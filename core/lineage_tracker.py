#!/usr/bin/env python3
"""
Data Lineage Tracker - Aggregate per-generator and per-validator statistics.

This module tracks validation outcomes by generator and validator to answer:
- "Which generator is producing the most rejections?"
- "Did model quality drop when generator G moved from v3 → v4?"
- "Are some validators hyper-aggressive compared to others?"

Statistics are written to status/data_lineage.json for dashboard consumption.

Usage:
    from core.lineage_tracker import LineageTracker
    from core.validation.validator import ValidationResult

    tracker = LineageTracker(status_dir=Path("status"))

    # After each validation
    result = validator.validate(file_path)
    tracker.record_validation(result)

    # Get current stats
    stats = tracker.get_stats()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading

logger = logging.getLogger(__name__)


@dataclass
class GeneratorStats:
    """Statistics for a single generator@version."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    by_validator: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_seen: Optional[str] = None
    error_reasons: Dict[str, int] = field(default_factory=dict)

    def record(self, passed: bool, validator_key: str, errors: List[str] = None):
        """Record a validation result."""
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        # Track by validator
        if validator_key not in self.by_validator:
            self.by_validator[validator_key] = {"total": 0, "passed": 0, "failed": 0}

        self.by_validator[validator_key]["total"] += 1
        if passed:
            self.by_validator[validator_key]["passed"] += 1
        else:
            self.by_validator[validator_key]["failed"] += 1

        # Track error reasons (first 3 unique errors only)
        if errors:
            for err in errors[:3]:
                # Truncate long errors
                err_key = err[:100] if len(err) > 100 else err
                self.error_reasons[err_key] = self.error_reasons.get(err_key, 0) + 1

        self.last_seen = datetime.now().isoformat()

    @property
    def fail_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.failed / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "fail_rate": round(self.fail_rate, 2),
            "by_validator": self.by_validator,
            "last_seen": self.last_seen,
            "top_errors": dict(sorted(
                self.error_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),  # Top 5 errors
        }


@dataclass
class ValidatorStats:
    """Statistics for a single validator@version."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    by_generator: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_seen: Optional[str] = None

    def record(self, passed: bool, generator_key: str):
        """Record a validation result."""
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        # Track by generator
        if generator_key not in self.by_generator:
            self.by_generator[generator_key] = {"total": 0, "passed": 0, "failed": 0}

        self.by_generator[generator_key]["total"] += 1
        if passed:
            self.by_generator[generator_key]["passed"] += 1
        else:
            self.by_generator[generator_key]["failed"] += 1

        self.last_seen = datetime.now().isoformat()

    @property
    def fail_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.failed / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "fail_rate": round(self.fail_rate, 2),
            "by_generator": self.by_generator,
            "last_seen": self.last_seen,
        }


class LineageTracker:
    """
    Aggregates per-generator and per-validator validation statistics.

    Thread-safe: Uses a lock for concurrent access.
    Persists to status/data_lineage.json with atomic writes.
    """

    def __init__(self, status_dir: Path):
        """
        Initialize the lineage tracker.

        Args:
            status_dir: Directory for status files (e.g., Path("status"))
        """
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.status_dir / "data_lineage.json"

        self._lock = threading.Lock()
        self._generators: Dict[str, GeneratorStats] = {}
        self._validators: Dict[str, ValidatorStats] = {}
        self._total_validations = 0
        self._last_updated: Optional[str] = None

        # Load existing stats on startup
        self._load()

    def _load(self):
        """Load existing stats from disk."""
        if not self.status_file.exists():
            return

        try:
            with open(self.status_file) as f:
                data = json.load(f)

            self._total_validations = data.get("total_validations", 0)
            self._last_updated = data.get("last_updated")

            # Reconstruct generator stats
            for gen_key, gen_data in data.get("generators", {}).items():
                stats = GeneratorStats(
                    total=gen_data.get("total", 0),
                    passed=gen_data.get("passed", 0),
                    failed=gen_data.get("failed", 0),
                    by_validator=gen_data.get("by_validator", {}),
                    last_seen=gen_data.get("last_seen"),
                    error_reasons=gen_data.get("top_errors", {}),
                )
                self._generators[gen_key] = stats

            # Reconstruct validator stats
            for val_key, val_data in data.get("validators", {}).items():
                stats = ValidatorStats(
                    total=val_data.get("total", 0),
                    passed=val_data.get("passed", 0),
                    failed=val_data.get("failed", 0),
                    by_generator=val_data.get("by_generator", {}),
                    last_seen=val_data.get("last_seen"),
                )
                self._validators[val_key] = stats

            logger.info(f"Loaded lineage stats: {self._total_validations} validations")

        except Exception as e:
            logger.warning(f"Failed to load lineage stats: {e}")

    def _save(self):
        """Save stats to disk (atomic write)."""
        try:
            data = {
                "total_validations": self._total_validations,
                "last_updated": self._last_updated,
                "generators": {
                    key: stats.to_dict()
                    for key, stats in self._generators.items()
                },
                "validators": {
                    key: stats.to_dict()
                    for key, stats in self._validators.items()
                },
                "summary": self._compute_summary(),
            }

            # Atomic write
            temp_file = self.status_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.status_file)

        except Exception as e:
            logger.warning(f"Failed to save lineage stats: {e}")

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for dashboard."""
        summary = {
            "total_generators": len(self._generators),
            "total_validators": len(self._validators),
            "worst_generator": None,
            "worst_validator": None,
            "overall_fail_rate": 0.0,
        }

        # Find worst generator (highest fail rate with >10 samples)
        worst_gen = None
        worst_gen_rate = 0.0
        for key, stats in self._generators.items():
            if stats.total >= 10 and stats.fail_rate > worst_gen_rate:
                worst_gen = key
                worst_gen_rate = stats.fail_rate

        if worst_gen:
            summary["worst_generator"] = {
                "id": worst_gen,
                "fail_rate": round(worst_gen_rate, 2),
                "total": self._generators[worst_gen].total,
            }

        # Find worst validator (highest fail rate with >10 samples)
        worst_val = None
        worst_val_rate = 0.0
        for key, stats in self._validators.items():
            if stats.total >= 10 and stats.fail_rate > worst_val_rate:
                worst_val = key
                worst_val_rate = stats.fail_rate

        if worst_val:
            summary["worst_validator"] = {
                "id": worst_val,
                "fail_rate": round(worst_val_rate, 2),
                "total": self._validators[worst_val].total,
            }

        # Overall fail rate
        total_failed = sum(s.failed for s in self._generators.values())
        if self._total_validations > 0:
            summary["overall_fail_rate"] = round(
                (total_failed / self._total_validations) * 100, 2
            )

        return summary

    def record_validation(
        self,
        valid: bool,
        validator_name: str,
        validator_version: str,
        generator_id: Optional[str] = None,
        generator_version: Optional[str] = None,
        errors: List[str] = None,
    ):
        """
        Record a validation result for lineage tracking.

        Args:
            valid: Whether validation passed
            validator_name: Name of the validator (e.g., "data_validator")
            validator_version: Version of the validator (e.g., "1.0.0")
            generator_id: ID of the generator that produced the data
            generator_version: Version of the generator
            errors: List of error messages if validation failed
        """
        with self._lock:
            self._total_validations += 1
            self._last_updated = datetime.now().isoformat()

            # Build keys
            gen_key = f"{generator_id or 'UNKNOWN'}@{generator_version or '?'}"
            val_key = f"{validator_name}@{validator_version}"

            # Update generator stats
            if gen_key not in self._generators:
                self._generators[gen_key] = GeneratorStats()
            self._generators[gen_key].record(valid, val_key, errors)

            # Update validator stats
            if val_key not in self._validators:
                self._validators[val_key] = ValidatorStats()
            self._validators[val_key].record(valid, gen_key)

            # Persist
            self._save()

    def record_from_result(self, result) -> None:
        """
        Record a validation result directly from a ValidationResult object.

        Args:
            result: ValidationResult with lineage fields
        """
        self.record_validation(
            valid=result.valid,
            validator_name=getattr(result, 'validator_name', 'unknown'),
            validator_version=getattr(result, 'validator_version', '?'),
            generator_id=getattr(result, 'generator_id', None),
            generator_version=getattr(result, 'generator_version', None),
            errors=result.errors if hasattr(result, 'errors') else None,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current lineage statistics."""
        with self._lock:
            return {
                "total_validations": self._total_validations,
                "last_updated": self._last_updated,
                "generators": {
                    key: stats.to_dict()
                    for key, stats in self._generators.items()
                },
                "validators": {
                    key: stats.to_dict()
                    for key, stats in self._validators.items()
                },
                "summary": self._compute_summary(),
            }

    def get_generator_stats(self, generator_id: str) -> Optional[Dict[str, Any]]:
        """Get stats for a specific generator (any version)."""
        with self._lock:
            result = {"versions": {}, "total": 0, "passed": 0, "failed": 0}
            for key, stats in self._generators.items():
                if key.startswith(f"{generator_id}@"):
                    result["versions"][key] = stats.to_dict()
                    result["total"] += stats.total
                    result["passed"] += stats.passed
                    result["failed"] += stats.failed

            if result["total"] == 0:
                return None
            result["fail_rate"] = round((result["failed"] / result["total"]) * 100, 2)
            return result

    def get_validator_stats(self, validator_name: str) -> Optional[Dict[str, Any]]:
        """Get stats for a specific validator (any version)."""
        with self._lock:
            result = {"versions": {}, "total": 0, "passed": 0, "failed": 0}
            for key, stats in self._validators.items():
                if key.startswith(f"{validator_name}@"):
                    result["versions"][key] = stats.to_dict()
                    result["total"] += stats.total
                    result["passed"] += stats.passed
                    result["failed"] += stats.failed

            if result["total"] == 0:
                return None
            result["fail_rate"] = round((result["failed"] / result["total"]) * 100, 2)
            return result

    def reset(self):
        """Reset all statistics (for testing)."""
        with self._lock:
            self._generators.clear()
            self._validators.clear()
            self._total_validations = 0
            self._last_updated = None
            self._save()


# =============================================================================
# GLOBAL SINGLETON (optional, for easy access)
# =============================================================================

_global_tracker: Optional[LineageTracker] = None


def get_global_tracker(status_dir: Path = None) -> LineageTracker:
    """Get or create the global lineage tracker."""
    global _global_tracker
    if _global_tracker is None:
        if status_dir is None:
            status_dir = Path("/path/to/training/status")
        _global_tracker = LineageTracker(status_dir)
    return _global_tracker


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Lineage Tracker")
    parser.add_argument("--status-dir", default="/path/to/training/status",
                       help="Status directory")
    parser.add_argument("--show", action="store_true", help="Show current stats")
    parser.add_argument("--reset", action="store_true", help="Reset all stats")
    args = parser.parse_args()

    tracker = LineageTracker(Path(args.status_dir))

    if args.reset:
        tracker.reset()
        print("Stats reset.")
    elif args.show:
        stats = tracker.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        # Demo: record some test validations
        print("Recording test validations...")
        tracker.record_validation(
            valid=True,
            validator_name="data_validator",
            validator_version="1.0.0",
            generator_id="discrimination",
            generator_version="1.0.0",
        )
        tracker.record_validation(
            valid=False,
            validator_name="data_validator",
            validator_version="1.0.0",
            generator_id="syllo_api",
            generator_version="1.0.0",
            errors=["Missing required field: messages"],
        )
        tracker.record_validation(
            valid=True,
            validator_name="spec_validator",
            validator_version="1.0.0",
            generator_id="syllo_api",
            generator_version="1.0.0",
        )

        print("\nCurrent stats:")
        stats = tracker.get_stats()
        print(json.dumps(stats, indent=2))
