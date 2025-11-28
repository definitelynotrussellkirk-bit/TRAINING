#!/usr/bin/env python3
"""
Forge Validator - Wraps DataValidator with eval leakage detection.

This is the main entry point for data validation in the Forge system.
It combines:
- Schema validation (from DataValidator)
- Content quality checks (from DataValidator)
- Eval leakage detection (from EvalBankManager)

Usage:
    from forge.validator import validate_file, validate_for_queue

    # Simple validation
    result = validate_file(Path("data.jsonl"))

    # Validation for queue ingestion (includes skill-specific leakage check)
    result = validate_for_queue(Path("data.jsonl"), skill_id="bin")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

# Version tracking
FORGE_VALIDATOR_VERSION = "1.0.0"


@dataclass
class ValidationResult:
    """Result of Forge validation."""
    passed: bool
    file_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    # Leakage detection
    leakage_detected: bool = False
    leakage_count: int = 0
    leakage_details: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    validated_at: str = ""
    validator_version: str = FORGE_VALIDATOR_VERSION
    skill_id: Optional[str] = None

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if self.passed:
            parts = [f"PASSED ({self.stats.get('total_examples', '?')} examples)"]
            if self.warnings:
                parts.append(f"{len(self.warnings)} warnings")
            return " - ".join(parts)
        else:
            parts = [f"FAILED"]
            if self.leakage_detected:
                parts.append(f"{self.leakage_count} eval leakage")
            if self.errors:
                parts.append(f"{len(self.errors)} errors")
            return " - ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "passed": self.passed,
            "file_path": self.file_path,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
            "leakage_detected": self.leakage_detected,
            "leakage_count": self.leakage_count,
            "leakage_details": self.leakage_details[:10],  # Limit for size
            "validated_at": self.validated_at,
            "validator_version": self.validator_version,
            "skill_id": self.skill_id,
        }


class ForgeValidator:
    """
    Main Forge validator combining DataValidator with eval leakage detection.

    Attributes:
        max_leakage_allowed: Maximum eval leakage items before rejection (default: 0)
        tokenizer: Optional tokenizer for length checks
        max_length: Maximum token length for training
    """

    def __init__(
        self,
        tokenizer=None,
        max_length: int = 4096,
        max_leakage_allowed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_leakage_allowed = max_leakage_allowed

        # Lazy-load eval banks
        self._eval_bank_manager = None

    @property
    def eval_bank_manager(self):
        """Lazy-load EvalBankManager."""
        if self._eval_bank_manager is None:
            from forge.leakage import EvalBankManager
            self._eval_bank_manager = EvalBankManager()
        return self._eval_bank_manager

    def validate(
        self,
        file_path: Path,
        skill_id: Optional[str] = None,
        deep: bool = True,
        check_leakage: bool = True,
    ) -> ValidationResult:
        """
        Validate a file for training.

        Args:
            file_path: Path to JSONL file
            skill_id: Optional skill ID for skill-specific leakage check
            deep: If True, run deep content checks
            check_leakage: If True, check for eval leakage

        Returns:
            ValidationResult with pass/fail and details
        """
        file_path = Path(file_path)
        result = ValidationResult(
            passed=True,
            file_path=str(file_path),
            validated_at=datetime.utcnow().isoformat() + "Z",
            skill_id=skill_id,
        )

        # Check file exists
        if not file_path.exists():
            result.passed = False
            result.errors.append(f"File not found: {file_path}")
            return result

        # Run DataValidator
        try:
            from core.validation.validator import DataValidator, ValidationLevel

            level = ValidationLevel.DEEP if deep else ValidationLevel.QUICK
            validator = DataValidator(
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            dv_result = validator.validate(file_path, level=level)

            # Copy results
            result.errors.extend(dv_result.errors)
            result.warnings.extend(dv_result.warnings)
            result.stats.update(dv_result.stats)

            if not dv_result.valid:
                result.passed = False

        except Exception as e:
            result.warnings.append(f"DataValidator error: {e}")
            logger.warning(f"DataValidator error: {e}")

        # Check eval leakage
        if check_leakage and skill_id:
            try:
                leakage_result = self._check_eval_leakage(file_path, skill_id)
                result.leakage_detected = leakage_result["detected"]
                result.leakage_count = leakage_result["count"]
                result.leakage_details = leakage_result["details"]

                if result.leakage_count > self.max_leakage_allowed:
                    result.passed = False
                    result.errors.append(
                        f"Eval leakage: {result.leakage_count} examples match eval bank for '{skill_id}'"
                    )
            except Exception as e:
                result.warnings.append(f"Leakage check error: {e}")
                logger.warning(f"Leakage check error: {e}")

        return result

    def _check_eval_leakage(
        self,
        file_path: Path,
        skill_id: str,
        max_samples: int = 10000,
    ) -> Dict[str, Any]:
        """
        Check file for eval leakage against skill's eval bank.

        Returns:
            {"detected": bool, "count": int, "details": [...]}
        """
        bank = self.eval_bank_manager.get_bank(skill_id)
        if not bank:
            return {"detected": False, "count": 0, "details": []}

        leakage_count = 0
        leakage_details = []

        with open(file_path) as f:
            for line_num, line in enumerate(f):
                if line_num >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Check ID match
                example_id = example.get("id", "")
                if example_id and example_id in bank.ids:
                    leakage_count += 1
                    if len(leakage_details) < 20:
                        leakage_details.append({
                            "line": line_num + 1,
                            "match_type": "id",
                            "matched_id": example_id,
                        })
                    continue

                # Check prompt hash match
                prompt = self._extract_prompt(example)
                if prompt:
                    prompt_hash = hash(prompt.strip().lower())
                    if prompt_hash in bank.prompt_hashes:
                        leakage_count += 1
                        if len(leakage_details) < 20:
                            leakage_details.append({
                                "line": line_num + 1,
                                "match_type": "prompt_hash",
                                "prompt_preview": prompt[:100],
                            })

        return {
            "detected": leakage_count > 0,
            "count": leakage_count,
            "details": leakage_details,
        }

    def _extract_prompt(self, example: Dict[str, Any]) -> Optional[str]:
        """Extract user prompt from various formats."""
        # Try common field names
        for field in ["user_prompt", "prompt", "question", "input"]:
            if field in example and example[field]:
                return example[field]

        # Try messages format
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")

        return None


# Convenience functions

def validate_file(
    file_path: Path,
    skill_id: Optional[str] = None,
    deep: bool = True,
) -> ValidationResult:
    """
    Validate a file for training.

    Args:
        file_path: Path to JSONL file
        skill_id: Optional skill ID for leakage check
        deep: Run deep content checks

    Returns:
        ValidationResult
    """
    validator = ForgeValidator()
    return validator.validate(file_path, skill_id=skill_id, deep=deep)


def validate_for_queue(
    file_path: Path,
    skill_id: Optional[str] = None,
) -> ValidationResult:
    """
    Validate a file before adding to training queue.

    This is the standard validation for queue ingestion.
    Uses deep validation and leakage checking.

    Args:
        file_path: Path to JSONL file
        skill_id: Optional skill ID (auto-detected from filename if not provided)

    Returns:
        ValidationResult
    """
    file_path = Path(file_path)

    # Auto-detect skill from filename
    if skill_id is None:
        skill_id = _detect_skill_from_filename(file_path.name)

    validator = ForgeValidator()
    return validator.validate(
        file_path,
        skill_id=skill_id,
        deep=True,
        check_leakage=True,
    )


def _detect_skill_from_filename(filename: str) -> Optional[str]:
    """
    Detect skill ID from filename patterns.

    Examples:
        "train_SYLLO_L1_20251128.jsonl" → "sy"
        "sparring_binary_L5_100.jsonl" → "bin"
        "bin_training_v1.jsonl" → "bin"
    """
    filename_lower = filename.lower()

    # Check for known patterns
    if "syllo" in filename_lower or "_sy_" in filename_lower:
        return "sy"
    if "binary" in filename_lower or "_bin_" in filename_lower or "bin_" in filename_lower:
        return "bin"
    if "sparring" in filename_lower:
        # Try to extract skill from sparring filename
        if "binary" in filename_lower:
            return "bin"
        if "sy" in filename_lower:
            return "sy"

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m forge.validator <file.jsonl> [skill_id]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    skill_id = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Validating: {file_path}")
    if skill_id:
        print(f"Skill: {skill_id}")

    result = validate_file(file_path, skill_id=skill_id)

    print(f"\nResult: {result.summary}")

    if result.errors:
        print(f"\nErrors:")
        for err in result.errors[:10]:
            print(f"  - {err}")

    if result.warnings:
        print(f"\nWarnings:")
        for warn in result.warnings[:10]:
            print(f"  - {warn}")

    if result.leakage_detected:
        print(f"\nEval Leakage: {result.leakage_count} matches")
        for detail in result.leakage_details[:5]:
            print(f"  - Line {detail['line']}: {detail['match_type']}")

    print(f"\nStats:")
    for key, value in result.stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    sys.exit(0 if result.passed else 1)
