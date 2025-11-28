"""
Retention Engine - Zone-aware checkpoint retention and archival.

This module provides a unified retention system that understands storage zones
(hot/warm/cold) and uses StorageHandles for location-agnostic operations.

Features:
- Zone-aware checkpoint discovery (knows where things live)
- Retention policies per zone (keep more on cold, less on hot)
- Archive operations (move from hot → warm → cold)
- Protection for promoted snapshots, latest, and best checkpoints

Usage:
    from management.retention_engine import RetentionEngine, get_retention_engine

    engine = get_retention_engine()

    # Run hot zone cleanup
    result = engine.apply_hot_policy()
    print(f"Deleted {result['deleted_count']} checkpoints, freed {result['freed_gb']:.1f} GB")

    # Archive old checkpoints to cold storage
    engine.archive_to_cold(max_age_days=30)

    # Get retention report
    report = engine.get_report()

Configuration:
    Retention policies are defined in config/storage_zones.json under "retention_policies"
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("retention_engine")


KB = 1024
MB = 1024 * KB
GB = 1024 * MB


@dataclass
class CheckpointInfo:
    """Information about a checkpoint for retention decisions."""
    path: Path
    step: int
    size_bytes: int
    mtime: float
    zone: str  # "hot", "warm", "cold"
    device: str  # Which device it's on
    is_promoted: bool = False  # Is this a promoted snapshot?
    is_best: bool = False  # Is this the best checkpoint?
    is_latest: bool = False  # Is this the latest checkpoint?
    canonical_name: Optional[str] = None

    @property
    def age_hours(self) -> float:
        """Age in hours since last modification."""
        return (time.time() - self.mtime) / 3600

    @property
    def age_days(self) -> float:
        """Age in days since last modification."""
        return self.age_hours / 24

    @property
    def date_str(self) -> str:
        """Date string YYYY-MM-DD from mtime."""
        return time.strftime("%Y-%m-%d", time.localtime(self.mtime))

    @property
    def size_gb(self) -> float:
        """Size in GB."""
        return self.size_bytes / GB


@dataclass
class RetentionPolicy:
    """Retention policy for a storage zone."""
    zone: str
    keep_recent_count: int = 50  # Keep N most recent checkpoints
    keep_promoted: bool = True  # Always keep promoted snapshots
    keep_best: bool = True  # Always keep best checkpoint
    keep_latest: bool = True  # Always keep latest checkpoint
    max_total_gb: float = 150.0  # Maximum total size in GB
    max_age_hours: Optional[float] = None  # Maximum age in hours
    min_age_hours: float = 36.0  # Minimum age before deletion (safety)
    one_per_day_after_count: int = 30  # After N checkpoints, keep only 1/day
    compress_on_archive: bool = False  # Compress when moving to colder zone


@dataclass
class RetentionResult:
    """Result of a retention operation."""
    zone: str
    analyzed_count: int = 0
    kept_count: int = 0
    deleted_count: int = 0
    archived_count: int = 0
    freed_bytes: int = 0
    errors: List[str] = field(default_factory=list)
    protected: List[str] = field(default_factory=list)  # Why checkpoints were kept

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / GB


class RetentionEngine:
    """
    Zone-aware retention engine.

    Manages checkpoint retention across storage zones (hot/warm/cold),
    applying zone-specific policies.
    """

    # Default policies per zone
    DEFAULT_POLICIES = {
        "hot": RetentionPolicy(
            zone="hot",
            keep_recent_count=50,
            max_total_gb=150.0,
            max_age_hours=168,  # 7 days
            min_age_hours=36,
            one_per_day_after_count=30,
        ),
        "warm": RetentionPolicy(
            zone="warm",
            keep_recent_count=100,
            max_total_gb=500.0,
            max_age_hours=2160,  # 90 days
            min_age_hours=168,  # 7 days
            one_per_day_after_count=50,
        ),
        "cold": RetentionPolicy(
            zone="cold",
            keep_recent_count=None,  # Keep all
            max_total_gb=None,  # No limit
            max_age_hours=None,  # No age limit
            min_age_hours=2160,  # 90 days
            compress_on_archive=True,
        ),
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize retention engine.

        Args:
            config_path: Path to storage_zones.json (auto-detected if None)
        """
        # Import here to avoid circular imports
        from core.devices import get_current_device_id
        from vault.storage_resolver import get_resolver

        self.resolver = get_resolver()
        self.device_id = get_current_device_id() or "trainer4090"

        # Load config
        if config_path:
            self.config_path = Path(config_path)
        else:
            base_dir = Path(__file__).parent.parent
            self.config_path = base_dir / "config" / "storage_zones.json"

        self._load_policies()

    def _load_policies(self) -> None:
        """Load retention policies from config."""
        self.policies = dict(self.DEFAULT_POLICIES)

        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)

                policy_config = config.get("retention_policies", {})
                for zone_name, policy_data in policy_config.items():
                    if zone_name in self.policies:
                        # Update default policy with config values
                        default = self.policies[zone_name]
                        self.policies[zone_name] = RetentionPolicy(
                            zone=zone_name,
                            keep_recent_count=policy_data.get(
                                "keep_recent_checkpoints", default.keep_recent_count
                            ),
                            keep_promoted=policy_data.get(
                                "keep_promoted_snapshots", default.keep_promoted
                            ),
                            keep_best=policy_data.get("keep_best", default.keep_best),
                            keep_latest=policy_data.get("keep_latest", default.keep_latest),
                            max_total_gb=policy_data.get("max_total_gb", default.max_total_gb),
                            max_age_hours=policy_data.get("max_age_hours", default.max_age_hours),
                            min_age_hours=policy_data.get("min_age_hours", default.min_age_hours),
                            one_per_day_after_count=policy_data.get(
                                "one_per_day_after_count", default.one_per_day_after_count
                            ),
                            compress_on_archive=policy_data.get(
                                "compress", default.compress_on_archive
                            ),
                        )
            except Exception as e:
                logger.warning(f"Failed to load retention policies: {e}")

    # =========================================================================
    # CHECKPOINT DISCOVERY
    # =========================================================================

    def discover_checkpoints(
        self,
        zone: str = "hot",
        include_snapshots: bool = False,
    ) -> List[CheckpointInfo]:
        """
        Discover all checkpoints in a zone.

        Args:
            zone: Storage zone to scan ("hot", "warm", "cold")
            include_snapshots: Include promoted snapshots in results

        Returns:
            List of CheckpointInfo sorted by step (newest first)
        """
        from core.storage_types import StorageZone, StorageKind

        checkpoints = []
        zone_enum = StorageZone(zone)

        # Get root for this zone on current device
        zone_root = self.resolver.zone_root(zone_enum)
        if not zone_root or not zone_root.exists():
            logger.debug(f"Zone {zone} not available on device {self.device_id}")
            return []

        # Find checkpoint directories
        # Pattern: checkpoint-XXXXX or checkpoint-XXXXX-YYYYMMDD-HHMM
        checkpoint_pattern = re.compile(r"checkpoint-(\d+)(?:-\d{8}-\d{4})?$")

        current_model_dir = zone_root / "models" / "current_model"
        if current_model_dir.exists():
            for candidate in current_model_dir.iterdir():
                if not candidate.is_dir():
                    continue

                match = checkpoint_pattern.match(candidate.name)
                if match:
                    step = int(match.group(1))
                    checkpoints.append(self._make_checkpoint_info(
                        candidate, step, zone, is_promoted=False
                    ))

        # Also check snapshots if requested
        if include_snapshots:
            snapshots_dir = zone_root / "snapshots"
            if snapshots_dir.exists():
                for candidate in snapshots_dir.iterdir():
                    if not candidate.is_dir():
                        continue
                    match = checkpoint_pattern.match(candidate.name)
                    if match:
                        step = int(match.group(1))
                        checkpoints.append(self._make_checkpoint_info(
                            candidate, step, zone, is_promoted=True
                        ))

        # Sort by step (newest first)
        checkpoints.sort(key=lambda c: c.step, reverse=True)

        # Mark latest and best
        if checkpoints:
            checkpoints[0].is_latest = True
            self._mark_best_checkpoint(checkpoints)

        return checkpoints

    def _make_checkpoint_info(
        self,
        path: Path,
        step: int,
        zone: str,
        is_promoted: bool,
    ) -> CheckpointInfo:
        """Create CheckpointInfo for a checkpoint directory."""
        try:
            stat = path.stat()
            mtime = stat.st_mtime
            size = self._get_dir_size(path)
        except OSError:
            mtime = time.time()
            size = 0

        return CheckpointInfo(
            path=path,
            step=step,
            size_bytes=size,
            mtime=mtime,
            zone=zone,
            device=self.device_id,
            is_promoted=is_promoted,
            canonical_name=path.name,
        )

    def _get_dir_size(self, path: Path) -> int:
        """Get directory size in bytes using du."""
        try:
            result = subprocess.run(
                ["du", "-sb", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return int(result.stdout.split()[0])
        except Exception:
            pass

        # Fallback to Python
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def _mark_best_checkpoint(self, checkpoints: List[CheckpointInfo]) -> None:
        """Mark the best checkpoint based on ledger data."""
        try:
            from core.checkpoint_ledger import get_ledger

            ledger = get_ledger()
            best = ledger.get_best(metric="train_loss")
            if best:
                for cp in checkpoints:
                    if cp.step == best.step:
                        cp.is_best = True
                        break
        except Exception:
            pass

    # =========================================================================
    # RETENTION PLANNING
    # =========================================================================

    def plan_retention(
        self,
        checkpoints: List[CheckpointInfo],
        policy: RetentionPolicy,
    ) -> Tuple[List[CheckpointInfo], List[CheckpointInfo]]:
        """
        Plan which checkpoints to keep and delete.

        Args:
            checkpoints: List of checkpoints (sorted newest first)
            policy: Retention policy to apply

        Returns:
            Tuple of (keep_list, delete_list)
        """
        if not checkpoints:
            return [], []

        keep = []
        delete = []
        seen_days: Set[str] = set()
        total_bytes = 0

        for i, cp in enumerate(checkpoints):
            reasons_to_keep = []

            # Protected checkpoints
            if policy.keep_latest and cp.is_latest:
                reasons_to_keep.append("latest")
            if policy.keep_best and cp.is_best:
                reasons_to_keep.append("best")
            if policy.keep_promoted and cp.is_promoted:
                reasons_to_keep.append("promoted")

            # Too young to delete
            if cp.age_hours < policy.min_age_hours:
                reasons_to_keep.append(f"too_young ({cp.age_hours:.0f}h < {policy.min_age_hours}h)")

            # Recent count protection
            if policy.keep_recent_count and i < policy.keep_recent_count:
                reasons_to_keep.append(f"recent (#{i+1})")

            # One per day after threshold
            if policy.one_per_day_after_count and i >= policy.one_per_day_after_count:
                if cp.date_str in seen_days:
                    # Already have one from this day
                    pass
                else:
                    reasons_to_keep.append(f"daily_keeper ({cp.date_str})")
                    seen_days.add(cp.date_str)

            # Age limit
            if policy.max_age_hours and cp.age_hours > policy.max_age_hours:
                if not reasons_to_keep:  # Not protected
                    delete.append(cp)
                    continue

            # Size limit (check after adding this checkpoint)
            if policy.max_total_gb:
                projected = (total_bytes + cp.size_bytes) / GB
                if projected > policy.max_total_gb and not reasons_to_keep:
                    delete.append(cp)
                    continue

            # Keep this checkpoint
            keep.append(cp)
            total_bytes += cp.size_bytes

            if i < policy.one_per_day_after_count or policy.one_per_day_after_count is None:
                seen_days.add(cp.date_str)

        return keep, delete

    # =========================================================================
    # RETENTION OPERATIONS
    # =========================================================================

    def apply_policy(
        self,
        zone: str = "hot",
        dry_run: bool = False,
    ) -> RetentionResult:
        """
        Apply retention policy to a zone.

        Args:
            zone: Zone to apply policy to
            dry_run: If True, don't actually delete anything

        Returns:
            RetentionResult with operation details
        """
        result = RetentionResult(zone=zone)

        policy = self.policies.get(zone)
        if not policy:
            result.errors.append(f"No policy for zone: {zone}")
            return result

        # Discover checkpoints
        checkpoints = self.discover_checkpoints(zone)
        result.analyzed_count = len(checkpoints)

        if not checkpoints:
            return result

        # Plan retention
        keep, delete = self.plan_retention(checkpoints, policy)
        result.kept_count = len(keep)
        result.deleted_count = len(delete)
        result.freed_bytes = sum(cp.size_bytes for cp in delete)

        # Record protected reasons
        for cp in keep:
            reasons = []
            if cp.is_latest:
                reasons.append("latest")
            if cp.is_best:
                reasons.append("best")
            if cp.is_promoted:
                reasons.append("promoted")
            if reasons:
                result.protected.append(f"{cp.canonical_name}: {', '.join(reasons)}")

        if dry_run:
            logger.info(
                f"[DRY RUN] Would delete {len(delete)} checkpoints, "
                f"freeing {result.freed_gb:.1f} GB"
            )
            return result

        # Actually delete
        for cp in delete:
            try:
                logger.info(f"Deleting checkpoint: {cp.path}")
                shutil.rmtree(cp.path)
            except Exception as e:
                result.errors.append(f"Failed to delete {cp.path}: {e}")
                result.deleted_count -= 1
                result.freed_bytes -= cp.size_bytes

        return result

    def apply_hot_policy(self, dry_run: bool = False) -> RetentionResult:
        """Apply retention policy to hot zone."""
        return self.apply_policy("hot", dry_run)

    def apply_warm_policy(self, dry_run: bool = False) -> RetentionResult:
        """Apply retention policy to warm zone."""
        return self.apply_policy("warm", dry_run)

    # =========================================================================
    # ARCHIVE OPERATIONS
    # =========================================================================

    def archive_to_warm(
        self,
        max_count: int = 10,
        min_age_hours: float = 168,  # 7 days
        dry_run: bool = False,
    ) -> RetentionResult:
        """
        Archive old checkpoints from hot to warm zone.

        Args:
            max_count: Maximum checkpoints to archive at once
            min_age_hours: Minimum age before archiving
            dry_run: If True, don't actually move anything
        """
        result = RetentionResult(zone="hot→warm")

        checkpoints = self.discover_checkpoints("hot")
        policy = self.policies["hot"]

        # Find candidates older than min_age_hours and outside keep_recent_count
        candidates = []
        for i, cp in enumerate(checkpoints):
            if i < policy.keep_recent_count:
                continue
            if cp.age_hours < min_age_hours:
                continue
            if cp.is_latest or cp.is_best or cp.is_promoted:
                continue
            candidates.append(cp)

        result.analyzed_count = len(candidates)

        # Limit to max_count
        to_archive = candidates[:max_count]

        for cp in to_archive:
            if dry_run:
                logger.info(f"[DRY RUN] Would archive: {cp.canonical_name}")
                result.archived_count += 1
                continue

            try:
                # Get warm zone path
                from core.storage_types import StorageKind, StorageZone, StorageHandle

                warm_handle = StorageHandle(
                    kind=StorageKind.SNAPSHOT,
                    key=cp.canonical_name,
                    zone=StorageZone.WARM,
                )
                warm_path = self.resolver.resolve(warm_handle)

                # Copy to warm zone
                warm_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(cp.path, warm_path)

                # Delete from hot
                shutil.rmtree(cp.path)

                logger.info(f"Archived {cp.canonical_name} to warm zone")
                result.archived_count += 1

            except Exception as e:
                result.errors.append(f"Failed to archive {cp.canonical_name}: {e}")

        return result

    def archive_to_cold(
        self,
        max_age_days: int = 90,
        dry_run: bool = False,
    ) -> RetentionResult:
        """
        Archive old snapshots from warm to cold zone.

        Args:
            max_age_days: Archive snapshots older than this
            dry_run: If True, don't actually move anything
        """
        result = RetentionResult(zone="warm→cold")

        snapshots = self.discover_checkpoints("warm", include_snapshots=True)
        min_age_hours = max_age_days * 24

        candidates = [
            cp for cp in snapshots
            if cp.age_hours > min_age_hours
            and not cp.is_latest
            and not cp.is_best
        ]

        result.analyzed_count = len(candidates)

        for cp in candidates:
            if dry_run:
                logger.info(f"[DRY RUN] Would archive to cold: {cp.canonical_name}")
                result.archived_count += 1
                continue

            try:
                from core.storage_types import StorageKind, StorageZone, StorageHandle

                cold_handle = StorageHandle(
                    kind=StorageKind.ARCHIVE,
                    key=cp.canonical_name,
                    zone=StorageZone.COLD,
                )
                cold_path = self.resolver.resolve(cold_handle)

                # Copy to cold zone (potentially compressed)
                cold_path.parent.mkdir(parents=True, exist_ok=True)

                policy = self.policies["cold"]
                if policy.compress_on_archive:
                    # Create tar.gz archive
                    archive_path = cold_path.with_suffix(".tar.gz")
                    subprocess.run(
                        ["tar", "-czf", str(archive_path), "-C", str(cp.path.parent), cp.path.name],
                        check=True,
                    )
                else:
                    shutil.copytree(cp.path, cold_path)

                # Delete from warm
                shutil.rmtree(cp.path)

                logger.info(f"Archived {cp.canonical_name} to cold zone")
                result.archived_count += 1

            except Exception as e:
                result.errors.append(f"Failed to archive {cp.canonical_name}: {e}")

        return result

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_report(self) -> Dict[str, Any]:
        """Get retention status report."""
        report = {
            "device": self.device_id,
            "zones": {},
            "total_checkpoints": 0,
            "total_size_gb": 0,
            "generated_at": datetime.now().isoformat(),
        }

        for zone_name in ["hot", "warm", "cold"]:
            try:
                checkpoints = self.discover_checkpoints(zone_name, include_snapshots=True)
                policy = self.policies.get(zone_name)

                zone_data = {
                    "checkpoint_count": len(checkpoints),
                    "total_size_gb": sum(cp.size_gb for cp in checkpoints),
                    "oldest_age_days": max(cp.age_days for cp in checkpoints) if checkpoints else 0,
                    "newest_age_hours": min(cp.age_hours for cp in checkpoints) if checkpoints else 0,
                    "policy": {
                        "keep_recent_count": policy.keep_recent_count if policy else None,
                        "max_total_gb": policy.max_total_gb if policy else None,
                        "max_age_hours": policy.max_age_hours if policy else None,
                    } if policy else None,
                }

                # Calculate what would be deleted
                if checkpoints and policy:
                    keep, delete = self.plan_retention(checkpoints, policy)
                    zone_data["would_delete_count"] = len(delete)
                    zone_data["would_free_gb"] = sum(cp.size_gb for cp in delete)

                report["zones"][zone_name] = zone_data
                report["total_checkpoints"] += len(checkpoints)
                report["total_size_gb"] += zone_data["total_size_gb"]

            except Exception as e:
                report["zones"][zone_name] = {"error": str(e)}

        return report


# =============================================================================
# SINGLETON
# =============================================================================

_engine: Optional[RetentionEngine] = None


def get_retention_engine() -> RetentionEngine:
    """Get or create the retention engine singleton."""
    global _engine
    if _engine is None:
        _engine = RetentionEngine()
    return _engine


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Retention Engine - Zone-aware checkpoint cleanup")
    parser.add_argument("command", nargs="?", default="report",
                        choices=["report", "apply", "archive"])
    parser.add_argument("--zone", default="hot", choices=["hot", "warm", "cold"],
                        help="Zone to operate on")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually delete/move")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    engine = get_retention_engine()

    if args.command == "report":
        report = engine.get_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("\n" + "=" * 60)
            print("RETENTION REPORT")
            print("=" * 60)
            print(f"Device: {report['device']}")
            print(f"Total checkpoints: {report['total_checkpoints']}")
            print(f"Total size: {report['total_size_gb']:.1f} GB")
            print()

            for zone, data in report["zones"].items():
                if "error" in data:
                    print(f"{zone.upper()}: ERROR - {data['error']}")
                else:
                    print(f"{zone.upper()}:")
                    print(f"  Checkpoints: {data['checkpoint_count']}")
                    print(f"  Size: {data['total_size_gb']:.1f} GB")
                    if data.get("would_delete_count"):
                        print(f"  Would delete: {data['would_delete_count']} ({data['would_free_gb']:.1f} GB)")
                    print()
            print("=" * 60)

    elif args.command == "apply":
        result = engine.apply_policy(args.zone, dry_run=args.dry_run)
        if args.json:
            print(json.dumps({
                "zone": result.zone,
                "analyzed": result.analyzed_count,
                "kept": result.kept_count,
                "deleted": result.deleted_count,
                "freed_gb": result.freed_gb,
                "errors": result.errors,
            }, indent=2))
        else:
            prefix = "[DRY RUN] " if args.dry_run else ""
            print(f"\n{prefix}Applied {args.zone} retention policy:")
            print(f"  Analyzed: {result.analyzed_count} checkpoints")
            print(f"  Kept: {result.kept_count}")
            print(f"  Deleted: {result.deleted_count}")
            print(f"  Freed: {result.freed_gb:.1f} GB")
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for err in result.errors[:5]:
                    print(f"    - {err}")

    elif args.command == "archive":
        if args.zone == "hot":
            result = engine.archive_to_warm(dry_run=args.dry_run)
        else:
            result = engine.archive_to_cold(dry_run=args.dry_run)

        if args.json:
            print(json.dumps({
                "zone": result.zone,
                "analyzed": result.analyzed_count,
                "archived": result.archived_count,
                "errors": result.errors,
            }, indent=2))
        else:
            prefix = "[DRY RUN] " if args.dry_run else ""
            print(f"\n{prefix}Archive operation ({result.zone}):")
            print(f"  Analyzed: {result.analyzed_count} checkpoints")
            print(f"  Archived: {result.archived_count}")
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
