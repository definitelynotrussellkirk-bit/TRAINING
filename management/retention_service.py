#!/usr/bin/env python3
"""
Retention Service - High-level adapter for daemon use

Wraps RetentionManager to provide a simple interface for the training daemon.
Handles multiple checkpoint directories and provides unified logging.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from retention_manager import RetentionManager

logger = logging.getLogger(__name__)


class RetentionService:
    """
    High-level retention service for training daemon.

    Unlike the old enforce_retention() which took a list of roots,
    this service manages a single output directory (current_model).

    Example usage:
        service = RetentionService(
            output_dir=Path("/path/to/current_model"),
            config={"enabled": True}
        )
        summary = service.enforce()
        print(f"Deleted {summary['deleted_checkpoints']} checkpoints")
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        custom_logger: Optional[logging.Logger] = None
    ):
        """
        Initialize retention service.

        Args:
            output_dir: Directory containing checkpoints (e.g., current_model)
            config: Optional configuration dict with:
                - enabled: bool (default True)
                - max_total_size_gb: int (default 150)
                - min_checkpoint_age_hours: int (default 36)
            custom_logger: Optional custom logger instance
        """
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.logger = custom_logger or logger

        # Extract config with defaults
        self.enabled = self.config.get("enabled", True)

        # Initialize manager (it uses global constants for limits)
        self.manager = RetentionManager(output_dir=self.output_dir)

        self.logger.info(f"RetentionService initialized: {self.output_dir}")
        self.logger.info(f"  Enabled: {self.enabled}")

    def enforce(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Enforce retention policy.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Summary dict with:
            - total_size_gb: Current total size
            - limit_gb: Configured limit
            - deleted_checkpoints: Number of checkpoints deleted
            - deleted_snapshots: Number of snapshots deleted
            - deleted_size_gb: Total GB freed
            - protected_checkpoints: Number of protected checkpoints
            - protected_snapshots: Number of protected snapshots
            - dry_run: Whether this was a dry run
            - skipped: True if retention is disabled
        """
        if not self.enabled:
            self.logger.debug("Retention disabled, skipping")
            return {
                "skipped": True,
                "reason": "retention disabled",
                "deleted_checkpoints": 0,
                "deleted_snapshots": 0,
                "deleted_size_gb": 0
            }

        if not self.output_dir.exists():
            self.logger.warning(f"Output directory does not exist: {self.output_dir}")
            return {
                "skipped": True,
                "reason": "directory not found",
                "deleted_checkpoints": 0,
                "deleted_snapshots": 0,
                "deleted_size_gb": 0
            }

        try:
            summary = self.manager.enforce_retention(dry_run=dry_run)
            summary["skipped"] = False
            return summary

        except Exception as e:
            self.logger.error(f"Retention enforcement failed: {e}", exc_info=True)
            return {
                "skipped": True,
                "reason": f"error: {str(e)}",
                "deleted_checkpoints": 0,
                "deleted_snapshots": 0,
                "deleted_size_gb": 0
            }

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        metrics: Optional[Dict[str, float]] = None,
        is_latest: bool = False
    ) -> bool:
        """
        Register a new checkpoint with the retention system.

        Args:
            checkpoint_path: Path to the checkpoint directory
            metrics: Training metrics (loss, eval_loss, etc.)
            is_latest: Whether this is the latest checkpoint

        Returns:
            True if registration successful
        """
        if not self.enabled:
            return False

        try:
            self.manager.register_checkpoint(
                checkpoint_path=checkpoint_path,
                metrics=metrics or {},
                is_latest=is_latest
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to register checkpoint: {e}")
            return False

    def create_daily_snapshot(self) -> Optional[Path]:
        """
        Create daily snapshot if one doesn't exist for today.

        Returns:
            Path to snapshot if created, None otherwise
        """
        if not self.enabled:
            return None

        try:
            return self.manager.create_daily_snapshot_if_needed()
        except Exception as e:
            self.logger.error(f"Failed to create daily snapshot: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current retention status.

        Returns:
            Status dict with checkpoint/snapshot counts and sizes
        """
        try:
            return self.manager.get_status()
        except Exception as e:
            self.logger.error(f"Failed to get retention status: {e}")
            return {"error": str(e)}


# Convenience function for backward compatibility
def enforce_retention_new(
    output_dir: Path,
    logger_instance: Optional[logging.Logger] = None,
    dry_run: bool = False,
    enabled: bool = True
) -> Dict[str, Any]:
    """
    Convenience function matching old enforce_retention signature pattern.

    Args:
        output_dir: Directory to manage
        logger_instance: Optional logger
        dry_run: If True, don't actually delete
        enabled: If False, skip retention

    Returns:
        Summary dict from RetentionService.enforce()
    """
    service = RetentionService(
        output_dir=output_dir,
        config={"enabled": enabled},
        custom_logger=logger_instance
    )
    return service.enforce(dry_run=dry_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retention Service CLI")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Directory to manage")
    parser.add_argument("--dry-run", action="store_true",
                       help="Report what would be deleted")
    parser.add_argument("--status", action="store_true",
                       help="Show current status")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    service = RetentionService(output_dir=args.output_dir)

    if args.status:
        status = service.get_status()
        print("\nRetention Status:")
        print(f"  Total size: {status.get('total_size_gb', 0):.1f} GB")
        print(f"  Checkpoints: {status.get('checkpoints', {}).get('count', 0)}")
        print(f"  Snapshots: {status.get('snapshots', {}).get('count', 0)}")
    else:
        print("\nEnforcing retention policy...")
        summary = service.enforce(dry_run=args.dry_run)
        print(f"\nSummary:")
        print(f"  Deleted checkpoints: {summary.get('deleted_checkpoints', 0)}")
        print(f"  Deleted snapshots: {summary.get('deleted_snapshots', 0)}")
        print(f"  Freed: {summary.get('deleted_size_gb', 0):.1f} GB")
        if summary.get('dry_run'):
            print("  (dry run - no changes made)")
