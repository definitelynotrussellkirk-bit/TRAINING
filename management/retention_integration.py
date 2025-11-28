#!/usr/bin/env python3
"""
Integration examples for RetentionManager with training daemon

This shows how to integrate the retention manager into your existing training flow.
"""

from pathlib import Path
from retention_manager import RetentionManager

# Use centralized path resolution for default
try:
    from core.paths import get_base_dir
    _DEFAULT_BASE_DIR = get_base_dir()
except ImportError:
    _DEFAULT_BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of management/


class TrainingDaemonIntegration:
    """Example integration with training daemon"""

    def __init__(self, base_dir: Path = None):
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR
        self.output_dir = self.base_dir / "models" / "current_model"
        self.base_model_path = self.base_dir / "models" / "Qwen3-0.6B"

        # Initialize retention manager
        self.retention_manager = RetentionManager(
            output_dir=self.output_dir,
            base_model_path=self.base_model_path
        )

    def after_checkpoint_save(self, checkpoint_path: str, metrics: dict):
        """
        Call this after HuggingFace Trainer saves a checkpoint

        Args:
            checkpoint_path: Path to the saved checkpoint
            metrics: Training metrics (loss, eval_loss, etc.)
        """
        # Register the new checkpoint
        self.retention_manager.register_checkpoint(
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            is_latest=True
        )

        # Lightweight cleanup (only if over limit)
        status = self.retention_manager.get_status()
        if status['usage_pct'] > 90:  # Only cleanup if > 90% of limit
            self.retention_manager.enforce_retention(dry_run=False)

    def daily_maintenance(self):
        """
        Call this once per day (e.g., at 03:00 via cron or daemon timer)

        Creates daily snapshot and enforces full retention policy
        """
        # Create today's snapshot if it doesn't exist
        self.retention_manager.create_daily_snapshot_if_needed()

        # Enforce retention policy
        self.retention_manager.enforce_retention(dry_run=False)

    def hourly_maintenance(self):
        """
        Call this hourly for light maintenance

        Just enforces retention without creating snapshots
        """
        status = self.retention_manager.get_status()

        # Only cleanup if approaching limit
        if status['usage_pct'] > 80:
            self.retention_manager.enforce_retention(dry_run=False)


# Example usage in HuggingFace Trainer callback
from transformers import TrainerCallback

class RetentionCallback(TrainerCallback):
    """Callback for HuggingFace Trainer to manage retention"""

    def __init__(self, retention_manager: RetentionManager):
        self.retention_manager = retention_manager

    def on_save(self, args, state, control, **kwargs):
        """Called when trainer saves a checkpoint"""
        checkpoint_dir = f"checkpoint-{state.global_step}"

        # Get latest metrics
        metrics = {}
        if state.log_history:
            last_log = state.log_history[-1]
            metrics = {k: v for k, v in last_log.items()
                      if isinstance(v, (int, float))}

        # Register checkpoint
        self.retention_manager.register_checkpoint(
            checkpoint_path=checkpoint_dir,
            metrics=metrics,
            is_latest=True
        )

        # Cleanup if needed
        status = self.retention_manager.get_status()
        if status['usage_pct'] > 90:
            self.retention_manager.enforce_retention(dry_run=False)

        return control


# Example: Modify training_daemon.py to use RetentionManager
def daemon_example():
    """
    Add this to your training_daemon.py:

    1. Import at top:
       from retention_manager import RetentionManager

    2. In __init__:
       self.retention_manager = RetentionManager(
           output_dir=self.current_model_dir,
           base_model_path=self.base_model_path
       )

    3. After training completes successfully:
       metrics = {
           "loss": final_loss,
           "eval_loss": eval_loss
       }
       self.retention_manager.register_checkpoint(
           checkpoint_path=latest_checkpoint,
           metrics=metrics,
           is_latest=True
       )

    4. In daily timer (check once per day):
       current_hour = datetime.now().hour
       if current_hour == 3 and not self.daily_snapshot_done_today:
           self.retention_manager.create_daily_snapshot_if_needed()
           self.retention_manager.enforce_retention(dry_run=False)
           self.daily_snapshot_done_today = True

    5. In hourly timer:
       if minutes_elapsed >= 60:
           status = self.retention_manager.get_status()
           if status['usage_pct'] > 80:
               self.retention_manager.enforce_retention(dry_run=False)
    """
    pass


# Example: Standalone cron job
def cron_example():
    """
    Set up a cron job to run daily maintenance:

    # crontab -e
    0 3 * * * cd $TRAINING_DIR && python3 -c "from management.retention_manager import RetentionManager; from core.paths import get_base_dir; base = get_base_dir(); m = RetentionManager(base / 'models/current_model'); m.create_daily_snapshot_if_needed(); m.enforce_retention()"

    Or create a shell script:

    #!/bin/bash
    # scripts/daily_retention.sh

    cd $(dirname $0)/..
    python3 management/retention_manager.py \\
        --output-dir models/current_model \\
        --base-model models/Qwen3-0.6B \\
        --snapshot \\
        --enforce

    Then in crontab:
    0 3 * * * $TRAINING_DIR/scripts/daily_retention.sh >> logs/retention.log 2>&1
    """
    pass


# Example: Manual operations
def manual_examples():
    """
    Manual operations you might want to run:

    # Check current status
    python3 management/retention_manager.py --output-dir models/current_model --status

    # Create snapshot manually
    python3 management/retention_manager.py --output-dir models/current_model --snapshot

    # Dry run cleanup (see what would be deleted)
    python3 management/retention_manager.py --output-dir models/current_model --enforce --dry-run

    # Actually cleanup
    python3 management/retention_manager.py --output-dir models/current_model --enforce

    # Rebuild index if corrupted
    python3 management/retention_manager.py --output-dir models/current_model --rebuild

    # Register a checkpoint manually
    python3 management/retention_manager.py --output-dir models/current_model --register checkpoints/checkpoint-1500
    """
    pass


if __name__ == '__main__':
    # Example: Quick test
    integration = TrainingDaemonIntegration()  # Uses auto-detected base_dir

    print("Getting status...")
    integration.retention_manager.print_status()
