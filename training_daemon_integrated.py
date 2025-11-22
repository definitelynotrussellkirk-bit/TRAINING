#!/usr/bin/env python3
"""
Training Daemon - RTX 3090 Continuous Training System (INTEGRATED)

Enhanced with:
- Control system (pause/stop/skip/resume)
- Priority queue management
- Graceful state transitions
- Signal-based control (no process killing)

Simple, focused purpose:
1. Watch INBOX for new training data
2. Train on it (1 epoch only) with graceful control
3. Delete data after successful training (keep failed files)
4. Save daily snapshots of model
5. Always train on newest model
6. Never touch old snapshots

Usage:
    python3 training_daemon_integrated.py --base-dir /training

Directory structure:
    /training/
        inbox/          ← Drop JSONL files here
        control/        ← Control signals (.pause, .stop, .skip, .resume)
        queue/          ← Priority queues (high/normal/low)
        current_model/  ← Active model being trained
        snapshots/      ← Daily snapshots (YYYY-MM-DD/)
        logs/           ← Training logs
        config.json     ← Training configuration
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add ultimate_trainer to path
sys.path.insert(0, str(Path(__file__).parent))

from train import UltimateTrainer
from training_controller import TrainingController
from training_queue import TrainingQueue


class IntegratedTrainingDaemon:
    """Continuous training daemon with full control system"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.inbox_dir = self.base_dir / "inbox"
        self.current_model_dir = self.base_dir / "current_model"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.logs_dir = self.base_dir / "logs"
        self.config_file = self.base_dir / "config.json"

        # Setup logging
        self.setup_logging()

        # Load config
        self.config = self.load_config()

        # Initialize control system
        self.controller = TrainingController(str(self.base_dir))
        self.logger.info("✅ Control system initialized")

        # Initialize queue system
        self.queue = TrainingQueue(str(self.base_dir))
        self.logger.info("✅ Queue system initialized")

        # Track last snapshot date
        self.last_snapshot_date = None

        # Track last consolidation date
        self.last_consolidation_date = None
        self.consolidation_marker = self.base_dir / ".last_consolidation"

        # Current training state
        self.current_file = None
        self.paused = False

    def setup_logging(self):
        """Setup logging to file and console"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.logs_dir / f"daemon_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)

    def load_config(self) -> dict:
        """Load training configuration"""
        if not self.config_file.exists():
            # Create default config
            default_config = {
                "model_name": "qwen3_0.6b",
                "model_path": None,  # Set this to your base model path
                "batch_size": 4,
                "gradient_accumulation": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "lora_r": 64,
                "lora_alpha": 32,
                "use_qlora": True,  # Use QLoRA to reduce VRAM usage
                "eval_steps": 100,
                "num_eval_samples": 5,
                "save_steps": 500,
                "poll_interval": 60,  # Check inbox every 60 seconds
                "snapshot_time": "02:00"  # Daily snapshot at 2 AM
            }

            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)

            self.logger.info(f"Created default config at {self.config_file}")
            self.logger.warning(f"IMPORTANT: Edit {self.config_file} and set model_path!")

            return default_config

        with open(self.config_file) as f:
            config = json.load(f)

        self.logger.info(f"Loaded config from {self.config_file}")
        return config

    def setup_directories(self):
        """Create directory structure"""
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.current_model_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Directory structure ready")
        self.logger.info(f"  Inbox: {self.inbox_dir}")
        self.logger.info(f"  Control: {self.controller.control_dir}")
        self.logger.info(f"  Queue: {self.queue.queue_dir}")
        self.logger.info(f"  Current model: {self.current_model_dir}")
        self.logger.info(f"  Snapshots: {self.snapshots_dir}")
        self.logger.info(f"  Logs: {self.logs_dir}")

    def initialize_model(self):
        """Initialize current model from base or latest snapshot"""
        # Check if current_model exists
        if (self.current_model_dir / "adapter_config.json").exists():
            self.logger.info(f"Current model already exists: {self.current_model_dir}")
            return

        # Check for latest snapshot
        snapshots = sorted(self.snapshots_dir.glob("20*"))
        if snapshots:
            latest_snapshot = snapshots[-1]
            self.logger.info(f"Restoring from latest snapshot: {latest_snapshot}")
            shutil.copytree(latest_snapshot, self.current_model_dir, dirs_exist_ok=True)
            return

        # No snapshot, need base model
        if not self.config.get("model_path"):
            self.logger.error("No current model, no snapshots, and no model_path in config!")
            self.logger.error(f"Please set model_path in {self.config_file}")
            sys.exit(1)

        self.logger.info("No existing model found. Will start fresh on first training.")

    def should_create_snapshot(self) -> bool:
        """Check if we should create a daily snapshot"""
        today = datetime.now().date()

        # Already created snapshot today?
        if self.last_snapshot_date == today:
            return False

        # Check if we're past snapshot time
        snapshot_time = datetime.strptime(self.config["snapshot_time"], "%H:%M").time()
        current_time = datetime.now().time()

        if current_time >= snapshot_time:
            return True

        return False

    def create_snapshot(self):
        """Create daily snapshot of current model (latest checkpoint only)"""
        today = datetime.now().date()
        snapshot_dir = self.snapshots_dir / today.strftime("%Y-%m-%d")

        if snapshot_dir.exists():
            self.logger.info(f"Snapshot already exists: {snapshot_dir}")
            self.last_snapshot_date = today
            return

        # Copy only latest checkpoint + essential files
        self.logger.info(f"Creating daily snapshot: {snapshot_dir}")

        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Find latest checkpoint
            checkpoints = sorted([d for d in self.current_model_dir.glob("checkpoint-*") if d.is_dir()])
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                self.logger.info(f"   Copying latest checkpoint: {latest_checkpoint.name}")
                shutil.copytree(latest_checkpoint, snapshot_dir / latest_checkpoint.name)

            # Copy essential adapter files (not in checkpoints)
            essential_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "added_tokens.json",
                "chat_template.jinja",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt"
            ]

            for filename in essential_files:
                src = self.current_model_dir / filename
                if src.exists():
                    shutil.copy2(src, snapshot_dir / filename)

            self.last_snapshot_date = today
            self.logger.info(f"✅ Snapshot created: {snapshot_dir}")

            # Log snapshot size
            snapshot_size = sum(f.stat().st_size for f in snapshot_dir.rglob('*') if f.is_file())
            self.logger.info(f"   Size: {snapshot_size / 1024 / 1024:.1f} MB")

        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")

    def should_consolidate(self) -> bool:
        """Check if we should consolidate (merge adapter into base model)"""
        today = datetime.now().date()

        # Already consolidated today?
        if self.last_consolidation_date == today:
            return False

        # Check if we're past consolidation time (3 AM)
        consolidation_time = datetime.strptime("03:00", "%H:%M").time()
        current_time = datetime.now().time()

        if current_time >= consolidation_time:
            return True

        return False

    def consolidate_model(self):
        """Consolidate adapter into base model using external script"""
        today = datetime.now().date()

        # Check if we already consolidated today (from marker file)
        if self.consolidation_marker.exists():
            last_date_str = self.consolidation_marker.read_text().strip()
            try:
                last_date = datetime.fromisoformat(last_date_str).date()
                if last_date == today:
                    self.logger.info(f"Already consolidated today: {today}")
                    self.last_consolidation_date = today
                    return
            except:
                pass  # Invalid marker file, proceed with consolidation

        # Check if there's an adapter to consolidate
        adapter_file = self.current_model_dir / 'adapter_model.safetensors'
        if not self.current_model_dir.exists() or not adapter_file.exists():
            self.logger.info("No adapter to consolidate - skipping")
            return

        self.logger.info("=" * 80)
        self.logger.info("🔄 STARTING MODEL CONSOLIDATION")
        self.logger.info("=" * 80)

        try:
            # Run consolidation script
            import subprocess
            consolidate_script = self.base_dir / 'consolidate_model.py'

            result = subprocess.run(
                ['python3', str(consolidate_script), '--base-dir', str(self.base_dir)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                self.logger.info(result.stdout)
                self.logger.info("✅ Consolidation completed successfully")

                # Mark as consolidated today
                self.last_consolidation_date = today
                self.consolidation_marker.write_text(datetime.now().isoformat())
            else:
                self.logger.error(f"Consolidation failed with code {result.returncode}")
                self.logger.error(result.stderr)

        except subprocess.TimeoutExpired:
            self.logger.error("Consolidation timed out after 30 minutes")
        except Exception as e:
            self.logger.error(f"Failed to run consolidation: {e}")

        self.logger.info("=" * 80)

    def process_inbox_to_queue(self):
        """
        Scan inbox and move files to appropriate priority queue

        Files in inbox/priority/ → high priority
        Files in inbox/ → normal priority
        """
        # Check for priority inbox
        priority_inbox = self.inbox_dir / "priority"

        # Scan priority inbox first
        if priority_inbox.exists():
            for file_path in priority_inbox.glob("*.jsonl"):
                if file_path.is_file():
                    self.queue.add_to_queue(file_path, priority="high")
                    self.logger.info(f"🚀 High priority: {file_path.name}")

        # Scan normal inbox
        for file_path in self.inbox_dir.glob("*.jsonl"):
            if file_path.is_file():
                self.queue.add_to_queue(file_path, priority="normal")
                self.logger.info(f"📥 Normal priority: {file_path.name}")

    def train_on_file(self, data_file: Path, batch_number: int = None, batch_queue_size: int = None):
        """Train on a single data file (1 epoch) with control checks"""
        self.current_file = data_file.name

        self.logger.info("=" * 80)
        self.logger.info(f"Training on: {data_file.name}")
        if batch_number and batch_queue_size:
            self.logger.info(f"Batch: {batch_number}/{batch_queue_size}")
        self.logger.info("=" * 80)

        # Get file stats
        file_size = data_file.stat().st_size / 1024 / 1024
        self.logger.info(f"File size: {file_size:.1f} MB")

        # Count lines
        with open(data_file) as f:
            num_examples = sum(1 for _ in f)
        self.logger.info(f"Examples: {num_examples:,}")

        # Check for skip signal before starting
        if self.controller.check_skip():
            self.logger.warning(f"⏭️  SKIP signal detected - skipping {data_file.name}")
            self.controller.clear_skip()
            self.queue.mark_skipped(data_file, reason="User requested skip")
            return "skipped"

        # Create args for UltimateTrainer
        class Args:
            pass

        args = Args()
        args.dataset = str(data_file)

        # Determine which model to use
        if (self.current_model_dir / "adapter_config.json").exists():
            # Continue training existing adapter
            args.model = str(self.current_model_dir)
            self.logger.info(f"Continuing training on: {self.current_model_dir}")
        else:
            # Start fresh from base model
            args.model = self.config.get("model_path") or self.config["model_name"]
            self.logger.info(f"Starting fresh training from: {args.model}")

        args.output_dir = str(self.current_model_dir)

        # Training params from config
        args.epochs = 1  # ALWAYS 1 EPOCH
        args.batch_size = self.config["batch_size"]
        args.gradient_accumulation = self.config["gradient_accumulation"]
        args.learning_rate = self.config["learning_rate"]
        args.warmup_steps = self.config["warmup_steps"]
        args.lora_r = self.config["lora_r"]
        args.lora_alpha = self.config["lora_alpha"]
        args.eval_steps = self.config["eval_steps"]
        args.num_eval_samples = self.config["num_eval_samples"]
        args.save_steps = self.config["save_steps"]
        args.use_qlora = self.config.get("use_qlora", False)

        # Skip validation (assume data is pre-validated)
        args.skip_validation = True
        args.yes = True  # No prompts
        args.system_prompt = "You are a helpful assistant."

        # Batch context (for progress tracking)
        args.current_file = data_file.name
        args.batch_number = batch_number
        args.batch_queue_size = batch_queue_size

        # Train
        self.logger.info("Starting training...")
        start_time = time.time()

        try:
            self.logger.info("Initializing UltimateTrainer...")
            trainer = UltimateTrainer(args)

            self.logger.info("Running training...")
            success = trainer.run()

            elapsed = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed/60:.1f} minutes")

            if success:
                self.logger.info("✅ Training successful")
                return True
            else:
                self.logger.error("❌ Training failed (trainer.run() returned False)")
                self.logger.error("Check if there were validation errors or other issues")
                return False

        except Exception as e:
            self.logger.error(f"❌ Training error: {e}")
            import traceback
            # Log full traceback to file
            error_details = traceback.format_exc()
            self.logger.error(f"Full traceback:\n{error_details}")
            return False

    def cleanup_data_file(self, data_file: Path):
        """Delete training data after use"""
        try:
            data_file.unlink()
            self.logger.info(f"🗑️  Deleted: {data_file.name}")
        except Exception as e:
            self.logger.error(f"Failed to delete {data_file}: {e}")

    def handle_pause_state(self):
        """Wait in paused state until resume signal or stop signal"""
        self.logger.info("⏸️  PAUSED - Waiting for resume signal...")
        self.logger.info("   (Create control/.resume to continue)")
        self.paused = True

        # Update state
        self.controller.update_state(status="paused", current_file=self.current_file)

        while True:
            # Check for resume signal
            if self.controller.check_resume():
                self.logger.info("▶️  RESUME signal detected - continuing...")
                self.controller.clear_resume()
                self.controller.clear_pause()
                self.paused = False
                return "resume"

            # Check for stop signal while paused
            if self.controller.check_stop():
                self.logger.info("🛑 STOP signal during pause - exiting...")
                return "stop"

            # Sleep briefly
            time.sleep(5)

    def run(self):
        """Main daemon loop with full control system integration"""
        self.logger.info("=" * 80)
        self.logger.info("🤖 INTEGRATED TRAINING DAEMON STARTING")
        self.logger.info("=" * 80)
        self.logger.info("")

        # Setup
        self.setup_directories()
        self.initialize_model()

        self.logger.info("")
        self.logger.info("Configuration:")
        self.logger.info(f"  Model: {self.config.get('model_path') or self.config['model_name']}")
        self.logger.info(f"  Batch size: {self.config['batch_size']}")
        self.logger.info(f"  Learning rate: {self.config['learning_rate']}")
        self.logger.info(f"  Poll interval: {self.config['poll_interval']}s")
        self.logger.info(f"  Daily snapshot: {self.config['snapshot_time']}")
        self.logger.info("")
        self.logger.info("Control System:")
        self.logger.info(f"  Pause:  touch {self.controller.pause_signal}")
        self.logger.info(f"  Stop:   touch {self.controller.stop_signal}")
        self.logger.info(f"  Skip:   touch {self.controller.skip_signal}")
        self.logger.info(f"  Resume: touch {self.controller.resume_signal}")
        self.logger.info("")
        self.logger.info("Daemon is running...")
        self.logger.info(f"  Drop JSONL files in: {self.inbox_dir}")
        self.logger.info(f"  High priority: {self.inbox_dir}/priority/")
        self.logger.info("")

        # Update state to running
        self.controller.update_state(status="running", current_file=None)

        iteration = 0

        while True:
            iteration += 1

            # Check for stop signal
            if self.controller.check_stop():
                self.logger.info("🛑 STOP signal detected!")
                self.controller.clear_stop()
                self.controller.update_state(status="stopped", current_file=None)
                self.logger.info("Daemon stopped cleanly")
                break

            # Check for pause signal (not during training, between files)
            if self.controller.check_pause() and not self.current_file:
                result = self.handle_pause_state()
                if result == "stop":
                    break
                # If resumed, continue loop

            # Check if we should create daily snapshot
            if self.should_create_snapshot():
                self.create_snapshot()

            # Check if we should consolidate (when idle after 3 AM)
            # Only check when queue is empty to avoid interrupting training
            queue_status = self.queue.get_queue_status()
            if queue_status["total_queued"] == 0 and self.should_consolidate():
                self.consolidate_model()

            # Process inbox → move files to priority queues
            self.process_inbox_to_queue()

            # Get next file from queue (respects priority)
            next_file = self.queue.get_next_file()

            if next_file:
                queue_status = self.queue.get_queue_status()
                total_remaining = queue_status["total_queued"]

                self.logger.info(f"📋 Queue: {total_remaining} files remaining")

                # Train on file
                result = self.train_on_file(
                    next_file,
                    batch_number=1,
                    batch_queue_size=total_remaining + 1
                )

                # Handle result
                if result == "skipped":
                    # Already handled by train_on_file
                    pass
                elif result:
                    # Success - mark completed and cleanup
                    self.queue.mark_completed(next_file)
                    self.cleanup_data_file(next_file)
                else:
                    # Failed - mark failed, keep file
                    self.queue.mark_failed(next_file)
                    self.logger.warning(f"⚠️  Training failed, file marked as failed")

                # Reset current file
                self.current_file = None

                # Check for control signals after training
                if self.controller.check_pause():
                    result = self.handle_pause_state()
                    if result == "stop":
                        break

                if self.controller.check_stop():
                    break

            else:
                # No files in queue
                if iteration % 10 == 0:  # Every 10 iterations
                    self.logger.info(f"Queue empty. Waiting... (iteration {iteration})")

                # Update state to idle
                if not self.paused:
                    self.controller.update_state(status="idle", current_file=None)

            # Sleep until next check
            time.sleep(self.config["poll_interval"])


def main():
    parser = argparse.ArgumentParser(description="Integrated Training Daemon for RTX 3090")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/training",
        help="Base directory for training system (default: /training)"
    )

    args = parser.parse_args()

    daemon = IntegratedTrainingDaemon(Path(args.base_dir))
    daemon.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
