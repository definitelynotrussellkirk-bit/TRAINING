#!/usr/bin/env python3
"""
Training Daemon - Continuous Training Orchestrator

This module implements a long-running daemon that continuously monitors for training data,
orchestrates training jobs, manages model checkpoints, and handles system lifecycle.

Key Components:
    - TrainingDaemon: Main daemon class with lifecycle management
    - Queue integration: Uses TrainingQueue for priority-based processing
    - Control integration: Uses TrainingController for pause/resume/stop signals
    - Data management: Auto-generation via DataManager when queue is empty
    - Snapshot management: Daily model snapshots and periodic consolidation
    - Checkpoint retention: Automatic cleanup of old checkpoints

Directory Structure:
    base_dir/
        inbox/             - Drop zone for new JSONL files
        queue/             - Priority queues (high/normal/low/processing/failed)
        current_model/     - Active model being trained
        snapshots/         - Daily snapshots (YYYY-MM-DD/)
        logs/              - Training and daemon logs
        status/            - training_status.json and other status files
        control/           - Control signals (state.json, .stop, .pause, .skip)
        config.json        - Training configuration
        .daemon.pid        - PID lock file (prevents multiple daemons)
        .stop              - Stop signal file (daemon exits cleanly)

Daemon Lifecycle:
    1. Startup: Setup directories, acquire PID lock, recover from crashes
    2. Main Loop:
        a. Check stop signal → exit cleanly
        b. Run checkpoint retention (cleanup old checkpoints)
        c. Check snapshot schedule → create daily snapshot
        d. Check consolidation schedule → consolidate checkpoints when idle
        e. Process inbox → move files to queue
        f. Process queue → train on files by priority
        g. Auto-generate data if queue empty
        h. Sleep poll_interval seconds → repeat
    3. Shutdown: Release PID lock, cleanup state

Integration Points:
    - core/training_queue.py: Priority queue management
    - core/training_controller.py: Control signals (pause/resume/stop)
    - core/train.py (UltimateTrainer): Actual training execution
    - data_manager.py: Auto-generation with quality testing
    - management/checkpoint_retention.py: Checkpoint cleanup

Usage:
    # Start daemon
    python3 training_daemon.py --base-dir /path/to/training

    # Drop training data
    cp my_data.jsonl /path/to/training/inbox/

    # Control daemon
    python3 core/training_controller.py pause
    python3 core/training_controller.py resume
    python3 core/training_controller.py stop

    # Stop daemon
    touch /path/to/training/.stop
"""

import os
import sys
import time
import json
import shutil
import argparse
import signal
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import logging
from urllib import request, error

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "management"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

# Extracted daemon services (TASK005)
from daemon.pid_manager import PIDManager
from daemon.file_watcher import InboxFlattener
from daemon.snapshot_service import SnapshotService, SnapshotConfig

# Unified validation (TASK008 + spec validation)
from validation.validator import DataValidator, ValidationLevel
from validation.spec import SpecValidator, SpecValidationError, DATASET_SPECS

# Checkpoint retention (new system)
from retention_service import RetentionService

from train import UltimateTrainer
from training_queue import TrainingQueue
from training_controller import TrainingController
from atomic_ops import write_json_atomic, safe_file_operation

# Data Manager (for auto-generation with quality testing)
sys.path.insert(0, str(Path(__file__).parent.parent / "data_manager"))
from data_manager import DataManager

# Add format variety support
import random
REPO_ROOT = Path(__file__).parent.parent / "singleSKILL"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from skill_syllo_variant.scripts.export_training_data import (
        DEFAULT_OUTPUT_VARIANT_DISTRIBUTION,
        OUTPUT_VARIANTS,
        choose_output_variant
    )
    FORMAT_VARIETY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Format variety not available: {e}")
    FORMAT_VARIETY_AVAILABLE = False
    DEFAULT_OUTPUT_VARIANT_DISTRIBUTION = {"standard": 1.0}
    OUTPUT_VARIANTS = {"standard": {"prompt_note": "", "transform": lambda p: p}}
    choose_output_variant = lambda rng, dist: "standard"


class TrainingDaemon:
    """
    Continuous training daemon that orchestrates the complete training lifecycle.

    Responsibilities:
        - Monitor inbox directory for new training data
        - Process files through priority queue (high/normal/low)
        - Execute training jobs via UltimateTrainer
        - Handle pause/resume/stop control signals
        - Create daily snapshots of trained models
        - Consolidate checkpoints when idle (reduce disk usage)
        - Enforce checkpoint retention policies (cap disk usage)
        - Auto-generate training data when queue empty
        - Recover from crashes (orphaned files, stale state)
        - Prevent multiple daemon instances (PID lock)

    Data Flow:
        1. Startup:
            → Setup directories (inbox, queue, logs, etc.)
            → Load config.json
            → Validate config (catch errors before training)
            → Acquire PID lock (.daemon.pid)
            → Recover from crashes (move processing → failed, cleanup state)

        2. Main Loop (every poll_interval seconds):
            → Check .stop file → exit cleanly
            → Run checkpoint retention (cleanup old checkpoints if caps exceeded)
            → Check snapshot schedule (daily at configured time) → create snapshot
            → Check consolidation schedule (after 3 AM when idle) → consolidate
            → Flatten inbox (move .jsonl files from subdirs to root)
            → Process inbox → move files to queue (high/normal/low)
            → Get queue status (counts by priority)
            → While queue not empty:
                - Check control signals (.stop, .pause, .skip)
                - Get next file by priority (high → normal → low)
                - Check disk space (skip if insufficient)
                - Train on file (via UltimateTrainer)
                - Mark completed (delete file) or failed (keep file)
            → If queue empty → maybe auto-generate data
            → Sleep poll_interval seconds

        3. Shutdown:
            → Release PID lock (.daemon.pid)
            → Cleanup state

    Lifecycle States (via TrainingController):
        - idle: Waiting for files
        - training: Actively training on a file
        - paused: Paused (waiting for resume signal)
        - stopped: Stopped by user (exit after current batch)

    Attributes:
        base_dir: Root directory for all training operations
        inbox_dir: Drop zone for new JSONL files (base_dir/inbox)
        current_model_dir: Active model directory (base_dir/current_model)
        snapshots_dir: Daily snapshots directory (base_dir/snapshots)
        logs_dir: Log files directory (base_dir/logs)
        config_file: Training configuration (base_dir/config.json)
        stop_file: Stop signal file (base_dir/.stop)
        pid_file: PID lock file (base_dir/.daemon.pid)
        config: Loaded configuration dict
        queue: TrainingQueue instance (priority queue management)
        controller: TrainingController instance (control signals)
        data_manager: DataManager instance (auto-generation with quality testing)
        logger: Python logger instance
        last_snapshot_date: Date of last snapshot (YYYY-MM-DD)
        last_consolidation_date: Date of last consolidation
        last_autogen_time: Timestamp of last auto-generation
        shutdown_requested: Flag set by signal handlers (SIGTERM, SIGINT)

    Example:
        # Start daemon
        daemon = TrainingDaemon(Path("/training"))
        daemon.run()  # Runs until .stop file created or signal received

        # In another terminal, drop training data
        $ cp my_data.jsonl /training/inbox/

        # Control daemon
        $ python3 core/training_controller.py pause
        $ python3 core/training_controller.py resume
        $ python3 core/training_controller.py stop

        # Stop daemon cleanly
        $ touch /training/.stop
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.inbox_dir = self.base_dir / "inbox"
        self.current_model_dir = self.base_dir / "current_model"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.logs_dir = self.base_dir / "logs"
        self.config_file = self.base_dir / "config.json"
        self.stop_file = self.base_dir / ".stop"
        self.pid_file = self.base_dir / ".daemon.pid"

        # Setup logging
        self.setup_logging()

        # Load config
        self.config = self.load_config()
        # Allow preset-specific model dir overrides
        if self.config.get("current_model_dir"):
            self.current_model_dir = Path(self.config["current_model_dir"])

        # Track last snapshot date
        self.last_snapshot_date = None

        # Track last consolidation date
        self.last_consolidation_date = None
        self.consolidation_marker = self.base_dir / ".last_consolidation"

        # Track API autogen cooldown
        self.last_autogen_time = 0.0

        # RNG for format variety
        self.format_rng = random.Random()

        # Initialize queue and control systems
        self.queue = TrainingQueue(str(self.base_dir))
        self.controller = TrainingController(str(self.base_dir))

        # Initialize Data Manager (handles auto-generation + quality testing)
        self.data_manager = DataManager(self.base_dir, self.config)

        # Extracted services (TASK005)
        self.pid_manager = PIDManager(self.pid_file)
        self.inbox_flattener = InboxFlattener(self.inbox_dir)
        self.quick_validator = DataValidator()  # For QUICK content checks on inbox files
        self.spec_validator = SpecValidator(
            registry=DATASET_SPECS,
            allow_default=True,  # Use chat_sft_v1 if no schema_id specified
            strict_mode=False    # Don't require metadata keys yet
        )
        self.snapshot_service = SnapshotService(SnapshotConfig(
            checkpoints_dir=self.current_model_dir,
            snapshots_dir=self.snapshots_dir,
            snapshot_time=self.config.get("snapshot_time", "02:00")
        ))

        # Checkpoint retention (new RetentionManager-based system)
        self.last_retention_check = 0
        retention_config = self.config.get("retention", {})
        self.retention_service = RetentionService(
            output_dir=self.current_model_dir,
            config=retention_config,
            custom_logger=None  # Will be set after logging is configured
        )

        # Shutdown flag for signal handling
        self.shutdown_requested = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

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
                "batch_size": 1,
                "gradient_accumulation": 16,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "load_in_4bit": False,  # 4-bit quantization (reduces VRAM but lower quality)
                "eval_steps": 100,
                "num_eval_samples": 5,
                "save_steps": 500,
                "poll_interval": 60,  # Check inbox every 60 seconds
                "snapshot_time": "02:00",  # Daily snapshot at 2 AM
                "auto_generate": {
                    "enabled": False,
                    "host": "127.0.0.1",
                    "port": 8091,
                    "count": 20000,
                    "priority": "normal",
                    "threshold": 0,
                    "seed": None,
                    "cooldown_sec": 120,
                    "payload": {}
                }
            }

            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)

            self.logger.info(f"Created default config at {self.config_file}")
            self.logger.warning(f"IMPORTANT: Edit {self.config_file} and set model_path!")

            return default_config

        with open(self.config_file) as f:
            config = json.load(f)

        config.setdefault("auto_generate", {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 8091,
            "count": 20000,
            "priority": "normal",
            "threshold": 0,
            "seed": None,
            "cooldown_sec": 120,
            "payload": {}
        })

        self.logger.info(f"Loaded config from {self.config_file}")

        # Validate config before using it
        self.validate_config(config)

        return config

    def validate_config(self, config: dict) -> None:
        """
        Validate configuration and fail fast on bad values.

        This is a strict validator: if any critical issues are found, a
        ValueError is raised and the daemon refuses to start until
        config.json is fixed.

        Args:
            config: Configuration dictionary loaded from config.json

        Raises:
            ValueError: If any validation errors are found

        Side Effects:
            - Logs all validation warnings and errors
            - Raises ValueError on errors (daemon will not start)

        Validation checks (currently implemented):
            - model_path/base_model path exists (if specified)
            - max_length in range [128, 32768]
            - learning_rate in range [1e-6, 1e-2]
            - batch_size in range [1, 128]
            - gradient_accumulation in range [1, 128]
            - lora_r in range [0, 1024] (if specified; 0 = full-model, no LoRA)
            - snapshot_time format "HH:MM" (if specified)
            - auto_generate.* fields are sane when enabled

        Example:
            config = {"batch_size": 256, "learning_rate": 0.1}
            self.validate_config(config)
            # Raises: ValueError("Config validation failed: ...")
        """
        errors = []

        # Check model path exists (accepts both model_path and legacy base_model)
        model_path = config.get("model_path") or config.get("base_model")
        if model_path:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                errors.append(f"Model path does not exist: {model_path}")

        # Check max_length is reasonable (if specified)
        if 'max_length' in config:
            max_len = config['max_length']
            if not (128 <= max_len <= 32768):
                errors.append(f"max_length out of range (128-32768): {max_len}")

        # Check learning rate is reasonable
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not (1e-6 <= lr <= 1e-2):
                errors.append(f"Learning rate out of range (1e-6 to 1e-2): {lr}")

        # Check batch size
        if 'batch_size' in config:
            bs = config['batch_size']
            if not (bs > 0 and bs <= 128):
                errors.append(f"Batch size out of range (1-128): {bs}")

        # Check gradient accumulation
        if 'gradient_accumulation' in config:
            ga = config['gradient_accumulation']
            if not (ga > 0 and ga <= 128):
                errors.append(f"Gradient accumulation out of range (1-128): {ga}")

        # Check LoRA rank (if specified)
        # lora_r=0 means full model training (no LoRA)
        if 'lora_r' in config:
            lora_r = config['lora_r']
            if not (lora_r >= 0 and lora_r <= 1024):
                errors.append(f"LoRA rank out of range (0-1024): {lora_r}")

        # Check snapshot_time format (if specified)
        snapshot_time = config.get("snapshot_time")
        if snapshot_time is not None:
            try:
                datetime.strptime(snapshot_time, "%H:%M")
            except ValueError:
                errors.append(f"Invalid snapshot_time '{snapshot_time}', expected HH:MM format")

        auto_cfg = config.get("auto_generate", {}) or {}
        if auto_cfg.get("enabled"):
            host = auto_cfg.get("host")
            port = auto_cfg.get("port")
            count = auto_cfg.get("count")
            if not host:
                errors.append("auto_generate.host is required when enabled")
            if not isinstance(port, int) or port <= 0:
                errors.append(f"auto_generate.port invalid: {port}")
            if not isinstance(count, int) or count <= 0:
                errors.append(f"auto_generate.count invalid: {count}")
            priority = auto_cfg.get("priority", "normal")
            if priority not in {"high", "normal", "low"}:
                errors.append(f"auto_generate.priority invalid: {priority}")

        if errors:
            self.logger.error("❌ CONFIG VALIDATION FAILED!")
            for error in errors:
                self.logger.error(f"   - {error}")
            self.logger.error(f"\nPlease fix {self.config_file} and restart daemon")
            raise ValueError(f"Config validation failed: {errors}")

        self.logger.info("✅ Config validation passed")

    def setup_directories(self):
        """Create directory structure"""
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.current_model_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Directory structure ready")
        self.logger.info(f"  Inbox: {self.inbox_dir}")
        self.logger.info(f"  Current model: {self.current_model_dir}")
        self.logger.info(f"  Snapshots: {self.snapshots_dir}")
        self.logger.info(f"  Logs: {self.logs_dir}")

    def initialize_model(self):
        """Initialize current model from base or latest snapshot"""
        # Check if current_model exists (HF checkpoint files for full model training)
        config_file = self.current_model_dir / "config.json"
        if config_file.exists():
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
        # Delegate to SnapshotService (TASK005)
        # Sync last_snapshot_date with service
        if self.last_snapshot_date:
            self.snapshot_service.last_snapshot_date = self.last_snapshot_date
        return self.snapshot_service.should_create_snapshot()

    def verify_snapshot(self, snapshot_dir: Path) -> bool:
        """Verify snapshot integrity"""
        # Delegate to SnapshotService (TASK005)
        return self.snapshot_service.verify_snapshot(snapshot_dir)

    def create_snapshot(self):
        """Create daily snapshot of current model (latest checkpoint only)"""
        # Delegate to SnapshotService (TASK005)
        result = self.snapshot_service.create_snapshot()

        if result.success:
            self.last_snapshot_date = self.snapshot_service.last_snapshot_date
            self.logger.info(f"✅ Snapshot created: {result.snapshot_path}")
            if result.checkpoint_name:
                self.logger.info(f"   Checkpoint: {result.checkpoint_name}")
            self.logger.info(f"   Size: {result.size_bytes / 1024 / 1024:.1f} MB")
        else:
            self.logger.error(f"Failed to create snapshot: {result.error}")

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

        # Check if there's a trained model to consolidate
        # For full model training, check for checkpoints
        checkpoints = list(self.current_model_dir.glob("checkpoint-*"))
        if not self.current_model_dir.exists() or not checkpoints:
            self.logger.info("No checkpoints to consolidate - skipping")
            return

        self.logger.info("=" * 80)
        self.logger.info("🔄 STARTING MODEL CONSOLIDATION")
        self.logger.info("=" * 80)

        try:
            # Run consolidation script
            import subprocess
            consolidate_script = self.base_dir / 'consolidate_model.py'

            # Generate description with current date
            description = f"Daily consolidation {datetime.now().strftime('%Y-%m-%d')}"

            result = subprocess.run(
                ['python3', str(consolidate_script), '--base-dir', str(self.base_dir),
                 '--description', description],
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

    def flatten_inbox(self):
        """Move any .jsonl files from subdirectories to inbox root"""
        # Delegate to InboxFlattener (TASK005)
        moved_count = self.inbox_flattener.flatten()
        if moved_count > 0:
            self.logger.info(f"✅ Flattened {moved_count} file(s) from subdirectories")

    def get_inbox_files(self):
        """Get all JSONL files in inbox"""
        return sorted(self.inbox_dir.glob("*.jsonl"))

    def quick_validate_inbox_files(self) -> int:
        """
        Run QUICK validation on inbox files, rejecting obviously bad ones.

        This catches schema errors (invalid JSON, missing fields) early,
        before files enter the processing queue. More comprehensive
        validation happens in validate_data_before_training().

        Returns:
            Number of files rejected
        """
        rejected = 0
        for inbox_file in self.get_inbox_files():
            result = self.quick_validator.validate(inbox_file, ValidationLevel.QUICK)
            if not result.should_proceed():
                # Move to failed queue immediately
                error_msg = "; ".join(result.errors[:3])  # First 3 errors
                self.logger.warning(f"Quick validation failed: {inbox_file.name}")
                self.logger.warning(f"  Errors: {error_msg}")
                try:
                    failed_path = self.base_dir / "queue" / "failed" / inbox_file.name
                    failed_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(inbox_file), str(failed_path))
                    self.logger.info(f"  Moved to: {failed_path}")
                    rejected += 1
                except Exception as e:
                    self.logger.error(f"  Failed to move: {e}")
        return rejected

    def validate_data_before_training(self, data_file: Path) -> bool:
        """
        GUARDRAIL: Validate training data against config settings

        Checks:
        - Tokenize sample of examples
        - Compare lengths against max_length setting
        - Warn if data will be truncated
        - Optionally auto-adjust config

        Returns: True if data is valid, False if issues found
        """
        try:
            from transformers import AutoTokenizer

            self.logger.info("🔍 Validating data against config...")

            # QUARANTINE: if file already failed recently, skip
            failed_path = self.base_dir / "queue" / "failed" / data_file.name
            if failed_path.exists():
                self.logger.error(f"❌ File previously failed; skipping: {data_file.name}")
                return False

            # Load tokenizer
            model_path = self.config.get("base_model") or self.config.get("model_path") or self.config["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Sample and tokenize examples (with output tracking)
            full_lengths = []
            output_lengths = []
            schema_errors = 0
            sample_size = min(100, sum(1 for _ in open(data_file)))  # Sample up to 100 examples

            with open(data_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break

                    try:
                        data = json.loads(line.strip())

                        # Build conversation text
                        if 'messages' in data:
                            messages = data['messages']
                            if not isinstance(messages, list) or not messages:
                                schema_errors += 1
                                continue
                            roles = {m.get('role') for m in messages}
                            if not {'user', 'assistant'} <= roles:
                                schema_errors += 1
                                continue

                            # Separate prompt and output
                            prompt_text = ""
                            output_text = ""

                            for msg in messages:
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if not isinstance(content, str):
                                    schema_errors += 1
                                    break

                                if role == 'assistant':
                                    output_text = content  # Last assistant message
                                else:
                                    prompt_text += f"{role}: {content}\n"

                            # Tokenize full conversation
                            full_text = prompt_text + f"assistant: {output_text}\n"
                            full_tokens = tokenizer.encode(full_text)
                            full_lengths.append(len(full_tokens))

                            # Tokenize output separately
                            if output_text:
                                output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
                                output_lengths.append(len(output_tokens))

                        elif 'text' in data:
                            text = data['text']
                            tokens = tokenizer.encode(text)
                            full_lengths.append(len(tokens))
                        else:
                            continue

                    except Exception as e:
                        self.logger.warning(f"   Skipping line {i+1}: {e}")
                        continue

            if not full_lengths:
                self.logger.error("   ❌ No valid examples found in data")
                return False
            if schema_errors > 0:
                self.logger.error(f"   ❌ Schema errors detected in sample (count={schema_errors}); quarantining file")
                return False

            # Compute statistics for full conversations
            full_lengths.sort()
            max_len = max(full_lengths)
            p95_len = full_lengths[int(len(full_lengths) * 0.95)]
            p99_len = full_lengths[int(len(full_lengths) * 0.99)]
            mean_len = sum(full_lengths) / len(full_lengths)

            self.logger.info(f"   Sampled {len(full_lengths)} examples:")
            self.logger.info(f"   📋 FULL CONVERSATIONS:")
            self.logger.info(f"      Max: {max_len} tokens | Mean: {mean_len:.1f} | p95: {p95_len} | p99: {p99_len}")

            # Output-specific statistics
            if output_lengths:
                output_lengths.sort()
                out_max = max(output_lengths)
                out_mean = sum(output_lengths) / len(output_lengths)
                out_p95 = output_lengths[int(len(output_lengths) * 0.95)]
                out_p99 = output_lengths[int(len(output_lengths) * 0.99)]

                self.logger.info(f"   🤖 ASSISTANT OUTPUTS:")
                self.logger.info(f"      Max: {out_max} tokens | Mean: {out_mean:.1f} | p95: {out_p95} | p99: {out_p99}")

            # Check against config
            config_max = self.config.get('max_length', 2048)
            self.logger.info(f"   Config max_length: {config_max}")

            # Validate FULL conversations
            has_issues = False

            if max_len > config_max:
                self.logger.warning(f"   ⚠️  WARNING: Longest example ({max_len} tokens) exceeds max_length ({config_max})")
                self.logger.warning(f"   ⚠️  Full conversations will be truncated!")

            if p95_len > config_max:
                self.logger.error(f"   ❌ CRITICAL: 95% of full conversations exceed max_length!")
                self.logger.error(f"   ❌ Recommended: Set max_length to at least {p99_len}")
                self.logger.error(f"   ❌ Either:")
                self.logger.error(f"       1. Update config.json: \"max_length\": {p99_len}")
                self.logger.error(f"       2. Run: python3 validate_data.py --auto-adjust")
                has_issues = True

            # Validate OUTPUTS specifically (NEW!)
            if output_lengths:
                if out_max > config_max:
                    self.logger.error(f"   🚨 CRITICAL: Assistant outputs exceed max_length!")
                    self.logger.error(f"   🚨 Max output: {out_max} tokens > max_length: {config_max}")
                    self.logger.error(f"   🚨 RESPONSES ARE BEING TRUNCATED!")
                    has_issues = True

                if out_p95 > config_max * 0.8:
                    self.logger.warning(f"   ⚠️  Large outputs detected: p95={out_p95} ({int(out_p95/config_max*100)}% of max_length)")

                if out_p99 > config_max * 0.9:
                    self.logger.warning(f"   ⚠️  Some outputs very close to limit: p99={out_p99} ({int(out_p99/config_max*100)}% of max_length)")

            if has_issues:
                return False

            if config_max > p99_len * 1.5:
                self.logger.info(f"   💡 NOTE: max_length could be reduced to ~{p99_len} to save memory")

            self.logger.info("   ✅ Data validation passed")
            return True

        except Exception as e:
            self.logger.error(f"   ❌ Validation error: {e}")
            self.logger.warning("   ⚠️  Proceeding anyway (validation is best-effort)")
            return True  # Don't block training on validation errors

    def train_on_file(self, data_file: Path, batch_number: int = None, batch_queue_size: int = None) -> bool:
        """
        Execute training on a single JSONL data file for one epoch.

        This is the core training execution method. It validates the data file,
        sets up the UltimateTrainer, executes training, handles errors, and
        reports training results.

        Args:
            data_file: Path to JSONL training data file
            batch_number: Optional batch number for logging (e.g., "3/10")
            batch_queue_size: Optional total queue size for logging

        Returns:
            True if training completed successfully, False if failed

        Data Flow:
            1. Validate file:
                - Check file size (max 10GB)
                - Check JSON line sizes (max 100MB per line)
                - Count examples (reject if 0)
                - Validate data against config (prompt/response lengths, etc.)
            2. Setup UltimateTrainer:
                - Create Args object with paths (dataset, output_dir, config)
                - Determine model (continue from current_model or start from base)
                - Choose format variant (if format variety enabled)
            3. Execute training:
                - Call trainer.train() (delegates to core/train.py)
                - Monitor progress via TrainingStatusWriter
                - Handle OOM errors → reduce batch size suggestion
                - Handle control signals (skip, pause, stop)
            4. Post-training:
                - Log training summary (examples, loss, time, etc.)
                - Compute batch size suggestions
                - Cleanup GPU memory
            5. Report result:
                - Return True if successful
                - Return False if failed (file moved to queue/failed)

        Side Effects:
            - Trains model (GPU operations)
            - Writes checkpoints to current_model/
            - Writes training_status.json
            - Logs training metrics
            - Clears GPU memory after training

        Raises:
            Does not raise exceptions - catches all errors and returns False

        Safety Checks:
            - File size limits (10GB max, 100MB per JSON line)
            - Empty file rejection
            - Prompt/response length validation against config.max_length
            - Truncation detection (rejects if responses truncated)

        Example:
            # Train on a single file
            success = daemon.train_on_file(
                Path("/training/queue/high/syllo_hard_1000.jsonl"),
                batch_number=3,
                batch_queue_size=10
            )
            if success:
                # File deleted, training complete
            else:
                # File moved to queue/failed, review logs
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Training on: {data_file.name}")
        if batch_number and batch_queue_size:
            self.logger.info(f"Batch: {batch_number}/{batch_queue_size}")
        self.logger.info("=" * 80)

        # Get file stats
        file_size = data_file.stat().st_size / 1024 / 1024
        self.logger.info(f"File size: {file_size:.1f} MB")

        # CRITICAL FIX #3: JSON size limits (prevent DoS)
        MAX_JSON_LINE_SIZE = 100 * 1024 * 1024  # 100MB per line
        MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB total

        if data_file.stat().st_size > MAX_FILE_SIZE:
            self.logger.error(f"❌ File too large: {file_size:.1f}MB (max 10GB)")
            return False

        # Count lines and validate JSON line sizes
        num_examples = 0
        try:
            with open(data_file, 'rb') as f:
                for line_num, line in enumerate(f, 1):
                    if len(line) > MAX_JSON_LINE_SIZE:
                        self.logger.error(f"❌ Line {line_num} too large: {len(line)/1024/1024:.1f}MB (max 100MB)")
                        return False
                    num_examples += 1
        except Exception as e:
            self.logger.error(f"❌ Error reading file: {e}")
            return False

        self.logger.info(f"Examples: {num_examples:,}")

        # CRITICAL FIX: Empty file check
        if num_examples == 0:
            self.logger.error(f"❌ Empty file: {data_file.name} (0 examples)")
            return False

        # SPEC VALIDATION: Ensure job maps to a known schema (deny-by-default)
        try:
            job_config = {
                "dataset_path": str(data_file),
                "base_model": self.config.get("model_path") or self.config.get("base_model"),
                "schema_id": self.config.get("schema_id"),  # Optional: from config.json
            }
            spec = self.spec_validator.validate_job(job_config)
            self.logger.info(f"📋 Spec validation passed: {spec.id} ({spec.kind})")
        except SpecValidationError as e:
            self.logger.error(f"❌ Spec validation failed: {e}")
            self.logger.error("   Add 'schema_id' to config.json or register a new spec")
            return False

        # GUARDRAIL: Validate data against config
        if not self.validate_data_before_training(data_file):
            self.logger.error("❌ Data validation failed - aborting training")
            self.logger.error("   Fix config or data, then try again")
            return False

        # Create minimal args for UltimateTrainer
        # TrainerConfig will load hyperparams from config.json
        class Args:
            pass

        args = Args()

        # Routing info only (where to read/write)
        args.dataset = str(data_file)
        args.output_dir = str(self.current_model_dir)

        # Point to config.json for TrainerConfig to load hyperparams
        args.config = str(self.config_file)

        # Validate current_model directory
        def is_valid_model_dir(path):
            """Check if directory contains required HF model files."""
            if not path.exists():
                return False
            required = ["config.json", "tokenizer.json"]
            return all((path / f).exists() for f in required)

        # Determine which model to use
        config_file = self.current_model_dir / "config.json"
        if config_file.exists():
            # Continue training existing model
            args.model = str(self.current_model_dir)
            self.logger.info(f"Continuing training on: {self.current_model_dir}")
        else:
            # Start fresh from base model
            args.model = self.config.get("model_path") or self.config["model_name"]

            # Initialize current_model_dir from base if invalid/empty
            if not is_valid_model_dir(self.current_model_dir):
                if self.current_model_dir.exists():
                    self.logger.warning("Current model dir incomplete/corrupt; will replace with fresh copy")
                    shutil.rmtree(self.current_model_dir)

                self.logger.info(f"Copying base model from: {args.model}")
                shutil.copytree(args.model, self.current_model_dir, dirs_exist_ok=True)
                self.logger.info(f"✅ Model copied to: {self.current_model_dir}")
            else:
                self.logger.info(f"Using existing model: {self.current_model_dir}")

            args.model = str(self.current_model_dir)

        # These are daemon-specific overrides (not hyperparams)
        args.epochs = 1  # ALWAYS 1 EPOCH (daemon trains 1 epoch per file)
        args.skip_validation = True  # Assume data is pre-validated
        args.yes = True  # No prompts
        args.system_prompt = "You are a helpful assistant."

        # Control/routing params (not hyperparams, needed by trainer logic)
        args.warmup_steps = self.config.get("warmup_steps", 100)
        args.log_steps = self.config.get("log_steps", 10)
        args.eval_steps = self.config.get("eval_steps", 500)
        args.save_steps = self.config.get("save_steps", 1000)
        args.num_eval_samples = self.config.get("num_eval_samples", 2)
        args.load_in_4bit = self.config.get("load_in_4bit", False)

        # Batch context for progress tracking
        args.current_file = data_file.name
        args.batch_number = batch_number
        args.batch_queue_size = batch_queue_size

        # Train
        self.logger.info("Starting training...")
        self.logger.info(f"🧭 Model dir: {self.current_model_dir}")
        start_time = time.time()

        try:
            self.logger.info("Initializing UltimateTrainer...")

            # CRITICAL FIX #6: GPU crash detection
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Test GPU accessibility
            except RuntimeError as e:
                if "CUDA" in str(e) or "GPU" in str(e):
                    self.logger.error("❌ GPU driver crashed or unavailable!")
                    self.logger.error(f"   Error: {e}")
                    self.logger.error("   Try: sudo nvidia-smi or restart system")
                    return False
                raise

            trainer = UltimateTrainer(args, controller=self.controller)

            self.logger.info("Running training...")
            success = trainer.run()
            summary = getattr(trainer, 'training_summary', None)
            if summary:
                summary.setdefault('dataset', data_file.name)
                self.log_run_summary(summary)

            elapsed = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed/60:.1f} minutes")

            if success:
                self.logger.info("✅ Training successful")

                # GUARDRAIL: Clean up GPU memory after training
                # Part of Phase 1 from CRITICAL_EDGE_CASES_AND_GUARDRAILS.md
                self.cleanup_gpu_memory()

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

    def log_run_summary(self, summary: dict):
        try:
            history_path = self.logs_dir / "run_history.jsonl"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")
            self.logger.info(f"📈 Logged run summary (dataset={summary.get('dataset')})")

            suggestion = self.compute_batch_suggestion(summary)
            if suggestion:
                self.logger.info(
                    "⚙️ Throughput advisor: observed %.2f GB at batch %s -> suggested batch %s (target %.1f GB)",
                    suggestion['observed_mem_gb'],
                    summary.get('batch_size'),
                    suggestion['recommended_batch_size'],
                    suggestion['target_mem_gb']
                )
        except Exception as e:
            self.logger.error(f"Failed to log run summary: {e}")

    def compute_batch_suggestion(self, summary: dict):
        peak = summary.get('gpu_peak_alloc_gb')
        batch = summary.get('batch_size')
        if not peak or not batch:
            return None
        try:
            peak = float(peak)
            batch = int(batch)
        except (TypeError, ValueError):
            return None
        if peak <= 0 or batch <= 0:
            return None

        target_mem = 21.0  # Aim for 19–22 GB usage
        ratio = target_mem / peak
        suggested = max(1, min(int(round(batch * ratio)), batch * 4))
        if suggested == batch:
            return None
        return {
            'observed_mem_gb': peak,
            'target_mem_gb': target_mem,
            'recommended_batch_size': suggested
        }

    def cleanup_data_file(self, data_file: Path):
        """Delete training data after use"""
        try:
            data_file.unlink()
            self.logger.info(f"🗑️  Deleted: {data_file.name}")
        except Exception as e:
            self.logger.error(f"Failed to delete {data_file}: {e}")

    def cleanup_gpu_memory(self):
        """Clean up GPU memory after training to prevent OOM errors.

        GUARDRAIL: Prevents GPU OOM when training multiple files sequentially.
        Part of Phase 1 from CRITICAL_EDGE_CASES_AND_GUARDRAILS.md

        Problem: GPU memory accumulates between training runs, causing OOM
        Solution: Force garbage collection + clear PyTorch cache
        """
        try:
            import gc
            import torch

            # Force Python garbage collection
            gc.collect()

            # Clear PyTorch GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all GPU operations to complete

                # Log memory state for monitoring
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9

                self.logger.info(f"🧹 GPU Memory cleaned up:")
                self.logger.info(f"   Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
                self.logger.info(f"   Cached: {cached:.2f} GB")

                # WARNING if still using >50% memory after cleanup
                if allocated > total * 0.5:
                    self.logger.warning(f"⚠️  GPU memory still high after cleanup: {allocated:.2f} GB")
                    self.logger.warning(f"   Consider restarting daemon if OOM occurs")
            else:
                self.logger.info("🧹 Cleaned up system memory (no GPU available)")

        except Exception as e:
            self.logger.error(f"Failed to cleanup GPU memory: {e}")
            # Don't fail training just because cleanup failed
            pass

    def should_stop(self) -> bool:
        """Check if daemon should stop"""
        return (self.stop_file.exists() or
                self.shutdown_requested or
                self.controller.should_stop_after_batch())

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        sig_name = signal.Signals(signum).name
        self.logger.info(f"⚠️  Received {sig_name} - will stop after current batch")
        self.shutdown_requested = True

    def acquire_lock(self):
        """Acquire PID file lock - prevents multiple daemons"""
        # Delegate to PIDManager (TASK005)
        if not self.pid_manager.acquire():
            running_pid = self.pid_manager.get_running_pid()
            self.logger.error(f"❌ Another daemon is running (PID {running_pid})")
            self.logger.error("   Stop it first or remove .daemon.pid if stale")
            sys.exit(1)

    def release_lock(self):
        """Release PID file lock"""
        # Delegate to PIDManager (TASK005)
        self.pid_manager.release()

    def recover_orphaned_files(self):
        """Move orphaned files from processing/ back to normal queue on startup"""
        processing_files = list(self.queue.processing.glob("*.jsonl"))

        if processing_files:
            self.logger.warning(f"⚠️  Found {len(processing_files)} orphaned files from previous crash")
            for file_path in processing_files:
                target = self.queue.normal_priority / file_path.name
                shutil.move(str(file_path), str(target))
                self.logger.info(f"   Recovered: {file_path.name}")
            self.logger.info("✅ Crash recovery complete")

    def cleanup_stale_state(self):
        """Clean up stale state from previous crash"""
        state = self.controller._load_state()

        if state.get("status") == "training":
            self.logger.warning("⚠️  Previous daemon crashed while training")
            self.controller.update_state("idle", reason="Recovered from crash")

        # Clear any stale signals
        self.controller.clear_signals()
        self.logger.info("✅ State cleanup complete")

    def check_disk_space(self) -> bool:
        """Check if enough disk space available"""
        stat = os.statvfs(self.base_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        if free_gb < 10:
            self.logger.error(f"❌ CRITICAL: Only {free_gb:.1f}GB free disk space")
            self.logger.error("   Need at least 10GB for safe checkpoint saves")
            return False

        if free_gb < 50:
            self.logger.warning(f"⚠️  Low disk space: {free_gb:.1f}GB free")

        return True

    def run_checkpoint_retention(self, force: bool = False):
        """
        Enforce checkpoint retention using RetentionManager.

        Policy:
        - 36-hour minimum age before deletion
        - 150GB total size limit
        - Protected: latest, best, today, yesterday
        """
        now = time.time()
        # Throttle to once per 30 minutes unless forced
        if not force and self.last_retention_check and (now - self.last_retention_check) < 1800:
            return

        try:
            summary = self.retention_service.enforce(dry_run=False)

            if summary.get("skipped"):
                self.logger.debug(f"Retention skipped: {summary.get('reason', 'unknown')}")
            else:
                deleted = summary.get("deleted_checkpoints", 0) + summary.get("deleted_snapshots", 0)
                if deleted > 0:
                    freed_gb = summary.get("deleted_size_gb", 0)
                    self.logger.info(
                        f"Retention: deleted {deleted} items, freed {freed_gb:.1f}GB "
                        f"(total: {summary.get('total_size_gb', 0):.1f}GB)"
                    )

            self.last_retention_check = now
        except Exception as e:
            self.logger.warning(f"Checkpoint retention failed: {e}")

    def run(self) -> None:
        """
        Main daemon loop that runs continuously until stopped.

        Lifecycle:
            1. Startup Phase:
                - Setup directories (inbox, queue, logs, etc.)
                - Initialize model (if empty current_model)
                - Acquire PID lock (prevents multiple daemons)
                - Recover from crashes (orphaned files, stale state)
                - Log configuration summary

            2. Main Loop (runs every poll_interval seconds):
                - Check stop signal (.stop file or SIGTERM/SIGINT) → exit cleanly
                - Run checkpoint retention (cleanup if exceeds disk caps)
                - Check snapshot schedule → create daily snapshot
                - Check consolidation schedule → consolidate when idle after 3 AM
                - Flatten inbox (move .jsonl from subdirs to root)
                - Process inbox → move files to priority queue
                - Get queue status (counts by priority)
                - While queue not empty:
                    * Check control signals (.stop, .pause, .skip)
                    * Get next file by priority (high → normal → low)
                    * Check disk space → skip if insufficient
                    * Train on file (via UltimateTrainer)
                    * Mark completed (delete) or failed (keep)
                    * Update queue status
                - If queue empty → maybe auto-generate data
                - Sleep poll_interval seconds
                - Repeat

            3. Shutdown Phase:
                - Release PID lock (.daemon.pid)
                - Log shutdown message

        Side Effects:
            - Creates/modifies files in base_dir (inbox, queue, logs, checkpoints, etc.)
            - Writes PID lock file (.daemon.pid)
            - Trains model (GPU operations)
            - Writes training_status.json
            - Deletes completed training files
            - Creates daily snapshots
            - Consolidates checkpoints
            - Auto-generates training data (if enabled)

        Raises:
            SystemExit: If PID lock acquisition fails (another daemon running)
            KeyboardInterrupt: Caught and handled gracefully (clean shutdown)
            Exception: Caught and logged (daemon exits with error message)

        Example:
            # Start daemon
            daemon = TrainingDaemon(Path("/training"))
            daemon.run()  # Blocks until stopped

            # Daemon runs indefinitely, processing files from queue
            # Stop by creating .stop file or sending SIGTERM/SIGINT
        """
        self.logger.info("=" * 80)
        self.logger.info("🤖 TRAINING DAEMON STARTING")
        self.logger.info("=" * 80)
        self.logger.info("")

        # Setup
        self.setup_directories()
        self.initialize_model()

        # Acquire PID lock (prevents multiple daemons)
        self.acquire_lock()

        # Recover from previous crash
        self.recover_orphaned_files()
        self.cleanup_stale_state()

        self.logger.info("")
        self.logger.info("Configuration:")
        self.logger.info(f"  Model: {self.config.get('model_path') or self.config['model_name']}")
        self.logger.info(f"  Batch size: {self.config['batch_size']}")
        self.logger.info(f"  Learning rate: {self.config['learning_rate']}")
        self.logger.info(f"  Poll interval: {self.config['poll_interval']}s")
        self.logger.info(f"  Daily snapshot: {self.config['snapshot_time']}")
        if FORMAT_VARIETY_AVAILABLE:
            variant_count = len(OUTPUT_VARIANTS)
            self.logger.info(f"  Format variety: {variant_count} variants enabled")
        else:
            self.logger.info("  Format variety: NOT AVAILABLE (using standard format only)")
        self.logger.info("")
        self.logger.info("Daemon is running...")
        self.logger.info(f"  Drop JSONL files in: {self.inbox_dir}")
        self.logger.info(f"  Create {self.stop_file} to stop daemon")
        self.logger.info("")

        iteration = 0

        try:
            while True:
                iteration += 1

                # Check for stop signal
                if self.should_stop():
                    self.logger.info("Stop signal detected!")
                    if self.stop_file.exists():
                        self.stop_file.unlink()
                    self.logger.info("Daemon stopped cleanly")
                    break

                # Enforce checkpoint retention periodically
                self.run_checkpoint_retention()

                # Check if we should create daily snapshot
                if self.should_create_snapshot():
                    self.create_snapshot()

                # Check if we should consolidate (when idle after 3 AM)
                # FIXED: Check queue processing status too, not just inbox
                inbox_files_check = self.get_inbox_files()
                queue_status_temp = self.queue.get_queue_status()
                if (not inbox_files_check and
                    queue_status_temp["total_queued"] == 0 and
                    queue_status_temp["processing"] == 0 and
                    self.should_consolidate()):
                    self.consolidate_model()

                # Flatten inbox (move .jsonl files from subdirs to root)
                self.flatten_inbox()

                # Quick validation: reject obviously bad files before queueing
                rejected = self.quick_validate_inbox_files()
                if rejected > 0:
                    self.logger.info(f"Quick validation rejected {rejected} file(s)")

                # Process inbox files into queue
                self.queue.process_inbox(default_priority="normal")

                # Get queue status
                queue_status = self.queue.get_queue_status()

                if queue_status["total_queued"] > 0:
                    self.logger.info(f"Queue: {queue_status['queued']['high']} high, {queue_status['queued']['normal']} normal, {queue_status['queued']['low']} low")

                    # Process files from queue
                    while queue_status["total_queued"] > 0:
                        # Check for stop signal
                        if self.controller.should_stop_after_batch():
                            self.controller.clear_stop()
                            self.controller.update_state("idle", reason="Stopped by user")
                            self.logger.info("🛑 Stopped by user")
                            break

                        # Check for pause signal
                        if self.controller.should_pause_after_batch():
                            self.controller.clear_pause()
                            self.controller.wait_for_resume()

                        # Get next file from queue (priority order)
                        data_file = self.queue.get_next_file()
                        if not data_file:
                            break

                        # NEW: Check disk space before training
                        if not self.check_disk_space():
                            self.logger.error(f"⚠️  Skipping {data_file.name} - insufficient disk space")
                            self.queue.mark_failed(data_file, error="Insufficient disk space")
                            continue

                        # Update controller state
                        self.controller.update_state("training", current_file=data_file.name)

                        # Train on file
                        success = self.train_on_file(
                            data_file,
                            batch_number=None,  # Queue handles ordering
                            batch_queue_size=queue_status["total_queued"]
                        )

                        # Handle result
                        if success:
                            self.queue.mark_completed(data_file, delete_file=True)
                        elif self.controller.should_skip_current_file():
                            self.controller.clear_skip()
                            self.queue.mark_skipped(data_file, reason="Skipped by user")
                            self.logger.info("⏭️  Skipped by user")
                        else:
                            self.queue.mark_failed(data_file, error="Training failed", keep_file=True)

                        # Update queue status for next iteration
                        queue_status = self.queue.get_queue_status()

                    # Update controller to idle after processing
                    self.controller.update_state("idle")

                else:
                    # No files, just log periodically
                    if iteration % 10 == 0:  # Every 10 iterations
                        self.logger.info(f"Queue empty. Waiting... (iteration {iteration})")
                    self.maybe_auto_generate(queue_status)

                # Sleep until next check
                time.sleep(self.config["poll_interval"])

        except KeyboardInterrupt:
            self.logger.info("⚠️  Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in daemon loop: {e}")
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            self.logger.error("   Daemon crashed - check logs for details")
        finally:
            self.logger.info("Shutting down daemon...")
            self.release_lock()

    def maybe_auto_generate(self, queue_status: dict):
        """
        Auto-generate training data using Data Manager

        Data Manager handles:
        - Remote GPU communication (192.168.x.x:8765)
        - Quality testing (5 test suites)
        - Queue management
        - Cooldown tracking
        """
        try:
            # Data Manager handles all checks internally
            success = self.data_manager.generate_and_queue(force=False)

            if success:
                self.logger.info("🤖 Data Manager: Successfully generated, tested, and queued new batch")

        except Exception as e:
            self.logger.error(f"Data Manager auto-generation failed: {e}")
            self.logger.debug(traceback.format_exc())

    # OLD METHODS REMOVED - Now handled by Data Manager
    # - fetch_autogen_puzzles() → RemoteGPUClient.generate_data()
    # - convert_puzzles_to_training() → Remote server returns training format
    # - build_user_prompt() → Remote server generates prompts
    # - build_assistant_payload() → Remote server generates payloads
    # - write_training_file() → DataManager.queue_data()


def main():
    # Import paths module for auto-detection
    from paths import get_base_dir

    parser = argparse.ArgumentParser(description="Training Daemon for RTX 3090")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for training system (default: auto-detect or $TRAINING_BASE_DIR)"
    )

    args = parser.parse_args()

    # Use provided base_dir or auto-detect
    base_dir = Path(args.base_dir) if args.base_dir else get_base_dir()
    daemon = TrainingDaemon(base_dir)
    daemon.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
