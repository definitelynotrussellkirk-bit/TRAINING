#!/usr/bin/env python3
"""
Training Daemon - RTX 3090 Continuous Training System

Simple, focused purpose:
1. Watch INBOX for new training data
2. Train on it (1 epoch only)
3. Delete data after successful training (keep failed files)
4. Save daily snapshots of model
5. Always train on newest model
6. Never touch old snapshots

Usage:
    python3 training_daemon.py --base-dir /training

Directory structure:
    /training/
        inbox/          ← Drop JSONL files here
        current_model/  ← Active model being trained
        snapshots/      ← Daily snapshots (YYYY-MM-DD/)
        logs/           ← Training logs
        config.json     ← Training configuration
        .stop           ← Create this file to stop daemon
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

# Checkpoint retention
from checkpoint_retention import enforce_retention

# Add ultimate_trainer to path
sys.path.insert(0, str(Path(__file__).parent))

from train import UltimateTrainer
from training_queue import TrainingQueue
from training_controller import TrainingController
from atomic_ops import write_json_atomic, safe_file_operation

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
    """Continuous training daemon for RTX 3090"""

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

        # Checkpoint retention tracking
        self.last_retention_check = 0
        self.recent_checkpoint_cap_gb = 100
        self.historic_checkpoint_cap_gb = 150

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

    def validate_config(self, config: dict):
        """Validate configuration to prevent training failures.

        GUARDRAIL: Catches config errors BEFORE training starts.
        Part of Phase 1 from CRITICAL_EDGE_CASES_AND_GUARDRAILS.md
        """
        errors = []

        # Check base model path exists (if specified)
        if 'base_model' in config and config['base_model']:
            base_model_path = Path(config['base_model'])
            if not base_model_path.exists():
                errors.append(f"Base model not found: {config['base_model']}")

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
        if 'lora_r' in config:
            lora_r = config['lora_r']
            if not (lora_r > 0 and lora_r <= 1024):
                errors.append(f"LoRA rank out of range (1-1024): {lora_r}")

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

    def verify_snapshot(self, snapshot_dir: Path) -> bool:
        """
        CRITICAL FIX #7: Verify snapshot integrity before trusting it

        Checks that essential files exist and are readable
        """
        try:
            adapter_file = snapshot_dir / "adapter_model.safetensors"
            config_file = snapshot_dir / "adapter_config.json"

            if not adapter_file.exists() or not config_file.exists():
                return False

            # Verify files are readable
            adapter_size = adapter_file.stat().st_size
            if adapter_size == 0:
                return False

            # Try to read config as JSON
            import json
            with open(config_file) as f:
                json.load(f)

            return True
        except Exception as e:
            self.logger.error(f"Snapshot verification failed: {e}")
            return False

    def create_snapshot(self):
        """Create daily snapshot of current model (latest checkpoint only)"""
        today = datetime.now().date()
        snapshot_dir = self.snapshots_dir / today.strftime("%Y-%m-%d")

        # TOCTOU FIX: Use try/except instead of checking exists first
        try:
            if snapshot_dir.exists():
                # Verify existing snapshot
                if self.verify_snapshot(snapshot_dir):
                    self.logger.info(f"Snapshot already exists and verified: {snapshot_dir}")
                    self.last_snapshot_date = today
                    return
                else:
                    self.logger.warning(f"Existing snapshot corrupt - recreating")
                    shutil.rmtree(snapshot_dir)
        except Exception as e:
            self.logger.warning(f"Error checking snapshot: {e}")

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

            # CRITICAL: Verify snapshot after creation
            if not self.verify_snapshot(snapshot_dir):
                self.logger.error("❌ Snapshot verification failed after creation!")
                self.logger.error("   Removing corrupt snapshot")
                shutil.rmtree(snapshot_dir)
                raise Exception("Snapshot creation produced corrupt files")

            self.last_snapshot_date = today
            self.logger.info(f"✅ Snapshot created and verified: {snapshot_dir}")

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
        # Find all .jsonl files in subdirectories (not at root)
        subdirs = [d for d in self.inbox_dir.iterdir() if d.is_dir()]

        if not subdirs:
            return  # No subdirectories, nothing to do

        moved_count = 0

        for subdir in subdirs:
            # Find all .jsonl files recursively in this subdir
            jsonl_files = list(subdir.rglob("*.jsonl"))

            for jsonl_file in jsonl_files:
                # Construct new filename at inbox root
                # Use subdirectory name to make it unique
                subdir_name = subdir.name
                original_name = jsonl_file.stem  # filename without extension
                new_name = f"{original_name}_{subdir_name}.jsonl"
                dest_path = self.inbox_dir / new_name

                # Handle collision (if file already exists at destination)
                counter = 1
                while dest_path.exists():
                    new_name = f"{original_name}_{subdir_name}_{counter}.jsonl"
                    dest_path = self.inbox_dir / new_name
                    counter += 1

                # Move file
                try:
                    shutil.move(str(jsonl_file), str(dest_path))
                    self.logger.info(f"📁 Flattened: {jsonl_file.relative_to(self.inbox_dir)} → {new_name}")
                    moved_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to move {jsonl_file}: {e}")

            # Clean up empty subdirectory
            try:
                if subdir.exists() and not any(subdir.iterdir()):
                    subdir.rmdir()
                    self.logger.info(f"🗑️  Removed empty subdir: {subdir.name}")
            except Exception as e:
                self.logger.warning(f"Could not remove subdir {subdir.name}: {e}")

        if moved_count > 0:
            self.logger.info(f"✅ Flattened {moved_count} file(s) from subdirectories")

    def get_inbox_files(self):
        """Get all JSONL files in inbox"""
        return sorted(self.inbox_dir.glob("*.jsonl"))

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

    def train_on_file(self, data_file: Path, batch_number: int = None, batch_queue_size: int = None):
        """Train on a single data file (1 epoch)"""
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

        # GUARDRAIL: Validate data against config
        if not self.validate_data_before_training(data_file):
            self.logger.error("❌ Data validation failed - aborting training")
            self.logger.error("   Fix config or data, then try again")
            return False

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
            # Initialize current_model_dir from base if empty
            if not self.current_model_dir.exists() or not any(self.current_model_dir.iterdir()):
                self.logger.info("Current model dir empty; copying base model for throughput resume safety")
                shutil.copytree(args.model, self.current_model_dir, dirs_exist_ok=True)

        args.output_dir = str(self.current_model_dir)

        # Training params from config
        args.epochs = 1  # ALWAYS 1 EPOCH
        args.batch_size = self.config["batch_size"]
        args.gradient_accumulation = self.config["gradient_accumulation"]
        args.learning_rate = self.config["learning_rate"]
        args.warmup_steps = self.config["warmup_steps"]
        args.lora_r = self.config["lora_r"]
        args.lora_alpha = self.config["lora_alpha"]
        args.log_steps = self.config.get("log_steps", 10)  # Default to 10 if not in config
        args.eval_steps = self.config["eval_steps"]
        args.num_eval_samples = self.config["num_eval_samples"]
        args.save_steps = self.config["save_steps"]
        args.use_qlora = self.config.get("use_qlora", False)
        args.max_length = self.config.get("max_length")

        # Skip validation (assume data is pre-validated)
        args.skip_validation = True
        args.yes = True  # No prompts
        args.system_prompt = "You are a helpful assistant."

        # Batch context (NEW: for progress tracking)
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
        if self.pid_file.exists():
            try:
                old_pid = int(self.pid_file.read_text().strip())
                # Check if process is still running
                os.kill(old_pid, 0)  # Doesn't kill, just checks
                self.logger.error(f"❌ Another daemon is running (PID {old_pid})")
                self.logger.error("   Stop it first or remove .daemon.pid if stale")
                sys.exit(1)
            except (OSError, ValueError):
                # Process not running, clean up stale PID file
                self.logger.warning(f"⚠️  Removing stale PID file (old PID: {old_pid if 'old_pid' in locals() else 'unknown'})")
                self.pid_file.unlink()

        # Write our PID
        self.pid_file.write_text(str(os.getpid()))
        self.logger.info(f"✅ Acquired daemon lock (PID: {os.getpid()})")

    def release_lock(self):
        """Release PID file lock"""
        if self.pid_file.exists():
            self.pid_file.unlink()
        self.logger.info("✅ Released daemon lock")

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
        """Enforce checkpoint caps: 100GB recent, 150GB historic (daily)."""
        now = time.time()
        # Throttle to once per 30 minutes unless forced
        if not force and self.last_retention_check and (now - self.last_retention_check) < 1800:
            return

        try:
            roots = [
                self.current_model_dir,
                self.base_dir / "current_model_small",
                self.snapshots_dir,
            ]
            enforce_retention(
                roots,
                recent_limit_gb=self.recent_checkpoint_cap_gb,
                historic_limit_gb=self.historic_checkpoint_cap_gb,
                logger=self.logger,
                dry_run=False,
            )
            self.last_retention_check = now
        except Exception as e:
            self.logger.warning(f"Checkpoint retention failed: {e}")

    def run(self):
        """Main daemon loop"""
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
        auto_cfg = self.config.get("auto_generate", {}) or {}
        if not auto_cfg.get("enabled"):
            return

        threshold = auto_cfg.get("threshold", 0)
        if queue_status.get("total_queued", 0) > threshold:
            return
        if queue_status.get("processing", 0) > 0:
            return
        if self.get_inbox_files():
            return

        cooldown = auto_cfg.get("cooldown_sec", 120)
        now = time.time()
        if now - self.last_autogen_time < cooldown:
            return

        try:
            puzzles = self.fetch_autogen_puzzles(auto_cfg)
            entries = self.convert_puzzles_to_training(puzzles)
            if not entries:
                self.logger.warning("Auto-generate: API returned no puzzles")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"syllo_autogen_{timestamp}_count{len(entries)}.jsonl"
            inbox_path = self.inbox_dir / filename
            self.write_training_file(inbox_path, entries)

            priority = auto_cfg.get("priority", "normal")
            self.queue.add_to_queue(inbox_path, priority)
            self.logger.info(f"🤖 Auto-generated {len(entries)} puzzles via API and queued {filename}")
        except Exception as e:
            self.logger.error(f"Auto-generation failed: {e}")
        finally:
            self.last_autogen_time = now

    def fetch_autogen_puzzles(self, auto_cfg: dict):
        host = auto_cfg.get("host", "127.0.0.1")
        port = auto_cfg.get("port", 8080)
        url = f"http://{host}:{port}/generate"

        payload = {
            "count": auto_cfg.get("count", 1000)
        }
        if auto_cfg.get("seed") is not None:
            payload["seed"] = auto_cfg["seed"]
        extra = auto_cfg.get("payload") or {}
        payload.update(extra)

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = request.Request(url, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=300) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        parsed = json.loads(body)
        if isinstance(parsed, dict):
            if "puzzles" in parsed:
                puzzles = parsed["puzzles"]
            elif "examples" in parsed:
                puzzles = parsed["examples"]
            else:
                puzzles = parsed.get("data", [])
        else:
            puzzles = parsed

        if not puzzles:
            raise RuntimeError("API response contained no puzzles")
        return puzzles

    def convert_puzzles_to_training(self, puzzles):
        entries = []
        for puzzle in puzzles:
            # Choose format variant
            variant_key = choose_output_variant(self.format_rng, DEFAULT_OUTPUT_VARIANT_DISTRIBUTION)

            # Build base payload
            base_payload = self.build_assistant_payload(puzzle)

            # Apply format transformation
            transform = OUTPUT_VARIANTS[variant_key]["transform"]
            assistant_payload = transform(base_payload)

            # Convert to JSON string
            assistant_text = json.dumps(assistant_payload, ensure_ascii=False)

            # Build user prompt with variant note
            user_prompt = self.build_user_prompt(puzzle, variant_key)

            entry = {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text}
                ],
                "metadata": {
                    "dataset": "syllo_api_autogen",
                    "puzzle_id": puzzle.get("puzzle_id"),
                    "word_count": len(puzzle.get("words", [])),
                    "syllable_bank_size": len(puzzle.get("syllable_bank", [])),
                    "rules": puzzle.get("rules", {}),
                    "output_variant": variant_key
                }
            }
            entries.append(entry)
        return entries

    def build_user_prompt(self, puzzle: dict, variant_key: str = "standard") -> str:
        puzzle_id = puzzle.get("puzzle_id", "syllo_autogen")
        rules = puzzle.get("rules", {})
        difficulty = rules.get("difficulty", "Unknown")
        word_count = rules.get("word_count", len(puzzle.get("words", [])))
        syllable_bank = puzzle.get("syllable_bank", [])
        total_tiles = len(syllable_bank)
        total_needed = sum(len(word.get("syllables", [])) for word in puzzle.get("words", []))
        red_herring_count = max(0, total_tiles - total_needed)
        notes = rules.get("notes", "")

        lines = [
            f"SYLLO Puzzle {puzzle_id}",
            "You must recover every hidden word by assigning syllable tiles to definitions.",
            f"Difficulty: {difficulty}",
            "Rules:",
            f"- {word_count} target words (always between 4 and 8).",
            "- Each word lists its syllable count via blank slots.",
            "- Syllable tiles may repeat across clues when the bank includes duplicates.",
            "- Return your answers as JSON with keys `solutions` and `inventory_check`.",
            "",
            "Word slots:"
        ]

        for idx, word in enumerate(puzzle.get("words", []), 1):
            blanks = " ".join(["___"] * word.get("syllable_count", len(word.get("syllables", []))))
            clue = word.get("definition") or ", ".join(word.get("available_hints", [])[:1]) or word.get("label", "Unknown clue")
            lines.append(f"{idx}. {blanks} — {clue}")

        note_line = "Note: "
        if red_herring_count > 0:
            note_line += f"{red_herring_count} tile(s) in the bank are red herrings and do not belong to any answer."
        else:
            note_line += "All tiles belong to some answer."
        if notes:
            note_line += f" {notes}"
        lines.extend(["", note_line, "", "Syllable bank (shuffled):", " | ".join(syllable_bank), ""])
        # Get variant-specific format note
        variant = OUTPUT_VARIANTS.get(variant_key, OUTPUT_VARIANTS.get("standard", {}))
        variant_note = variant.get("prompt_note", "")

        lines.extend([
            "Output contract:",
            "- Return a single JSON object.",
            "- Top-level keys: `solutions` (array) and `inventory_check` (object).",
            "- Each `solutions` entry contains `ans_num` (1-indexed clue number),",
            "  the ordered `syllables` you used, and the final UPPERCASE `answer`.",
            "- `inventory_check` must include `total_tiles`, a `usage` map of tile→count,",
            "  the `used` counts per tile, and a short `status` string.",
            "- If red herrings exist, include them under `inventory_check.unused_tiles`.",
            "Do not include literal JSON examples or commentary outside the payload."
        ])

        # Add variant-specific format instruction
        if variant_note:
            lines.append(variant_note)

        return "\n".join(lines)

    def build_assistant_payload(self, puzzle: dict) -> dict:
        from collections import Counter

        words = puzzle.get("words", [])
        syllable_bank = puzzle.get("syllable_bank", [])
        solutions = []
        used_counter = Counter()
        for idx, word in enumerate(words, 1):
            syllables = word.get("syllables", [])
            used_counter.update(syllables)
            solutions.append({
                "ans_num": idx,
                "syllables": syllables,
                "answer": word.get("label", "").upper()
            })

        bank_counter = Counter(syllable_bank)
        unused_tiles = []
        for tile, count in bank_counter.items():
            unused = count - used_counter.get(tile, 0)
            if unused > 0:
                unused_tiles.extend([tile] * unused)

        usage_map = dict(bank_counter)
        used_map = {tile: used_counter.get(tile, 0) for tile in bank_counter if used_counter.get(tile, 0)}
        if unused_tiles:
            status = "All target words completed; red herrings unused: " + ", ".join(sorted(set(unused_tiles))) + "."
        else:
            status = "All target words completed; no unused tiles."

        analysis = {
            "word_count": len(words),
            "unused_tile_count": len(set(unused_tiles)),
            "unused_tiles": sorted(set(unused_tiles)),
            "tile_usage_span": [[tile, used_counter.get(tile, 0)] for tile in sorted(bank_counter.keys())]
        }

        return {
            "solutions": solutions,
            "inventory_check": {
                "total_tiles": len(syllable_bank),
                "usage": usage_map,
                "used": used_map,
                "unused_tiles": sorted(set(unused_tiles)),
                "status": status
            },
            "analysis": analysis
        }

    def write_training_file(self, path: Path, entries):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False))
                fh.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Training Daemon for RTX 3090")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/training",
        help="Base directory for training system (default: /training)"
    )

    args = parser.parse_args()

    daemon = TrainingDaemon(Path(args.base_dir))
    daemon.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
