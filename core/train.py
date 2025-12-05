#!/usr/bin/env python3
"""
Ultimate Trainer - CLI Interface (Layer 3)

This is a Layer 3 interface that provides CLI access to TrainerEngine (Layer 1).
See ARCHITECTURE.md "Training Flow Architecture" for the layer model.

Layer Responsibilities:
- Layer 3 (this file): Parse CLI args, create TrainerConfig, call engine
- Layer 2 (daemon): Schedule training jobs, manage queue
- Layer 1 (TrainerEngine): Execute training loop

Default Behavior:
- Uses TrainerEngine for all training (engine-first)
- Legacy path available via USE_LEGACY_TRAINER=1 (deprecated)

Usage:
    # Standard (uses TrainerEngine)
    python3 train.py --dataset data.jsonl --model qwen3_0.6b --output-dir models/

    # Force legacy path (deprecated)
    USE_LEGACY_TRAINER=1 python3 train.py --dataset data.jsonl

Architecture Note:
    The legacy _run_legacy() method contains ~1000 lines of training code that
    duplicates TrainerEngine functionality. It is kept only for backward
    compatibility and will be removed in a future version.

    NEW CODE SHOULD NOT USE THE LEGACY PATH.
"""

import argparse
import os
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import OrderedDict
from datetime import datetime

# Fix CUDA multiprocessing issue - MUST be before torch import
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    LogitsProcessorList,
)
from datasets import Dataset

# Add paths for modules in other directories
sys.path.insert(0, str(Path(__file__).parent.parent))  # For trainer/ module
sys.path.insert(0, str(Path(__file__).parent.parent / "management"))
sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring" / "servers"))

# Import our components
from validator import DatasetValidator
from model_db import ModelDatabase
from time_estimator import TimeEstimator
from live_monitor import LiveInferenceMonitor
from training_status import TrainingStatusWriter, DEFAULT_STATUS_FILE
from evolution_tracker import EvolutionTracker
from custom_collator import DataCollatorForCompletionOnly
from logit_penalty import (
    build_think_penalty_processor,
    build_eos_penalty_processor,
    build_post_stop_penalty_processor,
    reset_processor_states,
    DEFAULT_PENALTY_SCHEDULE,
    collect_penalty_stats,
)
from layer_monitor import LayerMonitor

# Optimizer factory (Muon support)
try:
    from trainer.optimizers import create_optimizer as create_custom_optimizer
    from trainer.optimizers import get_param_group_summary
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    create_custom_optimizer = None
    get_param_group_summary = None

# Optional: auto self-correction (may not be available)
try:
    from auto_self_correction import create_self_correction_monitor
    SELF_CORRECTION_AVAILABLE = True
except ImportError:
    SELF_CORRECTION_AVAILABLE = False
    create_self_correction_monitor = None

# Import new trainer modules (refactored architecture)
from trainer.config import ConfigLoader, TrainerConfig, create_default_config
from trainer.profiles import get_profile
# NOTE: trainer.monitoring.TrainingStatusWriter exists but we use core/training_status.py for now (backward compat)

# Import TrainerEngine for delegation (Phase 3 refactor)
try:
    from trainer.core.engine import TrainerEngine, TrainingResult, MonitorContext
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    TrainerEngine = None
    TrainingResult = None
    MonitorContext = None

# Import LiveMonitorCallback from extracted module (Phase 4 refactor)
from trainer.monitoring.callbacks import LiveMonitorCallback

# Import chat template override (fixes Qwen3 <think> injection conflict)
from core.chat_templates import apply_chat_template_override

# Import Data Manager for remote eval
sys.path.insert(0, str(Path(__file__).parent.parent / "data_manager"))
try:
    from remote_evaluator import RemoteEvaluator
    REMOTE_EVAL_AVAILABLE = True
except ImportError:
    REMOTE_EVAL_AVAILABLE = False
    RemoteEvaluator = None

# Thinking emoji pool - each example gets a RANDOM emoji and count
THINKING_EMOJIS = [
    "🤔",  # Classic thinking
    "💭",  # Thought bubble
    "🧠",  # Brain
    "💡",  # Lightbulb (idea)
    "🎯",  # Target (focus)
    "🔍",  # Magnifying glass (analyze)
    "🤨",  # Raised eyebrow (skeptical)
    "🧐",  # Monocle (scrutinize)
    "⚡",  # Lightning (quick thought)
    "✨",  # Sparkles (insight)
]

# Stop emoji (stays consistent)
# Stop emoji pool - randomly select from these
STOP_EMOJI_POOL = ["🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"]
STOP_COUNT_MIN = 2
STOP_COUNT_MAX = 4

# Helper functions for variable stop sequences
def get_random_stop_emoji() -> str:
    """
    Select a random stop emoji from the pool.

    Returns:
        str: One emoji from STOP_EMOJI_POOL (e.g., "🛑", "⛔", "🚫")

    Example:
        >>> emoji = get_random_stop_emoji()
        >>> emoji in STOP_EMOJI_POOL
        True
    """
    import random
    return random.choice(STOP_EMOJI_POOL)

def get_random_stop_count() -> int:
    """
    Select a random stop count (2-4).

    Returns:
        int: Random integer between STOP_COUNT_MIN and STOP_COUNT_MAX (inclusive)

    Example:
        >>> count = get_random_stop_count()
        >>> 2 <= count <= 4
        True
    """
    import random
    return random.randint(STOP_COUNT_MIN, STOP_COUNT_MAX)

def get_stop_instruction(emoji: str, count: int) -> str:
    """
    Generate stop instruction text for user prompts.

    Args:
        emoji: Stop emoji to use (e.g., "🛑")
        count: Number of times to repeat emoji (2-4)

    Returns:
        str: Instruction text to append to user messages

    Example:
        >>> get_stop_instruction("🛑", 3)
        'When finished, emit 🛑 /three times/ to signal completion.'
    """
    count_words = {2: "twice", 3: "three times", 4: "four times"}
    count_text = count_words.get(count, f"{count} times")
    return f"When finished, emit {emoji} /{count_text}/ to signal completion."

def get_stop_suffix(emoji: str, count: int) -> str:
    """
    Generate stop suffix for assistant responses.

    Args:
        emoji: Stop emoji to use (e.g., "🛑")
        count: Number of times to repeat emoji (2-4)

    Returns:
        str: Suffix to append to assistant responses (newline + emoji * count)

    Example:
        >>> get_stop_suffix("🛑", 3)
        '\\n🛑🛑🛑'
    """
    return "\n" + emoji * count

def get_thinking_pattern(example_index):
    """
    Get RANDOM thinking emoji and count for this example.

    Each example gets:
    - Random emoji from THINKING_EMOJIS pool
    - Random count between 2-8

    Uses example_index as seed for reproducibility.
    """
    import random
    random.seed(example_index)  # Reproducible randomness

    emoji = random.choice(THINKING_EMOJIS)
    count = random.randint(2, 8)

    count_words = ["two", "three", "four", "five", "six", "seven", "eight"]
    count_word = count_words[count - 2]  # count 2 -> index 0

    prefix = emoji * count + "\n"
    instruction = f"For this task, think with {emoji} /{count_word}/ times."

    return emoji, count, count_word, prefix, instruction

from pattern_tracker import PatternTracker, get_default_patterns

# Try to import detailed monitoring (optional)
try:
    from detail_collector import add_detail_collector_to_trainer
    DETAILED_MONITORING_AVAILABLE = True
except ImportError:
    DETAILED_MONITORING_AVAILABLE = False

# Desktop notifier stub (optional feature, not implemented)
class DesktopNotifier:
    """Stub for optional desktop notifications. No-op by default."""
    def __init__(self): pass
    def notify(self, *args, **kwargs): pass
    def notify_success(self, *args, **kwargs): pass
    def notify_error(self, *args, **kwargs): pass
    def training_complete(self, *args, **kwargs): pass
    def training_crashed(self, *args, **kwargs): pass
    def oom_error(self, *args, **kwargs): pass


class UltimateTrainer:
    """
    CLI training interface (Layer 3) that delegates to TrainerEngine.

    ARCHITECTURE (see ARCHITECTURE.md "Training Flow Architecture"):
        - This class is a Layer 3 THIN WRAPPER
        - All training logic lives in TrainerEngine (Layer 1)
        - This class only: parses CLI args, creates config, calls engine

    Default Behavior:
        - run() calls _run_with_engine() which delegates to TrainerEngine
        - Legacy _run_legacy() path is DEPRECATED (USE_LEGACY_TRAINER=1 to force)

    DEPRECATION WARNING:
        The following methods are DEPRECATED and will be removed:
        - _run_legacy()
        - load_model()
        - prepare_dataset()
        - train()
        - validate_dataset()
        - estimate_time()
        - setup_live_monitor()
        - enforce_locked_config()

        These methods duplicate TrainerEngine functionality and are kept only
        for backward compatibility. New code should use TrainerEngine directly.

    Configuration:
        Uses trainer.config.TrainerConfig from config.json + CLI args.
        TrainerConfig is passed to TrainerEngine.run_job().

    Key Components Integrated:
        - DatasetValidator: Pre-training validation
        - ModelDatabase: Model selection and adapter management
        - TimeEstimator: Training duration estimation
        - TrainingStatusWriter: Real-time status updates to JSON
        - LiveInferenceMonitor: Periodic inference during training
        - EvolutionTracker: Track model evolution over time
        - LayerMonitor: Monitor internal layer statistics
        - DataProfile: Apply transformations (emoji patterns, stop signals, etc.)

    Attributes:
        args: Command-line arguments (argparse.Namespace)
        controller: Optional training controller for pause/resume/stop
        config: TrainerConfig from new config system (or None if fallback)
        model_db: ModelDatabase for model selection
        validator: DatasetValidator instance
        model: Loaded HuggingFace model (AutoModelForCausalLM)
        tokenizer: Loaded tokenizer
        train_dataset: Tokenized training dataset
        val_dataset: Tokenized validation dataset (if split enabled)
        fixed_val_dataset: Fixed validation set for generalization metrics
        raw_train_examples: Original untokenized examples (for snapshots)
        live_monitor: LiveInferenceMonitor (optional)
        evolution_tracker: EvolutionTracker (tracks model evolution)
        status_writer: TrainingStatusWriter (writes status JSON)
        layer_monitor: LayerMonitor (optional, tracks layer statistics)
        training_summary: Dict with final training metrics

    Side Effects:
        - Writes checkpoints to output_dir/checkpoint-{step}/
        - Writes training_status.json to status/ directory
        - Writes evolution data to data/evolution/ (if enabled)
        - Writes layer statistics to status/layer_monitor.json (if enabled)
        - Creates daily snapshots with model + metadata
        - May create profile files (stop signal patterns, etc.)
        - Sends desktop notifications on training completion

    Example:
        from argparse import Namespace

        args = Namespace(
            dataset="data.jsonl",
            model="qwen3_0.6b",
            output_dir="outputs",
            epochs=2,
            batch_size=16,
            learning_rate=2e-4
        )

        trainer = UltimateTrainer(args)
        trainer.run()  # Execute full pipeline

        # Access results
        print(f"Final loss: {trainer.training_summary['loss']}")
        print(f"Trained {trainer.training_summary['global_step']} steps")
    """

    def __init__(self, args, controller=None):
        self.args = args
        self.controller = controller  # Training control system (pause/stop)

        # Create TrainerConfig from args (new config system)
        # This provides single source of truth for all config values
        # Also stores raw config dict for extra fields (optimizer, liger_kernel, campaign)
        try:
            self.config, self.config_dict = ConfigLoader.from_args_and_json_with_raw(args)
            print(f"✓ TrainerConfig created (profile: {self.config.profile.name}, precision: {self.config.hyperparams.fp_precision})")
            # Show campaign info if using campaign system
            campaign = self.config_dict.get('campaign', {})
            if campaign.get('hero_id'):
                print(f"  Campaign: {campaign.get('hero_id')}/{campaign.get('campaign_id')} - {campaign.get('campaign_name')}")
        except Exception as e:
            print(f"⚠️  Could not create TrainerConfig: {e}")
            print(f"   Falling back to args-only mode")
            self.config = None
            self.config_dict = {}

        self.model_db = ModelDatabase()
        self.validator = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.fixed_val_dataset = None  # Fixed validation set for true generalization metrics
        self.tokenized_fixed_val_small = None  # Tiny fixed eval set for cheap micro-evals
        self.raw_train_examples = None  # Keep original un-tokenized data
        self.live_monitor = None
        self.evolution_tracker = None  # Will be initialized with dataset name
        self.avg_seq_len = 0  # Rolling estimate from tokenized train set
        self.model_label = self._load_model_label()

        # Create status writer with config values (or fallback to defaults)
        max_output_tokens = 2048  # Default
        context_window = 2048  # Default

        if self.config:
            if self.config.monitoring:
                max_output_tokens = self.config.monitoring.max_output_tokens
            if self.config.hyperparams:
                context_window = self.config.hyperparams.max_length
        else:
            # Fallback to args if config unavailable
            context_window = getattr(args, 'max_length', 2048)

        self.status_writer = TrainingStatusWriter(
            DEFAULT_STATUS_FILE,
            max_output_tokens=max_output_tokens,
            context_window=context_window,
            model_name=self.model_label
        )
        self.notifier = DesktopNotifier()
        self.training_start_time = None
        self.logits_processor = None  # Optional logit penalties for generation
        self.training_summary: Dict[str, Any] | None = None
        self.layer_monitor: LayerMonitor | None = None

        # TrainerEngine for delegation (engine-first refactor)
        # Engine is now the DEFAULT path. To use legacy: USE_LEGACY_TRAINER=1
        self.engine = None
        if ENGINE_AVAILABLE:
            self.engine = TrainerEngine(self.status_writer)
            print("TrainerEngine available (engine-first default)")
        else:
            print("TrainerEngine not available, will use legacy path")

        # Run ledger health check on startup
        self._run_ledger_health_check()

    def _run_ledger_health_check(self):
        """Run checkpoint ledger health check on startup."""
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            result = ledger.run_health_check(fix_issues=True)

            if result["orphans_adopted"] > 0:
                print(f"✓ Adopted {result['orphans_adopted']} orphan checkpoints")
            if result["stale_removed"] > 0:
                print(f"✓ Removed {result['stale_removed']} stale ledger entries")
            if result["healthy"]:
                if result["ledger_count"] > 0:
                    print(f"✓ Ledger healthy ({result['ledger_count']} checkpoints)")
        except Exception as e:
            print(f"⚠️  Ledger health check failed: {e}")

    def _get_hyperparam(self, name: str, default=None):
        """Get hyperparam from config (preferred) or args (fallback)."""
        if self.config and hasattr(self.config, 'hyperparams'):
            return getattr(self.config.hyperparams, name, default)
        return getattr(self.args, name, default)

    def sanitize_example(self, example: dict) -> dict:
        """Strip disallowed tags like <think> from conversation content."""
        cleaned_messages = []
        for msg in example.get('messages', []):
            content = msg.get('content')
            if msg.get('role') == 'assistant' and isinstance(content, str):
                content = content.replace("<think>", "").replace("</think>", "")
            cleaned_messages.append({**msg, "content": content})
        new_ex = dict(example)
        new_ex['messages'] = cleaned_messages
        return new_ex

    def enforce_thinking_requirement(self, messages: List[Dict[str, Any]], example_index: int = 0) -> List[Dict[str, Any]]:
        """
        Apply thinking pattern to messages with RANDOM emoji and count.

        Args:
            messages: List of message dicts
            example_index: Position in dataset (determines random pattern)

        Each example gets a random thinking emoji and repetition count (2-8).
        """
        # Get RANDOM pattern for this specific example
        emoji, count, count_word, prefix, instruction = get_thinking_pattern(example_index)

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            if role == "user":
                # Check if ANY thinking instruction is already present (any emoji variant)
                has_instruction = any(f"think with {e}" in content.lower() for e in THINKING_EMOJIS) or "think with" in content.lower()
                if not has_instruction:
                    content = content.rstrip() + "\n\n" + instruction

            elif role == "assistant":
                # Check if starts with ANY thinking emoji
                has_prefix = any(content.startswith(e) for e in THINKING_EMOJIS)
                if not has_prefix:
                    content = prefix + content.lstrip()

            msg["content"] = content

        return messages

    def enforce_stop_requirement(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enforce stop emoji pattern in conversations.

        Adds:
        - Stop instruction to user messages (after think instruction)
        - Stop suffix to assistant responses (at end, before EOT)

        Uses random stop emoji and count (2-4) for each conversation.
        """
        # Pick ONE random stop emoji and count for this entire conversation
        stop_emoji = get_random_stop_emoji()
        stop_count = get_random_stop_count()
        stop_instruction = get_stop_instruction(stop_emoji, stop_count)
        stop_suffix = get_stop_suffix(stop_emoji, stop_count)

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "user":
                # Add stop instruction to USER messages
                # Check if ANY stop instruction already present
                has_stop_instruction = any(
                    emoji in content and "When finished" in content
                    for emoji in STOP_EMOJI_POOL
                )
                if not has_stop_instruction:
                    content = content.rstrip() + "\n\n" + stop_instruction
            elif role == "assistant":
                # Append stop suffix to ASSISTANT responses (at END)
                # Check if ANY stop emoji sequence already present at end
                has_stop_suffix = any(
                    content.rstrip().endswith(emoji * count)
                    for emoji in STOP_EMOJI_POOL
                    for count in range(STOP_COUNT_MIN, STOP_COUNT_MAX + 1)
                )
                if not has_stop_suffix:
                    content = content.rstrip() + stop_suffix
            msg["content"] = content
        return messages

    def _load_model_label(self) -> str:
        """Derive a human-readable model label for status reporting."""
        config_path = Path("config.json")
        label = None
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                label = cfg.get("model_display_name") or cfg.get("model_name")
                if not label:
                    base_model = cfg.get("base_model") or cfg.get("model_path")
                    if base_model:
                        label = Path(base_model).name
            except Exception:
                label = None
        if not label:
            label = str(getattr(self.args, "model", "unknown-model"))
        return label

    def enforce_locked_config(self):
        """Hard guard: abort if locked config values are violated."""
        try:
            import json
            cfg_path = Path("config.json")
            if not cfg_path.exists():
                print("⚠️  Locked-config check: config.json not found; skipping hard guard.")
                return

            with open(cfg_path) as f:
                cfg = json.load(f)

            errors = []
            # Enforce CLI args align with config for critical keys
            key_map = {
                "batch_size": "batch_size",
                "gradient_accumulation": "gradient_accumulation",
                "load_in_4bit": "load_in_4bit",
                "max_length": "max_length"
            }
            for cfg_key, arg_key in key_map.items():
                cfg_val = cfg.get(cfg_key)
                arg_val = getattr(self.args, arg_key, None)
                if cfg_val is not None and arg_val is not None and cfg_val != arg_val:
                    errors.append(f"{cfg_key} mismatch: config={cfg_val}, args={arg_val}")

            # Base/model path checks
            base_model_cfg = cfg.get("base_model")
            model_path_cfg = cfg.get("model_path")
            current_model_dir_cfg = cfg.get("current_model_dir")
            if base_model_cfg:
                if not Path(base_model_cfg).exists():
                    errors.append(f"base_model path missing: {base_model_cfg}")
            if model_path_cfg:
                if not Path(model_path_cfg).exists():
                    errors.append(f"model_path missing: {model_path_cfg}")
                else:
                    # If args.model is a path, ensure it matches model_path to avoid wrong model
                    if Path(self.args.model).exists():
                        args_model_path = Path(self.args.model).resolve()
                        model_path_resolved = Path(model_path_cfg).resolve()
                        # Allow training from current_model_dir as adapter resume
                        if current_model_dir_cfg and args_model_path == Path(current_model_dir_cfg).resolve():
                            pass
                        elif args_model_path != model_path_resolved:
                            errors.append(f"args.model ({self.args.model}) differs from config model_path ({model_path_cfg})")

            if errors:
                print("\n❌ Locked configuration violation detected. Aborting to protect training.")
                for e in errors:
                    print(f"   - {e}")
                print("Fix config.json or CLI args to match locked values before training.")
                sys.exit(1)
            else:
                print("✅ Locked-config check passed.")
        except Exception as e:
            print(f"⚠️  Locked-config check failed: {e} (continuing, but investigate)")

    def _run_with_engine(self) -> bool:
        """
        Execute training using TrainerEngine delegation.

        This is the new simplified path enabled by USE_ENGINE=1.
        Keeps CLI orchestration (validation, time estimate, notifications)
        but delegates core HF training to TrainerEngine.

        Returns:
            bool: True if training succeeded, False otherwise
        """
        print("Using TrainerEngine delegation (Phase 3 refactor)")
        print("-" * 80)

        # Step 0: Enforce locked config
        self.enforce_locked_config()

        # Reset CUDA peak memory stats
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        # Step 1: Validate dataset
        if not self.args.skip_validation:
            print("\nSTEP 1: Validating Dataset")
            print("-" * 80)
            if not self.validate_dataset():
                print("\nValidation failed! Aborting.")
                return False
            print("Dataset validation passed!")
        else:
            print("\nSkipping validation (not recommended!)")

        # Step 2: Time estimation
        print("\nSTEP 2: Time Estimation")
        print("-" * 80)
        estimate = self.estimate_time()
        TimeEstimator.display_estimate(estimate)

        # Ask for confirmation
        if not self.args.yes:
            response = input("\nContinue with training? [yes/no]: ").strip().lower()
            if response != 'yes':
                print("Training cancelled by user")
                return False

        # Step 3: Delegate to TrainerEngine
        print("\nSTEP 3: Training (via TrainerEngine)")
        print("-" * 80)

        self.training_start_time = time.time()

        # Build MonitorContext with available components
        # Note: live_monitor=None is OK - inference preview is disabled in callback
        # The callback still handles: progress, throughput, ledger, control signals
        monitors = MonitorContext(
            live_monitor=None,  # Inference preview disabled, not needed
            evolution_tracker=self.evolution_tracker,
            layer_monitor=None,  # Engine will create if available
            controller=self.controller,
            raw_train_examples=[],  # Engine will populate from dataset
            logits_processor=self.logits_processor,
            remote_eval_config=self.config_dict.get("remote_eval", {}),
            current_file=getattr(self.args, "dataset", None),
            batch_number=getattr(self.args, "batch_number", None),
            batch_queue_size=getattr(self.args, "batch_queue_size", None),
            fixed_val_dataset=None,  # Could be set if available
            micro_eval_inputs=None,  # Could be set if available
            micro_eval_interval=self.config.monitoring.micro_eval_interval
                if self.config and self.config.monitoring else 500,
            status_writer=self.status_writer,
        ) if MonitorContext else None

        print("   MonitorContext built (controller, status_writer, remote_eval)")

        # Run the engine
        result = self.engine.run_job(
            config=self.config,
            config_dict=self.config_dict,
            monitors=monitors,
            callbacks=None,  # Engine creates LiveMonitorCallback from monitors
        )

        # Handle result
        if result.success:
            elapsed = time.time() - self.training_start_time
            duration_str = f"{elapsed/60:.1f} minutes"

            print("\n" + "=" * 80)
            print("TRAINING COMPLETE!")
            print("=" * 80)
            print(f"\nModel saved to: {result.last_checkpoint_path}")
            print(f"Final loss: {result.final_loss:.4f}")
            print(f"Global step: {result.global_step}")
            print(f"Runtime: {duration_str}")

            # Store training summary
            self.training_summary = result.summary

            # Mark completed
            self.status_writer.mark_completed(result.global_step, result.global_step)

            # Send notification
            self.notifier.training_complete(duration_str)

            return True
        else:
            print("\n" + "=" * 80)
            print("TRAINING FAILED!")
            print("=" * 80)
            print(f"\nError: {result.error_message}")

            # Mark crashed
            self.status_writer.mark_crashed(result.error_message, "TrainerEngine")

            # Send notification
            if result.error_message and "out of memory" in result.error_message.lower():
                self.notifier.oom_error()
            else:
                self.notifier.training_crashed(result.error_message or "Unknown error")

            return False

    def run(self):
        """
        Execute full training pipeline.

        Routing logic (engine-first refactor):
        1. Default: Use TrainerEngine (engine-first)
        2. Override: USE_LEGACY_TRAINER=1 forces legacy path
        3. Fallback: If engine unavailable, use legacy
        """
        print("\n" + "=" * 80)
        print("ULTIMATE TRAINER")
        print("=" * 80)

        # Check for legacy override (USE_LEGACY_TRAINER=1)
        use_legacy = os.environ.get("USE_LEGACY_TRAINER", "0").lower() in ("1", "true", "yes")

        if use_legacy:
            print("Using LEGACY trainer (USE_LEGACY_TRAINER=1)")
            return self._run_legacy()

        # Default: Use TrainerEngine (engine-first)
        if self.engine and self.config:
            return self._run_with_engine()

        # Fallback: If engine unavailable, use legacy
        print("TrainerEngine unavailable, falling back to legacy path")
        return self._run_legacy()

    def _run_legacy(self) -> bool:
        """
        DEPRECATED: Legacy training path.

        This method contains the original training implementation that
        manually handles model loading, dataset prep, and trainer creation.

        Use _run_with_engine() instead (default since engine-first refactor).
        This path remains for fallback and backward compatibility.

        To force legacy: USE_LEGACY_TRAINER=1

        Returns:
            bool: True if training succeeded, False otherwise
        """
        print("Running legacy training path...")
        print("-" * 80)

        # Step 0: Enforce locked config values to prevent bad runs
        self.enforce_locked_config()

        # Reset CUDA peak memory stats to capture per-run metrics
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        # Step 1: Validate dataset
        if not self.args.skip_validation:
            print("\n📋 STEP 1: Validating Dataset")
            print("─" * 80)

            if not self.validate_dataset():
                print("\n❌ Validation failed! Aborting.")
                print("Fix the issues above or use --skip-validation (not recommended)")
                return False

            print("✅ Dataset validation passed!")
        else:
            print("\n⚠️  Skipping validation (not recommended!)")

        # Step 2: Select/Load model
        print("\n📋 STEP 2: Loading Model")
        print("─" * 80)

        if not self.load_model():
            print("\n❌ Failed to load model!")
            return False

        print("✅ Model loaded successfully!")

        # Step 3: Load and prepare dataset
        print("\n📋 STEP 3: Preparing Dataset")
        print("─" * 80)

        if not self.prepare_dataset():
            print("\n❌ Failed to prepare dataset!")
            return False

        print(f"✅ Dataset ready: {len(self.train_dataset)} train examples")

        # Step 4: Time estimation
        print("\n📋 STEP 4: Time Estimation")
        print("─" * 80)

        estimate = self.estimate_time()
        TimeEstimator.display_estimate(estimate)

        # Ask for confirmation
        if not self.args.yes:
            response = input("\n🚦 Continue with training? [yes/no]: ").strip().lower()
            if response != 'yes':
                print("❌ Training cancelled by user")
                return False

        # Step 5: Setup live monitoring
        print("\n📋 STEP 5: Setting Up Live Monitoring")
        print("─" * 80)

        self.setup_live_monitor()
        print(f"✅ Will run inference every {self.args.eval_steps} steps on {len(self.val_dataset)} examples")

        # Step 6: Train!
        print("\n📋 STEP 6: Training")
        print("─" * 80)
        print("🚀 Starting training...")
        print()

        success = self.train()

        if success:
            print("\n" + "=" * 80)
            print("🎉 TRAINING COMPLETE!")
            print("=" * 80)
            print(f"\nModel saved to: {self.args.output_dir}")
            return True
        else:
            print("\n❌ Training failed!")
            return False

    def validate_dataset(self) -> bool:
        """Run pre-training validation."""
        self.validator = DatasetValidator(Path(self.args.dataset))

        if not self.validator.run_full_validation():
            return False

        # Ask user to confirm
        if not self.args.yes:
            print("\n" + "=" * 80)
            response = input("Proceed with this dataset? [yes/no]: ").strip().lower()
            if response != 'yes':
                return False

        return True

    def load_model(self) -> bool:
        """
        DEPRECATED: Load base model and tokenizer.

        This method is used by the legacy training path (_run_legacy).
        For new code, use TrainerEngine._load_model_and_tokenizer() instead.

        The engine-first refactor delegates model loading to TrainerEngine.
        """
        try:
            # Find model in database
            model_info = self.model_db.get_model(self.args.model)

            if model_info:
                model_path = model_info.path
                print(f"   Loading: {model_info.name}")
                print(f"   Size: {model_info.size_gb} GB ({model_info.num_layers} layers)")
            else:
                # Assume it's a path or HF model name
                model_path = self.args.model
                print(f"   Loading: {model_path}")

            # Setup quantization config if using 4-bit quantization
            quantization_config = None
            if getattr(self.args, 'load_in_4bit', False):
                print("   🔧 Enabling 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                print("   ✓ 4-bit quantization config created (NF4, bfloat16 compute)")

            # Load model (try Qwen3VL first, then AutoModel, fallback to AutoModelForCausalLM)
            print("   Loading model...")

            # Try flash_attention_2 if available, fallback to sdpa
            attn_impl = "sdpa"
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print("   ✓ Flash Attention 2 detected, will use optimized attention")
            except ImportError:
                print("   Using SDPA attention (flash_attention_2 not installed)")

            model_kwargs = {
                # "device_map": "auto",  # DISABLED - was causing OOM by pre-allocating 90% of GPU
                "trust_remote_code": True,
                "attn_implementation": attn_impl
            }

            # Add quantization config if using QLoRA, otherwise set precision
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                # Use precision from config if available, otherwise default to bfloat16
                if self.config and self.config.hyperparams.fp_precision:
                    precision = self.config.hyperparams.fp_precision
                    if precision == "bf16":
                        model_kwargs["torch_dtype"] = torch.bfloat16
                        print(f"   Using BF16 precision (from config)")
                    elif precision == "fp16":
                        model_kwargs["torch_dtype"] = torch.float16
                        print(f"   Using FP16 precision (from config)")
                    elif precision == "fp32":
                        model_kwargs["torch_dtype"] = torch.float32
                        print(f"   Using FP32 precision (from config)")
                    else:
                        model_kwargs["torch_dtype"] = torch.bfloat16
                        print(f"   ⚠️  Unknown precision '{precision}', defaulting to BF16")
                else:
                    # Fallback to original behavior
                    model_kwargs["torch_dtype"] = torch.bfloat16
                    print(f"   Using BF16 precision (default)")

            # Try loading as Qwen3VL (for VL models)
            self.is_vision_model = False  # Track model type for formatting
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
                print("   ✓ Loaded as Qwen3VLForConditionalGeneration")
                self.is_vision_model = True

                # Freeze vision and video towers for text-only training
                print("   🔒 Freezing vision/video towers for text-only training...")
                frozen_params = 0
                for n, p in self.model.named_parameters():
                    if any(k in n for k in ["vision_model", "video_model", "visual"]):
                        p.requires_grad = False
                        frozen_params += 1
                print(f"   ✓ Froze {frozen_params} vision/video parameters")

                # Use AutoProcessor for Qwen3VL
                print("   Loading processor...")
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
                print("   ✓ Loaded AutoProcessor")

            except Exception as e:
                # Qwen3VL failed, try standard CausalLM (e.g., Qwen2.5, Llama, etc.)
                print(f"   Qwen3VL failed ({str(e)[:50]}...), trying AutoModelForCausalLM...")
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                print("   ✓ Loaded as AutoModelForCausalLM")

                # Load tokenizer
                print("   Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("   ✓ Set pad_token = eos_token")

            # Override chat template if needed (fixes Qwen3 <think> injection conflict)
            # Must be done BEFORE formatting examples for training
            profile_name = self.config.profile.name if self.config and self.config.profile else None
            apply_chat_template_override(self.tokenizer, profile_name=profile_name, verbose=True)

            # Disable KV cache for training (saves memory)
            self.model.config.use_cache = False
            print("   ✓ Disabled KV cache (use_cache=False)")

            # Enable gradient checkpointing if configured (saves VRAM, trades compute)
            use_grad_ckpt = False
            if self.config and hasattr(self.config.hyperparams, 'use_gradient_checkpointing'):
                use_grad_ckpt = self.config.hyperparams.use_gradient_checkpointing
            if use_grad_ckpt and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("   ✓ Enabled gradient checkpointing (saves ~40% VRAM, ~15% slower training)")
            else:
                print("   ⚠️  Gradient checkpointing disabled (higher VRAM usage)")

            # Full model training (no LoRA)
            print("   ✓ Full model fine-tuning enabled (all weights trainable)")

            # Load profile for logit processors
            # If config available, use profile from config; otherwise use legacy emoji_think
            if self.config and self.config.profile.name:
                profile_name = self.config.profile.name
                try:
                    self.profile = get_profile(profile_name)
                    print(f"   ✓ Loaded profile: {profile_name}")

                    # Use profile's logit processors
                    combined_processors = self.profile.build_logits_processors(self.tokenizer)
                    print(f"   ✓ Built logit processors from profile ({len(combined_processors)} processors)")
                except Exception as e:
                    print(f"   ⚠️  Failed to load profile '{profile_name}': {e}")
                    print(f"      Falling back to legacy emoji_think logic")
                    self.profile = None
                    combined_processors = None
            else:
                self.profile = None
                combined_processors = None

            # Fallback to legacy logit processors if profile not available
            if combined_processors is None:
                print("   Building legacy emoji_think logit processors...")
                combined_processors = LogitsProcessorList()

                think_processors = build_think_penalty_processor(
                    self.tokenizer,
                    penalty=80.0,
                    schedule=DEFAULT_PENALTY_SCHEDULE,
                )
                if len(think_processors) > 0:
                    combined_processors.extend(think_processors)
                    print("   ✓ Enabled logit penalty for <think> tags")

                # Penalize tokens after stop emoji sequences with escalating penalties
                post_stop_processors = build_post_stop_penalty_processor(
                    self.tokenizer,
                    stop_emoji_pool=STOP_EMOJI_POOL,
                    stop_count_min=STOP_COUNT_MIN,
                    stop_count_max=STOP_COUNT_MAX,
                    base_penalty=100.0,
                    escalation_rate=10.0,
                    eot_reward=50.0,
                    eot_sequence=None,
                )
                if len(post_stop_processors) > 0:
                    combined_processors.extend(post_stop_processors)
                    print("   ✓ Enabled escalating penalty for tokens after stop signal")

            self.logits_processor = combined_processors if len(combined_processors) > 0 else None

            return True

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_dataset(self) -> bool:
        """
        DEPRECATED: Load and prepare dataset.

        This method is used by the legacy training path (_run_legacy).
        For new code, use TrainerEngine._prepare_dataset() instead.

        The engine-first refactor delegates dataset preparation to TrainerEngine.
        """
        try:
            # Build enforced system prompt with date (use profile, CLI arg, or default)
            from datetime import datetime

            # Priority: profile > CLI arg > default
            if self.profile and hasattr(self.profile, 'get_system_prompt_template'):
                # Profile provides template with {date} or {current_date} placeholder
                base_prompt = self.profile.get_system_prompt_template()
                current_date = datetime.now().strftime('%Y-%m-%d')
                # Support both placeholder formats
                enforced_system_prompt = base_prompt.replace("{date}", current_date).replace("{current_date}", current_date)
                print(f"   Using system prompt from profile: {self.profile.__class__.__name__}")
            elif hasattr(self.args, 'system_prompt') and self.args.system_prompt:
                # Use CLI arg with date prefix
                base_prompt = self.args.system_prompt
                enforced_system_prompt = f"Today is {datetime.now().strftime('%Y-%m-%d')}. {base_prompt}"
                print(f"   Using system prompt from CLI arg")
            else:
                # Fallback to default (from single source of truth)
                from core.prompts import format_system_prompt
                enforced_system_prompt = format_system_prompt()
                print(f"   Using default system prompt")

            self.enforced_system_prompt = enforced_system_prompt

            # Load examples
            examples = []
            with open(self.args.dataset) as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))

            # Shuffle
            random.shuffle(examples)

            # Split train/val
            val_size = min(100, int(len(examples) * 0.05))
            val_examples = examples[:val_size]
            train_examples = examples[val_size:]

            # Use profile loaded in load_model() if available
            # (profile is loaded early for logit processors)
            profile = self.profile if hasattr(self, 'profile') else None

            # Transform examples using profile (new) or legacy logic (fallback)
            if profile:
                profile_name = self.config.profile.name if self.config else "unknown"
                # NEW: Use profile.transform_example()
                def transform_example(ex, idx):
                    """Transform example using loaded profile."""
                    transformed = profile.transform_example(ex, idx, self.enforced_system_prompt)
                    return self.sanitize_example(transformed)

                print(f"   Using profile transformation: {profile_name}")
            else:
                # LEGACY: Use hard-coded emoji/stop logic
                def transform_example(ex, idx):
                    """Inject system prompt and apply random thinking pattern based on index."""
                    msgs = ex.get('messages', [])
                    if not msgs or msgs[0].get('role') != 'system':
                        msgs = [{"role": "system", "content": self.enforced_system_prompt}] + msgs
                    else:
                        msgs[0]["content"] = self.enforced_system_prompt
                    new_ex = dict(ex)
                    # Pass index to get RANDOM thinking pattern for this example
                    new_ex['messages'] = self.enforce_thinking_requirement(msgs, example_index=idx)
                    new_ex['messages'] = self.enforce_stop_requirement(new_ex['messages'])
                    return self.sanitize_example(new_ex)

                print(f"   Using legacy transformation (hard-coded emoji_think)")

            # Apply transformation to train/val examples
            train_examples = [transform_example(ex, idx) for idx, ex in enumerate(train_examples)]
            val_examples = [transform_example(ex, idx) for idx, ex in enumerate(val_examples)]

            # Keep raw examples for eval display (with enforced system prompt)
            self.raw_train_examples = train_examples

            print(f"   Total: {len(examples):,} examples")
            print(f"   Train: {len(train_examples):,}")
            print(f"   Val: {len(val_examples):,}")
            print(f"   System prompt: \"{self.enforced_system_prompt}\"")

            # Format for training
            def format_example(ex):
                messages = ex['messages']

                # Format messages based on model type
                formatted_messages = []
                for msg in messages:
                    content = msg.get('content', '')

                    # Convert non-string content to JSON string
                    if not isinstance(content, str):
                        import json
                        content = json.dumps(content, ensure_ascii=False)

                    # Wrap as list for vision models (Qwen3VL), keep as string for text models (Qwen2.5, etc)
                    if self.is_vision_model:
                        content = [{"type": "text", "text": content}]

                    formatted_messages.append({
                        "role": msg['role'],
                        "content": content
                    })

                text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return {"text": text}

            train_data = [format_example(ex) for ex in train_examples]
            self.train_dataset = Dataset.from_list(train_data)

            # Format validation data for eval_dataset
            val_data = [format_example(ex) for ex in val_examples]
            self.val_dataset_formatted = Dataset.from_list(val_data)

            # Keep val examples in original format for live monitoring
            self.val_dataset = val_examples

            # MEMORY FIX: Free intermediate data structures
            del examples
            del train_data
            del val_data
            import gc
            gc.collect()

            # Initialize evolution tracker
            dataset_name = Path(self.args.dataset).stem  # Get filename without extension
            base_dir = Path(self.args.dataset).parent if hasattr(self.args, 'base_dir') else Path.cwd()
            self.evolution_tracker = EvolutionTracker(base_dir, dataset_name)
            print(f"   Evolution tracking enabled for: {dataset_name}")

            # Load fixed validation set (if available)
            self.load_fixed_validation_set()

            return True

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_fixed_validation_set(self):
        """Load fixed validation set from data/validation directory."""
        try:
            # Look for validation file in data/validation/
            validation_file = Path("data/validation/syllo_validation_1000.jsonl")

            if not validation_file.exists():
                print(f"   ⚠️  No fixed validation set found at {validation_file}")
                print(f"   ℹ️  Will use training split for validation (less ideal)")
                self.fixed_val_dataset = None
                return

            # Load validation examples
            val_examples = []
            with open(validation_file) as f:
                for line in f:
                    if line.strip():
                        val_examples.append(json.loads(line))

            # Sample for efficiency (100 examples is enough for validation)
            if len(val_examples) > 100:
                import random
                random.seed(42)  # Reproducible sampling
                val_examples = random.sample(val_examples, 100)

            # Apply same transformations as training data (thinking/stop patterns)
            def inject_system_val(ex, idx):
                """Apply same transformations as training data for consistency."""
                msgs = ex.get('messages', [])
                if not msgs or msgs[0].get('role') != 'system':
                    msgs = [{"role": "system", "content": self.enforced_system_prompt}] + msgs
                else:
                    msgs[0]["content"] = self.enforced_system_prompt
                new_ex = dict(ex)
                # Apply thinking and stop patterns (same as training data)
                new_ex['messages'] = self.enforce_thinking_requirement(msgs, example_index=idx)
                new_ex['messages'] = self.enforce_stop_requirement(new_ex['messages'])
                return new_ex

            # Process examples with same pipeline as training data
            val_examples = [self.sanitize_example(inject_system_val(ex, idx)) for idx, ex in enumerate(val_examples)]

            # Format validation examples same as training data
            def format_example(ex):
                messages = ex['messages']

                # Format messages based on model type
                formatted_messages = []
                for msg in messages:
                    content = msg.get('content', '')

                    # Convert non-string content to JSON string
                    if not isinstance(content, str):
                        import json
                        content = json.dumps(content, ensure_ascii=False)

                    # Wrap as list for vision models
                    if self.is_vision_model:
                        content = [{"type": "text", "text": content}]

                    formatted_messages.append({
                        "role": msg['role'],
                        "content": content
                    })

                text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return {"text": text}

            # Create Dataset
            formatted_val = [format_example(ex) for ex in val_examples]
            self.fixed_val_dataset = Dataset.from_list(formatted_val)

            print(f"   ✅ Loaded {len(self.fixed_val_dataset)} fixed validation examples")
            print(f"   📊 This set is separate from training data for true generalization metrics")

            # Tokenize validation dataset (same approach as training data)
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self._get_hyperparam('max_length', 2048),
                    padding=False
                )

            # Tokenize the validation dataset
            self.tokenized_fixed_val_small = self.fixed_val_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=10,
                remove_columns=["text"],
                num_proc=None  # Disable multiprocessing
            )
            print(f"   🔎 Tokenized {len(self.tokenized_fixed_val_small)} validation examples")

        except Exception as e:
            print(f"   ⚠️  Failed to load fixed validation set: {e}")
            self.fixed_val_dataset = None
            self.tokenized_fixed_val_small = None

    def estimate_time(self):
        """Estimate training time."""
        # Get model size
        model_info = self.model_db.get_model(self.args.model)
        if model_info:
            # Rough estimate: hidden_size * num_layers → billions of params
            model_size_b = (model_info.hidden_size * model_info.num_layers) / 1000
        else:
            model_size_b = 8.0  # Default guess

        effective_batch = self._get_hyperparam('batch_size', 1) * self._get_hyperparam('gradient_accumulation', 1)

        return TimeEstimator.estimate_training(
            num_examples=len(self.train_dataset),
            batch_size=effective_batch,
            num_epochs=self.args.epochs,
            model_size_b=model_size_b
        )

    def setup_live_monitor(self):
        """Setup live inference monitoring."""
        # Get monitoring settings from config
        # Use config if available, otherwise load from config.json (legacy)
        max_eval_tokens = 2048  # Default
        self_correction_config = {}

        if self.config and self.config.monitoring:
            max_eval_tokens = self.config.monitoring.max_eval_tokens
            # Still need to get self_correction_config from config.json for now
            config_path = Path("config.json")
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                        self_correction_config = cfg.get("self_correction", {})
                except Exception as e:
                    print(f"⚠️  Could not load self-correction config: {e}")
        else:
            # Fallback to reading config.json directly (legacy)
            config_path = Path("config.json")
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                        self_correction_config = cfg.get("self_correction", {})
                        max_eval_tokens = cfg.get("eval_max_tokens", 2048)
                except Exception as e:
                    print(f"⚠️  Could not load self-correction config: {e}")

        # Create self-correction generator if enabled
        self_correction_gen = None
        if self_correction_config.get("enabled", False) and SELF_CORRECTION_AVAILABLE:
            try:
                max_examples = self_correction_config.get("max_examples", 200)
                generation_interval = self_correction_config.get("generation_interval")

                self_correction_gen = create_self_correction_monitor(
                    enable=True,
                    auto_queue=self_correction_config.get("auto_queue", True),
                    max_examples=max_examples,
                    generation_interval=generation_interval
                )

                if generation_interval:
                    print(f"✅ Auto self-correction enabled (every {max_examples} examples OR {generation_interval} steps)")
                else:
                    print(f"✅ Auto self-correction enabled (auto-drop every {max_examples} examples)")
            except Exception as e:
                print(f"⚠️  Could not enable self-correction: {e}")

        self.live_monitor = LiveInferenceMonitor(
            model=self.model,
            tokenizer=self.tokenizer,
            val_examples=self.val_dataset,
            num_samples=self.args.num_eval_samples,
            max_new_tokens=max_eval_tokens,
            logits_processor=self.logits_processor,
            self_correction_generator=self_correction_gen
        )

    def train(self) -> bool:
        """
        DEPRECATED: Run actual training.

        This method is used by the legacy training path (_run_legacy).
        For new code, use TrainerEngine.run_job() instead.

        The engine-first refactor delegates training execution to TrainerEngine.
        """
        try:
            # Tokenize dataset - get max_length from config
            max_length = self._get_hyperparam('max_length', 2048)

            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=max_length,
                    padding=False
                )

            # MEMORY FIX: Add parameters to prevent memory explosion
            # CUDA FIX: num_proc=None completely disables multiprocessing to avoid CUDA fork errors
            tokenized_dataset = self.train_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=10,  # Process in very small chunks to avoid OOM
                remove_columns=["text"],
                num_proc=None,  # Completely disable multiprocessing (not even 1 worker)
                load_from_cache_file=False,  # Don't cache to prevent accumulation
                writer_batch_size=10,  # Write in smaller batches
                desc="Tokenizing dataset"
            )

            # Pack dataset for efficiency (fill context windows more completely)
            enable_packing = os.environ.get("ENABLE_PACKING", "1").lower() in ("1", "true", "yes", "on")
            if enable_packing:
                try:
                    from trl import pack_dataset
                    print(f"   📦 Packing dataset (max_length={max_length})...")
                    print(f"      Before packing: {len(tokenized_dataset)} examples")
                    tokenized_dataset = pack_dataset(
                        tokenized_dataset,
                        seq_length=max_length,
                        strategy="bfd"  # Best Fit Decreasing - preserves sequence boundaries
                    )
                    print(f"      After packing: {len(tokenized_dataset)} packed sequences")

                    # Remove seq_lengths metadata (list of lists) - collator can't handle it
                    if 'seq_lengths' in tokenized_dataset.column_names:
                        tokenized_dataset = tokenized_dataset.remove_columns(['seq_lengths'])
                        print(f"      ✅ Removed seq_lengths metadata (incompatible with collator)")

                    print(f"      ✅ Packing enabled - filling {max_length}-token blocks efficiently")
                except Exception as e:
                    print(f"      ⚠️  Packing failed ({e}), continuing without packing")
            else:
                print(f"   ⚠️  Packing disabled via ENABLE_PACKING=0")

            # Tokenize validation dataset for eval_loss
            tokenized_val_dataset = self.val_dataset_formatted.map(
                tokenize_function,
                batched=True,
                batch_size=10,
                remove_columns=["text"],
                num_proc=None,
                load_from_cache_file=False,
                writer_batch_size=10,
                desc="Tokenizing validation dataset"
            )
            print(f"   ✅ Tokenized {len(tokenized_val_dataset)} validation examples for eval_loss")

            # Estimate average sequence length from a small sample for throughput metrics
            sample_count = min(100, len(tokenized_dataset))
            if sample_count > 0:
                total_tokens = 0
                for i in range(sample_count):
                    total_tokens += len(tokenized_dataset[i]["input_ids"])
                self.avg_seq_len = total_tokens / sample_count
                print(f"   📏 Avg seq length (sample {sample_count}): {self.avg_seq_len:.1f} tokens")

            # Force garbage collection after tokenization to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Calculate steps for this dataset
            effective_batch = self._get_hyperparam('batch_size', 1) * self._get_hyperparam('gradient_accumulation', 1)
            steps_for_this_file = len(tokenized_dataset) // effective_batch
            if len(tokenized_dataset) % effective_batch != 0:
                steps_for_this_file += 1  # Account for partial batch

            # Check for existing checkpoint to get current global_step
            current_global_step = 0
            print(f"🔍 Checking for checkpoints in: {self.args.output_dir}")
            if Path(self.args.output_dir).exists():
                checkpoints = list(Path(self.args.output_dir).glob("checkpoint-*"))
                print(f"🔍 Found {len(checkpoints)} checkpoints")
                checkpoint_candidates = []
                for cp in checkpoints:
                    parts = cp.name.split("-", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        step = int(parts[1])
                    except ValueError:
                        continue
                    checkpoint_candidates.append((step, cp))

                if checkpoint_candidates:
                    latest_checkpoint = max(checkpoint_candidates, key=lambda item: item[0])[1]
                    print(f"🔍 Latest checkpoint: {latest_checkpoint}")
                    trainer_state_file = latest_checkpoint / "trainer_state.json"
                    print(f"🔍 State file exists: {trainer_state_file.exists()}")
                    if trainer_state_file.exists():
                        with open(trainer_state_file, 'r') as f:
                            trainer_state = json.load(f)
                            current_global_step = trainer_state.get('global_step', 0)
                            print(f"📊 Current training progress: {current_global_step} steps completed")
                    else:
                        print("⚠️  Latest checkpoint missing trainer_state.json; treating as fresh start.")
                else:
                    print("🔍 No checkpoints found - starting from step 0")
            else:
                print(f"🔍 Output dir doesn't exist: {self.args.output_dir}")

            # Calculate max_steps for continuous training
            # This ensures each new file adds its steps to the running total
            max_steps = current_global_step + steps_for_this_file

            print(f"📈 Total steps after this batch: {current_global_step} + {steps_for_this_file} = {max_steps}")

            # Determine precision flags from config
            # Use config precision if available, otherwise default to bf16 (matches model loading)
            use_fp16 = False
            use_bf16 = False
            if self.config and self.config.hyperparams.fp_precision:
                precision = self.config.hyperparams.fp_precision
                if precision == "bf16":
                    use_bf16 = True
                elif precision == "fp16":
                    use_fp16 = True
                # fp32: both False
            else:
                # Fallback to match model loading default (bf16) - avoids BF16/FP16 mismatch
                use_bf16 = True

            # Training arguments
            # Use max_steps for continuous training instead of num_train_epochs
            # Determine last checkpoint step to control save frequency
            # HuggingFace Trainer saves at END of trainer.train() by default
            # We need to track this ourselves to avoid checkpoints every file
            last_checkpoint_step = 0
            checkpoint_paths = sorted(Path(self.args.output_dir).glob("checkpoint-*"))
            if checkpoint_paths:
                try:
                    last_checkpoint_step = int(checkpoint_paths[-1].name.split("-")[1])
                except (ValueError, IndexError):
                    pass

            # Calculate next checkpoint boundary
            save_steps = self.args.save_steps
            next_checkpoint_at = ((last_checkpoint_step // save_steps) + 1) * save_steps

            # Only enable saving if we'll reach the next checkpoint boundary
            # This prevents end-of-file saves that create checkpoints at every step
            should_enable_saving = max_steps >= next_checkpoint_at

            training_args = TrainingArguments(
                output_dir=self.args.output_dir,
                max_steps=max_steps,  # Use max_steps for continuous training
                per_device_train_batch_size=self._get_hyperparam('batch_size', 1),
                per_device_eval_batch_size=1,  # MEMORY FIX: Match train batch size to avoid eval spikes
                gradient_accumulation_steps=self._get_hyperparam('gradient_accumulation', 1),
                learning_rate=self._get_hyperparam('learning_rate', 2e-4),
                warmup_steps=self.args.warmup_steps,
                lr_scheduler_type="constant",  # FIX: Don't decay LR for continuous training on varied files
                logging_steps=10,
                save_steps=save_steps,
                eval_steps=self.args.eval_steps,
                eval_strategy="steps" if self.args.eval_steps else "no",
                # CRITICAL: Disable saves unless we'll cross a checkpoint boundary
                # This prevents the end-of-training save that creates checkpoints at every file
                save_strategy="steps" if should_enable_saving else "no",
                save_total_limit=None,  # Keep all checkpoints - manually clean by date later
                fp16=use_fp16,   # Set from config precision
                bf16=use_bf16,   # Set from config precision
                tf32=True,  # Enable TF32 for matrix math speedups
                gradient_checkpointing=True,  # MEMORY FIX: Enable to save ~40-50% activation memory
                optim="adamw_torch_fused",  # Faster than default AdamW
                dataloader_num_workers=16,  # INCREASED: Leverage 7950X (32 threads) + DDR5
                dataloader_pin_memory=True,  # Faster CPU->GPU transfer
                report_to="none",
                remove_unused_columns=False,
            )

            # Log active hyperparameters so we know what's running
            active_lr = training_args.learning_rate
            active_scheduler = training_args.lr_scheduler_type
            print(f"⚡ ACTIVE CONFIG: lr={active_lr}, scheduler={active_scheduler}, warmup={training_args.warmup_steps}")

            if should_enable_saving:
                print(f"💾 Checkpoints enabled: next save at step {next_checkpoint_at}")
            else:
                print(f"💾 Checkpoints disabled: won't reach {next_checkpoint_at} (max_steps={max_steps})")

            # Data collator - ONLY train on assistant response, NOT the user prompt
            # This prevents the model from learning to output the full conversation format
            data_collator = DataCollatorForCompletionOnly(
                tokenizer=self.tokenizer,
                response_template="<|im_start|>assistant\n"
            )

            print("   ✅ Using response-only collator (instruction portion will be masked)")
            print("   ℹ️  Model will ONLY learn to generate the assistant's answer")

            # VERIFY masking is working with comprehensive validation
            # CRITICAL: Prevents training on instructions (bug discovered 2025-11-27)
            if len(tokenized_dataset) > 0:
                # Test on multiple samples for packed sequences
                sample_count = min(5, len(tokenized_dataset))
                test_samples = [tokenized_dataset[i] for i in range(sample_count)]
                test_batch = data_collator(test_samples)

                # Basic stats
                masked_count = (test_batch['labels'] == -100).sum().item()
                total_count = test_batch['labels'].numel()
                trained_count = total_count - masked_count
                masked_pct = 100 * masked_count / total_count

                print(f"   🔍 Masking verification ({sample_count} samples):")
                print(f"      Total tokens: {total_count}")
                print(f"      Masked (instruction): {masked_count} ({masked_pct:.1f}%)")
                print(f"      Trained (response): {trained_count} ({100*trained_count/total_count:.1f}%)")

                # Run full validation suite
                try:
                    from masking_validators import validate_masking, MaskingValidationError
                    enable_packing = os.environ.get("ENABLE_PACKING", "1").lower() in ("1", "true", "yes", "on")

                    report = validate_masking(
                        batch=test_batch,
                        tokenizer=self.tokenizer,
                        response_template="<|im_start|>assistant\n",
                        end_token="<|im_end|>",
                        packing_enabled=enable_packing,
                        strict=False,  # Don't raise, we'll check results
                    )

                    if report.passed:
                        print(f"   ✅ All masking validations passed")
                    else:
                        print(f"   ❌ MASKING VALIDATION FAILED!")
                        for r in report.results:
                            if not r.passed:
                                print(f"      - {r.validator_name}: {r.message}")
                        print(f"   ⚠️  This could cause the model to learn instructions!")
                        print(f"   ⚠️  Check custom_collator.py and packing settings")
                        # Fail hard to prevent bad training (15% for short-response data)
                        if masked_pct < 15:
                            print(f"   ❌ ABORTING: Masking too low ({masked_pct:.1f}% < 15%)")
                            return False
                except ImportError:
                    # Fallback to basic check if validators not available
                    if masked_count == 0:
                        print(f"   ❌ ERROR: NO MASKING DETECTED!")
                        print(f"      Training would learn full conversation format")
                        print(f"      ABORTING to prevent bad training")
                        return False
                    elif masked_pct < 20:
                        print(f"   ⚠️  WARNING: Very little masking ({masked_pct:.1f}% < 20%)")
                        print(f"      This might indicate training on instructions!")
                    else:
                        print(f"   ✅ Masking looks correct (instruction is masked)")

            # Use the calculated values from earlier:
            # - current_global_step (from checkpoint)
            # - steps_for_this_file (for this dataset)
            # - max_steps (target total)

            # Get batch context from args (passed by daemon)
            current_file = getattr(self.args, 'current_file', None)
            batch_number = getattr(self.args, 'batch_number', None)
            batch_queue_size = getattr(self.args, 'batch_queue_size', None)

            if self.layer_monitor is None:
                try:
                    self.layer_monitor = LayerMonitor(self.model)
                except Exception as exc:
                    print(f"⚠️ Layer monitor disabled: {exc}")
                    self.layer_monitor = None

            # Create optimizer (Muon or AdamW based on config)
            custom_optimizer = None
            custom_scheduler = None
            optimizer_type = "adamw"  # default

            if MUON_AVAILABLE:
                # Get optimizer config from raw config dict (includes extra fields)
                opt_config = self.config_dict.get("optimizer", {})
                optimizer_type = opt_config.get("type", "adamw") if isinstance(opt_config, dict) else "adamw"

                if optimizer_type == "muon":
                    print("\n" + "=" * 60)
                    print("MUON OPTIMIZER - Orthogonalized Momentum")
                    print("=" * 60)

                    # Show parameter grouping summary
                    summary = get_param_group_summary(self.model)
                    total = summary.muon_params + summary.adam_params
                    print(f"  Hidden weights (Muon): {summary.muon_params:,} params ({100*summary.muon_params/total:.1f}%)")
                    print(f"  Other (AdamW):         {summary.adam_params:,} params ({100*summary.adam_params/total:.1f}%)")

                    # Create optimizer with config
                    custom_optimizer, custom_scheduler, opt_info = create_custom_optimizer(
                        self.model,
                        self.config_dict,
                        optimizer_type="muon",
                        num_training_steps=max_steps,
                        num_warmup_steps=self.args.warmup_steps,
                    )

                    print(f"  Hidden LR: {opt_info.hidden_lr}")
                    print(f"  Aux LR:    {opt_info.aux_lr}")
                    print("=" * 60 + "\n")

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_val_dataset,  # Use validation split from current batch
                data_collator=data_collator,
                callbacks=[LiveMonitorCallback(
                    self.live_monitor,
                    self.status_writer,
                    self.args.eval_steps,
                    max_steps,  # ✅ FIX: Use max_steps (cumulative total)
                    self.raw_train_examples,
                    self.tokenizer,
                    self.model,
                    batch_total_steps=steps_for_this_file,  # ✅ FIX: Use steps_for_this_file
                    current_global_step=current_global_step,
                    evolution_tracker=self.evolution_tracker,
                    current_file=current_file,
                    batch_number=batch_number,
                    batch_queue_size=batch_queue_size,
                    controller=self.controller,  # Pass controller for pause/stop detection
                    fixed_val_dataset=self.fixed_val_dataset,  # Pass fixed validation set
                    remote_eval_config=getattr(self.config, "remote_eval", {}) if self.config else {},  # Remote evaluation config
                    avg_seq_len=self.avg_seq_len,
                    effective_batch=effective_batch,
                    micro_eval_inputs=self.tokenized_fixed_val_small,
                    micro_eval_interval=max(500, self.args.eval_steps),
                    logits_processor=self.logits_processor,
                    layer_monitor=self.layer_monitor,
                )],
                optimizers=(custom_optimizer, custom_scheduler) if custom_optimizer else (None, None),
            )

            # Enable detailed monitoring if available
            # DISABLED: Causes hangs every 50 steps
            if False and DETAILED_MONITORING_AVAILABLE and hasattr(self, 'val_dataset'):
                try:
                    add_detail_collector_to_trainer(
                        trainer=trainer,
                        tokenizer=self.tokenizer,
                        eval_dataset=self.val_dataset,
                        update_frequency=50  # Update every 50 steps
                    )
                    print("✅ Detailed monitoring enabled - visit http://localhost:8081")
                except Exception as e:
                    print(f"⚠️  Could not enable detailed monitoring: {e}")

            # Train!
            self.training_start_time = time.time()

            # Check for existing checkpoint to resume from (preserves optimizer state)
            resume_from_checkpoint = None
            if Path(self.args.output_dir).exists():
                checkpoint_dir = Path(self.args.output_dir)
                checkpoint_candidates = []
                for cp in checkpoint_dir.glob("checkpoint-*"):
                    parts = cp.name.split("-", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        step = int(parts[1])
                    except ValueError:
                        continue
                    checkpoint_candidates.append((step, cp))

                if checkpoint_candidates:
                    # Use the latest checkpoint based on step number
                    latest_checkpoint = max(checkpoint_candidates, key=lambda item: item[0])[1]
                    trainer_state_path = latest_checkpoint / "trainer_state.json"
                    if trainer_state_path.exists():
                        resume_from_checkpoint = str(latest_checkpoint)
                        print(f"📦 Resuming from checkpoint: {resume_from_checkpoint}")
                        print(f"   (This preserves optimizer state for smooth continuation)")
                    else:
                        print(
                            f"⚠️  Skipping checkpoint resume: {trainer_state_path} is missing "
                            "trainer_state.json (optimizer state will be reinitialized)."
                        )

            # Delete old scheduler state to force fresh constant scheduler
            # This prevents the decayed LR from being restored on resume
            if resume_from_checkpoint:
                old_scheduler = Path(resume_from_checkpoint) / "scheduler.pt"
                if old_scheduler.exists():
                    old_scheduler.unlink()
                    print(f"🔄 Deleted old scheduler.pt to force fresh constant LR")

            # Train with checkpoint resumption if available
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # CRITICAL: Save checkpoint at end of each file to preserve progress
            # Without this, next file resumes from last save_steps checkpoint (loses progress)
            final_step = trainer.state.global_step
            checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{final_step}"
            if not checkpoint_dir.exists():
                print(f"💾 Saving end-of-file checkpoint: {checkpoint_dir.name}")
                # Use internal method to save FULL checkpoint (model + optimizer + scheduler + state)
                trainer._save_checkpoint(model=trainer.model, trial=None)
            else:
                print(f"✅ Checkpoint already exists: {checkpoint_dir.name}")

            # Save final adapter model (for inference/deployment)
            # NOTE: Checkpoints are auto-managed by Trainer via save_total_limit
            # This save_model() is just for having a clean adapter at root for inference
            print(f"💾 Saving final adapter model to: {self.args.output_dir}")
            trainer.save_model(self.args.output_dir)
            self.tokenizer.save_pretrained(self.args.output_dir)

            # Checkpoints managed via save_steps from config (default: 10000)
            # Saves disabled for files that won't cross a checkpoint boundary
            checkpoint_paths = sorted(Path(self.args.output_dir).glob("checkpoint-*"))
            if checkpoint_paths:
                latest_checkpoint = checkpoint_paths[-1]
                print(f"✅ Latest checkpoint ready for next batch: {latest_checkpoint.name}")
                print(f"   Contains: adapter weights, optimizer state, scheduler state, global_step")
                print(f"   Total checkpoints: {len(checkpoint_paths)} (auto-managed by save_total_limit)")
            else:
                print("⚠️  Warning: No checkpoints found!")
                print("   This may indicate training completed before first save_steps interval")

            # Mark training as completed
            self.status_writer.mark_completed(max_steps, max_steps)

            # Send completion notification
            elapsed = time.time() - self.training_start_time
            duration_str = f"{elapsed/60:.1f} minutes"
            self.notifier.training_complete(duration_str)

            self._record_training_summary(success=True, runtime_sec=elapsed)

            return True

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Training error: {error_msg}")
            import traceback
            traceback.print_exc()

            # Mark as crashed
            self.status_writer.mark_crashed(error_msg, type(e).__name__)

            runtime = None
            if self.training_start_time:
                runtime = time.time() - self.training_start_time

            # Send crash notification
            if "out of memory" in error_msg.lower() or "OOM" in error_msg:
                self.notifier.oom_error()
            else:
                self.notifier.training_crashed(error_msg)

            self._record_training_summary(success=False, runtime_sec=runtime)

            return False

    def _record_training_summary(self, success: bool, runtime_sec: Optional[float]):
        try:
            dataset_name = Path(self.args.dataset).name if getattr(self.args, 'dataset', None) else None
            sample_count = len(self.train_dataset) if self.train_dataset is not None else None
            max_golden = getattr(self.status_writer, 'max_golden_output_length', None)
            max_model = getattr(self.status_writer, 'max_model_output_length', None)
            max_length_cfg = getattr(self.args, 'max_length', None)
            headroom_tokens = None
            if max_golden:
                headroom_tokens = math.ceil(min(max_length_cfg or max_golden, max_golden * 1.1))

            gpu_peak_gb = None
            try:
                if torch.cuda.is_available():
                    peak_bytes = torch.cuda.max_memory_allocated()
                    if peak_bytes:
                        gpu_peak_gb = peak_bytes / (1024 ** 3)
            except Exception:
                pass

            summary = {
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset_name,
                "samples": sample_count,
                "success": success,
                "runtime_sec": runtime_sec,
                "batch_size": getattr(self.args, 'batch_size', None),
                "gradient_accumulation": getattr(self.args, 'gradient_accumulation', None),
                "effective_batch": (self._get_hyperparam('batch_size', 1) * self._get_hyperparam('gradient_accumulation', 1))
                if getattr(self.args, 'batch_size', None) and getattr(self.args, 'gradient_accumulation', None) else None,
                "max_length_config": max_length_cfg,
                "observed_max_tokens": max_golden,
                "observed_max_tokens_headroom": headroom_tokens,
                "observed_max_model_tokens": max_model,
                "tokens_per_sec_avg": self.tokens_per_sec_ema,
                "gpu_peak_alloc_gb": gpu_peak_gb
            }
            self.training_summary = summary
        except Exception:
            # Do not let summary collection crash training flow
            self.training_summary = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ultimate Trainer - Train with all safety checks")

    # Required
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSONL")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for adapter")

    # Training params
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")

    # Monitoring
    parser.add_argument("--eval-steps", type=int, default=100, help="Run live inference every N steps")
    parser.add_argument("--num-eval-samples", type=int, default=5, help="Number of examples for live monitoring")
    parser.add_argument("--save-steps", type=int, default=10000, help="Save checkpoint every N steps")

    # Flags
    parser.add_argument("--skip-validation", action="store_true", help="Skip pre-training validation (not recommended!)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")

    # System prompt
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt to prepend to all training examples (default: 'You are a helpful assistant.')")

    return parser.parse_args()


def main():
    args = parse_args()

    trainer = UltimateTrainer(args)
    success = trainer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
