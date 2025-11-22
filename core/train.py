#!/usr/bin/env python3
"""
Ultimate Trainer - Main Orchestrator

Brings together all components:
1. Validator - Check data BEFORE training
2. Model DB - Select model & adapters
3. Time Estimator - Know what you're getting into
4. Live Monitor - See what's learning DURING training
5. Training - Actually do it with all safety checks

Usage:
    python3 train.py \
        --dataset path/to/data.jsonl \
        --model qwen3_0.6b \
        --output-dir ~/my_adapter \
        --epochs 2 \
        --batch-size 4
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
def get_random_stop_emoji():
    """Select a random stop emoji from the pool."""
    import random
    return random.choice(STOP_EMOJI_POOL)

def get_random_stop_count():
    """Select a random stop count (2-4)."""
    import random
    return random.randint(STOP_COUNT_MIN, STOP_COUNT_MAX)

def get_stop_instruction(emoji: str, count: int) -> str:
    """Generate stop instruction for a specific emoji and count."""
    count_words = {2: "twice", 3: "three times", 4: "four times"}
    count_text = count_words.get(count, f"{count} times")
    return f"When finished, emit {emoji} /{count_text}/ to signal completion."

def get_stop_suffix(emoji: str, count: int) -> str:
    """Generate stop suffix for a specific emoji and count."""
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

# Try to import desktop notifier (optional)
try:
    from desktop_notifier import DesktopNotifier
except ImportError:
    # Create stub class if not available
    class DesktopNotifier:
        def __init__(self): pass
        def notify(self, *args, **kwargs): pass
        def notify_success(self, *args, **kwargs): pass
        def notify_error(self, *args, **kwargs): pass


class UltimateTrainer:
    """Main trainer orchestrator."""

    def __init__(self, args, controller=None):
        self.args = args
        self.controller = controller  # Training control system (pause/stop)

        # Create TrainerConfig from args (new config system)
        # This provides single source of truth for all config values
        try:
            self.config = ConfigLoader.from_args_and_json(args)
            print(f"✓ TrainerConfig created (profile: {self.config.profile.name}, precision: {self.config.hyperparams.fp_precision})")
        except Exception as e:
            print(f"⚠️  Could not create TrainerConfig: {e}")
            print(f"   Falling back to args-only mode")
            self.config = None

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
        context_window = getattr(args, 'max_length', 2048)  # Default

        if self.config and self.config.monitoring:
            max_output_tokens = self.config.monitoring.max_output_tokens
            context_window = self.config.hyperparams.max_length

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
                "use_qlora": "use_qlora",
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

    def run(self):
        """Execute full training pipeline."""
        print("\n" + "=" * 80)
        print("🚀 ULTIMATE TRAINER")
        print("=" * 80)

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
        """Load base model and tokenizer."""
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

            # Setup quantization config if using QLoRA
            quantization_config = None
            if getattr(self.args, 'use_qlora', False):
                print("   🔧 Enabling QLoRA (4-bit quantization)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                print("   ✓ QLoRA config created (4-bit NF4, bfloat16 compute)")

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

            # Disable KV cache for training (saves memory)
            self.model.config.use_cache = False
            print("   ✓ Disabled KV cache (use_cache=False)")

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
        """Load and prepare dataset."""
        try:
            # Build enforced system prompt with date (use CLI arg if provided)
            from datetime import datetime

            # Use CLI system prompt if provided, otherwise use default
            base_prompt = self.args.system_prompt if hasattr(self.args, 'system_prompt') else "You are a helpful assistant."

            # Inject current date
            enforced_system_prompt = f"Today is {datetime.now().strftime('%Y-%m-%d')}. {base_prompt}"

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
                    max_length=self.args.max_length,
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

        effective_batch = self.args.batch_size * self.args.gradient_accumulation

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
        """Run actual training."""
        try:
            # Tokenize dataset
            max_length = getattr(self.args, 'max_length', 2048)

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

            # Pack dataset for efficiency (fill 4k blocks completely)
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
                print(f"      ✅ Packing enabled - filling {max_length}-token blocks efficiently")
            except Exception as e:
                print(f"      ⚠️  Packing failed ({e}), continuing without packing")

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
            effective_batch = self.args.batch_size * self.args.gradient_accumulation
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
            # Use config precision if available, otherwise default to fp16
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
                # Fallback to original behavior (fp16)
                use_fp16 = True

            # Training arguments
            # Use max_steps for continuous training instead of num_train_epochs
            training_args = TrainingArguments(
                output_dir=self.args.output_dir,
                max_steps=max_steps,  # ✅ FIX: Use max_steps for continuous training
                per_device_train_batch_size=self.args.batch_size,
                gradient_accumulation_steps=self.args.gradient_accumulation,
                learning_rate=self.args.learning_rate,
                warmup_steps=self.args.warmup_steps,
                logging_steps=10,
                save_steps=self.args.save_steps,
                eval_steps=self.args.eval_steps,
                eval_strategy="steps" if self.args.eval_steps else "no",  # Enable eval loss calculation
                save_total_limit=None,  # Keep all checkpoints - manually clean by date later
                fp16=use_fp16,   # Set from config precision
                bf16=use_bf16,   # Set from config precision
                tf32=True,  # Enable TF32 for matrix math speedups
                gradient_checkpointing=False,  # Keep disabled for speed (only enable if OOM)
                optim="adamw_torch_fused",  # Faster than default AdamW
                dataloader_num_workers=8,  # High workers to feed GPU faster
                dataloader_pin_memory=True,  # Faster CPU->GPU transfer
                report_to="none",
                remove_unused_columns=False,
            )

            # Data collator - ONLY train on assistant response, NOT the user prompt
            # This prevents the model from learning to output the full conversation format
            data_collator = DataCollatorForCompletionOnly(
                tokenizer=self.tokenizer,
                response_template="<|im_start|>assistant\n"
            )

            print("   ✅ Using response-only collator (instruction portion will be masked)")
            print("   ℹ️  Model will ONLY learn to generate the assistant's answer")

            # VERIFY masking is working with a test sample
            if len(tokenized_dataset) > 0:
                test_batch = data_collator([tokenized_dataset[0]])
                masked_count = (test_batch['labels'][0] == -100).sum().item()
                total_count = test_batch['labels'][0].shape[0]
                trained_count = total_count - masked_count

                print(f"   🔍 Masking verification:")
                print(f"      Total tokens: {total_count}")
                print(f"      Masked (instruction): {masked_count} ({100*masked_count/total_count:.1f}%)")
                print(f"      Trained (response): {trained_count} ({100*trained_count/total_count:.1f}%)")

                if masked_count == 0:
                    print(f"   ❌ ERROR: NO MASKING DETECTED!")
                    print(f"      Training would learn full conversation format")
                    print(f"      ABORTING to prevent bad training")
                    return False
                elif masked_count < total_count * 0.3:
                    print(f"   ⚠️  WARNING: Very little masking (< 30%)")
                    print(f"      This might indicate a problem")
                else:
                    print(f"   ✅ Masking looks correct (instruction is masked)")

            # Custom callback for live monitoring
            from transformers import TrainerCallback

            class LiveMonitorCallback(TrainerCallback):
                def __init__(self, monitor, status_writer, eval_steps, total_steps, raw_train_examples, tokenizer, model,
                             batch_total_steps, current_global_step, evolution_tracker=None, current_file=None, batch_number=None, batch_queue_size=None, controller=None, fixed_val_dataset=None,
                             avg_seq_len: float = 0.0, effective_batch: int = 1, micro_eval_inputs=None, micro_eval_interval: int = 500,
                             logits_processor=None, layer_monitor: LayerMonitor | None = None):
                    self.monitor = monitor
                    self.status_writer = status_writer
                    self.eval_steps = eval_steps
                    self.total_steps = total_steps
                    self.raw_train_examples = raw_train_examples
                    self.tokenizer = tokenizer
                    self.model_ref = model
                    self.evolution_tracker = evolution_tracker
                    self.controller = controller  # Training control system
                    self.fixed_val_dataset = fixed_val_dataset  # Fixed validation set
                    self.last_update_time = time.time()
                    self.last_prompt_snapshot_time = time.time()
                    self.update_interval = 2  # Status JSON refresh cadence (seconds)
                    self.prompt_snapshot_interval = 20  # Update prompt/golden cache without inference
                    # Keep inference lightweight: short outputs, frequent previews
                    self.inference_interval_steps = max(10, (self.eval_steps or 200) // 10)
                    self.last_inference_step = 0
                    # Micro-eval settings
                    self.micro_eval_inputs = micro_eval_inputs
                    self.micro_eval_interval = micro_eval_interval
                    self.last_micro_eval_step = 0
                    self.control_check_interval = 10  # Check control signals every 10 steps
                    self.last_control_check_step = 0
                    self.last_val_loss = None  # Track validation loss
                    # Throughput tracking
                    self.prev_step_time = time.time()
                    self.steps_per_sec_ema = None
                    self.tokens_per_sec_ema = None
                    self.avg_seq_len = avg_seq_len
                    self.effective_batch = max(1, effective_batch)
                    # Loss stability tracking
                    self.loss_window = []
                    self.loss_window_size = 50
                    self.loss_trend = None
                    # Alert tracking baseline
                    self.throughput_baseline = None
                    # NEW: Batch progress tracking
                    self.batch_total_steps = batch_total_steps
                    self.current_global_step = current_global_step  # Starting global_step for this batch
                    self.current_file = current_file
                    self.batch_number = batch_number
                    self.batch_queue_size = batch_queue_size
                    self.logits_processor = logits_processor
                    self.pattern_tracker = PatternTracker(get_default_patterns())
                    self.layer_monitor = layer_monitor
                    self.total_vram = None
                    self.low_vram_counter = 0
                    self.last_vram_ratio = None
                    if torch.cuda.is_available():
                        try:
                            props = torch.cuda.get_device_properties(torch.cuda.current_device())
                            self.total_vram = props.total_memory / (1024 ** 3)
                        except Exception:
                            self.total_vram = None
                    self.penalty_last_hits: Dict[str, int] = {}
                    self.penalty_totals: Dict[str, int] = {}
                    self.penalty_files: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
                    self.latest_penalty_delta: Dict[str, int] = {}
                    self.vram_samples = []
                    self.queue_velocity = None
                    self.enable_penalty_metrics = os.environ.get("ENABLE_PENALTY_MONITORING", "0").lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )

                def on_step_end(self, args, state, control, **kwargs):
                    current_time = time.time()
                    current_loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
                    current_lr = state.log_history[-1].get('learning_rate', args.learning_rate) if state.log_history else args.learning_rate
                    current_epoch = state.epoch if state.epoch else 0
                    # Track loss stability
                    self.loss_window.append(current_loss)
                    if len(self.loss_window) > self.loss_window_size:
                        self.loss_window.pop(0)
                    loss_variance = None
                    if len(self.loss_window) > 1:
                        mean_loss = sum(self.loss_window) / len(self.loss_window)
                        loss_variance = sum((x - mean_loss) ** 2 for x in self.loss_window) / len(self.loss_window)
                        if len(self.loss_window) >= 20:
                            recent = self.loss_window[-10:]
                            earlier = self.loss_window[:10]
                            if earlier:
                                delta = (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))
                                if delta < -0.001:
                                    self.loss_trend = "improving"
                                elif delta > 0.001:
                                    self.loss_trend = "rising"
                                else:
                                    self.loss_trend = "stable"

                    # CRITICAL FIX #5: NaN detection
                    import math
                    if math.isnan(current_loss) or math.isinf(current_loss):
                        print(f"\n❌ CRITICAL: NaN/Inf loss detected at step {state.global_step}!")
                        print(f"   Loss value: {current_loss}")
                        print(f"   This indicates model corruption - training will be stopped")
                        print(f"   The model will need to be restored from last good checkpoint")
                        # Stop training immediately
                        try:
                            self.status_writer.mark_crashed(error="NaN/Inf loss detected", error_type="NaN")
                        except Exception:
                            pass
                        control.should_training_stop = True
                        return control

                    # Check for pause/stop signals (every N steps)
                    if self.controller and (state.global_step - self.last_control_check_step) >= self.control_check_interval:
                        self.last_control_check_step = state.global_step

                        # Check for stop signal
                        if self.controller.should_stop_after_batch():
                            print(f"\n🛑 STOP signal detected at step {state.global_step}")
                            print(f"   Stopping training gracefully after current batch...")
                            control.should_training_stop = True
                            return control

                        # Check for pause signal
                        if self.controller.should_pause_after_batch():
                            print(f"\n⏸️  PAUSE signal detected at step {state.global_step}")
                            print(f"   Pausing training gracefully after current batch...")
                            control.should_training_stop = True
                            return control

                    # Calculate batch-relative step (step within current file)
                    # state.global_step is cumulative, so subtract the starting point
                    batch_step = state.global_step - self.current_global_step

                    # Throughput tracking (steps/sec and rough tokens/sec)
                    step_time = current_time - self.prev_step_time if self.prev_step_time else None
                    tokens_per_sec = None
                    steps_per_sec = None
                    if step_time and step_time > 0:
                        steps_per_sec = 1.0 / step_time
                        tokens_per_step = self.avg_seq_len * self.effective_batch if self.avg_seq_len else None
                        tokens_per_sec = steps_per_sec * tokens_per_step if tokens_per_step else None
                        self.steps_per_sec_ema = steps_per_sec if self.steps_per_sec_ema is None else 0.1 * steps_per_sec + 0.9 * self.steps_per_sec_ema
                        if tokens_per_sec is not None:
                            self.tokens_per_sec_ema = tokens_per_sec if self.tokens_per_sec_ema is None else 0.1 * tokens_per_sec + 0.9 * self.tokens_per_sec_ema
                            if self.throughput_baseline is None and state.global_step > 20:
                                self.throughput_baseline = self.tokens_per_sec_ema

                    current_vram = None
                    if torch.cuda.is_available():
                        try:
                            current_vram = max(torch.cuda.memory_allocated(), torch.cuda.memory_reserved()) / (1024 ** 3)
                        except Exception:
                            current_vram = None
                    ratio = None
                    if current_vram is not None and self.total_vram:
                        ratio = current_vram / self.total_vram
                        self.last_vram_ratio = ratio
                        if ratio < 0.6:
                            self.low_vram_counter += 1
                        else:
                            self.low_vram_counter = 0
                    else:
                        self.low_vram_counter = 0
                    if steps_per_sec:
                        samples_per_sec = steps_per_sec * self.effective_batch
                        self.queue_velocity = {
                            "samples_per_sec": samples_per_sec,
                            "samples_per_hour": samples_per_sec * 3600,
                            "effective_batch": self.effective_batch,
                        }
                    if tokens_per_sec is not None and current_vram is not None:
                        self.vram_samples.append({
                            "step": state.global_step,
                            "tokens_per_sec": tokens_per_sec,
                            "vram_gb": round(current_vram, 3),
                            "penalty": None,
                        })
                        if len(self.vram_samples) > 120:
                            self.vram_samples.pop(0)

                    self.prev_step_time = current_time

                    # Lightweight micro-eval on tiny fixed set
                    val_loss = self.last_val_loss
                    if (
                        self.micro_eval_inputs is not None
                        and state.global_step > 0
                        and state.global_step % self.micro_eval_interval == 0
                        and state.global_step != self.last_micro_eval_step
                    ):
                        try:
                            self.model_ref.eval()
                            with torch.no_grad():
                                micro_inputs = {k: v.to(self.model_ref.device) for k, v in self.micro_eval_inputs.items()}
                                outputs = self.model_ref(**micro_inputs, labels=micro_inputs["input_ids"])
                                val_loss = outputs.loss.item()
                                self.last_val_loss = val_loss
                                self.last_micro_eval_step = state.global_step
                        except Exception as e:
                            print(f"Warning: Micro-eval failed at step {state.global_step}: {e}")
                        finally:
                            self.model_ref.train()

                    # Build simple alerts within the 10% overhead budget
                    alerts = []
                    alert_summary = None
                    if self.throughput_baseline and self.tokens_per_sec_ema:
                        if self.tokens_per_sec_ema < 0.6 * self.throughput_baseline:
                            alerts.append({"severity": "warn", "type": "throughput_drop", "detail": "Throughput <60% of baseline"})
                    if val_loss is not None and current_loss is not None:
                        gap = val_loss - current_loss
                        if gap > 0.3:
                            alerts.append({"severity": "warn", "type": "val_gap", "detail": f"val-train gap {gap:.3f}"})
                    if loss_variance is not None and self.loss_window:
                        mean_loss = sum(self.loss_window) / len(self.loss_window)
                        if mean_loss > 0 and loss_variance > (mean_loss ** 2) * 0.25:
                            alerts.append({"severity": "info", "type": "loss_variance", "detail": "High loss variance"})
                    if self.last_vram_ratio is not None and self.low_vram_counter >= 15:
                        severity = "warn" if self.last_vram_ratio < 0.5 else "info"
                        detail = f"VRAM usage at {self.last_vram_ratio*100:.1f}% capacity"
                        alerts.append({"severity": severity, "type": "low_vram_utilization", "detail": detail})
                    if alerts:
                        summary = {}
                        for a in alerts:
                            summary[a["severity"]] = summary.get(a["severity"], 0) + 1
                        alert_summary = summary

                    # Time-based status updates (every ~2 seconds)
                    if current_time - self.last_update_time >= self.update_interval:
                        penalty_stats = None
                        penalty_heatmap_payload = None
                        penalty_deltas: Dict[str, int] = {}
                        if self.enable_penalty_metrics and self.logits_processor:
                            penalty_stats = collect_penalty_stats(self.logits_processor)
                            if penalty_stats:
                                for stat in penalty_stats:
                                    label = stat.get("label", "penalty")
                                    hits = stat.get("hits", 0)
                                    previous = self.penalty_last_hits.get(label, 0)
                                    delta = max(0, hits - previous)
                                    self.penalty_last_hits[label] = hits
                                    if delta > 0:
                                        penalty_deltas[label] = delta
                                        self.penalty_totals[label] = self.penalty_totals.get(label, 0) + delta
                                        file_key = self.current_file or "unknown"
                                        file_stats = self.penalty_files.setdefault(file_key, {})
                                        file_stats[label] = file_stats.get(label, 0) + delta
                                        while len(self.penalty_files) > 8:
                                            self.penalty_files.popitem(last=False)
                                if self.penalty_files:
                                    penalty_heatmap_payload = {
                                        "totals": dict(self.penalty_totals),
                                        "per_file": {k: dict(v) for k, v in self.penalty_files.items()}
                                    }

                        if self.vram_samples:
                            if penalty_deltas:
                                self.vram_samples[-1]["penalty"] = penalty_deltas
                            else:
                                self.vram_samples[-1]["penalty"] = None

                        serialized_samples = []
                        for sample in self.vram_samples:
                            serialized_samples.append(sample)
                        penalty_stats_payload = penalty_stats

                        self.status_writer.update_progress(
                            step=state.global_step,
                            total_steps=self.total_steps,
                            epoch=int(current_epoch),
                            loss=current_loss,
                            lr=current_lr,
                            val_loss=self.last_val_loss,  # Use validation loss from on_evaluate callback
                            batch_step=batch_step,
                            batch_total_steps=self.batch_total_steps,
                            batch_number=self.batch_number,
                            batch_queue_size=self.batch_queue_size,
                            current_file=self.current_file,
                            tokens_per_sec=tokens_per_sec,
                            tokens_per_sec_avg=self.tokens_per_sec_ema,
                            tokens_per_sec_baseline=self.throughput_baseline,
                            loss_variance=loss_variance,
                            loss_trend=self.loss_trend,
                            active_alerts=alerts if alerts else None,
                            alert_summary=alert_summary,
                            throughput_vram_samples=serialized_samples,
                            queue_velocity=self.queue_velocity,
                            logit_penalty_stats=penalty_stats_payload,
                            penalty_heatmap=penalty_heatmap_payload,
                        )
                        self.last_update_time = current_time

                    # Inference updates (periodic) - show a live example with golden/model output
                    if (
                        self.raw_train_examples
                        and state.global_step > 0
                        and (state.global_step - self.last_inference_step) >= self.inference_interval_steps
                        and state.global_step != self.last_inference_step
                    ):
                        try:
                            print(f"[InferencePreview] step={state.global_step} interval={self.inference_interval_steps} last={self.last_inference_step}")
                            # Get the raw example from ORIGINAL dataset (before tokenization)
                            if self.raw_train_examples and len(self.raw_train_examples) > 0:
                                # Get example based on current step (with wraparound)
                                dataset_idx = state.global_step % len(self.raw_train_examples)
                                current_example = self.raw_train_examples[dataset_idx]

                                # Extract messages
                                if 'messages' in current_example:
                                    # Extract from messages
                                    messages = current_example['messages']
                                    system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
                                    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
                                    golden_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)

                                    if user_msg and golden_msg:
                                        # Run inference on this exact example
                                        self.model_ref.eval()
                                        with torch.no_grad():
                                            # Format prompt
                                            prompt_messages = [{"role": "user", "content": user_msg}]
                                            text = self.tokenizer.apply_chat_template(
                                                prompt_messages,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )

                                            # Generate
                                            inputs = self.tokenizer(text, return_tensors="pt").to(self.model_ref.device)
                                            reset_processor_states(self.logits_processor)
                                            outputs = self.model_ref.generate(
                                                **inputs,
                                                max_new_tokens=2048,  # Full output for reasoning tasks
                                                temperature=0.1,
                                                do_sample=False,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                logits_processor=self.logits_processor,
                                                min_new_tokens=1
                                            )

                                            # Decode model output
                                            model_output = self.tokenizer.decode(
                                                outputs[0][inputs['input_ids'].shape[1]:],
                                                skip_special_tokens=True
                                            ).strip()

                                        # Calculate loss on this specific example
                                        example_loss = None
                                        try:
                                            # Tokenize the full conversation (user + golden assistant)
                                            full_messages = [
                                                {"role": "user", "content": user_msg},
                                                {"role": "assistant", "content": golden_msg}
                                            ]
                                            full_text = self.tokenizer.apply_chat_template(
                                                full_messages,
                                                tokenize=False,
                                                add_generation_prompt=False
                                            )

                                            # Tokenize
                                            full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model_ref.device)

                                            # Get model logits
                                            with torch.no_grad():
                                                outputs = self.model_ref(**full_inputs, labels=full_inputs['input_ids'])
                                                example_loss = outputs.loss.item()
                                        except Exception as e:
                                            print(f"Warning: Could not calculate loss for this example: {e}")

                                        self.model_ref.train()

                                        # Clear GPU cache after inference to prevent OOM
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()

                                        # Estimate token lengths for metadata/analysis
                                        golden_token_len = None
                                        model_token_len = None
                                        try:
                                            golden_token_len = len(self.tokenizer.encode(golden_msg, add_special_tokens=False))
                                        except Exception:
                                            pass
                                        try:
                                            model_token_len = len(self.tokenizer.encode(model_output, add_special_tokens=False))
                                        except Exception:
                                            pass

                                        # Check if match
                                        matches = golden_msg.strip() == model_output.strip()

                                        # Update pattern tracker for heatmap coverage
                                        pattern_matrix = None
                                        pattern_id = None
                                        bin_name = None
                                        if self.pattern_tracker:
                                            try:
                                                response_tokens = model_token_len if model_token_len is not None else 0
                                                pattern_id, bin_name = self.pattern_tracker.classify(
                                                    user_msg or "",
                                                    response_tokens
                                                )
                                                self.pattern_tracker.record(pattern_id, bin_name, matches)
                                                pattern_matrix = self.pattern_tracker.get_matrix()
                                            except Exception as e:
                                                print(f"Warning: Pattern tracker update failed: {e}")
                                        pattern_metadata = None
                                        if pattern_id:
                                            pattern_metadata = {
                                                "pattern_id": pattern_id,
                                                "length_bin": bin_name,
                                                "timestamp": datetime.now().isoformat(),
                                                "loss": example_loss,
                                            }

                                        layer_summary = None
                                        if self.layer_monitor:
                                            try:
                                                layer_summary = self.layer_monitor.snapshot()
                                            except Exception as e:
                                                print(f"Warning: Layer monitor snapshot failed: {e}")

                                        # Display in terminal
                                        print("\n" + "=" * 80)
                                        print(f"🔍 CURRENT TRAINING EXAMPLE - Step {state.global_step:,}")
                                        print("=" * 80)
                                        print(f"📝 PROMPT:\n{user_msg[:500]}...")
                                        print(f"\n✅ GOLDEN:\n{golden_msg[:200]}...")
                                        print(f"\n🤖 MODEL:\n{model_output[:200]}...")
                                        status = "✅ MATCH" if matches else "❌ NO MATCH"
                                        print(f"\n{status}")
                                        if example_loss is not None:
                                            print(f"📉 LOSS ON THIS EXAMPLE: {example_loss:.4f}")
                                        print("=" * 80 + "\n")

                                        # Calculate batch-relative step
                                        batch_step = state.global_step - self.current_global_step

                                        # Update status JSON (use example-specific loss if available)
                                        display_loss = example_loss if example_loss is not None else current_loss
                                        self.status_writer.update_inference(
                                            step=state.global_step,
                                            total_steps=self.total_steps,
                                            epoch=int(current_epoch),
                                            loss=display_loss,
                                            lr=current_lr,
                                            prompt=user_msg,
                                            golden=golden_msg,
                                            model_output=model_output,
                                            matches=matches,
                                            system_prompt=system_msg,
                                            batch_step=batch_step,
                                            batch_total_steps=self.batch_total_steps,
                                            batch_number=self.batch_number,
                                            batch_queue_size=self.batch_queue_size,
                                            current_file=self.current_file,
                                            golden_output_length=golden_token_len,
                                            model_output_length=model_token_len,
                                            pattern_heatmap=pattern_matrix,
                                            layer_activity_summary=layer_summary,
                                            pattern_metadata=pattern_metadata
                                        )
                                        self.last_update_time = current_time
                                        self.last_inference_step = state.global_step
                        except Exception as e:
                            print(f"Warning: Could not display current training example at step {state.global_step}: {e}")
                            import traceback
                            traceback.print_exc()

                    # Evolution tracking (at special steps only)
                    # DISABLED: Takes 60-300s per snapshot (too slow)
                    if False and self.evolution_tracker and self.raw_train_examples:
                        try:
                            snapshot_id = self.evolution_tracker.capture_snapshot(
                                model=self.model_ref,
                                tokenizer=self.tokenizer,
                                examples=self.raw_train_examples,
                                current_step=state.global_step,
                                model_version="training",
                                max_examples=100  # Limit for performance
                            )
                            if snapshot_id:
                                print(f"📊 Evolution snapshot saved: {snapshot_id}")
                        except Exception as e:
                            print(f"Warning: Evolution snapshot failed at step {state.global_step}: {e}")

                def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                    """Capture validation loss after evaluation runs."""
                    if metrics and 'eval_loss' in metrics:
                        self.last_val_loss = metrics['eval_loss']
                        print(f"\n📊 Validation Loss: {self.last_val_loss:.4f}")

                        # Calculate train/val gap if we have recent training loss
                        if state.log_history:
                            recent_train_loss = state.log_history[-1].get('loss', None)
                            if recent_train_loss:
                                gap = self.last_val_loss - recent_train_loss
                                print(f"   Train Loss: {recent_train_loss:.4f}")
                                print(f"   Val-Train Gap: {gap:+.4f}")
                                if gap > 0.5:
                                    print(f"   ⚠️  Large gap detected - possible overfitting!")

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
                    avg_seq_len=self.avg_seq_len,
                    effective_batch=effective_batch,
                    micro_eval_inputs=self.tokenized_fixed_val_small,
                    micro_eval_interval=max(500, self.args.eval_steps),
                    logits_processor=self.logits_processor,
                    layer_monitor=self.layer_monitor,
                )]
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

            # Run a quick initial preview to seed status/logs
            try:
                self.run_initial_preview(current_global_step=current_global_step, total_steps=max_steps)
            except Exception as e:
                print(f"Warning: Initial preview failed: {e}")

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

            # Train with checkpoint resumption if available
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save final adapter model (for inference/deployment)
            # NOTE: Checkpoints are auto-managed by Trainer via save_total_limit
            # This save_model() is just for having a clean adapter at root for inference
            print(f"💾 Saving final adapter model to: {self.args.output_dir}")
            trainer.save_model(self.args.output_dir)
            self.tokenizer.save_pretrained(self.args.output_dir)

            # Trainer has automatically managed checkpoints during training:
            # - Saved checkpoints every save_steps (100 steps)
            # - Kept only last save_total_limit (3) checkpoints
            # - Latest checkpoint contains full state for resumption
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
                "effective_batch": (self.args.batch_size * self.args.gradient_accumulation)
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

    def run_initial_preview(self, current_global_step: int, total_steps: int):
        """Run a lightweight preview on the first example to seed status/logs."""
        if not self.raw_train_examples:
            print("Initial preview skipped: no raw_train_examples.")
            return
        try:
            example = self.raw_train_examples[0]
            messages = example.get('messages', [])
            system_msg = next((m['content'] for m in messages if m.get('role') == 'system'), None)
            user_msg = next((m['content'] for m in messages if m.get('role') == 'user'), None)
            golden_msg = next((m['content'] for m in messages if m.get('role') == 'assistant'), None)
            if not user_msg or not golden_msg:
                print("Initial preview skipped: missing user/golden.")
                return
            self.model.eval()
            with torch.no_grad():
                prompt_messages = [{"role": "user", "content": user_msg}]
                text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                reset_processor_states(self.logits_processor)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=self.logits_processor,
                    min_new_tokens=1
                )
                model_output = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
            self.model.train()

            # Clear GPU cache after initial inference to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_step = 0
            layer_summary = None
            if self.layer_monitor:
                try:
                    layer_summary = self.layer_monitor.snapshot()
                except Exception as exc:
                    print(f"Warning: Layer monitor snapshot failed (initial preview): {exc}")
            self.status_writer.update_inference(
                step=current_global_step,
                total_steps=total_steps,
                epoch=0,
                loss=0.0,
                lr=self.args.learning_rate,
                prompt=user_msg,
                golden=golden_msg,
                model_output=model_output,
                matches=(golden_msg.strip() == model_output.strip()),
                system_prompt=system_msg,
                batch_step=batch_step,
                batch_total_steps=total_steps,
                batch_number=getattr(self.args, 'batch_number', None),
                batch_queue_size=getattr(self.args, 'batch_queue_size', None),
                current_file=getattr(self.args, 'current_file', None),
                golden_output_length=len(golden_msg),
                model_output_length=len(model_output),
                layer_activity_summary=layer_summary
            )
            self.last_inference_step = current_global_step
            print("[InitialPreview] Completed initial inference preview.")
        except Exception as e:
            print(f"Initial preview error: {e}")

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
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")

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
