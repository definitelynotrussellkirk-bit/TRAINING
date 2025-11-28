#!/usr/bin/env python3
"""
Training Engine - Full Implementation

Clean API-style trainer that orchestrates all training operations.

Usage:
    from trainer.core import TrainerEngine
    from trainer.config import create_default_config
    from trainer.monitoring import TrainingStatusWriter

    config = create_default_config(...)
    status_writer = TrainingStatusWriter("status/training_status.json")
    engine = TrainerEngine(status_writer)
    result = engine.run_job(config)

Architecture:
    - Single entry point: run_job(config)
    - Delegates to profiles for data transformation
    - Uses ConfigLoader for all config
    - Returns TrainingResult with metrics

Features:
    - Qwen3VL model fallback with vision tower freezing
    - Flash Attention 2 detection and selection
    - Response-only masking (DataCollatorForCompletionOnly)
    - Packing support (trl.pack_dataset)
    - Muon optimizer support
    - Checkpoint resumption
    - Masking validation
    - Callback injection for monitoring
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Callable
from pathlib import Path
from datetime import datetime
import json
import random
import time
import sys
import os
import logging
import gc

# Set up module logger
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from datasets import Dataset

from trainer.config.schema import TrainerConfig
from trainer.config.loader import ConfigLoader
from trainer.monitoring.status_writer import TrainingStatusWriter
from trainer.profiles import get_profile

# Optional: Qwen3VL support
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None

# Optional: Flash Attention 2
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Optional: Packing from TRL
try:
    from trl import ConstantLengthDataset
    TRL_PACKING_AVAILABLE = True
except ImportError:
    TRL_PACKING_AVAILABLE = False

# Optional: Muon optimizer
try:
    from trainer.optimizers import create_optimizer as create_custom_optimizer
    from trainer.optimizers import get_param_group_summary
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    create_custom_optimizer = None
    get_param_group_summary = None

# Optional: LiveMonitorCallback for full monitoring
try:
    from trainer.monitoring.callbacks import LiveMonitorCallback
    LIVE_MONITOR_AVAILABLE = True
except ImportError:
    LIVE_MONITOR_AVAILABLE = False
    LiveMonitorCallback = None

# Optional: LayerMonitor for layer statistics
try:
    from monitoring.servers.layer_monitor import LayerMonitor
    LAYER_MONITOR_AVAILABLE = True
except ImportError:
    LAYER_MONITOR_AVAILABLE = False
    LayerMonitor = None


@dataclass
class TrainingResult:
    """
    Result of a training job execution.

    Contains all relevant information about training outcome including
    success status, performance metrics, checkpoint locations, and error
    information if training failed.

    Used as the return value from TrainerEngine.run_job() to provide
    complete feedback about training execution.
    """
    success: bool  # True if training completed without errors
    global_step: int  # Total number of training steps executed
    runtime_sec: float  # Wall-clock time in seconds (end - start)
    last_checkpoint_path: Optional[str]  # Path to final checkpoint or None if failed
    final_loss: float  # Last recorded training loss value
    summary: Dict[str, Any]  # Additional metrics (config, profile, precision, etc.)
    error_message: Optional[str] = None  # Error description if success=False

    @classmethod
    def from_error(cls, error: str) -> 'TrainingResult':
        """
        Create a TrainingResult representing a failed training run.

        Args:
            error: Error message describing what went wrong

        Returns:
            TrainingResult with success=False and error_message set
        """
        return cls(
            success=False,
            global_step=0,
            runtime_sec=0.0,
            last_checkpoint_path=None,
            final_loss=0.0,
            summary={},
            error_message=error
        )


@dataclass
class MonitorContext:
    """
    Context for training monitors passed from UltimateTrainer.

    Contains all monitoring components that can be injected into
    the training loop via callbacks. This context provides everything
    LiveMonitorCallback needs to enable full monitoring features:
    - Progress tracking (steps, loss, learning rate)
    - Validation loss tracking (micro-eval)
    - Throughput monitoring (tokens/sec, VRAM usage)
    - Pattern tracking (heatmaps)
    - Layer monitoring
    - Control signal handling (pause/stop)
    - Smart alerts
    - Checkpoint ledger recording
    - Remote evaluation sync
    """
    # Core monitors
    live_monitor: Any = None  # LiveInferenceMonitor
    evolution_tracker: Any = None  # EvolutionTracker
    layer_monitor: Any = None  # LayerMonitor (can be created by engine if None)
    controller: Any = None  # TrainingController (pause/stop)

    # Training data context
    raw_train_examples: List[Dict] = field(default_factory=list)
    logits_processor: Any = None  # LogitsProcessorList for generation

    # Remote evaluation
    remote_eval_config: Dict[str, Any] = field(default_factory=dict)

    # Batch/file context (for progress display)
    current_file: Optional[str] = None
    batch_number: Optional[int] = None
    batch_queue_size: Optional[int] = None

    # Validation datasets
    fixed_val_dataset: Any = None  # Fixed validation set for generalization metrics
    micro_eval_inputs: Any = None  # Tokenized micro eval inputs

    # Monitoring intervals
    micro_eval_interval: int = 500  # How often to run micro-eval

    # Status writer (optional - engine has its own, but can be overridden)
    status_writer: Any = None


class TrainerEngine:
    """
    Core training engine with stable public API.

    Provides a clean, testable interface for training operations:
    - Single entry point: run_job(config)
    - All configuration via TrainerConfig (no scattered reads)
    - Profile-based data transformation (emoji_think, regime3, etc.)
    - Structured result object with metrics and errors
    - Exception-safe (returns errors in TrainingResult)

    Enhanced Features:
        - Qwen3VL model loading with vision tower freezing
        - Flash Attention 2 detection and auto-selection
        - Response-only masking via DataCollatorForCompletionOnly
        - Dataset packing via trl.pack_dataset
        - Muon optimizer support (orthogonalized momentum)
        - Checkpoint resumption with scheduler management
        - Masking validation to catch misconfigured collators
        - Callback injection for external monitoring

    Attributes:
        status_writer: TrainingStatusWriter for writing status JSON
        model: HuggingFace model instance (None until loaded)
        tokenizer: HuggingFace tokenizer instance (None until loaded)
        profile: DataProfile instance for transformations (None until loaded)
        is_vision_model: True if model is Qwen3VL (needs special handling)
        avg_seq_len: Average sequence length (for throughput estimation)
    """

    def __init__(self, status_writer: TrainingStatusWriter):
        """
        Initialize TrainerEngine with monitoring.

        Args:
            status_writer: Status writer for live UI updates.
                Writes training_status.json with current step, loss, etc.

        Side Effects:
            - Initializes instance attributes to None
            - No file I/O or GPU operations occur until run_job() called
        """
        self.status_writer = status_writer
        self.model = None
        self.tokenizer = None
        self.profile = None
        self.is_vision_model = False
        self.avg_seq_len = 0.0

    def run_job(
        self,
        config: TrainerConfig,
        config_dict: Optional[Dict[str, Any]] = None,
        monitors: Optional[MonitorContext] = None,
        callbacks: Optional[List[TrainerCallback]] = None
    ) -> TrainingResult:
        """
        Execute a complete training job with the given configuration.

        This is the ONLY public method. All training operations go through here.
        The method is exception-safe: all errors are caught and returned in the
        TrainingResult object rather than being raised.

        Args:
            config: TrainerConfig containing all training parameters
            config_dict: Optional raw config dict (for optimizer settings, etc.)
            monitors: Optional MonitorContext with external monitors
            callbacks: Optional list of additional TrainerCallback instances

        Returns:
            TrainingResult containing success status, metrics, and error info

        Raises:
            No exceptions are raised. All errors are caught and returned
            in the TrainingResult.error_message field.
        """
        print("\n" + "=" * 80)
        print("TRAINER ENGINE - ENHANCED IMPLEMENTATION")
        print("=" * 80)
        print(f"Profile: {config.profile.name}")
        print(f"Model: {config.model.model_path}")
        print(f"Dataset: {config.data.dataset_path}")
        print(f"Output: {config.output.output_dir}")
        print(f"Precision: {config.hyperparams.fp_precision}")
        print(f"Batch size: {config.hyperparams.batch_size}")
        print("=" * 80 + "\n")

        start_time = time.time()
        config_dict = config_dict or {}

        try:
            # 1. Validate config
            print("Step 1: Validating configuration")
            ConfigLoader.validate_locked_config(config)
            print("   Config validated\n")

            # 2. Load profile
            print(f"Step 2: Loading profile '{config.profile.name}'")
            self.profile = get_profile(config.profile.name)
            print(f"   Profile loaded: {self.profile.__class__.__name__}\n")

            # 3. Load model & tokenizer (enhanced)
            print("Step 3: Loading model and tokenizer")
            self.model, self.tokenizer = self._load_model_and_tokenizer(config)
            print("   Model and tokenizer loaded\n")

            # 4. Prepare datasets (with packing)
            print("Step 4: Preparing datasets")
            train_dataset, val_dataset = self._prepare_dataset(config)
            print("   Datasets prepared\n")

            # 5. Find checkpoint for resumption
            print("Step 5: Checking for checkpoint resumption")
            resume_checkpoint, current_global_step = self._find_resume_checkpoint(
                config.output.output_dir
            )
            if resume_checkpoint:
                print(f"   Will resume from: {resume_checkpoint}")
                print(f"   Current global step: {current_global_step}")
            else:
                print("   Starting fresh (no checkpoint found)")
            print()

            # 6. Create trainer (with callbacks, optimizer, collator)
            print("Step 6: Creating HuggingFace Trainer")
            trainer, data_collator = self._create_trainer(
                config=config,
                config_dict=config_dict,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                monitors=monitors,
                callbacks=callbacks,
                current_global_step=current_global_step,
            )
            print("   Trainer created\n")

            # 7. Validate masking
            print("Step 7: Validating response masking")
            masking_stats = self._validate_masking(train_dataset, data_collator)
            print(f"   Masked (instruction): {masking_stats['masked_pct']:.1f}%")
            print(f"   Trained (response): {masking_stats['trained_pct']:.1f}%")
            print("   Masking validation passed\n")

            # 8. Execute training
            print("Step 8: Executing training")
            print("=" * 80)
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
            print("=" * 80)
            print("   Training complete\n")

            # 9. Save final checkpoint
            print("Step 9: Saving final checkpoint")
            final_checkpoint = Path(config.output.output_dir) / "final_model"
            trainer.save_model(str(final_checkpoint))
            self.tokenizer.save_pretrained(str(final_checkpoint))
            print(f"   Saved to {final_checkpoint}\n")

            # 10. Return result
            runtime_sec = time.time() - start_time

            return TrainingResult(
                success=True,
                global_step=train_result.global_step,
                runtime_sec=runtime_sec,
                last_checkpoint_path=str(final_checkpoint),
                final_loss=train_result.metrics.get('train_loss', 0.0),
                summary={
                    **train_result.metrics,
                    'config': config.to_dict(),
                    'profile': config.profile.name,
                    'precision': config.hyperparams.fp_precision,
                    'masking_stats': masking_stats,
                }
            )

        except Exception as e:
            print(f"\n Training failed: {e}")
            import traceback
            traceback.print_exc()

            return TrainingResult.from_error(str(e))

    # ========================================================================
    # Model Loading (Enhanced)
    # ========================================================================

    def _load_model_and_tokenizer(
        self,
        config: TrainerConfig
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Load model and tokenizer with enhanced features.

        Enhancements over basic loading:
        - Flash Attention 2 detection and auto-selection
        - Qwen3VL fallback with vision tower freezing
        - Chat template override for emoji_think/regime3 profiles
        - Gradient checkpointing support

        Args:
            config: TrainerConfig with model settings

        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = config.model.model_path
        print(f"   Loading from: {model_path}")

        # Detect attention implementation
        attn_impl = "sdpa"
        if FLASH_ATTN_AVAILABLE:
            attn_impl = "flash_attention_2"
            print("   Flash Attention 2 detected")
        else:
            print("   Using SDPA attention (flash_attention_2 not installed)")

        # Setup quantization (if requested)
        quantization_config = None
        if config.model.load_in_4bit:
            print("   Enabling 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Get precision
        torch_dtype = self._get_torch_dtype(config.hyperparams.fp_precision)
        print(f"   Precision: {config.hyperparams.fp_precision} ({torch_dtype})")

        # Model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        # Try Qwen3VL first (for VL models)
        self.is_vision_model = False
        model = None
        tokenizer = None

        if QWEN3VL_AVAILABLE:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, **model_kwargs
                )
                print("   Loaded as Qwen3VLForConditionalGeneration")
                self.is_vision_model = True

                # Freeze vision and video towers for text-only training
                print("   Freezing vision/video towers for text-only training...")
                frozen_params = 0
                for n, p in model.named_parameters():
                    if any(k in n for k in ["vision_model", "video_model", "visual"]):
                        p.requires_grad = False
                        frozen_params += 1
                print(f"   Froze {frozen_params} vision/video parameters")

                # Use AutoProcessor for Qwen3VL
                processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
                tokenizer = processor.tokenizer
                print("   Loaded AutoProcessor")

            except Exception as e:
                print(f"   Qwen3VL failed ({str(e)[:50]}...), trying AutoModelForCausalLM...")
                model = None

        # Fallback to standard CausalLM
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            print("   Loaded as AutoModelForCausalLM")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   Set pad_token = eos_token")

        # Apply chat template override (fixes Qwen3 <think> injection)
        try:
            from core.chat_templates import apply_chat_template_override
            profile_name = config.profile.name if config.profile else None
            apply_chat_template_override(tokenizer, profile_name=profile_name, verbose=True)
        except ImportError:
            print("   Chat template override not available")

        # Disable KV cache for training
        model.config.use_cache = False
        print("   Disabled KV cache (use_cache=False)")

        # Enable gradient checkpointing if configured
        use_grad_ckpt = getattr(config.hyperparams, 'use_gradient_checkpointing', True)
        if use_grad_ckpt and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("   Enabled gradient checkpointing")
        else:
            print("   Gradient checkpointing disabled")

        return model, tokenizer

    # ========================================================================
    # Dataset Preparation (Enhanced with Packing)
    # ========================================================================

    def _prepare_dataset(
        self,
        config: TrainerConfig
    ) -> Tuple[Dataset, Dataset]:
        """
        Load, transform, tokenize, and optionally pack datasets.

        Enhancements:
        - Dataset packing via trl.pack_dataset for efficiency
        - Proper handling of packed sequence metadata
        - Memory-efficient processing with garbage collection
        """
        # Load raw data
        dataset_path = Path(config.data.dataset_path)
        print(f"   Loading from: {dataset_path}")

        examples = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        print(f"   Total examples: {len(examples):,}")

        # Shuffle
        if config.data.shuffle:
            random.seed(config.data.seed)
            random.shuffle(examples)
            print(f"   Shuffled (seed={config.data.seed})")

        # Split train/val
        val_size = min(
            config.monitoring.validation_max_samples,
            int(len(examples) * config.monitoring.validation_split)
        )
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        print(f"   Train: {len(train_examples):,}")
        print(f"   Val: {len(val_examples):,}")

        # Build system prompt
        system_prompt = self._build_system_prompt(config)
        print(f"   System prompt: \"{system_prompt[:50]}...\"")

        # Transform using profile
        print(f"   Applying profile transformations...")
        train_examples = [
            self.profile.transform_example(ex, idx, system_prompt)
            for idx, ex in enumerate(train_examples)
        ]
        val_examples = [
            self.profile.transform_example(ex, idx, system_prompt)
            for idx, ex in enumerate(val_examples)
        ]
        print("   Profile transformations applied")

        # Format for tokenization
        def format_messages(ex):
            """Format messages for chat template."""
            messages = ex['messages']
            formatted_messages = []

            for msg in messages:
                content = msg.get('content', '')
                if not isinstance(content, str):
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

        # Format all examples
        train_data = [format_messages(ex) for ex in train_examples]
        val_data = [format_messages(ex) for ex in val_examples]

        # Create HF datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        # Tokenize
        max_length = config.hyperparams.max_length

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False
            )

        print("   Tokenizing...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=["text"],
            num_proc=None,  # Disable multiprocessing (CUDA fork issues)
            load_from_cache_file=False,
        )
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=["text"],
            num_proc=None,
            load_from_cache_file=False,
        )
        print("   Tokenization complete")

        # Calculate average sequence length
        sample_count = min(100, len(train_dataset))
        if sample_count > 0:
            total_tokens = sum(
                len(train_dataset[i]["input_ids"])
                for i in range(sample_count)
            )
            self.avg_seq_len = total_tokens / sample_count
            print(f"   Avg seq length: {self.avg_seq_len:.1f} tokens")

        # Pack dataset for efficiency (if enabled)
        enable_packing = os.environ.get("ENABLE_PACKING", "1").lower() in ("1", "true", "yes")
        if enable_packing:
            train_dataset = self._pack_dataset(train_dataset, max_length)

        # Memory cleanup
        del train_data, val_data, examples
        gc.collect()

        return train_dataset, val_dataset

    def _pack_dataset(self, dataset: Dataset, max_length: int) -> Dataset:
        """
        Pack dataset for more efficient training.

        Uses trl.pack_dataset to combine multiple short sequences
        into full-length blocks, reducing padding waste.
        """
        try:
            from trl import pack_dataset
            print(f"   Packing dataset (max_length={max_length})...")
            print(f"      Before packing: {len(dataset)} examples")

            packed = pack_dataset(
                dataset,
                seq_length=max_length,
                strategy="bfd"  # Best Fit Decreasing
            )

            print(f"      After packing: {len(packed)} packed sequences")

            # Remove seq_lengths metadata (collator can't handle it)
            if 'seq_lengths' in packed.column_names:
                packed = packed.remove_columns(['seq_lengths'])
                print("      Removed seq_lengths metadata")

            print("      Packing enabled")
            return packed

        except ImportError:
            print("      trl not available, skipping packing")
            return dataset
        except Exception as e:
            print(f"      Packing failed ({e}), continuing without packing")
            return dataset

    # ========================================================================
    # Checkpoint Resumption
    # ========================================================================

    def _find_resume_checkpoint(
        self,
        output_dir: str
    ) -> Tuple[Optional[str], int]:
        """
        Find latest checkpoint for resumption.

        Also deletes old scheduler.pt to force fresh constant LR scheduler.

        Args:
            output_dir: Directory containing checkpoints

        Returns:
            Tuple of (checkpoint_path, global_step) or (None, 0)
        """
        checkpoint_dir = Path(output_dir)
        if not checkpoint_dir.exists():
            return None, 0

        candidates = []
        for cp in checkpoint_dir.glob("checkpoint-*"):
            parts = cp.name.split("-", 1)
            if len(parts) != 2:
                continue
            try:
                step = int(parts[1].split("-")[0])  # Handle checkpoint-123-date-time format
            except ValueError:
                continue

            trainer_state = cp / "trainer_state.json"
            if trainer_state.exists():
                candidates.append((step, cp))

        if not candidates:
            return None, 0

        # Get latest checkpoint
        latest_step, latest_checkpoint = max(candidates, key=lambda x: x[0])

        # Read actual global_step from trainer_state
        trainer_state_file = latest_checkpoint / "trainer_state.json"
        with open(trainer_state_file) as f:
            state = json.load(f)
            current_global_step = state.get('global_step', latest_step)

        # Delete old scheduler to force fresh constant LR
        old_scheduler = latest_checkpoint / "scheduler.pt"
        if old_scheduler.exists():
            old_scheduler.unlink()
            print(f"   Deleted old scheduler.pt to force fresh constant LR")

        return str(latest_checkpoint), current_global_step

    # ========================================================================
    # Trainer Creation (Enhanced)
    # ========================================================================

    def _create_trainer(
        self,
        config: TrainerConfig,
        config_dict: Dict[str, Any],
        train_dataset: Dataset,
        val_dataset: Dataset,
        monitors: Optional[MonitorContext],
        callbacks: Optional[List[TrainerCallback]],
        current_global_step: int,
    ) -> Tuple[Trainer, Any]:
        """
        Create HuggingFace Trainer with enhanced features.

        Enhancements:
        - Response-only masking via DataCollatorForCompletionOnly
        - Muon optimizer support
        - Callback injection from monitors
        - Custom optimizer/scheduler handling
        """
        # Calculate max_steps
        effective_batch = (
            config.hyperparams.batch_size *
            config.hyperparams.gradient_accumulation
        )
        steps_for_this_file = len(train_dataset) // effective_batch
        if len(train_dataset) % effective_batch != 0:
            steps_for_this_file += 1

        max_steps = current_global_step + steps_for_this_file
        print(f"   Steps for this dataset: {steps_for_this_file}")
        print(f"   Max steps (cumulative): {max_steps}")

        # Training arguments
        training_args = self._build_training_arguments(config, max_steps)
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Learning rate: {training_args.learning_rate}")

        # Data collator - response-only masking
        data_collator = self._create_data_collator()
        print("   Using response-only collator (instruction masked)")

        # Create optimizer (Muon or default AdamW)
        custom_optimizer, custom_scheduler = self._create_optimizer(
            config_dict, max_steps, config.hyperparams.warmup_steps
        )

        # Build callbacks list
        all_callbacks = list(callbacks or [])

        # Wire LiveMonitorCallback from MonitorContext (engine-first refactor)
        if LIVE_MONITOR_AVAILABLE and monitors and monitors.live_monitor:
            # Use provided status_writer or fall back to engine's
            status_writer = monitors.status_writer or self.status_writer

            # Create LayerMonitor if not provided but available
            layer_monitor = monitors.layer_monitor
            if layer_monitor is None and LAYER_MONITOR_AVAILABLE:
                try:
                    layer_monitor = LayerMonitor(self.model)
                    print("   Created LayerMonitor for layer statistics")
                except Exception as e:
                    print(f"   LayerMonitor disabled: {e}")
                    layer_monitor = None

            # Create LiveMonitorCallback with all monitoring features
            monitor_cb = LiveMonitorCallback(
                monitor=monitors.live_monitor,
                status_writer=status_writer,
                eval_steps=config.hyperparams.eval_steps,
                total_steps=max_steps,
                raw_train_examples=monitors.raw_train_examples,
                tokenizer=self.tokenizer,
                model=self.model,
                batch_total_steps=steps_for_this_file,
                current_global_step=current_global_step,
                evolution_tracker=monitors.evolution_tracker,
                current_file=monitors.current_file,
                batch_number=monitors.batch_number,
                batch_queue_size=monitors.batch_queue_size,
                controller=monitors.controller,
                fixed_val_dataset=monitors.fixed_val_dataset,
                avg_seq_len=self.avg_seq_len,
                effective_batch=effective_batch,
                micro_eval_inputs=monitors.micro_eval_inputs,
                micro_eval_interval=monitors.micro_eval_interval,
                logits_processor=monitors.logits_processor,
                layer_monitor=layer_monitor,
                remote_eval_config=monitors.remote_eval_config,
            )
            all_callbacks.append(monitor_cb)
            print("   LiveMonitorCallback enabled (full monitoring)")
        elif monitors and monitors.live_monitor:
            print("   Warning: LiveMonitorCallback not available, monitoring disabled")

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=all_callbacks if all_callbacks else None,
            optimizers=(custom_optimizer, custom_scheduler) if custom_optimizer else (None, None),
        )

        return trainer, data_collator

    def _create_data_collator(self):
        """
        Create data collator for response-only training.

        Uses DataCollatorForCompletionOnly which masks instruction tokens,
        training only on assistant responses.
        """
        try:
            from core.custom_collator import DataCollatorForCompletionOnly
            return DataCollatorForCompletionOnly(
                tokenizer=self.tokenizer,
                response_template="<|im_start|>assistant\n"
            )
        except ImportError:
            # Fallback to standard collator
            print("   Warning: DataCollatorForCompletionOnly not available")
            from transformers import DataCollatorForLanguageModeling
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

    def _create_optimizer(
        self,
        config_dict: Dict[str, Any],
        num_training_steps: int,
        num_warmup_steps: int
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Create optimizer (Muon or AdamW) based on config.

        Args:
            config_dict: Raw config dict with optimizer settings
            num_training_steps: Total training steps
            num_warmup_steps: Warmup steps

        Returns:
            Tuple of (optimizer, scheduler) or (None, None) to use defaults
        """
        if not MUON_AVAILABLE:
            return None, None

        opt_config = config_dict.get("optimizer", {})
        optimizer_type = opt_config.get("type", "adamw") if isinstance(opt_config, dict) else "adamw"

        if optimizer_type != "muon":
            return None, None

        print("\n" + "=" * 60)
        print("MUON OPTIMIZER - Orthogonalized Momentum")
        print("=" * 60)

        # Show parameter grouping summary
        summary = get_param_group_summary(self.model)
        total = summary.muon_params + summary.adam_params
        print(f"  Hidden weights (Muon): {summary.muon_params:,} params ({100*summary.muon_params/total:.1f}%)")
        print(f"  Other (AdamW):         {summary.adam_params:,} params ({100*summary.adam_params/total:.1f}%)")

        # Create optimizer
        optimizer, scheduler, opt_info = create_custom_optimizer(
            self.model,
            config_dict,
            optimizer_type="muon",
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        print(f"  Hidden LR: {opt_info.hidden_lr}")
        print(f"  Aux LR:    {opt_info.aux_lr}")
        print("=" * 60 + "\n")

        return optimizer, scheduler

    # ========================================================================
    # Masking Validation
    # ========================================================================

    def _validate_masking(
        self,
        dataset: Dataset,
        collator,
        sample_count: int = 5
    ) -> Dict[str, float]:
        """
        Validate that masking is working correctly.

        Checks that a reasonable percentage of tokens are masked (instruction)
        vs trained (response). Raises error if masking looks wrong.

        Args:
            dataset: Tokenized dataset
            collator: Data collator with masking
            sample_count: Number of samples to check

        Returns:
            Dict with masked_pct and trained_pct

        Raises:
            ValueError: If masking percentage is suspiciously low (<25%)
        """
        if len(dataset) == 0:
            return {"masked_pct": 0.0, "trained_pct": 0.0}

        samples = [dataset[i] for i in range(min(sample_count, len(dataset)))]
        batch = collator(samples)

        masked_count = (batch['labels'] == -100).sum().item()
        total_count = batch['labels'].numel()
        trained_count = total_count - masked_count

        masked_pct = 100 * masked_count / total_count
        trained_pct = 100 * trained_count / total_count

        # Validate masking looks reasonable
        if masked_pct < 25:
            raise ValueError(
                f"Masking too low ({masked_pct:.1f}% < 25%). "
                "This may indicate training on instructions. "
                "Check collator configuration."
            )

        return {
            "masked_pct": masked_pct,
            "trained_pct": trained_pct,
            "masked_count": masked_count,
            "total_count": total_count,
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """Convert precision string to PyTorch dtype."""
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        elif precision == "fp32":
            return torch.float32
        else:
            print(f"   Unknown precision '{precision}', defaulting to bf16")
            return torch.bfloat16

    def _build_system_prompt(self, config: TrainerConfig) -> str:
        """Build system prompt by filling template with current date."""
        template = config.monitoring.system_prompt_base
        date_str = datetime.now().strftime('%Y-%m-%d')
        return template.format(date=date_str)

    def _build_training_arguments(
        self,
        config: TrainerConfig,
        max_steps: int
    ) -> TrainingArguments:
        """
        Build HuggingFace TrainingArguments from TrainerConfig.

        Uses max_steps for continuous training instead of epochs.
        """
        # Determine precision flags
        use_fp16 = config.hyperparams.fp_precision == "fp16"
        use_bf16 = config.hyperparams.fp_precision == "bf16"

        return TrainingArguments(
            output_dir=config.output.output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=config.hyperparams.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.hyperparams.gradient_accumulation,
            learning_rate=config.hyperparams.learning_rate,
            warmup_steps=config.hyperparams.warmup_steps,
            lr_scheduler_type="constant",  # Don't decay LR for continuous training
            save_steps=config.hyperparams.save_steps,
            save_total_limit=config.hyperparams.save_total_limit,
            eval_steps=config.hyperparams.eval_steps,
            eval_strategy=config.hyperparams.eval_strategy,
            fp16=use_fp16,
            bf16=use_bf16,
            tf32=True,  # Enable TF32 for speedup
            gradient_checkpointing=True,
            optim="adamw_torch_fused",  # Faster than default
            dataloader_num_workers=16,
            dataloader_pin_memory=True,
            logging_steps=config.environment.logging_steps,
            report_to=config.environment.report_to,
            remove_unused_columns=False,
            overwrite_output_dir=config.output.overwrite_output_dir,
            save_safetensors=config.output.save_safetensors,
            max_grad_norm=config.environment.max_grad_norm,
        )


__all__ = ["TrainerEngine", "TrainingResult", "MonitorContext"]
