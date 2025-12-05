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

# Configure CUDA memory allocator BEFORE importing torch
# This prevents OOM from memory fragmentation by using expandable segments
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

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
from trainer.monitoring.context import MonitorContext, ProgressContext, EvalContext, ControlContext
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

# Optional: Fortune Teller loss
try:
    from trainer.core.fortune_teller_trainer import FortuneTellerTrainer
    from trainer.losses import FortuneTellerTracker
    FORTUNE_TELLER_AVAILABLE = True
except ImportError:
    FORTUNE_TELLER_AVAILABLE = False
    FortuneTellerTrainer = None
    FortuneTellerTracker = None

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

# Optional: PEFT for LoRA/QLoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    prepare_model_for_kbit_training = None


@dataclass
class DryRunResult:
    """
    Result of a dry-run validation pass.

    Contains validation status and resource estimates without actually
    loading the model or starting training.
    """
    valid: bool  # True if config and data are valid
    error_message: Optional[str] = None  # Error description if valid=False
    dataset_path: Optional[str] = None  # Validated dataset path
    dataset_examples: int = 0  # Number of examples in dataset
    vram_estimate_gb: float = 0.0  # Estimated VRAM requirement
    estimated_steps: int = 0  # Estimated training steps
    config_summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: str) -> 'DryRunResult':
        """Create a DryRunResult for validation failure."""
        return cls(valid=False, error_message=error)


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
        verbose: If True, log progress to stdout (default True)
    """

    def __init__(self, status_writer: TrainingStatusWriter, verbose: bool = True):
        """
        Initialize TrainerEngine with monitoring.

        Args:
            status_writer: Status writer for live UI updates.
                Writes training_status.json with current step, loss, etc.
            verbose: If True, print progress to stdout. Set False for tests
                or library usage where output should be suppressed.

        Side Effects:
            - Initializes instance attributes to None
            - No file I/O or GPU operations occur until run_job() called
        """
        self.status_writer = status_writer
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.profile = None
        self.is_vision_model = False
        self.avg_seq_len = 0.0

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)

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
        self._log("\n" + "=" * 80)
        self._log("TRAINER ENGINE - ENHANCED IMPLEMENTATION")
        self._log("=" * 80)
        self._log(f"Profile: {config.profile.name}")
        self._log(f"Model: {config.model.model_path}")
        self._log(f"Dataset: {config.data.dataset_path}")
        self._log(f"Output: {config.output.output_dir}")
        self._log(f"Precision: {config.hyperparams.fp_precision}")
        self._log(f"Batch size: {config.hyperparams.batch_size}")
        self._log("=" * 80 + "\n")

        start_time = time.time()
        config_dict = config_dict or {}

        try:
            # 1. Validate config
            self._log("Step 1: Validating configuration")
            ConfigLoader.validate_locked_config(config)
            self._log("   Config validated\n")

            # 2. Load profile
            self._log(f"Step 2: Loading profile '{config.profile.name}'")
            self.profile = get_profile(config.profile.name)
            self._log(f"   Profile loaded: {self.profile.__class__.__name__}\n")

            # 3. Load model & tokenizer (enhanced)
            self._log("Step 3: Loading model and tokenizer")
            self.model, self.tokenizer = self._load_model_and_tokenizer(config)
            self._log("   Model and tokenizer loaded")

            # 3.5. Apply LoRA if training_mode is lora or qlora
            training_mode = getattr(config, 'training_mode', 'full')
            if training_mode in ('lora', 'qlora'):
                self._log(f"   Applying {training_mode.upper()} adapters...")
                is_quantized = (training_mode == 'qlora') or config.model.load_in_4bit
                self.model = self._apply_lora(self.model, config, is_quantized=is_quantized)
                self._log(f"   {training_mode.upper()} adapters applied")
            self._log("")

            # 4. Prepare datasets (with packing)
            self._log("Step 4: Preparing datasets")
            train_dataset, val_dataset = self._prepare_dataset(config)
            self._log("   Datasets prepared\n")

            # 5. Find checkpoint for resumption
            self._log("Step 5: Checking for checkpoint resumption")
            resume_checkpoint, current_global_step = self._find_resume_checkpoint(
                config.output.output_dir
            )
            if resume_checkpoint:
                self._log(f"   Will resume from: {resume_checkpoint}")
                self._log(f"   Current global step: {current_global_step}")
            else:
                self._log("   Starting fresh (no checkpoint found)")
            self._log("")

            # 6. Create trainer (with callbacks, optimizer, collator)
            self._log("Step 6: Creating HuggingFace Trainer")
            trainer, data_collator = self._create_trainer(
                config=config,
                config_dict=config_dict,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                monitors=monitors,
                callbacks=callbacks,
                current_global_step=current_global_step,
            )
            self._log("   Trainer created\n")

            # 7. Validate masking
            self._log("Step 7: Validating response masking")
            masking_stats = self._validate_masking(train_dataset, data_collator)
            self._log(f"   Masked (instruction): {masking_stats['masked_pct']:.1f}%")
            self._log(f"   Trained (response): {masking_stats['trained_pct']:.1f}%")
            self._log("   Masking validation passed\n")

            # 8. Execute training
            self._log("Step 8: Executing training")
            self._log("=" * 80)
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
            self._log("=" * 80)
            self._log("   Training complete\n")

            # 8.5. Save Fortune Teller history if enabled
            if config.fortune_teller.enabled and hasattr(self, 'fortune_teller_tracker') and self.fortune_teller_tracker:
                history_path = config.fortune_teller.history_path or str(Path(config.output.output_dir) / "fortune_teller_history.json")
                self.fortune_teller_tracker.save(history_path)
                self._log(f"   Saved Fortune Teller history to {history_path}")

                # Log summary stats
                stats = self.fortune_teller_tracker.get_stats()
                if stats:
                    self._log(f"   - Avg surprise: {stats.get('avg_surprise', 0):.3f}")
                    self._log(f"   - Records: {stats.get('num_records', 0)}\n")

            # 9. Save final checkpoint
            self._log("Step 9: Saving final checkpoint")
            final_checkpoint = Path(config.output.output_dir) / "final_model"
            trainer.save_model(str(final_checkpoint))
            self.tokenizer.save_pretrained(str(final_checkpoint))
            self._log(f"   Saved to {final_checkpoint}\n")

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
            self._log(f"\n Training failed: {e}")
            import traceback
            traceback.print_exc()

            return TrainingResult.from_error(str(e))

    def dry_run(self, config: TrainerConfig) -> DryRunResult:
        """
        Validate configuration and estimate resources without training.

        This is a lightweight check that validates:
        - Config structure and locked values
        - Dataset existence and format
        - VRAM estimate (based on model size and batch size)
        - Estimated training steps

        Does NOT load the model or tokenizer. Use this before run_job()
        to catch configuration errors early.

        Args:
            config: TrainerConfig to validate

        Returns:
            DryRunResult with validation status and estimates
        """
        self._log("\n" + "=" * 60)
        self._log("TRAINER ENGINE - DRY RUN")
        self._log("=" * 60)

        try:
            # 1. Validate locked config
            self._log("Checking config...")
            ConfigLoader.validate_locked_config(config)
            self._log("   Config validation passed")

            # 2. Check dataset exists
            dataset_path = Path(config.data.dataset_path)
            if not dataset_path.exists():
                return DryRunResult.from_error(f"Dataset not found: {dataset_path}")
            self._log(f"   Dataset found: {dataset_path}")

            # 3. Count examples
            example_count = 0
            with open(dataset_path) as f:
                for line in f:
                    if line.strip():
                        example_count += 1
            self._log(f"   Examples: {example_count:,}")

            if example_count == 0:
                return DryRunResult.from_error("Dataset is empty")

            # 4. Estimate VRAM (rough heuristic based on model name)
            model_path = config.model.model_path
            vram_estimate = self._estimate_vram(model_path, config)
            self._log(f"   Estimated VRAM: {vram_estimate:.1f} GB")

            # 5. Estimate steps
            effective_batch = (
                config.hyperparams.batch_size *
                config.hyperparams.gradient_accumulation
            )
            estimated_steps = example_count // effective_batch
            if example_count % effective_batch != 0:
                estimated_steps += 1
            self._log(f"   Estimated steps: {estimated_steps:,}")

            self._log("=" * 60)
            self._log("DRY RUN PASSED")
            self._log("=" * 60 + "\n")

            return DryRunResult(
                valid=True,
                dataset_path=str(dataset_path),
                dataset_examples=example_count,
                vram_estimate_gb=vram_estimate,
                estimated_steps=estimated_steps,
                config_summary={
                    'model_path': model_path,
                    'profile': config.profile.name,
                    'batch_size': config.hyperparams.batch_size,
                    'gradient_accumulation': config.hyperparams.gradient_accumulation,
                    'learning_rate': config.hyperparams.learning_rate,
                    'max_length': config.hyperparams.max_length,
                    'precision': config.hyperparams.fp_precision,
                }
            )

        except Exception as e:
            self._log(f"DRY RUN FAILED: {e}")
            return DryRunResult.from_error(str(e))

    def _estimate_vram(self, model_path: str, config: TrainerConfig) -> float:
        """
        Estimate VRAM usage based on model and config.

        This is a rough heuristic. Actual usage may vary.
        """
        # Base VRAM by model size (from model name patterns)
        model_lower = model_path.lower()

        if "0.5b" in model_lower or "0.6b" in model_lower:
            base_vram = 4.0
        elif "1.5b" in model_lower or "1b" in model_lower:
            base_vram = 6.0
        elif "3b" in model_lower:
            base_vram = 10.0
        elif "4b" in model_lower:
            base_vram = 12.0
        elif "7b" in model_lower or "8b" in model_lower:
            base_vram = 18.0
        elif "13b" in model_lower or "14b" in model_lower:
            base_vram = 28.0
        else:
            base_vram = 8.0  # Default assumption

        # Adjust for batch size (roughly linear for small batches)
        batch_multiplier = 1.0 + (config.hyperparams.batch_size - 1) * 0.3

        # Adjust for sequence length
        seq_multiplier = config.hyperparams.max_length / 2048

        # Adjust for precision
        if config.hyperparams.fp_precision == "fp32":
            precision_multiplier = 2.0
        elif config.hyperparams.fp_precision == "fp16":
            precision_multiplier = 1.0
        else:  # bf16
            precision_multiplier = 1.0

        # Gradient checkpointing reduces memory
        grad_ckpt_multiplier = 0.6 if getattr(config.hyperparams, 'use_gradient_checkpointing', True) else 1.0

        return base_vram * batch_multiplier * seq_multiplier * precision_multiplier * grad_ckpt_multiplier

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
        self._log(f"   Loading from: {model_path}")

        # Detect attention implementation
        attn_impl = "sdpa"
        if FLASH_ATTN_AVAILABLE:
            attn_impl = "flash_attention_2"
            self._log("   Flash Attention 2 detected")
        else:
            self._log("   Using SDPA attention (flash_attention_2 not installed)")

        # Setup quantization (if requested)
        quantization_config = None
        if config.model.load_in_4bit:
            self._log("   Enabling 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Get precision
        torch_dtype = self._get_torch_dtype(config.hyperparams.fp_precision)
        self._log(f"   Precision: {config.hyperparams.fp_precision} ({torch_dtype})")

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
                self._log("   Loaded as Qwen3VLForConditionalGeneration")
                self.is_vision_model = True

                # Freeze vision and video towers for text-only training
                self._log("   Freezing vision/video towers for text-only training...")
                frozen_params = 0
                for n, p in model.named_parameters():
                    if any(k in n for k in ["vision_model", "video_model", "visual"]):
                        p.requires_grad = False
                        frozen_params += 1
                self._log(f"   Froze {frozen_params} vision/video parameters")

                # Use AutoProcessor for Qwen3VL
                processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
                tokenizer = processor.tokenizer
                self._log("   Loaded AutoProcessor")

            except Exception as e:
                self._log(f"   Qwen3VL failed ({str(e)[:50]}...), trying AutoModelForCausalLM...")
                model = None

        # Fallback to standard CausalLM
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            self._log("   Loaded as AutoModelForCausalLM")

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self._log("   Set pad_token = eos_token")

        # Apply chat template override (fixes Qwen3 <think> injection)
        try:
            from core.chat_templates import apply_chat_template_override
            profile_name = config.profile.name if config.profile else None
            apply_chat_template_override(tokenizer, profile_name=profile_name, verbose=self.verbose)
        except ImportError:
            self._log("   Chat template override not available")

        # Disable KV cache for training
        model.config.use_cache = False
        self._log("   Disabled KV cache (use_cache=False)")

        # Enable gradient checkpointing if configured
        use_grad_ckpt = getattr(config.hyperparams, 'use_gradient_checkpointing', True)
        if use_grad_ckpt and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self._log("   Enabled gradient checkpointing")
        else:
            self._log("   Gradient checkpointing disabled")

        return model, tokenizer

    def _apply_lora(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        is_quantized: bool = False
    ) -> torch.nn.Module:
        """
        Apply LoRA adapters to the model.

        Args:
            model: The base model
            config: TrainerConfig with LoRA settings
            is_quantized: Whether model is 4-bit quantized (for QLoRA)

        Returns:
            Model with LoRA adapters
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA training. "
                "Install with: pip install peft"
            )

        # Get LoRA config from config or use defaults
        lora_config = getattr(config, 'lora', None)

        # Handle both dataclass (LoRAConfig) and dict formats
        if lora_config is None:
            lora_r = 16
            lora_alpha = 32
            lora_dropout = 0.05
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif hasattr(lora_config, 'r'):  # It's a dataclass
            lora_r = lora_config.r
            lora_alpha = lora_config.alpha
            lora_dropout = lora_config.dropout
            target_modules = lora_config.target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
        else:  # It's a dict
            lora_r = lora_config.get('r', 16)
            lora_alpha = lora_config.get('alpha', 32)
            lora_dropout = lora_config.get('dropout', 0.05)
            target_modules = lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ])

        self._log(f"   Applying LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        self._log(f"   Target modules: {target_modules}")

        # Prepare model for k-bit training if quantized
        if is_quantized:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )
            self._log("   Prepared model for k-bit training")

        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self._log(
            f"   LoRA applied: {trainable_params/1e6:.2f}M trainable / "
            f"{total_params/1e6:.2f}M total ({100*trainable_params/total_params:.2f}%)"
        )

        return model

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
        self._log(f"   Loading from: {dataset_path}")

        examples = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        self._log(f"   Total examples: {len(examples):,}")

        # Shuffle
        if config.data.shuffle:
            random.seed(config.data.seed)
            random.shuffle(examples)
            self._log(f"   Shuffled (seed={config.data.seed})")

        # Split train/val
        val_size = min(
            config.monitoring.validation_max_samples,
            int(len(examples) * config.monitoring.validation_split)
        )
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        self._log(f"   Train: {len(train_examples):,}")
        self._log(f"   Val: {len(val_examples):,}")

        # Build system prompt
        system_prompt = self._build_system_prompt(config)
        self._log(f"   System prompt: \"{system_prompt[:50]}...\"")

        # Transform using profile
        self._log("   Applying profile transformations...")
        train_examples = [
            self.profile.transform_example(ex, idx, system_prompt)
            for idx, ex in enumerate(train_examples)
        ]
        val_examples = [
            self.profile.transform_example(ex, idx, system_prompt)
            for idx, ex in enumerate(val_examples)
        ]
        self._log("   Profile transformations applied")

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

        self._log("   Tokenizing...")
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
        self._log("   Tokenization complete")

        # Calculate average sequence length
        sample_count = min(100, len(train_dataset))
        if sample_count > 0:
            total_tokens = sum(
                len(train_dataset[i]["input_ids"])
                for i in range(sample_count)
            )
            self.avg_seq_len = total_tokens / sample_count
            self._log(f"   Avg seq length: {self.avg_seq_len:.1f} tokens")

        # Pack dataset for efficiency (config is primary, env is override for debugging)
        enable_packing = config.data.enable_packing
        if os.environ.get("ENABLE_PACKING") == "0":
            enable_packing = False
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
            self._log(f"   Packing dataset (max_length={max_length})...")
            self._log(f"      Before packing: {len(dataset)} examples")

            packed = pack_dataset(
                dataset,
                seq_length=max_length,
                strategy="bfd"  # Best Fit Decreasing
            )

            self._log(f"      After packing: {len(packed)} packed sequences")

            # Remove seq_lengths metadata (collator can't handle it)
            if 'seq_lengths' in packed.column_names:
                packed = packed.remove_columns(['seq_lengths'])
                self._log("      Removed seq_lengths metadata")

            self._log("      Packing enabled")
            return packed

        except ImportError:
            self._log("      trl not available, skipping packing")
            return dataset
        except Exception as e:
            self._log(f"      Packing failed ({e}), continuing without packing")
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
            self._log("   Deleted old scheduler.pt to force fresh constant LR")

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
        self._log(f"   Steps for this dataset: {steps_for_this_file}")
        self._log(f"   Max steps (cumulative): {max_steps}")

        # Training arguments
        training_args = self._build_training_arguments(config, max_steps)
        self._log(f"   Batch size: {training_args.per_device_train_batch_size}")
        self._log(f"   Learning rate: {training_args.learning_rate}")

        # Data collator - response-only masking
        data_collator = self._create_data_collator()
        self._log("   Using response-only collator (instruction masked)")

        # Create optimizer (Muon or default AdamW)
        custom_optimizer, custom_scheduler = self._create_optimizer(
            config_dict, max_steps, config.hyperparams.warmup_steps
        )

        # Build callbacks list
        all_callbacks = list(callbacks or [])

        # Wire LiveMonitorCallback from MonitorContext (engine-first refactor)
        # NOTE: live_monitor can be None - callback still handles status/heartbeats
        if LIVE_MONITOR_AVAILABLE and monitors:
            # Use provided status_writer or fall back to engine's
            status_writer = monitors.status_writer or self.status_writer

            # Create LayerMonitor if not provided but available
            layer_monitor = monitors.layer_monitor
            if layer_monitor is None and LAYER_MONITOR_AVAILABLE:
                try:
                    layer_monitor = LayerMonitor(self.model)
                    self._log("   Created LayerMonitor for layer statistics")
                except Exception as e:
                    self._log(f"   LayerMonitor disabled: {e}")
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
            self._log("   LiveMonitorCallback enabled (full monitoring)")
        elif monitors:
            self._log("   Warning: LiveMonitorCallback not available, status updates disabled")

        # Create trainer (with optional Fortune Teller loss)
        if config.fortune_teller.enabled:
            if not FORTUNE_TELLER_AVAILABLE:
                self._log("   WARNING: Fortune Teller requested but not available, using standard trainer")
                trainer_cls = Trainer
                trainer_kwargs = {}
            else:
                self._log(f"   Fortune Teller enabled (metric: {config.fortune_teller.surprise_metric})")
                trainer_cls = FortuneTellerTrainer

                # Create tracker
                tracker = FortuneTellerTracker() if config.fortune_teller.save_history else None

                # Prepare loss config
                loss_config = {
                    "surprise_metric": config.fortune_teller.surprise_metric,
                    "min_surprise": config.fortune_teller.min_surprise,
                    "max_surprise": config.fortune_teller.max_surprise,
                    "normalize_batch": config.fortune_teller.normalize_batch,
                    "temperature": config.fortune_teller.temperature,
                }
                trainer_kwargs = {
                    "loss_config": loss_config,
                    "tracker": tracker,
                }
                self._log(f"   - min_surprise: {config.fortune_teller.min_surprise}")
                self._log(f"   - max_surprise: {config.fortune_teller.max_surprise}")
                self._log(f"   - normalize_batch: {config.fortune_teller.normalize_batch}")
                self._log(f"   - temperature: {config.fortune_teller.temperature}")
        else:
            trainer_cls = Trainer
            trainer_kwargs = {}

        trainer = trainer_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=all_callbacks if all_callbacks else None,
            optimizers=(custom_optimizer, custom_scheduler) if custom_optimizer else (None, None),
            **trainer_kwargs
        )

        # Save tracker reference if using Fortune Teller
        if config.fortune_teller.enabled and FORTUNE_TELLER_AVAILABLE:
            self.fortune_teller_tracker = getattr(trainer, 'tracker', None)

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
            self._log("   Warning: DataCollatorForCompletionOnly not available")
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

        # Handle 8-bit Adam (saves ~8GB VRAM for 4B models)
        if optimizer_type == "adamw_8bit":
            self._log("\n" + "=" * 60)
            self._log("8-BIT ADAM OPTIMIZER - Memory Efficient")
            self._log("=" * 60)

            optimizer, scheduler, opt_info = create_custom_optimizer(
                self.model,
                config_dict,
                optimizer_type="adamw_8bit",
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
            )

            self._log(f"  Learning rate: {opt_info.learning_rate}")
            self._log(f"  Trainable params: {opt_info.trainable_params:,}")
            self._log(f"  Saves ~8GB VRAM vs regular AdamW")
            self._log("=" * 60 + "\n")

            return optimizer, scheduler

        # Handle GaLore (low-rank gradient projection - full fine-tune with less memory)
        if optimizer_type in ("galore", "galore_8bit"):
            self._log("\n" + "=" * 60)
            self._log(f"GALORE OPTIMIZER - Low-Rank Gradient Projection")
            self._log("=" * 60)

            optimizer, scheduler, opt_info = create_custom_optimizer(
                self.model,
                config_dict,
                optimizer_type=optimizer_type,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
            )

            self._log(f"  Learning rate: {opt_info.learning_rate}")
            self._log(f"  Projection rank: {opt_info.galore_rank}")
            self._log(f"  Trainable params: {opt_info.trainable_params:,}")
            is_8bit = "8bit" in optimizer_type
            self._log(f"  Mode: {'8-bit quantized' if is_8bit else 'full precision'}")
            self._log(f"  Saves ~{'8x' if is_8bit else '4x'} optimizer memory vs AdamW")
            self._log("=" * 60 + "\n")

            return optimizer, scheduler

        if optimizer_type != "muon":
            return None, None

        self._log("\n" + "=" * 60)
        self._log("MUON OPTIMIZER - Orthogonalized Momentum")
        self._log("=" * 60)

        # Show parameter grouping summary
        summary = get_param_group_summary(self.model)
        total = summary.muon_params + summary.adam_params
        self._log(f"  Hidden weights (Muon): {summary.muon_params:,} params ({100*summary.muon_params/total:.1f}%)")
        self._log(f"  Other (AdamW):         {summary.adam_params:,} params ({100*summary.adam_params/total:.1f}%)")

        # Create optimizer
        optimizer, scheduler, opt_info = create_custom_optimizer(
            self.model,
            config_dict,
            optimizer_type="muon",
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        self._log(f"  Hidden LR: {opt_info.hidden_lr}")
        self._log(f"  Aux LR:    {opt_info.aux_lr}")
        self._log("=" * 60 + "\n")

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
        # Note: 15% threshold accommodates short-response data (binary arithmetic, etc.)
        if masked_pct < 15:
            raise ValueError(
                f"Masking too low ({masked_pct:.1f}% < 15%). "
                "This may indicate training on instructions. "
                "Check collator configuration."
            )

        # Check for too much masking (not training on enough content)
        if masked_pct > 95:
            raise ValueError(
                f"Masking too high ({masked_pct:.1f}% > 95%). "
                "Not training on enough response content. "
                "Check data format - responses may be empty or very short."
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
            self._log(f"   Unknown precision '{precision}', defaulting to bf16")
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

        # Build base args
        args_kwargs = dict(
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
            optim=config.environment.optimizer_type,  # Configurable: adamw_torch_fused, paged_adamw_32bit, etc.
            dataloader_num_workers=4,  # Reduced from 16 - too many workers causes I/O contention
            dataloader_pin_memory=True,
            logging_steps=config.environment.logging_steps,
            report_to=config.environment.report_to,
            remove_unused_columns=False,
            overwrite_output_dir=config.output.overwrite_output_dir,
            save_safetensors=config.output.save_safetensors,
            max_grad_norm=config.environment.max_grad_norm,
        )

        # Add DeepSpeed config if specified (enables CPU offloading for large models)
        if config.environment.deepspeed_config:
            from pathlib import Path
            ds_path = Path(config.environment.deepspeed_config)
            if ds_path.exists():
                args_kwargs["deepspeed"] = str(ds_path)
                logger.info(f"🚀 DeepSpeed enabled: {ds_path}")
            else:
                logger.warning(f"⚠️  DeepSpeed config not found: {ds_path}")

        return TrainingArguments(**args_kwargs)


__all__ = ["TrainerEngine", "TrainingResult", "DryRunResult", "MonitorContext"]
