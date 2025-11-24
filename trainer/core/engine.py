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
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import json
import random
import time
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import Dataset

from trainer.config.schema import TrainerConfig
from trainer.config.loader import ConfigLoader
from trainer.monitoring.status_writer import TrainingStatusWriter
from trainer.profiles import get_profile


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

    Responsibilities:
        - Validate configuration before training starts
        - Load and initialize model/tokenizer with correct precision
        - Apply data profile transformations to examples
        - Create train/validation splits with proper shuffling
        - Execute HuggingFace Trainer with monitoring
        - Save checkpoints and return comprehensive results

    Data Flow:
        1. Validate config → check locked parameters, paths exist
        2. Load profile → get transformation functions for data
        3. Load model & tokenizer → setup precision (bf16/fp16/fp32)
        4. Prepare datasets → load JSONL, apply profile, tokenize
        5. Create HF Trainer → setup args, collator, callbacks
        6. Execute training → run training loop with monitoring
        7. Save checkpoint → write final model to disk
        8. Return result → structured TrainingResult with metrics

    Attributes:
        status_writer: TrainingStatusWriter for writing status JSON
        model: HuggingFace model instance (None until loaded)
        tokenizer: HuggingFace tokenizer instance (None until loaded)
        profile: DataProfile instance for transformations (None until loaded)

    Example:
        >>> from trainer.config import create_default_config
        >>> from trainer.monitoring import TrainingStatusWriter
        >>>
        >>> config = create_default_config(
        ...     dataset_path="data.jsonl",
        ...     model_path="models/Qwen3-0.6B",
        ...     output_dir="outputs/run1"
        ... )
        >>> status_writer = TrainingStatusWriter("status/training_status.json")
        >>> engine = TrainerEngine(status_writer)
        >>> result = engine.run_job(config)
        >>>
        >>> if result.success:
        ...     print(f"Trained {result.global_step} steps in {result.runtime_sec:.1f}s")
        ...     print(f"Final loss: {result.final_loss:.4f}")
        ...     print(f"Checkpoint: {result.last_checkpoint_path}")
        ... else:
        ...     print(f"Training failed: {result.error_message}")
    """

    def __init__(self, status_writer: TrainingStatusWriter):
        """
        Initialize TrainerEngine with monitoring.

        Args:
            status_writer: Status writer for live UI updates.
                Writes training_status.json with current step, loss, etc.

        Side Effects:
            - Initializes instance attributes (model, tokenizer, profile) to None
            - No file I/O or GPU operations occur until run_job() called
        """
        self.status_writer = status_writer
        self.model = None
        self.tokenizer = None
        self.profile = None

    def run_job(self, config: TrainerConfig) -> TrainingResult:
        """
        Execute a complete training job with the given configuration.

        This is the ONLY public method. All training operations go through here.
        The method is exception-safe: all errors are caught and returned in the
        TrainingResult object rather than being raised.

        Args:
            config: TrainerConfig containing all training parameters:
                - data: DataConfig (dataset_path, validation_split, shuffle, seed)
                - hyperparams: Hyperparams (batch_size, learning_rate, epochs, etc.)
                - profile: ProfileConfig (name="emoji_think" or "regime3", etc.)
                - output: OutputConfig (output_dir, save_steps, overwrite, etc.)
                - monitoring: MonitoringConfig (status_file, system_prompt, etc.)
                - model: ModelConfig (model_path, tokenizer_path, quantization)
                - environment: EnvironmentConfig (logging_steps, report_to, etc.)

        Returns:
            TrainingResult containing:
                - success: bool (True if training completed without errors)
                - global_step: int (total training steps executed)
                - runtime_sec: float (wall-clock time in seconds)
                - last_checkpoint_path: Optional[str] (path to final checkpoint)
                - final_loss: float (last recorded training loss value)
                - summary: Dict[str, Any] (config, metrics, profile name, etc.)
                - error_message: Optional[str] (error description if failed)

        Raises:
            No exceptions are raised. All errors are caught and returned
            in the TrainingResult.error_message field.

        Side Effects:
            - Loads model and tokenizer into GPU memory
            - Reads dataset from config.data.dataset_path
            - Creates checkpoint directories (config.output.output_dir)
            - Writes checkpoint files every config.hyperparams.save_steps
            - Writes final model to {output_dir}/final_model/
            - Updates status_writer (writes training_status.json)
            - Logs training progress to stdout
            - May create profile-specific files (stop signals, etc.)

        Example:
            >>> config = TrainerConfig(
            ...     data=DataConfig(dataset_path="data.jsonl"),
            ...     hyperparams=Hyperparams(batch_size=16, learning_rate=2e-4),
            ...     profile=ProfileConfig(name="regime3"),
            ...     output=OutputConfig(output_dir="outputs/run1")
            ... )
            >>> engine = TrainerEngine(status_writer)
            >>> result = engine.run_job(config)
            >>>
            >>> if result.success:
            ...     print(f"Success! Trained {result.global_step} steps")
            ...     print(f"Final loss: {result.final_loss:.4f}")
            ...     print(f"Model saved to: {result.last_checkpoint_path}")
            ... else:
            ...     print(f"Training failed: {result.error_message}")
        """
        print("\n" + "=" * 80)
        print("🚀 TRAINER ENGINE - FULL IMPLEMENTATION")
        print("=" * 80)
        print(f"Profile: {config.profile.name}")
        print(f"Model: {config.model.model_path}")
        print(f"Dataset: {config.data.dataset_path}")
        print(f"Output: {config.output.output_dir}")
        print(f"Precision: {config.hyperparams.fp_precision}")
        print(f"Batch size: {config.hyperparams.batch_size}")
        print("=" * 80 + "\n")

        start_time = time.time()

        try:
            # 1. Validate config
            print("📋 Step 1: Validating configuration")
            ConfigLoader.validate_locked_config(config)
            print("   ✓ Config validated\n")

            # 2. Load profile
            print(f"📋 Step 2: Loading profile '{config.profile.name}'")
            self.profile = get_profile(config.profile.name)
            print(f"   ✓ Profile loaded: {self.profile.__class__.__name__}\n")

            # 3. Load model & tokenizer
            print("📋 Step 3: Loading model and tokenizer")
            self.model, self.tokenizer = self._load_model_and_tokenizer(config)
            print("   ✓ Model and tokenizer loaded\n")

            # 4. Prepare datasets
            print("📋 Step 4: Preparing datasets")
            train_dataset, val_dataset = self._prepare_dataset(config)
            print("   ✓ Datasets prepared\n")

            # 5. Create trainer
            print("📋 Step 5: Creating HuggingFace Trainer")
            trainer = self._create_trainer(config, train_dataset, val_dataset)
            print("   ✓ Trainer created\n")

            # 6. Execute training
            print("📋 Step 6: Executing training")
            print("=" * 80)
            train_result = trainer.train()
            print("=" * 80)
            print("   ✓ Training complete\n")

            # 7. Save checkpoint
            print("📋 Step 7: Saving final checkpoint")
            final_checkpoint = Path(config.output.output_dir) / "final_model"
            trainer.save_model(str(final_checkpoint))
            print(f"   ✓ Saved to {final_checkpoint}\n")

            # 8. Return result
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
                }
            )

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()

            return TrainingResult.from_error(str(e))

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _load_model_and_tokenizer(
        self,
        config: TrainerConfig
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Load model and tokenizer from disk with proper precision settings.

        Loads HuggingFace model and tokenizer from paths specified in config.
        Applies quantization if requested. Sets precision (bf16/fp16/fp32).
        Disables KV cache for training. Sets pad_token if not present.

        Args:
            config: TrainerConfig with model settings:
                - model.model_path: Path to model directory
                - model.tokenizer_path: Path to tokenizer (or None to use model_path)
                - model.load_in_4bit: Whether to use 4-bit quantization
                - hyperparams.fp_precision: Precision ("bf16", "fp16", or "fp32")

        Returns:
            Tuple of (model, tokenizer):
                - model: torch.nn.Module (AutoModelForCausalLM instance)
                - tokenizer: HuggingFace tokenizer with pad_token set

        Raises:
            OSError: If model_path or tokenizer_path doesn't exist
            ValueError: If model format is invalid or incompatible
            RuntimeError: If GPU memory insufficient for model

        Side Effects:
            - Loads model into GPU memory (~1-2GB for 0.6B model)
            - Prints loading progress to stdout
            - Modifies model.config.use_cache (sets to False)
            - May modify tokenizer.pad_token (sets to eos_token if None)
        """
        model_path = config.model.model_path
        print(f"   Loading from: {model_path}")

        # Setup quantization (if requested)
        quantization_config = None
        if config.model.load_in_4bit:
            print("   🔧 Enabling 4-bit quantization")
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
            "attn_implementation": "sdpa",
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Load tokenizer
        tokenizer_path = config.model.tokenizer_path or model_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   ✓ Set pad_token = eos_token")

        # Disable cache for training
        model.config.use_cache = False
        print("   ✓ Disabled KV cache (use_cache=False)")

        return model, tokenizer

    def _prepare_dataset(
        self,
        config: TrainerConfig
    ) -> Tuple[Dataset, Dataset]:
        """
        Load, transform, and tokenize datasets for training.

        Loads JSONL data, applies profile transformations (emoji_think, regime3),
        creates train/val splits, builds system prompts, and tokenizes examples.

        Data Flow:
            1. Load JSONL file → list of raw examples
            2. Shuffle if config.data.shuffle=True
            3. Split into train/val based on validation_split
            4. Build system prompt with current date
            5. Apply profile.transform_example() to each example
            6. Tokenize using chat template
            7. Create HuggingFace Dataset objects

        Args:
            config: TrainerConfig with data settings:
                - data.dataset_path: Path to JSONL file
                - data.shuffle: Whether to shuffle examples
                - data.seed: Random seed for shuffling
                - monitoring.validation_split: Fraction for validation (e.g., 0.1)
                - monitoring.validation_max_samples: Max validation examples
                - monitoring.system_prompt_base: Template for system prompt
                - hyperparams.max_length: Max sequence length for truncation

        Returns:
            Tuple of (train_dataset, val_dataset):
                - train_dataset: HuggingFace Dataset with tokenized examples
                - val_dataset: HuggingFace Dataset with tokenized examples
                Both datasets have columns: input_ids, attention_mask, labels

        Raises:
            FileNotFoundError: If dataset_path doesn't exist
            json.JSONDecodeError: If JSONL format is invalid
            ValueError: If examples don't have required fields after transform

        Side Effects:
            - Reads entire dataset file into memory
            - Prints dataset statistics to stdout
            - May create temporary files if profile requires (stop signals)
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
        print("   ✓ Profile transformations applied")

        # Tokenize
        def tokenize_function(example):
            """Tokenize a single example"""
            messages = example['messages']

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            result = self.tokenizer(
                text,
                truncation=True,
                max_length=config.hyperparams.max_length,
                padding=False,  # Dynamic padding in collator
            )

            # Add labels (same as input_ids for causal LM)
            result['labels'] = result['input_ids'].copy()

            return result

        print("   Tokenizing...")
        train_dataset = Dataset.from_list(train_examples).map(
            tokenize_function,
            remove_columns=['messages']
        )
        val_dataset = Dataset.from_list(val_examples).map(
            tokenize_function,
            remove_columns=['messages']
        )
        print("   ✓ Tokenization complete")

        return train_dataset, val_dataset

    def _create_trainer(
        self,
        config: TrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset
    ) -> Trainer:
        """
        Create HuggingFace Trainer with all configuration.

        Builds TrainingArguments from config, creates data collator,
        and instantiates HuggingFace Trainer ready for training.

        Args:
            config: TrainerConfig with all training parameters
            train_dataset: Tokenized training dataset with columns:
                - input_ids: List[int]
                - attention_mask: List[int]
                - labels: List[int]
            val_dataset: Tokenized validation dataset (same structure)

        Returns:
            Trainer: Configured HuggingFace Trainer instance ready to call .train()

        Side Effects:
            - Prints trainer configuration to stdout
            - Creates output directory if it doesn't exist
        """
        # Training arguments
        training_args = self._build_training_arguments(config)
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Epochs: {training_args.num_train_epochs}")

        # Data collator (simple padding)
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        return trainer

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """
        Convert precision string to PyTorch dtype.

        Args:
            precision: One of "bf16", "fp16", or "fp32"

        Returns:
            torch.dtype: Corresponding PyTorch dtype
                - "bf16" → torch.bfloat16
                - "fp16" → torch.float16
                - "fp32" → torch.float32
                - unknown → torch.bfloat16 (with warning)
        """
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        elif precision == "fp32":
            return torch.float32
        else:
            print(f"   ⚠️  Unknown precision '{precision}', defaulting to bf16")
            return torch.bfloat16

    def _build_system_prompt(self, config: TrainerConfig) -> str:
        """
        Build system prompt by filling template with current date.

        Args:
            config: TrainerConfig with monitoring.system_prompt_base template.
                Template should contain {date} placeholder.

        Returns:
            str: System prompt with {date} replaced by YYYY-MM-DD format

        Example:
            >>> config.monitoring.system_prompt_base = "Today is {date}. You are helpful."
            >>> engine._build_system_prompt(config)
            "Today is 2025-11-24. You are helpful."
        """
        template = config.monitoring.system_prompt_base
        date_str = datetime.now().strftime('%Y-%m-%d')
        return template.format(date=date_str)

    def _build_training_arguments(self, config: TrainerConfig) -> TrainingArguments:
        """
        Build HuggingFace TrainingArguments from TrainerConfig.

        Converts our TrainerConfig structure to HuggingFace's TrainingArguments
        format. Handles precision flags (fp16 vs bf16), batch size, learning rate,
        eval strategy, checkpointing, and logging.

        Args:
            config: TrainerConfig with all training parameters

        Returns:
            TrainingArguments: HuggingFace TrainingArguments instance with:
                - output_dir, batch_size, learning_rate, warmup_steps
                - num_train_epochs or max_steps
                - save_steps, save_total_limit
                - eval_steps, eval_strategy
                - fp16/bf16 flags based on config.hyperparams.fp_precision
                - logging_steps, report_to
                - other training configuration
        """
        # Determine precision flags
        use_fp16 = config.hyperparams.fp_precision == "fp16"
        use_bf16 = config.hyperparams.fp_precision == "bf16"

        return TrainingArguments(
            output_dir=config.output.output_dir,
            per_device_train_batch_size=config.hyperparams.batch_size,
            gradient_accumulation_steps=config.hyperparams.gradient_accumulation,
            learning_rate=config.hyperparams.learning_rate,
            warmup_steps=config.hyperparams.warmup_steps,
            num_train_epochs=config.hyperparams.num_epochs,
            max_steps=config.hyperparams.max_steps if config.hyperparams.max_steps else -1,
            save_steps=config.hyperparams.save_steps,
            save_total_limit=config.hyperparams.save_total_limit,
            eval_steps=config.hyperparams.eval_steps,
            eval_strategy=config.hyperparams.eval_strategy,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=config.environment.logging_steps,
            report_to=config.environment.report_to,
            remove_unused_columns=False,
            overwrite_output_dir=config.output.overwrite_output_dir,
            save_safetensors=config.output.save_safetensors,
            max_grad_norm=config.environment.max_grad_norm,
        )


__all__ = ["TrainerEngine", "TrainingResult"]
