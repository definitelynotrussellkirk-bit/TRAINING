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
    """Result of a training job"""
    success: bool
    global_step: int
    runtime_sec: float
    last_checkpoint_path: Optional[str]
    final_loss: float
    summary: Dict[str, Any]
    error_message: Optional[str] = None

    @classmethod
    def from_error(cls, error: str) -> 'TrainingResult':
        """Create error result"""
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
    Core training engine - stable API surface.

    This provides a clean, testable API for training:
    - Single entry point: run_job(config)
    - No scattered config reads
    - Profile-based data transformation
    - Returns structured results

    Architecture Flow:
        1. Validate config
        2. Load profile
        3. Load model & tokenizer
        4. Prepare datasets
        5. Create HF Trainer
        6. Execute training
        7. Save checkpoint
        8. Return result
    """

    def __init__(self, status_writer: TrainingStatusWriter):
        """
        Initialize TrainerEngine.

        Args:
            status_writer: Status writer for UI updates
        """
        self.status_writer = status_writer
        self.model = None
        self.tokenizer = None
        self.profile = None

    def run_job(self, config: TrainerConfig) -> TrainingResult:
        """
        Execute a training job.

        This is the ONLY public method. All training goes through here.

        Args:
            config: Complete training configuration

        Returns:
            TrainingResult with success status and metrics

        Raises:
            No exceptions - all errors are caught and returned in TrainingResult
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
        Load model and tokenizer from config.

        Args:
            config: TrainerConfig with model settings

        Returns:
            (model, tokenizer) tuple
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
        Load and prepare datasets.

        Args:
            config: TrainerConfig with data settings

        Returns:
            (train_dataset, val_dataset) tuple
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
        Create HuggingFace Trainer.

        Args:
            config: TrainerConfig
            train_dataset: Tokenized training dataset
            val_dataset: Tokenized validation dataset

        Returns:
            Trainer instance
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
        """Convert precision string to torch dtype"""
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
        """Build system prompt from config"""
        template = config.monitoring.system_prompt_base
        date_str = datetime.now().strftime('%Y-%m-%d')
        return template.format(date=date_str)

    def _build_training_arguments(self, config: TrainerConfig) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config"""
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
