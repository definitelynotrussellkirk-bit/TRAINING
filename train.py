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
        --model qwen3_8b \
        --output-dir ~/my_adapter \
        --epochs 2 \
        --batch-size 4
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Import our components
from validator import DatasetValidator
from model_db import ModelDatabase
from time_estimator import TimeEstimator
from live_monitor import LiveInferenceMonitor
from training_status import TrainingStatusWriter, DEFAULT_STATUS_FILE
from evolution_tracker import EvolutionTracker
from custom_collator import DataCollatorForCompletionOnly

# Try to import detailed monitoring (optional)
try:
    from detail_collector import add_detail_collector_to_trainer
    DETAILED_MONITORING_AVAILABLE = True
except ImportError:
    DETAILED_MONITORING_AVAILABLE = False
from desktop_notifier import DesktopNotifier


class UltimateTrainer:
    """Main trainer orchestrator."""

    def __init__(self, args, controller=None):
        self.args = args
        self.controller = controller  # Training control system (pause/stop)
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
        # Pass config values to status writer (hardcoded max_new_tokens=getattr(self.args, "eval_max_tokens", 512) from train.py:482)
        self.status_writer = TrainingStatusWriter(
            DEFAULT_STATUS_FILE,
            max_output_tokens=2048,  # From train.py line 482
            context_window=getattr(args, 'max_length', 2048)
        )
        self.notifier = DesktopNotifier()
        self.training_start_time = None

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
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "attn_implementation": "sdpa"
            }

            # Add quantization config if using QLoRA, otherwise use bfloat16
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16

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

            # Prepare for LoRA
            self.model = prepare_model_for_kbit_training(self.model)

            # Add LoRA
            lora_config = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

            return True

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_dataset(self) -> bool:
        """Load and prepare dataset."""
        try:
            # Build enforced system prompt with date
            from datetime import datetime
            enforced_system_prompt = (
                f"Today is {datetime.now().strftime('%Y-%m-%d')}. "
                "You are happy. You enjoy being helpful. "
                "Do your best to follow the directions given. "
                "If you are asked to think with a given token, you will emit that token "
                "the specified number of times before beginning your response. "
                "You no longer use <think>"
            )
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

            # Inject enforced system prompt into copies of examples
            def inject_system(ex):
                msgs = ex.get('messages', [])
                if not msgs or msgs[0].get('role') != 'system':
                    msgs = [{"role": "system", "content": self.enforced_system_prompt}] + msgs
                else:
                    msgs[0]["content"] = self.enforced_system_prompt
                new_ex = dict(ex)
                new_ex['messages'] = msgs
                return new_ex

            train_examples = [inject_system(ex) for ex in train_examples]
            val_examples = [inject_system(ex) for ex in val_examples]

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

            # Keep val examples in original format for live monitoring
            self.val_dataset = val_examples

            # MEMORY FIX: Free intermediate data structures
            del examples
            del train_data
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

            # Format validation examples same as training data
            def format_example(ex):
                messages = ex['messages']

                # Insert system prompt if not already present
                if not messages or messages[0].get('role') != 'system':
                    messages = [{"role": "system", "content": self.enforced_system_prompt}] + messages
                else:
                    messages[0]["content"] = self.enforced_system_prompt

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

            # Build tiny, tokenized subset for fast micro-evals (keep overhead low)
            mini_count = min(16, len(self.fixed_val_dataset))
            if mini_count > 0:
                mini_subset = self.fixed_val_dataset.select(range(mini_count))
                tokenized = self.tokenizer(
                    mini_subset["text"],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt"
                )
                self.tokenized_fixed_val_small = tokenized
                print(f"   🔎 Prepared {mini_count} mini eval samples for quick checks")

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
        self.live_monitor = LiveInferenceMonitor(
            model=self.model,
            tokenizer=self.tokenizer,
            val_examples=self.val_dataset,
            num_samples=self.args.num_eval_samples
        )

    def train(self) -> bool:
        """Run actual training."""
        try:
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=2048,
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
                if checkpoints:
                    # Use the latest checkpoint based on step number
                    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
                    print(f"🔍 Latest checkpoint: {latest_checkpoint}")
                    trainer_state_file = latest_checkpoint / "trainer_state.json"
                    print(f"🔍 State file exists: {trainer_state_file.exists()}")
                    if trainer_state_file.exists():
                        with open(trainer_state_file, 'r') as f:
                            trainer_state = json.load(f)
                            current_global_step = trainer_state.get('global_step', 0)
                            print(f"📊 Current training progress: {current_global_step} steps completed")
                else:
                    print("🔍 No checkpoints found - starting from step 0")
            else:
                print(f"🔍 Output dir doesn't exist: {self.args.output_dir}")

            # Calculate max_steps for continuous training
            # This ensures each new file adds its steps to the running total
            max_steps = current_global_step + steps_for_this_file

            print(f"📈 Total steps after this batch: {current_global_step} + {steps_for_this_file} = {max_steps}")

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
                save_total_limit=None,  # Keep all checkpoints - manually clean by date later
                fp16=True,
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
                             avg_seq_len: float = 0.0, effective_batch: int = 1, micro_eval_inputs=None, micro_eval_interval: int = 500):
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
                    if step_time and step_time > 0:
                        steps_per_sec = 1.0 / step_time
                        tokens_per_step = self.avg_seq_len * self.effective_batch if self.avg_seq_len else None
                        tokens_per_sec = steps_per_sec * tokens_per_step if tokens_per_step else None
                        self.steps_per_sec_ema = steps_per_sec if self.steps_per_sec_ema is None else 0.1 * steps_per_sec + 0.9 * self.steps_per_sec_ema
                        if tokens_per_sec is not None:
                            self.tokens_per_sec_ema = tokens_per_sec if self.tokens_per_sec_ema is None else 0.1 * tokens_per_sec + 0.9 * self.tokens_per_sec_ema
                            if self.throughput_baseline is None and state.global_step > 20:
                                self.throughput_baseline = self.tokens_per_sec_ema
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
                    if alerts:
                        summary = {}
                        for a in alerts:
                            summary[a["severity"]] = summary.get(a["severity"], 0) + 1
                        alert_summary = summary

                    # Time-based status updates (every ~2 seconds)
                    if current_time - self.last_update_time >= self.update_interval:
                        self.status_writer.update_progress(
                            step=state.global_step,
                            total_steps=self.total_steps,
                            epoch=int(current_epoch),
                            loss=current_loss,
                            lr=current_lr,
                            val_loss=val_loss,
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
                            alert_summary=alert_summary
                        )
                        self.last_update_time = current_time

                    # Prompt/golden snapshots (no generation) to keep UI populated
                    if (
                        self.raw_train_examples
                        and (current_time - self.last_prompt_snapshot_time) >= self.prompt_snapshot_interval
                    ):
                        try:
                            dataset_idx = state.global_step % len(self.raw_train_examples)
                            current_example = self.raw_train_examples[dataset_idx]
                            if 'messages' in current_example:
                                messages = current_example['messages']
                                system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
                                user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
                                golden_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), None)
                                if user_msg and golden_msg:
                                    # Truncate to avoid bloating status JSON
                                    self.status_writer.update_prompt_snapshot(
                                        prompt=user_msg[:1000],
                                        golden=golden_msg[:1000],
                                        system_prompt=system_msg[:500] if system_msg else None
                                    )
                                    self.last_prompt_snapshot_time = current_time
                        except Exception as e:
                            print(f"Warning: Could not record prompt snapshot at step {state.global_step}: {e}")

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
                                            outputs = self.model_ref.generate(
                                                **inputs,
                                                max_new_tokens=512,  # Longer preview output
                                                temperature=0.1,
                                                do_sample=False,
                                                pad_token_id=self.tokenizer.eos_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id
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

                                        # Check if match
                                        matches = golden_msg.strip() == model_output.strip()

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
                                            current_file=self.current_file
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

            # Use the calculated values from earlier:
            # - current_global_step (from checkpoint)
            # - steps_for_this_file (for this dataset)
            # - max_steps (target total)

            # Get batch context from args (passed by daemon)
            current_file = getattr(self.args, 'current_file', None)
            batch_number = getattr(self.args, 'batch_number', None)
            batch_queue_size = getattr(self.args, 'batch_queue_size', None)

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
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
                    micro_eval_interval=max(500, self.args.eval_steps)
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
                checkpoints = list(Path(self.args.output_dir).glob("checkpoint-*"))
                if checkpoints:
                    # Use the latest checkpoint based on step number
                    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
                    resume_from_checkpoint = str(latest_checkpoint)
                    print(f"📦 Resuming from checkpoint: {resume_from_checkpoint}")
                    print(f"   (This preserves optimizer state for smooth continuation)")

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

            return True

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Training error: {error_msg}")
            import traceback
            traceback.print_exc()

            # Mark as crashed
            self.status_writer.mark_crashed(error_msg, type(e).__name__)

            # Send crash notification
            if "out of memory" in error_msg.lower() or "OOM" in error_msg:
                self.notifier.oom_error()
            else:
                self.notifier.training_crashed(error_msg)

            return False

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
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                model_output = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
            self.model.train()
            batch_step = 0
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
                model_output_length=len(model_output)
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

    # LoRA params
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit quantization) to reduce VRAM usage")

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
