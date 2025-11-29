"""FLO Trainer - GaLore + 8-bit AdamW for 4B+ models."""

import json
import time
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from arena.trainers.base import BaseTrainer, TrainingResult
from core.paths import get_base_dir

# GaLore optimizer
try:
    from trainer.optimizers.galore_muon import create_galore_muon_optimizer
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False

# Liger kernel for memory efficiency
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False


class FLOProgressCallback(TrainerCallback):
    """Log training progress with loss and VRAM usage."""
    
    def __init__(self, status_file: Path = None):
        self.status_file = status_file
        self.last_loss = None
        self.start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        loss = logs.get("loss")
        if loss is not None:
            self.last_loss = loss
            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm", 0)
            
            print(f"\n  [Step {state.global_step}/{state.max_steps}] "
                  f"Loss: {loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.2f}", flush=True)
            
            if self.status_file:
                try:
                    now = datetime.now().isoformat()
                    flo_status = {
                        "step": state.global_step,
                        "max_steps": state.max_steps,
                        "loss": loss,
                        "lr": lr,
                        "epoch": state.epoch,
                        "timestamp": now,
                    }
                    with open(self.status_file, "w") as f:
                        json.dump(flo_status, f)
                    
                    # Tavern-compatible status
                    tavern_status = {
                        "status": "training",
                        "current_step": state.global_step,
                        "global_step": state.global_step,
                        "max_steps": state.max_steps,
                        "loss": loss,
                        "learning_rate": lr,
                        "current_file": "FLO GaLore Training",
                        "timestamp": now,
                        "hero": "FLO",
                    }
                    tavern_file = self.status_file.parent / "training_status.json"
                    with open(tavern_file, "w") as f:
                        json.dump(tavern_status, f)
                except Exception:
                    pass
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  [VRAM] {allocated:.1f}GB / {reserved:.1f}GB reserved")


class FLOTrainer(BaseTrainer):
    """
    FLO Trainer for Qwen3-4B with GaLore optimizer.
    
    Uses GaLore (Gradient Low-Rank Projection) to reduce optimizer
    memory from ~32GB to ~4GB, making 4B training possible on 24GB.
    """
    
    @property
    def name(self) -> str:
        return "flo"
    
    def _load_training_data(self, data_path: Path) -> list:
        """Load samples from JSONL."""
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    
    def _apply_liger_kernel(self):
        """Apply Liger kernel optimizations."""
        if LIGER_AVAILABLE:
            liger_config = self.hero_config.get("training_defaults", {}).get("liger_kernel", {})
            if liger_config.get("enabled", True):
                apply_liger_kernel_to_qwen2(
                    rope=liger_config.get("fused_rope", True),
                    rms_norm=liger_config.get("fused_rms_norm", True),
                    swiglu=liger_config.get("fused_swiglu", True),
                    cross_entropy=False,
                    fused_linear_cross_entropy=liger_config.get("fused_linear_cross_entropy", True),
                )
                print("  Liger kernel applied (fused ops)")
    
    def train(self, data_path: Path) -> TrainingResult:
        """Run training with GaLore optimizer."""
        if not GALORE_AVAILABLE:
            return TrainingResult(
                success=False,
                error_message="GaLore not available. Install with: pip install galore-torch"
            )
        
        start_time = time.time()
        base_dir = get_base_dir()
        
        # Get training config
        training_defaults = self.hero_config.get("training_defaults", {})
        trainer_config = self.hero_config.get("trainer", {})
        galore_config = trainer_config.get("galore", {})
        
        model_path = self.get_model_path()
        output_dir = self.checkpoints_dir
        
        rank = galore_config.get("rank", 256)
        hidden_lr = training_defaults.get("learning_rate", 0.02)
        aux_lr = 3e-4  # Fixed for aux params
        
        print("\n" + "=" * 60)
        print(f"FLO TRAINING - GaLore")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"Output: {output_dir}")
        print(f"Data: {data_path}")
        print(f"GaLore Rank: {rank}")
        
        # Apply Liger kernel BEFORE loading model
        self._apply_liger_kernel()
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load and process training data
        print(f"\nLoading training data...")
        raw_samples = self._load_training_data(data_path)
        print(f"  Loaded {len(raw_samples)} samples")
        
        max_length = training_defaults.get("max_length", 1024)
        
        def tokenize_messages(example):
            messages = example["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": tokens["input_ids"].squeeze(),
            }
        
        print("  Tokenizing...")
        tokenized_samples = []
        for sample in raw_samples:
            try:
                tokenized = tokenize_messages(sample)
                tokenized_samples.append(tokenized)
            except Exception:
                continue
        
        dataset = Dataset.from_list(tokenized_samples)
        print(f"  Dataset ready: {len(dataset)} samples")
        
        # Load model
        print("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="auto",
        )
        
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
        
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,} ({params/1e9:.2f}B)")
        
        # Create GaLore optimizer
        print("\nCreating GaLore optimizer...")
        optimizer = create_galore_muon_optimizer(
            model,
            rank=rank,
            hidden_lr=hidden_lr,
            aux_lr=aux_lr,
        )
        
        # Training arguments
        batch_size = training_defaults.get("batch_size", 1)
        grad_accum = training_defaults.get("gradient_accumulation", 32)
        epochs = training_defaults.get("epochs", 1)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=hidden_lr,
            bf16=True,
            logging_steps=training_defaults.get("log_steps", 10),
            save_strategy="steps",
            save_steps=training_defaults.get("save_steps", 500),
            warmup_steps=training_defaults.get("warmup_steps", 100),
            lr_scheduler_type="cosine",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to="none",
            gradient_checkpointing=True,
            optim="adamw_torch",
        )
        
        # Create trainer
        status_file = base_dir / "status" / "flo_training.json"
        status_file.parent.mkdir(exist_ok=True)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            optimizers=(optimizer, None),
            callbacks=[FLOProgressCallback(status_file)],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        # Train
        print("\n" + "=" * 60)
        print("STARTING TRAINING - GALORE + 8-bit AdamW")
        print("=" * 60 + "\n")
        
        torch.cuda.reset_peak_memory_stats()
        
        try:
            result = trainer.train()
            
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            duration = time.time() - start_time
            
            print(f"\nTraining complete!")
            print(f"Peak VRAM: {peak_mem:.2f} GB")
            
            # Save final model
            print(f"\nSaving model to {output_dir}...")
            trainer.save_model()
            tokenizer.save_pretrained(str(output_dir))
            
            return TrainingResult(
                success=True,
                steps_completed=result.global_step,
                final_loss=result.training_loss,
                peak_vram_gb=peak_mem,
                duration_seconds=duration,
                checkpoint_path=output_dir,
                metrics={
                    "samples": len(dataset),
                    "epochs": epochs,
                    "galore_rank": rank,
                }
            )
            
        except torch.cuda.OutOfMemoryError as e:
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            return TrainingResult(
                success=False,
                peak_vram_gb=peak_mem,
                duration_seconds=time.time() - start_time,
                error_message=f"OOM at {peak_mem:.2f} GB. Try reducing batch size or GaLore rank."
            )
        except Exception as e:
            return TrainingResult(
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
