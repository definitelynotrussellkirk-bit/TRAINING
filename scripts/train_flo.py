#!/usr/bin/env python3
"""
FLO Training Script - Qwen3-4B with Muon+Adam Hybrid Optimizer

Supports two training modes:
1. Muon (default): Hybrid Muon+AdamW optimizer for faster convergence
   - Muon for hidden weight matrices (attention, MLP)
   - AdamW for embeddings, heads, biases
   - Uses gradient checkpointing + Liger for memory efficiency

2. DeepSpeed ZeRO-3: For when you need maximum memory efficiency
   - CPU offload for optimizer states and parameters
   - Uses standard AdamW (DeepSpeed incompatible with custom optimizers)

Usage:
    # Train with Muon (recommended)
    python3 scripts/train_flo.py --data data/training.jsonl

    # Train with DeepSpeed ZeRO-3
    deepspeed --num_gpus=1 scripts/train_flo.py --data data/training.jsonl --deepspeed

    # Generate data and train
    python3 scripts/train_flo.py --generate --count 1000

    # Dry run (show config, don't train)
    python3 scripts/train_flo.py --dry-run
"""

import os
import sys
import json
import argparse
import requests
import torch
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Muon optimizer
from trainer.optimizers import create_optimizer, split_transformer_params, SingleDeviceMuonWithAuxAdam
from trainer.optimizers.param_groups import print_param_group_summary

# GaLore optimizer
try:
    from trainer.optimizers.galore_muon import GaLoreMuonOptimizer, create_galore_muon_optimizer
    GALORE_AVAILABLE = True
except ImportError:
    GALORE_AVAILABLE = False

# Apply Liger BEFORE loading model for memory efficiency
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Warning: Liger kernel not available. Training will use more VRAM.")


def generate_skill_data(skill: str, level: int, count: int) -> list:
    """Generate training data from skill server in messages format."""
    ports = {"sy": 8080, "bin": 8090}
    port = ports.get(skill, 8080)

    url = f"http://localhost:{port}/generate"
    payload = {"level": level, "count": count}

    print(f"  Generating {count} samples from {skill.upper()} level {level}...")

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        samples = []
        items = data.get("samples", []) or data.get("puzzles", [])
        for item in items:
            prompt = item.get("user_prompt", "")
            response = item.get("assistant_response", "")
            if prompt and response:
                samples.append({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                })

        print(f"    Got {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"    Error: {e}")
        return []


def generate_mixed_data(skills: list, count_per_skill: int, levels: list = None) -> list:
    """Generate mixed training data from multiple skills."""
    all_samples = []

    for skill in skills:
        skill_levels = levels or [1, 2, 3]
        count_per_level = count_per_skill // len(skill_levels)

        for level in skill_levels:
            samples = generate_skill_data(skill, level, count_per_level)
            all_samples.extend(samples)

    import random
    random.shuffle(all_samples)

    return all_samples


def save_training_data(samples: list, output_path: Path) -> None:
    """Save samples as JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")


def load_training_data(data_path: Path) -> list:
    """Load samples from JSONL."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_campaign_config(campaign_path: Path) -> dict:
    """Load campaign config."""
    config_file = campaign_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config at {config_file}")

    with open(config_file) as f:
        return json.load(f)


class TrainingProgressCallback(TrainerCallback):
    """Log training progress with loss and VRAM usage."""

    def __init__(self, status_file: Path = None):
        self.status_file = status_file
        self.last_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is None:
            return

        loss = logs.get("loss")
        if loss is not None:
            self.last_loss = loss
            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm", 0)

            # Print progress on new line (tqdm overwrites \r)
            print(f"\n  [Step {state.global_step}/{state.max_steps}] "
                  f"Loss: {loss:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.2f}", flush=True)

            # Update status files for Tavern
            if self.status_file:
                try:
                    from datetime import datetime
                    now = datetime.now().isoformat()

                    # FLO-specific status
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

                    # Tavern-compatible training_status.json
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
        # Log VRAM every 20 steps
        if state.global_step % 20 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  [VRAM] {allocated:.1f}GB / {reserved:.1f}GB reserved")


class MuonTrainer(Trainer):
    """
    Custom Trainer that supports Muon optimizer.

    Overrides create_optimizer to use Muon+AdamW hybrid instead of default optimizer.
    """

    def __init__(self, muon_config: dict = None, **kwargs):
        self.muon_config = muon_config or {}
        super().__init__(**kwargs)

    def create_optimizer(self):
        """Create Muon+AdamW hybrid optimizer."""
        if self.optimizer is not None:
            return self.optimizer

        # Get Muon config
        hidden_lr = self.muon_config.get("hidden_lr", 0.02)
        aux_lr = self.muon_config.get("aux_lr", 3e-4)
        momentum = self.muon_config.get("momentum", 0.95)
        weight_decay = self.muon_config.get("weight_decay", 0.0)
        use_8bit = self.muon_config.get("use_8bit_adam", True)  # Default to 8-bit for memory savings

        # Split parameters
        param_groups = split_transformer_params(
            self.model,
            hidden_lr=hidden_lr,
            aux_lr=aux_lr,
            hidden_momentum=momentum,
            weight_decay=weight_decay,
        )

        # Create Muon optimizer with 8-bit AdamW for aux params
        self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups, use_8bit_adam=use_8bit)

        print(f"\nMuon Optimizer Created:")
        print(f"  Hidden LR (Muon): {hidden_lr}")
        print(f"  Aux LR (AdamW): {aux_lr}")
        print(f"  8-bit AdamW: {use_8bit}")
        print(f"  Momentum: {momentum}")

        return self.optimizer


def run_muon_training(config: dict, data_path: Path, dry_run: bool = False):
    """Run training with Muon+AdamW hybrid optimizer."""
    base_dir = get_base_dir()
    model_path = base_dir / config["model_path"]
    output_dir = base_dir / config["current_model_dir"]

    # Get optimizer settings from config or use defaults
    opt_config = config.get("optimizer", {})
    muon_config = {
        "hidden_lr": opt_config.get("muon", {}).get("hidden_lr", 0.02),
        "aux_lr": opt_config.get("muon", {}).get("aux_lr", 3e-4),
        "momentum": opt_config.get("muon", {}).get("momentum", 0.95),
        "weight_decay": opt_config.get("muon", {}).get("weight_decay", 0.0),
    }

    print("\n" + "=" * 60)
    print(f"FLO TRAINING - {config.get('hero_name', 'FLO')} with MUON")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Data: {data_path}")
    print(f"Optimizer: Muon (hidden) + AdamW (aux)")
    print(f"  Hidden LR: {muon_config['hidden_lr']}")
    print(f"  Aux LR: {muon_config['aux_lr']}")
    print(f"Batch: {config.get('batch_size', 1)} x {config.get('gradient_accumulation', 32)}")
    print(f"Max length: {config.get('max_length', 1024)}")

    if dry_run:
        print("\n[DRY RUN] Would train with above config")
        return

    # Step 1: Apply Liger kernel BEFORE loading model
    if LIGER_AVAILABLE and config.get("liger", {}).get("enabled", True):
        print("\n" + "=" * 60)
        print("LIGER KERNEL - Fused Operations")
        print("=" * 60)
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )
        print("  Fused Linear CrossEntropy")
        print("  Fused RMSNorm + SwiGLU + RoPE")
        print("=" * 60)

    # Step 2: Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Load and process training data
    print(f"\nLoading training data from {data_path}...")
    raw_samples = load_training_data(data_path)
    print(f"  Loaded {len(raw_samples)} samples")

    max_length = config.get("max_length", 1024)

    def tokenize_messages(example):
        """Convert messages format to tokenized input."""
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

    # Step 4: Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing enabled")

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,} ({params/1e9:.2f}B)")
    print(f"  Trainable: {trainable:,}")

    # Show param split preview
    print_param_group_summary(model)

    # Step 5: Training arguments (no DeepSpeed)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 32),
        num_train_epochs=config.get("epochs", 1),
        learning_rate=muon_config["hidden_lr"],  # Main LR (Muon handles its own)
        bf16=True,
        logging_steps=config.get("log_steps", 10),
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        warmup_steps=config.get("warmup_steps", 100),
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=True,
        # No DeepSpeed - using custom Muon optimizer
        optim="adamw_torch",  # Placeholder, MuonTrainer overrides this
    )

    # Step 6: Create MuonTrainer
    status_file = base_dir / "status" / "flo_training.json"
    status_file.parent.mkdir(exist_ok=True)
    trainer = MuonTrainer(
        muon_config=muon_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[TrainingProgressCallback(status_file)],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Step 7: Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING - MUON + ADAM")
    print("=" * 60)
    print("  Muon: Hidden weight matrices (attention projections, MLP)")
    print("  AdamW: Embeddings, output head, layer norms, biases")
    print("=" * 60 + "\n")

    torch.cuda.reset_peak_memory_stats()

    try:
        trainer.train()

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nTraining complete!")
        print(f"Peak VRAM: {peak_mem:.2f} GB")

        # Save final model
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))

    except torch.cuda.OutOfMemoryError as e:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nOOM! Peak VRAM was {peak_mem:.2f} GB")
        print("Try: --deepspeed for ZeRO-3 CPU offload (but uses AdamW instead of Muon)")
        raise


def run_galore_training(config: dict, data_path: Path, dry_run: bool = False):
    """Run training with GaLore + 8-bit AdamW (memory-efficient for 4B+)."""
    if not GALORE_AVAILABLE:
        print("ERROR: GaLore not available. Install with: pip install galore-torch")
        return

    base_dir = get_base_dir()
    model_path = base_dir / config["model_path"]
    output_dir = base_dir / config["current_model_dir"]

    # Get optimizer settings
    opt_config = config.get("optimizer", {})
    galore_config = opt_config.get("galore", {})
    rank = galore_config.get("rank", 256)
    hidden_lr = opt_config.get("muon", {}).get("hidden_lr", 0.02)
    aux_lr = opt_config.get("muon", {}).get("aux_lr", 3e-4)

    print("\n" + "=" * 60)
    print(f"FLO TRAINING - {config.get('hero_name', 'FLO')} with GALORE")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Data: {data_path}")
    print(f"Optimizer: GaLore + 8-bit AdamW")
    print(f"  GaLore Rank: {rank}")
    print(f"  Hidden LR: {hidden_lr}")
    print(f"  Aux LR: {aux_lr}")
    print(f"Batch: {config.get('batch_size', 1)} x {config.get('gradient_accumulation', 32)}")
    print(f"Max length: {config.get('max_length', 1024)}")

    if dry_run:
        print("\n[DRY RUN] Would train with above config")
        return

    # Step 1: Apply Liger kernel
    if LIGER_AVAILABLE and config.get("liger", {}).get("enabled", True):
        print("\n" + "=" * 60)
        print("LIGER KERNEL - Fused Operations")
        print("=" * 60)
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )
        print("  Fused Linear CrossEntropy")
        print("  Fused RMSNorm + SwiGLU + RoPE")
        print("=" * 60)

    # Step 2: Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Load and process training data
    print(f"\nLoading training data from {data_path}...")
    raw_samples = load_training_data(data_path)
    print(f"  Loaded {len(raw_samples)} samples")

    max_length = config.get("max_length", 1024)

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

    # Step 4: Load model
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,} ({params/1e9:.2f}B)")
    print(f"  Trainable: {trainable:,}")

    # Step 5: Create GaLore optimizer
    print("\nCreating GaLore optimizer...")
    optimizer = create_galore_muon_optimizer(
        model,
        rank=rank,
        hidden_lr=hidden_lr,
        aux_lr=aux_lr,
    )

    # Step 6: Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 32),
        num_train_epochs=config.get("epochs", 1),
        learning_rate=hidden_lr,
        bf16=True,
        logging_steps=config.get("log_steps", 10),
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        warmup_steps=config.get("warmup_steps", 100),
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # Step 7: Create trainer with custom optimizer
    status_file = base_dir / "status" / "flo_training.json"
    status_file.parent.mkdir(exist_ok=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(optimizer, None),  # Pass custom optimizer, no scheduler
        callbacks=[TrainingProgressCallback(status_file)],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Step 8: Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING - GALORE + 8-bit AdamW")
    print("=" * 60)
    print(f"  GaLore rank: {rank} (reduces optimizer memory ~8x)")
    print("  8-bit AdamW for embeddings, heads, norms")
    print("=" * 60 + "\n")

    torch.cuda.reset_peak_memory_stats()

    try:
        trainer.train()

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nTraining complete!")
        print(f"Peak VRAM: {peak_mem:.2f} GB")

        print(f"\nSaving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))

    except torch.cuda.OutOfMemoryError as e:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nOOM! Peak VRAM was {peak_mem:.2f} GB")
        print("Try reducing --rank or --batch-size")
        raise


def run_deepspeed_training(config: dict, data_path: Path, dry_run: bool = False):
    """Run training with DeepSpeed ZeRO-3 (original behavior)."""
    base_dir = get_base_dir()
    model_path = base_dir / config["model_path"]
    output_dir = base_dir / config["current_model_dir"]
    deepspeed_config = base_dir / "configs" / "deepspeed" / "zero3_offload.json"

    print("\n" + "=" * 60)
    print(f"FLO TRAINING - {config.get('hero_name', 'FLO')} with DeepSpeed")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Data: {data_path}")
    print(f"DeepSpeed: ZeRO-3 with CPU Offload")
    print(f"Batch: {config.get('batch_size', 1)} x {config.get('gradient_accumulation', 32)}")
    print(f"Max length: {config.get('max_length', 1024)}")

    if dry_run:
        print("\n[DRY RUN] Would train with above config")
        return

    # Step 1: Apply Liger kernel BEFORE loading model
    if LIGER_AVAILABLE and config.get("liger", {}).get("enabled", True):
        print("\n" + "=" * 60)
        print("LIGER KERNEL - Fused Operations")
        print("=" * 60)
        apply_liger_kernel_to_qwen2(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )
        print("  Fused Linear CrossEntropy")
        print("  Fused RMSNorm + SwiGLU + RoPE")
        print("=" * 60)

    # Step 2: Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Load and process training data
    print(f"\nLoading training data from {data_path}...")
    raw_samples = load_training_data(data_path)
    print(f"  Loaded {len(raw_samples)} samples")

    max_length = config.get("max_length", 1024)

    def tokenize_messages(example):
        """Convert messages format to tokenized input."""
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

    # Step 4: Load model with DeepSpeed ZeRO-3 context
    print("\nLoading model with DeepSpeed ZeRO-3 Init...")
    import deepspeed
    from transformers.integrations import HfDeepSpeedConfig

    dschf = HfDeepSpeedConfig(str(deepspeed_config))

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing enabled")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e9:.2f}B)")

    # Step 5: Training arguments with DeepSpeed
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 32),
        num_train_epochs=config.get("epochs", 1),
        learning_rate=config.get("learning_rate", 2e-5),
        bf16=True,
        logging_steps=config.get("log_steps", 10),
        save_strategy="steps",
        save_steps=config.get("save_steps", 500),
        warmup_steps=config.get("warmup_steps", 100),
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed=str(deepspeed_config),
    )

    # Step 6: Create trainer
    status_file = base_dir / "status" / "flo_training.json"
    status_file.parent.mkdir(exist_ok=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[TrainingProgressCallback(status_file)],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Step 7: Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING - DeepSpeed ZeRO-3")
    print("=" * 60)
    print("  CPU offload enabled for optimizer states and parameters")
    print("  Your 64GB RAM will be used for offloading")
    print("=" * 60 + "\n")

    torch.cuda.reset_peak_memory_stats()

    try:
        trainer.train()

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nTraining complete!")
        print(f"Peak VRAM: {peak_mem:.2f} GB")

        # Save final model
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))

    except torch.cuda.OutOfMemoryError as e:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nOOM! Peak VRAM was {peak_mem:.2f} GB")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train FLO on SY+BIN skills with Muon+Adam or DeepSpeed")
    parser.add_argument("--generate", action="store_true", help="Generate training data from skill servers")
    parser.add_argument("--count", type=int, default=1000, help="Samples per skill (default: 1000)")
    parser.add_argument("--levels", type=str, default="1,2,3", help="Skill levels to use (default: 1,2,3)")
    parser.add_argument("--data", type=str, help="Use existing training data file")
    parser.add_argument("--campaign", type=str, default="campaigns/titan-qwen3-4b/campaign-001",
                        help="Campaign directory")
    parser.add_argument("--dry-run", action="store_true", help="Show config without training")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Use DeepSpeed ZeRO-3 with CPU offload (uses AdamW, not Muon)")
    parser.add_argument("--galore", action="store_true",
                        help="Use GaLore + 8-bit AdamW (recommended for 4B+ on 24GB)")
    parser.add_argument("--rank", type=int, default=256,
                        help="GaLore rank (default: 256, lower = less memory)")
    # DeepSpeed launcher adds this
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    args = parser.parse_args()

    base_dir = get_base_dir()
    campaign_path = base_dir / args.campaign

    # Load campaign config
    print("Loading campaign config...")
    config = load_campaign_config(campaign_path)
    print(f"  Hero: {config.get('hero_name', 'FLO')}")
    print(f"  Model: {config.get('model_display_name', 'Unknown')}")

    # Determine data path
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = base_dir / data_path
    else:
        data_dir = campaign_path / "data"
        data_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = data_dir / f"training_{timestamp}.jsonl"

        if args.generate or not list(data_dir.glob("training_*.jsonl")):
            print("\nGenerating training data...")
            skills = config.get("curriculum", {}).get("skills", ["bin", "sy"])
            levels = [int(l) for l in args.levels.split(",")]

            samples = generate_mixed_data(
                skills=skills,
                count_per_skill=args.count,
                levels=levels
            )

            if not samples:
                print("ERROR: No samples generated. Are skill servers running?")
                print("  Check: curl http://localhost:8080/health")
                print("  Check: curl http://localhost:8090/health")
                sys.exit(1)

            save_training_data(samples, data_path)
        else:
            # Find most recent data file
            data_files = list(data_dir.glob("training_*.jsonl"))
            if data_files:
                data_path = max(data_files, key=lambda p: p.stat().st_mtime)
                print(f"Using existing data: {data_path}")
            else:
                print("No data found. Use --generate to create training data.")
                sys.exit(1)

    # Inject rank override if provided
    if args.rank != 256:
        if "optimizer" not in config:
            config["optimizer"] = {}
        if "galore" not in config["optimizer"]:
            config["optimizer"]["galore"] = {}
        config["optimizer"]["galore"]["rank"] = args.rank

    # Run training
    if args.deepspeed:
        run_deepspeed_training(config, data_path, dry_run=args.dry_run)
    elif args.galore:
        run_galore_training(config, data_path, dry_run=args.dry_run)
    else:
        # Default: GaLore (best for 4B on 24GB)
        run_galore_training(config, data_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
