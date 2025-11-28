#!/usr/bin/env python3
"""
Full Fine-Tune Qwen3-4B with Paged 8-bit Adam + Liger

VRAM Budget (24GB RTX 4090):
- Model (bf16): 8 GB
- Gradients (bf16): 8 GB
- Activations: ~3-5 GB (with checkpointing + Liger)
- Paged 8-bit Adam: Offloads to CPU when GPU full

Key optimizations:
1. Paged 8-bit Adam - Optimizer states offload to CPU on demand
2. Liger Kernel - Fused ops, no materialized logits
3. Gradient checkpointing - Trade compute for memory
4. bf16 precision - Half the memory
5. Batch size 1 + gradient accumulation
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import Dataset

# Import Liger BEFORE loading model
from liger_kernel.transformers import apply_liger_kernel_to_qwen2


def setup_liger():
    """Apply Liger kernel patches for memory efficiency."""
    print("=" * 60)
    print("LIGER KERNEL - Fused Operations")
    print("=" * 60)
    apply_liger_kernel_to_qwen2(
        rope=True,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )
    print("  ✓ Fused Linear CrossEntropy")
    print("  ✓ Fused RMSNorm + SwiGLU + RoPE")
    print("=" * 60 + "\n")


def load_model(model_path: str):
    """Load model with memory-optimal settings."""
    print(f"Loading model: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    print("  ✓ Gradient checkpointing enabled")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} ({params/1e9:.2f}B)")

    # Memory report
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after load: {allocated:.2f} GB")

    return model, tokenizer


class VRAMCallback(TrainerCallback):
    """Log VRAM usage during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [Step {state.global_step}] VRAM: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")


def create_dummy_dataset(tokenizer, num_samples=100, max_length=512):
    """Create a dummy dataset for testing."""
    samples = []
    for i in range(num_samples):
        text = f"""<|im_start|>user
What is {i} + {i}?<|im_end|>
<|im_start|>assistant
{i} + {i} = {i + i}<|im_end|>"""

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        samples.append({
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze(),
        })

    return Dataset.from_list(samples)


def main():
    # Config - paths relative to base_dir
    from core.paths import get_base_dir
    base_dir = get_base_dir()
    MODEL_PATH = str(base_dir / "models" / "Qwen3-4B-Instruct-2507")
    OUTPUT_DIR = str(base_dir / "outputs" / "qwen3-4b-8bit")
    MAX_LENGTH = 512
    BATCH_SIZE = 1
    GRAD_ACCUM = 32

    print("\n" + "=" * 60)
    print("FULL FINE-TUNE: Qwen3-4B with Paged 8-bit Adam")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Optimizer: Paged 8-bit Adam (offloads to CPU)")
    print("=" * 60 + "\n")

    # Step 1: Apply Liger patches BEFORE loading model
    setup_liger()

    # Step 2: Load model and tokenizer
    model, tokenizer = load_model(MODEL_PATH)

    # Step 3: Create dummy dataset
    print("Creating test dataset...")
    dataset = create_dummy_dataset(tokenizer, num_samples=100, max_length=MAX_LENGTH)
    print(f"  {len(dataset)} samples, max_length={MAX_LENGTH}")

    # Step 4: Training arguments with 8-bit Adam
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=True,
        # Paged 8-bit Adam - offloads optimizer states when GPU is full
        optim="paged_adamw_8bit",
    )

    # Step 5: Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[VRAMCallback()],
    )

    # Report VRAM before training
    torch.cuda.reset_peak_memory_stats()

    # Step 6: Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    try:
        trainer.train()

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n✅ Training complete!")
        print(f"Peak VRAM: {peak_mem:.2f} GB")

    except torch.cuda.OutOfMemoryError as e:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n❌ OOM! Peak VRAM was {peak_mem:.2f} GB")
        raise


if __name__ == "__main__":
    main()
