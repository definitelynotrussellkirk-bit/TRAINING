# Continuous Training Implementation Guide

**Date:** 2025-11-11
**Status:** Complete Reference Implementation

This document explains the PROPER way to implement continuous training with HuggingFace Trainer + PEFT/LoRA. The key insight: **Stop fighting Trainer's built-in checkpoint system**.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [What Needs to Persist](#what-needs-to-persist)
3. [How Trainer Checkpoints Work](#how-trainer-checkpoints-work)
4. [The Common Mistakes](#the-common-mistakes)
5. [The Correct Implementation](#the-correct-implementation)
6. [Directory Structure](#directory-structure)
7. [Verification Tests](#verification-tests)
8. [Troubleshooting](#troubleshooting)

---

## Core Principles

### Principle 1: Let Trainer Own Checkpoints

HuggingFace Trainer is **designed** for continuous training. It has built-in support for:
- Saving checkpoints at regular intervals
- Resuming from checkpoints seamlessly
- Managing checkpoint cleanup automatically
- Preserving optimizer state, scheduler state, and global_step

**DO NOT** try to manually manage these checkpoints. Trainer knows what it's doing.

### Principle 2: Checkpoints ≠ Final Model Saves

- **Checkpoints** = Full training state (model + optimizer + scheduler + global_step)
- **Final saves** = Just the trained weights (adapter only)

Checkpoints are for **resuming training**. Final saves are for **inference/deployment**.

### Principle 3: global_step Must Never Reset

The `global_step` counter tracks total training progress across ALL batches. It must:
- ✅ Increment continuously: 0 → 118 → 236 → 354 → ...
- ❌ NEVER reset between batches
- ✅ Be preserved in `trainer_state.json` inside checkpoints
- ✅ Resume from the last checkpoint's global_step

If global_step resets, optimizer momentum and learning rate schedule break.

---

## What Needs to Persist

Between training batches, these must be preserved:

| Component | Stored In | Why Critical |
|-----------|-----------|--------------|
| **LoRA adapter weights** | `adapter_model.safetensors` | The learned knowledge |
| **Optimizer state** | `optimizer.pt` | Momentum, adaptive LR history |
| **LR scheduler state** | `scheduler.pt` | Learning rate schedule position |
| **Global step counter** | `trainer_state.json` | Total training progress |
| **RNG state** | `rng_state*.pth` | Reproducibility (optional) |

All of these are automatically saved by Trainer in checkpoint directories.

---

## How Trainer Checkpoints Work

### Checkpoint Structure

```
current_model/
├── checkpoint-100/              ← First checkpoint (step 100)
│   ├── adapter_model.safetensors  ← LoRA weights at step 100
│   ├── optimizer.pt                ← Optimizer state (momentum, etc.)
│   ├── scheduler.pt                ← LR scheduler state
│   ├── trainer_state.json          ← Contains global_step=100
│   ├── rng_state_0.pth            ← RNG state for GPU 0
│   └── training_args.bin          ← Training config
├── checkpoint-200/              ← Second checkpoint (step 200)
│   └── ... (same files)
└── checkpoint-300/              ← Latest checkpoint
    └── ... (same files)
```

### What Trainer Does Automatically

**During Training:**
1. Every `save_steps` steps, Trainer creates a new `checkpoint-N/` directory
2. Saves all state files listed above
3. If `save_total_limit=3`, deletes oldest checkpoints (keeps only last 3)

**On Resume:**
1. You call `trainer.train(resume_from_checkpoint="checkpoint-300")`
2. Trainer loads: adapter weights + optimizer + scheduler + global_step
3. Training continues from step 300 (not step 0!)
4. Next checkpoint will be at step 400 (if save_steps=100)

**The Magic:** You don't manage any of this. Just point Trainer at the latest checkpoint.

---

## The Common Mistakes

### ❌ Mistake 1: Manual Checkpoint Deletion

```python
# BAD: Fighting Trainer's cleanup
for old_checkpoint in old_checkpoints:
    shutil.rmtree(old_checkpoint)  # Breaks Trainer expectations!
```

**Why bad:** Trainer expects to manage its own checkpoints via `save_total_limit`. Manual deletion can delete checkpoints Trainer is still tracking.

**Solution:** Use `save_total_limit` in TrainingArguments. That's what it's for.

### ❌ Mistake 2: Calling save_model() for Checkpointing

```python
# BAD: This only saves weights, not optimizer state!
trainer.train()
trainer.save_model(output_dir)  # Missing optimizer.pt, scheduler.pt!
```

**Why bad:** `save_model()` saves only adapter weights. If you resume from this, optimizer state is lost (momentum resets, LR schedule breaks).

**Solution:** Let `save_strategy="steps"` handle checkpoints. Use `save_model()` only for final deployment saves.

### ❌ Mistake 3: Resetting global_step

```python
# BAD: Resetting the step counter
trainer_state["global_step"] = 0  # BREAKS EVERYTHING!
```

**Why bad:** Optimizer momentum and LR schedule depend on global_step. Resetting it causes:
- LR warmup to restart (wrong learning rate)
- Momentum to reset (unstable gradients)
- Loss to spike or plateau

**Solution:** NEVER touch global_step. Trainer manages it automatically.

### ❌ Mistake 4: Copying Checkpoint Files to Root

```python
# BAD: Duplicating state files
shutil.copy("checkpoint-300/optimizer.pt", "current_model/optimizer.pt")
```

**Why bad:** Trainer doesn't look for state files in the root directory. They must be in `checkpoint-N/` folders. Copying them wastes space and creates confusion.

**Solution:** Leave checkpoints where Trainer creates them.

---

## The Correct Implementation

### Step 1: TrainingArguments (Set and Forget)

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="current_model",

    # Checkpoint management (Trainer owns this!)
    save_strategy="steps",
    save_steps=100,              # Checkpoint every 100 steps
    save_total_limit=3,          # Keep only last 3 checkpoints (auto-cleanup!)
    save_safetensors=True,       # Use safetensors format

    # Training params
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=100,

    # Logging
    logging_steps=10,
    report_to="none",

    # Performance
    fp16=True,                   # Or bf16 if supported
    dataloader_num_workers=4,
)
```

**Key points:**
- `save_steps=100` must be less than steps per batch (e.g., 118) to ensure checkpoints exist between batches
- `save_total_limit=3` keeps only the last 3 checkpoints (auto-deletes old ones)
- Don't set `resume_from_checkpoint` here (pass it to `.train()` instead)

### Step 2: Finding Latest Checkpoint

```python
from pathlib import Path

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output_dir"""
    output_path = Path(output_dir)

    if not output_path.exists():
        return None

    # Find all checkpoint-* directories
    checkpoints = sorted(
        output_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])  # Sort by step number
    )

    if not checkpoints:
        return None

    latest = checkpoints[-1]
    print(f"Found latest checkpoint: {latest.name}")
    return str(latest)
```

### Step 3: Training with Resume

```python
from transformers import Trainer
from peft import LoraConfig, get_peft_model

# Setup model with LoRA (do this once at start)
model = AutoModelForCausalLM.from_pretrained(base_model_path, ...)
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Find latest checkpoint
resume_checkpoint = get_latest_checkpoint(training_args.output_dir)

# Train (with resume if checkpoint exists)
if resume_checkpoint:
    print(f"Resuming from {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
else:
    print("Starting fresh training")
    trainer.train()

# Optional: Save final adapter for deployment (not for resumption!)
trainer.save_model(training_args.output_dir)
```

**What happens:**
1. **First batch:** No checkpoint exists → trains from step 0 → saves checkpoints at 100, 200, etc.
2. **Second batch:** Latest checkpoint is `checkpoint-200` → resumes from step 200 → continues to 318, etc.
3. **Third batch:** Latest checkpoint is `checkpoint-400` → resumes from step 400 → continues...

**The beauty:** You don't track global_step. Trainer does it automatically.

### Step 4: Daemon Loop (Continuous Training)

```python
def train_batch(data_file, output_dir, base_model_path):
    """Train on one batch of data, automatically resuming from latest checkpoint"""

    # Load dataset
    dataset = load_dataset_from_jsonl(data_file)

    # Setup model (Trainer will load checkpoint if it exists)
    model = setup_model_with_lora(base_model_path)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # Auto-cleanup old checkpoints!
        num_train_epochs=1,
        # ... other args
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Find and resume from latest checkpoint
    resume_checkpoint = get_latest_checkpoint(output_dir)

    # Train
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # Done! Trainer has saved checkpoints automatically.
    # DO NOT manually delete checkpoints - save_total_limit handles it!

# Daemon loop
while True:
    data_files = scan_inbox()
    for data_file in data_files:
        train_batch(data_file, "current_model", "base_model_path")
        delete_data_file(data_file)  # Only delete data after successful training
    time.sleep(poll_interval)
```

**Key points:**
- Each batch automatically resumes from the latest checkpoint
- No manual checkpoint management needed
- `save_total_limit=3` prevents disk bloat
- Global step increments continuously across all batches

---

## Directory Structure

### During Training (Rolling Checkpoints)

```
/path/to/training/
├── current_model/                    ← Active training directory
│   ├── checkpoint-100/               ← Oldest checkpoint (will be deleted when checkpoint-400 is created)
│   ├── checkpoint-200/
│   ├── checkpoint-300/               ← Latest checkpoint
│   ├── adapter_model.safetensors    ← Final save (optional, for deployment)
│   ├── adapter_config.json
│   └── trainer_state.json            ← May exist at root, but checkpoint-* is source of truth
├── snapshots/                        ← Daily backups
│   ├── 2025-11-10/
│   │   └── checkpoint-25600/         ← Snapshot of checkpoint at end of day
│   └── 2025-11-11/
│       └── checkpoint-38400/
├── inbox/                            ← Data drops
│   └── batch_001.jsonl
└── config.json                       ← Training config
```

### After Training Completes

```
current_model/
├── checkpoint-300/               ← Latest checkpoint (for resumption)
├── adapter_model.safetensors    ← Final deployment save (weights only)
└── adapter_config.json
```

**Important:**
- Keep `checkpoint-*` directories for resuming training
- `adapter_model.safetensors` at root is optional (for inference/deployment)
- Daily snapshots preserve full checkpoints for backup/audit

---

## Verification Tests

### Test 1: Verify global_step Increments

```bash
# After first batch
cat current_model/checkpoint-100/trainer_state.json | jq '.global_step'
# Expected: 100

# After second batch (should NOT reset!)
cat current_model/checkpoint-200/trainer_state.json | jq '.global_step'
# Expected: 200 (not 100!)

# After third batch
cat current_model/checkpoint-300/trainer_state.json | jq '.global_step'
# Expected: 300+
```

**If global_step resets to 0 or low numbers between batches, your checkpoint system is broken.**

### Test 2: Verify Checkpoint Contents

```bash
# Check latest checkpoint has all required files
ls -lh current_model/checkpoint-*/
# Must see: adapter_model.safetensors, optimizer.pt, scheduler.pt, trainer_state.json
```

**If `optimizer.pt` or `scheduler.pt` are missing, resumption will be incomplete.**

### Test 3: Verify Automatic Cleanup

```bash
# Start training with save_total_limit=3
# After checkpoint-400 is created, checkpoint-100 should be deleted
ls current_model/checkpoint-*/
# Should see: checkpoint-200, checkpoint-300, checkpoint-400 (only 3!)
```

**If old checkpoints accumulate, `save_total_limit` is not working.**

### Test 4: Verify Loss Continuity

```python
# Extract loss from logs across batches
# Batch 1: loss should decrease from ~2.0 → ~1.5
# Batch 2: loss should continue from ~1.5 → ~1.2 (NOT jump back to 2.0!)
# Batch 3: loss should continue from ~1.2 → ~1.0

# If loss resets between batches, optimizer state is not being preserved
```

**Smooth loss decrease across batches confirms proper checkpoint resumption.**

---

## Troubleshooting

### Problem: Loss spikes or resets between batches

**Symptoms:**
- Batch 1: loss 2.0 → 1.5
- Batch 2: loss 2.0 → 1.6 (restarted from high loss!)

**Cause:** Optimizer state not preserved (missing `optimizer.pt` or not resuming correctly)

**Fix:**
1. Check that `checkpoint-*/optimizer.pt` exists
2. Verify `resume_from_checkpoint` is passed to `trainer.train()`
3. Check that global_step is not being reset

### Problem: global_step stuck or resetting

**Symptoms:**
- Checkpoint names don't increase across batches
- `trainer_state.json` shows global_step=0 or low numbers after multiple batches

**Cause:** Code is manually resetting global_step or not resuming from checkpoints

**Fix:**
1. Remove any code that modifies `trainer_state.json`
2. Ensure `resume_from_checkpoint` points to latest checkpoint
3. Don't manually create checkpoint directories

### Problem: Checkpoints accumulating (disk full)

**Symptoms:**
- 50+ checkpoint directories in `current_model/`
- Disk usage growing rapidly

**Cause:** `save_total_limit` not set or manual checkpoint management interfering

**Fix:**
1. Add `save_total_limit=3` to TrainingArguments
2. Remove any manual checkpoint deletion code
3. Let Trainer handle cleanup automatically

### Problem: "Checkpoint not found" errors

**Symptoms:**
- Error: `checkpoint-300 does not exist`
- Training starts from step 0 unexpectedly

**Cause:** Checkpoints were manually deleted or moved

**Fix:**
1. Stop manually deleting checkpoints
2. If you need to clean up, delete the entire `current_model/` directory (start fresh)
3. Use `save_total_limit` to control checkpoint count, not manual deletion

---

## Summary: The Golden Rules

1. **Use `save_total_limit`** in TrainingArguments to control checkpoint count
2. **Pass `resume_from_checkpoint`** to `trainer.train()` with latest checkpoint path
3. **Never reset global_step** or modify `trainer_state.json`
4. **Never manually delete checkpoints** while training is active
5. **Let Trainer save checkpoints** via `save_strategy="steps"`
6. **Don't use `save_model()` for checkpointing** (it's for final deployment saves)

**If you follow these rules, continuous training will work flawlessly.**

---

## Additional Resources

- HuggingFace Trainer docs: https://huggingface.co/docs/transformers/main_classes/trainer
- PEFT docs: https://huggingface.co/docs/peft
- TrainingArguments API: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
