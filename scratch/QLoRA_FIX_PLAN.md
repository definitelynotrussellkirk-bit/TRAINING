# Full Model Training Fix Plan - 2025-11-24

## Problem Summary
Currently calling it "QLoRA" but not actually using LoRA adapters. Just doing 4-bit quantized full-model training (or full-precision full-model). This causes:
- Snapshot verification fails (looks for adapter files that don't exist)
- Consolidation always skips (expects adapter files)
- Confusing naming throughout codebase

## Changes Needed

### ✅ COMPLETED
1. **schema.py defaults fixed** - Safe defaults for 24GB GPU
   - batch_size: 1, gradient_accumulation: 16, max_length: 2048
   - use_gradient_checkpointing: True, fp_precision: "bf16"

2. **config.json updated** - gradient_accumulation: 16

### 🔧 IN PROGRESS
3. **Daemon snapshot verification** (core/training_daemon.py ~line 1250)
   - Current: Looks for `adapter_model.safetensors`, `adapter_config.json`
   - Fix: Look for HF checkpoint files: `config.json`, `model.safetensors`, tokenizer files

### 📋 TODO
4. **Rename use_qlora → load_in_4bit** throughout:
   - trainer/config/schema.py
   - core/train.py (multiple references + comments)
   - core/training_daemon.py (default config + validation)
   - config.json

5. **Remove LoRA config fields** from daemon default config:
   - lora_r, lora_alpha references

6. **Update consolidation logic** (training_daemon.py ~line 1450)
   - Either simplify for full-model or disable

## Files to Modify
- ✅ trainer/config/schema.py
- ✅ config.json
- 🔧 core/training_daemon.py (snapshot + consolidation + use_qlora → load_in_4bit)
- ⏳ core/train.py (use_qlora → load_in_4bit)

## Testing Plan
1. Verify snapshot verification works with HF checkpoints
2. Verify training starts without errors
3. Verify no OOM with new defaults
4. Commit changes

---
Generated: 2025-11-24
Status: In Progress
