# Permanent Error Training Backup

**Created:** 2025-11-21
**Reason:** Preserve failed training files for analysis

## Files (1.1 GB total)

1. `syllo_autogen_20251120_232642_count100000.jsonl` (318M)
2. `syllo_autogen_20251120_233134_count100000.jsonl` (318M)
3. `syllo_autogen_20251121_102028_count100000.jsonl` (318M)
4. `training_samples.jsonl` (116M)

## Why These Failed

These files repeatedly hit OOM errors during training with various batch sizes:
- Originally failed at batch_size=40 (effective 80)
- Failed again at batch_size=36
- Failed again at batch_size=32

## Issue Analysis

The crashes were caused by:
1. `device_map="auto"` pre-allocating 90% of GPU for model
2. Leaving insufficient memory for training activations/gradients
3. max_length=4096 requiring more memory than shorter sequences

## Resolution

Fixed by:
- Disabled `device_map="auto"` in train.py:445
- Reduced batch_size to 28
- Now using dynamic memory allocation

**DO NOT DELETE THIS DIRECTORY** - Keep for future reference and debugging.
