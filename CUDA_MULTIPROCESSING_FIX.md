# CUDA Multiprocessing Fix Documentation

**Date:** 2025-11-16
**Issue:** Training crashes with CUDA multiprocessing errors
**Status:** FIXED ✅

## Problem Description

### Symptoms
Training kept crashing repeatedly with the following error:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with
multiprocessing, you must use the 'spawn' start method
```

### Root Cause
The HuggingFace `datasets` library's `.map()` function was attempting to use forked
subprocesses for parallel processing, even with `num_proc=1`. This conflicted with
CUDA, which was already initialized in the parent process when loading the model.

When CUDA is initialized in a process and that process is then forked (the default
multiprocessing method on Linux), the forked child process inherits the CUDA context
but cannot properly re-initialize it, leading to crashes.

## Solution

### The Fix
Changed the `num_proc` parameter in the dataset tokenization from `1` to `None`
to completely disable multiprocessing in the `datasets.map()` call.

**File:** `/path/to/training/train.py`
**Line:** 414

**Before:**
```python
tokenized_dataset = self.train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    num_proc=1,  # ❌ Still uses multiprocessing internally
    load_from_cache_file=False,
    writer_batch_size=1000,
    desc="Tokenizing dataset"
)
```

**After:**
```python
# MEMORY FIX: Add parameters to prevent memory explosion
# CUDA FIX: num_proc=None completely disables multiprocessing to avoid CUDA fork errors
tokenized_dataset = self.train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=10,  # Reduced to avoid memory issues
    remove_columns=["text"],
    num_proc=None,  # ✅ Completely disable multiprocessing
    load_from_cache_file=False,
    writer_batch_size=10,
    desc="Tokenizing dataset"
)
```

### Additional Changes
Also added multiprocessing configuration at the top of `train.py` (lines 30-35),
though this ended up not being necessary with `num_proc=None`:

```python
# Fix CUDA multiprocessing issue - MUST be before torch import
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch
```

## Why This Works

### Understanding num_proc Values

1. **`num_proc=N` (where N > 0):**
   - Uses N worker processes via multiprocessing
   - On Linux, defaults to `fork` start method
   - FAILS with CUDA because forked processes inherit CUDA context

2. **`num_proc=1`:**
   - Still uses multiprocessing pool with 1 worker
   - Worker is created via fork, inheriting CUDA context
   - STILL FAILS for the same reason

3. **`num_proc=None`:**
   - Completely disables multiprocessing
   - Processes data sequentially in the main process
   - Works perfectly because no forking occurs

### Performance Impact
- **Tokenization time:** Slightly slower due to sequential processing
- **Training time:** No impact - training itself doesn't use multiprocessing
- **Overall impact:** Negligible for small datasets, acceptable tradeoff for stability

## Verification

### Test Results
After applying the fix:
- ✅ Training completes successfully without crashes
- ✅ Tokenization proceeds sequentially without errors
- ✅ Evolution tracking captures snapshots correctly
- ✅ No CUDA initialization errors

### Example Success Log
```
Tokenizing dataset: 100%|██████████| 95/95 [00:00<00:00, 4312.98 examples/s]
2025-11-16 03:43:48,899 [INFO] Training completed in 0.6 minutes
2025-11-16 03:43:48,899 [INFO] ✅ Training successful
```

## Future-Proofing

### Considerations
1. **If using very large datasets (>100k examples):**
   - May want to investigate using `num_proc` with `spawn` start method
   - Would require ensuring CUDA is initialized only in child processes
   - Current solution is simpler and more reliable

2. **Alternative approaches (not recommended):**
   - Setting `CUDA_VISIBLE_DEVICES` differently in each worker
   - Lazy CUDA initialization after fork
   - Using `torch.multiprocessing` with start method='spawn'
   - All add complexity without clear benefits for this use case

3. **If this issue returns:**
   - Check for other `.map()` calls in the codebase
   - Ensure all use `num_proc=None`
   - Verify no other multiprocessing is happening after CUDA init

## Related Issues
- Memory optimization (reduced batch_size to 10)
- Evolution tracking integration (works correctly with fixed training)

## References
- PyTorch CUDA + multiprocessing: https://pytorch.org/docs/stable/notes/multiprocessing.html
- HuggingFace datasets multiprocessing: https://huggingface.co/docs/datasets/process
- Python multiprocessing start methods: https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

---

**Summary:** Changing `num_proc=1` to `num_proc=None` completely disables multiprocessing
in dataset tokenization, eliminating CUDA fork errors and ensuring stable training.
