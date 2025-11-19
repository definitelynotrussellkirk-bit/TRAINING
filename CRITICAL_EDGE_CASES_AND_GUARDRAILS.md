# CRITICAL EDGE CASES AND CODE GUARDRAILS

**Created:** 2025-11-16
**Purpose:** Prevent bugs like the `UnboundLocalError` and `TypeError` that just occurred
**Priority:** CRITICAL - These bugs cost hours of training time

---

## 🚨 ROOT CAUSE ANALYSIS

### Bug 1: UnboundLocalError (`total_steps`)
**What happened:** Variable `total_steps` was used at line 515 before being calculated at line 839.

**Why it happened:**
1. Code was refactored/moved without checking dependencies
2. No static analysis or linting to catch undefined variables
3. No tests to verify the training initialization flow
4. Variable calculation was buried deep in function (839 lines in)

### Bug 2: TypeError (`NoneType > 0`)
**What happened:** Set `num_train_epochs=None`, but HuggingFace Trainer compared `None > 0`.

**Why it happened:**
1. Didn't check HuggingFace Trainer source code requirements
2. Assumed `None` would be acceptable for an optional parameter
3. No integration test with actual Trainer initialization
4. API contract violation (Trainer expects int or no parameter, not `None`)

---

## 🛡️ GUARDRAILS TO PREVENT FUTURE BUGS

### 1. STATIC ANALYSIS (Add to CI/CD)

```bash
# Add to pre-commit hook or CI pipeline
pylint train.py training_daemon.py --disable=C,R  # Errors and warnings only
mypy train.py training_daemon.py --ignore-missing-imports
```

**What this catches:**
- ✅ Undefined variables (like `total_steps`)
- ✅ Type errors (like `None > 0`)
- ✅ Unreachable code
- ✅ Missing imports

### 2. CODE ORDERING VALIDATION

Add this assertion RIGHT AFTER variable calculations:

```python
# train.py line ~528 (after total_steps calculation)
assert 'total_steps' in locals(), "CRITICAL: total_steps must be defined before TrainingArguments!"
assert isinstance(total_steps, int), f"total_steps must be int, got {type(total_steps)}"
assert total_steps > 0, f"total_steps must be > 0, got {total_steps}"
```

**Why:** Fails FAST with clear error instead of cryptic exception later.

### 3. PARAMETER VALIDATION

Add validation for ALL TrainingArguments parameters:

```python
# Before creating TrainingArguments
def validate_training_args(total_steps, batch_size, lr, warmup_steps, etc):
    assert isinstance(total_steps, int) and total_steps > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(lr, float) and 0 < lr < 1
    assert isinstance(warmup_steps, int) and warmup_steps >= 0
    # ... etc for all params

validate_training_args(total_steps, self.args.batch_size, self.args.learning_rate, ...)
```

### 4. INTEGRATION SMOKE TEST

Create `test_training_init.py`:

```python
def test_trainer_initialization():
    """Smoke test: can we create a Trainer without errors?"""
    # Mock minimal setup
    model, tokenizer, dataset = create_minimal_test_setup()

    # This should NOT raise any exceptions
    trainer = UltimateTrainer(args=test_args)
    trainer.load_model()
    # Don't actually train, just verify initialization works

    assert trainer.model is not None
    assert trainer.tokenizer is not None
```

**Run before every commit:** `pytest test_training_init.py -v`

### 5. GPU MEMORY GUARDRAILS

**Problem:** GPU OOM errors when daemon doesn't clean up memory between files.

**Solution:**

```python
# In training_daemon.py, after each training file completes:
def cleanup_gpu_memory():
    import gc
    import torch

    # Force Python garbage collection
    gc.collect()

    # Clear PyTorch GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete

    # Log memory state
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

        # WARNING if still using >50% memory after cleanup
        if allocated > 12.0:  # 24GB GPU, >50% still used
            logger.warning(f"⚠️  GPU memory still high after cleanup: {allocated:.2f}GB")
```

### 6. CONFIGURATION VALIDATION

**Problem:** Config changes can break training (wrong max_length, wrong model path, etc).

**Solution:** Add config validation BEFORE training starts:

```python
def validate_config(config_path):
    """Validate config.json before starting daemon."""
    with open(config_path) as f:
        config = json.load(f)

    # Check base model exists
    assert Path(config['base_model']).exists(), f"Base model not found: {config['base_model']}"

    # Check max_length is reasonable
    assert 128 <= config['max_length'] <= 32768, f"max_length out of range: {config['max_length']}"

    # Check learning rate is reasonable
    assert 1e-6 <= config['learning_rate'] <= 1e-2, f"LR out of range: {config['learning_rate']}"

    # Check batch size
    assert config['batch_size'] > 0, "batch_size must be > 0"

    # Check gradient accumulation
    assert config['gradient_accumulation'] > 0, "gradient_accumulation must be > 0"

    return config
```

### 7. DEFENSIVE PROGRAMMING - NULL CHECKS

```python
# Always check for None before using values
if checkpoint_dir is not None and checkpoint_dir.exists():
    # ... use checkpoint

# Use .get() with defaults for dictionaries
current_step = trainer_state.get("global_step", 0)  # ✅ Good
current_step = trainer_state["global_step"]          # ❌ KeyError if missing
```

### 8. FAIL-FAST ASSERTIONS

Add these at the START of critical functions:

```python
def train(self):
    # FAIL FAST: Check all prerequisites
    assert self.model is not None, "Model not loaded!"
    assert self.tokenizer is not None, "Tokenizer not loaded!"
    assert self.train_dataset is not None, "No training data!"
    assert len(self.train_dataset) > 0, "Training dataset is empty!"

    # Continue with training...
```

---

## 📋 COMPREHENSIVE EDGE CASES TO TEST

### Category 1: Data Issues

1. **Empty training file** (0 examples)
2. **Single example file** (< batch_size examples)
3. **Extremely long sequences** (> max_length, require truncation)
4. **Malformed JSON** (syntax errors, missing fields)
5. **Missing 'messages' field**
6. **Empty 'messages' array**
7. **Wrong role names** (not 'user' or 'assistant')
8. **Non-string content** (numbers, objects, arrays)
9. **Emoji/Unicode in data** (UTF-8 encoding issues)
10. **Extremely large files** (>1GB, memory issues)

### Category 2: Model/Checkpoint Issues

11. **No base model exists** (path doesn't exist)
12. **Corrupted model files** (incomplete download)
13. **Mismatched adapter/base** (adapter trained on different base)
14. **No checkpoints exist** (fresh start)
15. **Checkpoint directory exists but empty**
16. **Trainer_state.json missing** (no global_step info)
17. **Trainer_state.json corrupted** (invalid JSON)
18. **Checkpoint from different model** (incompatible)
19. **Very old checkpoint** (outdated format)
20. **Checkpoint mid-step** (partial save)

### Category 3: GPU/Memory Issues

21. **GPU OOM during model load**
22. **GPU OOM during tokenization**
23. **GPU OOM during training**
24. **System RAM OOM**
25. **Disk full** (can't save checkpoints)
26. **No GPU available** (CUDA not found)
27. **Multiple GPUs** (need to specify which one)
28. **GPU crash mid-training**
29. **Memory fragmentation** (many small allocations)
30. **Swap thrashing** (system using swap heavily)

### Category 4: Configuration Issues

31. **Missing config.json**
32. **Invalid JSON in config**
33. **max_length too small** (< shortest example)
34. **max_length too large** (> GPU memory)
35. **batch_size = 0**
36. **batch_size too large** (GPU OOM)
37. **learning_rate = 0**
38. **learning_rate too high** (divergence)
39. **Invalid model_name** (not in database)
40. **Mismatched tokenizer/model**

### Category 5: Process/System Issues

41. **Daemon killed mid-training** (SIGKILL)
42. **Power loss mid-training**
43. **System reboot mid-training**
44. **Disk I/O errors**
45. **Network mount disconnected** (if data on network)
46. **Permission errors** (can't write checkpoints)
47. **Clock skew** (timestamps inconsistent)
48. **Multiple daemons running** (PID file stale)
49. **Zombie processes** (defunct but not cleaned)
50. **File locks** (can't delete/move files)

### Category 6: Queue/File Management

51. **File deleted mid-queue** (race condition)
52. **File moved mid-processing**
53. **Duplicate files in queue** (same name)
54. **Circular symlinks**
55. **Files with special characters** (spaces, quotes)
56. **Files with no extension** (not .jsonl)
57. **Empty failed queue** (no files to retry)
58. **Queue metadata corrupted**
59. **Priority inversion** (low priority starves high)
60. **Infinite retry loop** (file keeps failing)

### Category 7: Training Flow Issues

61. **Zero training steps** (dataset too small)
62. **Negative global_step** (corrupted state)
63. **global_step > max_steps** (already trained)
64. **Loss is NaN** (training diverged)
65. **Loss is inf** (numerical overflow)
66. **Loss doesn't decrease** (stuck in plateau)
67. **Validation loss increases** (overfitting)
68. **Think tag percentage = 100%** (model not learning)
69. **Accuracy = 0%** (model completely broken)
70. **Training speed = 0** (hang/freeze)

### Category 8: Code/Variable Issues

71. **UnboundLocalError** (variable used before defined) ✅ JUST HAPPENED
72. **NameError** (variable doesn't exist)
73. **AttributeError** (object has no attribute)
74. **KeyError** (dictionary key missing)
75. **IndexError** (list index out of range)
76. **TypeError** (wrong type for operation) ✅ JUST HAPPENED
77. **ValueError** (value out of acceptable range)
78. **ZeroDivisionError** (divide by zero)
79. **RecursionError** (infinite recursion)
80. **MemoryError** (Python out of memory)

---

## 🔧 IMPLEMENTATION PLAN

### Phase 1: Immediate (Do Now)

1. ✅ Add assertions before TrainingArguments (validate total_steps exists)
2. ✅ Remove `num_train_epochs=None` (use default or omit)
3. ⏳ Add GPU memory cleanup between files
4. ⏳ Add config validation on daemon start

### Phase 2: Short-term (This Week)

5. Add static analysis (pylint/mypy) to pre-commit hook
6. Create smoke test for training initialization
7. Add parameter validation function
8. Add fail-fast assertions to all critical functions

### Phase 3: Medium-term (Next Week)

9. Create comprehensive test suite for all 80 edge cases
10. Add monitoring/alerting for OOM, NaN loss, etc.
11. Add automatic recovery for common failures
12. Document all edge cases and expected behavior

### Phase 4: Long-term (Ongoing)

13. Continuously add tests as new edge cases discovered
14. Regular code reviews with edge case checklist
15. Quarterly audit of all guardrails
16. Update documentation with lessons learned

---

## 🎯 SUCCESS METRICS

**How we'll know guardrails are working:**

1. **Zero UnboundLocalError bugs** - Static analysis catches all
2. **Zero TypeError bugs** - Type checking catches all
3. **Zero GPU OOM between files** - Memory cleanup works
4. **<5% failed training runs** - Robust error handling
5. **<10 min recovery time** - Fast rollback/restart
6. **100% edge case test coverage** - All 80 cases tested

---

## 📝 MAINTENANCE

**Weekly:**
- Run full test suite
- Check for new failed training patterns
- Update edge case list if new issues found

**Monthly:**
- Review all assertions (are they still needed?)
- Update static analysis rules
- Refactor code to reduce complexity

**Quarterly:**
- Full audit of all guardrails
- Performance review of validation overhead
- Team retrospective on recent bugs

---

## 💡 LESSONS LEARNED

### From Today's Bugs:

1. **Don't assume parameter order is safe** - Add assertions
2. **Don't assume API contracts** - Read the docs, validate inputs
3. **Don't set parameters to `None` without checking** - Use defaults or omit
4. **Don't bury critical calculations deep in functions** - Move to top
5. **Don't skip testing edge cases** - They WILL happen in production

### General Principles:

- **Fail fast, fail loud** - Don't let bugs propagate
- **Validate inputs at boundaries** - Check before using
- **Test the sad path** - Edge cases matter more than happy path
- **Monitor in production** - Catch issues before they become critical
- **Document assumptions** - Make implicit constraints explicit

---

**END OF DOCUMENT**
