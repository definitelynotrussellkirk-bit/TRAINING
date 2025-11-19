# CRITICAL BUG: Training Hangs on Inference

**Date:** 2025-11-16  03:02 AM
**Status:** BLOCKING - Evolution tracker can't work because training hangs

## The Real Problem

Evolution tracker IS initialized and the code IS correct.
**BUT: Training hangs during eval inference generation and never completes.**

## Evidence

1. **Log stops mid-inference:**
   ```
   DEBUG: Starting inference (user_msg len=14, golden_msg len=3)
   The following generation flags are not valid...
   [END OF LOG - stuck here for 17+ minutes]
   ```

2. **Log file not growing:**
   - 612 lines at 02:46
   - Still 612 lines at 03:02 (17 minutes later)

3. **GPU active but stuck:**
   - GPU utilization: 46%
   - Memory used: 17.5 GB
   - Process running but not progressing

4. **Evolution tracker code NEVER executes:**
   - My debug messages (🔍) should appear AFTER inference
   - They never appear because inference never finishes
   - Evolution snapshots can't be captured if code never runs

## Root Cause

The `model.generate()` call during eval is **hanging indefinitely**:

```python
# train.py ~line 790
outputs = self.model_ref.generate(
    inputs,
    max_new_tokens=self.max_output_tokens,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True
)
# ← HANGS HERE, never returns
```

## Why This Breaks Everything

```
Training Step 10 flow:
1. on_step_end() called ✅
2. Eval triggered (step 10 % 10 == 0) ✅
3. Extract example from dataset ✅
4. Start inference ✅
5. Call model.generate() ← HANGS HERE ❌
6. [NEVER REACHED] Decode output
7. [NEVER REACHED] Calculate loss
8. [NEVER REACHED] Update status
9. [NEVER REACHED] Call evolution_tracker.capture_snapshot()
10. [NEVER REACHED] Continue training
```

**Result:** Training completely frozen, evolution tracker never called, no progress.

## Possible Causes

1. **Infinite loop in generation:**
   - Model not generating EOS token
   - `max_new_tokens` might be too high (currently 2048)
   - Model might be stuck in repetition loop

2. **Tokenizer/model mismatch:**
   - Chat template might be malformed
   - Missing special tokens causing endless generation

3. **Memory/attention issue:**
   - Context too long
   - Attention mask problem
   - Out of memory (silent)

## Immediate Fix Options

### Option 1: Add generation timeout ⭐ RECOMMENDED
```python
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Before generate():
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    outputs = self.model_ref.generate(...)
    signal.alarm(0)  # Cancel alarm
except TimeoutException:
    print("⚠️  Generation timed out, skipping this eval")
    return  # Skip this eval, continue training
```

### Option 2: Reduce max_new_tokens drastically
```python
outputs = self.model_ref.generate(
    inputs,
    max_new_tokens=128,  # Was: 2048 (way too high!)
    ...
)
```

### Option 3: Disable sampling, use greedy
```python
outputs = self.model_ref.generate(
    inputs,
    max_new_tokens=128,
    do_sample=False,  # Greedy decoding (faster, deterministic)
)
```

### Option 4: Skip eval generation entirely (temporary)
```python
# Just comment out the whole inference section
# Still do evolution tracking on the training data itself
```

## Next Steps

1. **Kill hung process** and clean up
2. **Implement Option 1 + 2** (timeout + reduce tokens)
3. **Test with small token limit** (64 tokens)
4. **Once inference works**, evolution tracker will automatically work
5. **Verify snapshots appear** in data/evolution_snapshots/

## Timeline

- 02:45:39 - Training started
- 02:46:xx - Reached step 10, started inference
- 03:02:26 - Still hung (17+ minutes)
- **Next:** Kill, fix, restart

## Commands to Fix

```bash
# 1. Kill hung process
ps aux | grep training_daemon | grep -v grep | awk '{print $2}' | xargs kill -9

# 2. Clean state
rm -rf current_model/ __pycache__/

# 3. Edit train.py to add timeout + reduce max_new_tokens

# 4. Restart
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# 5. Monitor (should complete in ~5 minutes with fixes)
tail -f training_output.log | grep -E "(step|snapshot|📸)"
```

##Status
**BLOCKER IDENTIFIED - Ready to fix**
