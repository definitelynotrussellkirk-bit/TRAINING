# Bug Fix Instructions

## The Problem

Training crashed at step 624 (5% complete) with this error:

```
File "/path/to/training/live_monitor.py", line 65, in run_inference
    expected = example['messages'][1]['content'].strip()
AttributeError: 'int' object has no attribute 'strip'
```

**Root Cause**: Some training examples have integers in the `content` field instead of strings. The code assumes all content is a string and calls `.strip()` on it.

## The Fix

### File to Edit
`/path/to/training/live_monitor.py`

### Changes Needed

#### Change 1: Line 64
```python
# BEFORE (BROKEN):
user_content = example['messages'][0]['content']

# AFTER (FIXED):
user_content = str(example['messages'][0]['content'])
```

#### Change 2: Line 65
```python
# BEFORE (BROKEN):
expected = example['messages'][1]['content'].strip()

# AFTER (FIXED):
expected = str(example['messages'][1]['content']).strip()
```

### Complete Fix Command

```bash
cd /path/to/training

# Backup the original file
cp live_monitor.py live_monitor.py.backup

# Apply the fix using sed
sed -i '64s/example\[.messages.\]\[0\]\[.content.\]/str(example["messages"][0]["content"])/' live_monitor.py
sed -i '65s/example\[.messages.\]\[1\]\[.content.\]/str(example["messages"][1]["content"])/' live_monitor.py
```

### Or Manual Fix

1. Open the file:
```bash
nano /path/to/training/live_monitor.py
```

2. Go to line 64 and 65 (around line 64-65 in the `run_inference` method)

3. Make the changes shown above

4. Save and exit (Ctrl+X, then Y, then Enter)

## Verification

After fixing, verify the changes:

```bash
# Check the fixed lines
grep -n "user_content\|expected" /path/to/training/live_monitor.py | head -5
```

Should show:
```
64:                user_content = str(example['messages'][0]['content'])
65:                expected = str(example['messages'][1]['content']).strip()
```

## Additional Improvements (Optional)

### 1. Update Monitoring Frequency

Current: Updates every 625 steps (takes 30+ minutes)
Better: Update every 10 steps (updates every ~35 seconds)

**File**: `/path/to/training/train.py`

**Find**:
```python
detail_collector = DetailCollector(
    output_dir=status_dir,
    tokenizer=tokenizer,
    eval_dataset=val_dataset,
    update_frequency=625  # ← CHANGE THIS
)
```

**Change to**:
```python
detail_collector = DetailCollector(
    output_dir=status_dir,
    tokenizer=tokenizer,
    eval_dataset=val_dataset,
    update_frequency=10  # ← NOW UPDATES EVERY 10 STEPS
)
```

### 2. Add Training Config to Status JSON

To show model name, batch size, etc. in the dashboard, edit `/path/to/training/detail_collector.py`:

**Add to the `_write_detail` method**:
```python
data['config'] = {
    'model': 'Qwen2.5-7B-Instruct',
    'batch_size': 1,
    'gradient_accumulation': 8,
    'effective_batch_size': 8,
    'learning_rate': 0.0002,
    'total_steps': 12488,
    'dataset': 'leo_100k_compositional_fixed.jsonl',
    'method': 'QLoRA (4-bit)'
}
```

## After Fixing

### 1. Restart the Daemon

```bash
# Stop training
touch /path/to/training/.stop

# Wait for daemon to stop (check with ps aux | grep training_daemon)
sleep 10

# Remove stop file
rm -f /path/to/training/.stop

# Restart daemon
cd /path/to/training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 training_daemon.py --base-dir /path/to/training > /tmp/daemon.log 2>&1 &
```

### 2. Monitor Progress

```bash
# Watch the daemon log
tail -f /tmp/daemon.log

# Open the enhanced monitor
xdg-open http://localhost:8082  # or open http://localhost:8082 on Mac

# Check GPU usage
watch -n 1 nvidia-smi
```

### 3. Verify Fix Worked

Training should now:
- ✅ Start successfully
- ✅ Progress past step 624
- ✅ Not crash with AttributeError
- ✅ Continue to completion (~12 hours)

## Rollback (If Needed)

If something goes wrong:

```bash
# Restore original file
cp /path/to/training/live_monitor.py.backup \
   /path/to/training/live_monitor.py

# Restart daemon
touch /path/to/training/.stop
sleep 10
rm -f /path/to/training/.stop
cd /path/to/training
nohup python3 training_daemon.py --base-dir /path/to/training > /tmp/daemon.log 2>&1 &
```

## Testing the Fix

Before restarting full training, you can test on a small subset:

```bash
cd /path/to/training

# Test with just 100 examples
python3 train.py \
  --dataset inbox/leo_10k_qlora.jsonl \
  --model model_qwen25 \
  --output-dir /tmp/test_fix \
  --epochs 1 \
  --max-steps 100 \
  --skip-validation \
  --yes \
  --use-qlora
```

If this completes without error, the fix worked!

## Summary

1. **Fix**: Convert content to string before calling .strip()
2. **Verify**: Check the fix was applied correctly
3. **Optional**: Update monitoring frequency to 10 steps
4. **Restart**: Stop and restart the training daemon
5. **Monitor**: Watch at http://localhost:8082

**Expected Result**: Training completes successfully in ~12 hours!

---

**Last Updated**: 2025-11-07 01:15 AM
