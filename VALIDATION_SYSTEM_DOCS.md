# Validation System Documentation

**Date:** 2025-11-16
**Status:** ✅ COMPLETE and WORKING

---

## Overview

The validation system measures **true generalization** by computing loss on a fixed set of unseen examples. This tells us if the model is learning general patterns or just memorizing training data.

---

## Components

### 1. Fixed Validation Set
- **File:** `data/validation/syllo_validation_1000.jsonl`
- **Size:** 1000 syllogism examples
- **Sampled:** 100 examples used per validation check (for speed)
- **Seed:** 42 (reproducible)
- **Source:** Held out from training data

### 2. Validation Loss Computation
- **Location:** `train.py` - `LiveMonitorCallback.compute_validation_loss()`
- **Frequency:** Every eval step (every 10 steps)
- **Method:** Same as training loss (cross-entropy on tokens, padding excluded)
- **Speed:** ~1-2 seconds per validation check (20 examples)

### 3. Status Tracking
- **Fields in `training_status.json`:**
  - `validation_loss` - Loss on unseen validation data
  - `val_train_gap` - Difference (validation - training loss)
  - `think_tag_count` - Count of outputs with `<think>` tags
  - `think_tag_percent` - Percentage of outputs with `<think>` tags

### 4. UI Display
- **New Modular Monitor:** http://localhost:8080/live_monitor_ui_v2.html
  - Train vs Val loss chart
  - Think tag percentage chart
  - Color-coded gap indicator
  - Real-time updates

---

## How It Works

### Validation Loss Computation:

```python
# Every 10 steps during training:
1. Sample 20 examples from fixed validation set
2. Tokenize with same tokenizer as training
3. Set padding tokens to -100 (ignored in loss)
4. Run model forward pass (no gradients)
5. Compute average cross-entropy loss
6. Compare to current training loss
```

### Key Fix (2025-11-16):

**BEFORE:**
```python
labels = input_ids  # WRONG - includes padding!
```

**AFTER:**
```python
labels = input_ids.clone()
labels[labels == pad_token_id] = -100  # Exclude padding ✅
```

This fixed the validation loss from 10.97 → 0.127 (realistic value!)

---

## Interpreting Results

### Train/Val Gap Meanings:

| Gap Size | Status | Interpretation | Action |
|----------|--------|----------------|--------|
| < 0.2 | ✅ Excellent | Model generalizes perfectly | Keep training |
| 0.2 - 0.4 | ✅ Good | Normal generalization | Monitor |
| 0.4 - 0.6 | ⚠️ Warning | May be overfitting | Consider stopping soon |
| > 0.6 | 🚨 Overfitting | Memorizing training data | Stop or add more data |
| Negative | ✅ Great | Val easier than training | Keep training |

### Current Training (Step 1617+):
- **Train Loss:** 0.132
- **Val Loss:** 0.127
- **Gap:** -0.005 ✅ **EXCELLENT!**
- **Interpretation:** Model is learning general patterns, not overfitting

### Think Tag Tracking:

**Problem:** Base model (DIO) outputs `<think></think>` tags from pre-training, but our training data is clean (no think tags).

**Solution:** Track percentage of outputs with think tags:

| Think % | Status | Interpretation |
|---------|--------|----------------|
| 100% | 🔴 Not learning format | Model hasn't learned to skip think tags yet |
| 60-99% | 🟡 Learning | Model starting to output clean format |
| 20-60% | 🟢 Good progress | Most outputs clean |
| < 20% | ✅ Excellent | Model learned clean format |
| 0% | 🎯 Perfect | No unwanted think tags |

**Current:** 100% (expected early in training, will decrease)

---

## Files Modified

### Code:
- `train.py` (lines 754-802) - Validation loss computation
- `training_status.py` (lines 95-101, 125, 155, 216-220) - Status tracking
- `training_daemon.py` - Loads validation set automatically

### Data:
- `data/validation/syllo_validation_1000.jsonl` - Fixed validation set

### UI:
- `live_monitor_ui_v2.html` - New modular monitor
- `js/chart_manager.js` - Validation charts
- `css/live_monitor.css` - Styling

---

## Usage

### Check Validation Metrics:

```bash
# Quick check
cat status/training_status.json | jq '{
  step: .current_step,
  train_loss: .loss,
  val_loss: .validation_loss,
  gap: .val_train_gap,
  think_pct: .think_tag_percent
}'

# Output:
{
  "step": 1617,
  "train_loss": 0.132,
  "val_loss": 0.127,
  "gap": -0.005,
  "think_pct": 100.0
}
```

### View in UI:

1. Open http://localhost:8080/live_monitor_ui_v2.html
2. Check status bar for current metrics
3. View charts showing trends over time
4. Gap is color-coded (green = good, yellow = warning, red = bad)

---

## Technical Details

### Why Exclude Padding?

**Padding tokens are filler** to make sequences the same length for GPU batching. They're not real data.

**If we include padding in loss:**
- Loss dominated by predicting padding (wrong signal)
- Misleading comparison to training loss
- Model incentivized to predict padding well (useless)

**Correct approach:**
- Set padding token labels to -100
- PyTorch automatically ignores -100 in cross-entropy loss
- Only compute loss on actual content tokens

### Why Sample 20 Examples?

**Trade-off:**
- More examples = more accurate loss estimate
- Fewer examples = faster validation check

**20 examples:**
- ✅ Fast (~1-2 seconds)
- ✅ Stable loss estimate
- ✅ Doesn't slow training

**100 examples per check would be:**
- ❌ Too slow (~10 seconds)
- ❌ Delays training
- ✅ Slightly more accurate (not worth it)

---

## Future Enhancements

### Possible Additions:

1. **Multiple validation sets** (easy/medium/hard difficulty)
2. **Accuracy on validation set** (% correct, not just loss)
3. **Auto-pause on high gap** (stop training if gap > 1.0)
4. **Validation set browser** (browse validation examples in UI)
5. **Per-pattern breakdown** (validation loss by puzzle type)

---

## Troubleshooting

### Validation loss is null:

**Cause:** Validation set not loaded or training hasn't reached eval step yet.

**Fix:** Wait for step 10, 20, 30, etc. (multiples of eval_steps)

### Validation loss very high (> 5):

**Possible causes:**
1. Padding tokens included (BUG - should be fixed now)
2. Validation set has different format than training
3. Model hasn't learned anything yet (early in training)

**Check:**
```bash
# Verify padding fix is applied
grep -A5 "labels\[labels == " train.py
# Should see: labels[labels == self.tokenizer.pad_token_id] = -100
```

### Gap increasing over time:

**Cause:** Model is overfitting (memorizing training examples).

**Solutions:**
1. Stop training and use current checkpoint
2. Add more training data
3. Increase regularization (dropout)
4. Use smaller model or reduce LoRA rank

---

**Questions?** Check `VALIDATION_SYSTEM_SUMMARY.md` for overview or `REFACTOR_COMPLETE.md` for UI details.
