# Advanced Anomaly Detection Examples

**Smart Monitor - Real-World Detection Scenarios**

Last Updated: 2025-11-12

---

## 🎯 What Gets Detected

The enhanced smart monitor now catches sophisticated anomalies that indicate training issues.

---

## 📊 Statistical Anomalies (Z-Scores)

### What is a Z-Score?

A z-score tells you how many standard deviations away from the mean a value is.

- **Z-score = 0:** Perfectly average
- **Z-score = ±1:** Within normal range (68% of data)
- **Z-score = ±2:** Unusual but possible (95% of data)
- **Z-score = ±3:** Very unusual! (99.7% of data)

**Trigger:** Z-score > 3.0 or < -3.0

### Example 1: Loss Spike Detection via Z-Score

```
Recent loss history: [0.72, 0.71, 0.73, 0.70, 0.72, 0.71, 0.73, 0.72]
Mean: 0.718
Std Dev: 0.010

Step 17500: Loss = 1.15
Z-score = (1.15 - 0.718) / 0.010 = 43.2

Trigger: "loss_zscore_43.20"
Snapshot: snapshots/anomaly_20251112_loss_zscore_43.20/
```

**What it means:** Loss jumped dramatically - something went wrong!

### Example 2: Learning Rate Schedule Glitch

```
Recent LR history: [0.00019, 0.000185, 0.00018, 0.000175, 0.00017]
Mean: 0.000182
Std Dev: 0.0000089

Step 18000: LR = 0.0005 (accidentally reset?)
Z-score = (0.0005 - 0.000182) / 0.0000089 = 35.7

Trigger: "lr_zscore_35.70"
Snapshot: snapshots/anomaly_20251112_lr_zscore_35.70/
```

**What it means:** Learning rate jumped - scheduler bug or manual change.

---

## 🎭 Prediction Anomalies (Loss/Accuracy Mismatch)

### Anomaly Type 1: Perfect Answer, High Loss

**Scenario:** Model generates perfect answer but loss is very high.

**Example:**
```json
{
  "prompt": "What is 2+2?",
  "golden_answer": "4",
  "model_answer": "4",
  "matches": true,
  "loss": 2.34  ← Should be near 0!
}
```

**Trigger:** `perfect_answer_high_loss_2.34`

**What it means:**
- Loss calculation bug?
- Model is uncertain despite being correct
- Tokenization mismatch?
- Label smoothing too aggressive?

**Action:** Investigate training code, check loss function.

### Anomaly Type 2: Wrong Answer, Low Loss

**Scenario:** Model generates wrong answer but loss is very low.

**Example:**
```json
{
  "prompt": "What is the capital of France?",
  "golden_answer": "Paris",
  "model_answer": "London",
  "matches": false,
  "loss": 0.12  ← Should be high!
}
```

**Trigger:** `wrong_answer_low_loss_0.12`

**What it means:**
- Model is confident but wrong (dangerous!)
- Training data might have errors
- Model memorizing wrong patterns
- Overfitting to noise

**Action:** Review training data quality, check for label errors.

### Anomaly Type 3: Inverted Loss Pattern

**Scenario:** Statistically, correct answers have HIGHER loss than incorrect ones.

**Normal pattern:**
```
Correct answers avg loss: 0.45
Incorrect answers avg loss: 1.23
```

**Inverted pattern (BAD):**
```
Correct answers avg loss: 1.45  ← High!
Incorrect answers avg loss: 0.67  ← Low!
```

**Trigger:** `inverted_loss_pattern_correct_high`

**What it means:**
- Training is fundamentally broken
- Loss function is backwards?
- Labels are inverted?
- Model learning opposite of what you want

**Action:** STOP TRAINING! Check loss function and data labels.

---

## 🔍 Combined Anomaly Examples

### Example 1: Catastrophic Failure

```
Step 15000: Multiple triggers detected
├─ loss_spike_87.3pct
├─ loss_zscore_12.45
├─ divergence_detected
├─ inverted_loss_pattern_correct_high
└─ lr_zscore_-3.21

Snapshot: snapshots/anomaly_20251112_150000_MULTIPLE_TRIGGERS/
```

**What happened:** Training completely broke. Learning rate crashed, loss exploded, model now doing opposite of correct.

**Action:** Rollback to previous checkpoint, investigate what changed.

### Example 2: Subtle Overfitting

```
Step 22000: Detection sequence over time

Step 20000: best_model_loss_0.6234
Step 21000: (normal training)
Step 22000: accuracy_drop_12.3pct
Step 22500: wrong_answer_low_loss_0.18
Step 23000: wrong_answer_low_loss_0.21

Snapshot: Multiple snapshots showing progression
```

**What happened:** Model found best point at 20k, then started overfitting. Getting confident about wrong answers.

**Action:** Use step 20000 model, stop training or get more diverse data.

### Example 3: Data Quality Issue

```
Step 17800: perfect_answer_high_loss_2.87

Investigation shows:
Prompt: "Translate: Hello"
Golden: "Bonjour"
Model: "Bonjour"
Loss: 2.87

But also found:
Prompt: "Translate: Hello" (duplicate)
Golden: "Hola" (different!)
```

**What happened:** Training data has conflicting labels for same input.

**Action:** Clean training data, remove duplicates with different labels.

---

## 📈 Real-World Detection Timeline

Here's what smart monitor catches during typical training:

```
Step 1000:   best_model_loss_1.2345 (initial best)
Step 2500:   best_model_loss_0.9876 (improving)
Step 5000:   best_model_loss_0.7654 (good progress)
Step 5100:   loss_spike_45.2pct (GPU hiccup? recovered)
Step 8000:   best_model_loss_0.6543 (continuing to improve)
Step 12000:  best_model_loss_0.5432 (peak performance)
Step 15000:  accuracy_drop_8.5pct (starting to overfit?)
Step 15500:  wrong_answer_low_loss_0.23 (confirming overfit)
Step 16000:  inverted_loss_pattern_correct_high (overfitting badly)
```

**Conclusion:** Best model was at step 12000. After that, overfitting began.

---

## 🎓 How to Use These Snapshots

### 1. Review Metadata

```bash
cat snapshots/anomaly_20251112_150000_*/metadata.json | jq
```

**Look for:**
- `triggers`: What anomalies were detected
- `z_scores`: How unusual was this?
- `recent_examples`: Pattern of correct/incorrect
- `loss_history`: Trend leading up to this

### 2. Compare to Normal Checkpoints

```bash
# Load normal checkpoint
python3 test_model.py --model current_model/checkpoint-15000

# Load anomaly snapshot
python3 test_model.py --model snapshots/anomaly_20251112_150000_*/model/

# Compare outputs
```

### 3. Investigate Root Cause

```python
# Check what changed
import json

# Load metadata
with open('snapshots/anomaly_*/metadata.json') as f:
    meta = json.load(f)

# Analyze
print(f"Triggers: {meta['triggers']}")
print(f"Loss Z-score: {meta['z_scores']['loss']}")
print(f"Recent examples: {meta['recent_examples']}")

# Look for patterns
correct_losses = [ex['loss'] for ex in meta['recent_examples'] if ex['matches']]
incorrect_losses = [ex['loss'] for ex in meta['recent_examples'] if not ex['matches']]

print(f"Avg loss when correct: {sum(correct_losses)/len(correct_losses)}")
print(f"Avg loss when wrong: {sum(incorrect_losses)/len(incorrect_losses)}")
```

---

## 🚨 Critical Anomalies (Immediate Action Required)

### 1. Inverted Loss Pattern
**Signal:** `inverted_loss_pattern_*`
**Action:** STOP TRAINING - Something fundamentally wrong
**Fix:** Check loss function, verify data labels, review training code

### 2. Extreme Z-Scores (>10)
**Signal:** `*_zscore_15.23`
**Action:** Investigate immediately - not normal variation
**Fix:** Review what changed, check logs, verify inputs

### 3. Multiple Simultaneous Triggers
**Signal:** 3+ triggers at same step
**Action:** Major problem - rollback to last good checkpoint
**Fix:** Systematic debugging needed

---

## ✅ Normal Anomalies (Expected Occasionally)

### 1. Loss Spike (Z-score 3-5)
**Frequency:** Maybe once per 5000 steps
**Cause:** Hard example batch, normal variation
**Action:** Monitor, usually recovers

### 2. Best Model
**Frequency:** Decreasing over time (common early, rare late)
**Cause:** Normal improvement
**Action:** None - this is good!

### 3. Accuracy Drop (5-10%)
**Frequency:** Occasional after best model
**Cause:** Exploring harder examples
**Action:** Watch trend - OK if recovers

---

## 📊 Anomaly Statistics Dashboard

Create summary of detected anomalies:

```bash
# Count by type
grep "Created snapshot" logs/smart_monitor_*.log | \
  cut -d'_' -f3- | sort | uniq -c

# Output:
# 15 best_model_loss_*
#  3 loss_spike_*
#  1 accuracy_drop_*
#  2 wrong_answer_low_loss_*
#  0 inverted_loss_pattern (good!)
```

---

## 🎯 Summary

**Statistical Anomalies:** Catch unusual values (z-score > 3)
- Loss spikes
- LR schedule glitches
- Unexpected metric changes

**Prediction Anomalies:** Catch logic errors
- Perfect answer but high loss
- Wrong answer but low loss
- Inverted loss patterns

**Combined:** Multiple triggers = serious problem

**Auto-saved to:** `snapshots/anomaly_*/` with full metadata

**Use for:** Debugging, finding best models, catching training issues early

---

**Monitor is active!** Check logs to see what it's detecting in real-time.
