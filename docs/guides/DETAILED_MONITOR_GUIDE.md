# Detailed Training Monitor - Complete Guide

**Created:** 2025-11-03
**Status:** ✅ Ready to use

---

## 🎯 What This Provides

A **real-time web dashboard** showing during training:

✅ **Current loss / eval loss** - Live training metrics
✅ **Complete prompt context** - Full conversation (system + user + assistant)
✅ **User input** - What the user asked
✅ **Assistant output so far** - Partial generation
✅ **Golden assistant** - Expected/correct output
✅ **Current guess** - Model's prediction
✅ **Token-by-token comparison** - Color-coded accuracy

**URL:** http://localhost:8081
**Updates:** Every 50 training steps (configurable)

---

## 📁 Files Created

1. **`detailed_monitor.py`** - Web dashboard server (Flask)
2. **`detail_collector.py`** - Training data collector (callback)
3. **`enable_detailed_monitoring.py`** - Integration script
4. **`DETAILED_MONITOR_GUIDE.md`** - This guide

---

## 🚀 Quick Start (3 Steps)

### Step 1: Enable Monitoring (One-Time Setup)

```bash
cd /path/to/training
python3 enable_detailed_monitoring.py
```

**What this does:**
- Patches `train.py` to collect detailed data
- Creates backup (`train.py.backup`)
- Creates launch script

**Output:**
```
✅ train.py patched successfully
✅ Created launch script
✅ DETAILED MONITORING ENABLED
```

### Step 2: Start the Dashboard

```bash
# Terminal 1
python3 detailed_monitor.py

# Output:
# 🔬 DETAILED TRAINING MONITOR
# Server starting at: http://localhost:8081
```

### Step 3: Start Training

```bash
# Terminal 2
python3 train.py \
  --dataset inbox/leo_10k_with_system.jsonl \
  --model model \
  --output-dir adapters/test \
  --epochs 1 \
  --use-qlora
```

### Step 4: View Dashboard

Open browser: **http://localhost:8081**

---

## 🖥️ Dashboard Features

### Top Metrics Cards

```
┌─────────────┬──────────────┬────────┬────────┐
│Training Loss│  Eval Loss   │  Step  │ Epoch  │
│   2.1234    │   1.9876     │  450   │  0.36  │
└─────────────┴──────────────┴────────┴────────┘
```

### Complete Prompt Context

Shows full conversation with color-coded roles:

```
[SYSTEM] (purple border)
You enjoy helping others. Your goal is produce what the user
WANTS...

[USER] (blue border)
What items have property X? Return as JSON.

[ASSISTANT] (green border)
[Currently being generated...]
```

### Golden vs Predicted Comparison

Side-by-side view with token matching:

```
┌─────────────────────┬─────────────────────┐
│ ✓ Golden (Expected) │ 🤖 Predicted        │
├─────────────────────┼─────────────────────┤
│ The quick brown fox │ The quick brown fox │
│ [match] [match]     │ [match] [match]     │
│                     │                     │
│ jumps over dog      │ jumps over cat      │
│ [match] [mismatch]  │ [match] [mismatch]  │
└─────────────────────┴─────────────────────┘

Accuracy: 85.7% | Matches: 6/7 | Diff: 1
```

**Color Coding:**
- 🟢 **Green** - Tokens match
- 🔴 **Red** - Tokens mismatch
- 🟡 **Yellow** - Missing tokens

---

## ⚙️ Configuration

### Update Frequency

Edit `detail_collector.py`:

```python
add_detail_collector_to_trainer(
    trainer=trainer,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    update_frequency=50  # Update every 50 steps (default)
)
```

**Options:**
- `25` - Very frequent updates (slower training)
- `50` - Balanced (recommended)
- `100` - Less frequent (faster training)
- `200` - Sparse updates

### Dashboard Port

Edit `detailed_monitor.py`:

```python
app.run(host='0.0.0.0', port=8081, debug=False)
#                           ^^^^
#                           Change port here
```

---

## 🔍 How It Works

### Data Flow

```
Training Loop
      ↓
DetailCollector (callback)
      ↓
Captures every N steps:
- Current metrics
- Sample from validation set
- Model prediction
      ↓
Writes to: status/training_detail.json
      ↓
Dashboard reads JSON
      ↓
Browser displays (auto-refresh every 2s)
```

### What Gets Captured

**During Training:**
- Step number
- Epoch
- Training loss
- Learning rate
- GPU memory

**During Evaluation (every N steps):**
- Everything above, plus:
- Full prompt (all messages)
- Golden response (expected)
- Model prediction (generated)
- Token-by-token comparison

---

## 📊 Example Output

### training_detail.json

```json
{
  "status": "training",
  "step": 450,
  "epoch": 0.36,
  "train_loss": 2.1234,
  "eval_loss": 1.9876,
  "learning_rate": 0.0001998,
  "sample_idx": 0,
  "prompt": {
    "messages": [
      {
        "role": "system",
        "content": "You enjoy helping others..."
      },
      {
        "role": "user",
        "content": "What items have property X?"
      }
    ]
  },
  "golden": "[{\"name\": \"item1\", \"properties\": {...}}]",
  "predicted": "[{\"name\": \"item1\", \"property\": {...}}]",
  "gpu_memory": "19.43GB",
  "timestamp": "2025-11-03T14:52:30.123456"
}
```

---

## 🐛 Troubleshooting

### Issue: Dashboard shows "Waiting for training data..."

**Cause:** Training hasn't reached an evaluation step yet

**Solution:**
- Wait for training to reach step 50 (default update frequency)
- Or reduce `update_frequency` to 25

### Issue: Dashboard shows old data

**Cause:** training_detail.json not being updated

**Solution:**
```bash
# Check if collector is enabled
grep -i "detail collector enabled" logs/daemon_*.log

# Check if file is being written
ls -lh status/training_detail.json
stat status/training_detail.json

# Check file contents
cat status/training_detail.json
```

### Issue: "Module not found: detail_collector"

**Cause:** Not in training directory

**Solution:**
```bash
cd /path/to/training
python3 enable_detailed_monitoring.py  # Re-run setup
```

### Issue: Port 8081 already in use

**Solution:**
```bash
# Find process using port
lsof -i :8081

# Kill it
kill <PID>

# Or change port in detailed_monitor.py
```

---

## 🔧 Advanced Usage

### Multiple Training Runs

Run dashboard once, monitor multiple training runs:

```bash
# Terminal 1 - Dashboard (leave running)
python3 detailed_monitor.py

# Terminal 2 - First training
python3 train.py --dataset data1.jsonl --output-dir run1 ...

# Terminal 3 - Second training (after first completes)
python3 train.py --dataset data2.jsonl --output-dir run2 ...
```

Dashboard automatically updates with current training data.

### Custom Metrics

Edit `detail_collector.py` to add custom metrics:

```python
detail_data = {
    # ... existing metrics ...
    'custom_metric': your_calculation_here,
    'perplexity': torch.exp(torch.tensor(train_loss)),
    # etc.
}
```

### Save Snapshots

Capture interesting samples:

```bash
# While training, save current detail
cp status/training_detail.json snapshots/step_450.json
```

---

## 📚 Integration with Existing Tools

### Works With

✅ **Training Daemon** - Auto-processes files from inbox
✅ **Web UI** (port 7860) - General training control
✅ **Live Monitor** (port 8080) - Basic metrics
✅ **Detailed Monitor** (port 8081) - This tool ⭐

### Ports Used

```
:7860  - Training Control Center (Gradio)
:8080  - Live Monitor (basic metrics)
:8081  - Detailed Monitor (this dashboard)
```

---

## 🎓 Understanding the Display

### What Each Section Means

**Metrics Cards**
- Shows current training state
- Updates in real-time
- Compare train vs eval loss to check for overfitting

**Complete Prompt Context**
- Shows what the model sees
- Includes system prompt (your personality injection)
- Full conversation history

**Golden vs Predicted**
- **Golden** = What the model should generate
- **Predicted** = What the model actually generates
- Token colors show where model matches/diverges

**Sample Information**
- Which validation sample is shown
- Current learning rate
- GPU memory usage

---

## 💡 Tips

1. **Watch for divergence** - If predicted text diverges early, model needs more training
2. **Check token matches** - High match rate (>80%) indicates good learning
3. **Monitor eval loss** - Should decrease over time
4. **System prompt** - Visible in prompt context, shows model sees your personality
5. **Update frequency** - Too frequent slows training, too sparse misses issues

---

## 🆘 Emergency Commands

```bash
# Stop dashboard
pkill -f detailed_monitor

# Restart dashboard
python3 detailed_monitor.py &

# Check if running
ps aux | grep detailed_monitor

# View current detail file
cat status/training_detail.json | python3 -m json.tool

# Reset monitoring (remove patch)
cp train.py.backup train.py
```

---

## 📖 Quick Reference

```bash
# Setup (one-time)
python3 enable_detailed_monitoring.py

# Start dashboard
python3 detailed_monitor.py

# Start training
python3 train.py --dataset inbox/data.jsonl --model model \
  --output-dir adapters/test --epochs 1 --use-qlora

# View
firefox http://localhost:8081
```

---

## ✅ Checklist

Before first use:

- [ ] Run `enable_detailed_monitoring.py`
- [ ] Verify `train.py.backup` was created
- [ ] Start `detailed_monitor.py`
- [ ] Open http://localhost:8081 in browser
- [ ] Start training
- [ ] Wait for step 50 (first update)
- [ ] Verify dashboard shows data

---

## 🔗 Related Documentation

- `QUICK_START.md` - Main training guide
- `SYSTEM_PROMPT_GUIDE.md` - System prompt injection
- `WEB_UI_GUIDE.md` - Training Control Center (port 7860)
- `README.md` - System overview

---

**Status:** ✅ Fully functional
**Last Updated:** 2025-11-03
**Dashboard URL:** http://localhost:8081
**Update Frequency:** Every 50 steps (configurable)
