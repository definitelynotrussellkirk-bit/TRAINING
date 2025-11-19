# Smart Monitor Guide - Intelligent Anomaly Detection

**Last Updated:** 2025-11-12

Automated snapshot system that watches training and saves checkpoints when anomalies occur.

---

## 🎯 What It Does

The Smart Monitor continuously watches your training and automatically creates snapshots when:

### 1. **Loss Spikes** 🔥
Detects sudden increases in loss (default: >30%)
```
Example: Loss was 0.74, suddenly jumps to 1.02
→ Saves: snapshots/anomaly_20251112_143022_loss_spike_37.8pct/
```

### 2. **Accuracy Drops** 📉
Detects sudden decreases in accuracy (default: >10%)
```
Example: Accuracy was 75%, drops to 62%
→ Saves: snapshots/anomaly_20251112_143500_accuracy_drop_13.0pct/
```

### 3. **Best Model Found** 🏆
Automatically saves whenever a new best model is achieved
```
Example: Loss reaches new low of 0.6321 (was 0.6589)
→ Saves: snapshots/anomaly_20251112_144200_best_model_loss_0.6321/
```

### 4. **Training Divergence** ⚠️
Detects if loss keeps increasing (training going wrong)
```
Example: Loss increasing for 8+ consecutive evaluations
→ Saves: snapshots/anomaly_20251112_145000_divergence_detected/
```

---

## 🚀 Quick Start

### Start Monitoring

```bash
# Basic (with defaults)
nohup python3 smart_monitor.py > /dev/null 2>&1 &

# With custom thresholds
python3 smart_monitor.py \
  --loss-spike-threshold 0.2 \
  --accuracy-drop-threshold 15 \
  --min-steps-between-saves 250
```

### Check Status

```bash
# See what's been detected
ls -lt snapshots/anomaly_*/ | head

# View logs
tail -f logs/smart_monitor_$(date +%Y%m%d).log
```

### Stop Monitoring

```bash
# Find and kill
ps aux | grep smart_monitor | grep -v grep
kill <PID>
```

---

## ⚙️ Configuration

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-dir` | `/path/to/training` | Training directory |
| `--loss-spike-threshold` | `0.3` | 30% increase triggers save |
| `--accuracy-drop-threshold` | `10` | 10% drop triggers save |
| `--poll-interval` | `10` | Check every 10 seconds |
| `--min-steps-between-saves` | `500` | Min 500 steps between saves |
| `--loss-window` | `10` | Average last 10 values |

### Examples

**Conservative (More Snapshots):**
```bash
python3 smart_monitor.py \
  --loss-spike-threshold 0.1 \
  --accuracy-drop-threshold 5 \
  --min-steps-between-saves 250
```

**Moderate (Default):**
```bash
python3 smart_monitor.py
```

**Aggressive (Fewer Snapshots):**
```bash
python3 smart_monitor.py \
  --loss-spike-threshold 0.5 \
  --accuracy-drop-threshold 20 \
  --min-steps-between-saves 1000
```

---

## 📂 Snapshot Structure

Each anomaly snapshot contains:

```
snapshots/anomaly_20251112_143022_loss_spike_37.8pct/
├── model/                          # Saved checkpoint
│   ├── adapter_model.safetensors   # Model weights
│   └── checkpoint-17500/           # Latest checkpoint
├── metadata.json                   # Why it was saved
└── training_status.json            # Full status at save time
```

### Metadata Example

```json
{
  "timestamp": "20251112_143022",
  "reason": "loss_spike_37.8pct",
  "step": 17500,
  "loss": 1.0234,
  "accuracy": 58.3,
  "triggers": ["loss_spike_37.8pct"],
  "loss_history": [0.74, 0.73, 0.72, 0.71, 1.02],
  "best_loss": 0.6987,
  "best_loss_step": 16800
}
```

---

## 🎓 How Detection Works

### Loss Spike Detection

1. Maintains rolling window of last N loss values (default: 10)
2. Calculates recent average (last 5 values)
3. If current loss > average × (1 + threshold), triggers
4. Example: Avg=0.75, threshold=30%, trigger if loss > 0.975

### Accuracy Drop Detection

1. Maintains rolling window of last N accuracy values
2. Calculates recent average
3. If current < average - threshold, triggers
4. Example: Avg=75%, threshold=10%, trigger if < 65%

### Best Model Detection

1. Tracks all-time best (lowest) loss
2. When new best found with >2% improvement, saves
3. Ensures you never lose your best checkpoint

### Divergence Detection

1. Checks if 80%+ of recent losses are increasing
2. Indicates training instability
3. Helps catch gradient explosions early

---

## 🔍 Use Cases

### Use Case 1: Catch Learning Rate Too High

**Symptom:** Loss oscillating or spiking
**Detection:** Loss spike triggers
**Action:** Snapshot saved, review and rollback to earlier checkpoint

### Use Case 2: Find Optimal Stopping Point

**Symptom:** Accuracy plateaus then drops (overfitting)
**Detection:** Best model saved, then accuracy drop saves "before overfit"
**Action:** Use best_model snapshot for deployment

### Use Case 3: Debug Training Issues

**Symptom:** Training suddenly goes wrong
**Detection:** Divergence or multiple triggers
**Action:** Review metadata.json to see what happened

### Use Case 4: Never Lose Best Model

**Symptom:** Training continues but model gets worse
**Detection:** Best model auto-saved at peak
**Action:** Always have access to optimal checkpoint

---

## 📊 Monitoring Dashboard

View real-time stats in logs:

```bash
# Live monitoring
tail -f logs/smart_monitor_$(date +%Y%m%d).log

# Recent anomalies
grep "ALERT" logs/smart_monitor_$(date +%Y%m%d).log

# Statistics
grep "Monitoring: Step" logs/smart_monitor_$(date +%Y%m%d).log | tail -10
```

---

## 🛠️ Integration with Training

### Run Together

Start both training daemon and smart monitor:

```bash
# Terminal 1: Training
nohup python3 training_daemon.py --base-dir $(pwd) > training_output.log 2>&1 &

# Terminal 2: Smart monitoring
nohup python3 smart_monitor.py > smart_monitor_output.log 2>&1 &

# Terminal 3: Regular monitors
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &
nohup python3 enhanced_monitor.py > /dev/null 2>&1 &
```

### Automated Startup

Add to startup script:

```bash
#!/bin/bash
cd /path/to/training

# Start training
nohup python3 training_daemon.py --base-dir $(pwd) > training_output.log 2>&1 &

# Start smart monitor
nohup python3 smart_monitor.py > smart_monitor_output.log 2>&1 &

# Start web monitors
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &
nohup python3 enhanced_monitor.py > /dev/null 2>&1 &
nohup python3 memory_stats_api.py > /dev/null 2>&1 &
```

---

## 🎯 Best Practices

### 1. Adjust Thresholds Based on Data

**Early training (first 5k steps):**
- Higher spike threshold (0.5) - loss more volatile
- Lower accuracy threshold (5%) - still stabilizing

**Stable training (after 5k steps):**
- Standard thresholds (0.3, 10%) - should be smooth

**Fine-tuning (near convergence):**
- Lower spike threshold (0.1) - catch tiny regressions
- Higher min steps (1000) - reduce noise

### 2. Review Snapshots Regularly

```bash
# Weekly review
ls -lt snapshots/anomaly_*/ | head -20

# Check why things were saved
cat snapshots/anomaly_*/metadata.json | jq '.reason'

# Find best models
ls snapshots/anomaly_*best_model*/
```

### 3. Clean Up Old Anomalies

```bash
# Keep only last 30 days
find snapshots/ -name "anomaly_*" -mtime +30 -exec rm -rf {} \;

# Or keep only specific types
find snapshots/ -name "anomaly_*loss_spike*" -mtime +7 -exec rm -rf {} \;
# But keep best_model snapshots forever
```

### 4. Use with A/B Testing

When trying different hyperparameters:

```bash
# Run 1: Conservative LR
# smart_monitor saves best model

# Run 2: Aggressive LR
# smart_monitor detects if it diverges

# Compare best_model snapshots from each run
```

---

## 🚨 Troubleshooting

### Monitor Not Detecting Anything

**Check:**
```bash
# Is it running?
ps aux | grep smart_monitor

# Are thresholds too strict?
# Try looser thresholds
python3 smart_monitor.py --loss-spike-threshold 0.1

# Is training actually varying?
tail status/training_status.json
```

### Too Many Snapshots

**Solutions:**
```bash
# Increase min_steps_between_saves
python3 smart_monitor.py --min-steps-between-saves 1000

# Tighten thresholds
python3 smart_monitor.py --loss-spike-threshold 0.5

# Reduce sensitivity
python3 smart_monitor.py --loss-window 20
```

### Missed Important Event

**Check logs:**
```bash
grep "Step 17500" logs/smart_monitor_*.log
# See what monitor saw at that step
```

**Adjust thresholds:**
```bash
# Lower thresholds to catch more
python3 smart_monitor.py --loss-spike-threshold 0.15
```

---

## 📈 Performance Impact

**CPU:** < 1% (polls every 10 seconds)
**Memory:** ~50MB (maintains small history buffers)
**Disk I/O:** Only on snapshot creation (~4GB per snapshot)
**Training Impact:** None (runs independently)

---

## 🔗 Related Tools

- `training_daemon.py` - Runs the training
- `maintenance.sh` - Regular checkpoint cleanup
- `cleanup_checkpoints.sh` - 3-tier retention policy
- `memory_monitor.sh` - RAM usage alerts
- Live monitor UI - Real-time visualization

---

## 📝 Examples

### Example 1: Catch Overfitting

```bash
# Start monitor with accuracy focus
python3 smart_monitor.py --accuracy-drop-threshold 5

# Training runs
# At step 18000: accuracy 85% → best_model saved
# At step 22000: accuracy 78% → accuracy_drop saved
# → Rollback to step 18000 model
```

### Example 2: Learning Rate Too High

```bash
# Monitor detects:
# Step 15000: loss 0.72
# Step 15100: loss 0.71
# Step 15200: loss 1.15 → loss_spike_61.2pct saved
# → Review snapshot, reduce LR, resume training
```

### Example 3: Systematic Best Model Collection

```bash
# Over days of training, monitor automatically saves:
snapshots/
├── anomaly_20251112_best_model_loss_0.8234/
├── anomaly_20251113_best_model_loss_0.7456/
├── anomaly_20251114_best_model_loss_0.6891/
└── anomaly_20251115_best_model_loss_0.6234/

# Always have history of improvements!
```

---

## ✅ Quick Reference

**Start:** `python3 smart_monitor.py &`
**Stop:** `pkill -f smart_monitor`
**Logs:** `tail -f logs/smart_monitor_$(date +%Y%m%d).log`
**Snapshots:** `ls -lt snapshots/anomaly_*/`
**Best models:** `ls snapshots/anomaly_*best_model*/`

---

**Status:** ✅ Ready to use
**Dependencies:** Python 3, json, shutil (all standard library)
**Runs:** Independently from training
**Cost:** Minimal CPU/memory, disk only on snapshots
