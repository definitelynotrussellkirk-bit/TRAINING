# Scripts Reference

Quick reference for all scripts in the training system.

---

## 🚀 Core Training Scripts

### **train.py** (36K)
The main training script. Handles:
- QLoRA 4-bit training
- Dataset loading and tokenization
- Live inference monitoring
- Checkpoint management

**Usage:**
```bash
python3 train.py --dataset inbox/data.jsonl --output current_model/
```

### **training_daemon.py** (20K)
Auto-ingestion daemon. Monitors inbox/ and trains automatically.

**Usage:**
```bash
python3 training_daemon.py --base-dir /path/to/training
```

**Control:**
```bash
# Stop gracefully
touch .stop

# Check status
ps aux | grep training_daemon
```

---

## 📊 Monitoring Scripts

### **launch_live_monitor.py** (20K)
Launches the main web UI on port 8080.

**Usage:**
```bash
python3 launch_live_monitor.py
# Access: http://localhost:8080/live_monitor_ui.html
```

### **enhanced_monitor.py** (24K)
Gradio-based monitoring interface on port 8082.

**Usage:**
```bash
python3 enhanced_monitor.py
# Access: http://localhost:8082
```

### **detailed_monitor.py** (24K)
Advanced monitoring with detailed metrics.

**Usage:**
```bash
python3 detailed_monitor.py
```

### **memory_stats_api.py**
Memory monitoring API on port 8081.

**Usage:**
```bash
python3 memory_stats_api.py
# API: http://localhost:8081/api/memory_stats
```

### **gpu_stats_api.py**
GPU statistics API endpoint.

**Usage:**
```bash
# Runs as part of launch_live_monitor.py
curl http://localhost:8080/api/gpu_stats
```

---

## 🔧 Utility Scripts

### **consolidate_model.py**
Merges LoRA adapter into base model.

**Usage:**
```bash
python3 consolidate_model.py --base-dir /path/to/training
```

**When to use:**
- Manual consolidation (normally happens at 3 AM daily)
- Creating deployment model
- Clearing adapter to start fresh

### **validator.py**
Validates training data format.

**Usage:**
```bash
python3 validator.py inbox/data.jsonl
```

### **convert_leo_data.py**
Converts LEO system outputs to training format.

**Usage:**
```bash
python3 convert_leo_data.py input.json output.jsonl
```

### **add_system_prompt.py**
Adds system prompt to training data.

**Usage:**
```bash
python3 add_system_prompt.py --input data.jsonl --output data_with_prompt.jsonl
```

### **edit_config.py**
Interactive config.json editor.

**Usage:**
```bash
python3 edit_config.py
```

---

## 🧪 Testing Scripts

### **test_model.py**
Tests trained model with sample prompts.

**Usage:**
```bash
python3 test_model.py --model current_model/ --prompt "Your question"
```

### **test_specific.py**
Tests specific model functionality.

**Usage:**
```bash
python3 test_specific.py
```

### **interactive_train.py** (20K)
Interactive training session (not daemon).

**Usage:**
```bash
python3 interactive_train.py
```

---

## 🗑️ Maintenance Scripts

### **cleanup_checkpoints.sh** (NEW!)
Cleans old checkpoints using graduated retention policy.

**Usage:**
```bash
./cleanup_checkpoints.sh
```

**Retention Policy:**
- **Keeps last 20 checkpoints** (dense recent history, every 100 steps)
- **Keeps every 1000th checkpoint** before that (sparse older history)
- **Deletes everything else**

**Example:**
- Have checkpoints: 100, 200, 300 ... 16600, 16700
- Keep: 16500-16700 (last 20, every 100)
- Keep: 1000, 2000, 3000 ... 16000 (every 1000th)
- Delete: 100-900, 1100-1900, 2100-2900, etc.

**What it does:**
- Analyzes all checkpoints
- Shows what will be kept/deleted
- Calculates space to free
- Prompts for confirmation
- Performs deletion

**Benefits:**
- Preserves recent training history (last 2000 steps)
- Maintains older reference points (every 1000 steps)
- Frees significant disk space (~500GB typically)

**Run when:**
- Disk space low
- More than 30 checkpoints accumulated
- After training large batches

### **maintenance.sh** (NEW!)
Automated maintenance tasks.

**Usage:**
```bash
./maintenance.sh
```

**What it does:**
- Checks checkpoint count
- Archives old logs (> 7 days)
- Cleans temporary files
- Shows disk usage
- Checks service status
- Reports memory usage

**Run when:**
- Weekly maintenance
- After training sessions
- When system feels slow

### **memory_monitor.sh**
Monitors RAM usage and alerts on high usage.

**Usage:**
```bash
./memory_monitor.sh &
tail -f memory_alerts.log
```

---

## 🚀 Launcher Scripts

### **launch_detailed_monitor.sh**
Launches detailed monitor in background.

### **launch_web_ui.sh**
Launches web UI services.

### **restart_web_ui.sh**
Restarts web monitoring services.

### **monitor_training.sh**
Displays training status in terminal.

---

## 🗂️ Support Scripts

### **model_db.py**
Model database for tracking training runs.

### **training_status.py**
Training status tracking utilities.

### **time_estimator.py**
Training time estimation.

### **detail_collector.py**
Collects detailed training metrics.

### **desktop_notifier.py**
Desktop notifications for training events.

### **streaming_trainer.py**
Streaming data training support.

### **training_web_ui.py**
Alternative web UI implementation.

### **web_ui_gradio.py**
Gradio web interface.

### **convert_format.py**
Data format conversion utilities.

---

## 📋 Quick Command Reference

### Start Everything
```bash
# 1. Start monitoring
nohup python3 enhanced_monitor.py > /dev/null 2>&1 &
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &
nohup python3 memory_stats_api.py > /dev/null 2>&1 &

# 2. Start training
rm -f .stop
nohup python3 training_daemon.py --base-dir $(pwd) > training_output.log 2>&1 &
```

### Stop Everything
```bash
# Graceful stop
touch .stop

# Or kill all
pkill -f "training_daemon|launch_live_monitor|enhanced_monitor|memory_stats"
```

### Check Status
```bash
# What's running?
ps aux | grep python | grep -v grep

# Training status
cat status/training_status.json | jq

# Memory status
curl -s http://localhost:8081/api/memory_stats | jq
```

### Maintenance
```bash
# Weekly maintenance
./maintenance.sh

# Clean checkpoints (when needed)
./cleanup_checkpoints.sh

# Archive old logs
find logs/ -name "*.log" -mtime +7 -exec mv {} logs/archive/ \;
```

---

## 🎯 Which Script to Use?

### I want to...

**Train a model:**
→ Drop data in inbox/, daemon handles it
→ Or: `python3 train.py --dataset data.jsonl`

**Monitor training:**
→ Open http://localhost:8080/live_monitor_ui.html
→ Or: `./monitor_training.sh`

**Test trained model:**
→ `python3 test_model.py --model current_model/`

**Validate data:**
→ `python3 validator.py inbox/data.jsonl`

**Free disk space:**
→ `./cleanup_checkpoints.sh`

**System maintenance:**
→ `./maintenance.sh`

**Merge adapter:**
→ `python3 consolidate_model.py`

**Check memory:**
→ `./memory_monitor.sh &`
→ Or: `curl http://localhost:8081/api/memory_stats`

---

## ⚠️ Important Notes

1. **Don't run multiple training_daemon.py** - Only one instance!
2. **Monitors can run concurrently** - All 3 monitors can run together
3. **Cleanup is manual** - Run cleanup_checkpoints.sh when needed
4. **Logs rotate daily** - Old logs should be archived periodically

---

## 🔗 Related Documentation

- [README.md](README.md) - Main documentation
- [QUICK_START.md](QUICK_START.md) - Getting started
- [docs/guides/CONFIG_GUIDE.md](docs/guides/CONFIG_GUIDE.md) - Configuration
- [docs/guides/TROUBLESHOOTING.md](docs/guides/TROUBLESHOOTING.md) - Problem solving

---

**Last Updated:** 2025-11-12
