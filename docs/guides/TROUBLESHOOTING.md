# Training Troubleshooting Guide

## Current Issue: Training Failing Silently

### What's Happening
- Training starts but fails after ~2 minutes
- No error details in daemon log (stdout not captured)
- File was deleted by old daemon code
- Need to restart with updated code

### Immediate Actions

#### 1. Stop Current Daemon (Multiple Instances Running!)
```bash
pkill -f "training_daemon.py"
```

#### 2. Regenerate Training Data (File Was Deleted)
```bash
cd /home/user/leo_composition_system
python3 -m generators.training_pipeline --count 100000 --seed 42
python3 /path/to/training/convert_leo_data.py \
  outputs/training_runs/*/training_samples.jsonl \
  /path/to/training/inbox/leo_100k_$(date +%Y%m%d).jsonl
```

#### 3. Start Updated Daemon
```bash
cd /path/to/training
python3 training_daemon.py --base-dir /path/to/training &
```

#### 4. Monitor Training
Open http://localhost:7860 and go to "📊 Training Logs" tab
- Shows real-time daemon log
- Auto-refreshes every 5 seconds
- Shows GPU usage and training PID

### Updated Features

✅ **Daemon now:**
- Only deletes files after SUCCESSFUL training
- Keeps failed files for retry
- Better error logging (full tracebacks)

✅ **Web UI now has:**
- Training Logs tab with real-time monitoring
- Daemon log viewer (last 100 lines)
- Auto-refresh every 5 seconds

### Common Failure Causes

1. **Model Path Wrong**
   - Check config.json `model_path` setting
   - Should point to actual model directory

2. **OOM (Out of Memory)**
   - Reduce batch_size in config.json
   - Reduce lora_r value

3. **Dataset Format Issues**
   - Messages format required: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
   - Validation happens unless --skip-validation is set

4. **Missing Dependencies**
   - Check if transformers, peft, torch are installed

### How to Debug

**Option 1: Web UI (Easiest)**
```bash
# Open browser to http://localhost:7860
# Go to "Training Logs" tab
# Watch real-time output
```

**Option 2: Command Line**
```bash
# Watch daemon log
tail -f /path/to/training/logs/daemon_*.log

# Check GPU
watch -n 1 nvidia-smi

# Check training process
ps aux | grep train.py
```

**Option 3: Manual Test (Bypass Daemon)**
```bash
cd /path/to/training
python3 train.py \
  --dataset inbox/leo_100k_*.jsonl \
  --model /path/to/training/model \
  --output-dir test_output \
  --epochs 1 \
  --skip-validation \
  --yes
```
This will show ALL output and help identify the exact failure point.

### Next Steps

Once training works:
1. Training runs for 1 epoch on 100k examples
2. Takes ~4-8 hours on RTX 4090 (depends on config)
3. Model saved to current_model/
4. Daily snapshot created automatically at 3 AM
5. Desktop notification on completion or crash
