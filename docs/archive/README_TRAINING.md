# Training System Documentation

**Quick Links**:
- 📊 **Enhanced Monitor**: http://localhost:8082 (RECOMMENDED)
- 🔬 **Detailed Monitor**: http://localhost:8081
- 📈 **Basic Monitor**: http://localhost:8080

## Documentation Files

### 🚨 [TRAINING_STATUS.md](./TRAINING_STATUS.md)
**START HERE** - Current training status, configuration, and what crashed.

**Contains**:
- Current training configuration (model, dataset, hyperparameters)
- Progress before crash (step 624, 5% complete)
- The bug that caused the crash
- How to restart training

### 🔧 [BUG_FIX_INSTRUCTIONS.md](./BUG_FIX_INSTRUCTIONS.md)
**NEXT** - Step-by-step instructions to fix the bug and restart.

**Contains**:
- Exact code changes needed
- Commands to apply the fix
- Verification steps
- Restart instructions

### 📊 [MONITOR_GUIDE.md](./MONITOR_GUIDE.md)
**REFERENCE** - Complete guide to all monitoring dashboards.

**Contains**:
- Guide to each monitor (ports 8080, 8081, 8082)
- What each dashboard shows
- How to start/stop/restart monitors
- Troubleshooting tips

## Quick Start

### To Fix and Restart Training:

1. **Read the status**:
   ```bash
   cat /path/to/training/TRAINING_STATUS.md
   ```

2. **Apply the fix**:
   ```bash
   cd /path/to/training

   # Fix line 64
   sed -i '64s/example\["messages"\]\[0\]\["content"\]/str(example["messages"][0]["content"])/' live_monitor.py

   # Fix line 65
   sed -i '65s/example\["messages"\]\[1\]\["content"\]/str(example["messages"][1]["content"])/' live_monitor.py
   ```

3. **Restart daemon**:
   ```bash
   # Stop current training
   touch /path/to/training/.stop
   sleep 10

   # Remove stop file
   rm -f /path/to/training/.stop

   # Start fresh
   cd /path/to/training
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   nohup python3 training_daemon.py --base-dir /path/to/training > /tmp/daemon.log 2>&1 &
   ```

4. **Monitor at**: http://localhost:8082

## What Happened?

Training was going well (loss decreasing from 2.14 → 0.55) but crashed at step 624 because:

```python
# This line crashed:
expected = example['messages'][1]['content'].strip()

# Because some training examples have integers in 'content'
# AttributeError: 'int' object has no attribute 'strip'
```

## The Fix

Convert content to string before calling `.strip()`:

```python
# Fixed version:
expected = str(example['messages'][1]['content']).strip()
```

## Training Configuration

- **Model**: Qwen2.5-7B-Instruct
- **Method**: QLoRA (4-bit quantization)
- **Dataset**: 100,000 examples
- **Batch Size**: 1 (with 8x gradient accumulation = effective batch 8)
- **Steps**: 12,488 total
- **Time**: ~12 hours estimated
- **GPU**: RTX 4090, using ~8GB with QLoRA

## Monitor Summary

| Port | Purpose | Update Frequency |
|------|---------|------------------|
| 8082 | **Real-time metrics + charts** | Every 2 seconds |
| 8081 | Token-by-token analysis | Every 625 steps (fix to 10) |
| 8080 | Basic status | Real-time |

## Files Structure

```
/path/to/training/
├── README_TRAINING.md          ← You are here
├── TRAINING_STATUS.md          ← Current status & config
├── BUG_FIX_INSTRUCTIONS.md     ← How to fix & restart
├── MONITOR_GUIDE.md            ← Monitor documentation
│
├── training_daemon.py          ← Main training daemon
├── train.py                    ← Training script
├── live_monitor.py             ← Monitor (needs fix!)
├── detailed_monitor.py         ← Port 8081 monitor
├── enhanced_monitor.py         ← Port 8082 monitor (NEW!)
│
├── inbox/                      ← Training data
│   ├── leo_100k_compositional_fixed.jsonl  (100k examples)
│   └── leo_10k_qlora.jsonl                  (10k examples)
│
├── model_qwen25/               ← Qwen2.5-7B-Instruct model
├── current_model/              ← Training output
│   └── status/
│       └── training_detail.json  ← Status file (all monitors read this)
│
└── logs/                       ← Log files
```

## Monitoring Files

- **Daemon log**: `/tmp/daemon.log`
- **Enhanced monitor log**: `/tmp/enhanced_monitor.log`
- **Detailed monitor log**: `/tmp/detailed_monitor.log`
- **Status JSON**: `/path/to/training/current_model/status/training_detail.json`

## Commands Cheat Sheet

```bash
# Check daemon status
ps aux | grep training_daemon

# Watch daemon log
tail -f /tmp/daemon.log

# Check GPU
nvidia-smi

# Check training status
cat /path/to/training/current_model/status/training_detail.json

# Stop training
touch /path/to/training/.stop

# Check monitors
netstat -tlnp | grep -E "8080|8081|8082"

# Kill all monitors
pkill -f "monitor.py"
```

## Next Steps

1. ✅ Read [TRAINING_STATUS.md](./TRAINING_STATUS.md)
2. ✅ Follow [BUG_FIX_INSTRUCTIONS.md](./BUG_FIX_INSTRUCTIONS.md)
3. ✅ Open http://localhost:8082 to watch training
4. ✅ Wait ~12 hours for completion
5. ✅ Check results in `/path/to/training/current_model/`

---

**Last Updated**: 2025-11-07 01:20 AM
**Status**: ⚠️ Ready to fix and restart
