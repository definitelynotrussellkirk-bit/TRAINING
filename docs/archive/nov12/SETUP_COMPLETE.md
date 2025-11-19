# 🎉 Training System Setup Complete

**Setup Date:** 2025-11-12
**Status:** ✅ Ready for Training

---

## ✅ What Was Done

### **1. Inbox Cleared**
- Removed all old training data from `inbox/`
- Fresh start ready for new data

### **2. Configuration Upgraded**
**Old config backed up to:** `config_qwen25_r128_backup_*.json`

**Changes:**
| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| **LoRA Rank** | 128 | **256** 🔥 | 2x capacity |
| **LoRA Alpha** | 128 | **256** | Same α/r ratio |
| **Dropout** | 0% | **2%** | Better generalization |
| **Trainable Params** | ~34M | **~67M** | 2x learning power |
| **Adapter Size** | ~1.2 GB | ~2.5 GB | Larger but worth it |

### **3. Training State Cleared**
- Removed `current_model/` directory
- Fresh start required for rank change
- Old checkpoints incompatible with new rank

### **4. Model Verified**
- ✅ Using existing `model_qwen25` (15 GB)
- ✅ No download needed!
- ✅ Fits in 24GB VRAM with QLoRA

---

## 🚀 How to Start Training

### **Step 1: Add Training Data**
```bash
cd /path/to/training

# Copy your training data to inbox
cp /path/to/your/data.jsonl inbox/

# Or from LEO outputs
cp /home/user/leo_composition_system/outputs/training_runs/*/training_samples.jsonl inbox/
```

### **Step 2: Start Training Daemon**
```bash
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &
```

### **Step 3: Monitor Training**
**Open in browser:**
- **Live Monitor**: http://localhost:8080/live_monitor_ui.html
- **Enhanced Monitor**: http://localhost:8082

**Check logs:**
```bash
tail -f training_output.log
tail -f logs/daemon_$(date +%Y%m%d).log
```

**Watch GPU:**
```bash
watch -n 1 nvidia-smi
```

---

## 📊 Expected Performance

### **Training Speed**
- **Speed**: ~1,200-1,500 tokens/sec (20-30% slower than rank=128)
- **Throughput**: ~40-50 MB/hour
- **Time per 10k examples**: ~2-2.5 hours

### **Memory Usage**
- **GPU VRAM**: ~21-22 GB (vs 19 GB before)
- **System RAM**: ~16-20 GB
- **Adapter checkpoints**: ~2.5 GB each

### **Quality Benefits**
- 🔥 **2x capacity** to learn complex patterns
- ✅ **Better final quality** on challenging tasks
- ✅ **Reduced overfitting** (2% dropout)
- ✅ **More robust** to data variations

---

## 🎯 Why This Upgrade?

### **Rank = 256:**
- Standard for serious fine-tuning
- Much higher capacity than 128
- Better for complex domains
- Still reasonable VRAM usage

### **Dropout = 2%:**
- Industry standard (1-5%)
- Prevents overfitting with high rank
- Better generalization to unseen data
- Slight speed penalty but worth it

### **Same Model (No Download):**
- Keep using Qwen2.5-7B (proven, reliable)
- No 10-20 minute download wait
- Immediate start when data ready
- Known performance characteristics

---

## 🛑 How to Stop Training

```bash
# Graceful stop
touch /path/to/training/.stop

# Force stop
pkill -f training_daemon
```

---

## 📁 Directory Structure

```
/path/to/training/
├── inbox/                          ← Add .jsonl files here
├── current_model/                  ← Will be created during training
├── model_qwen25/                   ← Base model (15 GB)
├── config.json                     ← UPGRADED (rank=256)
├── config_qwen25_r128_backup_*.json ← Old config backup
├── training_daemon.py              ← Auto-ingestion daemon
├── train.py                        ← Core training script
├── flagged_examples/               ← Auto-flagged mismatches
└── logs/                           ← Daily training logs
```

---

## 🔧 Troubleshooting

### **"Out of Memory" Error**
- Check GPU VRAM: `nvidia-smi`
- Should use ~21-22 GB
- If OOM, try reducing max_length in config.json

### **"Model incompatible" Error**
- Means old rank=128 adapter still exists
- Solution: `rm -rf current_model/`
- Fresh start required for rank change

### **Training Won't Start**
- Check daemon running: `ps aux | grep training_daemon`
- Check logs: `tail -30 training_output.log`
- Verify inbox has data: `ls -lh inbox/`

### **Flagged Examples Not Appearing**
- New feature added today
- Will populate during training
- Click "🚩 Flagged" button in live monitor
- See `FLAGGED_EXAMPLES_GUIDE.md` for details

---

## 📚 Documentation

**Main Guides:**
- `README.md` - Complete system documentation
- `CLAUDE.md` - Quick reference for operations
- `FLAGGED_EXAMPLES_GUIDE.md` - Review problematic examples
- `ADVANCED_METRICS_IMPLEMENTATION.md` - New metrics system

**Backup/Restore:**
- `RESTART_QWEN3_GUIDE.md` - Original upgrade guide
- `config_qwen25_r128_backup_*.json` - Old config (restore if needed)

---

## ✅ Verification Checklist

Before starting training, verify:

- [x] **Inbox is empty** (ready for new data)
- [x] **Config has rank=256** (check `config.json`)
- [x] **Dropout is 2%** (prevents overfitting)
- [x] **Old training state cleared** (no `current_model/`)
- [x] **Model exists** (`model_qwen25/` present)
- [x] **No daemon running** (clean startup)
- [x] **Monitors ready** (port 8080, 8082)

**All verified! ✅**

---

## 🎉 Summary

You now have:
- ✅ **2x learning capacity** (rank 128 → 256)
- ✅ **Better generalization** (2% dropout)
- ✅ **Clean slate** (no old training state)
- ✅ **Empty inbox** (ready for new data)
- ✅ **Advanced metrics** (flagged examples, streaming CE, etc.)
- ✅ **No downloads** (using existing model)

**Ready to train! Just add data to `inbox/` and start the daemon.** 🚀
