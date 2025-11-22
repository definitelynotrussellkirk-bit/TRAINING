# Qwen3-VL Training System for LEO Compositional Tasks

**Status:** ✅ Ready to train (Fixed: 2025-11-03)

---

## 🚀 QUICK START

**1. Start the training daemon:**
```bash
cd /path/to/training
bin/launch_training_daemon.sh
```

**2. Monitor training:**
- Web UI: http://localhost:7860
- Live stats: http://localhost:8080/live_monitor_ui.html

**3. Done!** The guarded launcher prevents multiple daemons; training auto-pulls from `inbox/`.

---

## 📖 Documentation

**Start here:**
- **[QUICK_START.md](QUICK_START.md)** ← **READ THIS FIRST**

**Reference:**
- [FIXES_APPLIED.md](FIXES_APPLIED.md) - Technical fixes (9 debug sessions)
- [WEB_UI_GUIDE.md](WEB_UI_GUIDE.md) - Web interface
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

## 🎯 What's This?

Trains **Qwen3 0.6B** on **LEO composition data** using:
- QLoRA (4-bit) - fits in 24GB GPU
- Text-only training - vision layers frozen
- Automated pipeline - just drop files in `inbox/`

**Model Location:** `/path/to/training/consolidated_models/20251119_152444/` (1.5GB, text-only)

**Training Data:** `inbox/leo_10k_qlora.jsonl` (10k samples, ready to go)

---

## ⚡ One-Line Training

```bash
cd /path/to/training && python3 train.py --dataset inbox/leo_10k_qlora.jsonl --model model --output-dir adapters/leo_10k --epochs 1 --use-qlora
```

**Time:** ~1h 12m | **VRAM:** ~19GB / 24GB

---

## 📁 Directory Structure

```
/path/to/training/
├── README.md              ← You are here
├── QUICK_START.md         ← Complete training guide
├── models/Qwen3-0.6B/     ← Base model (1.5GB)
├── inbox/                 ← Drop training data here
│   └── leo_10k_qlora.jsonl
├── adapters/              ← Trained adapters saved here
├── archive/               ← Processed files moved here
├── logs/                  ← Training logs
├── config.json            ← Training config (FIXED)
├── train.py               ← Main trainer (FIXED)
└── training_daemon.py     ← Auto-training daemon
```

---

## ✅ Current Status

- ✅ Model downloaded (Qwen3 0.6B, 1.5GB)
- ✅ Data ready (10k LEO samples)
- ✅ Config fixed (batch_size=1, QLoRA enabled)
- ✅ Data format fixed (handles non-string content)
- ✅ Memory optimized (19GB VRAM usage)
- ✅ System tested and verified
- ✅ GPU clean (696MB used, 23GB free)

---

## 🔄 Generate More Training Data

```bash
# Generate 100k samples
cd /home/user/leo_composition_system
python3 -m generators.training_pipeline --count 100000 --seed 42

# Convert to training format
cd /path/to/training
python3 convert_leo_data.py \
  /home/user/leo_composition_system/outputs/training_runs/*/training_samples.jsonl \
  inbox/leo_100k.jsonl

# Daemon will auto-train when file appears
```

### On-demand SYLLO batches via API
If `skill_syllo_variant/api_server.py` is running you can auto-request fresh packs
whenever the queue is low:

```bash
# Start the API server (example)
python skill_syllo_variant/api_server.py --host 127.0.0.1 --port 8080 \
  --word-db HELPERS/data/word_db.jsonl

# Generate a 20k pack as soon as <=1 files remain in the queue
python3 generate_syllo_batch.py --count 20000 --seed 93001 --threshold 1
```

`generate_syllo_batch.py` writes the JSONL to `inbox/` and immediately queues it
for training. Use `--payload '{...}'` to pass custom parameters (e.g., hard-only
puzzles) straight through to the API.

---

## 🐛 Issues? Check These

1. **Read [QUICK_START.md](QUICK_START.md)** - Complete guide
2. **Check GPU:** `nvidia-smi` (should show ~700MB used)
3. **Check logs:** `tail -f logs/daemon_*.log`
4. **Verify config:** `cat config.json | grep batch_size` (must be 1)
5. **Read [FIXES_APPLIED.md](FIXES_APPLIED.md)** - Known fixes

---

## 💡 Key Settings (DO NOT CHANGE)

```json
{
  "batch_size": 1,              ← REQUIRED (batch_size=2 causes OOM)
  "gradient_accumulation": 8,   ← Maintains effective batch size
  "use_qlora": true            ← REQUIRED (4-bit quantization)
}
```

---

## 🆘 Emergency Commands

```bash
# Stop everything
pkill -f "train.py"
pkill -f "training_daemon"

# Check GPU
nvidia-smi

# Clean GPU
pkill -f python3

# Restart daemon
cd /path/to/training
python3 training_daemon.py --base-dir /path/to/training &
```

---

## 📊 Expected Performance

- **Training time:** 1h 12m for 10k samples
- **VRAM usage:** 19.4 GB / 24.6 GB (79%)
- **GPU utilization:** 99%
- **Speed:** ~3.3 sec/step
- **Adapter size:** ~200-500 MB

---

**Last Updated:** 2025-11-03 14:37 PST
**System:** Ubuntu 24.04, RTX 4090 24GB
**Model:** Qwen3 0.6B (text-only)
**Status:** ✅ Fully operational
