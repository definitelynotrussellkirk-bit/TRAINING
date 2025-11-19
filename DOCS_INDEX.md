# Documentation Index

**Read documents in this order:**

---

## 🚀 Getting Started

1. **[README.md](README.md)** - Overview and quick commands
2. **[QUICK_START.md](QUICK_START.md)** - Complete training guide ← **START HERE**

---

## 📚 Reference Documentation

3. **[FIXES_APPLIED.md](FIXES_APPLIED.md)** - Technical fixes (9 debug sessions)
   - Data format fix (TypeError resolution)
   - Memory optimization (OOM fix)
   - What changed and why

4. **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Configuration options
   - Batch size, learning rate, LoRA settings
   - When to change settings (rarely)

5. **[WEB_UI_GUIDE.md](WEB_UI_GUIDE.md)** - Web interface guide
   - How to use http://localhost:7860
   - Live monitoring tools

6. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues
   - Error messages and solutions
   - System health checks

---

## 📖 Quick Reference

### To Train

```bash
cd /path/to/training
python3 training_daemon.py --base-dir /path/to/training &
```

### To Monitor

- Web UI: http://localhost:7860
- Live stats: http://localhost:8080/live_monitor_ui.html
- Logs: `tail -f logs/daemon_*.log`

### Model Location

**Base Model:** `/path/to/training/model/`
- Qwen3-VL-8B-Thinking (17GB)
- Vision-language model (8.9B params)
- Text-only training (174M trainable params)

### Training Data

**Current:** `/path/to/training/inbox/leo_10k_qlora.jsonl`
- 10,000 LEO composition samples
- Ready to train

**Generate More:**
```bash
cd /home/user/leo_composition_system
python3 -m generators.training_pipeline --count 100000 --seed 42
```

---

## 🔧 Critical Settings

**DO NOT CHANGE THESE:**
- `batch_size: 1` (batch_size=2 causes OOM)
- `use_qlora: true` (required for 24GB GPU)

**Safe to Change:**
- `learning_rate` (default: 2e-4)
- `lora_r` / `lora_alpha` (default: 32)
- `epochs` (default: 1)

---

## 🐛 Debugging

If training fails:

1. Check [FIXES_APPLIED.md](FIXES_APPLIED.md) - Known issues
2. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Solutions
3. Run health check:
   ```bash
   nvidia-smi  # Check GPU
   cat config.json | grep batch_size  # Verify config
   tail -100 logs/daemon_*.log  # Check logs
   ```

---

## 📊 System Status

**Last Verified:** 2025-11-03 14:37 PST

- ✅ Model ready (17GB, Qwen3-VL 8B)
- ✅ Data ready (10k samples)
- ✅ Config fixed (batch_size=1, QLoRA enabled)
- ✅ Code fixed (data format handling)
- ✅ GPU clean (696MB used, 23GB free)
- ✅ System tested (successful training verified)

**Expected Performance:**
- Training time: ~1h 12m for 10k samples
- VRAM usage: ~19GB / 24GB
- GPU utilization: 99%
- Speed: ~3.3 sec/step

---

**Need Help?** Start with [QUICK_START.md](QUICK_START.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
