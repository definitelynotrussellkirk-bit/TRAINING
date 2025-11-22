# ⚡ Quick Start - Integrated Training System

**5-Minute Guide to Get Started**

---

## 🚀 START TRAINING (2 minutes)

###1. Start the Integrated Daemon

```bash
cd /path/to/training

# Kill old daemon (if running)
pkill -f training_daemon

# Start new integrated daemon
nohup python3 training_daemon_integrated.py \
  --base-dir /path/to/training \
  > training_output.log 2>&1 &

# Watch it start
tail -f training_output.log
```

### 2. Drop Training Data

```bash
# Normal priority (default)
cp my_data.jsonl inbox/

# High priority (trains immediately)
cp urgent_data.jsonl inbox/priority/
```

### 3. Monitor Progress

```bash
# Check queue
python3 training_queue.py status

# Watch logs
tail -f logs/daemon_$(date +%Y%m%d).log
```

**Done!** Training starts automatically.

---

## 🎮 CONTROL TRAINING (1 minute each)

### Pause (Finish Current Batch, Then Wait)
```bash
python3 training_controller.py pause

# Resume when ready
python3 training_controller.py resume
```

### Stop (Finish Current Batch, Then Exit)
```bash
python3 training_controller.py stop
```

### Skip Current File
```bash
python3 training_controller.py skip
```

### Check Status
```bash
python3 training_controller.py status
python3 training_queue.py status
```

---

## 🔬 CONSOLIDATE MODEL (2 minutes)

After training completes, merge adapter into base model:

```bash
python3 consolidate_model.py \
  --base-dir /path/to/training \
  --description "What I trained on"
```

**Result:**
- ✅ New version created (v001, v002, etc.)
- ✅ Backups verified
- ✅ Evolution data preserved
- ✅ Zero data loss

---

## 📊 VIEW RESULTS

### Check Versions
```bash
python3 model_versioner.py list
```

### View Learning Evolution
```bash
# Open in browser
http://localhost:8080/evolution_viewer.html

# Or check snapshots
ls -lh data/evolution_snapshots/
```

### Restore Previous Version
```bash
python3 model_versioner.py restore v001
```

---

## 🆘 TROUBLESHOOTING

### Daemon Not Running?
```bash
# Check if running
ps aux | grep training_daemon_integrated

# View logs
tail -50 training_output.log

# Restart
pkill -f training_daemon && sleep 2
python3 training_daemon_integrated.py --base-dir /path/to/training
```

### Queue Not Processing?
```bash
# Check queue status
python3 training_queue.py status

# List files in queue
python3 training_queue.py list

# Check for pause signal
ls -la control/.pause
```

### Clear All Signals
```bash
# Remove all control signals
rm -f control/.pause control/.stop control/.skip control/.resume
```

---

## 📚 FULL DOCUMENTATION

- **Integration Guide:** `INTEGRATION_COMPLETE.md`
- **Quick Reference:** `CLAUDE.md`
- **Phase 2 Details:** `PHASE2_MODEL_VERSIONING_COMPLETE.md`
- **Phase 3 Details:** `PHASE3_CONTROL_SYSTEM_COMPLETE.md`

---

## ✅ THAT'S IT!

**You now have:**
- ✅ Graceful pause/resume/stop/skip
- ✅ Priority queue (high/normal/low)
- ✅ Version management with rollback
- ✅ Zero data loss guarantee
- ✅ Learning evolution tracking

**Start training with confidence!** 🚀
