# QUICK START GUIDE

**For: Starting training with full safety protections**

---

## Step 1: Health Check (30 seconds)

```bash
cd /path/to/training

# Check system health
python3 comprehensive_health_check.py

# Verify checkpoints (if resuming)
python3 verify_checkpoint_resume.py

# Validate config
python3 config_validator.py
```

**Expected:** All checks pass. If issues, fix before proceeding.

---

## Step 2: Start Services (1 minute)

```bash
# Start training daemon
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# Start monitors
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &  # Port 8080
nohup python3 memory_stats_api.py > /dev/null 2>&1 &      # Port 8081

# Start watchdog (auto-restart protection)
nohup python3 daemon_watchdog.py > logs/watchdog.log 2>&1 &

# OPTIONAL: Anti-stuck monitor (hang detection)
nohup python3 anti_stuck_monitor.py > logs/anti_stuck.log 2>&1 &
```

**Verify:**
```bash
ps aux | grep -E "training_daemon|daemon_watchdog" | grep -v grep
# Should see both processes running
```

---

## Step 3: Add Training Data (10 seconds)

```bash
# Copy data to inbox
cp /path/to/your/data.jsonl inbox/

# Daemon auto-processes every 30 seconds
```

---

## Step 4: Monitor Training (Ongoing)

**Web UI:**
```
http://localhost:8080/live_monitor_ui_v2.html
```

**Command line:**
```bash
# Check status
cat status/training_status.json | jq '{step, total, loss}'

# Watch logs
tail -f logs/daemon_$(date +%Y%m%d).log

# Watch watchdog
tail -f logs/watchdog.log
```

---

## Step 5: After Training Completes

```bash
# Clean old checkpoints
python3 safe_checkpoint_cleanup.py --keep 3

# Check for crashes (should be none!)
python3 crash_detector.py

# Consolidate model (optional)
python3 consolidate_model.py --description "What was trained"
```

---

## 🛡️ Safety Protections Active

When you follow this quick start:

✅ **Daemon Watchdog** - Auto-restarts on crash
✅ **Anti-Stuck Monitor** - Detects hangs (optional but recommended)
✅ **Config Locked** - Can't train on wrong version
✅ **Health Validated** - System checked before start
✅ **Checkpoint Verified** - Resume point confirmed

---

## ⚠️ DON'T DO THESE While Training

❌ Change `base_model` in config.json
❌ Delete files from `queue/processing/`
❌ Delete `current_model/checkpoint-*`
❌ Run consolidation
❌ Kill daemon with `kill -9`

---

## ✅ SAFE TO DO While Training

✅ Add files to `inbox/`
✅ Check `status.json`
✅ View logs
✅ Monitor web UI
✅ Check GPU with `nvidia-smi`

---

## 🚨 If Something Goes Wrong

**Training crashed:**
```bash
# 1. Check what happened
python3 crash_detector.py

# 2. Watchdog should auto-restart within 60s
# If not, check: tail -f logs/watchdog.log

# 3. Manual restart if needed
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &
```

**Training stuck:**
```bash
# Anti-stuck monitor detects within 15 min
# Or manually restart:
python3 training_controller.py stop
# Watchdog will auto-restart
```

**Need help:**
```bash
# Full diagnostic
python3 comprehensive_health_check.py

# Read guides
cat SAFEGUARDS_SUMMARY.md
cat CRASH_PREVENTION_GUIDE.md
```

---

## 📊 Expected Behavior

**Normal operation:**
- Status file updates every few seconds
- Step count increases
- Loss generally decreases
- Checkpoints saved every 100 steps
- Evals run every 500 steps (~90 seconds each)

**If watchdog working:**
- Daemon crashes detected within 30s
- Auto-restart within 60s
- Logs to `logs/watchdog.log`

**If anti-stuck working:**
- Hangs detected within 15 min
- Stuck evals detected within 30 min
- Auto-kills stuck process
- Watchdog restarts daemon

---

## 🎯 That's It!

Training is now:
- **Protected** - Auto-restart on crash
- **Monitored** - Hang detection active
- **Safe** - Config locked, checkpoints verified
- **Recoverable** - All edge cases handled

Just monitor the web UI and let it run. System handles the rest!

**For complete details:** See `SAFEGUARDS_SUMMARY.md`
