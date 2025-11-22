# Memory Monitoring Quick Reference

## 🚨 Quick Alert Thresholds

| Metric | Safe | Caution | Critical |
|--------|------|---------|----------|
| **System RAM** | < 43 GB (70%) | 43-52 GB (70-85%) | > 52 GB (85%) |
| **Training Process** | < 30 GB | 30-40 GB | > 40 GB |
| **OOM Risk** | LOW 🟢 | MEDIUM 🟡 | HIGH 🔴 |

## 📍 Where to Look

### Live Monitor UI: http://localhost:8080/live_monitor_ui.html

**Top Status Bar:**
```
[RAM: X.X GB ↑↓]  ← Color-coded, shows trend
```

**Hardware Status Section:**
```
💾 System RAM - X.X GB / 61.9 GB (XX%)
├─ Used RAM: X.X GB        ← Total system usage
├─ Training Process: X.X GB ← Just your training script
├─ Available RAM: X.X GB    ← Free memory
└─ OOM Risk: LOW/MEDIUM/HIGH
```

## 🎨 Color Code

| Color | Meaning | Action |
|-------|---------|--------|
| 🟢 **Green** | Normal (< 70%) | Continue training |
| 🟡 **Yellow** | Warning (70-85%) | Monitor closely |
| 🔴 **Red Flashing** | Critical (> 85%) | Stop training NOW! |

## ⚡ Quick Actions

### If RAM > 85% (RED):
```bash
# 1. Stop training gracefully
touch /path/to/training/.stop

# 2. Check what's using memory
ps aux --sort=-%mem | head -10

# 3. Wait for training to stop, then restart
# Training will automatically resume from checkpoint
```

### If Training Process > 40 GB:
**Memory leak detected!** The memory fixes should prevent this, but if it happens:
```bash
# Kill and restart daemon
pkill -f training_daemon
cd /path/to/training
nohup python3 training_daemon.py --base-dir $(pwd) > /dev/null 2>&1 &
```

### Check Memory History:
```bash
# Via memory monitor log
tail -50 memory_alerts.log

# Via system logs
journalctl -p err --since "1 hour ago" | grep -i oom
```

## 🔍 Manual Check Commands

```bash
# Quick system memory check
free -h

# Training process memory
ps aux | grep "python.*train" | awk '{print $6/1024 " MB"}'

# API endpoint (for scripts)
curl -s http://localhost:8081/api/memory_stats | jq '{used_gb, training_process_gb, oom_risk}'
```

## 📊 What's Normal?

**During Dataset Loading/Tokenization:**
- System RAM: 15-25 GB
- Training Process: 8-15 GB
- Duration: 1-2 minutes per batch

**During Active Training:**
- System RAM: 10-20 GB
- Training Process: 6-10 GB
- Stable (not growing)

**Between Batches:**
- System RAM: < 15 GB
- Training Process: < 8 GB
- Should drop after each batch completes

## 🚩 Warning Signs

**Memory Leak Indicators:**
1. Training process grows steadily (adds 1+ GB every 10 minutes)
2. System RAM doesn't decrease between batches
3. Swap usage starts increasing
4. OOM risk escalates from LOW → MEDIUM → HIGH

**If You See This Pattern:**
- The memory leak fix should prevent this
- If it still happens, report it as a bug
- Workaround: Restart daemon between large batches

## 🎯 Monitoring Best Practices

1. **Watch during tokenization** - highest memory usage
2. **Check between batches** - should drop
3. **Monitor training process separately** - isolates leaks
4. **Set up alerts** - run `memory_monitor.sh` in background
5. **Export data regularly** - keeps history for analysis

## 📱 Mobile/Remote Access

If accessing from another machine:
```
http://YOUR_IP:8080/live_monitor_ui.html
http://YOUR_IP:8081/api/memory_stats
```

Make sure firewall allows ports 8080 and 8081.

## 🛠️ Services to Keep Running

```bash
# Check all are up
ps aux | grep -E "(memory_stats_api|launch_live_monitor)" | grep -v grep

# If any missing, restart:
cd /path/to/training
nohup python3 memory_stats_api.py > /dev/null 2>&1 &
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &
```

## 📈 Export Data Includes Memory

Press **E** or click 💾 Export button:
```json
{
  "systemRAM": "14.0 GB",
  "trainingProcessRAM": "6.1 GB",
  "oomRisk": "LOW",
  ...
}
```

## 🔔 Remember

- **Memory fixes applied** - Leak should not occur anymore
- **Monitoring is preventive** - Catches issues before crash
- **Status bar always visible** - Scroll anywhere, still see RAM
- **Health indicator prioritizes RAM** - Shows critical issues first

---

**Need Help?** Check `UI_IMPROVEMENTS_SUMMARY.md` for full details.
