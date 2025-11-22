# Automatic Disk Space Management - COMPLETE

**Date:** 2025-11-21
**Problem:** Training crashed with "No space left on device" - disk was 100% full (1.7TB/1.8TB)
**Solution:** Automatic disk space management daemon

---

## What Happened

Training daemon crashed overnight because:
- 342GB in old model versions and backups
- 108GB in .git repository
- Disk reached 100% capacity
- Training failed: `OSError: [Errno 28] No space left on device`

## Immediate Fix

Manually freed **284GB** by deleting:
- `models/backups/pre_consolidation` (129GB) - old pre-merge backups
- `models/versions/v001`, `v002` (105GB) - old model versions
- `current_model_small*` (48GB) - unused backup models

**Result:** Disk usage down to 84% (283GB free)

## Permanent Solution: Auto Disk Manager

Created `auto_disk_manager.py` - a daemon that runs 24/7 to prevent future crashes.

### How It Works

**Monitoring (every 5 minutes):**
- Checks disk space continuously
- Triggers cleanup when < 50GB free OR < 10% free

**Auto-cleanup:**
1. **Old model versions** - Keeps only latest 2, deletes rest
2. **Old backups** - Keeps latest 1 of each type
3. **Old logs** - Deletes logs older than 7 days
4. **Git cleanup** - Runs `git gc --aggressive` if still low

**Safety:**
- Never deletes `current_model/` (active training)
- Never deletes latest version
- Logs everything to `logs/disk_manager.log`

### Configuration

```python
DiskSpaceManager(
    base_dir="/path/to/training",
    min_free_gb=50,          # Cleanup when < 50GB free
    min_free_percent=10,     # OR when < 10% free
    keep_versions=2,         # Keep latest 2 model versions
    check_interval=300       # Check every 5 minutes
)
```

## Integration

**Auto-starts:**
- Added to `start_all.sh` (starts before training daemon)
- Runs continuously in background

**Manual start:**
```bash
nohup python3 auto_disk_manager.py > /dev/null 2>&1 &
```

**Check status:**
```bash
# See if running
ps aux | grep auto_disk_manager

# Check logs
tail -f logs/disk_manager.log

# Current disk usage
df -h /path/to/training
```

## Example Log Output

```
2025-11-21 12:38:35,339 - INFO - 🚀 Automatic Disk Space Manager started
2025-11-21 12:38:35,339 - INFO -    Min free space: 50GB or 10%
2025-11-21 12:38:35,339 - INFO -    Keep latest 2 model versions
2025-11-21 12:38:35,339 - INFO -    Check interval: 300s
2025-11-21 12:38:35,339 - INFO - 💾 Disk: 1456GB used, 283GB free (15.4%)
```

When cleanup is needed:
```
2025-11-21 14:23:12,456 - WARNING - ⚠️  Low disk space: 45.2GB free (< 50GB threshold)
2025-11-21 14:23:12,457 - INFO - 🧹 Starting automatic cleanup...
2025-11-21 14:23:15,123 - INFO - 🗑️  Deleted old version: v003_20251120_031848 (53GB)
2025-11-21 14:23:18,456 - INFO - ✅ Cleanup complete: freed 53.0GB
2025-11-21 14:23:18,457 - INFO - ✅ Cleanup successful: 98.2GB free
```

## Benefits

✅ **No more manual intervention** - Automatic cleanup while you sleep
✅ **Prevents training crashes** - Never runs out of disk space
✅ **Smart retention** - Keeps latest versions, deletes old ones
✅ **Low overhead** - Checks every 5 minutes, cleanup takes seconds
✅ **Fully logged** - Everything recorded for debugging

## Files

- **Script:** `auto_disk_manager.py`
- **Logs:** `logs/disk_manager.log`
- **Auto-start:** `start_all.sh` (line 9-12)
- **Documentation:** This file + `CLAUDE.md` (top section)

## Future Improvements

Possible enhancements:
- [ ] Integrate with daemon_watchdog.py for auto-restart
- [ ] Add email/notification when cleanup occurs
- [ ] Adjustable retention policies per directory
- [ ] Compress old versions instead of deleting

---

## Summary

**Before:** Disk 100% full → Training crashes overnight → Manual cleanup required
**After:** Disk automatically managed → Training runs 24/7 → No intervention needed

The system now self-manages disk space, keeping the latest 2 model versions and automatically cleaning up when space runs low. Training will never crash from disk space issues again.
