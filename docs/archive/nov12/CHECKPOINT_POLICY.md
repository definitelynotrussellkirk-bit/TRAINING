# Checkpoint Retention Policy

**Last Updated:** 2025-11-12

This document explains the checkpoint management strategy for continuous training.

---

## 📋 The Problem

With continuous training saving checkpoints every 100 steps:
- **169 checkpoints** = 614GB disk space
- Training for months would accumulate thousands of checkpoints
- Running out of disk space would crash training

**Solution:** Graduated retention policy that keeps recent history dense but thins older checkpoints.

---

## 🎯 Retention Policy

### Strategy: 3-Tier Graduated Retention

```
Checkpoint Timeline (Example with 172 checkpoints):
│
├─ Tier 3 (Oldest - Sparse)
│  ├─ checkpoint-1000    ← KEEP (every 1000th)
│  ├─ checkpoint-1100    ← Delete
│  ├─ ...
│  ├─ checkpoint-2000    ← KEEP (every 1000th)
│  ├─ ...
│  ├─ checkpoint-13000   ← KEEP (every 1000th)
│  ├─ checkpoint-13100   ← Delete
│  ├─ checkpoint-13200   ← Tier 2 starts
│  │
├─ Tier 2 (Medium - Every Other)
│  ├─ checkpoint-13200   ← KEEP (even 200)
│  ├─ checkpoint-13300   ← Delete (odd)
│  ├─ checkpoint-13400   ← KEEP (even 200)
│  ├─ checkpoint-13500   ← Delete (odd)
│  ├─ ...
│  ├─ checkpoint-15000   ← KEEP (even 200)
│  ├─ checkpoint-15100   ← Tier 1 starts
│  │
└─ Tier 1 (Recent - Dense)
   ├─ checkpoint-15100   ← KEEP (all)
   ├─ checkpoint-15200   ← KEEP (all)
   ├─ ...
   ├─ checkpoint-16900   ← KEEP (all)
   └─ checkpoint-17000   ← KEEP (all, newest)
```

### Rules

1. **Tier 1 - Last 20 checkpoints:** Keep ALL (100% density, every 100 steps)
   - Example: steps 15100-17000 (newest)
   - Reason: Dense recent history for detailed analysis
   - Space: ~72GB (20 × 3.6GB)

2. **Tier 2 - Next 20 checkpoints:** Keep every other (50% density, every 200 steps)
   - Example: 13200, 13400, 13600 ... 14800, 15000
   - Reason: Medium-term history at reduced density
   - Space: ~36GB (10 × 3.6GB)

3. **Tier 3 - Older checkpoints:** Keep every 1000th step (sparse, <10% density)
   - Example: 1000, 2000, 3000 ... 12000, 13000
   - Reason: Long-term reference points, minimal space
   - Space: ~47GB (13 × 3.6GB)

4. **Everything else:** DELETE
   - All checkpoints not matching above rules
   - Reason: Redundant with nearby checkpoints

---

## 📊 Space Savings Example

### Current State (172 checkpoints)
```
Total: 172 checkpoints × ~3.6GB = 619GB
```

### After 3-Tier Cleanup
```
Tier 1 (last 20):      20 checkpoints × 3.6GB = 72GB
Tier 2 (next 20@50%):  10 checkpoints × 3.6GB = 36GB
Tier 3 (older@1:1000): 13 checkpoints × 3.6GB = 47GB
--------------------------------------------------
Total kept:            43 checkpoints = 155GB
Freed:                129 checkpoints = 464GB (75% reduction!)
```

---

## 🔄 When to Run Cleanup

### Automatic (Recommended)
Set up weekly cron job:
```bash
# Every Sunday at 4 AM
0 4 * * 0 cd /path/to/training && ./cleanup_checkpoints.sh -y >> logs/cleanup.log 2>&1
```

### Manual
Run when:
- More than 30 checkpoints accumulated
- Disk space warning
- After training large batches
- Before starting new training run

```bash
cd /path/to/training
./cleanup_checkpoints.sh
```

---

## 🛡️ Safety Features

### What's Protected
- ✅ **Last 20 checkpoints** - Always preserved
- ✅ **Every 1000th checkpoint** - Long-term references preserved
- ✅ **Current training** - Script checks if training active
- ✅ **Confirmation prompt** - Must confirm before deletion

### What Can Be Recovered
If you delete too much:
- Recent checkpoints (last 20): Can resume training from any
- Daily snapshots: Check `snapshots/` directory
- Consolidated models: Check `consolidated_models/` directory

### What Cannot Be Recovered
- Deleted intermediate checkpoints (100, 200, 300, etc.)
- But these are redundant with nearby checkpoints at 1000, 2000, etc.

---

## 📈 Long-Term Benefits

### For 6 Months of Training

**Without policy:**
```
~50,000 steps ÷ 100 = 500 checkpoints × 3.6GB = 1.8TB
```

**With 3-tier policy:**
```
Tier 1: 20 checkpoints × 3.6GB = 72GB
Tier 2: 10 checkpoints × 3.6GB = 36GB
Tier 3: 50 checkpoints × 3.6GB = 180GB
Total: 80 checkpoints = 288GB (84% savings!)
```

### For 1 Year of Training

**Without policy:**
```
~100,000 steps ÷ 100 = 1,000 checkpoints × 3.6GB = 3.6TB
```

**With 3-tier policy:**
```
Tier 1: 20 checkpoints × 3.6GB = 72GB
Tier 2: 10 checkpoints × 3.6GB = 36GB
Tier 3: 100 checkpoints × 3.6GB = 360GB
Total: 130 checkpoints = 468GB (87% savings!)
```

---

## 🎓 Why This Policy?

### Tier 1: Dense Recent History (Last 20)
**Why:** Need for detailed analysis and debugging
- Compare recent performance trends step-by-step
- Rollback if latest training degraded
- Analyze what changed in last ~2000 steps
- Quick resume from very recent point
- A/B test between very close checkpoints

### Tier 2: Medium-Term History (Next 20 at 50%)
**Why:** Bridge between dense and sparse
- Track medium-term trends (2000-4000 steps back)
- Still granular enough for rollback
- 200-step intervals adequate for analysis
- Reduces storage by 50% vs keeping all

### Tier 3: Long-Term Reference (Every 1000th)
**Why:** Sparse reference points for historical tracking
- Track improvement over months
- Compare to older model versions
- Recover from catastrophic issues
- See long-term training trends
- 1000-step intervals sufficient for coarse comparison

### Delete Intermediate (Everything else)
**Why:** Redundant and wasteful
- checkpoint-100 vs checkpoint-200: < 0.1% training difference
- checkpoint-13300 vs checkpoint-13200/13400: interpolatable
- Not worth 3.6GB per checkpoint for tiny deltas

---

## 🔧 Customizing the Policy

Edit `cleanup_checkpoints.sh` to adjust:

```bash
KEEP_LAST=20          # Tier 1: Keep last N checkpoints (100% density)
TIER2_COUNT=20        # Tier 2: Keep next N checkpoints at 50% density
TIER3_INTERVAL=1000   # Tier 3: Keep every Nth step (sparse oldest)
```

**Examples:**

### More Conservative (Keep more checkpoints)
```bash
KEEP_LAST=50          # Tier 1: Last 5000 steps dense
TIER2_COUNT=30        # Tier 2: Next 3000 steps at 50%
TIER3_INTERVAL=500    # Tier 3: Every 500th older checkpoint
```

### More Aggressive (Save more space)
```bash
KEEP_LAST=10          # Tier 1: Last 1000 steps dense
TIER2_COUNT=10        # Tier 2: Next 1000 steps at 50%
TIER3_INTERVAL=5000   # Tier 3: Every 5000th older checkpoint
```

### Balanced (Default - Recommended)
```bash
KEEP_LAST=20          # Tier 1: Last 2000 steps dense (20 × 100)
TIER2_COUNT=20        # Tier 2: Next 2000 steps at 50% (20 × 100)
TIER3_INTERVAL=1000   # Tier 3: Every 1000th step
```

---

## 📝 Running the Cleanup

### Interactive Mode (Default)
```bash
./cleanup_checkpoints.sh
```

Shows:
1. Current checkpoint count
2. What will be kept/deleted
3. Space to be freed
4. Prompts for confirmation

### Automatic Mode (For Cron)
```bash
./cleanup_checkpoints.sh -y  # Auto-confirm
```

Add to crontab:
```bash
crontab -e

# Add this line (runs weekly Sunday 4 AM):
0 4 * * 0 cd /path/to/training && ./cleanup_checkpoints.sh -y >> logs/cleanup.log 2>&1
```

---

## 🚨 Emergency Recovery

### If You Deleted Too Much

**Option 1: Use Daily Snapshot**
```bash
# Copy from snapshot (taken at 3 AM daily)
cp -r snapshots/2025-11-11/ current_model/
```

**Option 2: Use Consolidated Model**
```bash
# Copy from consolidated backup
cp -r consolidated_models/20251111_120000/ current_model/
```

**Option 3: Start Fresh**
```bash
# Remove current_model, training will start from base
rm -rf current_model/
```

### If Training Crashes After Cleanup

Training automatically resumes from latest available checkpoint:
- Looks for highest numbered checkpoint
- Loads that checkpoint
- Continues training from there
- No manual intervention needed

---

## ✅ Best Practices

1. **Run maintenance weekly**
   ```bash
   ./maintenance.sh
   ```

2. **Check before cleanup**
   ```bash
   # See what would be deleted
   ./cleanup_checkpoints.sh
   # Review, then confirm
   ```

3. **Monitor disk space**
   ```bash
   df -h /path/to/training
   ```

4. **Keep daily snapshots**
   - Automatic at 3 AM via daemon
   - Backup before major cleanup

5. **Test recovery periodically**
   ```bash
   # Verify checkpoints work
   python3 test_model.py --model current_model/checkpoint-16000
   ```

---

## 📚 Related Documentation

- [maintenance.sh](maintenance.sh) - Automated maintenance script
- [cleanup_checkpoints.sh](cleanup_checkpoints.sh) - Checkpoint cleanup script
- [SCRIPTS_README.md](SCRIPTS_README.md) - All scripts reference
- [docs/technical/CONTINUOUS_TRAINING_GUIDE.md](docs/technical/CONTINUOUS_TRAINING_GUIDE.md) - How checkpoints work

---

**Questions?**
- Check `TROUBLESHOOTING.md` for issues
- Review `README.md` for system overview

---

**Last Updated:** 2025-11-12
**Policy Version:** 1.0 (Graduated Retention)
