# Training Recovery Investigation

**Date:** 2025-11-16
**Issue:** Missing Qwen3 model with "3 days of training"
**Status:** Investigation in progress

---

## 🔍 What We Found

### 1. Consolidated Model from Nov 15 (DELETED)
**From daemon logs** (`daemon_20251115.log`):
```
2025-11-15 03:08:50 - Consolidation created:
  - /path/to/training/consolidated_models/20251115_030850
  - /path/to/training/consolidated_backups/20251115_030850
2025-11-15 03:11:42 - ✅ Consolidation completed successfully
```

**Problem:** According to `CLAUDE.md`:
> "Cleaned up 1.2TB of old Qwen 2.5 training artifacts on 2025-11-15"
> "Removed: consolidated_models, consolidated_backups, snapshots, model_qwen25"

**Impact:** The consolidated model was **deleted** during cleanup, thinking it was "old Qwen 2.5" when it may have been valuable Qwen3 training.

---

### 2. Current Training State (Nov 15 22:00)
- **Active model:** Qwen3-8B (DIO_20251114)
- **Adapter:** 1.4 GB at step 2488
- **Status:** Fresh training from Nov 15 evening
- **This is NOT the missing 3-day trained model**

---

### 3. Older Qwen3 Adapters Found

#### A. /tmp/leo_compositions_qwen3_8b_lora (Nov 2)
```
Location: /tmp/leo_compositions_qwen3_8b_lora/
Size: 2.7 GB
Latest checkpoint: checkpoint-10962
Global step: 10,962 (2 epochs)
Date: Nov 2, 2025 04:14
Adapter size: 667 MB
```

**Status:** This is from Nov 2 - possibly part of earlier training, but not recent.

#### B. /path/to/compositions/adapters/ (Nov 2)
```
Location: /path/to/compositions/adapters/trained_overnight/
Size: 682 MB
Date: Nov 2, 2025 22:10
```

**Status:** Also from Nov 2 - old training run.

---

### 4. What's MISSING
According to search:
- ❌ No consolidated_models/ directory
- ❌ No consolidated_backups/ directory
- ❌ No large recent models besides current_model/
- ❌ Nothing in trash
- ❌ No snapshots with substantial data

---

## 📊 Training Timeline Reconstruction

### Nov 2, 2025
- Qwen3 training in `/tmp/leo_compositions_qwen3_8b_lora`
- Reached step 10,962 (2 epochs)
- Saved to `compositions_model/adapters/trained_overnight/`

### Nov 11, 2025
- Large log file (172 KB) - significant activity
- Possible consolidation attempts (errors seen)

### Nov 14, 2025
- Training sessions: 378 minutes each (~6.3 hours)
- Multiple quick completions
- Checkpoint-900 copied

### Nov 15, 2025 03:08 AM
- **Consolidation happened** - created `consolidated_models/20251115_030850`
- Merged adapter into base model
- Backed up to `consolidated_backups/20251115_030850`

### Nov 15, 2025 (Later)
- **1.2TB cleanup executed** - removed consolidated directories
- **Lost the merged model**

### Nov 15, 2025 22:00
- Fresh training started from step 0
- Current state: step 2488

---

## ❓ CRITICAL QUESTIONS FOR USER

1. **When did the "3 days of training" happen?**
   - Was it Nov 12-15?
   - Was it earlier (like Nov 2-5)?
   - Or a different time period?

2. **What was the highest step number you remember seeing?**
   - The Nov 15 consolidation had some training
   - The Nov 2 adapter reached step 10,962
   - Was it higher than these?

3. **Was it definitely in /path/to/training?**
   - Or could it have been in a different directory?
   - Like `/tmp/leo_compositions_qwen3_8b_lora`?
   - Or `/path/to/compositions/`?

4. **Do you remember the adapter size?**
   - Current adapters are ~667 MB to 1.4 GB
   - Was yours larger?

---

## 💡 Recovery Options

### If it was the Nov 15 consolidated model:
- ❌ **Not recoverable** - deleted, not in trash
- 🔄 **Alternative:** The adapter that was merged still exists as backup somewhere
- 📍 **Look for:** `consolidated_backups/20251115_030850/adapter/`

### If it was the Nov 2 training:
- ✅ **Still exists** in `/tmp/leo_compositions_qwen3_8b_lora/`
- ✅ **Also exists** in `/path/to/compositions/adapters/`
- 🎯 **Action:** Can copy these adapters to current training system

### If it was something else:
- 🔍 **Need more info** about timing and location
- 📂 **Can search** other directories if you remember more details

---

## 🎯 Next Steps

**Please answer the questions above so we can:**
1. Identify exactly which training run was lost
2. Determine if any backups exist
3. Plan recovery strategy

**Meanwhile, I'll check:**
- Whether the adapter backup from consolidation still exists somewhere
- Other large directories that might contain the model
- Git history or other version control

---

**Generated:** 2025-11-16 01:15 UTC
**Investigator:** Claude
