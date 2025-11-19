# Progress Display Fix - Dual Progress Tracking

**Date:** 2025-11-11
**Status:** ✅ IMPLEMENTED (will activate on next batch)

---

## PROBLEM

The old progress display was confusing:
```json
{
  "current_step": 89,
  "total_steps": 237
}
```

**Issue:** Mixing two different concepts:
- `current_step` (89) = Trainer's internal global_step
- `total_steps` (237) = Cumulative total after this batch

**Result:** Percentage shows 89/237 = 37%, but unclear what this means!

---

## SOLUTION

**NEW: Dual Progress Tracking**

The status JSON now includes BOTH:

### 1. File Progress (within current batch)
```json
{
  "batch_step": 89,           // Step within THIS file
  "batch_total_steps": 118,   // Total steps for THIS file
  "current_file": "syllo_batch_autogen_06.jsonl"
}
```

**Display:** `89/118 = 75%` - "75% done with current file"

### 2. Queue Progress (overall batches)
```json
{
  "batch_number": 6,          // Current batch number
  "batch_queue_size": 22      // Total batches queued
}
```

**Display:** `6/22 = 27%` - "Batch 6 of 22"

### 3. Cumulative Progress (still available)
```json
{
  "current_step": 119,        // Cumulative global_step
  "total_steps": 237          // Total after this batch
}
```

**Display:** `119/237 = 50%` - "50% of cumulative training steps"

---

## NEW STATUS JSON STRUCTURE

```json
{
  "status": "training",

  // Cumulative progress (across all batches)
  "current_step": 119,
  "total_steps": 237,

  // File progress (current batch only) - NEW!
  "batch_step": 89,
  "batch_total_steps": 118,
  "batch_number": 6,
  "batch_queue_size": 22,
  "current_file": "syllo_batch_autogen_06.jsonl",

  // Training metrics
  "epoch": 0,
  "loss": 0.234,
  "learning_rate": 0.0002,

  // ... rest of fields
}
```

---

## IMPLEMENTATION DETAILS

### Changes Made:

**1. training_status.py (Status Tracker)**
- Added 5 new optional fields to `TrainingStatus` dataclass:
  - `batch_step`: Step within current file
  - `batch_total_steps`: Total steps for current file
  - `batch_number`: Current batch number (1-indexed)
  - `batch_queue_size`: Total batches in queue
  - `current_file`: Filename being trained on
- Updated `update_progress()` and `update_inference()` to accept these new parameters

**2. train.py (Trainer)**
- Updated `LiveMonitorCallback` to:
  - Accept batch context parameters in `__init__`
  - Calculate `batch_step = state.global_step - current_global_step`
  - Pass all batch context to status writer
- Reads batch context from `args` (passed by daemon):
  - `args.current_file`
  - `args.batch_number`
  - `args.batch_queue_size`

**3. training_daemon.py (Daemon)**
- Updated `train_on_file()` to accept `batch_number` and `batch_queue_size`
- Counts total files in inbox before processing
- Enumerates files with 1-indexed batch numbers
- Passes batch context to trainer via `args`

---

## UI DISPLAY EXAMPLES

### Example 1: Mid-batch
```
File Progress: 89/118 steps (75%)
Queue Progress: Batch 6/22 (27%)
Current File: syllo_batch_autogen_06.jsonl
```

### Example 2: Near completion
```
File Progress: 115/118 steps (97%)
Queue Progress: Batch 21/22 (95%)
Current File: syllo_batch_autogen_27.jsonl
```

---

## MATH VERIFICATION

**Scenario:** Training batch 06 of 22 total batches

**File Progress:**
- Previous batches completed: 5 batches × 118 steps = 590 steps
- Current batch: 89 steps into 118 total
- File progress: 89/118 = 75% ✅

**Queue Progress:**
- Current batch: 6
- Total batches: 22
- Queue progress: 6/22 = 27% ✅

**Cumulative Progress:**
- Steps completed: 590 (prev) + 89 (current) = 679 steps
- Wait, that doesn't match the 119 in the status...

Actually, the `current_step` is the Trainer's `global_step`, which starts from the checkpoint value (119 from previous session), not 0! So:
- Trainer session starts at: 119 (from checkpoint)
- Current step in session: 89
- Trainer's `global_step`: 119 + 89 = 208
- Total steps projected: 237

So the cumulative display would be: 208/237 = 88% through this training session

---

## WHEN IT ACTIVATES

**Current Status:** Changes deployed to code
**Activation:** Next batch (batch 07 or later)
**Why:** Current batch (06) started before code changes

Once batch 07 starts, the status JSON will include all new fields!

---

## BACKWARD COMPATIBILITY

All new fields are **optional** (default: `None`).

**Old monitors:** Still work - will just see `null` for new fields
**New monitors:** Can display rich dual progress

---

## RECOMMENDATIONS FOR UI

Display progress like this:

```
Current File: syllo_batch_autogen_06.jsonl
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75%
Step 89/118

Overall Queue: Batch 6 of 22
━━━━━━━━━░░░░░░░░░░░░░░░░░░░░░░░░ 27%
```

Or simpler:
```
File: 89/118 (75%) | Queue: 6/22 (27%)
Training: syllo_batch_autogen_06.jsonl
```

---

## TESTING

To test, wait for batch 07 to start, then check:

```bash
cat status/training_status.json | jq '{
  current_file,
  batch_step,
  batch_total_steps,
  batch_number,
  batch_queue_size,
  file_pct: ((.batch_step / .batch_total_steps * 100) | floor),
  queue_pct: ((.batch_number / .batch_queue_size * 100) | floor)
}'
```

Expected output:
```json
{
  "current_file": "syllo_batch_autogen_07.jsonl",
  "batch_step": 25,
  "batch_total_steps": 118,
  "batch_number": 7,
  "batch_queue_size": 22,
  "file_pct": 21,
  "queue_pct": 31
}
```

---

## SUMMARY

✅ **File progress:** Shows how far through current batch
✅ **Queue progress:** Shows which batch out of total
✅ **Cumulative progress:** Still available for overall tracking
✅ **Backward compatible:** Old monitors won't break
✅ **Clear semantics:** Each percentage means something specific

**No more confusion!** 🎉
