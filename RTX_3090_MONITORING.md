# RTX 3090 Monitoring Systems Guide

**Last Updated:** 2025-11-23
**Machine:** RTX 3090 (192.168.x.x)
**Purpose:** Inference + Monitoring + Validation

---

## Overview

The RTX 3090 runs 8 autonomous monitoring systems that evaluate training checkpoints synced from the RTX 4090. All systems load checkpoints directly (no API calls) and write results to JSON files.

---

## Running Systems

### 1. Checkpoint Sync Daemon
**Status:** ✅ Working
**File:** `monitoring/checkpoint_sync_daemon.py`
**Interval:** 5 minutes
**Purpose:** Auto-sync latest checkpoint from 4090

**Output Files:**
- `logs/checkpoint_sync.log` - Sync activity log
- `status/checkpoint_sync.json` - Current sync status
- `current_model/checkpoint-XXXXX/` - Latest synced checkpoint (keeps only 1)

**How to View:**
```bash
# Check sync status
ssh 192.168.x.x "tail -30 ~/TRAINING/logs/checkpoint_sync.log"

# See current checkpoint
ssh 192.168.x.x "ls -lh ~/TRAINING/current_model/"

# View sync status
ssh 192.168.x.x "cat ~/TRAINING/status/checkpoint_sync.json | jq"
```

---

### 2. Curriculum Optimizer
**Status:** ⚠️ Running but returning inf/0%
**File:** `monitoring/curriculum_optimizer.py`
**Interval:** 5 minutes
**Purpose:** Test checkpoints against easy/medium/hard validation sets

**Output Files:**
- `logs/curriculum_optimizer.log` - Evaluation log
- `status/curriculum_optimization.json` - Results + recommended curriculum mix

**How to View:**
```bash
# Check recent evaluations
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/curriculum_optimizer.log"

# View results
ssh 192.168.x.x "cat ~/TRAINING/status/curriculum_optimization.json | jq '.evaluations[-1]'"

# See recommended curriculum
ssh 192.168.x.x "cat ~/TRAINING/status/curriculum_optimization.json | jq '.recommendation'"
```

**Expected Data:**
- Loss per difficulty level
- Accuracy per difficulty level
- Recommended training mix (% easy/medium/hard)

**Current Issue:** Returns Loss=inf, Acc=0.00% (validation not working)

---

### 3. Adversarial Miner
**Status:** ⚠️ Running but no adversarial examples found
**File:** `monitoring/adversarial_miner.py`
**Interval:** 5 minutes
**Purpose:** Find hard examples where model fails

**Output Files:**
- `logs/adversarial_miner.log` - Mining activity
- `status/adversarial_mining.json` - Found examples summary
- `data/adversarial_examples/*.jsonl` - Hard examples for training

**How to View:**
```bash
# Check mining log
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/adversarial_miner.log"

# View found examples
ssh 192.168.x.x "cat ~/TRAINING/status/adversarial_mining.json | jq"

# List generated training data
ssh 192.168.x.x "ls -lh ~/TRAINING/data/adversarial_examples/"
```

---

### 4. Regression Monitor
**Status:** ⚠️ Running but no regressions detected (due to inf values)
**File:** `monitoring/continuous_regression_monitor.py`
**Interval:** 5 minutes
**Purpose:** Detect bad checkpoints (>15% loss increase)

**Output Files:**
- `logs/regression_monitor.log` - Monitoring log
- `status/regression_monitoring.json` - Regression alerts

**How to View:**
```bash
# Check for regressions
ssh 192.168.x.x "cat ~/TRAINING/status/regression_monitoring.json | jq '.checks'"

# View log
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/regression_monitor.log"
```

**Alert Triggers:**
- Loss increase >15% from previous checkpoint
- Accuracy drop >10% on any difficulty level

---

### 5. Model Comparison Engine
**Status:** ⚠️ Running but unable to rank checkpoints (inf values)
**File:** `monitoring/model_comparison_engine.py`
**Interval:** 10 minutes
**Purpose:** Rank checkpoints by composite score

**Output Files:**
- `logs/model_comparison.log` - Comparison log
- `status/model_comparisons.json` - Ranked checkpoint list

**How to View:**
```bash
# See ranked checkpoints
ssh 192.168.x.x "cat ~/TRAINING/status/model_comparisons.json | jq '.rankings'"

# Check comparison log
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/model_comparison.log"
```

**Ranking Criteria:**
- Average loss (lower is better)
- Easy/medium/hard balance
- Consistency across difficulty levels

---

### 6. Confidence Calibrator
**Status:** ⚠️ Running but no calibration data (inf values)
**File:** `monitoring/confidence_calibrator.py`
**Interval:** 10 minutes
**Purpose:** Measure model confidence calibration

**Output Files:**
- `logs/confidence_calibrator.log` - Calibration log
- `status/confidence_calibration.json` - 6 confidence bins

**How to View:**
```bash
# View calibration data
ssh 192.168.x.x "cat ~/TRAINING/status/confidence_calibration.json | jq"

# Check log
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/confidence_calibrator.log"
```

**Bins:**
- 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-0.95, 0.95-1.0

---

### 7. Self-Correction Loop
**Status:** ⚠️ Running but no data to process
**File:** `monitoring/self_correction_loop.py`
**Interval:** 5 minutes
**Purpose:** Capture errors and generate correction examples

**Output Files:**
- `logs/self_correction.log` - Processing log
- `queue/corrections/*.jsonl` - Correction training data
- `logs/error_patterns/*.json` - Error analysis

**How to View:**
```bash
# Check processing log
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/self_correction.log"

# View generated corrections
ssh 192.168.x.x "ls -lh ~/TRAINING/queue/corrections/"

# See error patterns
ssh 192.168.x.x "ls -lh ~/TRAINING/logs/error_patterns/"
```

---

### 8. Automated Testing Daemon
**Status:** ⚠️ Running but tests not executing (validation issues)
**File:** `monitoring/automated_testing_daemon.py`
**Interval:** 10 minutes
**Purpose:** Run fixed validation suite against checkpoints

**Output Files:**
- `logs/automated_testing.log` - Test execution log
- `status/automated_testing.json` - Test results

**How to View:**
```bash
# View test results
ssh 192.168.x.x "cat ~/TRAINING/status/automated_testing.json | jq"

# Check test log
ssh 192.168.x.x "tail -50 ~/TRAINING/logs/automated_testing.log"
```

---

## Problem Summary

**All 8 systems are running BUT:**
- All returning `Loss=inf` and `Accuracy=0.00%`
- Systems load checkpoints correctly
- Validation data exists and is properly formatted
- Issue is in the evaluation function (`_evaluate_difficulty`)

**Root Cause:**
The monitoring systems load models directly from checkpoints and try to run inference, but the evaluation is returning default/empty results instead of actual predictions.

---

## Validation Data

**Location:** `~/TRAINING/data/validation/`

**Files:**
- `easy_50.jsonl` - 50 easy puzzles
- `medium_50.jsonl` - 50 medium puzzles
- `hard_50.jsonl` - 50 hard puzzles
- `easy_100.jsonl`, `easy_200.jsonl` - Larger sets
- `medium_100.jsonl`, `medium_200.jsonl` - Larger sets
- `hard_100.jsonl`, `hard_200.jsonl` - Larger sets

**Format:** JSONL with `messages` array (user/assistant pairs)

---

## Web Interface (NOT WORKING YET)

**Intended URLs:**
- Main dashboard: http://192.168.x.x:8080/dashboard.html
- Live samples: http://192.168.x.x:8080/samples.html
- API health: http://192.168.x.x:5000/health (works)
- API generate: http://192.168.x.x:5000/generate (doesn't exist)

**Issue:** Web interface exists but shows no data because all monitoring systems return inf/0%

---

## Quick Health Check

```bash
# Check all running processes
ssh 192.168.x.x "ps aux | grep python3 | grep monitoring | grep -v grep"

# Should show 8 processes:
# 1. curriculum_optimizer
# 2. adversarial_miner
# 3. regression_monitor
# 4. model_comparison_engine
# 5. confidence_calibrator
# 6. self_correction_loop
# 7. automated_testing_daemon
# 8. checkpoint_sync_daemon

# Check all status files
ssh 192.168.x.x "ls -lh ~/TRAINING/status/*.json"

# Check recent activity
ssh 192.168.x.x "ls -lht ~/TRAINING/logs/*.log | head -10"
```

---

## Next Steps to Fix

1. **Debug _evaluate_difficulty function** - Find why it returns inf/0%
2. **Test validation manually** - Run quick_validation.py to isolate issue
3. **Check tokenizer compatibility** - Ensure checkpoint tokenizer works
4. **Verify inference** - Test actual model.generate() calls
5. **Fix web UI** - Once data is flowing, web interface will populate

---

## Memory Usage

All 8 systems combined: ~18GB RAM on RTX 3090
- Each monitoring system: ~900MB-1GB RAM
- Checkpoint sync: ~15MB RAM
- API server (main.py): ~1.8GB RAM

**Total:** ~20GB / 32GB RAM used
