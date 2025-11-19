# Monitoring Costs Guide - Preventing Training Hangs

**Created:** 2025-11-16
**Context:** Lost hours debugging training hangs caused by expensive monitoring operations

## Problem Summary

**What Happened:**
- Training hung at steps 10, 25, 50 for 20+ minutes each
- GPU dropped from 98% to 60% utilization
- No visible progress, appeared completely frozen

**Root Causes:**
1. **evolution_tracker.py** - Ran inference on 100 examples at checkpoint steps → 60-300s per call
2. **detail_collector.py** - Processed validation dataset every 50 steps → caused hangs

**Why It Went Unnoticed:**
- No cost tracking for monitoring operations
- No warnings when operations became expensive
- Assumed monitoring was "lightweight"
- No visibility into time spent in monitoring vs training

---

## The Solution: Cost-Aware Monitoring

### 1. Monitoring Cost Tracker (`monitoring_cost_tracker.py`)

**Purpose:** Track and budget all monitoring operation costs

**Usage:**
```python
from monitoring_cost_tracker import MonitoringCostTracker

tracker = MonitoringCostTracker(output_dir="current_model/status")

# Track any monitoring operation
with tracker.track("evolution_snapshot"):
    capture_evolution_snapshot(model, examples)

# Check if over budget
if tracker.is_over_budget():
    print("⚠️  Monitoring consuming too much time!")

# Get detailed report
report = tracker.get_report()
print(f"Monitoring overhead: {report['total_percentage']:.1f}%")
```

**Automatic Warnings:**
- Warns if single operation takes >30 seconds
- Warns if total monitoring >5% of training time
- Critical alert if >10% of training time

**Output:**
- `status/monitoring_costs.json` - Real-time cost tracking
- Console warnings when budgets exceeded
- Cost breakdown showing most expensive operations

---

## Known Operation Costs

| Operation | Time/Call | Frequency | Cost per 1000 steps | Impact | Status |
|-----------|-----------|-----------|---------------------|--------|--------|
| **Evolution Snapshot** | 60-300s | Every 500 steps | 120-600s (12-60%) | 🚨 CRITICAL | DISABLED |
| **Detail Collector** | 0.1-0.5s | Every 50 steps | 2-10s (0.2-1%) | ⚠️ MODERATE | DISABLED |
| **Validation Loss** | 30-120s | Every 200 steps | 150-600s (15-60%) | ⚠️ HIGH | Active (1000 examples) |
| **Live Inference** | 1-5s | Every 200 steps | 5-25s (0.5-2.5%) | ✅ ACCEPTABLE | Active |
| **Status Updates** | <0.01s | Every 5s | ~2s (0.2%) | ✅ NEGLIGIBLE | Active |

### Cost Calculation Example

For 1000 training steps at 3.5s/step = 3500s total training time:

**Acceptable Setup (Current):**
- Live inference (200 step): 5 calls × 3s = 15s (0.4%)
- Status updates: ~2s (0.06%)
- **Total:** 17s (0.5%) ✅

**Previous Setup (HUNG):**
- Evolution snapshots (500 step): 2 calls × 180s = 360s (10%)
- Detail collector (50 step): 20 calls × 0.3s = 6s (0.2%)
- Live inference: 15s (0.4%)
- **Total:** 381s (11%) 🚨 OVER BUDGET!

---

## Decision Framework: When to Enable Monitoring

### ✅ ALWAYS SAFE (Enable by Default)
- **Live inference** (eval_steps=200) - Quick examples showing model learning
- **Status updates** (every 5s) - Progress tracking
- **Checkpoints** (every 100 steps) - Model saving

### ⚠️ USE CAUTIOUSLY (Consider Cost)
- **Validation loss** - Use smaller validation set (<100 examples) or less frequent
- **Detail collector** - Only for debugging specific issues, disable for production
- **Training data browser** - Only when actively monitoring

### 🚨 NEVER IN PRODUCTION (Debug Only)
- **Evolution tracking** - WAY too expensive, only for research
- **Per-step inference** - Blocks training completely
- **Large validation sets** - Use <100 examples max

---

## Monitoring Budget Guidelines

**Target:** <5% of total training time spent on monitoring

**Calculation:**
```
monitoring_overhead = (total_monitoring_time / total_training_time) * 100

If monitoring_overhead > 5%:
    → Reduce frequency OR reduce scope OR disable feature
```

**Quick Check:**
```bash
# Check current monitoring costs
cat current_model/status/monitoring_costs.json | jq '{
  budget_status,
  total_percentage,
  top_operations: .operations | to_entries |
    sort_by(.value.total_time) | reverse | .[0:3]
}'
```

---

## Best Practices

### 1. **Estimate Costs Before Enabling**
```python
# Example: Estimate evolution tracking cost
steps_per_batch = 2487
frequency = 500  # Every 500 steps
time_per_snapshot = 180  # 3 minutes

calls = steps_per_batch / frequency  # 4.9 calls
total_cost = calls * time_per_snapshot  # 882 seconds

training_time = steps_per_batch * 3.5  # 8704 seconds
overhead = (total_cost / training_time) * 100  # 10.1%

# Result: Too expensive! Don't enable.
```

### 2. **Start Lightweight, Add Selectively**
- Begin with minimal monitoring
- Add features one at a time
- Measure impact before keeping enabled
- Disable anything >5% overhead

### 3. **Use Sampling for Expensive Operations**
- Don't need to track EVERY example
- Sample 10-20 examples instead of 100
- Run expensive analysis offline after training

### 4. **Check Costs Regularly**
```bash
# During training, check monitoring overhead
python3 -c "
from monitoring_cost_tracker import get_global_tracker
tracker = get_global_tracker()
tracker.print_summary()
"
```

---

## Troubleshooting: Training Seems Slow

### Step 1: Check Monitoring Costs
```bash
# Are monitoring operations consuming time?
cat current_model/status/monitoring_costs.json | jq .total_percentage

# If > 5%, identify culprit:
cat current_model/status/monitoring_costs.json | jq '.operations |
  to_entries | sort_by(.value.total_time) | reverse'
```

### Step 2: Check GPU Utilization
```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Should be >90% during training
# If <70%, monitoring is likely blocking
```

### Step 3: Identify Expensive Operations
Signs of expensive monitoring:
- GPU drops to 50-70% periodically
- Training "pauses" at regular intervals
- Specific steps take much longer (e.g., every 50 or 200 steps)
- Logs show operations taking >10s

### Step 4: Disable Expensive Features
```python
# In train.py, disable suspects:
self.evolution_tracker = None  # Line 366
if False and DETAILED_MONITORING_AVAILABLE:  # Line 798
```

---

## Configuration Recommendations

### For Production Training
```json
{
  "eval_steps": 200,           // Show examples every 200 steps
  "save_steps": 100,           // Checkpoint every 100 steps
  "validation_examples": 20,   // Small validation set
  "enable_evolution": false,   // Too expensive
  "enable_detail_collector": false,  // Too expensive
  "monitoring_budget": 5.0     // Max 5% overhead
}
```

### For Research/Debugging
```json
{
  "eval_steps": 50,            // More frequent examples
  "save_steps": 50,            // More frequent checkpoints
  "validation_examples": 100,  // Larger validation set
  "enable_evolution": true,    // ONLY if needed, VERY expensive
  "enable_detail_collector": true,  // ONLY for debugging
  "monitoring_budget": 15.0    // Accept higher overhead
}
```

### For Final Model Evaluation
```json
{
  "eval_steps": 1000,          // Infrequent evals
  "save_steps": 500,           // Infrequent checkpoints
  "validation_examples": 1000, // Full validation set
  "enable_evolution": false,   // Not needed
  "enable_detail_collector": false,  // Not needed
  "monitoring_budget": 2.0     // Minimal overhead
}
```

---

## Monitoring Cost Checklist

Before enabling ANY monitoring feature:

- [ ] Estimated time per call?
- [ ] How often will it run?
- [ ] Total cost per 1000 steps?
- [ ] Percentage of training time?
- [ ] Is it <5% overhead?
- [ ] Can it be made cheaper (smaller sample, less frequent)?
- [ ] Is there a cheaper alternative?
- [ ] Added to cost tracker with `tracker.track()`?

**If overhead >5%, either:**
1. Reduce frequency (eval_steps)
2. Reduce scope (fewer examples)
3. Run offline (after training)
4. Don't enable it

---

## Lessons Learned (2025-11-16)

### What We Did Wrong
❌ Enabled expensive monitoring without measuring cost
❌ No budget for monitoring overhead
❌ No warnings when operations took too long
❌ Assumed monitoring was "free"
❌ No visibility into monitoring time vs training time

### What We're Doing Now
✅ Track all monitoring costs with `MonitoringCostTracker`
✅ Budget of <5% overhead for monitoring
✅ Automatic warnings when over budget
✅ Cost estimates documented for all features
✅ Default to lightweight monitoring, add cautiously
✅ Measure impact before keeping features enabled

### Impact
- **Before:** 3+ hours lost to hung training, manual restarts, debugging
- **After:** Training runs smoothly, early warnings prevent issues
- **Cost:** Monitoring now <1% overhead vs 10-15% before

---

## Quick Reference

**Check monitoring costs:**
```bash
cat current_model/status/monitoring_costs.json | jq .total_percentage
```

**Is training slow due to monitoring?**
```bash
# Check GPU util - should be >90%
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Check monitoring overhead - should be <5%
cat current_model/status/monitoring_costs.json | jq .budget_status
```

**Disable expensive features:**
```bash
# Edit train.py to disable:
# - evolution_tracker (line 366)
# - detail_collector (line 798)
```

**Safe monitoring setup:**
- eval_steps: 200 (not less)
- validation: <50 examples
- evolution: disabled
- detail_collector: disabled
- Expected overhead: <1%
