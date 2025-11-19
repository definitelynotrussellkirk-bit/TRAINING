# Master Training System Refactor

**Date:** 2025-11-16
**Priority:** CRITICAL - Prevent future catastrophic losses + Enable learning insights

---

## 🎯 PRIMARY GOAL: LEARNING EVOLUTION TRACKING

**THE MOST IMPORTANT FEATURE:**
> "I want to see my training data and what the machine is guessing at each stage of training"

This enables:
- See which examples the model learns easily vs struggles with
- Track improvement/regression on specific patterns
- Identify when model "gets it" (learning curves)
- Debug why certain things aren't being learned
- Make data-driven decisions about training

### Example Use Case:
```
Example #42: "What is 2+2?"
- Before training:  "The capital of France is Paris" (loss: 4.2)
- After 100 steps:  "2+2 is 5" (loss: 2.1)
- After 500 steps:  "2+2 equals 4" (loss: 0.3) ← Learned!
- After 1000 steps: "2+2=4" (loss: 0.1) ← Solidified!
```

---

## 🏗️ NEW ARCHITECTURE

### Directory Structure

```
TRAINING/
│
├── README.md                           # Start here
├── QUICK_START.md                      # 5-min guide
├── config.json                         # Main config
│
├── bin/                                # All executables
│   ├── train_daemon.py                 # Main daemon
│   ├── evolution_tracker.py            # Learning tracker
│   ├── model_versioner.py              # Version manager
│   ├── backup_manager.py               # Backup system
│   └── control_panel.py                # Control interface
│
├── control/                            # Training control
│   ├── .pause                          # Pause signal
│   ├── .stop                           # Stop signal
│   ├── .skip                           # Skip signal
│   ├── queue.json                      # Training queue
│   └── current_state.json              # Current status
│
├── inbox/                              # Data ingestion
│   ├── priority/                       # Urgent training
│   └── normal/                         # Regular queue
│
├── data/                               # Training data tracking
│   ├── active/                         # Currently training
│   │   └── TIMESTAMP_dataset.jsonl
│   ├── completed/                      # Successfully trained
│   │   └── TIMESTAMP_dataset.jsonl
│   └── evolution_snapshots/            # Model predictions over time
│       └── TIMESTAMP_dataset/
│           ├── step_0000.json          # Before training
│           ├── step_0100.json          # After 100 steps
│           ├── step_0500.json          # After 500 steps
│           └── analysis.json           # Learning curve data
│
├── models/                             # Model storage
│   ├── base/                           # Base models
│   │   └── qwen3-8b/                   # Current base
│   │       ├── model files...
│   │       └── VERSION.txt             # Base version
│   │
│   ├── versions/                       # Trained model versions
│   │   ├── v001_20251116_initial/
│   │   │   ├── adapter.safetensors
│   │   │   ├── metadata.json           # What was trained
│   │   │   ├── evolution/              # Learning evolution data
│   │   │   └── checkpoints/            # Training checkpoints
│   │   ├── v002_20251117_updated/
│   │   └── latest -> v002_20251117_updated
│   │
│   └── backups/                        # Safety backups
│       ├── daily/                      # Daily snapshots
│       │   └── 2025-11-16/
│       ├── pre_consolidation/          # Before merges
│       │   └── TIMESTAMP/
│       └── pre_deletion/               # Before any deletion
│           └── TIMESTAMP/
│
├── monitoring/                         # Web UI & monitors
│   ├── ui/
│   │   ├── index.html                  # Main dashboard
│   │   ├── evolution_viewer.html       # Learning evolution UI
│   │   └── model_history.html          # Version history
│   ├── servers/
│   │   ├── main_server.py              # Port 8080
│   │   └── evolution_api.py            # Port 8081
│   └── assets/
│       ├── css/
│       └── js/
│
├── logs/                               # All logs
│   ├── training/
│   │   └── YYYYMMDD.log
│   ├── daemon/
│   │   └── YYYYMMDD.log
│   └── evolution/
│       └── YYYYMMDD.log
│
├── docs/                               # Documentation
│   ├── README.md                       # Docs index
│   ├── guides/                         # User guides
│   │   ├── getting_started.md
│   │   ├── evolution_tracking.md
│   │   └── model_versioning.md
│   ├── technical/                      # Technical docs
│   └── archive/                        # Old docs
│
└── temp/                               # Temporary files
    └── .gitignore
```

---

## 🔬 LEARNING EVOLUTION TRACKING SYSTEM

### Core Concept

**Store model predictions at regular intervals during training**

### Data Structure

```json
// evolution_snapshots/20251116_dataset/step_0100.json
{
  "snapshot_id": "20251116_dataset_step_0100",
  "training_step": 100,
  "timestamp": "2025-11-16T02:00:00Z",
  "model_version": "v001_20251116_initial",
  "examples": [
    {
      "example_id": "ex_001",
      "input": "What is 2+2?",
      "expected_output": "4",
      "model_output": "2+2 is 5",
      "loss": 2.1,
      "accuracy": 0.0,
      "confidence": 0.65,
      "metadata": {
        "improved_from_last": false,
        "learning_rate": 0.0002,
        "batch_position": 5
      }
    },
    // ... more examples
  ],
  "summary": {
    "avg_loss": 1.8,
    "avg_accuracy": 0.45,
    "examples_improving": 120,
    "examples_regressing": 30,
    "examples_stable": 50
  }
}
```

### Evolution Tracking Schedule

```python
# Snapshot frequency (configurable)
SNAPSHOT_SCHEDULE = {
    "before_training": 0,      # Always capture baseline
    "early": [10, 25, 50],     # Dense early on
    "mid": [100, 250, 500],    # Less frequent mid-training
    "late": [1000, 2500, 5000],# Sparse late training
    "final": "end"             # Always capture final
}
```

### Learning Curve Analysis

For each example, track:
- Initial prediction (before training)
- Predictions at each snapshot
- Loss trajectory
- When it "learned" (loss < threshold)
- Whether it regressed
- Final performance

### Evolution Viewer UI

```
┌─ LEARNING EVOLUTION VIEWER ──────────────────────────────┐
│                                                            │
│ Example: "What is 2+2?"                                   │
│                                                            │
│ Loss Over Time:                                           │
│ 5.0 │ ●                                                   │
│ 4.0 │                                                     │
│ 3.0 │                                                     │
│ 2.0 │   ●                                                 │
│ 1.0 │       ●───●───●─────●                              │
│ 0.0 └─────┴─────┴─────┴─────┴─────→                      │
│     0    100   250   500  1000  steps                     │
│                                                            │
│ Predictions:                                              │
│ Step 0:    "The capital of France..."  ❌ Loss: 4.2      │
│ Step 100:  "2+2 is 5"                  ❌ Loss: 2.1      │
│ Step 250:  "2+2 equals 4"              ✓  Loss: 0.8      │
│ Step 500:  "2+2=4"                     ✓  Loss: 0.3      │
│ Step 1000: "2+2=4"                     ✓  Loss: 0.1      │
│                                                            │
│ Analysis: Learned at step ~250 ✓                         │
│                                                            │
│ [Previous Example] [Next Example] [Export Data]          │
└────────────────────────────────────────────────────────────┘
```

---

## 🔐 MODEL VERSIONING & BACKUP SYSTEM

### Versioning Strategy

**Every trained model gets a version:**

```
v001_20251116_baseline          # First training
v002_20251117_math_fixes        # After math training
v003_20251118_consolidated      # After consolidation
v004_20251119_reasoning_boost   # After reasoning data
```

### Version Metadata

```json
// models/versions/v002_20251117_math_fixes/metadata.json
{
  "version": "v002",
  "created": "2025-11-17T10:30:00Z",
  "parent_version": "v001_20251116_baseline",
  "base_model": "qwen3-8b",
  "training_data": {
    "file": "math_corrections_10k.jsonl",
    "examples": 10000,
    "md5": "abc123..."
  },
  "training_config": {
    "learning_rate": 0.0002,
    "lora_r": 128,
    "steps": 1250,
    "final_loss": 0.45
  },
  "performance": {
    "baseline_accuracy": 0.42,
    "final_accuracy": 0.89,
    "improvement": 0.47
  },
  "evolution_data": "evolution/v002_snapshots/",
  "notes": "Fixed math calculation errors, significant improvement"
}
```

### Backup Strategy (CRITICAL)

**3-2-1 Backup Rule:**
- **3** copies of every important model
- **2** different storage locations
- **1** offsite/external backup

**Automatic Backups:**
```python
# Before any destructive operation
def safe_operation(operation_type):
    backup_path = f"backups/pre_{operation_type}/{timestamp}/"

    # Copy current model
    backup_current_model(backup_path)

    # Copy evolution data
    backup_evolution_data(backup_path)

    # Create backup manifest
    create_manifest(backup_path, operation_type)

    # Perform operation
    try:
        perform_operation()
    except Exception as e:
        # Restore from backup
        restore_from_backup(backup_path)
        raise
```

**Daily Backups:**
- Every day at 3 AM: Full snapshot to `backups/daily/YYYY-MM-DD/`
- Keep last 7 days
- Keep weekly backups (4 weeks)
- Keep monthly backups (12 months)

**Pre-Operation Backups:**
- Before consolidation
- Before model deletion
- Before major config changes
- Before switching base models

---

## 🎮 ENHANCED CONTROL SYSTEM

### Control Files

```bash
# In control/ directory

.pause          # Pause training (saves checkpoint)
.stop           # Stop current, queue it, move to next
.skip           # Skip current batch entirely
.priority       # Process priority queue
.snapshot       # Take evolution snapshot NOW
.consolidate    # Safe consolidation with backups
```

### Queue System

```json
// control/queue.json
{
  "active": {
    "id": "train_20251116_143000",
    "file": "math_fixes.jsonl",
    "priority": "high",
    "started": "2025-11-16T14:30:00Z",
    "progress": {
      "step": 450,
      "total_steps": 1250,
      "percent": 36,
      "eta_minutes": 18
    },
    "status": "training"
  },
  "queued": [
    {
      "id": "train_20251116_150000",
      "file": "normal_data.jsonl",
      "priority": "normal",
      "queued_at": "2025-11-16T15:00:00Z",
      "estimated_start": "2025-11-16T14:48:00Z"
    }
  ],
  "paused": [],
  "completed": [
    {
      "id": "train_20251116_120000",
      "file": "baseline.jsonl",
      "completed_at": "2025-11-16T14:25:00Z",
      "version_created": "v001_20251116_baseline"
    }
  ]
}
```

---

## 🔄 SAFE CONSOLIDATION PROCESS

### Never Lose Data Again

```python
def safe_consolidation():
    """
    Consolidate with triple-redundancy backups
    """
    timestamp = get_timestamp()

    # Step 1: Pre-consolidation backup
    backup_dir = f"backups/pre_consolidation/{timestamp}/"
    logger.info(f"Creating backup at {backup_dir}")

    # Backup everything
    copy_tree("current_model/", f"{backup_dir}/adapter/")
    copy_tree("evolution_snapshots/", f"{backup_dir}/evolution/")
    copy_file("config.json", f"{backup_dir}/config.json")

    # Create manifest
    create_manifest(backup_dir, {
        "operation": "consolidation",
        "base_model": get_base_model(),
        "adapter_size": get_size("current_model/"),
        "evolution_data": get_size("evolution_snapshots/"),
        "timestamp": timestamp
    })

    # Step 2: Verify backup
    if not verify_backup(backup_dir):
        raise Exception("Backup verification failed!")

    # Step 3: Perform consolidation
    try:
        merged_model = merge_adapter_with_base()

        # Step 4: Create new version
        version_id = create_new_version(merged_model, timestamp)

        # Step 5: Keep old adapter as backup
        archive_adapter(f"backups/adapters/{timestamp}/")

        # Step 6: Update pointers
        update_latest_version(version_id)

        logger.info(f"✓ Consolidation complete: {version_id}")
        logger.info(f"✓ Backup at: {backup_dir}")
        logger.info(f"✓ Old adapter archived")

    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        logger.info("Restoring from backup...")
        restore_from_backup(backup_dir)
        raise
```

---

## 📊 WEB UI FEATURES

### Dashboard

```
┌─ TRAINING DASHBOARD ──────────────────────────────────────┐
│                                                            │
│ Current Training:                                         │
│  ▶ math_fixes.jsonl (Priority)                           │
│    Step 450/1250 (36%) - ETA: 18 min                     │
│    Loss: 0.82 │ Accuracy: 67% │ Learning: ↑              │
│                                                            │
│ Learning Evolution:                                       │
│  [View Evolution Snapshots] [Compare Versions]           │
│                                                            │
│ Queue (1 pending):                                        │
│  1. normal_data.jsonl (50k examples) - ETA: +45min      │
│                                                            │
│ Recent Versions:                                          │
│  v002_20251117_math_fixes    [Active]   [View]          │
│  v001_20251116_baseline                  [Compare]       │
│                                                            │
│ Controls:                                                 │
│  [Pause] [Stop] [Skip] [Snapshot] [Consolidate]         │
│                                                            │
│ System Health: ✓ All systems operational                 │
└────────────────────────────────────────────────────────────┘
```

### Evolution Viewer (MOST IMPORTANT)

```
┌─ EVOLUTION VIEWER ────────────────────────────────────────┐
│                                                            │
│ Dataset: math_fixes.jsonl                                 │
│ Training: v002_20251117_math_fixes                        │
│                                                            │
│ Overall Progress:                                         │
│ ████████████████████░░░░░░ 67% learned (67/100 examples) │
│                                                            │
│ Learning Categories:                                      │
│  ✓ Learned (67)   ⚠ Struggling (20)   ✗ Not learned (13)│
│                                                            │
│ Example Browser:                                          │
│  [Filter: All ▾] [Sort: Learning Speed ▾] [Search...]    │
│                                                            │
│  #1  ✓ "What is 2+2?" - Learned at step 250              │
│  #2  ✓ "Calculate 5+3" - Learned at step 180             │
│  #3  ⚠ "Solve 15*23" - Improving slowly                  │
│  #4  ✗ "Factor 143" - Not learning                       │
│                                                            │
│  [Click example to view detailed learning curve]         │
│                                                            │
│ Export: [CSV] [JSON] [Report]                            │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 IMPLEMENTATION PHASES

### Phase 1: Learning Evolution Tracker (Week 1)
**Priority: CRITICAL**

1. **Create evolution tracker core** (2 hours)
   - Snapshot capture system
   - Data storage format
   - Scheduling system

2. **Integrate with training loop** (2 hours)
   - Hook into training steps
   - Capture predictions
   - Store efficiently

3. **Build evolution viewer UI** (3 hours)
   - Web interface
   - Learning curves
   - Example browser

4. **Testing** (1 hour)
   - Train on sample data
   - Verify snapshots
   - Check UI

**Deliverable:** Can see learning evolution for any training run

### Phase 2: Model Versioning System (Week 1)
**Priority: HIGH**

1. **Version manager** (2 hours)
   - Create version structure
   - Metadata system
   - Version tracking

2. **Backup system** (2 hours)
   - Automatic backups
   - Pre-operation backups
   - Verification

3. **Safe consolidation** (2 hours)
   - Backup before consolidate
   - Error recovery
   - Archiving

4. **Testing** (1 hour)
   - Test backups
   - Test consolidation
   - Verify recovery

**Deliverable:** Never lose a model again

### Phase 3: Control System (Week 2)
**Priority: MEDIUM**

1. **Priority queue** (2 hours)
2. **Pause/stop/skip** (2 hours)
3. **Queue management UI** (2 hours)
4. **Testing** (1 hour)

**Deliverable:** Full training control

### Phase 4: Directory Reorganization (Week 2)
**Priority: LOW**

1. **Create new structure** (1 hour)
2. **Migrate files** (2 hours)
3. **Update imports** (2 hours)
4. **Testing** (1 hour)

**Deliverable:** Clean organization

### Phase 5: Documentation (Week 2)
**Priority: MEDIUM**

1. **Update all docs** (3 hours)
2. **Create tutorials** (2 hours)
3. **API documentation** (1 hour)

**Deliverable:** Complete documentation

---

## ✅ SUCCESS CRITERIA

### Must Have
- [ ] Can view learning evolution for any example
- [ ] Can see learning curves
- [ ] Can compare model versions
- [ ] Automatic backups before any operation
- [ ] Safe consolidation with recovery
- [ ] Version history visible
- [ ] Never lose training data

### Should Have
- [ ] Priority queue system
- [ ] Pause/resume training
- [ ] Clean directory structure
- [ ] Comprehensive docs

### Nice to Have
- [ ] Evolution data export
- [ ] Automated reports
- [ ] Learning insights/recommendations

---

## 🎯 WHY THIS MATTERS

### Learning Evolution Tracking
- **Debug training:** See exactly what's being learned
- **Optimize data:** Identify hard examples
- **Track progress:** Measure real improvement
- **Make decisions:** Data-driven training choices

### Version Control
- **Never lose work:** Multiple backups
- **Track history:** Know what changed when
- **Easy rollback:** Return to previous versions
- **Safe experiments:** Can always recover

### Control System
- **Flexibility:** Pause/stop/prioritize
- **Efficiency:** Don't waste GPU time
- **Responsiveness:** Handle urgent training

---

## 📝 NEXT STEPS

1. **Review this plan** ← You are here
2. **Approve/modify**
3. **Implement Phase 1** (Evolution tracking)
4. **Test thoroughly**
5. **Implement Phase 2** (Versioning)
6. **Continue...**

---

**Ready to start?**

I recommend starting with **Phase 1** (Learning Evolution Tracking) since that's your top priority. It will give you immediate value and can be implemented without disrupting current training.
