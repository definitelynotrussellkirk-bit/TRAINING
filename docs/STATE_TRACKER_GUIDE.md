# State Tracker Guide - Phase 4

**Created:** 2025-11-16  
**Purpose:** Machine-readable system state for bulletproof training system

---

## 🎯 What is the State Tracker?

The state tracker is your **first line of defense** against catastrophic mistakes. It generates a complete, machine-readable snapshot of your training system that both you AND future Claude instances can read to know EXACTLY what exists.

**Think of it as:** A health check + inventory + warning system all in one command.

---

## 🚀 Quick Start

```bash
# Run this at the start of EVERY session
python3 state_tracker.py --check
```

That's it! You now know:
- What models exist (don't delete them!)
- System health status
- Any warnings that need attention
- Complete system state

---

## 📋 What It Checks

### 1. Current Model
- **Exists?** Is there a `current_model/` directory?
- **Size:** How much disk space is it using?
- **Training Steps:** How much training has been done?
- **Has Adapter?** Is there an active LoRA adapter?
- **Last Training:** When was it last trained?

**Example:**
```
📦 CURRENT MODEL:
   ✓ Exists: current_model/
   Size: 32.66 GB
   Training Steps: 685
   Last Training: 2025-11-16T06:41:10
   Has Adapter: Yes
```

### 2. Saved Versions
- **Count:** How many versions are saved?
- **Details:** ID, date, description, size for each
- **Storage:** Total disk space used by versions

**Example:**
```
📚 VERSIONS: 2 saved
   • v001: Math training (2025-11-14, 30.2GB)
   • v002: Reasoning training (2025-11-15, 31.5GB)
```

### 3. Configuration
- **Model Name:** Current model identifier
- **Base Model:** Path to base model
- **Max Length:** Maximum sequence length (critical!)
- **Locked Parameters:** Which params shouldn't change

**Example:**
```
⚙️  CONFIG:
   Model: qwen3_8b
   Base: /path/to/training/DIO_20251114
   Max Length: 2048
   Locked Params: max_length, base_model, model_name
```

### 4. Training Status
- **Status:** idle, training, paused, error
- **Current Step:** Where in training are we?
- **Total Evals:** How many evaluations done?
- **Last Update:** When was status last updated?

**Example:**
```
🎯 TRAINING STATUS:
   Status: training
   Steps: 685
   Evaluations: 57
```

### 5. Daemon Status
- **Running?** Is training daemon active?
- Checks for `training_daemon.py` process

**Example:**
```
🔄 DAEMON: Running ✓
```

### 6. Disk Space
- **Available:** Free space in GB
- **Used %:** Percentage of disk used
- **Warnings:** Alerts if space is low

**Example:**
```
💾 DISK SPACE:
   Available: 1297 GB
   Used: 26%
```

### 7. Warnings
- **Aggregated Warnings:** All issues needing attention
- **Categories:** model, config, daemon, disk, backup
- **Actionable:** Each warning suggests what to do

**Example:**
```
⚠️  WARNINGS:
   • [backup] Current model exists but no versions saved - create backup!
   • [disk] Disk space getting low: 45GB free
```

---

## 💻 Usage Examples

### Basic Usage

```bash
# Full report (recommended)
python3 state_tracker.py --check

# Quick warnings check
python3 state_tracker.py --warnings

# Update state file
python3 state_tracker.py

# JSON output (for scripts)
python3 state_tracker.py --json
```

### Example Workflows

**Workflow 1: Starting a new session**
```bash
# 1. Run state tracker
python3 state_tracker.py --check

# 2. Read the output - ESPECIALLY WARNINGS
# 3. Address any warnings before proceeding
# 4. Now safe to work
```

**Workflow 2: Before deleting anything**
```bash
# 1. Check what exists
python3 state_tracker.py --check

# 2. Review model size, training steps
# 3. Check if versions are saved
# 4. Only delete if you're SURE it's safe
```

**Workflow 3: Quick health check**
```bash
# Just show warnings
python3 state_tracker.py --warnings

# If no warnings = system healthy
# If warnings = read and address them
```

**Workflow 4: Scripted monitoring**
```bash
# Get JSON output
python3 state_tracker.py --json > current_state.json

# Parse with jq
cat current_state.json | jq '.warnings'
cat current_state.json | jq '.current_model.training_steps'
```

---

## 📁 Output Files

### `.system_state.json`

**Location:** `/path/to/training/.system_state.json`

**Purpose:** Machine-readable system state

**Format:**
```json
{
  "last_updated": "2025-11-16T06:41:11.480706",
  "current_model": {
    "exists": true,
    "path": "current_model/",
    "size_gb": 32.66,
    "has_adapter": true,
    "training_steps": 685,
    "last_training": "2025-11-16T06:41:10.995484",
    "warning": null
  },
  "versions": [
    {
      "id": "v001",
      "date": "2025-11-14",
      "description": "Math training",
      "training_steps": 5000,
      "size_gb": 30.2
    }
  ],
  "config": {
    "exists": true,
    "model_name": "qwen3_8b",
    "base_model": "/path/to/training/DIO_20251114",
    "max_length": 2048,
    "locked_params": ["max_length", "base_model", "model_name"],
    "warning": null
  },
  "training_status": {
    "status": "training",
    "current_step": 685,
    "total_evals": 57,
    "last_update": "2025-11-16T06:41:10"
  },
  "daemon_running": true,
  "disk_space": {
    "available_gb": 1297,
    "used_percent": 26,
    "warning": null
  },
  "warnings": [
    ["backup", "Current model exists but no versions saved - create backup!"]
  ]
}
```

**Use Cases:**
- Future Claude reads this file to know system state
- Scripts can parse it for automation
- Provides snapshot for debugging
- Audit trail of system state over time

---

## 🚨 Warning Types

### model
**Cause:** Issues with current model
**Examples:**
- "No current model - fresh start"
- "No adapter found - base model only"

**Action:** Usually informational, but check if expected

### config
**Cause:** Configuration issues
**Examples:**
- "Config file missing!"
- "Error reading config: [error message]"

**Action:** Fix config.json or ask user

### daemon
**Cause:** Training daemon status issues
**Examples:**
- "Training daemon not running (model exists with training progress)"

**Action:** Restart daemon if training should be active

### disk
**Cause:** Low disk space
**Examples:**
- "LOW DISK SPACE: Only 8GB free!"
- "Disk space getting low: 45GB free"

**Action:** Free up space or add storage

### backup
**Cause:** No backups for current model
**Examples:**
- "Current model exists but no versions saved - create backup!"

**Action:** Create version snapshot

---

## 🔄 When to Run

### Required (Run Every Time)
1. **Start of new session** - Know what exists
2. **Before deletion** - See what you're deleting
3. **Before consolidation** - Check progress
4. **After major changes** - Verify system state

### Recommended
5. **After training** - Update state file
6. **Weekly** - Health check
7. **Before risky operations** - Verify assumptions

### Optional
8. **Debugging** - Understand system state
9. **Reporting** - Export JSON for analysis

---

## 🛠️ Troubleshooting

### State tracker not found
```bash
# Make sure you're in the right directory
cd /path/to/training

# Make it executable
chmod +x state_tracker.py

# Run with python3 explicitly
python3 state_tracker.py --check
```

### Permission errors
```bash
# Ensure you have read access
ls -l .system_state.json

# Regenerate state file
python3 state_tracker.py
```

### Warnings not showing
```bash
# Some warnings only appear in certain conditions
# Run full check to see all details
python3 state_tracker.py --check
```

### JSON parsing errors
```bash
# Validate JSON
cat .system_state.json | jq .

# Regenerate if corrupted
rm .system_state.json
python3 state_tracker.py
```

---

## 🎯 Integration with Other Tools

### With Model Versioner
```bash
# Check versions
python3 state_tracker.py --check | grep "VERSIONS:"

# Then manage versions
python3 model_versioner.py list
```

### With Training Controller
```bash
# Check daemon status
python3 state_tracker.py --warnings | grep daemon

# Then control if needed
python3 training_controller.py status
```

### With Backup Manager
```bash
# Check if backups needed
python3 state_tracker.py --warnings | grep backup

# Create backup if warned
python3 backup_manager.py backup current_model/
```

---

## 📊 State File Schema

```typescript
interface SystemState {
  last_updated: string;  // ISO 8601 timestamp
  
  current_model: {
    exists: boolean;
    path: string;
    size_gb: number;
    has_adapter?: boolean;
    training_steps: number;
    last_training: string | null;
    warning: string | null;
  };
  
  versions: Array<{
    id: string;           // v001, v002, etc.
    date: string;         // YYYY-MM-DD
    description: string;
    training_steps: number;
    size_gb: number;
  }>;
  
  config: {
    exists: boolean;
    model_name?: string;
    base_model?: string;
    max_length?: number;
    locked_params: string[];
    warning: string | null;
  };
  
  training_status: {
    status: string;       // idle, training, paused, error
    current_step: number;
    total_evals: number;
    last_update: string | null;
  };
  
  daemon_running: boolean;
  
  disk_space: {
    available_gb: number;
    used_percent: number;
    warning: string | null;
  };
  
  warnings: Array<[string, string]>;  // [category, message]
}
```

---

## 🎓 Best Practices

### DO
- ✅ Run at start of every session
- ✅ Read warnings carefully
- ✅ Address warnings before proceeding
- ✅ Use `--check` for full report
- ✅ Use `--warnings` for quick check
- ✅ Parse JSON for automation

### DON'T
- ❌ Ignore warnings
- ❌ Skip state check before operations
- ❌ Assume system state without checking
- ❌ Delete `.system_state.json`

---

## 🚀 Future Enhancements

**Possible additions:**
- Historical state tracking (state over time)
- Automated alerting (email/slack on warnings)
- Web UI for state visualization
- State comparison (diff between states)
- Scheduled state checks (cron job)

**Current status:** Phase 4 complete, working well

---

## 📝 Quick Reference Card

```
STATE TRACKER QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASIC USAGE:
  python3 state_tracker.py --check     Full report
  python3 state_tracker.py --warnings  Warnings only
  python3 state_tracker.py            Update state file
  python3 state_tracker.py --json     JSON output

WHEN TO RUN:
  ✓ Start of every session (REQUIRED)
  ✓ Before deletion
  ✓ Before consolidation
  ✓ After major changes
  ✓ When uncertain

OUTPUT FILE:
  .system_state.json (machine-readable)

INTEGRATION:
  Works with: model_versioner.py
              training_controller.py
              backup_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

**END OF STATE TRACKER GUIDE**
