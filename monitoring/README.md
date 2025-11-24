# monitoring/ - Autonomous Intelligence & Monitoring Systems

**Status:** Production (10 systems running 24/7 on RTX 3090)
**Version:** 2.0.0
**Architecture:** Distributed intelligence with centralized API

---

## 📋 OVERVIEW

The `monitoring/` system provides 24/7 autonomous intelligence for training optimization, quality control, and infrastructure management. 10 independent systems run continuously, analyzing checkpoints, generating data, detecting regressions, and auto-correcting errors.

**Key Capabilities:**
- Autonomous curriculum optimization (A/B testing strategies)
- Adversarial example mining (hard case discovery)
- Regression detection (bad checkpoint alerts)
- Model comparison & ranking
- Confidence calibration
- Self-correction loops
- Automated testing
- Checkpoint deployment
- Data generation automation
- Real-time web UI

**Architecture:** 2 machines working together
- **RTX 4090 (Training):** Trains models, generates data
- **RTX 3090 (Inference):** Runs all 10 monitoring systems + web UI

---

## 🏗️ ARCHITECTURE

### System Overview

```
RTX 4090 (Training)              RTX 3090 (Inference + Monitoring)
===================              ==================================

Training Daemon          →       Checkpoint Sync Daemon
  ↓                                      ↓
Checkpoints Saved        →       10 Autonomous Systems:
  ↓                              1. Curriculum Optimizer
Data Generation ←               2. Adversarial Miner
Automation                       3. Regression Monitor
  ↓                              4. Model Comparison
Best Checkpoint ←               5. Confidence Calibrator
Auto-Deployment                  6. Self-Correction Loop
                                 7. Automated Testing
                                 8. Checkpoint Auto-Deploy
                                 9. Data Generation (also 4090)
                                10. Checkpoint Sync
                                      ↓
                                 Web UI (port 8080)
                                 API Server (port 5000)
```

### Data Flow

```
1. Training produces checkpoint
   ↓
2. Checkpoint synced to 3090 (checkpoint_sync_daemon.py)
   ↓
3. Automated testing runs (automated_testing_daemon.py)
   ↓
4. Model comparison ranks checkpoint (model_comparison_engine.py)
   ↓
5. Regression check (continuous_regression_monitor.py)
   ↓
6. If best checkpoint: auto-deploy (checkpoint_auto_deployment.py)
   ↓
7. Curriculum optimizer A/B tests strategies (curriculum_optimizer.py)
   ↓
8. Adversarial miner finds hard examples (adversarial_miner.py)
   ↓
9. Self-correction validates & fixes (self_correction_loop.py)
   ↓
10. Data generation creates new batches (data_generation_automation.py)
```

---

## 🤖 10 AUTONOMOUS SYSTEMS

### 1. Curriculum Optimizer
**File:** `curriculum_optimizer.py`
**Location:** RTX 3090
**Interval:** Every 5 minutes

**Purpose:** A/B tests different curriculum strategies to find optimal difficulty progression.

**How it works:**
1. Loads recent checkpoints (last 3)
2. Tests 5 curriculum strategies:
   - Easy→Medium→Hard (linear)
   - Random mix
   - Hard-first (reverse)
   - Interleaved (alternating)
   - Adaptive (based on loss)
3. Runs inference on validation set for each strategy
4. Compares accuracy/loss
5. Updates `status/curriculum_optimization.json` with best strategy

**Output:** `status/curriculum_optimization.json`
```json
{
  "best_strategy": "easy_to_hard",
  "accuracy": 0.85,
  "strategies_tested": 5,
  "last_update": "2025-11-24T10:30:00"
}
```

**Usage:**
```bash
# Manual run
python3 monitoring/curriculum_optimizer.py --base-dir ~/TRAINING --samples 50

# Daemon (runs on 3090)
nohup python3 monitoring/curriculum_optimizer.py \
  --base-dir ~/TRAINING --interval 300 --samples 50 \
  > logs/curriculum_optimizer.log 2>&1 &
```

---

### 2. Adversarial Miner
**File:** `adversarial_miner.py`
**Location:** RTX 3090
**Interval:** Every 5 minutes

**Purpose:** Mines hard examples from model failures to create targeted training data.

**How it works:**
1. Loads latest checkpoint
2. Runs inference on validation set
3. Identifies examples where model failed (wrong answer)
4. Analyzes failure patterns (specific skills, difficulty levels)
5. Generates similar hard examples
6. Saves to `data/adversarial_examples/`

**Output:** `data/adversarial_examples/hard_examples_YYYYMMDD.jsonl`
```jsonl
{"prompt": "...", "golden": "...", "model_failed": true, "pattern": "math_reasoning"}
{"prompt": "...", "golden": "...", "model_failed": true, "pattern": "logical_deduction"}
```

**Status:** `status/adversarial_mining.json`

**Usage:**
```bash
python3 monitoring/adversarial_miner.py \
  --base-dir ~/TRAINING --interval 300 --samples 100
```

---

### 3. Regression Monitor
**File:** `continuous_regression_monitor.py`
**Location:** RTX 3090
**Interval:** Every 5 minutes

**Purpose:** Detects bad checkpoints with significant loss increases (>15%).

**How it works:**
1. Loads last 5 checkpoints
2. Runs inference on fixed validation set
3. Compares loss to baseline (best checkpoint so far)
4. If loss increase >15%: Mark as regression
5. Alerts via `status/regression_monitoring.json`

**Output:** `status/regression_monitoring.json`
```json
{
  "regression_detected": true,
  "checkpoint": "checkpoint-1200",
  "baseline_loss": 0.45,
  "current_loss": 0.58,
  "increase_percent": 28.9,
  "alert_level": "critical"
}
```

**Usage:**
```bash
python3 monitoring/continuous_regression_monitor.py \
  --base-dir ~/TRAINING --interval 300
```

---

### 4. Model Comparison Engine
**File:** `model_comparison_engine.py`
**Location:** RTX 3090
**Interval:** Every 10 minutes

**Purpose:** Ranks checkpoints by composite score (accuracy, loss, calibration).

**How it works:**
1. Loads all available checkpoints
2. Runs comprehensive evaluation:
   - Accuracy on validation set
   - Loss on fixed set
   - Confidence calibration (ECE)
   - Output quality (length, format)
3. Calculates composite score (weighted average)
4. Ranks checkpoints best→worst
5. Updates `status/model_comparisons.json`

**Output:** `status/model_comparisons.json`
```json
{
  "rankings": [
    {
      "checkpoint": "checkpoint-1500",
      "score": 0.92,
      "accuracy": 0.87,
      "loss": 0.42,
      "ece": 0.08
    },
    {
      "checkpoint": "checkpoint-1400",
      "score": 0.89,
      "accuracy": 0.85,
      "loss": 0.45,
      "ece": 0.10
    }
  ],
  "best_checkpoint": "checkpoint-1500"
}
```

**Usage:**
```bash
python3 monitoring/model_comparison_engine.py \
  --base-dir ~/TRAINING --interval 600
```

---

### 5. Confidence Calibrator
**File:** `confidence_calibrator.py`
**Location:** RTX 3090
**Interval:** Every 10 minutes

**Purpose:** Calibrates model prediction confidence into 6 bins.

**How it works:**
1. Loads latest checkpoint
2. Runs inference with logit outputs
3. Extracts prediction confidence (softmax probabilities)
4. Bins into 6 confidence levels: [0-0.2, 0.2-0.4, ..., 0.8-1.0]
5. Calculates accuracy within each bin
6. Updates `status/confidence_calibration.json`

**Output:** `status/confidence_calibration.json`
```json
{
  "bins": [
    {"range": "0.0-0.2", "accuracy": 0.15, "count": 45},
    {"range": "0.2-0.4", "accuracy": 0.32, "count": 78},
    {"range": "0.8-1.0", "accuracy": 0.95, "count": 234}
  ],
  "ece": 0.08,
  "well_calibrated": true
}
```

**Usage:**
```bash
python3 monitoring/confidence_calibrator.py \
  --base-dir ~/TRAINING --interval 600
```

---

### 6. Self-Correction Loop
**File:** `self_correction_loop.py`
**Location:** RTX 3090
**Interval:** Every 5 minutes

**Purpose:** Validates data quality, captures errors, generates correction examples.

**How it works:**
1. Loads training data from queue
2. Validates format (ChatML structure, required fields)
3. Runs inference to check if model can answer
4. Captures errors (parsing failures, wrong answers)
5. Generates correction examples (prompt + error analysis + correct answer)
6. Saves to `queue/corrections/`

**Output:**
- `queue/corrections/corrections_YYYYMMDD.jsonl` - Correction examples
- `logs/error_patterns/*.json` - Error analysis by pattern

**Status:** `status/self_correction.json`

**Usage:**
```bash
python3 monitoring/self_correction_loop.py \
  --continuous --interval 300 --error-threshold 50
```

---

### 7. Automated Testing Daemon
**File:** `automated_testing_daemon.py`
**Location:** RTX 3090
**Interval:** Every 10 minutes

**Purpose:** Runs fixed validation suite against checkpoints, detects regressions.

**How it works:**
1. Loads fixed validation set (100 examples, difficulty-balanced)
2. Tests latest checkpoint
3. Calculates accuracy by difficulty (easy/medium/hard)
4. Compares to baseline (best checkpoint)
5. Detects regressions (>5% accuracy drop)
6. Updates `status/automated_testing.json`

**Output:** `status/automated_testing.json`
```json
{
  "checkpoint": "checkpoint-1500",
  "overall_accuracy": 0.87,
  "by_difficulty": {
    "easy": 0.95,
    "medium": 0.85,
    "hard": 0.72
  },
  "regression_detected": false,
  "baseline": "checkpoint-1400"
}
```

**Usage:**
```bash
python3 monitoring/automated_testing_daemon.py --interval 600
```

---

### 8. Checkpoint Auto-Deployment
**File:** `checkpoint_auto_deployment.py`
**Location:** RTX 4090
**Interval:** Every 10 minutes

**Purpose:** Auto-deploys best checkpoint to 3090 for inference.

**How it works:**
1. Reads `status/model_comparisons.json` from 3090
2. Gets best checkpoint by score
3. Copies to 3090: `~/llm/models/Qwen3-0.6B/`
4. Restarts inference API on 3090
5. Updates `status/last_deployment.json`

**Output:** `status/last_deployment.json`
```json
{
  "checkpoint": "checkpoint-1500",
  "deployed_at": "2025-11-24T10:30:00",
  "score": 0.92,
  "success": true
}
```

**Usage:**
```bash
python3 monitoring/checkpoint_auto_deployment.py
```

---

### 9. Data Generation Automation
**File:** `data_generation_automation.py`
**Location:** RTX 4090 (also monitors 3090 curriculum)
**Interval:** Every 5 minutes

**Purpose:** Auto-generates training data when queue < 2 files.

**How it works:**
1. Checks queue size (`queue/normal/`)
2. If < 2 files: Trigger data generation
3. Reads curriculum strategy from 3090
4. Generates data with optimal difficulty distribution
5. Saves to `queue/normal/`

**Output:** New JSONL files in `queue/normal/`

**Usage:**
```bash
python3 monitoring/data_generation_automation.py \
  --check-interval 300 --min-files 2
```

---

### 10. Checkpoint Sync Daemon
**File:** `checkpoint_sync_daemon.py`
**Location:** RTX 3090
**Interval:** Every 5 minutes

**Purpose:** Syncs checkpoints from 4090 (training) to 3090 (inference).

**How it works:**
1. SSH to 4090
2. List checkpoints in `models/current_model/`
3. Rsync new checkpoints to 3090: `~/TRAINING/current_model/`
4. Keep last N checkpoints (default: 3)
5. Delete old checkpoints to save space

**Output:** Checkpoints in `~/TRAINING/current_model/` on 3090

**Usage:**
```bash
python3 monitoring/checkpoint_sync_daemon.py \
  --remote-host 192.168.x.x \
  --remote-dir /path/to/training/current_model \
  --local-dir /home/user/TRAINING/current_model \
  --interval 300 --keep 3
```

---

## 🌐 WEB UI & API

### Live Monitor (Port 8080)
**File:** `servers/live_monitor.py`
**Purpose:** Real-time training dashboard

**Features:**
- Live training progress (step, loss, lr)
- Inference previews (prompt, golden, model output)
- GPU stats (memory, utilization)
- Recent accuracy
- Validation metrics
- Think tag tracking

**Access:** http://192.168.x.x:8080/live_monitor_ui_v2.html

### API Server (Port 5000)
**File:** `api/server.py`
**Purpose:** Centralized API for all monitoring systems

**Plugin Architecture:**
- `api/plugins/training_status.py` - Training progress
- `api/plugins/adversarial.py` - Adversarial mining status
- `api/plugins/curriculum.py` - Curriculum optimization
- `api/plugins/regression.py` - Regression monitoring
- `api/plugins/model_comparison.py` - Checkpoint rankings
- `api/plugins/confidence.py` - Confidence calibration
- `api/plugins/self_correction.py` - Self-correction loop
- `api/plugins/testing.py` - Automated testing
- `api/plugins/checkpoints.py` - Checkpoint sync
- `api/plugins/gpu_stats.py` - GPU monitoring

**Endpoints:**
```
GET  /api/status              - Overall system status
GET  /api/training            - Training progress
GET  /api/adversarial         - Adversarial mining
GET  /api/curriculum          - Curriculum optimization
GET  /api/regression          - Regression monitoring
GET  /api/model_comparison    - Checkpoint rankings
GET  /api/confidence          - Confidence calibration
GET  /api/self_correction     - Self-correction
GET  /api/testing             - Automated testing
GET  /api/checkpoints         - Checkpoint sync
GET  /api/gpu                 - GPU stats
```

**Access:** http://192.168.x.x:5000/api/status

---

## 📊 STATUS FILES

All systems write JSON status files to `status/`:

```
status/
├── training_status.json           # Training progress (core)
├── curriculum_optimization.json   # Best curriculum strategy
├── adversarial_mining.json        # Hard examples found
├── regression_monitoring.json     # Regression alerts
├── model_comparisons.json         # Checkpoint rankings
├── confidence_calibration.json    # Confidence bins
├── self_correction.json           # Error patterns
├── automated_testing.json         # Validation results
├── last_deployment.json           # Latest checkpoint deployed
└── checkpoint_sync.json           # Sync status
```

**Reading Status:**
```python
import json

# Check if regression detected
with open("status/regression_monitoring.json") as f:
    status = json.load(f)
    if status["regression_detected"]:
        print(f"Regression in {status['checkpoint']}")

# Get best checkpoint
with open("status/model_comparisons.json") as f:
    rankings = json.load(f)
    best = rankings["best_checkpoint"]
    print(f"Best checkpoint: {best}")
```

---

## 🚀 STARTING ALL SYSTEMS

### On RTX 3090 (Inference + Monitoring)

```bash
ssh 192.168.x.x

cd ~/TRAINING

# 1. Curriculum Optimizer
nohup python3 monitoring/curriculum_optimizer.py \
  --base-dir ~/TRAINING --interval 300 --samples 50 \
  > logs/curriculum_optimizer.log 2>&1 &

# 2. Adversarial Miner
nohup python3 monitoring/adversarial_miner.py \
  --base-dir ~/TRAINING --interval 300 --samples 100 \
  > logs/adversarial_miner.log 2>&1 &

# 3. Regression Monitor
nohup python3 monitoring/continuous_regression_monitor.py \
  --base-dir ~/TRAINING --interval 300 \
  > logs/regression_monitor.log 2>&1 &

# 4. Model Comparison
nohup python3 monitoring/model_comparison_engine.py \
  --base-dir ~/TRAINING --interval 600 \
  > logs/model_comparison.log 2>&1 &

# 5. Confidence Calibrator
nohup python3 monitoring/confidence_calibrator.py \
  --base-dir ~/TRAINING --interval 600 \
  > logs/confidence_calibrator.log 2>&1 &

# 6. Self-Correction Loop
nohup python3 monitoring/self_correction_loop.py \
  --continuous --interval 300 --error-threshold 50 \
  > logs/self_correction.log 2>&1 &

# 7. Automated Testing
nohup python3 monitoring/automated_testing_daemon.py \
  --interval 600 \
  > logs/automated_testing.log 2>&1 &

# 10. Checkpoint Sync
nohup python3 monitoring/checkpoint_sync_daemon.py \
  --remote-host 192.168.x.x \
  --remote-dir /path/to/training/current_model \
  --local-dir /home/user/TRAINING/current_model \
  --interval 300 --keep 3 \
  > logs/checkpoint_sync.log 2>&1 &

# Web UI
nohup python3 monitoring/servers/live_monitor.py \
  > logs/live_monitor.log 2>&1 &

# API Server
nohup python3 monitoring/api/server.py \
  > logs/api_server.log 2>&1 &

# Verify all running
ps aux | grep -E '(curriculum|adversarial|regression|comparison|calibrator|self_correction|automated_testing|checkpoint_sync)' | grep python | grep -v grep
```

### On RTX 4090 (Training + Automation)

```bash
cd ~/Desktop/TRAINING

# 8. Checkpoint Auto-Deployment
nohup python3 monitoring/checkpoint_auto_deployment.py \
  > logs/checkpoint_auto_deployment.log 2>&1 &

# 9. Data Generation Automation
nohup python3 monitoring/data_generation_automation.py \
  --check-interval 300 --min-files 2 \
  > logs/data_generation_automation.log 2>&1 &

# Verify running
ps aux | grep -E '(checkpoint_auto_deployment|data_generation_automation)' | grep python | grep -v grep
```

---

## 🔧 UTILITIES

### Preview Engine
**File:** `preview_engine.py`
**Purpose:** Quick inference preview at specific training step

```bash
python3 monitoring/preview_engine.py --step 66400 --count 5
```

### Quick Validation
**File:** `quick_validation.py`
**Purpose:** Fast validation of any checkpoint

```bash
python3 monitoring/quick_validation.py \
  --model-path ~/llm/models/Qwen3-0.6B \
  --samples 10 \
  --output /tmp/validation.json
```

### Queue Health
**File:** `queue_health.py`
**Purpose:** Check queue status and health

```bash
python3 monitoring/queue_health.py --base-dir ~/TRAINING
```

### Compare Checkpoints
**File:** `compare_checkpoints.py`
**Purpose:** Side-by-side checkpoint comparison

```bash
python3 monitoring/compare_checkpoints.py \
  --checkpoint1 models/checkpoint-1400 \
  --checkpoint2 models/checkpoint-1500
```

---

## 🐛 TROUBLESHOOTING

### System Not Running
```bash
# Check if running
ps aux | grep curriculum_optimizer | grep -v grep

# Check logs
tail -50 logs/curriculum_optimizer.log

# Restart
pkill -f curriculum_optimizer
nohup python3 monitoring/curriculum_optimizer.py \
  --base-dir ~/TRAINING --interval 300 --samples 50 \
  > logs/curriculum_optimizer.log 2>&1 &
```

### Out of Date Status
```bash
# Check timestamp
cat status/curriculum_optimization.json | jq '.last_update'

# Force update
python3 monitoring/curriculum_optimizer.py \
  --base-dir ~/TRAINING --samples 50
```

### API Server Down
```bash
# Check if running
ps aux | grep "api/server.py" | grep -v grep

# Check port
lsof -i :5000

# Restart
pkill -f "api/server.py"
python3 monitoring/api/server.py &
```

---

## 📈 PERFORMANCE

**System Resource Usage:**
- Each daemon: ~200-500MB RAM
- Total: ~4GB RAM on RTX 3090
- GPU usage: ~2GB VRAM (inference only)
- CPU: <5% per daemon

**Status File Sizes:**
- Each status file: ~1-50KB
- Total: <1MB for all status files
- Updated every 5-10 minutes

**Inference Load:**
- Each system runs 10-100 inferences per cycle
- Total: ~500-1000 inferences every 5 minutes
- RTX 3090 handles easily (~100ms per inference)

---

## 🤝 RELATED MODULES

- **core/train.py** - Training loop (writes checkpoints)
- **core/training_daemon.py** - Orchestrates training
- **core/training_status.py** - Status tracking
- **management/** - Checkpoint management
- **safety/** - Health checks & watchdogs

---

## 📝 HISTORY

- **2025-11-15:** Initial monitoring systems created
- **2025-11-18:** Added API plugin architecture
- **2025-11-20:** Launched 10 autonomous systems
- **2025-11-22:** Integrated with training pipeline
- **2025-11-23:** Full 24/7 autonomous operation

---

**For detailed information on specific systems, see their inline documentation.**
