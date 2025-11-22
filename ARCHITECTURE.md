# System Architecture

**Design Philosophy:** Pure training module - all training on this machine, all inference on remote RTX 3090.

## System Split

### This Machine (Training Module)
**Purpose:** Continuous model training with minimal overhead

**Responsibilities:**
- Accept training data (`.jsonl` files)
- Validate data before training
- Train model with full fine-tuning
- Save checkpoints at regular intervals
- Monitor training metrics (loss, validation loss)
- Manage disk space and backups
- Version control for model snapshots

**NOT responsible for:**
- Inference/generation
- Live example outputs
- Self-correction data generation
- Model evaluation on test sets

### Remote RTX 3090 (Inference Module)
**Purpose:** All inference, generation, and testing

**Responsibilities:**
- Run trained models for inference
- Generate training data (e.g., syllogistic logic examples)
- Evaluate model outputs
- Create self-correction training data
- Quality assessment and testing

## Core Components

### 1. Training Daemon (`core/training_daemon.py`)

**Main orchestrator running 24/7**

```
┌─────────────────────────────────────┐
│     Training Daemon (Main Loop)     │
│                                     │
│  1. Poll inbox/ every 30s          │
│  2. Validate new files             │
│  3. Move to priority queue         │
│  4. Load file from queue           │
│  5. Train with UltimateTrainer     │
│  6. Save checkpoints               │
│  7. Update status.json             │
│  8. Clean up completed files       │
│  9. Check for .stop signal         │
│  10. Repeat                        │
└─────────────────────────────────────┘
```

**Key Features:**
- Non-blocking file watching
- Graceful shutdown on `.stop` file
- Automatic recovery from crashes (via watchdog)
- Status updates to JSON for monitoring
- Queue-based processing (high/normal/low priority)

### 2. Training Script (`core/train.py`)

**HuggingFace Trainer wrapper with custom features**

```python
UltimateTrainer:
├── Validation (pre-training checks)
├── Model Loading (AutoModelForCausalLM)
├── Dataset Preparation
│   ├── Load JSONL
│   ├── Tokenize
│   └── Train/Val Split
├── Custom Features
│   ├── Logit Penalty Processors
│   │   ├── Think Tag Penalty (<think> suppression)
│   │   └── Post-Stop Penalty (tokens after stop emoji)
│   ├── Variable Stop Emojis (random selection)
│   └── Custom Data Collator
└── Training Loop (HuggingFace Trainer)
    ├── Forward pass
    ├── Loss calculation
    ├── Backward pass
    ├── Optimizer step
    └── Checkpoint saving
```

**Training Method:**
- **Full model fine-tuning** (no LoRA adapters)
- All weights trainable: ~677M parameters (Qwen3-0.6B)
- Direct weight updates via AdamW optimizer
- Learning rate: 2e-4 with warmup

**Why no LoRA?**
- Model is small enough (<1B params) to fit entirely in 24GB VRAM
- Full fine-tuning preserves complete model capacity
- No adapter overhead during training
- Simpler checkpoint management

### 3. Queue System (`core/training_queue.py`)

**Priority-based file processing**

```
inbox/
  └─> [Validation] ─> queue/high/    (priority 1)
                  ├─> queue/normal/  (priority 2)
                  └─> queue/low/     (priority 3)
                        │
                        v
                  [Daemon picks next file]
                        │
                        v
                  queue/processing/
                        │
                        ├─> [Success] ─> queue/recently_completed/
                        └─> [Failure] ─> queue/failed/
```

**Features:**
- Priority-based ordering
- Atomic file operations (prevents corruption)
- Metadata tracking (attempts, timestamps, errors)
- Retry logic (up to 3 attempts)
- Failed file isolation for debugging

### 4. Checkpoint Management

**Continuous training across multiple files**

```
HuggingFace Trainer Checkpoints:
├── models/current_model/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   ├── checkpoint-3000/
│   └── ... (auto-cleaned after N checkpoints)
│
└── Global Step Counter
    ├── Never resets between files
    ├── Accumulates across all training
    └── Enables true continuous learning
```

**Checkpoint Contents:**
- Model weights (safetensors format)
- Optimizer state (AdamW)
- Learning rate scheduler state
- Global step count
- RNG states (for reproducibility)

**Retention:**
- Keep latest N checkpoints (configurable)
- Auto-cleanup via `checkpoint_retention.py`
- Safe deletion (blocks during active training)

### 5. Model Versioning (`management/model_versioner.py`)

**Snapshot system for major milestones**

```
models/
├── Qwen3-0.6B/              # Base model (never modified)
├── current_model/           # Active training checkpoint
└── backups/
    └── consolidated/
        ├── v001_YYYYMMDD/   # Snapshot at milestone 1
        ├── v002_YYYYMMDD/   # Snapshot at milestone 2
        └── v003_YYYYMMDD/   # Snapshot at milestone 3
```

**Operations:**
- `consolidate_model.py` - Create numbered snapshot with description
- `model_versioner.py restore v002` - Restore specific version
- `model_versioner.py rollback` - Quick rollback to previous
- `backup_manager.py` - Emergency backups with retention policies

### 6. Safety Systems

**Watchdog (`safety/daemon_watchdog.py`):**
- Monitors daemon process every 30 seconds
- Auto-restarts on crash within 60 seconds
- Kills orphaned processes
- Logs all recovery actions

**Anti-Stuck Monitor (`safety/anti_stuck_monitor.py`):**
- Detects training hangs (15 min timeout)
- Kills stuck processes
- Moves file back to queue for retry

**Config Validator (`safety/config_validator.py`):**
- Locks config during training
- Prevents accidental changes to critical parameters
- Validates paths exist before training

**Crash Detector (`safety/crash_detector.py`):**
- Analyzes logs for crash patterns
- Categorizes: OOM, CUDA, disk space, etc.
- Suggests recovery actions

### 7. Monitoring System

**Web UI (`monitoring/ui/live_monitor_ui_v2.html`):**
- Real-time loss charts (training + validation)
- GPU/RAM usage graphs
- Time remaining estimates
- Overfitting detection (train/val gap)

**API Servers:**
- `monitoring/servers/live_monitor.py` (port 8080)
- `monitoring/servers/memory_stats_api.py` (port 8081)
- `monitoring/servers/enhanced_monitor.py` (port 8082)

**Status Tracking (`core/training_status.py`):**
- Updates `status/training_status.json` every N steps
- Contains: current_step, loss, val_loss, GPU usage, ETA
- Consumed by web UI and monitoring tools

### 8. Data Validation (`core/validator.py`)

**Pre-training checks:**
1. Token length analysis (sample 100 examples)
2. Check against `max_length` config
3. Warn if >95% exceed limit
4. Block training if severe truncation detected
5. Suggest config adjustments

**Validation during training:**
- Fixed validation set (1000 examples in `data/validation/`)
- Separate from training data
- Tracks generalization (train/val gap)

## Data Format

**Training files:** JSONL (JSON Lines)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
```

**Features:**
- Supports multi-turn conversations
- System prompts optional
- Assistant response is the training target
- User messages + previous context = input

## Training Process

**Single File Training:**

```
1. Load file from queue
2. Validate token lengths
3. Load model from latest checkpoint
4. Tokenize dataset
5. Split train/val (99.9% / 0.1%)
6. Training loop:
   ├── Forward pass (batch_size examples)
   ├── Calculate loss
   ├── Backward pass
   ├── Optimizer step
   ├── Every eval_steps: Compute val_loss
   ├── Every save_steps: Save checkpoint
   └── Every log_steps: Update status.json
7. Training complete
8. Move file to recently_completed/
9. Return to queue for next file
```

**Continuous Training:**
- Global step never resets
- Next file resumes from last checkpoint
- Optimizer state preserved
- Learning rate schedule continues

## Memory Management

**GPU Memory:**
- Model: ~3GB (Qwen3-0.6B in bfloat16)
- Activations: ~8-12GB (depends on batch_size)
- Gradients: ~3GB (same as model)
- Optimizer states: ~6GB (AdamW doubles memory)
- **Total:** ~20-24GB for batch_size=19

**Disk Space:**
- Auto-managed by `management/auto_disk_manager.py`
- Monitors every 5 minutes
- Deletes old snapshots when <50GB free
- Keeps latest 2 versions always

**Checkpoint Cleanup:**
- Keeps latest N checkpoints (default: 5)
- Deletes older checkpoints automatically
- Safe deletion (won't delete during training)

## Control Flow

**Start System:**
```bash
scripts/start_all.sh
```

Starts:
1. Auto disk manager
2. Training daemon
3. Live monitor (web UI)
4. Memory stats API
5. Enhanced monitor

**Control Training:**
```bash
# Pause (finishes current step, waits)
python3 core/training_controller.py pause

# Resume
python3 core/training_controller.py resume

# Stop (finishes current file, exits)
python3 core/training_controller.py stop

# Emergency stop (creates .stop file)
touch .stop
```

**Queue Operations:**
```bash
# Add file with priority
python3 core/training_queue.py add data.jsonl --priority high

# Check queue status
python3 core/training_queue.py status

# List all queued files
python3 core/training_queue.py list
```

## Configuration

**config.json structure:**

```json
{
  "model_name": "qwen2.5_0.5b",
  "model_path": "/path/to/base/model",
  "batch_size": 19,
  "gradient_accumulation": 1,
  "learning_rate": 0.0002,
  "warmup_steps": 100,
  "eval_steps": 50,
  "save_steps": 1000,
  "max_length": 4096,
  "poll_interval": 30
}
```

**Critical parameters (require user approval to change):**
- `max_length` - Affects memory and data truncation
- `model_name` - Identifier for versioning
- `base_model` - Path to base model

**Tunable parameters:**
- `batch_size` - Adjust based on VRAM availability
- `eval_steps` - Validation frequency (affects overhead)
- `save_steps` - Checkpoint frequency (affects disk usage)

## Future: Remote Inference Integration

**Planned architecture for inference on remote RTX 3090:**

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   Training Module       │      │   Inference Module      │
│   (This Machine)        │      │   (Remote RTX 3090)     │
│                         │      │                         │
│  - Train models         │◄────►│  - Run inference        │
│  - Save checkpoints     │ HTTP │  - Generate data        │
│  - Manage versions      │ API  │  - Evaluate outputs     │
│  - Monitor training     │      │  - Create corrections   │
└─────────────────────────┘      └─────────────────────────┘
```

**API Design (planned):**
- REST endpoints for model upload/download
- Inference requests with model version
- Batch processing for efficiency
- Result streaming for real-time feedback

**Benefits:**
- Keeps training machine focused on training
- Inference doesn't compete for GPU resources
- Can run multiple inference jobs in parallel
- Easier to scale inference independently
