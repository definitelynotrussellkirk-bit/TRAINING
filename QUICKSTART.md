# Quick Start Guide

Get up and running with the training system in 5 minutes.

## Prerequisites

- **GPU:** 24GB VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **OS:** Linux (tested on Ubuntu)
- **Python:** 3.10+
- **Disk:** 50GB+ free space

## Installation

```bash
# Clone or navigate to training directory
cd /path/to/TRAINING

# Optional: set base dir for all commands
export TRAINING_BASE_DIR="$(pwd)"

# Install dependencies (if not already installed)
pip install torch transformers datasets peft accelerate
pip install bitsandbytes  # For potential 4-bit quantization
pip install jq  # For JSON parsing in shell
```

## First-Time Setup

### 1. Verify Base Model Exists

```bash
ls -lh models/Qwen3-0.6B/
# Should show model.safetensors (~1.5GB)
```

If missing, download from HuggingFace:
```bash
# Download Qwen3-0.6B
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

### 2. Initialize Current Model

```bash
# Copy base model to current_model/ for first training
cp -r models/Qwen3-0.6B/* models/current_model/
```

### 3. Review Configuration

```bash
cat config.json
```

Key settings (values from `config.json`):
- `batch_size`: 1 (with gradient_accumulation: 16)
- `max_length`: 2048 (max tokens per example)
- `eval_steps`: 500 (validation frequency)
- `save_steps`: 1000 (checkpoint frequency)
- `profile.name`: "emoji_think" (data transformation profile)

## Basic Usage

### Start Training System

```bash
# Start all services
scripts/start_all.sh
```

This launches:
1. Training daemon (watches inbox/, processes queue)
2. Auto disk manager (monitors disk space)
3. Live monitor (web UI on port 8080)
4. Unified monitoring API (port 8081) - aggregates all system metrics

### Verify Services Running

```bash
# Check processes
ps aux | grep -E "training_daemon|auto_disk_manager|live_monitor" | grep -v grep

# Check daemon status
python3 core/training_controller.py status
```

### Add Training Data

**Option 1: Drop file in inbox**
```bash
# Copy .jsonl file to inbox
cp /path/to/your/training_data.jsonl inbox/

# Daemon will auto-detect within 30 seconds
```

**Option 2: Add directly to queue**
```bash
# Add with priority
python3 core/training_queue.py add /path/to/data.jsonl --priority high

# Check queue
python3 core/training_queue.py status
```

### Monitor Training

**Web UI (recommended):**
```
Open browser: http://localhost:8080/live_monitor_ui_v2.html
```

Shows:
- Real-time loss charts
- GPU/RAM usage
- Training progress
- Time remaining

**Command line:**
```bash
# Watch status
watch -n 5 'cat status/training_status.json | jq .'

# Tail logs
tail -f logs/daemon_$(date +%Y%m%d).log

# GPU monitoring
watch -n 1 nvidia-smi
```

## Data Format

Training files must be JSONL (JSON Lines) with OpenAI chat format:

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
{"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "6"}]}
```

**With system prompt:**
```json
{"messages": [{"role": "system", "content": "You are a math tutor."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

**Multi-turn conversations:**
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4"}]}
```

## Common Operations

### Control Training

```bash
# Pause (finishes current step, then waits)
python3 core/training_controller.py pause

# Resume
python3 core/training_controller.py resume

# Stop (finishes current file, then exits)
python3 core/training_controller.py stop

# Emergency stop
touch .stop
```

### Check Queue Status

```bash
# View queue
python3 core/training_queue.py list

# Check counts
python3 core/training_queue.py status
```

### Create Model Snapshot

```bash
# After training milestone, create versioned snapshot
python3 management/consolidate_model.py \
  --description "Trained on 100k syllogistic examples"
```

This creates:
- Numbered version (v001, v002, etc.)
- Timestamped backup
- Description file for tracking

### Validate Data Before Training

```bash
# Check if data fits in max_length
python3 tools/data/validate_data.py --file my_data.jsonl

# Auto-adjust config if needed
python3 tools/data/validate_data.py --auto-adjust
```

## Typical Workflow

### 1. Prepare Training Data

```bash
# Example: Generate syllogistic logic data (requires remote inference)
# This would be done on remote machine, then transferred here
cp /from/remote/machine/syllo_data.jsonl inbox/
```

### 2. Start Training

```bash
# Start system if not running
scripts/start_all.sh

# Data auto-trains within 30 seconds
```

### 3. Monitor Progress

```bash
# Open web UI
xdg-open http://localhost:8080/live_monitor_ui_v2.html

# Or watch logs
tail -f logs/daemon_$(date +%Y%m%d).log
```

### 4. Check Metrics

```bash
# Current training metrics
cat status/training_status.json | jq '{step: .current_step, loss: .loss, val_loss: .validation_loss, gap: .val_train_gap}'

# Interpret gap (validation_loss - training_loss):
# < 0.3: Good generalization
# 0.3-0.5: Monitor closely
# > 0.5: Possible overfitting
```

### 5. Create Checkpoint

```bash
# After significant training (e.g., 10k steps)
python3 management/consolidate_model.py \
  --description "10k steps on logic data"
```

### 6. Test Model (on remote machine)

Transfer latest checkpoint to remote inference server for testing:

```bash
# Copy to remote (see config/hosts.json for inference host details)
rsync -avz models/current_model/ "${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/latest/"

# Run inference on remote machine (not on this training machine)
```

## Health Checks

### System Health

```bash
# Comprehensive check
python3 safety/comprehensive_health_check.py

# Quick check
scripts/check_health.sh

# State tracker
python3 tools/analysis/state_tracker.py --check
```

### Common Issues

**Daemon not running:**
```bash
ps aux | grep training_daemon | grep -v grep
# If not found, restart:
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

**OOM errors:**
```bash
# Reduce batch size
python3 tools/config/edit_config.py batch_size 16

# Restart daemon
pkill -f training_daemon
sleep 3
nohup python3 core/training_daemon.py > training_output.log 2>&1 &
```

**Queue stuck:**
```bash
# Check for stuck files
ls -lh queue/processing/
ls -lh queue/failed/

# Move back to normal queue (if safe)
mv queue/processing/* queue/normal/
mv queue/failed/* queue/normal/
```

## Next Steps

- Read **ARCHITECTURE.md** for deep dive into system design
- Read **TROUBLESHOOTING.md** for detailed problem solving
- Check **DEVELOPMENT.md** for contributing to the codebase
- Track changes in **CHANGELOG.md**

## Quick Reference

**Key Directories:**
- `inbox/` - Drop training files here
- `queue/` - Priority queues (high/normal/low)
- `models/current_model/` - Active training checkpoint
- `logs/` - Training logs
- `status/` - Real-time status JSON

**Key Commands:**
```bash
# Start system
scripts/start_all.sh

# Control training
python3 core/training_controller.py [pause|resume|stop|status]

# Queue management
python3 core/training_queue.py [add|list|status]

# Health check
python3 safety/comprehensive_health_check.py

# Create snapshot
python3 management/consolidate_model.py --description "..."
```

**Key URLs:**
- Main UI: http://localhost:8080/live_monitor_ui_v2.html
- Unified API: http://localhost:8081/api/unified
- Status JSON: http://localhost:8080/status/training_status.json
