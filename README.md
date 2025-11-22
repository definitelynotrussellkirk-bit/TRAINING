# LLM Training System

**Purpose:** Pure training module for continuous fine-tuning of small language models.

## Architecture Overview

This system is designed as a **training-only module**:
- **This machine:** Training, checkpoint management, monitoring training metrics
- **Remote RTX 3090:** All inference, generation, evaluation, and testing

### Core Components

**Training Pipeline:**
- `core/train.py` - HuggingFace Trainer wrapper with custom features
- `core/training_daemon.py` - File watcher and orchestrator
- `core/training_queue.py` - Priority-based queue system
- `core/validator.py` - Pre-training data validation

**Model Management:**
- `management/backup_manager.py` - Backup system with retention policies
- `management/model_versioner.py` - Version control for model snapshots
- `management/consolidate_model.py` - Checkpoint consolidation
- `management/auto_disk_manager.py` - Automatic disk space management

**Safety & Monitoring:**
- `safety/daemon_watchdog.py` - Auto-restart crashed processes
- `safety/crash_detector.py` - Analyze crash logs
- `monitoring/servers/` - Web UI for training metrics

### Current Model

**Base Model:** Qwen3-0.6B
- Location: `models/Qwen3-0.6B/`
- Size: 1.5GB
- Architecture: Qwen3ForCausalLM (28 layers, 1024 hidden size)
- Training Method: Full model fine-tuning (all weights trainable, no LoRA)

### Data Flow

1. Drop `.jsonl` training file into `inbox/`
2. Daemon detects file (30-second polling)
3. Validation checks token lengths and format
4. File moved to priority queue (`queue/high/`, `queue/normal/`, or `queue/low/`)
5. Training processes one file at a time
6. Checkpoints saved to `models/current_model/` every N steps
7. Completed files archived to `queue/recently_completed/`
8. Failed files moved to `queue/failed/` for analysis

### Training Features

**Full Model Fine-tuning:**
- Updates all model weights directly (no adapter layers)
- Efficient for small models (<1B parameters)
- Preserves full model capacity

**Custom Training Features:**
- Logit bias penalties for unwanted patterns (e.g., `<think>` tags)
- Variable stop emoji sequences (random selection from pool)
- Real-time validation loss tracking
- Automatic checkpoint cleanup

**Continuous Training:**
- Global step counter never resets
- Seamless resumption from latest checkpoint
- Accumulates training across multiple data files

### Hardware Requirements

- **GPU:** 24GB VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **Disk:** ~50GB free space minimum (auto-managed)
- **RAM:** 32GB+ recommended

### Configuration

Active config: `config.json`

Key settings:
- `batch_size`: Training batch size (adjust based on VRAM)
- `learning_rate`: 2e-4 (default)
- `max_length`: 4096 tokens (max sequence length)
- `eval_steps`: Validation frequency
- `save_steps`: Checkpoint frequency
- `poll_interval`: Inbox polling frequency (seconds)

### Monitoring

**Web UI:** http://localhost:8080/live_monitor_ui_v2.html
- Real-time loss charts
- GPU/RAM monitoring
- Training progress and time estimates
- Overfitting detection (train/val gap)

**Command Line:**
```bash
# Check status
cat status/training_status.json | jq .

# Watch logs
tail -f logs/daemon_$(date +%Y%m%d).log

# GPU monitoring
nvidia-smi
```

### Getting Started

See `QUICKSTART.md` for setup and usage instructions.

### Architecture Details

See `ARCHITECTURE.md` for deep dive into system design.

### Troubleshooting

See `TROUBLESHOOTING.md` for common issues and solutions.
