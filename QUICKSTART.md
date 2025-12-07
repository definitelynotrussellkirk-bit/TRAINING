# Quick Start Guide

Get up and running with the Realm of Training in 10 minutes.

---

## SIMPLE MODE: Drag & Drop Training

**Just want to train on your own data? Skip the RPG features:**

### 1. Data Format (JSONL)

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
{"messages": [{"role": "user", "content": "Solve x^2=4"}, {"role": "assistant", "content": "x = 2 or x = -2"}]}
```

### 2. Drop Files & Train

```bash
# Drop your data file
cp my_training_data.jsonl inbox/

# Start training (uses active campaign)
python3 -m arena.hero_loop ojas-qwen3-8b/campaign-004
```

That's it. Files are processed oldest-first, moved to `completed/` when done.

### 3. Priority Folders (Optional)

```
inbox/
  high/    ← Processed FIRST
  normal/  ← Default priority (or just inbox/)
  low/     ← Processed LAST
```

```bash
# High priority - train on this immediately
cp urgent_data.jsonl inbox/high/

# Low priority - train when nothing else queued
cp background_data.jsonl inbox/low/
```

### 4. Disable Auto-Generation

Edit your hero config (`configs/heroes/<hero>.yaml`):

```yaml
idle_behavior:
  enabled: true
  generation:
    enabled: false  # ← Disables auto data generation when idle
```

This makes the system wait for your data instead of generating its own.

### 5. Feature Toggles (Hero Config)

| Feature | Config Key | Default | Effect |
|---------|------------|---------|--------|
| Auto-generation | `idle_behavior.generation.enabled` | true | Generate data when queue empty |
| Idle behavior | `idle_behavior.enabled` | true | Any idle actions at all |
| QLoRA | `qlora.enabled` | varies | Use adapter training |
| Gradient checkpointing | `training_defaults.gradient_checkpointing` | true | Save VRAM |

---

## Prerequisites

- **GPU:** 24GB VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **OS:** Linux (tested on Ubuntu)
- **Python:** 3.10+
- **Disk:** 50GB+ free space (more for larger models)

## Step 1: Clone & Bootstrap

```bash
# Clone the repository
git clone <repo-url> TRAINING
cd TRAINING

# Run bootstrap to set up directories and configs
./scripts/bootstrap_dev.sh

# Verify environment
python3 -m training doctor
```

The doctor will show what's ready and what needs attention.

## Step 2: Download a Base Model

Models go in the `models/` directory. Download using HuggingFace CLI:

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download a model (choose one based on your GPU)
# Small (fits any 24GB GPU easily):
huggingface-cli download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# Medium (good balance):
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B

# Larger (requires memory optimization):
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
```

**Which hero/model for your GPU?**

| Your VRAM | Recommended Hero | Model | Training Method | Download Size |
|-----------|------------------|-------|-----------------|---------------|
| 8-16GB | **DIO** | Qwen3-0.6B | Full fine-tuning | 1.2GB |
| 16-20GB | **GOU** | Qwen3-4B | Full or QLoRA | 8GB |
| 24GB+ | **OJAS** | Qwen3-8B | QLoRA required | 16GB |

**Quick decision:**
- **Learning/experimenting?** → DIO (fast, forgiving)
- **Serious training, 24GB GPU?** → OJAS (most capable)
- **In between?** → GOU (good balance)

## Step 3: Create Your Hero

Heroes are defined in `configs/heroes/`. Three heroes are pre-configured:

| Hero | Model | Description |
|------|-------|-------------|
| **DIO** | Qwen3-0.6B | The Skeptic - fast training, good for learning |
| **GOU** | Qwen3-4B | The Hound - balanced power and speed |
| **OJAS** | Qwen3-8B | The Vital Force - maximum capability (needs QLoRA) |

To create a new hero, copy the template:

```bash
# Copy template
cp configs/heroes/_template.yaml configs/heroes/my-hero.yaml
# Edit to match your downloaded model
```

## Step 4: Start a Campaign

A campaign links your hero to a model and tracks training progress.

**Option A: Activate an existing hero (recommended for first run)**

```bash
# Create the active campaign file
cat > control/active_campaign.json << 'EOF'
{
  "hero_id": "dio-qwen3-0.6b",
  "campaign_id": "campaign-001",
  "campaign_path": "campaigns/dio-qwen3-0.6b/campaign-001",
  "started_at": "2025-12-03T00:00:00"
}
EOF

# Create campaign directory
mkdir -p campaigns/dio-qwen3-0.6b/campaign-001
```

**Option B: Use GOU (4B model)**

```bash
cat > control/active_campaign.json << 'EOF'
{
  "hero_id": "gou-qwen3-4b",
  "campaign_id": "campaign-001",
  "campaign_path": "campaigns/gou-qwen3-4b/campaign-001",
  "started_at": "2025-12-03T00:00:00"
}
EOF

mkdir -p campaigns/gou-qwen3-4b/campaign-001
```

## Step 5: Configure the Model Path

Edit `config.json` to point to your model:

```bash
# For DIO (Qwen3-0.6B):
python3 -c "
import json
with open('config.json') as f:
    cfg = json.load(f)
cfg['model_path'] = 'models/Qwen3-0.6B'
cfg['base_model'] = 'models/Qwen3-0.6B'
cfg['model_name'] = 'qwen3_0.6b'
cfg['model_display_name'] = 'DIO - Qwen3-0.6B'
with open('config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
```

Or manually edit `config.json`:
```json
{
  "model_name": "qwen3_0.6b",
  "model_display_name": "DIO - Qwen3-0.6B",
  "model_path": "models/Qwen3-0.6B",
  "base_model": "models/Qwen3-0.6B",
  ...
}
```

## Step 6: Initialize Current Model

Copy the base model to the training directory:

```bash
# Create current_model from base model
cp -r models/Qwen3-0.6B/* models/current_model/
```

## Step 7: Start the Realm

```bash
# Start all services
python3 -m training start-all
```

This launches:
- **Tavern** (port 8888) - Game UI
- **VaultKeeper** (port 8767) - Asset registry
- **RealmState** (port 8866) - Real-time state
- **Training Daemon** - Watches queue, runs training

## Step 8: Visit the Tavern

Open your browser to **http://localhost:8888**

You should see:
- **Hero Card** - Your hero with Level 1, 0 XP
- **Status** - "Ready to train" (or "Idle" if no quests)
- **Quest Board** - Empty (drop files in inbox to create quests)
- **Vault** - Your base model checkpoint

## Step 9: Start Training

Training data goes in the `inbox/` directory as JSONL files.

**Format:**
```json
{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
```

**Add training data:**
```bash
# Copy a training file
cp /path/to/your/training_data.jsonl inbox/

# Or generate skill data (requires skill server running)
python3 guild/data_gen.py --skill sy --level 1 --count 1000
```

The daemon will:
1. Detect new files in inbox
2. Move them to the queue
3. Start training automatically

Watch training in the Tavern UI or via command line:
```bash
python3 core/training_controller.py status
```

---

## Quick Reference

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `models/` | Downloaded base models |
| `models/current_model/` | Active training checkpoint |
| `configs/heroes/` | Hero definitions (YAML) |
| `campaigns/` | Campaign data and analysis |
| `control/` | Active campaign, state files |
| `inbox/` | Drop training files here |
| `queue/` | Training queue (high/normal/low) |
| `status/` | Runtime status JSON |
| `logs/` | Training logs |

### Key Commands

```bash
# System health
python3 -m training doctor

# Start/stop services
python3 -m training start-all
python3 -m training stop-all

# Control training
python3 core/training_controller.py status
python3 core/training_controller.py pause
python3 core/training_controller.py resume
python3 core/training_controller.py stop

# Queue management
python3 core/training_queue.py status
python3 core/training_queue.py list
```

### Key URLs

| URL | Description |
|-----|-------------|
| http://localhost:8888 | Tavern - Main game UI |
| http://localhost:8888/quests | Quest Board |
| http://localhost:8888/oracle | Oracle - Chat with model |
| http://localhost:8888/vault | Vault - Checkpoint browser |
| http://localhost:8888/guild | Guild Hall - Skill progress |
| http://localhost:8888/settings | Settings & VRAM calculator |

### Common Issues

**"No Hero" in Tavern:**
- Check `control/active_campaign.json` exists
- Verify hero config exists in `configs/heroes/`

**"No model" or training errors:**
- Verify `models/current_model/` contains model files
- Check `config.json` model_path is correct

**Training not starting:**
- Check daemon is running: `ps aux | grep hero_loop`
- Check queue has files: `python3 core/training_queue.py status`
- Check inbox for stuck files: `ls inbox/`

---

## Next Steps

1. **Add training data** - Drop JSONL files in `inbox/`
2. **Watch progress** - Tavern UI shows real-time stats
3. **Level up** - As accuracy improves, skills level up
4. **Experiment** - Try different skills, models, techniques

Read **CLAUDE.md** for the full game design document and system reference.
