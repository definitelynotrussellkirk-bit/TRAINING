# RTX 3090 Inference Server

This directory contains the inference API server that runs on the RTX 3090 machine.

## Architecture

- **main.py** - FastAPI server providing inference, model management, and system APIs
- **inference_worker.py** - GPU worker that loads models and runs inference

## Data Directories (not in git)

The server uses `~/llm/` for data storage:

```
~/llm/
├── models/          # Model weights (Qwen3-0.6B, deployed/, etc.)
├── data/            # Training/eval data
├── checkpoints/     # Checkpoint storage
├── datasets/        # Dataset files
├── logs/            # Server logs
├── db.sqlite        # Job/model database
└── venv/            # Python virtual environment
```

## Setup on 3090

```bash
# Clone repo (if not already)
cd ~
git clone https://github.com/definitelynotuserellkirk-bit/TRAINING.git

# Ensure data directory exists
mkdir -p ~/llm/{models,data,checkpoints,datasets,logs}

# Activate venv
source ~/llm/venv/bin/activate

# Run server
cd ~/TRAINING/inference
python main.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health and GPU status |
| `/models` | GET | List registered models |
| `/models/info` | GET | Currently loaded model info |
| `/models/reload` | POST | Force reload deployed model |
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |
| `/gpu` | GET | Detailed GPU statistics |
| `/system` | GET | CPU/RAM/disk statistics |

## Deployment Flow

1. 4090 trains model, saves checkpoints
2. `deployment_orchestrator.py` (on 4090) syncs best checkpoint to 3090
3. Files go to `~/llm/models/deployed/`
4. Call `/models/reload` to load new model
5. Server serves inference from deployed model

## Port

Default: **8765**

```bash
curl http://192.168.x.x:8765/health
```
