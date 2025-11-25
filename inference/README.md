# RTX 3090 Inference Server

This directory contains the inference API server that runs on the RTX 3090 machine.

## Architecture

- **main.py** - FastAPI server providing inference, model management, and system APIs
- **inference_worker.py** - GPU worker that loads models and runs inference
- **auth.py** - API key authentication middleware

## Authentication

All endpoints (except `/health`) require API key authentication.

### Setup

Set environment variables before starting the server:

```bash
export INFERENCE_ADMIN_KEY="your-secret-admin-key"
export INFERENCE_READ_KEY="your-secret-read-key"
```

**Generate secure keys:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Security Levels

| Level | Endpoints | Use Case |
|-------|-----------|----------|
| **Public** | `/health` | Health checks, monitoring |
| **Read** | `/v1/chat/completions`, `/models/info`, `/models`, `/models/active`, `/generate` | Inference, model info |
| **Admin** | `/models/reload`, `/models/register`, `/models/set_active`, `/gpu`, `/system`, `/jobs/*`, `/settings/*`, `/config`, `/logs/*` | Model management, system control |

### Usage

Include API key in the `X-API-Key` header:

```bash
# Health check - no auth required
curl http://192.168.x.x:8765/health

# Inference - read key required
curl -X POST http://192.168.x.x:8765/v1/chat/completions \
  -H "X-API-Key: $INFERENCE_READ_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Model reload - admin key required
curl -X POST http://192.168.x.x:8765/models/reload \
  -H "X-API-Key: $INFERENCE_ADMIN_KEY"
```

### Client Configuration

The `PredictionClient` automatically uses environment variables:

```python
from prediction_client import PredictionClient

# Uses INFERENCE_READ_KEY and INFERENCE_ADMIN_KEY from environment
client = PredictionClient()
client.chat(messages=[{"role": "user", "content": "Hello"}])
client.reload_model()  # Uses admin key automatically
```

### Security Warning

**Never expose this API to the public internet without authentication enabled.**

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

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | - | System health and GPU status |
| `/models` | GET | Read | List registered models |
| `/models/info` | GET | Read | Currently loaded model info |
| `/models/reload` | POST | Admin | Force reload deployed model |
| `/models/register` | POST | Admin | Register new model |
| `/models/set_active` | POST | Admin | Set active model |
| `/v1/chat/completions` | POST | Read | OpenAI-compatible chat API |
| `/gpu` | GET | Admin | Detailed GPU statistics |
| `/system` | GET | Admin | CPU/RAM/disk statistics |
| `/settings/power_profile` | GET/POST | Admin | GPU power management |
| `/jobs` | GET | Admin | List job queue |
| `/config` | GET | Admin | Server configuration |

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
