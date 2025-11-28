# Remote Inference Server

**Purpose:** HTTP API server for all inference, evaluation, and data generation operations.

**Configuration:** See `config/hosts.json` for server IP, port, and SSH settings.

**Server:** `<INFERENCE_HOST>:<INFERENCE_PORT>` (default: port 8765)
**API Base URL:** `http://<INFERENCE_HOST>:<INFERENCE_PORT>`

---

## Environment Setup

Set these environment variables to customize access (or use `config/hosts.json`):

```bash
# From config/hosts.json or set manually
export INFERENCE_HOST="your.inference.host"    # IP or hostname
export INFERENCE_PORT="8765"
export INFERENCE_SSH_USER="youruser"
export INFERENCE_MODELS_DIR="~/llm/models"
export TRAINING_BASE_DIR="/path/to/TRAINING"
```

---

## Architecture

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   4090 (Training)       │      │   3090 (Inference)      │
│   Trainer Host          │      │   <INFERENCE_HOST>      │
│                         │      │                         │
│  - Train models         │◄────►│  HTTP API (port 8765)   │
│  - Save checkpoints     │ HTTP │                         │
│  - Monitor training     │      │  ┌──────────────────┐   │
│  - Request eval jobs    │      │  │  FastAPI Server  │   │
│                         │      │  └────────┬─────────┘   │
└─────────────────────────┘      │           │             │
                                 │  ┌────────▼─────────┐   │
                                 │  │  GPU Worker      │   │
                                 │  │  (Qwen3-0.6B)    │   │
                                 │  └──────────────────┘   │
                                 │                         │
                                 │  - Inference            │
                                 │  - Evaluation           │
                                 │  - Data generation      │
                                 └─────────────────────────┘
```

**Communication:** HTTP API (no SSH required for normal operations)

---

## Quick Reference

### Health Check
```bash
curl "http://${INFERENCE_HOST:-localhost}:${INFERENCE_PORT:-8765}/health"
```

### GPU Stats
```bash
curl "http://${INFERENCE_HOST:-localhost}:${INFERENCE_PORT:-8765}/gpu" | jq .
```

### List Models
```bash
curl "http://${INFERENCE_HOST:-localhost}:${INFERENCE_PORT:-8765}/models" | jq .
```

### System Stats
```bash
curl "http://${INFERENCE_HOST:-localhost}:${INFERENCE_PORT:-8765}/system" | jq .
```

---

## API Endpoints

### Health & Info

**GET /health** - System health and GPU status
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/health"
```
Response:
```json
{
  "status": "ok",
  "gpu": {
    "available": true,
    "device_count": 1,
    "device_name": "NVIDIA GeForce RTX 3090",
    "memory_allocated_gb": 0.0,
    "memory_reserved_gb": 0.0
  },
  "active_model": "Qwen3-0.6B",
  "worker_busy": false
}
```

**GET /info** - Server configuration
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/info"
```

**GET /version** - Software versions
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/version"
```
Response:
```json
{
  "api_version": "1.0.0",
  "python_version": "3.12.3",
  "torch_version": "2.9.1+cu128",
  "transformers_version": "4.57.1",
  "cuda_available": true,
  "cuda_version": "12.8"
}
```

---

### Model Management

**GET /models** - List all registered models
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models" | jq .
```

**POST /models/register** - Register new checkpoint from trainer
```bash
# After training, copy checkpoint to inference server:
scp -r models/current_model "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/qwen3-step-5000"

# Register it:
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models/register" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "qwen3-step-5000",
    "source": "4090",
    "tags": "step5000,math"
  }'
```

**POST /models/set_active** - Switch active model
```bash
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models/set_active" \
  -H "Content-Type: application/json" \
  -d '{"id": "qwen3-step-5000"}'
```

**GET /models/active** - Get currently active model
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models/active"
```

---

### Inference

**POST /generate** - Generate text (queued)
```bash
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "mode": "normal"
  }'
```

**POST /v1/chat/completions** - OpenAI-compatible chat
```bash
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

---

### Evaluation Jobs

**POST /eval/jobs** - Queue evaluation job
```bash
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/eval/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-step-5000",
    "name": "math_eval_step_5000",
    "dataset_ref": "math_eval.jsonl",
    "metrics": ["accuracy"],
    "max_samples": 1000,
    "per_example_logging": true
  }'
```

**GET /eval/jobs/{job_id}** - Get eval results
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/eval/jobs/eval_20251122_053000" | jq .
```

Response:
```json
{
  "job_id": "eval_20251122_053000",
  "type": "eval",
  "model_id": "qwen3-step-5000",
  "status": "done",
  "result": {
    "accuracy": 0.87,
    "total_samples": 1000,
    "correct": 870,
    "avg_loss": 0.42
  }
}
```

---

### Data Generation Jobs

**POST /data_gen/jobs** - Generate training data
```bash
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/data_gen/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen3-0.6B",
    "strategy": "self_instruct",
    "config": {
      "num_examples": 10000,
      "domain": "math",
      "output_dir": "train_shards_out"
    }
  }'
```

**GET /data_gen/jobs/{job_id}** - Check generation status
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/data_gen/jobs/datagen_20251122_060000" | jq .
```

---

### Job Management

**GET /jobs** - List all jobs (with filters)
```bash
# All jobs
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs" | jq .

# Filter by type
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs?type=eval" | jq .

# Filter by status
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs?status=pending" | jq .

# Combined filters
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs?type=eval&status=done&limit=10" | jq .
```

**GET /jobs/stats** - Queue statistics
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs/stats" | jq .
```

Response:
```json
{
  "pending": 3,
  "running": 1,
  "done": 247,
  "failed": 2,
  "jobs_last_hour": 15
}
```

---

### GPU & System Telemetry

**GET /gpu** - GPU statistics
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/gpu" | jq .
```

Response:
```json
{
  "name": "NVIDIA GeForce RTX 3090",
  "driver_version": "580.95.05",
  "memory_total_mb": 24576,
  "memory_used_mb": 181,
  "utilization_gpu": 0,
  "utilization_mem": 1,
  "temperature_gpu": 41,
  "power_draw_w": 21.38,
  "power_limit_w": 350.0,
  "fan_speed_pct": 0
}
```

**GET /system** - System statistics
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/system" | jq .
```

Response:
```json
{
  "cpu_load": 0.0,
  "ram_used_gb": 2.5,
  "ram_total_gb": 33.4,
  "ram_percent": 7.3,
  "disk": [
    {
      "mount": "/",
      "used_gb": 111.1,
      "total_gb": 249.8,
      "percent": 46.9
    }
  ]
}
```

---

### Power Management

**GET /settings/power_profile** - Current power profile
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/settings/power_profile" | jq .
```

Response:
```json
{
  "current": "normal",
  "profiles": {
    "quiet": {
      "power_limit_w": 220,
      "max_concurrent_jobs": 1,
      "description": "Low power, quiet operation"
    },
    "normal": {
      "power_limit_w": 280,
      "max_concurrent_jobs": 1,
      "description": "Balanced performance"
    },
    "max": {
      "power_limit_w": 350,
      "max_concurrent_jobs": 2,
      "description": "Maximum performance"
    }
  }
}
```

**POST /settings/power_profile** - Set power profile
```bash
# Quiet mode (220W) - for overnight jobs
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/settings/power_profile?profile=quiet"

# Normal mode (280W) - default
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/settings/power_profile?profile=normal"

# Max mode (350W) - for heavy workloads
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/settings/power_profile?profile=max"
```

---

### Ops & Maintenance

**GET /logs/{component}** - View component logs
```bash
# API server logs
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/logs/api_server?lines=50"

# GPU worker logs
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/logs/gpu_worker?lines=100"

# Job execution logs
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/logs/jobs?lines=200"
```

**GET /config** - Current configuration
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/config" | jq .
```

---

## Common Workflows

### Workflow 1: Evaluate Latest Checkpoint

```bash
# 1. Copy checkpoint to inference server
scp -r models/current_model "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/current_checkpoint"

# 2. Register the checkpoint
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models/register" \
  -H "Content-Type: application/json" \
  -d '{"id": "current_checkpoint", "source": "4090"}'

# 3. Set as active
curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models/set_active" \
  -H "Content-Type: application/json" \
  -d '{"id": "current_checkpoint"}'

# 4. Queue eval job
EVAL_JOB=$(curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/eval/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "current_checkpoint",
    "name": "latest_eval",
    "dataset_ref": "math_eval.jsonl",
    "metrics": ["accuracy"],
    "max_samples": 1000
  }' | jq -r '.job_id')

# 5. Poll for results
while true; do
  STATUS=$(curl -s "http://${INFERENCE_HOST}:${INFERENCE_PORT}/eval/jobs/$EVAL_JOB" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] && break
  sleep 10
done

# 6. Get final results
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/eval/jobs/$EVAL_JOB" | jq .result
```

### Workflow 2: Generate Training Data

```bash
# 1. Queue data generation job
JOB_ID=$(curl -X POST "http://${INFERENCE_HOST}:${INFERENCE_PORT}/data_gen/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen3-0.6B",
    "strategy": "self_instruct",
    "config": {
      "num_examples": 50000,
      "domain": "math",
      "output_dir": "train_shards_out"
    }
  }' | jq -r '.job_id')

# 2. Wait for completion
while true; do
  STATUS=$(curl -s "http://${INFERENCE_HOST}:${INFERENCE_PORT}/data_gen/jobs/$JOB_ID" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "done" ] && break
  sleep 30
done

# 3. Get output shards
SHARDS=$(curl -s "http://${INFERENCE_HOST}:${INFERENCE_PORT}/data_gen/jobs/$JOB_ID" | jq -r '.shards[]')

# 4. Copy shards to training machine
for shard in $SHARDS; do
  scp "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/../train_shards_out/$shard" inbox/
done

# 5. Training starts automatically
```

### Workflow 3: Monitor GPU During Jobs

```bash
# Watch GPU stats in real-time
watch -n 5 "curl -s 'http://${INFERENCE_HOST}:${INFERENCE_PORT}/gpu' | jq ."

# Or create a simple monitor
while true; do
  clear
  echo "=== GPU Stats ==="
  curl -s "http://${INFERENCE_HOST}:${INFERENCE_PORT}/gpu" | jq '{
    temp: .temperature_gpu,
    power: .power_draw_w,
    vram: .memory_used_mb,
    util: .utilization_gpu
  }'
  echo ""
  echo "=== Job Queue ==="
  curl -s "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs/stats" | jq .
  sleep 5
done
```

---

## Server Management

### Directory Structure

```
~/llm/
├── main.py                      # FastAPI server
├── venv/                        # Python virtualenv
├── db.sqlite                    # Job queue database
├── models/
│   ├── Qwen3-0.6B/             # Base model (active)
│   ├── qwen3-step-5000/         # Checkpoint from trainer
│   └── qwen3-step-10000/
├── runs/
│   ├── eval/
│   │   ├── run_001/
│   │   │   ├── config.json
│   │   │   ├── metrics.json
│   │   │   └── samples.jsonl
│   └── data_gen/
│       ├── job_001/
│       │   └── shard_000.jsonl
├── train_shards_out/            # Ready for trainer
│   ├── shard_2025-11-22_001.jsonl
│   └── shard_2025-11-22_002.jsonl
├── datasets/                    # Eval datasets
│   ├── math_eval.jsonl
│   └── logic_eval.jsonl
├── logs/
│   ├── api_server.log
│   ├── gpu_worker.log
│   └── jobs.log
└── checkpoints/                 # Temp storage for transfers
```

### Start/Stop Server

**Manual start:**
```bash
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "cd ~/llm && source venv/bin/activate && nohup python3 main.py > logs/api_server.log 2>&1 &"
```

**Check if running:**
```bash
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/health"
# or
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "ps aux | grep 'main.py' | grep -v grep"
```

**Stop server:**
```bash
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "pkill -f 'python3 main.py'"
```

**View logs:**
```bash
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "tail -f ~/llm/logs/api_server.log"
```

### Systemd Service (Optional)

Service file location: `/tmp/llm-api.service` on inference server

**Install service:**
```bash
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}"
sudo cp /tmp/llm-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable llm-api
sudo systemctl start llm-api
```

**Service management:**
```bash
# Check status
sudo systemctl status llm-api

# View logs
sudo journalctl -u llm-api -f

# Restart
sudo systemctl restart llm-api

# Stop
sudo systemctl stop llm-api
```

---

## SSH Access (Fallback)

**Connection:**
```bash
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}"
# Use SSH key or credentials from ~/.ssh/config
```

**Direct commands:**
```bash
# Check GPU
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'nvidia-smi'

# View server status
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'ps aux | grep main.py'

# Tail logs
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'tail -f ~/llm/logs/api_server.log'
```

**File transfers:**
```bash
# Transfer model to inference server
scp -r models/current_model "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/my_checkpoint"

# Pull generated data from inference server
scp "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:~/llm/train_shards_out/*.jsonl" inbox/

# Rsync (faster for large transfers)
rsync -avz --progress models/current_model/ "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/my_checkpoint/"
```

---

## Troubleshooting

### API Not Responding

```bash
# Check if server is running
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "ps aux | grep 'main.py' | grep -v grep"

# Check logs
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "tail -50 ~/llm/logs/api_server.log"

# Restart server
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "pkill -f 'python3 main.py' && cd ~/llm && source venv/bin/activate && nohup python3 main.py > logs/api_server.log 2>&1 &"

# Test connection
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/health"
```

### Model Loading Fails

```bash
# Check model files exist
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "ls -lh ~/llm/models/Qwen3-0.6B/"

# Test model loading
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "cd ~/llm && source venv/bin/activate && python3 -c 'from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained(\"models/Qwen3-0.6B\"); print(\"✓ Model loaded\")'"
```

### GPU Out of Memory

```bash
# Check VRAM usage
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/gpu" | jq '.memory_used_mb'

# Stop all jobs
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "sqlite3 ~/llm/db.sqlite 'UPDATE jobs SET status=\"cancelled\" WHERE status=\"pending\"'"

# Restart server to clear VRAM
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "pkill -f 'python3 main.py'"
```

### Disk Space Issues

```bash
# Check disk usage
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/system" | jq '.disk'

# Clean old job outputs
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "find ~/llm/runs/ -type f -mtime +7 -delete"

# Clean old shards
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" "rm ~/llm/train_shards_out/*.jsonl"
```

---

## Python Client Example

```python
import os
import requests
import time

class LLMInferenceClient:
    def __init__(self, base_url=None):
        if base_url is None:
            host = os.environ.get("INFERENCE_HOST", "localhost")
            port = os.environ.get("INFERENCE_PORT", "8765")
            base_url = f"http://{host}:{port}"
        self.base_url = base_url

    def health(self):
        return requests.get(f"{self.base_url}/health").json()

    def gpu_stats(self):
        return requests.get(f"{self.base_url}/gpu").json()

    def list_models(self):
        return requests.get(f"{self.base_url}/models").json()

    def register_model(self, model_id, source="4090", tags=None):
        return requests.post(
            f"{self.base_url}/models/register",
            json={"id": model_id, "source": source, "tags": tags}
        ).json()

    def set_active_model(self, model_id):
        return requests.post(
            f"{self.base_url}/models/set_active",
            json={"id": model_id}
        ).json()

    def eval_job(self, model_id, name, dataset_ref, max_samples=1000):
        return requests.post(
            f"{self.base_url}/eval/jobs",
            json={
                "model_id": model_id,
                "name": name,
                "dataset_ref": dataset_ref,
                "max_samples": max_samples
            }
        ).json()

    def get_eval_job(self, job_id):
        return requests.get(f"{self.base_url}/eval/jobs/{job_id}").json()

    def wait_for_job(self, job_id, poll_interval=10):
        while True:
            result = self.get_eval_job(job_id)
            status = result.get("status")
            print(f"Job {job_id}: {status}")
            if status in ["done", "failed"]:
                return result
            time.sleep(poll_interval)

# Usage
client = LLMInferenceClient()

# Check health
print(client.health())

# Run evaluation
job = client.eval_job("Qwen3-0.6B", "test_eval", "math_eval.jsonl")
result = client.wait_for_job(job["job_id"])
print(f"Accuracy: {result['result']['accuracy']}")
```

---

## Quick Command Reference

```bash
# Health check
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/health"

# GPU stats
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/gpu" | jq .

# List models
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/models" | jq .

# Job queue stats
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/jobs/stats" | jq .

# System stats
curl "http://${INFERENCE_HOST}:${INFERENCE_PORT}/system" | jq .

# Check server logs
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'tail -50 ~/llm/logs/api_server.log'

# Restart server
ssh "${INFERENCE_SSH_USER}@${INFERENCE_HOST}" 'pkill -f main.py && cd ~/llm && source venv/bin/activate && nohup python3 main.py > logs/api_server.log 2>&1 &'

# Transfer checkpoint
scp -r models/current_model "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:${INFERENCE_MODELS_DIR}/checkpoint_name"

# Pull generated data
scp "${INFERENCE_SSH_USER}@${INFERENCE_HOST}:~/llm/train_shards_out/*.jsonl" inbox/
```
