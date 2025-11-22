# RTX 3090 Inference Server - Complete Setup

**Purpose:** Remote inference, evaluation, and data generation server. One HTTP API to rule them all.

**Server:** 192.168.x.x
**SSH:** `ssh user@xxx.xxx.88.149` (password: Bullshit1!)

---

## Architecture Overview

```
┌─────────────────────────┐      ┌─────────────────────────┐
│   4090 (Training)       │      │   3090 (Inference)      │
│   This Machine          │      │   192.168.x.x        │
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

**Division of Labor:**
- **4090:** Pure training (no inference, no eval, just gradients)
- **3090:** All inference, eval, and data generation (never touches training)
- **Communication:** HTTP API only (no SSH after initial setup)

---

## Directory Structure on 3090

```
/srv/llm/
├── models/                      # Model snapshots
│   ├── qwen3-0.6b-base/         # Base model
│   ├── qwen3-0.6b-step-005000/  # Checkpoint from 4090
│   ├── qwen3-0.6b-step-010000/
│   └── qwen3-0.6b-step-015000/
│
├── runs/                        # Job outputs
│   ├── eval/
│   │   ├── run_001/
│   │   │   ├── config.json      # Eval config
│   │   │   ├── metrics.json     # Accuracy, loss, etc.
│   │   │   └── samples.jsonl    # Per-example results
│   │   └── run_002/
│   └── data_gen/
│       ├── job_001/
│       │   ├── config.json
│       │   ├── shard_000.jsonl  # Generated training data
│       │   └── shard_001.jsonl
│       └── job_002/
│
├── train_shards_out/            # Ready for 4090 consumption
│   ├── shard_2025-11-22_001.jsonl
│   ├── shard_2025-11-22_002.jsonl
│   └── ...
│
├── datasets/                    # Fixed eval datasets
│   ├── math_eval.jsonl
│   ├── logic_eval.jsonl
│   └── ...
│
├── logs/                        # All logs
│   ├── api_server.log
│   ├── gpu_worker.log
│   └── jobs.log
│
├── db.sqlite                    # Job queue and metadata
│
└── api/                         # API server code
    ├── main.py                  # FastAPI app
    ├── models.py                # Model management
    ├── worker.py                # GPU worker process
    ├── jobs.py                  # Job queue
    └── requirements.txt
```

---

## Initial Setup (One-Time, Over SSH)

### Step 1: Connect to 3090

```bash
ssh user@xxx.xxx.88.149
# Password: Bullshit1!
```

### Step 2: Create Directory Structure

```bash
sudo mkdir -p /srv/llm/{models,runs/{eval,data_gen},train_shards_out,datasets,logs,api}
sudo chown -R user:user /srv/llm
cd /srv/llm
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ if needed
python3 --version

# Install pip and system tools
sudo apt install -y python3-pip python3-venv
sudo apt install -y htop nvtop curl jq

# Install psutil for system monitoring
sudo apt install -y python3-psutil

# Create virtual environment
python3 -m venv /srv/llm/venv
source /srv/llm/venv/bin/activate

# Install core packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install fastapi uvicorn[standard]
pip install aiosqlite sqlalchemy
pip install pydantic pydantic-settings
pip install psutil  # For system stats
```

### Step 3b: Configure Power Management (Sudo Access)

Allow the service user to control GPU power without password:

```bash
# Create sudoers file for nvidia-smi
sudo nano /etc/sudoers.d/llm-gpu-power
```

Add this line:
```
user ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi -pl *
```

Save and test:
```bash
# Test power limit (should not ask for password)
sudo nvidia-smi -pl 280

# Verify current power limit
nvidia-smi -q -d POWER | grep "Power Limit"
```

### Step 4: Download Base Model

```bash
cd /srv/llm/models

# Download Qwen3-0.6B from HuggingFace
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b-base

# Verify
ls -lh qwen3-0.6b-base/
# Should show model.safetensors (~1.5GB)
```

### Step 5: Create API Server

Create `/srv/llm/api/main.py`:

```python
#!/usr/bin/env python3
"""
RTX 3090 Inference API Server
Single HTTP endpoint to control all inference/eval/data-gen
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# ===== Configuration =====
BASE_DIR = Path("/srv/llm")
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "runs"
DATASETS_DIR = BASE_DIR / "datasets"
SHARDS_OUT = BASE_DIR / "train_shards_out"
DB_PATH = BASE_DIR / "db.sqlite"

app = FastAPI(title="RTX 3090 Inference API", version="1.0.0")

# ===== Database Setup =====
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Models table
    c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            source TEXT,
            created_at TEXT,
            tags TEXT,
            is_active INTEGER DEFAULT 0
        )
    """)

    # Jobs table
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            model_id TEXT,
            config TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            result TEXT,
            error TEXT
        )
    """)

    conn.commit()
    conn.close()

# Initialize DB on startup
@app.on_event("startup")
async def startup():
    init_db()
    print("✓ Database initialized")

    # Register base model if not exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM models WHERE id = ?", ("qwen3-0.6b-base",))
    if not c.fetchone():
        c.execute("""
            INSERT INTO models (id, path, source, created_at, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "qwen3-0.6b-base",
            str(MODELS_DIR / "qwen3-0.6b-base"),
            "HuggingFace",
            datetime.now().isoformat(),
            1  # Active by default
        ))
        conn.commit()
        print("✓ Base model registered")
    conn.close()

# ===== API Models =====
class HealthResponse(BaseModel):
    status: str
    gpu: Dict[str, Any]
    active_model: Optional[str]
    worker_busy: bool

class ModelRegister(BaseModel):
    id: str
    source: str
    tags: Optional[str] = None

class ModelSetActive(BaseModel):
    id: str

class GenerateRequest(BaseModel):
    model_id: Optional[str] = None
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    mode: str = "normal"  # or "think"

class EvalJobRequest(BaseModel):
    model_id: str
    name: str
    dataset_ref: str
    metrics: List[str] = ["accuracy"]
    max_samples: int = 1000
    per_example_logging: bool = True

class DataGenJobRequest(BaseModel):
    model_id: str
    strategy: str  # "self_instruct", "fix_eval_failures", "custom"
    config: Dict[str, Any]

# ===== Health & Info =====
@app.get("/health", response_model=HealthResponse)
async def health():
    """System health and status"""
    try:
        import torch
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0,
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2) if torch.cuda.is_available() else 0,
        }
    except Exception as e:
        gpu_info = {"error": str(e)}

    # Get active model
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM models WHERE is_active = 1")
    active = c.fetchone()
    conn.close()

    return {
        "status": "ok",
        "gpu": gpu_info,
        "active_model": active[0] if active else None,
        "worker_busy": False  # TODO: Check worker status
    }

@app.get("/info")
async def info():
    """System information"""
    return {
        "base_dir": str(BASE_DIR),
        "models_dir": str(MODELS_DIR),
        "runs_dir": str(RUNS_DIR),
        "datasets_dir": str(DATASETS_DIR),
        "db_path": str(DB_PATH),
        "version": "1.0.0"
    }

# ===== Model Management =====
@app.get("/models")
async def list_models():
    """List all registered models"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, path, source, created_at, tags, is_active
        FROM models
        ORDER BY created_at DESC
    """)
    models = []
    for row in c.fetchall():
        models.append({
            "id": row[0],
            "path": row[1],
            "source": row[2],
            "created_at": row[3],
            "tags": row[4],
            "is_active": bool(row[5])
        })
    conn.close()
    return {"models": models}

@app.post("/models/register")
async def register_model(req: ModelRegister):
    """Register a new model snapshot"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if already exists
    c.execute("SELECT id FROM models WHERE id = ?", (req.id,))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail=f"Model {req.id} already registered")

    # Insert
    c.execute("""
        INSERT INTO models (id, path, source, created_at, tags)
        VALUES (?, ?, ?, ?, ?)
    """, (
        req.id,
        str(MODELS_DIR / req.id),
        req.source,
        datetime.now().isoformat(),
        req.tags
    ))
    conn.commit()
    conn.close()

    return {"status": "registered", "id": req.id}

@app.post("/models/set_active")
async def set_active_model(req: ModelSetActive):
    """Set active model for inference"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check model exists
    c.execute("SELECT id FROM models WHERE id = ?", (req.id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Model {req.id} not found")

    # Deactivate all
    c.execute("UPDATE models SET is_active = 0")

    # Activate requested
    c.execute("UPDATE models SET is_active = 1 WHERE id = ?", (req.id,))
    conn.commit()
    conn.close()

    # TODO: Signal GPU worker to reload model

    return {"status": "activated", "id": req.id}

@app.get("/models/active")
async def get_active_model():
    """Get currently active model"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, path FROM models WHERE is_active = 1")
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No active model")

    return {"id": row[0], "path": row[1]}

# ===== Inference =====
@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate text (synchronous for now, async later)"""
    # TODO: Implement actual generation via GPU worker
    # For now, just queue the job

    job_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO jobs (id, type, model_id, config, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        job_id,
        "inference",
        req.model_id or "active",
        json.dumps(req.dict()),
        "pending",
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Job queued. Poll /jobs/{job_id} for status"
    }

# ===== Eval Jobs =====
@app.post("/eval/jobs")
async def create_eval_job(req: EvalJobRequest):
    """Create evaluation job"""
    job_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO jobs (id, type, model_id, config, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        job_id,
        "eval",
        req.model_id,
        json.dumps(req.dict()),
        "pending",
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return {"job_id": job_id, "status": "pending"}

@app.get("/eval/jobs/{job_id}")
async def get_eval_job(job_id: str):
    """Get eval job status and results"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT type, model_id, config, status, created_at, started_at, completed_at, result, error
        FROM jobs WHERE id = ?
    """, (job_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job_id,
        "type": row[0],
        "model_id": row[1],
        "config": json.loads(row[2]) if row[2] else {},
        "status": row[3],
        "created_at": row[4],
        "started_at": row[5],
        "completed_at": row[6],
        "result": json.loads(row[7]) if row[7] else None,
        "error": row[8]
    }

# ===== Data Generation Jobs =====
@app.post("/data_gen/jobs")
async def create_data_gen_job(req: DataGenJobRequest):
    """Create data generation job"""
    job_id = f"datagen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO jobs (id, type, model_id, config, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        job_id,
        "data_gen",
        req.model_id,
        json.dumps(req.dict()),
        "pending",
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return {"job_id": job_id, "status": "pending"}

@app.get("/data_gen/jobs/{job_id}")
async def get_data_gen_job(job_id: str):
    """Get data generation job status"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT type, model_id, config, status, result
        FROM jobs WHERE id = ?
    """, (job_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = json.loads(row[4]) if row[4] else {}

    return {
        "job_id": job_id,
        "status": row[3],
        "shards": result.get("shards", [])
    }

# ===== Job Management =====
@app.get("/jobs")
async def list_jobs(type: Optional[str] = None, status: Optional[str] = None, limit: int = 100):
    """List jobs with optional filters"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    query = "SELECT id, type, model_id, status, created_at FROM jobs WHERE 1=1"
    params = []

    if type:
        query += " AND type = ?"
        params.append(type)

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    c.execute(query, params)
    jobs = []
    for row in c.fetchall():
        jobs.append({
            "id": row[0],
            "type": row[1],
            "model_id": row[2],
            "status": row[3],
            "created_at": row[4]
        })
    conn.close()

    return {"jobs": jobs}

# ===== GPU & System Telemetry =====
@app.get("/gpu")
async def gpu_stats():
    """Get GPU statistics"""
    try:
        import subprocess
        # Query nvidia-smi for detailed stats
        result = subprocess.run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,fan.speed",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(result.stderr)

        parts = result.stdout.strip().split(", ")
        return {
            "name": parts[0],
            "driver_version": parts[1],
            "memory_total_mb": int(parts[2]),
            "memory_used_mb": int(parts[3]),
            "utilization_gpu": int(parts[4]),
            "utilization_mem": int(parts[5]),
            "temperature_gpu": int(parts[6]),
            "power_draw_w": float(parts[7]),
            "power_limit_w": float(parts[8]),
            "fan_speed_pct": int(parts[9])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPU query failed: {str(e)}")

@app.get("/system")
async def system_stats():
    """Get system statistics"""
    import psutil

    # CPU
    cpu_load = psutil.cpu_percent(interval=1) / 100.0

    # RAM
    mem = psutil.virtual_memory()

    # Disk
    disks = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disks.append({
                "mount": partition.mountpoint,
                "used_gb": round(usage.used / 1e9, 1),
                "total_gb": round(usage.total / 1e9, 1),
                "percent": usage.percent
            })
        except:
            pass

    return {
        "cpu_load": round(cpu_load, 2),
        "ram_used_gb": round(mem.used / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "ram_percent": mem.percent,
        "disk": disks
    }

@app.get("/jobs/stats")
async def job_stats():
    """Get job queue statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Count by status
    c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
    status_counts = dict(c.fetchall())

    # Recent throughput (jobs completed in last hour)
    from datetime import datetime, timedelta
    one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    c.execute("SELECT COUNT(*) FROM jobs WHERE status = 'done' AND completed_at > ?", (one_hour_ago,))
    recent_completed = c.fetchone()[0]

    conn.close()

    return {
        "pending": status_counts.get("pending", 0),
        "running": status_counts.get("running", 0),
        "done": status_counts.get("done", 0),
        "failed": status_counts.get("failed", 0),
        "jobs_last_hour": recent_completed
    }

# ===== Power Management =====
POWER_PROFILES = {
    "quiet": {"power_limit_w": 220, "max_concurrent_jobs": 1, "description": "Low power, quiet operation"},
    "normal": {"power_limit_w": 280, "max_concurrent_jobs": 1, "description": "Balanced performance"},
    "max": {"power_limit_w": 350, "max_concurrent_jobs": 2, "description": "Maximum performance"}
}

@app.get("/settings/power_profile")
async def get_power_profile():
    """Get current power profile"""
    # TODO: Read from config/db
    return {
        "current": "normal",
        "profiles": POWER_PROFILES
    }

@app.post("/settings/power_profile")
async def set_power_profile(profile: str):
    """Set power profile"""
    if profile not in POWER_PROFILES:
        raise HTTPException(status_code=400, detail=f"Invalid profile. Choose from: {list(POWER_PROFILES.keys())}")

    config = POWER_PROFILES[profile]

    try:
        import subprocess
        # Set power limit via nvidia-smi
        result = subprocess.run([
            "sudo", "nvidia-smi", "-pl", str(config["power_limit_w"])
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(result.stderr)

        # TODO: Save to config/db for persistence

        return {
            "status": "success",
            "profile": profile,
            "power_limit_w": config["power_limit_w"],
            "max_concurrent_jobs": config["max_concurrent_jobs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set power profile: {str(e)}")

# ===== OpenAI-Compatible Endpoints =====
class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-0.6b"
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    # Convert messages to prompt
    prompt = ""
    for msg in req.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant:"

    # Queue inference job
    job_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO jobs (id, type, model_id, config, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        job_id,
        "inference",
        req.model,
        json.dumps({"prompt": prompt, "max_tokens": req.max_tokens, "temperature": req.temperature, "top_p": req.top_p}),
        "pending",
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    # TODO: Actually run inference (for now, return mock)
    return {
        "id": job_id,
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "[Job queued - implement GPU worker to get actual response]"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 10,
            "total_tokens": len(prompt.split()) + 10
        }
    }

# ===== Ops & Maintenance =====
@app.get("/logs/{component}")
async def get_logs(component: str, lines: int = 100):
    """Get recent logs for a component"""
    valid_components = ["api_server", "gpu_worker", "jobs"]
    if component not in valid_components:
        raise HTTPException(status_code=400, detail=f"Invalid component. Choose from: {valid_components}")

    log_file = BASE_DIR / "logs" / f"{component}.log"
    if not log_file.exists():
        return {"component": component, "lines": []}

    try:
        import subprocess
        result = subprocess.run(["tail", "-n", str(lines), str(log_file)], capture_output=True, text=True)
        return {
            "component": component,
            "lines": result.stdout.split("\n")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/version")
async def get_version():
    """Get system version info"""
    import sys
    import torch
    import transformers

    return {
        "api_version": "1.0.0",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "base_dir": str(BASE_DIR),
        "models_dir": str(MODELS_DIR),
        "runs_dir": str(RUNS_DIR),
        "datasets_dir": str(DATASETS_DIR),
        "db_path": str(DB_PATH),
        "port": 8765
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
```

Save this file, then make it executable:

```bash
chmod +x /srv/llm/api/main.py
```

### Step 6: Create Systemd Service

Create `/etc/systemd/system/llm-api.service`:

```ini
[Unit]
Description=RTX 3090 LLM Inference API
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/srv/llm/api
Environment="PATH=/srv/llm/venv/bin"
ExecStart=/srv/llm/venv/bin/python /srv/llm/api/main.py
Restart=always
RestartSec=10
StandardOutput=append:/srv/llm/logs/api_server.log
StandardError=append:/srv/llm/logs/api_server.log

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable llm-api
sudo systemctl start llm-api
sudo systemctl status llm-api
```

### Step 7: Test API

From the 3090:

```bash
# Health check
curl http://localhost:8765/health | jq .

# List models
curl http://localhost:8765/models | jq .

# Get active model
curl http://localhost:8765/models/active | jq .
```

From the 4090 (training machine):

```bash
# Health check
curl http://192.168.x.x:8765/health | jq .

# List models
curl http://192.168.x.x:8765/models | jq .
```

---

## API Reference

### Health & Info

**GET /health**
```json
{
  "status": "ok",
  "gpu": {
    "available": true,
    "device_name": "NVIDIA GeForce RTX 3090",
    "memory_allocated_gb": 2.1,
    "memory_reserved_gb": 3.5
  },
  "active_model": "qwen3-0.6b-step-005000",
  "worker_busy": false
}
```

**GET /info**
```json
{
  "base_dir": "/srv/llm",
  "models_dir": "/srv/llm/models",
  "version": "1.0.0"
}
```

### Model Management

**GET /models**
```json
{
  "models": [
    {
      "id": "qwen3-0.6b-step-005000",
      "path": "/srv/llm/models/qwen3-0.6b-step-005000",
      "source": "checkpoint from 4090",
      "created_at": "2025-11-22T10:30:00",
      "is_active": true
    }
  ]
}
```

**POST /models/register**
```bash
curl -X POST http://192.168.x.x:8765/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "qwen3-0.6b-step-010000",
    "source": "checkpoint from 4090 at step 10000"
  }'
```

**POST /models/set_active**
```bash
curl -X POST http://192.168.x.x:8765/models/set_active \
  -H "Content-Type: application/json" \
  -d '{"id": "qwen3-0.6b-step-010000"}'
```

### Inference

**POST /generate**
```bash
curl -X POST http://192.168.x.x:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Evaluation

**POST /eval/jobs**
```bash
curl -X POST http://192.168.x.x:8765/eval/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-0.6b-step-010000",
    "name": "math_eval_step_10k",
    "dataset_ref": "/srv/llm/datasets/math_eval.jsonl",
    "max_samples": 1000
  }'
```

**GET /eval/jobs/{job_id}**
```bash
curl http://192.168.x.x:8765/eval/jobs/eval_20251122_103000 | jq .
```

### Data Generation

**POST /data_gen/jobs**
```bash
curl -X POST http://192.168.x.x:8765/data_gen/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-0.6b-step-010000",
    "strategy": "self_instruct",
    "config": {
      "target_domain": "math",
      "num_samples": 1000
    }
  }'
```

### GPU & System Monitoring

**GET /gpu**
```bash
curl http://192.168.x.x:8765/gpu | jq .
```

Response:
```json
{
  "name": "NVIDIA GeForce RTX 3090",
  "driver_version": "535.129.03",
  "memory_total_mb": 24576,
  "memory_used_mb": 2048,
  "utilization_gpu": 15,
  "utilization_mem": 8,
  "temperature_gpu": 42,
  "power_draw_w": 85.3,
  "power_limit_w": 280.0,
  "fan_speed_pct": 30
}
```

**GET /system**
```bash
curl http://192.168.x.x:8765/system | jq .
```

Response:
```json
{
  "cpu_load": 0.12,
  "ram_used_gb": 8.3,
  "ram_total_gb": 32.0,
  "ram_percent": 25.9,
  "disk": [
    {"mount": "/", "used_gb": 120.5, "total_gb": 500.0, "percent": 24.1},
    {"mount": "/srv", "used_gb": 45.2, "total_gb": 1000.0, "percent": 4.5}
  ]
}
```

**GET /jobs/stats**
```bash
curl http://192.168.x.x:8765/jobs/stats | jq .
```

Response:
```json
{
  "pending": 3,
  "running": 1,
  "done": 127,
  "failed": 2,
  "jobs_last_hour": 12
}
```

### Power Management

**Get current power profile:**
```bash
curl http://192.168.x.x:8765/settings/power_profile | jq .
```

**Set to quiet mode (low power, low noise):**
```bash
curl -X POST http://192.168.x.x:8765/settings/power_profile?profile=quiet
```

**Set to max performance:**
```bash
curl -X POST http://192.168.x.x:8765/settings/power_profile?profile=max
```

Profiles:
- **quiet:** 220W, single job, minimal fan noise
- **normal:** 280W, single job, balanced
- **max:** 350W, up to 2 concurrent jobs, full performance

### OpenAI-Compatible Inference

**Chat completions:**
```bash
curl -X POST http://192.168.x.x:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }' | jq .
```

This endpoint is compatible with OpenAI client libraries:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.x.x:8765/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Ops & Maintenance

**View logs:**
```bash
# API server logs
curl http://192.168.x.x:8765/logs/api_server?lines=50 | jq .

# GPU worker logs
curl http://192.168.x.x:8765/logs/gpu_worker?lines=50 | jq .
```

**Check version:**
```bash
curl http://192.168.x.x:8765/version | jq .
```

**View config:**
```bash
curl http://192.168.x.x:8765/config | jq .
```

---

## Usage Patterns

### From 4090 (Training Machine)

**After training saves a new checkpoint:**

```bash
# 1. Register new model
curl -X POST http://192.168.x.x:8765/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "qwen3-0.6b-step-015000",
    "source": "checkpoint from 4090 at step 15000"
  }'

# 2. Set as active
curl -X POST http://192.168.x.x:8765/models/set_active \
  -H "Content-Type: application/json" \
  -d '{"id": "qwen3-0.6b-step-015000"}'

# 3. Run eval
curl -X POST http://192.168.x.x:8765/eval/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-0.6b-step-015000",
    "name": "math_eval_step_15k",
    "dataset_ref": "/srv/llm/datasets/math_eval.jsonl",
    "max_samples": 1000
  }'

# 4. Check eval status (poll until done)
curl http://192.168.x.x:8765/eval/jobs/eval_20251122_153000 | jq .

# 5. Generate new training data based on eval
curl -X POST http://192.168.x.x:8765/data_gen/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-0.6b-step-015000",
    "strategy": "fix_eval_failures",
    "config": {
      "source_eval_run_id": "eval_20251122_153000",
      "num_samples": 1000
    }
  }'
```

### Daily Monitoring Dashboard

Create a simple script on the 4090 or your laptop:

```bash
#!/bin/bash
# 3090_status.sh - Quick status check

echo "=== RTX 3090 Status ==="
echo ""

echo "GPU Stats:"
curl -s http://192.168.x.x:8765/gpu | jq '{
  temp: .temperature_gpu,
  power: .power_draw_w,
  mem_used: .memory_used_mb,
  utilization: .utilization_gpu
}'

echo ""
echo "Job Queue:"
curl -s http://192.168.x.x:8765/jobs/stats | jq .

echo ""
echo "Active Model:"
curl -s http://192.168.x.x:8765/models/active | jq .id
```

Run with: `./3090_status.sh`

---

## Next Steps: GPU Worker

The API server above handles job queuing but doesn't actually run inference yet. Next we need:

1. **GPU Worker Process** (`/srv/llm/api/worker.py`)
   - Polls job queue
   - Loads model on GPU
   - Executes inference/eval/data-gen jobs
   - Updates job status

2. **Systemd Service** for GPU worker
   - Runs alongside API server
   - Auto-restarts on crash

3. **Shared Model Loader**
   - Keeps one model in VRAM
   - Reloads when active model changes

Ready to implement the GPU worker?
