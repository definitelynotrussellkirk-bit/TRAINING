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
from datetime import datetime, timedelta
from pathlib import Path

# ===== Configuration =====
BASE_DIR = Path.home() / "llm"
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
    c.execute("SELECT id FROM models WHERE id = ?", ("Qwen3-0.6B",))
    if not c.fetchone():
        c.execute("""
            INSERT INTO models (id, path, source, created_at, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "Qwen3-0.6B",
            str(MODELS_DIR / "Qwen3-0.6B"),
            "4090",
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

@app.get("/models/info")
async def get_model_info():
    """Get currently loaded model information"""
    try:
        from inference_worker import get_worker
        worker = get_worker()

        if worker.model is None:
            return {
                "loaded": False,
                "model_id": None,
                "checkpoint_step": None,
                "loaded_at": None,
                "vram_usage_gb": 0
            }

        # Extract checkpoint step from model_id if available
        checkpoint_step = None
        if worker.loaded_model_id and "step" in worker.loaded_model_id:
            try:
                checkpoint_step = int(worker.loaded_model_id.split("step")[1].split("-")[0])
            except:
                pass

        # Get VRAM usage
        import torch
        vram_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0

        return {
            "loaded": True,
            "model_id": worker.loaded_model_id,
            "checkpoint_step": checkpoint_step,
            "loaded_from": str(worker.model_path / worker.loaded_model_id) if worker.loaded_model_id else None,
            "loaded_at": getattr(worker, 'model_loaded_at', None),
            "vram_usage_gb": vram_gb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.post("/models/reload")
async def reload_model():
    """Force reload of deployed model"""
    try:
        from inference_worker import get_worker
        import torch

        worker = get_worker()

        # Check if deployed model exists
        deployed_path = MODELS_DIR / "deployed"
        if not deployed_path.exists() or not (deployed_path / "config.json").exists():
            raise HTTPException(
                status_code=404,
                detail="No deployed model found. Deploy a checkpoint to models/deployed/ first."
            )

        # Unload current model
        if worker.model is not None:
            del worker.model
            del worker.tokenizer
            worker.model = None
            worker.tokenizer = None
            worker.loaded_model_id = None
            torch.cuda.empty_cache()

        # Load deployed model
        worker.load_model("deployed")

        # Record load time
        worker.model_loaded_at = datetime.now().isoformat()

        # Extract checkpoint step if available
        checkpoint_step = None
        try:
            # Try to read from checkpoint metadata
            if (deployed_path / "trainer_state.json").exists():
                import json
                with open(deployed_path / "trainer_state.json") as f:
                    state = json.load(f)
                    checkpoint_step = state.get("global_step")
        except:
            pass

        # Get VRAM usage
        vram_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0

        return {
            "status": "reloaded",
            "model_id": "deployed",
            "checkpoint_step": checkpoint_step,
            "loaded_from": str(deployed_path),
            "loaded_at": worker.model_loaded_at,
            "vram_usage_gb": vram_gb
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

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
    model: str = "Qwen3-0.6B"
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

    # Load model and run inference
    try:
        from inference_worker import get_worker
        worker = get_worker()
        worker.load_model(req.model)

        result = worker.generate(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=1.1
        )

        return {
            "id": job_id,
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["generated_text"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"]
            }
        }
    except Exception as e:
        # Return error response
        return {
            "id": job_id,
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                },
                "finish_reason": "error"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
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
