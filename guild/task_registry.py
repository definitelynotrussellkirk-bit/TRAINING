#!/usr/bin/env python3
"""
Task Registry - Available tasks for the Task Master

Defines tasks that can be scheduled when GPU resources are available.
The Weaver/Task Master checks GPU utilization and runs tasks opportunistically.

Task Types:
- INFERENCE: Requires 3090 (inference server)
- GENERATION: Requires skill APIs (CPU-bound)
- TRAINING: Requires 4090 (already managed by training daemon)

Usage:
    from guild.task_registry import get_available_tasks, get_task, run_task

    # Get tasks that can run on 3090
    tasks = get_available_tasks(gpu="3090")

    # Run a specific task
    result = run_task("sparring_binary")
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/path/to/training")


@dataclass
class Task:
    """A schedulable task"""
    id: str                          # Unique identifier
    name: str                        # Human-readable name
    description: str                 # What it does
    gpu: str                         # Which GPU needed: "3090", "4090", "none"
    min_gpu_free: float = 0.4        # Min GPU free (0-1) to run
    estimated_duration: int = 300    # Estimated seconds
    cooldown: int = 3600             # Min seconds between runs
    priority: int = 5                # 1-10, higher = more important
    command: List[str] = field(default_factory=list)  # Command to run
    enabled: bool = True             # Can be disabled
    tags: List[str] = field(default_factory=list)     # For filtering


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

TASKS: Dict[str, Task] = {
    # =========================================================================
    # EVAL QUEUE PROCESSOR - HIGHEST PRIORITY (require 3090 inference)
    # Evals are queued when checkpoints are saved - ONLY RUN ONCE PER CHECKPOINT
    # This task processes the pending eval queue, not re-runs!
    # =========================================================================
    "process_eval_queue": Task(
        id="process_eval_queue",
        name="Process Eval Queue",
        description="Process pending evals (queued from checkpoint saves) - HIGHEST PRIORITY",
        gpu="3090",
        min_gpu_free=0.3,
        estimated_duration=120,      # ~2 minutes per checkpoint eval
        cooldown=60,                 # Check queue every minute when idle
        priority=10,                 # HIGHEST PRIORITY - evals inform curriculum!
        command=[
            "python3", str(BASE_DIR / "core" / "eval_runner.py"),
            "--once",  # Process all PENDING evals, then exit
        ],
        tags=["eval", "queue", "curriculum", "critical"],
    ),

    # =========================================================================
    # SPARRING TASKS (require 3090 inference)
    # =========================================================================
    # SPARRING: Always HIGH priority queue - data becomes stale when checkpoint advances!
    "sparring_binary": Task(
        id="sparring_binary",
        name="Sparring: Binary",
        description="DIO spars with Binary trainer, generates correction training data (HIGH PRIORITY - stale when checkpoint advances)",
        gpu="3090",
        min_gpu_free=0.4,
        estimated_duration=600,      # ~10 minutes for 100 problems
        cooldown=1800,               # 30 min cooldown
        priority=9,                  # HIGH - must train before next checkpoint!
        command=[
            "python3", str(BASE_DIR / "guild" / "sparring.py"),
            "--skill", "binary",
            "--count", "100",
            "--priority", "high",    # Explicit high priority queue
        ],
        tags=["sparring", "binary", "self-correction", "time-sensitive"],
    ),

    "sparring_binary_large": Task(
        id="sparring_binary_large",
        name="Sparring: Binary (Large)",
        description="Extended binary sparring - 500 problems (HIGH PRIORITY - stale when checkpoint advances)",
        gpu="3090",
        min_gpu_free=0.5,
        estimated_duration=2400,     # ~40 minutes
        cooldown=7200,               # 2 hour cooldown
        priority=8,                  # HIGH - must train before next checkpoint!
        command=[
            "python3", str(BASE_DIR / "guild" / "sparring.py"),
            "--skill", "binary",
            "--count", "500",
            "--priority", "high",    # Explicit high priority queue
        ],
        tags=["sparring", "binary", "self-correction", "large", "time-sensitive"],
    ),

    "sparring_sy": Task(
        id="sparring_sy",
        name="Sparring: SYLLO",
        description="DIO spars with SYLLO trainer (HIGH PRIORITY - stale when checkpoint advances)",
        gpu="3090",
        min_gpu_free=0.4,
        estimated_duration=900,      # ~15 minutes
        cooldown=1800,
        priority=9,                  # HIGH - must train before next checkpoint!
        command=[
            "python3", str(BASE_DIR / "guild" / "sparring.py"),
            "--skill", "sy",
            "--count", "100",
            "--priority", "high",    # Explicit high priority queue
        ],
        enabled=False,  # Disabled until SYLLO skill is ready
        tags=["sparring", "syllo", "self-correction", "time-sensitive"],
    ),

    # =========================================================================
    # DATA GENERATION TASKS (CPU-bound, use skill APIs)
    # =========================================================================
    "generate_binary": Task(
        id="generate_binary",
        name="Generate: Binary Data",
        description="Generate binary training data via skill API",
        gpu="none",
        estimated_duration=60,
        cooldown=300,
        priority=6,
        command=[
            "python3", str(BASE_DIR / "data_manager" / "manager.py"),
            "generate", "--force",
        ],
        tags=["generation", "binary", "data"],
    ),

    # =========================================================================
    # MAINTENANCE TASKS
    # =========================================================================
    "checkpoint_cleanup": Task(
        id="checkpoint_cleanup",
        name="Checkpoint Cleanup",
        description="Clean up old checkpoints based on retention policy",
        gpu="none",
        estimated_duration=30,
        cooldown=3600,
        priority=3,
        command=[
            "python3", str(BASE_DIR / "management" / "checkpoint_retention.py"),
            "--apply",
        ],
        tags=["maintenance", "cleanup"],
    ),

    "health_check": Task(
        id="health_check",
        name="Health Check",
        description="Run comprehensive system health check",
        gpu="none",
        estimated_duration=15,
        cooldown=1800,
        priority=4,
        command=[
            "python3", str(BASE_DIR / "safety" / "comprehensive_health_check.py"),
        ],
        tags=["maintenance", "health"],
    ),

    # =========================================================================
    # VAULT & SYNC TASKS
    # =========================================================================
    "vault_scan": Task(
        id="vault_scan",
        name="Vault: Scan Assets",
        description="Scan and register new assets in VaultKeeper",
        gpu="none",
        estimated_duration=30,
        cooldown=3600,
        priority=3,
        command=[
            "curl", "-X", "POST", "http://localhost:8767/api/scan",
            "-H", "Content-Type: application/json",
            "-d", '{"path": "/path/to/training/models/current_model"}',
        ],
        tags=["vault", "maintenance", "sync"],
    ),

    # =========================================================================
    # INFERENCE TASKS (3090)
    # =========================================================================
    "model_warmup": Task(
        id="model_warmup",
        name="Model Warmup",
        description="Warm up inference model with test prompts",
        gpu="3090",
        min_gpu_free=0.3,
        estimated_duration=30,
        cooldown=7200,               # 2 hour cooldown
        priority=2,
        command=[
            "curl", "-X", "POST", "http://192.168.x.x:8765/generate",
            "-H", "Content-Type: application/json",
            "-H", "X-API-Key: admin123",
            "-d", '{"prompt": "Compute: todecimal(①①)", "max_tokens": 50}',
        ],
        tags=["inference", "warmup", "3090"],
    ),

    # =========================================================================
    # QUALITY & VALIDATION TASKS
    # =========================================================================
    "validate_queue": Task(
        id="validate_queue",
        name="Validate Queue Files",
        description="Run validation on pending queue files",
        gpu="none",
        estimated_duration=60,
        cooldown=1800,
        priority=4,
        command=[
            "python3", str(BASE_DIR / "guild" / "sparring_validator.py"),
            "--check-all",
        ],
        tags=["validation", "quality", "queue"],
    ),

    "lineage_report": Task(
        id="lineage_report",
        name="Data Lineage Report",
        description="Generate data lineage report (generator/validator stats)",
        gpu="none",
        estimated_duration=15,
        cooldown=3600,
        priority=2,
        command=[
            "curl", "-s", "http://localhost:8767/api/lineage",
        ],
        tags=["lineage", "reporting", "data"],
    ),
}


# =============================================================================
# TASK STATE TRACKING
# =============================================================================

class TaskState:
    """Tracks task execution state"""

    def __init__(self, state_file: Path = None):
        self.state_file = state_file or BASE_DIR / "status" / "task_state.json"
        self.state = self._load()

    def _load(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        return {"tasks": {}, "last_updated": None}

    def _save(self):
        self.state["last_updated"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_last_run(self, task_id: str) -> Optional[datetime]:
        """Get when a task was last run"""
        task_state = self.state.get("tasks", {}).get(task_id, {})
        last_run = task_state.get("last_run")
        if last_run:
            return datetime.fromisoformat(last_run)
        return None

    def record_run(self, task_id: str, success: bool, duration: float = 0):
        """Record a task execution"""
        if "tasks" not in self.state:
            self.state["tasks"] = {}

        if task_id not in self.state["tasks"]:
            self.state["tasks"][task_id] = {
                "run_count": 0,
                "success_count": 0,
                "total_duration": 0,
            }

        task_state = self.state["tasks"][task_id]
        task_state["last_run"] = datetime.now().isoformat()
        task_state["last_success"] = success
        task_state["run_count"] = task_state.get("run_count", 0) + 1
        if success:
            task_state["success_count"] = task_state.get("success_count", 0) + 1
        task_state["total_duration"] = task_state.get("total_duration", 0) + duration

        self._save()

    def can_run(self, task: Task) -> tuple[bool, str]:
        """Check if a task can run (cooldown check)"""
        last_run = self.get_last_run(task.id)
        if last_run:
            elapsed = (datetime.now() - last_run).total_seconds()
            if elapsed < task.cooldown:
                remaining = task.cooldown - elapsed
                return False, f"Cooldown: {remaining:.0f}s remaining"
        return True, "Ready"


# Singleton state tracker
_task_state: Optional[TaskState] = None

def get_task_state() -> TaskState:
    global _task_state
    if _task_state is None:
        _task_state = TaskState()
    return _task_state


# =============================================================================
# PUBLIC API
# =============================================================================

def get_task(task_id: str) -> Optional[Task]:
    """Get a task by ID"""
    return TASKS.get(task_id)


def get_all_tasks() -> List[Task]:
    """Get all registered tasks"""
    return list(TASKS.values())


def get_available_tasks(
    gpu: str = None,
    tags: List[str] = None,
    include_disabled: bool = False,
    check_cooldown: bool = True,
) -> List[Task]:
    """
    Get tasks that are available to run.

    Args:
        gpu: Filter by GPU requirement ("3090", "4090", "none")
        tags: Filter by tags (any match)
        include_disabled: Include disabled tasks
        check_cooldown: Check if cooldown has elapsed

    Returns:
        List of available tasks, sorted by priority (high first)
    """
    state = get_task_state()
    available = []

    for task in TASKS.values():
        # Skip disabled unless requested
        if not task.enabled and not include_disabled:
            continue

        # Filter by GPU
        if gpu and task.gpu != gpu:
            continue

        # Filter by tags
        if tags and not any(t in task.tags for t in tags):
            continue

        # Check cooldown
        if check_cooldown:
            can_run, reason = state.can_run(task)
            if not can_run:
                continue

        available.append(task)

    # Sort by priority (highest first)
    return sorted(available, key=lambda t: -t.priority)


def run_task(task_id: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Execute a task.

    Args:
        task_id: Task to run
        dry_run: Just log, don't actually run

    Returns:
        Result dict with success, output, duration
    """
    task = get_task(task_id)
    if not task:
        return {"success": False, "error": f"Unknown task: {task_id}"}

    if not task.enabled:
        return {"success": False, "error": f"Task disabled: {task_id}"}

    state = get_task_state()
    can_run, reason = state.can_run(task)
    if not can_run:
        return {"success": False, "error": reason}

    logger.info(f"🎯 Running task: {task.name}")
    logger.info(f"   Command: {' '.join(task.command)}")

    if dry_run:
        logger.info("   (dry run - not executing)")
        return {"success": True, "dry_run": True}

    start_time = datetime.now()

    try:
        result = subprocess.run(
            task.command,
            capture_output=True,
            text=True,
            timeout=task.estimated_duration * 3,  # 3x estimate as timeout
            cwd=str(BASE_DIR),
        )

        duration = (datetime.now() - start_time).total_seconds()
        success = result.returncode == 0

        state.record_run(task_id, success, duration)

        return {
            "success": success,
            "task_id": task_id,
            "duration": duration,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        state.record_run(task_id, False, duration)
        return {"success": False, "error": "Timeout", "duration": duration}

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        state.record_run(task_id, False, duration)
        return {"success": False, "error": str(e), "duration": duration}


def get_status() -> Dict[str, Any]:
    """Get task registry status for dashboard"""
    state = get_task_state()

    tasks_status = []
    for task in TASKS.values():
        can_run, reason = state.can_run(task)
        task_state = state.state.get("tasks", {}).get(task.id, {})

        tasks_status.append({
            "id": task.id,
            "name": task.name,
            "gpu": task.gpu,
            "enabled": task.enabled,
            "priority": task.priority,
            "can_run": can_run,
            "reason": reason,
            "last_run": task_state.get("last_run"),
            "run_count": task_state.get("run_count", 0),
            "success_rate": (
                task_state.get("success_count", 0) / task_state.get("run_count", 1)
                if task_state.get("run_count", 0) > 0 else None
            ),
            "tags": task.tags,
        })

    return {
        "total_tasks": len(TASKS),
        "enabled_tasks": sum(1 for t in TASKS.values() if t.enabled),
        "tasks": tasks_status,
        "updated_at": datetime.now().isoformat(),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task Registry - Available tasks")
    parser.add_argument("command", nargs="?", default="list",
                        choices=["list", "status", "run", "available"])
    parser.add_argument("--task", help="Task ID for run command")
    parser.add_argument("--gpu", choices=["3090", "4090", "none"], help="Filter by GPU")
    parser.add_argument("--tag", action="append", help="Filter by tag")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually run")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.command == "list":
        print("\n" + "="*70)
        print("TASK REGISTRY")
        print("="*70)
        for task in sorted(TASKS.values(), key=lambda t: (-t.priority, t.id)):
            status = "✓" if task.enabled else "✗"
            print(f"\n{status} [{task.priority}] {task.id}")
            print(f"   {task.name}: {task.description}")
            print(f"   GPU: {task.gpu} | Cooldown: {task.cooldown}s | Est: {task.estimated_duration}s")
            print(f"   Tags: {', '.join(task.tags)}")
        print("\n" + "="*70)

    elif args.command == "status":
        status = get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\n" + "="*70)
            print("TASK STATUS")
            print("="*70)
            for t in status["tasks"]:
                icon = "✓" if t["can_run"] else "⏳"
                runs = f"runs: {t['run_count']}" if t["run_count"] else "never run"
                print(f"{icon} {t['id']}: {t['reason']} ({runs})")
            print("="*70 + "\n")

    elif args.command == "available":
        tasks = get_available_tasks(gpu=args.gpu, tags=args.tag)
        if args.json:
            print(json.dumps([{"id": t.id, "name": t.name, "priority": t.priority} for t in tasks], indent=2))
        else:
            print(f"\nAvailable tasks (gpu={args.gpu}, tags={args.tag}):")
            for t in tasks:
                print(f"  [{t.priority}] {t.id}: {t.name}")
            print()

    elif args.command == "run":
        if not args.task:
            print("Error: --task required for run command")
            sys.exit(1)
        result = run_task(args.task, dry_run=args.dry_run)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print(f"✅ Task {args.task} completed in {result.get('duration', 0):.1f}s")
            else:
                print(f"❌ Task {args.task} failed: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
