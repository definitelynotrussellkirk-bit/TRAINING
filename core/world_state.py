"""
World State - Single authoritative snapshot of the Realm.

Aggregates:
- RealmMode (TRAINING/IDLE/etc)
- RunContext (hero/campaign/model)
- Training status (current job, progress)
- Eval status (running suites, pending jobs)
- Worker heartbeats
- GPU stats
- Health warnings

Usage:
    from core.world_state import get_world_state

    state = get_world_state()
    print(f"Mode: {state['realm_mode']}")
    print(f"Training: {state['training']['status']}")
    print(f"Warnings: {state['warnings']}")

The Tavern exposes this at /api/world-state for the UI.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_world_state() -> Dict[str, Any]:
    """
    Build a complete world state snapshot.

    This is the ONE place that answers "what is the Realm doing right now?"
    """
    state = {
        "generated_at": datetime.now().isoformat(),
        "realm_mode": "idle",
        "realm_mode_info": {},
        "run": {},
        "training": {},
        "evals": {},
        "jobs": {},
        "workers": [],
        "gpus": [],
        "warnings": [],
        "health": "unknown",
    }

    # 1. Realm Mode
    try:
        from core.realm_state import get_realm_state
        realm_state = get_realm_state()
        state["realm_mode"] = realm_state.mode.value
        state["realm_mode_info"] = {
            "description": realm_state.mode.description,
            "changed_at": realm_state.changed_at,
            "changed_by": realm_state.changed_by,
            "reason": realm_state.reason,
            "allows_training": realm_state.mode.allows_training,
            "allows_evals": realm_state.mode.allows_evals,
        }
    except Exception as e:
        state["warnings"].append(f"Failed to get realm mode: {e}")

    # 2. Run Context
    try:
        from core.run_context import get_run_context
        ctx = get_run_context()
        state["run"] = {
            "hero_id": ctx.hero_id,
            "hero_name": ctx.hero_name,
            "campaign_id": ctx.campaign_id,
            "campaign_name": ctx.campaign_name,
            "model_path": ctx.model_path,
            "current_model_dir": ctx.current_model_dir,
            "context_hash": ctx.context_hash(),
            "is_first_run": ctx.is_first_run,
            "is_legacy_mode": ctx.is_legacy_mode,
        }
    except Exception as e:
        state["warnings"].append(f"Failed to get run context: {e}")

    # 3. Training Status
    state["training"] = _get_training_status()

    # 4. Workers (heartbeats)
    state["workers"] = _get_worker_status()

    # 5. Eval Status
    state["evals"] = _get_eval_status()

    # 6. Job Counts
    state["jobs"] = _get_job_counts()

    # 7. GPU Stats
    state["gpus"] = _get_gpu_stats(state["workers"])

    # 8. Health Assessment
    state["health"], health_warnings = _assess_health(state)
    state["warnings"].extend(health_warnings)

    return state


def _get_training_status() -> Dict[str, Any]:
    """Get current training job status."""
    result = {
        "status": "idle",  # "idle", "running", "error"
        "current_job_id": None,
        "current_job_name": None,
        "progress": None,
        "last_heartbeat": None,
    }

    # Check training daemon heartbeat
    try:
        from core.heartbeat import get_training_worker
        worker = get_training_worker()

        if worker and worker.is_alive:
            result["status"] = worker.status
            result["current_job_id"] = worker.current_job_id
            result["last_heartbeat"] = worker.updated_at

            # Extract progress from extra
            if worker.extra:
                result["progress"] = {
                    "step": worker.extra.get("step"),
                    "total": worker.extra.get("total_steps"),
                    "it_per_sec": worker.extra.get("it_per_sec"),
                    "eta_seconds": worker.extra.get("eta_seconds"),
                }
                if worker.extra.get("current_file"):
                    result["current_job_name"] = worker.extra.get("current_file")

    except Exception as e:
        logger.debug(f"Failed to get training worker: {e}")

    # Also check training_status.json as fallback
    try:
        from core.paths import get_base_dir
        status_file = get_base_dir() / "status" / "training_status.json"
        if status_file.exists():
            with open(status_file) as f:
                ts = json.load(f)

            # If heartbeat says idle but status file has current_file, use status file
            if result["status"] == "idle" and ts.get("current_file"):
                result["status"] = "running"
                result["current_job_name"] = ts.get("current_file")

            if result["progress"] is None and ts.get("step"):
                result["progress"] = {
                    "step": ts.get("step"),
                    "total": ts.get("total_steps"),
                    "it_per_sec": ts.get("tokens_per_sec_avg", 0) / 500 if ts.get("tokens_per_sec_avg") else None,
                }

    except Exception as e:
        logger.debug(f"Failed to read training_status.json: {e}")

    return result


def _get_worker_status() -> List[Dict[str, Any]]:
    """Get status of all workers."""
    workers = []

    try:
        from core.heartbeat import get_all_heartbeats

        for hb in get_all_heartbeats():
            workers.append({
                "id": hb.worker_id,
                "role": hb.role,
                "status": "alive" if hb.is_alive else ("stale" if hb.is_stale else "unknown"),
                "device": hb.device,
                "current_job_id": hb.current_job_id,
                "current_job_type": hb.current_job_type,
                "last_heartbeat": hb.updated_at,
                "age_seconds": round(hb.age_seconds, 1),
            })
    except Exception as e:
        logger.debug(f"Failed to get heartbeats: {e}")

    return workers


def _get_eval_status() -> Dict[str, Any]:
    """Get eval suite status."""
    result = {
        "running_suites": [],
        "pending_jobs": 0,
        "recent_results": [],
    }

    # Check for running eval workers
    try:
        from core.heartbeat import get_workers_by_role

        eval_workers = get_workers_by_role("eval")
        running_jobs = {}  # suite_id -> count

        for w in eval_workers:
            if w.is_alive and w.current_job_id:
                # Try to extract suite_id from job
                suite_id = w.extra.get("suite_id", "unknown") if w.extra else "unknown"
                running_jobs[suite_id] = running_jobs.get(suite_id, 0) + 1

        result["running_suites"] = [
            {"suite_id": sid, "jobs": count}
            for sid, count in running_jobs.items()
        ]
    except Exception as e:
        logger.debug(f"Failed to get eval workers: {e}")

    # Get pending eval jobs from job store
    try:
        from vault.job_store import get_job_store
        store = get_job_store()
        pending = store.list_jobs(status="pending", job_type="eval")
        result["pending_jobs"] = len(pending)
    except Exception as e:
        logger.debug(f"Failed to get pending eval jobs: {e}")

    # Get recent eval results
    try:
        from core.evaluation_ledger import get_eval_ledger
        ledger = get_eval_ledger()
        recent = ledger.list_all(limit=5)
        result["recent_results"] = [
            {
                "skill": r.skill,
                "level": r.level,
                "accuracy": r.accuracy,
                "checkpoint_step": r.checkpoint_step,
                "timestamp": r.timestamp,
            }
            for r in recent
        ]
    except Exception as e:
        logger.debug(f"Failed to get recent evals: {e}")

    return result


def _get_job_counts() -> Dict[str, Any]:
    """Get job queue counts."""
    result = {
        "total_pending": 0,
        "total_running": 0,
        "total_completed": 0,
        "total_failed": 0,
    }

    try:
        from vault.job_store import get_job_store
        store = get_job_store()

        result["total_pending"] = len(store.list_jobs(status="pending"))
        result["total_running"] = len(store.list_jobs(status="running"))
        result["total_completed"] = len(store.list_jobs(status="completed"))
        result["total_failed"] = len(store.list_jobs(status="failed"))
    except Exception as e:
        logger.debug(f"Failed to get job counts: {e}")

    return result


def _get_gpu_stats(workers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get GPU stats and correlate with workers."""
    gpus = []

    try:
        # Run nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpu_id = f"GPU{parts[0]}"

                    # Find worker using this GPU
                    active_job_id = None
                    active_job_type = None
                    for w in workers:
                        if w.get("device") == gpu_id and w.get("status") == "alive":
                            active_job_id = w.get("current_job_id")
                            active_job_type = w.get("current_job_type") or w.get("role")
                            break

                    gpus.append({
                        "id": gpu_id,
                        "utilization": int(parts[1]) if parts[1].isdigit() else 0,
                        "memory_used_gb": round(float(parts[2]) / 1024, 1) if parts[2].replace(".", "").isdigit() else 0,
                        "memory_total_gb": round(float(parts[3]) / 1024, 1) if parts[3].replace(".", "").isdigit() else 0,
                        "power_w": int(float(parts[4])) if parts[4].replace(".", "").isdigit() else 0,
                        "temperature_c": int(parts[5]) if parts[5].isdigit() else 0,
                        "active_job_id": active_job_id,
                        "active_job_type": active_job_type,
                    })

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except FileNotFoundError:
        logger.debug("nvidia-smi not found")
    except Exception as e:
        logger.debug(f"Failed to get GPU stats: {e}")

    return gpus


def _assess_health(state: Dict[str, Any]) -> tuple[str, List[str]]:
    """
    Assess overall realm health.

    Returns (health_status, warnings)
    health_status: "healthy", "warning", "error"
    """
    warnings = []
    has_error = False
    has_warning = False

    # Check for stale training status (callback wiring bug)
    try:
        from core.status_monitor import check_status_staleness
        staleness = check_status_staleness(max_stale_seconds=120.0)
        if staleness.is_stale:
            warnings.append(
                f"Training status stale for {staleness.stale_seconds:.0f}s "
                f"(step stuck at {staleness.current_step}) - callbacks may be broken"
            )
            has_error = True
            state["training"]["status_stale"] = True
            state["training"]["status_stale_seconds"] = staleness.stale_seconds
    except Exception as e:
        logger.debug(f"Staleness check failed: {e}")

    # Check realm mode vs actual state
    mode = state.get("realm_mode", "idle")
    training_status = state.get("training", {}).get("status", "idle")

    if mode == "training":
        # In training mode, should have training daemon alive
        training_workers = [w for w in state.get("workers", []) if w.get("role") == "training"]
        if not training_workers:
            warnings.append("Mode is TRAINING but no training daemon heartbeat found")
            has_warning = True
        elif all(w.get("status") != "alive" for w in training_workers):
            warnings.append("Training daemon heartbeat is stale")
            has_warning = True

    # Check for stale workers
    stale_workers = [w for w in state.get("workers", []) if w.get("status") == "stale"]
    for w in stale_workers:
        warnings.append(f"Worker {w['id']} has stale heartbeat ({w.get('age_seconds', 0):.0f}s old)")
        has_warning = True

    # Check for GPU utilization without known job
    for gpu in state.get("gpus", []):
        if gpu.get("utilization", 0) > 50 and not gpu.get("active_job_id"):
            warnings.append(f"{gpu['id']} is {gpu['utilization']}% utilized but no known job")
            has_warning = True

    # Check run context
    run = state.get("run", {})
    if not run.get("hero_id") and not run.get("is_first_run"):
        warnings.append("No active hero/campaign configured")
        has_warning = True

    # Check for failed jobs
    jobs = state.get("jobs", {})
    if jobs.get("total_failed", 0) > 0:
        warnings.append(f"{jobs['total_failed']} failed jobs in queue")
        has_warning = True

    # Determine health status
    if has_error:
        health = "error"
    elif has_warning:
        health = "warning"
    else:
        health = "healthy"

    return health, warnings


# =============================================================================
# SNAPSHOT FILE (optional persistence)
# =============================================================================

def save_world_state_snapshot(state: Optional[Dict[str, Any]] = None):
    """
    Save world state to status/world_state.json.

    Called periodically by the aggregator daemon (if running).
    """
    if state is None:
        state = get_world_state()

    try:
        from core.paths import get_base_dir
        snapshot_file = get_base_dir() / "status" / "world_state.json"
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)

        with open(snapshot_file, "w") as f:
            json.dump(state, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to save world state snapshot: {e}")


def load_world_state_snapshot() -> Optional[Dict[str, Any]]:
    """
    Load cached world state snapshot.

    Returns None if no snapshot exists or it's too old.
    """
    try:
        from core.paths import get_base_dir
        snapshot_file = get_base_dir() / "status" / "world_state.json"

        if not snapshot_file.exists():
            return None

        with open(snapshot_file) as f:
            state = json.load(f)

        # Check freshness (reject if >30s old)
        generated_at = state.get("generated_at", "")
        try:
            ts = datetime.fromisoformat(generated_at)
            age = (datetime.now() - ts).total_seconds()
            if age > 30:
                return None
        except Exception:
            return None

        return state

    except Exception as e:
        logger.debug(f"Failed to load world state snapshot: {e}")
        return None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="World State Aggregator")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--save", action="store_true", help="Save snapshot to file")

    args = parser.parse_args()

    state = get_world_state()

    if args.save:
        save_world_state_snapshot(state)
        print("Saved world state snapshot")

    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print(f"World State - {state['generated_at'][:19]}")
        print("=" * 60)

        # Health
        health_icon = {"healthy": "🟢", "warning": "🟡", "error": "🔴"}.get(state["health"], "⚪")
        print(f"\n{health_icon} Health: {state['health'].upper()}")
        if state["warnings"]:
            for w in state["warnings"]:
                print(f"   ⚠️  {w}")

        # Mode
        mode_info = state.get("realm_mode_info", {})
        print(f"\n📍 Mode: {state['realm_mode'].upper()}")
        print(f"   {mode_info.get('description', '')}")
        if mode_info.get("changed_at"):
            print(f"   Changed: {mode_info['changed_at'][:19]} by {mode_info.get('changed_by', '?')}")

        # Run
        run = state.get("run", {})
        if run.get("hero_id"):
            print(f"\n🦸 Hero: {run.get('hero_name', run['hero_id'])}")
            print(f"   Campaign: {run.get('campaign_id', 'N/A')}")
            print(f"   Context hash: {run.get('context_hash', 'N/A')[:8]}...")

        # Training
        training = state.get("training", {})
        print(f"\n⚔️  Training: {training.get('status', 'unknown').upper()}")
        if training.get("current_job_name"):
            print(f"   Job: {training['current_job_name']}")
        if training.get("progress"):
            p = training["progress"]
            if p.get("step") and p.get("total"):
                pct = p["step"] / p["total"] * 100
                print(f"   Progress: {p['step']}/{p['total']} ({pct:.1f}%)")
            if p.get("it_per_sec"):
                print(f"   Speed: {p['it_per_sec']:.2f} it/s")

        # Workers
        workers = state.get("workers", [])
        if workers:
            print(f"\n👷 Workers ({len(workers)}):")
            for w in workers:
                icon = {"alive": "🟢", "stale": "🟡"}.get(w["status"], "🔴")
                job = f" -> {w['current_job_id'][:8]}..." if w.get("current_job_id") else ""
                print(f"   {icon} {w['id']} ({w['role']}) on {w.get('device', 'N/A')}{job}")

        # GPUs
        gpus = state.get("gpus", [])
        if gpus:
            print(f"\n🎮 GPUs ({len(gpus)}):")
            for g in gpus:
                job_info = f" -> {g['active_job_type']}" if g.get("active_job_type") else ""
                print(f"   {g['id']}: {g['utilization']}% util, {g['memory_used_gb']}/{g['memory_total_gb']}GB, {g['temperature_c']}°C{job_info}")

        # Jobs
        jobs = state.get("jobs", {})
        print(f"\n📋 Jobs: {jobs.get('total_pending', 0)} pending, {jobs.get('total_running', 0)} running, {jobs.get('total_failed', 0)} failed")

        # Evals
        evals = state.get("evals", {})
        if evals.get("running_suites"):
            print(f"\n📊 Eval Suites Running:")
            for s in evals["running_suites"]:
                print(f"   {s['suite_id']}: {s['jobs']} jobs")
        if evals.get("recent_results"):
            print(f"\n📈 Recent Eval Results:")
            for r in evals["recent_results"][:3]:
                print(f"   {r['skill']} L{r['level']}: {r['accuracy']*100:.1f}% (step {r['checkpoint_step']})")
