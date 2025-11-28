"""
Heterogeneous Cluster Routing - Smart job→worker matching.

This module implements resource-aware routing for the job system:
- ordered_job_types_for_worker(): Prioritized list of job types for a worker
- compute_cluster_mode(): Detect catch-up/idle/normal modes
- worker_can_run_job_type(): Capability checking

Usage:
    from jobs.routing import ordered_job_types_for_worker, compute_cluster_mode

    # Get prioritized job types for a worker
    ordered_types = ordered_job_types_for_worker(worker, cluster_mode, queue_depths)

    # Check cluster mode
    mode = compute_cluster_mode(job_store)
"""

import logging
from typing import Dict, List, Optional, Tuple

from jobs.registry import JOB_TYPE_REGISTRY, JobTypeConfig

logger = logging.getLogger("job_routing")


# =============================================================================
# CAPABILITY CHECKING
# =============================================================================

def worker_can_run_job_type(worker: dict, job_type: str) -> Tuple[bool, str]:
    """
    Check if worker can run a job type based on capabilities.

    Args:
        worker: Worker info dict with resource_class, capabilities, etc.
        job_type: The job type name

    Returns:
        Tuple of (can_run: bool, reason: str)
    """
    config = JOB_TYPE_REGISTRY.get(job_type)
    if not config:
        return False, "unknown_job_type"

    # Check roles
    worker_roles = set(worker.get("roles", []))
    if not worker_roles.intersection(config.allowed_roles):
        return False, "role_mismatch"

    # Check forbidden roles
    if config.forbidden_roles and worker_roles.intersection(config.forbidden_roles):
        return False, "forbidden_role"

    # Check required resource class
    if config.required_resource_classes:
        if worker.get("resource_class") not in config.required_resource_classes:
            return False, "resource_class_required"

    # Check GPU requirement
    if config.requires_gpu:
        worker_rc = worker.get("resource_class", "")
        if not worker_rc.startswith("gpu"):
            return False, "requires_gpu"

    # Check capabilities
    worker_caps = set(worker.get("capabilities", []))
    required_caps = set(config.required_capabilities)
    if required_caps and not required_caps.issubset(worker_caps):
        missing = required_caps - worker_caps
        return False, f"missing_capabilities:{','.join(missing)}"

    # Check VRAM
    if config.min_vram_gb:
        # Get VRAM from worker's reported hardware or device config
        gpus = worker.get("gpus", [])
        worker_vram = gpus[0].get("vram_gb", 0) if gpus else 0
        if worker_vram < config.min_vram_gb:
            return False, f"insufficient_vram:{worker_vram}<{config.min_vram_gb}"

    return True, "ok"


def get_allowed_job_types_for_worker(worker: dict) -> List[str]:
    """
    Get all job types a worker can run based on roles and capabilities.

    Args:
        worker: Worker info dict

    Returns:
        List of job type names
    """
    allowed = []
    for job_type in JOB_TYPE_REGISTRY:
        can_run, _ = worker_can_run_job_type(worker, job_type)
        if can_run:
            allowed.append(job_type)
    return allowed


# =============================================================================
# CLUSTER MODE
# =============================================================================

def compute_cluster_mode(queue_stats: Dict[str, Dict[str, int]]) -> str:
    """
    Determine current cluster mode based on queue state.

    Args:
        queue_stats: Dict of {job_type: {"pending": N, "running": M}}

    Returns:
        "catch_up" - Training backlog is high, protect critical workers
        "idle" - No critical jobs, analytics can use all resources
        "normal" - Balanced operation
    """
    # Check critical job backlog
    critical_pending = 0
    critical_running = 0

    for job_type, config in JOB_TYPE_REGISTRY.items():
        if config.job_priority_class == "critical":
            stats = queue_stats.get(job_type, {})
            critical_pending += stats.get("pending", 0)
            critical_running += stats.get("running", 0)

    # Check high priority jobs too
    high_pending = 0
    for job_type, config in JOB_TYPE_REGISTRY.items():
        if config.job_priority_class == "high":
            stats = queue_stats.get(job_type, {})
            high_pending += stats.get("pending", 0)

    # Catch-up mode: critical jobs backlogged
    if critical_pending > 10:
        return "catch_up"

    # Idle mode: no critical or high priority work
    if critical_pending == 0 and critical_running == 0 and high_pending < 5:
        return "idle"

    return "normal"


# =============================================================================
# SMART ROUTING
# =============================================================================

def ordered_job_types_for_worker(
    worker: dict,
    cluster_mode: str = "normal",
    queue_depths: Optional[Dict[str, int]] = None
) -> List[str]:
    """
    Return job types this worker should try to claim, in priority order.

    Considers:
    - Worker's resource_class and capabilities
    - Worker's priority_class (critical workers prioritize critical jobs)
    - Cluster mode (catch_up, normal, idle)
    - Current queue depths (prefer jobs with backlog)

    Args:
        worker: Worker info dict with resource_class, priority_class, roles, capabilities
        cluster_mode: Current cluster mode ("normal", "catch_up", "idle")
        queue_depths: Dict of {job_type: pending_count}

    Returns:
        List of job type names in priority order
    """
    # Get candidate job types this worker can run
    candidate_types = get_allowed_job_types_for_worker(worker)

    if not candidate_types:
        return []

    queue_depths = queue_depths or {}

    # Score each job type for this worker
    def score_job_type(job_type: str) -> Tuple[int, int, int, str]:
        """
        Score a job type for routing priority.
        Lower score = higher priority.
        Returns tuple for sorting: (priority_score, preference_score, backlog_penalty, name)
        """
        config = JOB_TYPE_REGISTRY[job_type]

        # Base priority score (lower = higher priority)
        priority_map = {"critical": 0, "high": 10, "normal": 20, "low": 30}
        priority_score = priority_map.get(config.job_priority_class, 20)

        # Preference score based on resource class match
        worker_rc = worker.get("resource_class", "")
        if config.required_resource_classes and worker_rc in config.required_resource_classes:
            preference_score = 0  # Perfect required match
        elif config.preferred_resource_classes and worker_rc in config.preferred_resource_classes:
            # Position in preference list matters
            try:
                idx = config.preferred_resource_classes.index(worker_rc)
                preference_score = idx + 1
            except ValueError:
                preference_score = 100
        elif not config.preferred_resource_classes and not config.required_resource_classes:
            preference_score = 50  # No preference = neutral
        else:
            preference_score = 100  # Not in preference list

        # Cluster mode adjustments
        worker_priority = worker.get("priority_class", "auxiliary")

        if cluster_mode == "catch_up":
            # In catch-up mode, critical workers should focus on critical/high jobs
            if worker_priority == "critical":
                if config.job_priority_class in ("low",):
                    priority_score += 100  # Push low priority to end
                elif config.job_priority_class == "normal":
                    priority_score += 50  # Deprioritize normal jobs slightly
            # Support workers can still do low-priority work
            elif worker_priority == "support":
                if config.job_priority_class == "low":
                    priority_score += 20  # Slight penalty

        elif cluster_mode == "idle":
            # In idle mode, all jobs are fair game, but prefer analytics
            if config.job_priority_class == "low":
                priority_score -= 5  # Slight boost for analytics/background jobs

        # Backlog bonus: prefer jobs with pending work
        depth = queue_depths.get(job_type, 0)
        if depth > 0:
            backlog_penalty = -min(depth, 10)  # Up to -10 bonus for backlog
        else:
            backlog_penalty = 5  # Slight penalty for empty queues

        return (priority_score, preference_score, backlog_penalty, job_type)

    # Sort by score
    candidate_types.sort(key=score_job_type)

    return candidate_types


def get_routing_explanation(
    worker: dict,
    job_type: str,
    cluster_mode: str = "normal"
) -> Dict[str, any]:
    """
    Get detailed explanation of routing decision for a worker/job type pair.

    Useful for debugging and observability.
    """
    config = JOB_TYPE_REGISTRY.get(job_type)
    if not config:
        return {"error": "unknown_job_type"}

    can_run, reason = worker_can_run_job_type(worker, job_type)

    worker_rc = worker.get("resource_class", "")
    preference_match = "none"
    if config.required_resource_classes and worker_rc in config.required_resource_classes:
        preference_match = "required"
    elif config.preferred_resource_classes and worker_rc in config.preferred_resource_classes:
        idx = config.preferred_resource_classes.index(worker_rc)
        preference_match = f"preferred[{idx}]"

    return {
        "job_type": job_type,
        "worker_id": worker.get("worker_id", worker.get("id", "unknown")),
        "can_run": can_run,
        "reason": reason,
        "cluster_mode": cluster_mode,
        "worker_resource_class": worker_rc,
        "worker_priority_class": worker.get("priority_class", "unknown"),
        "job_priority_class": config.job_priority_class,
        "job_resource_intensity": config.resource_intensity,
        "preference_match": preference_match,
        "required_resource_classes": config.required_resource_classes,
        "preferred_resource_classes": config.preferred_resource_classes,
        "required_capabilities": config.required_capabilities,
        "worker_capabilities": worker.get("capabilities", []),
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Demo the routing logic
    print("Heterogeneous Cluster Routing Demo")
    print("=" * 60)

    # Sample workers
    workers = [
        {
            "worker_id": "trainer4090.claiming",
            "device_id": "trainer4090",
            "resource_class": "gpu_heavy",
            "priority_class": "critical",
            "roles": ["trainer", "eval_worker", "storage_hot", "control_plane"],
            "capabilities": ["cuda_12", "flash_attn", "bf16", "training", "eval_gpu"],
            "gpus": [{"name": "RTX 4090", "vram_gb": 24}],
        },
        {
            "worker_id": "inference3090.claiming",
            "device_id": "inference3090",
            "resource_class": "gpu_medium",
            "priority_class": "support",
            "roles": ["inference", "eval_worker", "analytics", "storage_hot"],
            "capabilities": ["cuda_12", "inference", "analytics", "eval_gpu", "layer_analysis"],
            "gpus": [{"name": "RTX 3090", "vram_gb": 24}],
        },
        {
            "worker_id": "r730xd.claiming",
            "device_id": "r730xd",
            "resource_class": "cpu_heavy",
            "priority_class": "support",
            "roles": ["data_forge", "vault_worker", "analytics", "eval_worker", "storage_warm"],
            "capabilities": ["data_processing", "bulk_storage", "parallel_cpu", "eval_cpu", "archive"],
            "gpus": [],
        },
        {
            "worker_id": "macmini_eval_1.claiming",
            "device_id": "macmini_eval_1",
            "resource_class": "cpu_light",
            "priority_class": "auxiliary",
            "roles": ["eval_worker", "data_forge"],
            "capabilities": ["data_processing", "eval_cpu"],
            "gpus": [],
        },
    ]

    # Sample queue depths
    queue_depths = {
        "eval": 15,
        "data_validate": 30,
        "layer_stats": 5,
        "archive": 2,
        "inference": 3,
    }

    for mode in ["normal", "catch_up", "idle"]:
        print(f"\n{'=' * 60}")
        print(f"CLUSTER MODE: {mode.upper()}")
        print("=" * 60)

        for worker in workers:
            print(f"\n{worker['worker_id']} ({worker['resource_class']}, {worker['priority_class']}):")
            ordered = ordered_job_types_for_worker(worker, mode, queue_depths)
            for i, jt in enumerate(ordered[:8], 1):
                config = JOB_TYPE_REGISTRY[jt]
                depth = queue_depths.get(jt, 0)
                print(f"  {i}. {jt:20} (prio={config.job_priority_class:8}, depth={depth})")
