"""
Job Registry - Single source of truth for job contracts.

This module defines:
- JobTypeConfig: Configuration for each job type
- JOB_TYPE_REGISTRY: All job types and their contracts
- Validation functions for payloads

Usage:
    from jobs.registry import (
        validate_job_type,
        validate_payload,
        get_allowed_job_types,
        get_job_config,
    )

    # Validate a job submission
    config = validate_job_type("eval")
    validate_payload("eval", {"skill_id": "bin", "level": 5})

    # Get job types a worker can run
    types = get_allowed_job_types(["eval_worker", "data_forge"])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from guild.job_types import JobErrorCode

logger = logging.getLogger("job_registry")


@dataclass
class JobTypeConfig:
    """
    Configuration for a job type.

    This is the canonical definition of what a job type expects
    and how it should be handled.
    """

    # Identity
    name: str
    description: str

    # Payload contract
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    payload_version: int = 1

    # Execution
    default_timeout: int = 300  # seconds
    max_attempts: int = 3
    retryable_errors: List[JobErrorCode] = field(default_factory=list)

    # Routing
    allowed_roles: List[str] = field(default_factory=list)
    requires_gpu: bool = False

    # Backpressure
    max_pending: int = 100  # Queue limit
    max_running: int = 10  # Concurrent limit
    queue_full_policy: str = "warn"  # "reject" | "warn" | "allow"

    def __post_init__(self):
        # Default retryable errors if not specified
        if not self.retryable_errors:
            self.retryable_errors = [
                JobErrorCode.TRANSPORT_ERROR,
                JobErrorCode.CONNECTION_REFUSED,
                JobErrorCode.INFERENCE_ERROR,
                JobErrorCode.TIMEOUT,
                JobErrorCode.LEASE_EXPIRED,
            ]


# =============================================================================
# JOB TYPE REGISTRY
# =============================================================================

JOB_TYPE_REGISTRY: Dict[str, JobTypeConfig] = {

    # =========================================================================
    # INFERENCE-REQUIRING JOBS
    # =========================================================================

    "eval": JobTypeConfig(
        name="eval",
        description="Run skill evaluation suite on a model",
        required_fields=["skill_id"],
        optional_fields=["level", "batch_size", "model_ref", "checkpoint_step"],
        payload_version=1,
        default_timeout=600,  # 10 min
        max_attempts=2,
        retryable_errors=[
            JobErrorCode.TRANSPORT_ERROR,
            JobErrorCode.INFERENCE_ERROR,
            JobErrorCode.TIMEOUT,
        ],
        allowed_roles=["eval_worker"],
        requires_gpu=False,  # Uses remote inference
        max_pending=50,
        max_running=3,
        queue_full_policy="warn",
    ),

    "sparring": JobTypeConfig(
        name="sparring",
        description="Self-correction sparring session",
        required_fields=["skill_id"],
        optional_fields=["count", "checkpoint", "threshold", "output_path"],
        payload_version=1,
        default_timeout=1800,  # 30 min
        max_attempts=1,  # Expensive, don't retry
        retryable_errors=[],
        allowed_roles=["eval_worker"],
        requires_gpu=False,
        max_pending=10,
        max_running=1,
        queue_full_policy="reject",
    ),

    "inference": JobTypeConfig(
        name="inference",
        description="Direct inference request",
        required_fields=["prompt"],
        optional_fields=["max_tokens", "temperature", "model_ref", "stop_sequences"],
        payload_version=1,
        default_timeout=60,  # 1 min
        max_attempts=2,
        retryable_errors=[
            JobErrorCode.TRANSPORT_ERROR,
            JobErrorCode.INFERENCE_ERROR,
        ],
        allowed_roles=["inference", "eval_worker"],
        requires_gpu=False,
        max_pending=100,
        max_running=10,
        queue_full_policy="reject",
    ),

    # =========================================================================
    # DATA GENERATION JOBS (CPU-BOUND)
    # =========================================================================

    "data_gen": JobTypeConfig(
        name="data_gen",
        description="Generate training data",
        required_fields=["generator", "count"],
        optional_fields=["skill_id", "level", "output_path", "seed"],
        payload_version=1,
        default_timeout=300,  # 5 min
        max_attempts=2,
        retryable_errors=[JobErrorCode.EXECUTION_ERROR],
        allowed_roles=["data_forge"],
        requires_gpu=False,
        max_pending=20,
        max_running=2,
        queue_full_policy="warn",
    ),

    "data_filter": JobTypeConfig(
        name="data_filter",
        description="Filter or validate training data",
        required_fields=["input_path"],
        optional_fields=["output_path", "filter_type", "threshold"],
        payload_version=1,
        default_timeout=300,
        max_attempts=2,
        retryable_errors=[],
        allowed_roles=["data_forge"],
        requires_gpu=False,
        max_pending=10,
        max_running=2,
        queue_full_policy="warn",
    ),

    "data_convert": JobTypeConfig(
        name="data_convert",
        description="Convert data between formats",
        required_fields=["input_path", "output_format"],
        optional_fields=["output_path"],
        payload_version=1,
        default_timeout=300,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["data_forge"],
        requires_gpu=False,
        max_pending=10,
        max_running=2,
        queue_full_policy="warn",
    ),

    # =========================================================================
    # STORAGE/ARCHIVAL JOBS
    # =========================================================================

    "archive": JobTypeConfig(
        name="archive",
        description="Archive checkpoints to cold storage",
        required_fields=["source_zone", "target_zone"],
        optional_fields=["checkpoint_pattern", "keep_last_n", "dry_run"],
        payload_version=1,
        default_timeout=3600,  # 1 hour
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["vault_worker"],
        requires_gpu=False,
        max_pending=5,
        max_running=1,
        queue_full_policy="reject",
    ),

    "retention": JobTypeConfig(
        name="retention",
        description="Apply retention policy to zone",
        required_fields=["zone"],
        optional_fields=["policy", "dry_run", "max_delete"],
        payload_version=1,
        default_timeout=600,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["vault_worker"],
        requires_gpu=False,
        max_pending=3,
        max_running=1,
        queue_full_policy="reject",
    ),

    "sync": JobTypeConfig(
        name="sync",
        description="Sync assets between zones",
        required_fields=["source_zone", "target_zone"],
        optional_fields=["asset_type", "pattern", "dry_run"],
        payload_version=1,
        default_timeout=3600,
        max_attempts=1,
        retryable_errors=[
            JobErrorCode.TRANSPORT_ERROR,
            JobErrorCode.CONNECTION_REFUSED,
        ],
        allowed_roles=["vault_worker"],
        requires_gpu=False,
        max_pending=3,
        max_running=1,
        queue_full_policy="reject",
    ),

    # =========================================================================
    # REPORTING/ANALYTICS JOBS
    # =========================================================================

    "analytics": JobTypeConfig(
        name="analytics",
        description="Run analytics or metrics calculation",
        required_fields=["report_type"],
        optional_fields=["start_date", "end_date", "output_format"],
        payload_version=1,
        default_timeout=300,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["analytics"],
        requires_gpu=False,
        max_pending=10,
        max_running=2,
        queue_full_policy="allow",
    ),

    "report": JobTypeConfig(
        name="report",
        description="Generate a report",
        required_fields=["report_type"],
        optional_fields=["parameters", "output_path"],
        payload_version=1,
        default_timeout=120,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["analytics"],
        requires_gpu=False,
        max_pending=10,
        max_running=3,
        queue_full_policy="allow",
    ),

    "health_check": JobTypeConfig(
        name="health_check",
        description="System health check",
        required_fields=[],
        optional_fields=["components", "deep", "timeout"],
        payload_version=1,
        default_timeout=60,
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["eval_worker", "data_forge", "vault_worker", "analytics"],
        requires_gpu=False,
        max_pending=10,
        max_running=5,
        queue_full_policy="allow",
    ),

    # =========================================================================
    # MODEL ARCHAEOLOGY JOBS (INTERPRETABILITY)
    # =========================================================================

    "layer_stats": JobTypeConfig(
        name="layer_stats",
        description="Compute per-layer weight and activation stats for a checkpoint",
        required_fields=["campaign_id", "hero_id", "checkpoint_path"],
        optional_fields=[
            "model_ref",                  # e.g., 'qwen3-0.6b', 'qwen3-4b'
            "reference_checkpoint_path",  # For drift calculation
            "probe_dataset",              # e.g., 'probes/core_v1.jsonl'
            "max_probe_tokens",           # Limit probe size (default: 4096)
            "output_path",                # Override default analysis dir
            "compute_activations",        # default: True
            "compute_drift",              # default: True if reference provided
        ],
        payload_version=1,
        default_timeout=1800,       # 30 min - model loading is slow
        max_attempts=1,             # Don't retry - fix the issue
        retryable_errors=[],
        allowed_roles=["analytics"],
        requires_gpu=True,          # Need GPU for activation computation
        max_pending=10,
        max_running=1,              # Only one at a time (VRAM constraint)
        queue_full_policy="warn",
    ),

    "layer_drift": JobTypeConfig(
        name="layer_drift",
        description="Compare layer weight drift between two checkpoints",
        required_fields=[
            "campaign_id",
            "hero_id",
            "base_checkpoint_path",
            "target_checkpoint_path",
        ],
        optional_fields=[
            "metrics",              # ['l2', 'cosine', 'frobenius']
            "output_path",
        ],
        payload_version=1,
        default_timeout=600,        # 10 min - no inference needed
        max_attempts=1,
        retryable_errors=[],
        allowed_roles=["analytics"],
        requires_gpu=False,         # Can be CPU-only (loads state_dicts)
        max_pending=20,
        max_running=2,
        queue_full_policy="allow",
    ),
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_job_type(job_type: str) -> JobTypeConfig:
    """
    Get configuration for a job type.

    Args:
        job_type: The job type string

    Returns:
        JobTypeConfig for the type

    Raises:
        ValueError: If job type is unknown
    """
    if job_type not in JOB_TYPE_REGISTRY:
        valid_types = sorted(JOB_TYPE_REGISTRY.keys())
        raise ValueError(
            f"Unknown job type: '{job_type}'. "
            f"Valid types: {valid_types}"
        )
    return JOB_TYPE_REGISTRY[job_type]


def get_job_config(job_type: str) -> Optional[JobTypeConfig]:
    """
    Get configuration for a job type (returns None if unknown).

    Args:
        job_type: The job type string

    Returns:
        JobTypeConfig or None
    """
    return JOB_TYPE_REGISTRY.get(job_type)


def validate_payload(job_type: str, payload: Dict[str, Any]) -> List[str]:
    """
    Validate payload against job type contract.

    Args:
        job_type: The job type string
        payload: The payload dictionary

    Returns:
        List of warning messages (empty if all good)

    Raises:
        ValueError: If required fields are missing
    """
    config = validate_job_type(job_type)

    # Check required fields
    missing = [f for f in config.required_fields if f not in payload]
    if missing:
        raise ValueError(
            f"Missing required fields for '{job_type}': {missing}. "
            f"Required: {config.required_fields}"
        )

    # Check for unknown fields (warning only)
    warnings = []
    known = set(config.required_fields + config.optional_fields)
    unknown = [f for f in payload if f not in known]
    if unknown:
        warnings.append(
            f"Unknown fields for '{job_type}': {unknown}. "
            f"Known fields: {sorted(known)}"
        )

    return warnings


def get_allowed_job_types(roles: List[str]) -> List[str]:
    """
    Get job types a worker with given roles can execute.

    Args:
        roles: List of role strings

    Returns:
        List of job type names the worker can handle
    """
    worker_roles = set(roles)
    return [
        config.name
        for config in JOB_TYPE_REGISTRY.values()
        if set(config.allowed_roles) & worker_roles
    ]


def get_roles_for_job_type(job_type: str) -> List[str]:
    """
    Get roles that can execute a job type.

    Args:
        job_type: The job type string

    Returns:
        List of role names
    """
    config = get_job_config(job_type)
    return config.allowed_roles if config else []


def check_queue_limits(job_type: str, pending: int, running: int) -> tuple:
    """
    Check if queue can accept new job of this type.

    Args:
        job_type: The job type
        pending: Current pending count
        running: Current running count

    Returns:
        Tuple of (can_accept: bool, reason: str, warning: Optional[str])
    """
    config = get_job_config(job_type)
    if not config:
        return True, "ok", None

    warning = None

    # Check pending limit
    if pending >= config.max_pending:
        if config.queue_full_policy == "reject":
            return False, f"queue_full", f"Queue full: {pending}/{config.max_pending} pending"
        elif config.queue_full_policy == "warn":
            warning = f"Queue near capacity: {pending}/{config.max_pending} pending"

    # Check running limit
    if running >= config.max_running:
        if config.queue_full_policy == "reject":
            return False, "at_capacity", f"At capacity: {running}/{config.max_running} running"
        elif config.queue_full_policy == "warn":
            warning = warning or f"At capacity: {running}/{config.max_running} running"

    return True, "ok", warning


def get_registry_summary() -> Dict[str, Any]:
    """Get summary of all registered job types."""
    return {
        "job_types": {
            name: {
                "description": config.description,
                "required_fields": config.required_fields,
                "allowed_roles": config.allowed_roles,
                "requires_gpu": config.requires_gpu,
                "default_timeout": config.default_timeout,
                "max_pending": config.max_pending,
                "max_running": config.max_running,
            }
            for name, config in JOB_TYPE_REGISTRY.items()
        },
        "total_types": len(JOB_TYPE_REGISTRY),
        "by_role": _group_by_role(),
    }


def _group_by_role() -> Dict[str, List[str]]:
    """Group job types by allowed role."""
    by_role: Dict[str, List[str]] = {}
    for config in JOB_TYPE_REGISTRY.values():
        for role in config.allowed_roles:
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(config.name)
    return by_role


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import json

    print("Job Type Registry")
    print("=" * 60)
    print()

    for name, config in sorted(JOB_TYPE_REGISTRY.items()):
        print(f"{name}:")
        print(f"  description: {config.description}")
        print(f"  required: {config.required_fields}")
        print(f"  optional: {config.optional_fields}")
        print(f"  roles: {config.allowed_roles}")
        print(f"  timeout: {config.default_timeout}s")
        print(f"  limits: {config.max_pending} pending, {config.max_running} running")
        print()

    print("\nBy Role:")
    print("-" * 40)
    for role, types in _group_by_role().items():
        print(f"  {role}: {types}")

    print("\n\nValidation Examples:")
    print("-" * 40)

    # Test valid payload
    try:
        warnings = validate_payload("eval", {"skill_id": "bin", "level": 5})
        print("eval with skill_id+level: OK")
        if warnings:
            print(f"  Warnings: {warnings}")
    except ValueError as e:
        print(f"eval error: {e}")

    # Test missing required
    try:
        validate_payload("eval", {"level": 5})
        print("eval without skill_id: OK (unexpected!)")
    except ValueError as e:
        print(f"eval without skill_id: {e}")

    # Test unknown type
    try:
        validate_job_type("unknown_job")
    except ValueError as e:
        print(f"unknown_job: {e}")
