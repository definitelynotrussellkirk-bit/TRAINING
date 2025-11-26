#!/usr/bin/env python3
"""
Central path configuration for the training system.

This module provides path utilities that:
1. Support environment variable overrides (TRAINING_BASE_DIR)
2. Auto-detect repository root when not configured
3. Provide consistent paths across all modules

Usage:
    from core.paths import get_base_dir, get_models_dir

    base = get_base_dir()  # Auto-detects or uses $TRAINING_BASE_DIR
    models = get_models_dir()  # base / "models"
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# Track whether we've logged the resolution (to avoid spam)
_resolution_logged = False


@lru_cache(maxsize=1)
def get_base_dir() -> Path:
    """
    Get the base training directory.

    Resolution order:
    1. TRAINING_BASE_DIR environment variable (if set and exists)
    2. Auto-detect by searching for CLAUDE.md from current file location
    3. Fallback to common locations

    The resolution method is logged once at INFO level for debugging.

    Returns:
        Path to the base training directory

    Raises:
        RuntimeError: If base directory cannot be determined
    """
    global _resolution_logged

    # Check environment variable first
    if "TRAINING_BASE_DIR" in os.environ:
        path = Path(os.environ["TRAINING_BASE_DIR"])
        if path.exists():
            if not _resolution_logged:
                logger.info(f"Base dir from $TRAINING_BASE_DIR: {path}")
                _resolution_logged = True
            return path
        # If env var is set but doesn't exist, warn but continue to auto-detect
        import warnings
        warnings.warn(
            f"TRAINING_BASE_DIR={path} does not exist, auto-detecting",
            RuntimeWarning
        )

    # Auto-detect by looking for CLAUDE.md
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            if not _resolution_logged:
                logger.info(f"Base dir auto-detected (CLAUDE.md): {current}")
                _resolution_logged = True
            return current
        current = current.parent

    # Fallback: try common locations
    fallbacks = [
        Path.home() / "Desktop" / "TRAINING",
        Path.home() / "TRAINING",
        Path("/path/to/training"),  # Legacy default
    ]

    for fallback in fallbacks:
        if fallback.exists() and (fallback / "CLAUDE.md").exists():
            if not _resolution_logged:
                logger.info(f"Base dir from fallback path: {fallback}")
                _resolution_logged = True
            return fallback

    raise RuntimeError(
        "Could not determine base directory. "
        "Set TRAINING_BASE_DIR environment variable or run from repository."
    )


def get_models_dir() -> Path:
    """Get the models directory (base/models)."""
    return get_base_dir() / "models"


def get_current_model_dir() -> Path:
    """Get the current model directory (base/current_model)."""
    return get_base_dir() / "current_model"


def get_status_dir() -> Path:
    """Get the status directory (base/status)."""
    return get_base_dir() / "status"


def get_logs_dir() -> Path:
    """Get the logs directory (base/logs)."""
    return get_base_dir() / "logs"


def get_queue_dir() -> Path:
    """Get the queue directory (base/queue)."""
    return get_base_dir() / "queue"


def get_inbox_dir() -> Path:
    """Get the inbox directory (base/inbox)."""
    return get_base_dir() / "inbox"


def get_data_dir() -> Path:
    """Get the data directory (base/data)."""
    return get_base_dir() / "data"


def get_config_path() -> Path:
    """Get the config file path (base/config.json)."""
    return get_base_dir() / "config.json"


def get_control_dir() -> Path:
    """Get the control directory (base/control)."""
    return get_base_dir() / "control"


def get_backups_dir() -> Path:
    """Get the backups directory (base/backups)."""
    return get_base_dir() / "backups"


# Remote server paths (3090)
REMOTE_HOST = os.environ.get("INFERENCE_HOST", "192.168.x.x")
REMOTE_PORT = int(os.environ.get("INFERENCE_PORT", "8765"))
REMOTE_SCHEDULER_PORT = int(os.environ.get("SCHEDULER_PORT", "8766"))
REMOTE_MODELS_DIR = Path(os.environ.get("REMOTE_MODELS_DIR", "/path/to/models"))


def get_remote_api_url() -> str:
    """Get the remote inference API URL."""
    return f"http://{REMOTE_HOST}:{REMOTE_PORT}"


def get_scheduler_api_url() -> str:
    """Get the GPU task scheduler API URL."""
    return f"http://{REMOTE_HOST}:{REMOTE_SCHEDULER_PORT}"


def get_pids_dir() -> Path:
    """Get the PID files directory (base/.pids)."""
    return get_base_dir() / ".pids"


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory (base/models/current_model)."""
    return get_base_dir() / "models" / "current_model"


def get_snapshots_dir() -> Path:
    """Get the snapshots directory (base/backups/snapshots)."""
    return get_base_dir() / "backups" / "snapshots"


def get_test_results_dir() -> Path:
    """Get the test results directory (base/test_results)."""
    return get_base_dir() / "test_results"


if __name__ == "__main__":
    # Quick test
    print("Path Configuration:")
    print(f"  Base dir: {get_base_dir()}")
    print(f"  Models: {get_models_dir()}")
    print(f"  Current model: {get_current_model_dir()}")
    print(f"  Status: {get_status_dir()}")
    print(f"  Config: {get_config_path()}")
    print(f"  Logs: {get_logs_dir()}")
    print(f"  Queue: {get_queue_dir()}")
    print(f"  Inbox: {get_inbox_dir()}")
    print(f"  PIDs: {get_pids_dir()}")
    print(f"  Checkpoints: {get_checkpoints_dir()}")
    print(f"  Snapshots: {get_snapshots_dir()}")
    print(f"  Test results: {get_test_results_dir()}")
    print(f"\nRemote Configuration:")
    print(f"  Host: {REMOTE_HOST}")
    print(f"  Remote API: {get_remote_api_url()}")
    print(f"  Scheduler API: {get_scheduler_api_url()}")
