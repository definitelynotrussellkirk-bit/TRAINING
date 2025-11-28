#!/usr/bin/env python3
"""
Central path configuration for the training system.

This module provides path utilities that:
1. Use Guild facility resolver when available
2. Support environment variable overrides (TRAINING_BASE_DIR, GUILD_BASE_DIR)
3. Auto-detect repository root when not configured
4. Provide consistent paths across all modules

Usage:
    from core.paths import get_base_dir, get_models_dir, resolve_path

    base = get_base_dir()  # Auto-detects or uses guild resolver
    models = get_models_dir()  # base / "models"
    path = resolve_path("@checkpoints")  # Guild facility shorthand
"""

import os
import sys
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

# Ensure project root is in sys.path for guild imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

# Track whether we've logged the resolution (to avoid spam)
_resolution_logged = False
_guild_available: Optional[bool] = None


def _check_guild_available() -> bool:
    """Check if guild resolver is available and configured."""
    global _guild_available
    if _guild_available is not None:
        return _guild_available

    try:
        from guild.facilities.resolver import init_resolver, get_resolver
        try:
            init_resolver()
            _guild_available = True
        except FileNotFoundError:
            _guild_available = False
    except ImportError:
        _guild_available = False

    return _guild_available


def _detect_base_dir_legacy() -> Path:
    """Legacy base directory detection."""
    global _resolution_logged

    # Check environment variable first
    env_path = os.environ.get("GUILD_BASE_DIR") or os.environ.get("TRAINING_BASE_DIR")
    if env_path:
        path = Path(env_path)
        if path.exists():
            if not _resolution_logged:
                logger.info(f"Base dir from environment: {path}")
                _resolution_logged = True
            return path

    # Auto-detect by looking for CLAUDE.md
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            if not _resolution_logged:
                logger.info(f"Base dir auto-detected (CLAUDE.md): {current}")
                _resolution_logged = True
            return current
        current = current.parent

    # Fallback: try common locations (no hardcoded usernames)
    fallbacks = [
        Path.home() / "Desktop" / "TRAINING",
        Path.home() / "TRAINING",
        Path.cwd(),
    ]

    for fallback in fallbacks:
        if fallback.exists() and (fallback / "CLAUDE.md").exists():
            if not _resolution_logged:
                logger.info(f"Base dir from fallback path: {fallback}")
                _resolution_logged = True
            return fallback

    # No valid directory found - fail clearly
    raise RuntimeError(
        "Cannot detect training base directory. Either:\n"
        "  1. Set TRAINING_BASE_DIR environment variable\n"
        "  2. Run from the training repository root\n"
        "  3. Ensure CLAUDE.md exists in the project root"
    )


@lru_cache(maxsize=1)
def get_base_dir() -> Path:
    """
    Get the base training directory.

    Resolution order:
    1. Guild facility resolver (if configured)
    2. GUILD_BASE_DIR or TRAINING_BASE_DIR environment variable
    3. Auto-detect by searching for CLAUDE.md from current file location
    4. Fallback to common locations

    Returns:
        Path to the base training directory
    """
    global _resolution_logged

    # Try guild resolver first
    if _check_guild_available():
        try:
            from guild.facilities.resolver import get_resolver
            resolver = get_resolver()
            facility = resolver.get_facility(resolver.current_facility_id)
            base_path = Path(os.path.expandvars(facility.base_path)).expanduser()
            if not _resolution_logged:
                logger.info(f"Base dir from guild resolver ({facility.id}): {base_path}")
                _resolution_logged = True
            return base_path
        except Exception as e:
            logger.debug(f"Guild resolver failed, using legacy: {e}")

    # Fall back to legacy detection
    return _detect_base_dir_legacy()


def resolve_path(path_spec: str) -> Path:
    """
    Resolve a path specification.

    Supports:
    - facility:id:path - Guild facility paths
    - @path - Current facility shorthand
    - ~/path - Home expansion
    - /absolute - Unchanged
    - relative - Relative to base_dir

    Args:
        path_spec: Path specification to resolve

    Returns:
        Resolved Path object
    """
    if path_spec.startswith("facility:") or path_spec.startswith("@"):
        if _check_guild_available():
            try:
                from guild.facilities.resolver import resolve
                return resolve(path_spec)
            except Exception as e:
                logger.debug(f"Guild resolve failed for {path_spec}: {e}")

        # Can't resolve guild paths without guild
        if path_spec.startswith("@"):
            # Strip @ and treat as relative to base
            return get_base_dir() / path_spec[1:].split("/")[0]
        raise ValueError(f"Guild resolver not available for: {path_spec}")

    if path_spec.startswith("~"):
        return Path(path_spec).expanduser()

    if path_spec.startswith("/"):
        return Path(path_spec)

    # Relative path - resolve relative to base_dir
    return get_base_dir() / path_spec


# Convenience functions
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


def get_pids_dir() -> Path:
    """Get the PID files directory (base/.pids)."""
    return get_base_dir() / ".pids"


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory (base/current_model)."""
    return get_base_dir() / "current_model"


def get_snapshots_dir() -> Path:
    """Get the snapshots directory (base/backups/snapshots)."""
    return get_base_dir() / "backups" / "snapshots"


def get_test_results_dir() -> Path:
    """Get the test results directory (base/test_results)."""
    return get_base_dir() / "test_results"


# Remote server configuration - use core.hosts for service discovery
# These are deprecated - use get_service_url("inference") from core.hosts instead
def _get_inference_host() -> str:
    """Get inference host from hosts.json or environment."""
    if "INFERENCE_HOST" in os.environ:
        return os.environ["INFERENCE_HOST"]
    try:
        from core.hosts import get_host
        host = get_host("3090")
        return host.host if host else "localhost"
    except Exception:
        return os.environ.get("INFERENCE_HOST", "localhost")

def _get_inference_port() -> int:
    """Get inference port from hosts.json or environment."""
    if "INFERENCE_PORT" in os.environ:
        return int(os.environ["INFERENCE_PORT"])
    try:
        from core.hosts import get_host
        host = get_host("3090")
        if host and "inference" in host.services:
            return host.services["inference"].port
    except Exception:
        pass
    return int(os.environ.get("INFERENCE_PORT", "8765"))

def _get_remote_models_dir() -> Path:
    """Get remote models directory from hosts.json or environment."""
    if "REMOTE_MODELS_DIR" in os.environ:
        return Path(os.environ["REMOTE_MODELS_DIR"])
    try:
        from core.hosts import get_host
        host = get_host("3090")
        if host and host.models_dir:
            return Path(host.models_dir)
    except Exception:
        pass
    return Path(os.environ.get("REMOTE_MODELS_DIR", "/tmp/models"))

# Module-level access (deprecated - use core.hosts.get_service_url() instead)
# These call the functions above on each access for backward compatibility
def __getattr__(name):
    """Lazy attribute access for deprecated remote config variables."""
    if name == "REMOTE_HOST":
        return _get_inference_host()
    elif name == "REMOTE_PORT":
        return _get_inference_port()
    elif name == "REMOTE_SCHEDULER_PORT":
        return _get_inference_port() + 1
    elif name == "REMOTE_MODELS_DIR":
        return _get_remote_models_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_remote_api_url() -> str:
    """Get the remote inference API URL. Prefer core.hosts.get_service_url('inference')."""
    return f"http://{_get_inference_host()}:{_get_inference_port()}"


def get_scheduler_api_url() -> str:
    """Get the GPU task scheduler API URL. Prefer core.hosts.get_service_url('scheduler')."""
    return f"http://{_get_inference_host()}:{_get_inference_port() + 1}"


if __name__ == "__main__":
    # Reset caching to get fresh results in test mode
    _guild_available = None
    _resolution_logged = False
    get_base_dir.cache_clear()

    # Quick test
    print("Path Configuration:")
    print(f"  Guild available: {_check_guild_available()}")
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
    print(f"\nPath Resolution:")
    print(f"  resolve_path('status'): {resolve_path('status')}")
    print(f"  resolve_path('~/test'): {resolve_path('~/test')}")
    print(f"  resolve_path('/absolute'): {resolve_path('/absolute')}")
