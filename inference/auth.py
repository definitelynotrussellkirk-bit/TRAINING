#!/usr/bin/env python3
"""
Authentication middleware for inference API

Security levels:
- PUBLIC: No auth required (health checks)
- READ: Requires read key (inference, model info)
- ADMIN: Requires admin key (model management, system settings, jobs)

Environment variables:
- INFERENCE_ADMIN_KEY: Required for admin operations
- INFERENCE_READ_KEY: Required for inference operations (admin key also works)
"""

import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load keys from environment
ADMIN_KEY = os.environ.get("INFERENCE_ADMIN_KEY", "")
READ_KEY = os.environ.get("INFERENCE_READ_KEY", "")


def require_admin(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Require admin-level authentication.

    Use for: model management, system settings, power profiles, job queue
    """
    if not ADMIN_KEY:
        raise HTTPException(
            status_code=500,
            detail="INFERENCE_ADMIN_KEY not configured. Set this environment variable."
        )

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )

    if api_key != ADMIN_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin API key."
        )

    return api_key


def require_read(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Require read-level authentication.

    Use for: inference endpoints, model info
    Admin key also grants read access.
    """
    if not READ_KEY and not ADMIN_KEY:
        raise HTTPException(
            status_code=500,
            detail="No API keys configured. Set INFERENCE_READ_KEY or INFERENCE_ADMIN_KEY."
        )

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header."
        )

    # Accept either read key or admin key
    valid_keys = [k for k in [READ_KEY, ADMIN_KEY] if k]
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    return api_key


def check_auth_configured() -> dict:
    """
    Check authentication configuration status.

    Returns dict with status info (for debugging/health checks)
    """
    return {
        "admin_key_configured": bool(ADMIN_KEY),
        "read_key_configured": bool(READ_KEY),
        "any_key_configured": bool(ADMIN_KEY or READ_KEY)
    }
