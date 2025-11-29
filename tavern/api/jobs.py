"""
Jobs API - Distributed job execution

Extracted from tavern/server.py for better organization.
Handles:
- /api/jobs - List jobs with filters
- /api/jobs/stats - Job statistics
- /api/jobs/{job_id} - Get specific job
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def _get_job_client():
    """Get job store client."""
    try:
        from jobs.client import JobStoreClient
        return JobStoreClient()
    except ImportError:
        return None


def serve_jobs_list(handler: "TavernHandler", query: dict):
    """
    GET /api/jobs - List jobs with optional filters.

    Query params:
    - status: Filter by status
    - type: Filter by job type
    - limit: Max jobs to return (default 100)
    """
    client = _get_job_client()
    if not client:
        handler._send_json({"error": "Jobs module not available"}, 500)
        return

    try:
        status = query.get("status", [None])[0]
        job_type = query.get("type", [None])[0]
        limit = int(query.get("limit", [100])[0])

        jobs = client.list(status=status, job_type=job_type, limit=limit)

        handler._send_json({
            "count": len(jobs),
            "jobs": jobs,
        })

    except Exception as e:
        logger.error(f"Jobs list error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_jobs_stats(handler: "TavernHandler"):
    """
    GET /api/jobs/stats - Job statistics.
    """
    client = _get_job_client()
    if not client:
        handler._send_json({"error": "Jobs module not available"}, 500)
        return

    try:
        stats = client.stats()
        handler._send_json(stats)

    except Exception as e:
        logger.error(f"Jobs stats error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_job(handler: "TavernHandler", job_id: str):
    """
    GET /api/jobs/{job_id} - Get specific job.
    """
    client = _get_job_client()
    if not client:
        handler._send_json({"error": "Jobs module not available"}, 500)
        return

    try:
        job = client.get(job_id)
        if job:
            handler._send_json(job)
        else:
            handler._send_json({"error": f"Job not found: {job_id}"}, 404)

    except Exception as e:
        logger.error(f"Jobs get error: {e}")
        handler._send_json({"error": str(e)}, 500)
