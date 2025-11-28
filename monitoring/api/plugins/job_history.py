#!/usr/bin/env python3
"""
Job History Plugin - Provides job history data for unified API

Reads from status/job_history.jsonl to provide:
- Recent jobs summary
- Job counts by status
- Aggregate statistics
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BasePlugin, PluginError


class JobHistoryPlugin(BasePlugin):
    """
    Plugin that provides job history data.

    Reads from job_history.jsonl and provides:
    - Recent jobs (last 10)
    - Counts by status
    - Total training hours
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache_duration = config.get('cache_duration', 60)  # 1 min cache

    def get_name(self) -> str:
        return "job_history"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "description": "Training job history and statistics",
            "machine": "4090",
            "location": "status/job_history.jsonl",
            "refresh_interval": 60,
            "critical": False,
            "data_type": "file"
        }

    def fetch(self) -> Dict[str, Any]:
        """Fetch job history data."""
        try:
            from core.paths import get_status_dir
            job_file = get_status_dir() / "job_history.jsonl"
        except ImportError:
            job_file = Path(__file__).parent.parent.parent.parent / "status" / "job_history.jsonl"

        if not job_file.exists():
            return {
                "recent_jobs": [],
                "total_jobs": 0,
                "by_status": {},
                "total_hours": 0,
                "avg_duration_hours": None
            }

        # Read and dedupe jobs
        jobs = {}
        try:
            with open(job_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            job_id = record.get("job_id")
                            if job_id:
                                jobs[job_id] = record
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise PluginError(f"Failed to read job history: {e}")

        # Compute statistics
        by_status = {}
        total_hours = 0
        durations = []

        for job in jobs.values():
            status = job.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1

            if job.get("actual_hours"):
                total_hours += job["actual_hours"]
                durations.append(job["actual_hours"])

        # Get recent jobs
        sorted_jobs = sorted(
            jobs.values(),
            key=lambda j: j.get("created_at", ""),
            reverse=True
        )[:10]

        avg_duration = sum(durations) / len(durations) if durations else None

        return {
            "recent_jobs": sorted_jobs,
            "total_jobs": len(jobs),
            "by_status": by_status,
            "total_hours": round(total_hours, 2),
            "avg_duration_hours": round(avg_duration, 2) if avg_duration else None,
            "completed_count": by_status.get("completed", 0),
            "failed_count": by_status.get("failed", 0),
            "running_count": by_status.get("running", 0)
        }
