"""
API Ritual - Detailed HTTP API endpoint validation.

This ritual performs comprehensive checks on Tavern API endpoints:
- World state API (/api/world-state)
- Skills API (/api/skills)
- Jobs API (/api/jobs/stats)
- Cluster API (/api/cluster/summary)
"""

import json
import os
from datetime import datetime
from typing import List
import urllib.request
import urllib.error

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("api", "Ritual of APIs", "Detailed HTTP API endpoint validation")
def run() -> List[RitualCheckResult]:
    """Execute all API ritual checks."""
    results = []
    results.append(_check_world_state_api())
    results.append(_check_skills_api())
    results.append(_check_jobs_api())
    results.append(_check_cluster_api())
    return results


def _tavern_url(path: str) -> str:
    """Build Tavern API URL."""
    port = int(os.environ.get("TAVERN_PORT", 8888))
    return f"http://localhost:{port}{path}"


def _fetch_json(url: str, timeout: float = 3.0) -> dict:
    """Fetch JSON from URL, raise on error."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _check_world_state_api() -> RitualCheckResult:
    """Check that /api/world-state returns valid data."""
    start = datetime.utcnow()
    url = _tavern_url("/api/world-state")

    try:
        data = _fetch_json(url)

        # Validate expected fields
        has_health = "health" in data
        has_mode = "realm_mode" in data or "mode" in data
        has_training = "training" in data

        if has_health or has_mode or has_training:
            status = "ok"
        else:
            status = "warn"

        return RitualCheckResult(
            id="world_state_api",
            name="World State API",
            description="Check that /api/world-state returns valid data",
            status=status,
            category="network",
            details={
                "url": url,
                "has_health": has_health,
                "has_mode": has_mode,
                "has_training": has_training,
                "realm_mode": data.get("realm_mode") or data.get("mode"),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="world_state_api",
            name="World State API",
            description="Check that /api/world-state returns valid data",
            status="fail",
            category="network",
            details={"error": str(e.reason), "url": url},
            remediation="Check Tavern server is running",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="world_state_api",
            name="World State API",
            description="Check that /api/world-state returns valid data",
            status="fail",
            category="network",
            details={"error": str(e), "url": url},
            remediation="Check Tavern server is running",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_skills_api() -> RitualCheckResult:
    """Check that /api/skills returns skill list."""
    start = datetime.utcnow()
    url = _tavern_url("/api/skills")

    try:
        data = _fetch_json(url)

        skills = data.get("skills", [])
        skill_count = len(skills)
        skill_ids = [s.get("id") for s in skills[:5]]  # First 5

        status = "ok" if skill_count > 0 else "warn"

        return RitualCheckResult(
            id="skills_api",
            name="Skills API",
            description="Check that /api/skills returns skill list",
            status=status,
            details={
                "url": url,
                "skill_count": skill_count,
                "skill_ids": skill_ids,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="skills_api",
            name="Skills API",
            description="Check that /api/skills returns skill list",
            status="fail",
            details={"error": str(e.reason), "url": url},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="skills_api",
            name="Skills API",
            description="Check that /api/skills returns skill list",
            status="fail",
            details={"error": str(e), "url": url},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_jobs_api() -> RitualCheckResult:
    """Check that /api/jobs/stats returns job statistics."""
    start = datetime.utcnow()
    url = _tavern_url("/api/jobs/stats")

    try:
        data = _fetch_json(url)

        # Check for error response
        if data.get("error"):
            return RitualCheckResult(
                id="jobs_api",
                name="Jobs API",
                description="Check that /api/jobs/stats returns job statistics",
                status="warn",
                details={
                    "url": url,
                    "error": data.get("error"),
                },
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        queue_depth = data.get("queue_depth", 0)
        by_status = data.get("by_status", {})

        return RitualCheckResult(
            id="jobs_api",
            name="Jobs API",
            description="Check that /api/jobs/stats returns job statistics",
            status="ok",
            details={
                "url": url,
                "queue_depth": queue_depth,
                "by_status": by_status,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="jobs_api",
            name="Jobs API",
            description="Check that /api/jobs/stats returns job statistics",
            status="warn",  # Jobs module may not be available
            details={"error": str(e.reason), "url": url},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="jobs_api",
            name="Jobs API",
            description="Check that /api/jobs/stats returns job statistics",
            status="warn",
            details={"error": str(e), "url": url},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_cluster_api() -> RitualCheckResult:
    """Check that /api/cluster/summary returns cluster info."""
    start = datetime.utcnow()
    url = _tavern_url("/api/cluster/summary")

    try:
        data = _fetch_json(url)

        # Check for error response
        if data.get("error"):
            return RitualCheckResult(
                id="cluster_api",
                name="Cluster API",
                description="Check that /api/cluster/summary returns cluster info",
                status="warn",
                details={
                    "url": url,
                    "error": data.get("error"),
                },
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        total_hosts = data.get("total_hosts", 0)
        hosts_online = data.get("hosts_online", 0)

        return RitualCheckResult(
            id="cluster_api",
            name="Cluster API",
            description="Check that /api/cluster/summary returns cluster info",
            status="ok",
            details={
                "url": url,
                "total_hosts": total_hosts,
                "hosts_online": hosts_online,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="cluster_api",
            name="Cluster API",
            description="Check that /api/cluster/summary returns cluster info",
            status="warn",  # Cluster module may not be available
            details={"error": str(e.reason), "url": url},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="cluster_api",
            name="Cluster API",
            description="Check that /api/cluster/summary returns cluster info",
            status="warn",
            details={"error": str(e), "url": url},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
