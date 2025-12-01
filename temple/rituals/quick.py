"""
Quick Ritual - Fast sanity checks on core services.

This ritual performs basic health checks on the Realm's core systems:
- Realm state store connectivity
- Tavern API responsiveness
- VaultKeeper API responsiveness
- Training queue depth
- Active campaign configuration
"""

import json
import os
from datetime import datetime
from typing import List
import urllib.request
import urllib.error

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("quick", "Ritual of Quick", "Fast sanity checks on core services")
def run() -> List[RitualCheckResult]:
    """Execute all quick ritual checks."""
    results = []
    results.append(_check_realm_state())
    results.append(_check_tavern_api())
    results.append(_check_vault_api())
    results.append(_check_queue_depth())
    results.append(_check_active_campaign())
    return results


def _check_realm_state() -> RitualCheckResult:
    """Check that Realm state can be read from the store."""
    start = datetime.utcnow()
    try:
        from core.realm_store import get_store
        store = get_store()
        data = store.get_all()
        mode = data.get("state", {}).get("mode", "unknown")
        training = data.get("state", {}).get("training", {})

        return RitualCheckResult(
            id="realm_state_read",
            name="Realm state readable",
            description="Check that Realm state can be read from the store",
            status="ok",
            details={
                "mode": mode,
                "training_status": training.get("status"),
                "training_step": training.get("step"),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="realm_state_read",
            name="Realm state readable",
            description="Check that Realm state can be read from the store",
            status="fail",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_tavern_api() -> RitualCheckResult:
    """Check that the Tavern API responds to health checks."""
    start = datetime.utcnow()
    port = int(os.environ.get("TAVERN_PORT", 8888))
    url = f"http://localhost:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body) if body else {}
            status = "ok" if resp.status == 200 else "fail"
            details = {
                "status_code": resp.status,
                "url": url,
                "response": data,
            }
    except urllib.error.URLError as e:
        status = "fail"
        details = {"error": str(e.reason), "url": url}
    except Exception as e:
        status = "fail"
        details = {"error": str(e), "url": url}

    return RitualCheckResult(
        id="tavern_api_health",
        name="Tavern API responsive",
        description="Check that the Tavern API responds to health checks",
        status=status,
        details=details,
        started_at=start,
        finished_at=datetime.utcnow(),
    )


def _check_vault_api() -> RitualCheckResult:
    """Check that the VaultKeeper API can be reached."""
    start = datetime.utcnow()
    port = int(os.environ.get("VAULTKEEPER_PORT", 8767))
    url = f"http://localhost:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body) if body else {}
            status = "ok" if resp.status == 200 else "fail"
            details = {
                "status_code": resp.status,
                "url": url,
                "response": data,
            }
    except urllib.error.URLError as e:
        # VaultKeeper is optional, so warn instead of fail
        status = "warn"
        details = {"error": str(e.reason), "url": url}
    except Exception as e:
        status = "warn"
        details = {"error": str(e), "url": url}

    return RitualCheckResult(
        id="vault_api_health",
        name="VaultKeeper health",
        description="Check that the VaultKeeper API can be reached",
        status=status,
        details=details,
        started_at=start,
        finished_at=datetime.utcnow(),
    )


def _check_queue_depth() -> RitualCheckResult:
    """Check that training queue has data."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        queue_dir = get_base_dir() / "queue"

        counts = {}
        total = 0
        for priority in ["high", "normal", "low"]:
            pdir = queue_dir / priority
            if pdir.exists():
                count = len(list(pdir.glob("*.jsonl")))
                counts[priority] = count
                total += count
            else:
                counts[priority] = 0

        # Determine status based on queue depth
        if total > 5:
            status = "ok"
        elif total > 0:
            status = "warn"
        else:
            status = "fail"

        return RitualCheckResult(
            id="queue_depth",
            name="Training queue populated",
            description="Check that training queue has data",
            status=status,
            details={"total": total, **counts},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="queue_depth",
            name="Training queue populated",
            description="Check that training queue has data",
            status="fail",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_active_campaign() -> RitualCheckResult:
    """Check that an active campaign is set."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        campaign_file = get_base_dir() / "control" / "active_campaign.json"

        if not campaign_file.exists():
            return RitualCheckResult(
                id="active_campaign",
                name="Active campaign configured",
                description="Check that an active campaign is set",
                status="warn",
                details={"error": "No active_campaign.json found"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        with open(campaign_file) as f:
            data = json.load(f)

        hero_id = data.get("hero_id")
        campaign_id = data.get("campaign_id")

        # Verify campaign directory exists
        campaign_dir = get_base_dir() / "campaigns" / hero_id / campaign_id if hero_id and campaign_id else None
        campaign_exists = campaign_dir.exists() if campaign_dir else False

        return RitualCheckResult(
            id="active_campaign",
            name="Active campaign configured",
            description="Check that an active campaign is set",
            status="ok" if hero_id and campaign_exists else "warn",
            details={
                "hero_id": hero_id,
                "campaign_id": campaign_id,
                "campaign_exists": campaign_exists,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="active_campaign",
            name="Active campaign configured",
            description="Check that an active campaign is set",
            status="fail",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
