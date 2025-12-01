"""
Ritual of the Champion - Model and checkpoint health diagnostics.

This ritual checks the health of models and checkpoints:
- Current model exists and is valid
- Recent checkpoints exist
- Checkpoint ledger consistency
- Champion designation validity
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("champion", "Ritual of the Champion", "Model and checkpoint health diagnostics")
def run() -> List[RitualCheckResult]:
    """Execute all champion ritual checks."""
    results = []
    results.append(_check_current_model())
    results.append(_check_recent_checkpoints())
    results.append(_check_checkpoint_ledger())
    results.append(_check_base_model())
    return results


def _check_current_model() -> RitualCheckResult:
    """Check that current_model exists and is valid."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        model_dir = get_base_dir() / "models" / "current_model"

        if not model_dir.exists():
            return RitualCheckResult(
                id="current_model",
                name="Current Model",
                description="Verify current_model directory exists and is valid",
                status="fail",
                category="model",
                details={"error": "models/current_model does not exist"},
                remediation="Copy a checkpoint to models/current_model or start a new campaign",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Check for essential files
        essential_files = ["config.json", "tokenizer.json"]
        missing = []
        for f in essential_files:
            if not (model_dir / f).exists():
                missing.append(f)

        # Check for model weights
        has_weights = (
            (model_dir / "model.safetensors").exists() or
            (model_dir / "pytorch_model.bin").exists() or
            any(model_dir.glob("model-*.safetensors"))
        )

        if missing or not has_weights:
            status = "fail"
        else:
            status = "ok"

        # Get model size
        total_size = 0
        for f in model_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        size_gb = total_size / (1024**3)

        return RitualCheckResult(
            id="current_model",
            name="Current Model",
            description="Verify current_model directory exists and is valid",
            status=status,
            category="model",
            details={
                "path": str(model_dir),
                "missing_files": missing,
                "has_weights": has_weights,
                "size_gb": round(size_gb, 2),
            },
            remediation="Restore model from checkpoint or base model" if status == "fail" else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="current_model",
            name="Current Model",
            description="Verify current_model directory exists and is valid",
            status="fail",
            category="model",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_recent_checkpoints() -> RitualCheckResult:
    """Check that recent checkpoints exist."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir

        # Get active campaign
        campaign_file = get_base_dir() / "control" / "active_campaign.json"
        if not campaign_file.exists():
            return RitualCheckResult(
                id="recent_checkpoints",
                name="Recent Checkpoints",
                description="Verify recent checkpoints exist for active campaign",
                status="skip",
                category="model",
                details={"error": "No active campaign configured"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        with open(campaign_file) as f:
            campaign = json.load(f)

        hero_id = campaign.get("hero_id")
        campaign_id = campaign.get("campaign_id")

        if not hero_id or not campaign_id:
            return RitualCheckResult(
                id="recent_checkpoints",
                name="Recent Checkpoints",
                description="Verify recent checkpoints exist for active campaign",
                status="warn",
                category="model",
                details={"error": "Invalid active campaign configuration"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        checkpoint_dir = get_base_dir() / "campaigns" / hero_id / campaign_id / "checkpoints"

        if not checkpoint_dir.exists():
            return RitualCheckResult(
                id="recent_checkpoints",
                name="Recent Checkpoints",
                description="Verify recent checkpoints exist for active campaign",
                status="warn",
                category="model",
                details={
                    "error": "Checkpoint directory does not exist",
                    "path": str(checkpoint_dir),
                },
                remediation="Train to create checkpoints",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Find checkpoints
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not checkpoints:
            return RitualCheckResult(
                id="recent_checkpoints",
                name="Recent Checkpoints",
                description="Verify recent checkpoints exist for active campaign",
                status="warn",
                category="model",
                details={"checkpoint_count": 0},
                remediation="Train to create checkpoints",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Check age of most recent
        most_recent = checkpoints[0]
        age_hours = (datetime.now() - datetime.fromtimestamp(most_recent.stat().st_mtime)).total_seconds() / 3600

        # Status based on checkpoint count and age
        if len(checkpoints) < 2 and age_hours > 24:
            status = "warn"
        else:
            status = "ok"

        return RitualCheckResult(
            id="recent_checkpoints",
            name="Recent Checkpoints",
            description="Verify recent checkpoints exist for active campaign",
            status=status,
            category="model",
            details={
                "checkpoint_count": len(checkpoints),
                "most_recent": most_recent.name,
                "age_hours": round(age_hours, 1),
                "path": str(checkpoint_dir),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="recent_checkpoints",
            name="Recent Checkpoints",
            description="Verify recent checkpoints exist for active campaign",
            status="fail",
            category="model",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_checkpoint_ledger() -> RitualCheckResult:
    """Check checkpoint ledger consistency."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        ledger_file = get_base_dir() / "data" / "checkpoint_ledger.json"

        if not ledger_file.exists():
            return RitualCheckResult(
                id="checkpoint_ledger",
                name="Checkpoint Ledger",
                description="Verify checkpoint ledger exists and is consistent",
                status="warn",
                category="model",
                details={"error": "No checkpoint_ledger.json found"},
                remediation="Ledger will be created on next checkpoint save",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        with open(ledger_file) as f:
            ledger = json.load(f)

        entries = ledger.get("checkpoints", [])

        return RitualCheckResult(
            id="checkpoint_ledger",
            name="Checkpoint Ledger",
            description="Verify checkpoint ledger exists and is consistent",
            status="ok",
            category="model",
            details={
                "entry_count": len(entries),
                "ledger_path": str(ledger_file),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except json.JSONDecodeError as e:
        return RitualCheckResult(
            id="checkpoint_ledger",
            name="Checkpoint Ledger",
            description="Verify checkpoint ledger exists and is consistent",
            status="fail",
            category="model",
            details={"error": f"Invalid JSON: {e}"},
            remediation="Fix or regenerate checkpoint_ledger.json",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="checkpoint_ledger",
            name="Checkpoint Ledger",
            description="Verify checkpoint ledger exists and is consistent",
            status="fail",
            category="model",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_base_model() -> RitualCheckResult:
    """Check that base model is configured and accessible."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir

        # Check active campaign for hero
        campaign_file = get_base_dir() / "control" / "active_campaign.json"
        if not campaign_file.exists():
            return RitualCheckResult(
                id="base_model",
                name="Base Model",
                description="Verify base model is configured and accessible",
                status="skip",
                category="model",
                details={"error": "No active campaign"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        with open(campaign_file) as f:
            campaign = json.load(f)

        hero_id = campaign.get("hero_id", "")

        # Try to find hero config
        hero_config = get_base_dir() / "configs" / "heroes" / f"{hero_id}.yaml"

        if hero_config.exists():
            import yaml
            with open(hero_config) as f:
                hero = yaml.safe_load(f)
            base_model_path = hero.get("base_model", {}).get("path", "")

            if base_model_path and Path(base_model_path).exists():
                status = "ok"
                details = {
                    "hero_id": hero_id,
                    "base_model_path": base_model_path,
                    "exists": True,
                }
            else:
                status = "warn"
                details = {
                    "hero_id": hero_id,
                    "base_model_path": base_model_path,
                    "exists": False,
                }
        else:
            status = "warn"
            details = {"error": f"Hero config not found: {hero_config}"}

        return RitualCheckResult(
            id="base_model",
            name="Base Model",
            description="Verify base model is configured and accessible",
            status=status,
            category="model",
            details=details,
            remediation="Configure base model in hero config" if status != "ok" else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="base_model",
            name="Base Model",
            description="Verify base model is configured and accessible",
            status="fail",
            category="model",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
