"""
Setup API - First-run detection and onboarding status.

Handles:
- GET /api/setup/status - Detect first-run state and what steps are needed

Used by the UI to show appropriate welcome/onboarding experience.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List

from core.paths import get_base_dir
from guild.campaigns import CampaignManager
from guild.integration.queue_adapter import get_queue_adapter

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def get_setup_status() -> Dict[str, Any]:
    """
    Check system state for first-run detection.

    Returns dict with:
        is_first_run: True if truly fresh install (no campaigns)
        has_campaigns: Whether any campaigns exist
        has_active_campaign: Whether a campaign is currently active
        has_queue_data: Whether training queue has data
        has_models: Whether any models exist
        next_step: The next action needed
        steps_completed: List of completed setup steps
        steps_remaining: List of remaining setup steps
    """
    base_dir = get_base_dir()

    # Check campaigns
    try:
        mgr = CampaignManager(base_dir)
        heroes = mgr.list_heroes()
        has_campaigns = len(heroes) > 0
        active = mgr.get_active()
        has_active_campaign = active is not None
    except Exception as e:
        logger.warning(f"Error checking campaigns: {e}")
        has_campaigns = False
        has_active_campaign = False

    # Check queue
    try:
        queue = get_queue_adapter()
        status = queue.get_status()
        if status.success:
            has_queue_data = not status.data.is_empty
        else:
            has_queue_data = False
    except Exception as e:
        logger.warning(f"Error checking queue: {e}")
        has_queue_data = False

    # Check models directory
    models_dir = base_dir / "models"
    has_models = False
    if models_dir.exists():
        # Look for actual model directories (not just empty dir)
        for item in models_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                has_models = True
                break

    # Determine steps
    steps_completed = []
    steps_remaining = []

    # Step 1: Models available
    if has_models:
        steps_completed.append("models_available")
    else:
        steps_remaining.append("download_model")

    # Step 2: Campaign exists
    if has_campaigns:
        steps_completed.append("campaign_created")
    else:
        steps_remaining.append("create_campaign")

    # Step 3: Campaign active
    if has_active_campaign:
        steps_completed.append("campaign_active")
    elif has_campaigns:
        steps_remaining.append("activate_campaign")

    # Step 4: Queue has data
    if has_queue_data:
        steps_completed.append("queue_has_data")
    else:
        steps_remaining.append("generate_data")

    # Determine next step
    if not has_campaigns:
        next_step = "create_campaign"
    elif not has_active_campaign:
        next_step = "activate_campaign"
    elif not has_queue_data:
        next_step = "generate_data"
    else:
        next_step = "ready_to_train"

    # Is this a true first run?
    is_first_run = not has_campaigns

    return {
        "is_first_run": is_first_run,
        "has_campaigns": has_campaigns,
        "has_active_campaign": has_active_campaign,
        "has_queue_data": has_queue_data,
        "has_models": has_models,
        "next_step": next_step,
        "steps_completed": steps_completed,
        "steps_remaining": steps_remaining,
        "active_campaign": {
            "hero_id": active.hero_id if active else None,
            "campaign_id": active.id if active else None,
            "name": active.name if active else None,
        } if has_active_campaign else None,
    }


def serve_setup_status(handler: "TavernHandler"):
    """
    GET /api/setup/status - Get first-run and setup status.

    Returns:
        {
            "is_first_run": true/false,
            "has_campaigns": true/false,
            "has_active_campaign": true/false,
            "has_queue_data": true/false,
            "next_step": "create_campaign" | "activate_campaign" | "generate_data" | "ready_to_train",
            "steps_completed": [...],
            "steps_remaining": [...]
        }
    """
    try:
        status = get_setup_status()
        handler._send_json(status)
    except Exception as e:
        logger.error(f"Setup status error: {e}")
        handler._send_json({
            "error": str(e),
            "is_first_run": True,
            "next_step": "unknown",
        }, 500)


def serve_quick_start(handler: "TavernHandler"):
    """
    POST /api/setup/quick-start - One-click setup for new users.

    Creates a default campaign, generates initial training data, and optionally starts daemon.

    Body (optional):
        {
            "hero_id": "dio-qwen3-0.6b",  // defaults to first available
            "campaign_name": "My First Campaign",
            "examples_count": 1000,
            "start_daemon": true           // also start the training daemon
        }
    """
    try:
        import json
        import subprocess
        from guild.skills import get_engine

        base_dir = get_base_dir()

        # Parse optional body
        content_length = int(handler.headers.get("Content-Length", 0))
        body = {}
        if content_length:
            body = json.loads(handler.rfile.read(content_length).decode())

        hero_id = body.get("hero_id")
        campaign_name = body.get("campaign_name", "First Adventure")
        examples_count = int(body.get("examples_count", 1000))
        start_daemon = body.get("start_daemon", True)  # Default to True for true one-click

        # Get campaign manager
        mgr = CampaignManager(base_dir)

        # Check if already have active campaign
        active = mgr.get_active()
        if active:
            handler._send_json({
                "ok": True,
                "message": "Already have an active campaign",
                "campaign_id": active.id,
                "hero_id": active.hero_id,
                "skipped": True,
            })
            return

        # Find a hero if not specified
        if not hero_id:
            heroes = mgr.list_heroes()
            if heroes:
                hero_id = heroes[0]
            else:
                # No heroes exist - need to detect available models
                models_dir = base_dir / "models"
                if models_dir.exists():
                    for item in models_dir.iterdir():
                        if item.is_dir() and (item / "config.json").exists():
                            hero_id = item.name
                            break

                if not hero_id:
                    handler._send_json({
                        "ok": False,
                        "error": "No models found. Please download a model first.",
                    }, 400)
                    return

        # Check if hero has campaigns, create one if not
        existing = mgr.list_campaigns(hero_id)
        if existing:
            campaign = existing[0]
            logger.info(f"[Quick Start] Using existing campaign: {campaign.id}")
        else:
            # Create new campaign
            campaign = mgr.create_campaign(
                hero_id=hero_id,
                name=campaign_name,
            )
            logger.info(f"[Quick Start] Created campaign: {campaign.id}")

        # Activate it
        mgr.activate(campaign)
        logger.info(f"[Quick Start] Activated campaign: {campaign.id}")

        # Generate training data
        engine = get_engine()
        skills = engine.list_skills()

        generated_files = []
        if skills:
            # Generate for first skill
            skill_id = skills[0]
            skill = engine.get(skill_id)
            state = engine.get_state(skill_id)

            training_data = skill.generate_training_batch(
                level=state.level,
                count=examples_count,
            )

            if training_data:
                # Queue it
                queue = get_queue_adapter()
                from datetime import datetime

                jsonl_lines = [json.dumps(ex) for ex in training_data]
                content = "\n".join(jsonl_lines)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"quickstart_{skill_id}_L{state.level}_{examples_count}_{timestamp}.jsonl"

                result = queue.submit_content(
                    content=content,
                    filename=filename,
                    priority="normal",
                )

                if result.success:
                    generated_files.append({
                        "skill": skill_id,
                        "level": state.level,
                        "count": len(training_data),
                        "file": filename,
                    })
                    logger.info(f"[Quick Start] Generated {len(training_data)} examples → {filename}")

        # Start daemon if requested
        daemon_started = False
        if start_daemon:
            try:
                import os
                from core.momentum import get_daemon_status

                daemon = get_daemon_status()
                if not daemon["running"]:
                    # Start the daemon
                    daemon_script = base_dir / "core" / "training_daemon.py"
                    log_file = base_dir / "logs" / "training_daemon.log"
                    log_file.parent.mkdir(parents=True, exist_ok=True)

                    # Set PYTHONPATH so daemon can import trainer, guild, etc.
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(base_dir)

                    subprocess.Popen(
                        ["nohup", "python3", str(daemon_script)],
                        stdout=open(log_file, "a"),
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        cwd=str(base_dir),
                        env=env
                    )
                    daemon_started = True
                    logger.info("[Quick Start] Started training daemon")
                else:
                    logger.info("[Quick Start] Daemon already running")
            except Exception as e:
                logger.warning(f"[Quick Start] Failed to start daemon: {e}")

        handler._send_json({
            "ok": True,
            "message": "Quick start complete!",
            "campaign_id": campaign.id,
            "hero_id": hero_id,
            "generated": generated_files,
            "daemon_started": daemon_started,
        })

    except Exception as e:
        logger.error(f"Quick start error: {e}")
        handler._send_json({
            "ok": False,
            "error": str(e),
        }, 500)
