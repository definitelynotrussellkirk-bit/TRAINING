"""
Heroes & Campaigns API

Extracted from tavern/server.py for better organization.
Handles:
- /api/hero - Active hero info
- /api/hero-config/{hero_id} - Hero config YAML as JSON
- /api/heroes - List all heroes
- /api/campaigns - List all campaigns
- /api/active-campaign - Get active campaign with peak tracking
- /api/hero-model-info - Merged hero + model info for settings
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core import paths

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_hero_info(handler: "TavernHandler"):
    """
    GET /api/hero - Active hero information.

    Returns hero data from the active campaign.
    """
    try:
        from core.hero import get_active_hero
        hero = get_active_hero()
        handler._send_json(hero)
    except Exception as e:
        logger.error(f"Hero info error: {e}")
        # Fallback to generic hero
        handler._send_json({
            "name": "Hero",
            "rpg_name": "The Apprentice",
            "icon": "🦸",
            "model_name": "Unknown",
            "hero_id": "",
            "campaign_id": "",
            "error": str(e)
        })


def serve_hero_config(handler: "TavernHandler", hero_id: str):
    """
    GET /api/hero-config/{hero_id} - Hero configuration.

    Returns the YAML config for a specific hero as JSON.
    """
    try:
        import yaml
        hero_file = paths.get_heroes_config_dir() / f"{hero_id}.yaml"

        if not hero_file.exists():
            handler._send_json({"error": f"Hero config not found: {hero_id}"}, 404)
            return

        with open(hero_file) as f:
            config = yaml.safe_load(f)

        handler._send_json(config)
    except Exception as e:
        logger.error(f"Hero config error for {hero_id}: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_heroes_data(handler: "TavernHandler"):
    """
    GET /api/heroes - List all available heroes.

    Returns list of hero configs with model info.
    """
    try:
        from guild.heroes import list_heroes, get_hero

        hero_ids = list_heroes()
        heroes = []
        for hero_id in hero_ids:
            hero = get_hero(hero_id)
            heroes.append({
                "id": hero.id,
                "name": hero.name,
                "rpg_name": hero.rpg_name,
                "description": hero.description,
                "model": {
                    "hf_name": hero.model.hf_name,
                    "family": hero.model.family,
                    "size_b": hero.model.size_b,
                },
                "display": {
                    "color": hero.display.color,
                    "emoji": hero.display.emoji,
                },
                "skills_affinity": hero.skills_affinity,
            })

        handler._send_json(heroes)

    except ImportError as e:
        logger.error(f"Hero module not available: {e}")
        handler._send_json({"error": "Hero system not installed"}, 503)
    except Exception as e:
        logger.error(f"Heroes data error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_campaigns_data(handler: "TavernHandler"):
    """
    GET /api/campaigns - List all campaigns.

    Returns campaigns grouped by hero, plus active campaign.
    """
    try:
        from guild.campaigns import CampaignManager

        base_dir = paths.get_base_dir()
        mgr = CampaignManager(base_dir)
        active = mgr.get_active()

        # Get all campaigns by hero
        campaigns = {}
        for hero_id in mgr.list_heroes():
            hero_campaigns = mgr.list_campaigns(hero_id, include_archived=True)
            campaigns[hero_id] = [c.to_dict() for c in hero_campaigns]

        result = {
            "active": active.to_dict() if active else None,
            "campaigns": campaigns,
        }

        handler._send_json(result)

    except ImportError as e:
        logger.error(f"Campaign module not available: {e}")
        handler._send_json({"error": "Campaign system not installed"}, 503)
    except Exception as e:
        logger.error(f"Campaigns data error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_active_campaign(handler: "TavernHandler"):
    """
    GET /api/active-campaign - Get the active campaign.

    Returns the active campaign with peak tracking fields:
    - peak_skill_levels: Highest skill levels achieved
    - peak_metrics: Best metrics (lowest_loss, highest_accuracy)
    - skill_effort: Cumulative effort per skill
    - journey_summary: One-line summary
    """
    try:
        from guild.campaigns import get_active_campaign

        base_dir = paths.get_base_dir()
        active = get_active_campaign(base_dir)

        if active:
            data = active.to_dict()
            # Ensure peak tracking fields are included
            data.setdefault("peak_skill_levels", {})
            data.setdefault("peak_metrics", {})
            data.setdefault("skill_effort", {})
            data.setdefault("level_transitions", [])
            data["journey_summary"] = active.journey_summary
            # Add effort summary if available
            try:
                data["effort_summary"] = active.get_effort_summary()
            except Exception:
                data["effort_summary"] = None
            handler._send_json(data)
        else:
            handler._send_json(None)

    except ImportError as e:
        logger.error(f"Campaign module not available: {e}")
        handler._send_json({"error": "Campaign system not installed"}, 503)
    except Exception as e:
        logger.error(f"Active campaign error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_hero_model_info(handler: "TavernHandler"):
    """
    GET /api/hero-model-info - Merged hero + model info for settings.

    Combines data from:
    - /api/hero (active hero)
    - /api/active-campaign (campaign info)
    - /api/hero-config/{hero_id} (model specs)

    Returns a single JSON with all the info the settings page needs.
    """
    try:
        import yaml
        from core.hero import get_active_hero
        from guild.campaigns import get_active_campaign

        base_dir = paths.get_base_dir()

        # Get active hero
        hero = {}
        try:
            hero = get_active_hero()
        except Exception as e:
            logger.warning(f"Could not get active hero: {e}")

        # Get active campaign
        campaign = None
        try:
            campaign = get_active_campaign(base_dir)
        except Exception as e:
            logger.warning(f"Could not get active campaign: {e}")

        # Get hero config YAML for model specs
        hero_config = {}
        hero_id = hero.get("hero_id")
        if hero_id:
            hero_file = paths.get_heroes_config_dir() / f"{hero_id}.yaml"
            if hero_file.exists():
                try:
                    with open(hero_file) as f:
                        hero_config = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Could not load hero config: {e}")

        # Extract model info
        model_config = hero_config.get("model", {})
        model_name = hero.get("model_name") or model_config.get("hf_name")
        architecture = model_config.get("architecture") or hero.get("model_family")
        context_length = model_config.get("context_length")
        vocab_size = model_config.get("vocab_size")

        # Build response
        result = {
            "hero_id": hero_id,
            "hero_name": hero.get("name") or hero.get("rpg_name"),
            "model_name": model_name,
            "architecture": architecture,
            "context_length": context_length,
            "vocab_size": vocab_size,
            "campaign_id": campaign.id if campaign else None,
            "campaign_path": str(campaign.path) if campaign else None,
            "campaign_name": campaign.name if campaign else None,
            # Peak tracking
            "peak_skill_levels": campaign.peak_skill_levels if campaign else {},
            "peak_metrics": campaign.peak_metrics if campaign else {},
            "journey_summary": campaign.journey_summary if campaign else None,
        }

        handler._send_json(result)

    except Exception as e:
        logger.error(f"Hero model info error: {e}")
        handler._send_json({"error": str(e)}, 500)
