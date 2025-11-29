"""
Campaign Loader - Safe helpers for loading campaigns in trainer/worker contexts.

This module provides defensive loading that:
- Never throws (logs warnings and returns None)
- Is safe to call from both trainer and tavern
- Caches manager to avoid repeated initialization

Usage:
    from guild.campaigns.loader import load_active_campaign

    campaign = load_active_campaign()
    if campaign:
        campaign.update_peak_skill("bin", 5)
        campaign.update_peak_metric("lowest_loss", 0.82, lower_is_better=True)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cached manager to avoid re-init
_cached_manager = None


def _get_manager():
    """Get or create campaign manager (cached)."""
    global _cached_manager
    if _cached_manager is not None:
        return _cached_manager

    try:
        from guild.campaigns import CampaignManager
        from core.paths import get_base_dir

        base_dir = get_base_dir()
        _cached_manager = CampaignManager(base_dir)
        return _cached_manager
    except Exception as e:
        logger.warning(f"Failed to create campaign manager: {e}")
        return None


def load_active_campaign():
    """
    Load the currently active campaign object, or None if none configured.

    Safe to call from both trainer and tavern. Never throws.

    Returns:
        Campaign instance or None
    """
    try:
        mgr = _get_manager()
        if mgr is None:
            return None

        return mgr.get_active()
    except Exception as e:
        logger.warning(f"Failed to load active campaign: {e}")
        return None


def reset_loader_cache():
    """Reset the cached manager (for testing)."""
    global _cached_manager
    _cached_manager = None
