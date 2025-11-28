"""
Guild Campaigns Module - Campaign management system.

Campaigns are training runs for heroes. Each campaign has:
- Its own checkpoints directory
- Its own status files (training_status, curriculum_state, etc.)
- Metadata (name, dates, milestones)
- Config overrides from hero defaults

Usage:
    from guild.campaigns import (
        CampaignManager,
        get_active_campaign,
        Campaign,
    )

    # Quick access to active campaign
    campaign = get_active_campaign()
    if campaign:
        print(f"Active: {campaign.hero_id}/{campaign.id}")
        print(f"Step: {campaign.current_step}")

    # Full manager operations
    mgr = CampaignManager()

    # Create new campaign
    campaign = mgr.create_campaign(
        hero_id="dio-qwen3-0.6b",
        name="Binary Focus",
        skills_focus=["bin"]
    )

    # Activate it
    mgr.activate(campaign)

    # Archive old campaign
    mgr.archive("dio-qwen3-0.6b", "campaign-001")

RPG Flavor:
    Every hero's journey is recorded in a Campaign - a saga of battles
    fought, levels gained, and skills mastered. The Campaign Manager
    is the Guild's Chronicler, keeping track of every adventure.
"""

from .types import (
    Campaign,
    CampaignStatus,
    Milestone,
    ActiveCampaignPointer,
)

from .manager import (
    CampaignManager,
    CampaignNotFoundError,
    CampaignExistsError,
    get_manager,
    get_active_campaign,
    list_all_campaigns,
)

__all__ = [
    # Types
    "Campaign",
    "CampaignStatus",
    "Milestone",
    "ActiveCampaignPointer",
    # Manager
    "CampaignManager",
    "CampaignNotFoundError",
    "CampaignExistsError",
    "get_manager",
    "get_active_campaign",
    "list_all_campaigns",
]
