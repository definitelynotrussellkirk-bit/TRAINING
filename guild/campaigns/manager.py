"""
Campaign Manager - Create, switch, and archive campaigns.

The Campaign Manager handles all campaign lifecycle operations:
- Creating new campaigns for heroes
- Switching between campaigns
- Archiving old campaigns to the Hall of Legends
- Querying campaign status and history

Usage:
    from guild.campaigns import CampaignManager, get_active_campaign

    # Get active campaign (convenience function)
    campaign = get_active_campaign()
    if campaign:
        print(f"Active: {campaign.hero_id}/{campaign.id}")

    # Use manager for full operations
    mgr = CampaignManager(base_dir)

    # List all campaigns for a hero
    campaigns = mgr.list_campaigns("dio-qwen3-0.6b")

    # Create new campaign
    campaign = mgr.create_campaign(
        hero_id="dio-qwen3-0.6b",
        name="Fresh Start",
        skills_focus=["bin", "sy"]
    )

    # Switch to it
    mgr.activate(campaign)

    # Archive old campaign
    mgr.archive("dio-qwen3-0.6b", "campaign-001")

RPG Flavor:
    The Campaign Manager is the Guild's Chronicler - recording every
    adventure, knowing where each hero's journey stands, and preserving
    the tales of past campaigns in the Hall of Legends.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .types import Campaign, ActiveCampaignPointer, CampaignStatus

logger = logging.getLogger("campaign_manager")


class CampaignNotFoundError(Exception):
    """Raised when a campaign is not found."""
    pass


class CampaignExistsError(Exception):
    """Raised when trying to create a campaign that already exists."""
    pass


class CampaignManager:
    """
    Manages campaign lifecycle operations.

    Handles creating, activating, archiving, and querying campaigns.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the campaign manager.

        Args:
            base_dir: Base directory (containing campaigns/, control/, etc.)
                      If None, auto-detects from file location.
        """
        if base_dir is None:
            # Auto-detect: guild/campaigns/manager.py -> TRAINING/
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = Path(base_dir)
        self.campaigns_dir = self.base_dir / "campaigns"
        self.archive_dir = self.campaigns_dir / "archive"
        self.control_dir = self.base_dir / "control"
        self.pointer_path = self.control_dir / "active_campaign.json"

        # Ensure directories exist
        self.campaigns_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.control_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_active(self) -> Optional[Campaign]:
        """
        Get the currently active campaign.

        Returns:
            Campaign instance or None if no active campaign
        """
        pointer = self._load_pointer()
        if pointer is None:
            return None

        try:
            return self.get_campaign(pointer.hero_id, pointer.campaign_id)
        except CampaignNotFoundError:
            logger.warning(
                f"Active campaign pointer references missing campaign: "
                f"{pointer.hero_id}/{pointer.campaign_id}"
            )
            return None

    def list_heroes(self) -> List[str]:
        """
        List all heroes that have campaigns.

        Returns:
            List of hero IDs with at least one campaign
        """
        heroes = []
        for item in self.campaigns_dir.iterdir():
            if item.is_dir() and not item.name.startswith((".", "archive")):
                # Check if it has any campaigns
                campaigns = list(item.glob("campaign-*"))
                if campaigns:
                    heroes.append(item.name)
        return sorted(heroes)

    def list_campaigns(self, hero_id: str, include_archived: bool = False) -> List[Campaign]:
        """
        List all campaigns for a hero.

        Args:
            hero_id: Hero identifier
            include_archived: Include archived campaigns

        Returns:
            List of Campaign instances
        """
        campaigns = []

        # Active campaigns
        hero_dir = self.campaigns_dir / hero_id
        if hero_dir.exists():
            for campaign_dir in hero_dir.glob("campaign-*"):
                if campaign_dir.is_dir():
                    try:
                        campaigns.append(self._load_campaign(campaign_dir))
                    except Exception as e:
                        logger.warning(f"Error loading campaign {campaign_dir}: {e}")

        # Archived campaigns
        if include_archived:
            archive_hero_dir = self.archive_dir / hero_id
            if archive_hero_dir.exists():
                for campaign_dir in archive_hero_dir.glob("campaign-*"):
                    if campaign_dir.is_dir():
                        try:
                            campaigns.append(self._load_campaign(campaign_dir))
                        except Exception as e:
                            logger.warning(f"Error loading archived campaign {campaign_dir}: {e}")

        # Sort by creation date
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        return campaigns

    def get_campaign(self, hero_id: str, campaign_id: str) -> Campaign:
        """
        Get a specific campaign.

        Args:
            hero_id: Hero identifier
            campaign_id: Campaign identifier

        Returns:
            Campaign instance

        Raises:
            CampaignNotFoundError: If campaign doesn't exist
        """
        # Check active campaigns
        campaign_dir = self.campaigns_dir / hero_id / campaign_id
        if campaign_dir.exists():
            return self._load_campaign(campaign_dir)

        # Check archive
        archive_dir = self.archive_dir / hero_id / campaign_id
        if archive_dir.exists():
            return self._load_campaign(archive_dir)

        raise CampaignNotFoundError(
            f"Campaign not found: {hero_id}/{campaign_id}"
        )

    # -------------------------------------------------------------------------
    # Create Operations
    # -------------------------------------------------------------------------

    def create_campaign(
        self,
        hero_id: str,
        name: str,
        description: str = "",
        skills_focus: Optional[List[str]] = None,
        config_overrides: Optional[Dict] = None,
        starting_checkpoint: Optional[str] = None,
    ) -> Campaign:
        """
        Create a new campaign for a hero.

        Args:
            hero_id: Hero identifier
            name: Display name for the campaign
            description: Optional description
            skills_focus: List of skill IDs to focus on
            config_overrides: Overrides from hero training defaults
            starting_checkpoint: Path to checkpoint to start from (None = base model)

        Returns:
            New Campaign instance

        Raises:
            CampaignExistsError: If campaign ID already exists
        """
        # Generate campaign ID
        existing = self.list_campaigns(hero_id)
        next_num = len(existing) + 1
        campaign_id = f"campaign-{next_num:03d}"

        # Check if exists (shouldn't happen but be safe)
        campaign_dir = self.campaigns_dir / hero_id / campaign_id
        if campaign_dir.exists():
            raise CampaignExistsError(f"Campaign already exists: {campaign_id}")

        # Create directory structure
        campaign_dir.mkdir(parents=True)
        (campaign_dir / "checkpoints").mkdir()
        (campaign_dir / "status").mkdir()
        (campaign_dir / "logs").mkdir()

        # Create campaign
        campaign = Campaign(
            id=campaign_id,
            hero_id=hero_id,
            name=name,
            path=campaign_dir,
            description=description,
            skills_focus=skills_focus or [],
            config_overrides=config_overrides or {},
            starting_checkpoint=starting_checkpoint,
            starting_step=0,
            current_step=0,
        )

        # Save metadata
        campaign.save()

        logger.info(f"Created campaign: {hero_id}/{campaign_id}")
        return campaign

    # -------------------------------------------------------------------------
    # Activation Operations
    # -------------------------------------------------------------------------

    # Status files that should be symlinked to active campaign
    STATUS_FILES = [
        "training_status.json",
        "curriculum_state.json",
        "checkpoint_ledger.json",
        "eval_results_history.json",
        "evaluation_ledger.json",
    ]

    def _is_training_active(self) -> bool:
        """Check if training is currently running."""
        state_file = self.control_dir / "state.json"
        if not state_file.exists():
            return False
        try:
            with open(state_file) as f:
                state = json.load(f)
            return state.get("status") == "training"
        except:
            return False

    def _pause_training(self, reason: str = "Campaign switch") -> bool:
        """
        Signal training to pause and wait for it.

        Returns:
            True if training paused (or wasn't running), False on timeout
        """
        if not self._is_training_active():
            logger.info("Training not active, no need to pause")
            return True

        # Create pause signal
        pause_file = self.control_dir / ".pause"
        pause_data = {
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        pause_file.write_text(json.dumps(pause_data))
        logger.info(f"Pause signal sent: {reason}")

        # Wait for training to pause (up to 60 seconds)
        state_file = self.control_dir / "state.json"
        for i in range(60):
            time.sleep(1)
            try:
                with open(state_file) as f:
                    state = json.load(f)
                if state.get("status") in ("paused", "idle", "stopped"):
                    logger.info(f"Training paused after {i+1}s")
                    return True
            except:
                pass

        logger.warning("Timeout waiting for training to pause")
        return False

    def _resume_training(self) -> None:
        """Signal training to resume."""
        resume_file = self.control_dir / ".resume"
        resume_file.touch()
        logger.info("Resume signal sent")

        # Clean up pause file
        pause_file = self.control_dir / ".pause"
        pause_file.unlink(missing_ok=True)

    def activate(self, campaign: Campaign, safe: bool = True) -> None:
        """
        Make a campaign the active one.

        This updates:
        - Active campaign pointer file
        - campaigns/active symlink
        - status/* symlinks to point to campaign's status/

        Args:
            campaign: Campaign to activate
            safe: If True, pause training during switch (recommended)
        """
        was_training = self._is_training_active()

        # Pause training if active
        if safe and was_training:
            logger.info("Training is active - pausing for safe switch...")
            if not self._pause_training("Campaign switch"):
                logger.warning("Could not pause training - switching anyway (may cause issues)")

        try:
            # Create pointer
            pointer = ActiveCampaignPointer(
                hero_id=campaign.hero_id,
                campaign_id=campaign.id,
                campaign_path=str(campaign.path.relative_to(self.base_dir)),
            )

            # Save pointer
            self._save_pointer(pointer)

            # Update symlink
            active_link = self.campaigns_dir / "active"
            if active_link.exists() or active_link.is_symlink():
                active_link.unlink()
            active_link.symlink_to(campaign.path)

            # Update status file symlinks
            self._update_status_symlinks(campaign)

            # Update campaign status
            campaign.status = CampaignStatus.ACTIVE
            campaign.save()

            logger.info(f"Activated campaign: {campaign.hero_id}/{campaign.id}")

        finally:
            # Resume training if it was active
            if safe and was_training:
                logger.info("Resuming training...")
                self._resume_training()

    def _update_status_symlinks(self, campaign: Campaign) -> None:
        """
        Update status/ symlinks to point to the campaign's status directory.

        For each status file:
        - If it's a regular file in status/, copy to campaign and replace with symlink
        - If it's already a symlink, update to point to campaign
        - If campaign doesn't have the file, create empty one
        """
        status_dir = self.base_dir / "status"
        campaign_status_dir = campaign.path / "status"
        campaign_status_dir.mkdir(parents=True, exist_ok=True)

        for filename in self.STATUS_FILES:
            status_file = status_dir / filename
            campaign_file = campaign_status_dir / filename

            # Ensure campaign has the file
            if not campaign_file.exists():
                # If original exists (as regular file), copy it
                if status_file.exists() and not status_file.is_symlink():
                    shutil.copy2(status_file, campaign_file)
                    logger.info(f"Copied {filename} to campaign")
                else:
                    # Create empty JSON file
                    campaign_file.write_text("{}")
                    logger.info(f"Created empty {filename} in campaign")

            # Remove old status file/symlink
            if status_file.exists() or status_file.is_symlink():
                status_file.unlink()

            # Create symlink to campaign
            status_file.symlink_to(campaign_file)
            logger.debug(f"Symlinked {filename} -> {campaign_file}")

    def deactivate(self) -> None:
        """
        Deactivate the current campaign (remove pointer).

        This doesn't archive - just clears the active pointer.
        """
        if self.pointer_path.exists():
            self.pointer_path.unlink()

        active_link = self.campaigns_dir / "active"
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()

        logger.info("Deactivated active campaign")

    # -------------------------------------------------------------------------
    # Archive Operations
    # -------------------------------------------------------------------------

    def archive(self, hero_id: str, campaign_id: str) -> Campaign:
        """
        Move a campaign to the Hall of Legends (archive).

        Args:
            hero_id: Hero identifier
            campaign_id: Campaign identifier

        Returns:
            Archived Campaign instance

        Raises:
            CampaignNotFoundError: If campaign doesn't exist
        """
        src_dir = self.campaigns_dir / hero_id / campaign_id
        if not src_dir.exists():
            raise CampaignNotFoundError(f"Campaign not found: {hero_id}/{campaign_id}")

        # Load campaign
        campaign = self._load_campaign(src_dir)

        # Check if it's the active campaign
        active = self.get_active()
        if active and active.id == campaign_id and active.hero_id == hero_id:
            self.deactivate()

        # Update metadata
        campaign.status = CampaignStatus.ARCHIVED
        campaign.archived_at = datetime.now().isoformat()
        campaign.save()

        # Move to archive
        dst_dir = self.archive_dir / hero_id / campaign_id
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_dir), str(dst_dir))

        # Update campaign path
        campaign.path = dst_dir

        logger.info(f"Archived campaign: {hero_id}/{campaign_id}")
        return campaign

    def restore(self, hero_id: str, campaign_id: str) -> Campaign:
        """
        Restore a campaign from the archive.

        Args:
            hero_id: Hero identifier
            campaign_id: Campaign identifier

        Returns:
            Restored Campaign instance

        Raises:
            CampaignNotFoundError: If campaign not in archive
        """
        src_dir = self.archive_dir / hero_id / campaign_id
        if not src_dir.exists():
            raise CampaignNotFoundError(
                f"Archived campaign not found: {hero_id}/{campaign_id}"
            )

        # Load campaign
        campaign = self._load_campaign(src_dir)

        # Update metadata
        campaign.status = CampaignStatus.PAUSED  # Restored but not active
        campaign.archived_at = None
        campaign.save()

        # Move back from archive
        dst_dir = self.campaigns_dir / hero_id / campaign_id
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_dir), str(dst_dir))

        # Update campaign path
        campaign.path = dst_dir

        logger.info(f"Restored campaign: {hero_id}/{campaign_id}")
        return campaign

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _load_campaign(self, campaign_dir: Path) -> Campaign:
        """Load campaign from directory."""
        config_path = campaign_dir / "campaign.json"
        if not config_path.exists():
            raise CampaignNotFoundError(
                f"No campaign.json in {campaign_dir}"
            )

        with open(config_path) as f:
            data = json.load(f)

        return Campaign.from_dict(data, campaign_dir)

    def _load_pointer(self) -> Optional[ActiveCampaignPointer]:
        """Load active campaign pointer."""
        if not self.pointer_path.exists():
            return None

        with open(self.pointer_path) as f:
            data = json.load(f)

        return ActiveCampaignPointer.from_dict(data)

    def _save_pointer(self, pointer: ActiveCampaignPointer) -> None:
        """Save active campaign pointer."""
        with open(self.pointer_path, "w") as f:
            json.dump(pointer.to_dict(), f, indent=2)


# Module-level singleton
_manager: Optional[CampaignManager] = None


def get_manager(base_dir: Optional[Path] = None) -> CampaignManager:
    """Get the global campaign manager singleton."""
    global _manager
    if _manager is None:
        _manager = CampaignManager(base_dir)
    return _manager


def get_active_campaign(base_dir: Optional[Path] = None) -> Optional[Campaign]:
    """
    Get the currently active campaign.

    Convenience function for quick access to active campaign.

    Args:
        base_dir: Optional base directory override

    Returns:
        Active Campaign or None
    """
    return get_manager(base_dir).get_active()


def list_all_campaigns(base_dir: Optional[Path] = None) -> Dict[str, List[Campaign]]:
    """
    List all campaigns grouped by hero.

    Returns:
        Dict mapping hero_id -> list of campaigns
    """
    mgr = get_manager(base_dir)
    result = {}
    for hero_id in mgr.list_heroes():
        result[hero_id] = mgr.list_campaigns(hero_id, include_archived=True)
    return result
