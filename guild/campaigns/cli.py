#!/usr/bin/env python3
"""
Campaign CLI - Manage training campaigns from the command line.

Commands:
    list        List all campaigns (or campaigns for a hero)
    active      Show the currently active campaign
    new         Create a new campaign
    switch      Switch to a different campaign
    archive     Archive a campaign to the Hall of Legends
    info        Show detailed info about a campaign

RPG Flavor (Pathfinder-inspired):
    The Campaign CLI is like the adventurer's journal - track your journeys,
    start new quests, and revisit old saves.

Usage:
    python3 -m guild.campaigns.cli list
    python3 -m guild.campaigns.cli active
    python3 -m guild.campaigns.cli new --hero titan-qwen3-4b --name "4B Binary Training"
    python3 -m guild.campaigns.cli switch dio-qwen3-0.6b campaign-001
    python3 -m guild.campaigns.cli info dio-qwen3-0.6b campaign-001
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from guild.heroes import get_hero, list_heroes, HeroNotFoundError


class CampaignCLI:
    """Command-line interface for campaign management."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = Path(base_dir)
        self.campaigns_dir = self.base_dir / "campaigns"
        self.control_dir = self.base_dir / "control"

    def list_campaigns(self, hero_id: Optional[str] = None) -> None:
        """List all campaigns, optionally filtered by hero."""
        print("\n" + "=" * 60)
        print("CAMPAIGN REGISTRY - All Playthroughs")
        print("=" * 60)

        if hero_id:
            heroes = [hero_id]
        else:
            # Find all hero directories
            heroes = [d.name for d in self.campaigns_dir.iterdir()
                     if d.is_dir() and not d.name.startswith(('.', 'archive'))]

        active = self._get_active_pointer()
        total_campaigns = 0

        for hid in sorted(heroes):
            hero_dir = self.campaigns_dir / hid
            if not hero_dir.exists():
                continue

            campaigns = [d.name for d in hero_dir.iterdir()
                        if d.is_dir() and d.name.startswith('campaign-')]

            if not campaigns:
                continue

            # Get hero display name
            try:
                hero = get_hero(hid)
                hero_display = f"{hero.name} ({hero.rpg_name})"
            except:
                hero_display = hid

            print(f"\n{hero_display} [{hid}]")
            print("-" * 40)

            for cid in sorted(campaigns):
                campaign_json = hero_dir / cid / "campaign.json"
                if campaign_json.exists():
                    with open(campaign_json) as f:
                        data = json.load(f)
                    name = data.get("name", cid)
                    step = data.get("current_step", 0)

                    # Check if this is the currently selected campaign
                    is_selected = (active and
                                active.get("hero_id") == hid and
                                active.get("campaign_id") == cid)
                    marker = " ★ SELECTED" if is_selected else ""

                    print(f"  {cid}: {name}{marker}")
                    print(f"    Step: {step:,}")
                    total_campaigns += 1

        print(f"\n{'=' * 60}")
        print(f"Total: {total_campaigns} campaign(s)")
        print()

    def show_active(self) -> None:
        """Show the currently active campaign."""
        active = self._get_active_pointer()
        if not active:
            print("\n⚠️  No active campaign!")
            print("   Use 'campaign switch <hero_id> <campaign_id>' to activate one.")
            return

        hero_id = active.get("hero_id")
        campaign_id = active.get("campaign_id")

        print("\n" + "=" * 60)
        print("ACTIVE CAMPAIGN - Current Playthrough")
        print("=" * 60)

        # Load campaign data
        campaign_path = self.campaigns_dir / hero_id / campaign_id / "campaign.json"
        if campaign_path.exists():
            with open(campaign_path) as f:
                data = json.load(f)

            # Load hero
            try:
                hero = get_hero(hero_id)
                print(f"\nHero: {hero.name} ({hero.rpg_name})")
                print(f"  Model: {hero.model.hf_name} ({hero.model.size_b}B)")
            except:
                print(f"\nHero: {hero_id}")

            print(f"\nCampaign: {data.get('name', campaign_id)}")
            print(f"  ID: {campaign_id}")
            print(f"  Status: {data.get('status', 'unknown')}")
            print(f"  Current Step: {data.get('current_step', 0):,}")
            print(f"  Created: {data.get('created_at', 'unknown')}")

            skills = data.get("skills_focus", [])
            if skills:
                print(f"  Skills Focus: {', '.join(skills)}")

            overrides = data.get("config_overrides", {})
            if overrides:
                print(f"  Config Overrides: {overrides}")

        print()

    def create_campaign(
        self,
        hero_id: str,
        name: str,
        starting_checkpoint: Optional[str] = None,
        skills_focus: Optional[list] = None,
        config_overrides: Optional[dict] = None,
        activate: bool = True,
    ) -> str:
        """Create a new campaign for a hero."""
        # Validate hero exists
        try:
            hero = get_hero(hero_id)
        except HeroNotFoundError as e:
            print(f"\n❌ {e}")
            print(f"   Available heroes: {list_heroes()}")
            return None

        # Find next campaign number
        hero_dir = self.campaigns_dir / hero_id
        hero_dir.mkdir(parents=True, exist_ok=True)

        existing = [d.name for d in hero_dir.iterdir()
                   if d.is_dir() and d.name.startswith('campaign-')]
        next_num = len(existing) + 1
        campaign_id = f"campaign-{next_num:03d}"

        # Create campaign directory structure
        campaign_dir = hero_dir / campaign_id
        (campaign_dir / "checkpoints").mkdir(parents=True)
        (campaign_dir / "status").mkdir()
        (campaign_dir / "logs").mkdir()

        # Parse starting checkpoint to extract lineage info
        parent_campaign = None
        starting_step = 0
        if starting_checkpoint:
            # Try to extract step number from checkpoint name
            # Formats: checkpoint-12345, checkpoint-12345-20251127-1430, /path/to/checkpoint-12345
            import re
            match = re.search(r'checkpoint-(\d+)', str(starting_checkpoint))
            if match:
                starting_step = int(match.group(1))

            # Try to find parent campaign from checkpoint path
            # Works even if checkpoint doesn't exist yet (parses path structure)
            checkpoint_str = str(starting_checkpoint)
            if 'campaigns/' in checkpoint_str or 'campaigns\\' in checkpoint_str:
                # Parse: campaigns/{hero_id}/{campaign_id}/checkpoints/checkpoint-*
                parts = checkpoint_str.replace('\\', '/').split('/')
                try:
                    idx = parts.index('campaigns')
                    if len(parts) > idx + 2:
                        parent_hero = parts[idx + 1]
                        parent_cid = parts[idx + 2]
                        parent_campaign = {
                            "hero_id": parent_hero,
                            "campaign_id": parent_cid,
                            "checkpoint": starting_checkpoint,
                            "step": starting_step,
                        }
                except (ValueError, IndexError):
                    pass

        # Create campaign.json
        campaign_data = {
            "id": campaign_id,
            "hero_id": hero_id,
            "name": name,
            "description": f"Campaign for {hero.name}",
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "starting_checkpoint": starting_checkpoint,
            "starting_step": starting_step,
            "current_step": starting_step,  # Start from parent step
            "total_examples": 0,
            "skills_focus": skills_focus or hero.skills_affinity,
            "config_overrides": config_overrides or {},
            "milestones": [],
            "archived_at": None,
            # Lineage tracking
            "parent_campaign": parent_campaign,
        }

        with open(campaign_dir / "campaign.json", "w") as f:
            json.dump(campaign_data, f, indent=2)

        # Initialize status files
        self._init_status_files(campaign_dir)

        print(f"\n✅ Created campaign: {campaign_id}")
        print(f"   Hero: {hero.name} ({hero_id})")
        print(f"   Name: {name}")
        print(f"   Path: {campaign_dir}")

        if activate:
            self._activate_campaign(hero_id, campaign_id)
            print(f"\n★ Campaign activated!")

        return campaign_id

    def switch_campaign(self, hero_id: str, campaign_id: str) -> None:
        """Switch to a different campaign."""
        campaign_path = self.campaigns_dir / hero_id / campaign_id
        if not campaign_path.exists():
            print(f"\n❌ Campaign not found: {hero_id}/{campaign_id}")
            return

        self._activate_campaign(hero_id, campaign_id)

        # Load and display info
        with open(campaign_path / "campaign.json") as f:
            data = json.load(f)

        print(f"\n★ Switched to campaign: {data.get('name', campaign_id)}")
        print(f"   Hero: {hero_id}")
        print(f"   Step: {data.get('current_step', 0):,}")

    def show_info(self, hero_id: str, campaign_id: str) -> None:
        """Show detailed info about a campaign."""
        campaign_path = self.campaigns_dir / hero_id / campaign_id
        if not campaign_path.exists():
            print(f"\n❌ Campaign not found: {hero_id}/{campaign_id}")
            return

        with open(campaign_path / "campaign.json") as f:
            data = json.load(f)

        print("\n" + "=" * 60)
        print(f"CAMPAIGN: {data.get('name', campaign_id)}")
        print("=" * 60)

        # Hero info
        try:
            hero = get_hero(hero_id)
            print(f"\nHero: {hero.name} ({hero.rpg_name})")
            print(f"  Model: {hero.model.hf_name}")
            print(f"  Size: {hero.model.size_b}B parameters")
            print(f"  Optimizer: {hero.training_defaults.optimizer}")
        except:
            print(f"\nHero: {hero_id}")

        # Campaign details
        print(f"\nCampaign Details:")
        print(f"  ID: {campaign_id}")
        print(f"  Status: {data.get('status', 'unknown')}")
        print(f"  Created: {data.get('created_at', 'unknown')}")
        print(f"  Current Step: {data.get('current_step', 0):,}")
        print(f"  Starting Checkpoint: {data.get('starting_checkpoint') or 'None (fresh)'}")

        # Lineage information
        parent = data.get('parent_campaign')
        if parent:
            print(f"\nLineage:")
            print(f"  Parent: {parent.get('hero_id')}/{parent.get('campaign_id')}")
            print(f"  Branched at: Step {parent.get('step', 0):,}")

        # Skills
        skills = data.get("skills_focus", [])
        print(f"\nSkills Focus: {', '.join(skills) if skills else 'None'}")

        # Config overrides
        overrides = data.get("config_overrides", {})
        if overrides:
            print(f"\nConfig Overrides:")
            for k, v in overrides.items():
                print(f"  {k}: {v}")

        # Milestones
        milestones = data.get("milestones", [])
        if milestones:
            print(f"\nMilestones:")
            for m in milestones[-5:]:  # Last 5
                print(f"  Step {m.get('step', '?'):,}: {m.get('note', '')}")

        # Checkpoints
        checkpoints_dir = campaign_path / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
            print(f"\nCheckpoints: {len(checkpoints)} saved")
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"  Latest: {latest.name}")

        print()

    def show_lineage(self, hero_id: str, campaign_id: str) -> None:
        """Show the lineage tree of a campaign."""
        campaign_path = self.campaigns_dir / hero_id / campaign_id
        if not campaign_path.exists():
            print(f"\n❌ Campaign not found: {hero_id}/{campaign_id}")
            return

        print("\n" + "=" * 60)
        print(f"LINEAGE TREE")
        print("=" * 60)

        # Build lineage chain
        chain = []
        current_hero = hero_id
        current_cid = campaign_id

        while current_hero and current_cid:
            path = self.campaigns_dir / current_hero / current_cid / "campaign.json"
            if not path.exists():
                break

            with open(path) as f:
                data = json.load(f)

            chain.append({
                "hero_id": current_hero,
                "campaign_id": current_cid,
                "name": data.get("name", current_cid),
                "step": data.get("current_step", 0),
                "created": data.get("created_at", "unknown"),
            })

            # Get parent
            parent = data.get("parent_campaign")
            if parent:
                current_hero = parent.get("hero_id")
                current_cid = parent.get("campaign_id")
            else:
                break

        # Print chain (reverse to show from root)
        chain.reverse()

        if not chain:
            print("\n  No lineage data available")
            return

        print()
        for i, entry in enumerate(chain):
            indent = "  " * i
            connector = "└── " if i > 0 else ""
            print(f"{indent}{connector}{entry['hero_id']}/{entry['campaign_id']}")
            print(f"{indent}    Name: {entry['name']}")
            print(f"{indent}    Step: {entry['step']:,}")

            if i < len(chain) - 1:
                print(f"{indent}    │")

        print()

    def archive_campaign(self, hero_id: str, campaign_id: str) -> None:
        """Archive a campaign to the Hall of Legends."""
        campaign_path = self.campaigns_dir / hero_id / campaign_id
        if not campaign_path.exists():
            print(f"\n❌ Campaign not found: {hero_id}/{campaign_id}")
            return

        # Check if it's active
        active = self._get_active_pointer()
        if (active and
            active.get("hero_id") == hero_id and
            active.get("campaign_id") == campaign_id):
            print(f"\n❌ Cannot archive active campaign!")
            print(f"   Switch to another campaign first.")
            return

        # Update status
        campaign_json = campaign_path / "campaign.json"
        with open(campaign_json) as f:
            data = json.load(f)
        data["status"] = "archived"
        data["archived_at"] = datetime.now().isoformat()
        with open(campaign_json, "w") as f:
            json.dump(data, f, indent=2)

        # Move to archive
        archive_dir = self.campaigns_dir / "archive" / hero_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        dest = archive_dir / f"{campaign_id}-archived"
        shutil.move(str(campaign_path), str(dest))

        print(f"\n📦 Archived campaign: {campaign_id}")
        print(f"   Moved to: {dest}")

    def _get_active_pointer(self) -> Optional[dict]:
        """Get active campaign pointer."""
        pointer_path = self.control_dir / "active_campaign.json"
        if pointer_path.exists():
            with open(pointer_path) as f:
                return json.load(f)
        return None

    def _activate_campaign(self, hero_id: str, campaign_id: str) -> None:
        """Set a campaign as active."""
        campaign_path = self.campaigns_dir / hero_id / campaign_id

        # Update pointer file
        pointer = {
            "hero_id": hero_id,
            "campaign_id": campaign_id,
            "campaign_path": f"campaigns/{hero_id}/{campaign_id}",
            "activated_at": datetime.now().isoformat(),
        }
        self.control_dir.mkdir(exist_ok=True)
        with open(self.control_dir / "active_campaign.json", "w") as f:
            json.dump(pointer, f, indent=2)

        # Update symlink
        active_link = self.campaigns_dir / "active"
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()
        active_link.symlink_to(campaign_path)

    def _init_status_files(self, campaign_dir: Path) -> None:
        """Initialize empty status files for a new campaign."""
        status_dir = campaign_dir / "status"

        # Training status
        with open(status_dir / "training_status.json", "w") as f:
            json.dump({
                "status": "idle",
                "current_step": 0,
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

        # Checkpoint ledger
        with open(status_dir / "checkpoint_ledger.json", "w") as f:
            json.dump({"checkpoints": []}, f, indent=2)

        # Curriculum state
        with open(status_dir / "curriculum_state.json", "w") as f:
            json.dump({"skills": {}}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Campaign Manager - Manage training campaigns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                           List all campaigns
  %(prog)s list dio-qwen3-0.6b            List campaigns for DIO
  %(prog)s active                         Show active campaign
  %(prog)s new --hero titan-qwen3-4b --name "4B Training"
  %(prog)s switch titan-qwen3-4b campaign-001
  %(prog)s info dio-qwen3-0.6b campaign-001
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list command
    list_parser = subparsers.add_parser("list", help="List campaigns")
    list_parser.add_argument("hero_id", nargs="?", help="Filter by hero ID")

    # active command
    subparsers.add_parser("active", help="Show active campaign")

    # new command
    new_parser = subparsers.add_parser("new", help="Create new campaign")
    new_parser.add_argument("--hero", required=True, help="Hero ID")
    new_parser.add_argument("--name", required=True, help="Campaign name")
    new_parser.add_argument("--checkpoint", help="Starting checkpoint")
    new_parser.add_argument("--skills", nargs="+", help="Skills to focus on")
    new_parser.add_argument("--no-activate", action="store_true",
                           help="Don't activate after creation")

    # switch command
    switch_parser = subparsers.add_parser("switch", help="Switch campaign")
    switch_parser.add_argument("hero_id", help="Hero ID")
    switch_parser.add_argument("campaign_id", help="Campaign ID")

    # info command
    info_parser = subparsers.add_parser("info", help="Show campaign info")
    info_parser.add_argument("hero_id", help="Hero ID")
    info_parser.add_argument("campaign_id", help="Campaign ID")

    # archive command
    archive_parser = subparsers.add_parser("archive", help="Archive campaign")
    archive_parser.add_argument("hero_id", help="Hero ID")
    archive_parser.add_argument("campaign_id", help="Campaign ID")

    # lineage command
    lineage_parser = subparsers.add_parser("lineage", help="Show campaign lineage tree")
    lineage_parser.add_argument("hero_id", help="Hero ID")
    lineage_parser.add_argument("campaign_id", help="Campaign ID")

    # heroes command (bonus)
    subparsers.add_parser("heroes", help="List available heroes")

    args = parser.parse_args()

    cli = CampaignCLI()

    if args.command == "list":
        cli.list_campaigns(args.hero_id)
    elif args.command == "active":
        cli.show_active()
    elif args.command == "new":
        cli.create_campaign(
            hero_id=args.hero,
            name=args.name,
            starting_checkpoint=args.checkpoint,
            skills_focus=args.skills,
            activate=not args.no_activate,
        )
    elif args.command == "switch":
        cli.switch_campaign(args.hero_id, args.campaign_id)
    elif args.command == "info":
        cli.show_info(args.hero_id, args.campaign_id)
    elif args.command == "archive":
        cli.archive_campaign(args.hero_id, args.campaign_id)
    elif args.command == "lineage":
        cli.show_lineage(args.hero_id, args.campaign_id)
    elif args.command == "heroes":
        print("\n" + "=" * 60)
        print("HERO ROSTER - Available Champions")
        print("=" * 60)
        for hid in list_heroes():
            try:
                hero = get_hero(hid)
                print(f"\n{hero.name} ({hero.rpg_name})")
                print(f"  ID: {hid}")
                print(f"  Model: {hero.model.hf_name} ({hero.model.size_b}B)")
                print(f"  Optimizer: {hero.training_defaults.optimizer}")
            except Exception as e:
                print(f"\n{hid}: Error loading - {e}")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
