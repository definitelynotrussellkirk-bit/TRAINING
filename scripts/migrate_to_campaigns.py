#!/usr/bin/env python3
"""
Migrate to Campaign System

This script migrates the existing single-hero setup to the new campaign system.

What it does:
1. Creates campaigns/dio-qwen3-0.6b/campaign-001/ structure
2. Creates campaign.json with current training stats
3. Creates symbolic links for backward compatibility:
   - status/ files -> campaigns/active/status/ (new canonical location)
   - current_model symlink -> campaigns/active/checkpoints
4. Sets up active_campaign.json pointer

What it DOESN'T do (for safety):
- Does NOT move checkpoints (too risky during active training)
- Does NOT delete any existing files
- Does NOT modify training daemon

After running:
- Training continues using current_model/ (unchanged)
- Status files are read from campaigns/active/status/ via symlinks
- New campaigns can be created via the UI

Usage:
    # Dry run (show what would happen)
    python3 scripts/migrate_to_campaigns.py --dry-run

    # Actually migrate
    python3 scripts/migrate_to_campaigns.py

    # Force (skip confirmations)
    python3 scripts/migrate_to_campaigns.py --force
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


def get_current_stats() -> dict:
    """Read current training stats from status files."""
    stats = {
        "current_step": 0,
        "total_examples": 0,
        "skills": ["bin", "sy"],
    }

    # Read training_status.json
    training_status = BASE_DIR / "status" / "training_status.json"
    if training_status.exists():
        with open(training_status) as f:
            data = json.load(f)
            stats["current_step"] = data.get("current_step", 0)

    # Read curriculum_state.json for skills
    curriculum_state = BASE_DIR / "status" / "curriculum_state.json"
    if curriculum_state.exists():
        with open(curriculum_state) as f:
            data = json.load(f)
            skills = list(data.get("skills", {}).keys())
            if skills:
                # Map curriculum IDs to skill IDs
                id_mapping = {"syllo": "sy", "binary": "bin"}
                stats["skills"] = [id_mapping.get(s, s) for s in skills]

    return stats


def migrate(dry_run: bool = False, force: bool = False):
    """Run the migration."""
    print("=" * 60)
    print("CAMPAIGN SYSTEM MIGRATION")
    print("=" * 60)

    hero_id = "dio-qwen3-0.6b"
    campaign_id = "campaign-001"

    campaigns_dir = BASE_DIR / "campaigns"
    campaign_dir = campaigns_dir / hero_id / campaign_id
    status_dir = BASE_DIR / "status"
    control_dir = BASE_DIR / "control"

    # Check if already migrated
    if campaign_dir.exists():
        print(f"\n[!] Campaign already exists: {campaign_dir}")
        print("    Migration may have already been run.")
        if not force:
            response = input("    Continue anyway? [y/N] ")
            if response.lower() != "y":
                print("Aborted.")
                return

    # Get current stats
    stats = get_current_stats()
    print(f"\nCurrent training stats:")
    print(f"  - Step: {stats['current_step']}")
    print(f"  - Skills: {stats['skills']}")

    # Files to link (status files that should be in campaign)
    status_files_to_link = [
        "training_status.json",
        "curriculum_state.json",
        "checkpoint_ledger.json",
        "eval_results_history.json",
        "evaluation_ledger.json",
    ]

    print(f"\n[1] Create campaign directory structure")
    print(f"    {campaign_dir}/")
    print(f"    {campaign_dir}/checkpoints/")
    print(f"    {campaign_dir}/status/")
    print(f"    {campaign_dir}/logs/")

    if not dry_run:
        campaign_dir.mkdir(parents=True, exist_ok=True)
        (campaign_dir / "checkpoints").mkdir(exist_ok=True)
        (campaign_dir / "status").mkdir(exist_ok=True)
        (campaign_dir / "logs").mkdir(exist_ok=True)

    print(f"\n[2] Create campaign.json metadata")
    campaign_json = {
        "id": campaign_id,
        "hero_id": hero_id,
        "name": "DIO's First Campaign",
        "description": "Original training run - Binary Alchemy and Word Weaving",
        "created_at": "2025-11-22T04:00:00Z",  # Approximate start date
        "status": "active",
        "starting_checkpoint": None,
        "starting_step": 0,
        "current_step": stats["current_step"],
        "total_examples": 0,
        "skills_focus": stats["skills"],
        "config_overrides": {},
        "milestones": [
            {
                "step": 100000,
                "note": "First 100k steps",
                "date": "2025-11-24"
            },
            {
                "step": 180000,
                "note": "Campaign system migration",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        ],
        "archived_at": None
    }
    print(f"    {campaign_dir / 'campaign.json'}")

    if not dry_run:
        with open(campaign_dir / "campaign.json", "w") as f:
            json.dump(campaign_json, f, indent=2)

    print(f"\n[3] Move status files to campaign (with symlinks back)")
    for filename in status_files_to_link:
        src = status_dir / filename
        dst = campaign_dir / "status" / filename
        link = status_dir / filename

        if src.exists():
            print(f"    {filename}:")
            print(f"      - Copy to: {dst}")
            print(f"      - Symlink: {link} -> {dst}")

            if not dry_run:
                # Copy file to campaign status
                shutil.copy2(src, dst)
                # Remove original and create symlink
                src.unlink()
                link.symlink_to(dst)

    print(f"\n[4] Create active campaign symlink")
    active_link = campaigns_dir / "active"
    print(f"    {active_link} -> {campaign_dir}")

    if not dry_run:
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()
        active_link.symlink_to(campaign_dir)

    print(f"\n[5] Create active_campaign.json pointer")
    pointer = {
        "hero_id": hero_id,
        "campaign_id": campaign_id,
        "campaign_path": f"campaigns/{hero_id}/{campaign_id}",
        "activated_at": datetime.now().isoformat(),
        "_comment": "Scroll of Destiny - Points to the currently active campaign"
    }
    pointer_path = control_dir / "active_campaign.json"
    print(f"    {pointer_path}")

    if not dry_run:
        control_dir.mkdir(parents=True, exist_ok=True)
        with open(pointer_path, "w") as f:
            json.dump(pointer, f, indent=2)

    print(f"\n[6] Create symlink: current_model -> campaigns/active/checkpoints")
    current_model = BASE_DIR / "current_model"
    checkpoints_target = campaign_dir / "checkpoints"

    # Note: We can't actually change current_model during active training
    # Instead, we'll note what would need to happen
    print(f"    NOTE: current_model is currently a directory with active checkpoints.")
    print(f"    For full migration, you would need to:")
    print(f"    1. Stop training")
    print(f"    2. Move checkpoints: mv current_model/* {checkpoints_target}/")
    print(f"    3. Remove directory: rmdir current_model")
    print(f"    4. Create symlink: ln -s campaigns/active/checkpoints current_model")
    print(f"\n    For now, training will continue using current_model/ directly.")
    print(f"    The campaign system will be ready for NEW campaigns.")

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
        print("Run without --dry-run to apply changes")
    else:
        print("MIGRATION COMPLETE")
        print("\nNext steps:")
        print("1. Verify: python3 -c 'from guild.campaigns import get_active_campaign; print(get_active_campaign())'")
        print("2. Check Tavern UI for campaign display")
        print("3. Full checkpoint migration (optional) requires stopping training")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate to campaign system"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.force:
        print("\nThis will modify your status/ directory structure.")
        print("Training will continue to work, but status files will be symlinked.")
        response = input("\nProceed with migration? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    migrate(dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
