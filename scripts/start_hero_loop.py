#!/usr/bin/env python3
"""
Hero Loop Launcher - Starts hero_loop for the active campaign.

Reads the active campaign from control/active_campaign.json and launches
the hero_loop for that campaign. Used by the service registry to start
training as a managed service.

Usage:
    python3 scripts/start_hero_loop.py
    python3 scripts/start_hero_loop.py --campaign gou-qwen3-4b/campaign-001

The active campaign is set via:
    - control/active_campaign.json
    - Or passed directly via --campaign argument
"""

import os
import sys
import json
import signal
import subprocess
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir


def load_active_campaign() -> str | None:
    """Load active campaign path from control/active_campaign.json."""
    base_dir = get_base_dir()
    active_file = base_dir / "control" / "active_campaign.json"

    if not active_file.exists():
        return None

    try:
        with open(active_file) as f:
            data = json.load(f)
        return data.get("campaign_path")
    except Exception as e:
        print(f"Error reading active_campaign.json: {e}")
        return None


def write_pid_file(pid: int):
    """Write PID file for service registry."""
    base_dir = get_base_dir()
    pid_file = base_dir / ".pids" / "hero_loop.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def remove_pid_file():
    """Remove PID file on exit."""
    base_dir = get_base_dir()
    pid_file = base_dir / ".pids" / "hero_loop.pid"
    if pid_file.exists():
        pid_file.unlink()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start hero_loop for active campaign")
    parser.add_argument("--campaign", "-c", help="Campaign path (e.g., gou-qwen3-4b/campaign-001)")
    args = parser.parse_args()

    # Determine campaign path
    campaign_path = args.campaign or load_active_campaign()

    if not campaign_path:
        print("No active campaign found. Set one via control/active_campaign.json")
        print("Or pass --campaign gou-qwen3-4b/campaign-001")
        sys.exit(1)

    base_dir = get_base_dir()
    full_campaign_path = base_dir / campaign_path

    if not full_campaign_path.exists():
        print(f"Campaign path does not exist: {full_campaign_path}")
        sys.exit(1)

    print(f"Starting hero_loop for: {campaign_path}")

    # Build command
    cmd = [
        sys.executable,
        "-m", "arena.hero_loop",
        campaign_path
    ]

    # Start the hero_loop as subprocess
    # We exec into it so this script's PID becomes the hero_loop's PID
    os.chdir(base_dir)

    # Write our PID (before exec replaces us)
    write_pid_file(os.getpid())

    # Handle signals to clean up PID file
    def cleanup(signum, frame):
        remove_pid_file()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Replace this process with hero_loop
    try:
        os.execvp(cmd[0], cmd)
    except Exception as e:
        print(f"Failed to start hero_loop: {e}")
        remove_pid_file()
        sys.exit(1)


if __name__ == "__main__":
    main()
