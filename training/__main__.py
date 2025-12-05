#!/usr/bin/env python3
"""
TRAINING CLI - Main entry point for Realm of Training.

Usage:
    python -m training play         # Enter the realm - start everything
    python -m training doctor       # Check environment and services
    python -m training start-all    # Start all services for dev
    python -m training stop-all     # Stop all services
    python -m training status       # Show current status
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="training",
        description="Realm of Training - CLI for managing training infrastructure"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # play command - THE main entry point
    p_play = subparsers.add_parser(
        "play",
        help="Enter the realm - start services and awaken heroes"
    )
    p_play.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    p_play.add_argument(
        "--hero",
        type=str,
        help="Only awaken a specific hero (e.g., titan-qwen3-4b/campaign-001)"
    )

    # doctor command
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Check environment, config, and services"
    )
    p_doctor.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues"
    )
    p_doctor.add_argument(
        "--check-hardcodes",
        action="store_true",
        help="Also check for hardcoded paths"
    )

    # start-all command
    p_start = subparsers.add_parser(
        "start-all",
        help="Start Vault, Tavern, and local worker for dev"
    )
    p_start.add_argument(
        "--no-worker",
        action="store_true",
        help="Skip starting the local worker"
    )

    # stop-all command
    p_stop = subparsers.add_parser(
        "stop-all",
        help="Stop all services"
    )

    # status command
    p_status = subparsers.add_parser(
        "status",
        help="Show current status of all services"
    )

    # reset command - multi-level reset system
    p_reset = subparsers.add_parser(
        "reset",
        help="Reset training environment (multi-level: campaign, hero, full, deep)"
    )
    p_reset.add_argument(
        "--level", "-l",
        choices=["campaign", "hero", "full", "deep"],
        default="full",
        help="Reset level: campaign (one), hero (all for hero), full (everything), deep (nuclear)"
    )
    p_reset.add_argument(
        "--target", "-t",
        help="Target for campaign/hero reset (e.g., gou-qwen3-4b/campaign-001)"
    )
    p_reset.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    p_reset.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    p_reset.add_argument(
        "--confirm",
        action="store_true",
        help="Required confirmation for DEEP level reset"
    )
    p_reset.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    p_reset.add_argument(
        "--keep-jobs",
        action="store_true",
        help="Keep job database (legacy, for full reset compatibility)"
    )

    # temple command - diagnostic rituals
    p_temple = subparsers.add_parser(
        "temple",
        help="Run diagnostic rituals on the realm"
    )
    p_temple.add_argument(
        "ritual",
        nargs="?",
        default="quick",
        help="Ritual to run (quick, api). Default: quick"
    )
    p_temple.add_argument(
        "--list",
        action="store_true",
        help="List available rituals"
    )
    p_temple.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Import and run the appropriate command
    from training.cli import run_command
    return run_command(args.command, args)


if __name__ == "__main__":
    sys.exit(main())
