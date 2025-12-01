#!/usr/bin/env python3
"""
Temple CLI - Run diagnostic rituals from command line.

Usage:
    python -m tavern.temple check         # Run critical healthchecks (exit 0/1)
    python -m tavern.temple quick         # Run quick ritual
    python -m tavern.temple <ritual_id>   # Run specific ritual
    python -m tavern.temple ceremony      # Run all rituals
    python -m tavern.temple list          # List available rituals
    python -m tavern.temple --json <cmd>  # Output as JSON

Examples:
    # CI integration
    python -m tavern.temple check && echo "All checks passed"

    # Cron health monitoring
    python -m tavern.temple check || notify-send "Temple check failed"

    # List all rituals
    python -m tavern.temple list

    # Run full ceremony and get JSON output
    python -m tavern.temple --json ceremony
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def format_status(status: str) -> str:
    """Format status with ANSI colors."""
    colors = {
        "ok": "\033[92m",      # Green
        "warn": "\033[93m",    # Yellow
        "fail": "\033[91m",    # Red
        "skip": "\033[90m",    # Gray
    }
    reset = "\033[0m"
    symbol = {
        "ok": "✓",
        "warn": "⚠",
        "fail": "✗",
        "skip": "○",
    }
    return f"{colors.get(status, '')}{symbol.get(status, '?')} {status.upper()}{reset}"


def format_duration(ms: Optional[float]) -> str:
    """Format duration in milliseconds."""
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.2f}s"


def print_check_result(check, verbose: bool = False):
    """Print a single check result."""
    status_str = format_status(check.status)
    duration = format_duration(check.duration_ms())

    print(f"  {status_str} {check.name} {duration}")

    if verbose and check.details:
        for key, value in check.details.items():
            if key != "error":
                print(f"      {key}: {value}")

    if check.status == "fail" and check.remediation:
        print(f"      → {check.remediation}")


def print_ritual_result(result, verbose: bool = False):
    """Print ritual result with checks."""
    status_str = format_status(result.status)
    duration = format_duration(result.duration_ms())

    print(f"\n{result.name} {status_str} {duration}")
    print("-" * 60)

    for check in result.checks:
        print_check_result(check, verbose)


def cmd_check(args):
    """
    Run critical healthchecks.

    Runs: quick, scribe (eval system)
    Exits 0 if all pass, 1 if any fail.
    """
    from temple.cleric import run_ceremony, get_ceremony_status

    # Critical rituals for basic system health
    critical_rituals = ["quick", "scribe"]

    if not args.json:
        print("=" * 60)
        print("TEMPLE SMOKE TEST")
        print("=" * 60)

    results = run_ceremony(critical_rituals)
    status = get_ceremony_status(results)

    if args.json:
        output = {
            "status": status,
            "rituals": {
                rid: {
                    "status": r.status,
                    "checks": [
                        {
                            "id": c.id,
                            "name": c.name,
                            "status": c.status,
                            "details": c.details,
                        }
                        for c in r.checks
                    ]
                }
                for rid, r in results.items()
            }
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        for rid, result in results.items():
            print_ritual_result(result, args.verbose)

        print("\n" + "=" * 60)
        print(f"OVERALL: {format_status(status)}")
        print("=" * 60)

    return 0 if status == "ok" else 1


def cmd_ritual(args):
    """Run a specific ritual."""
    from temple.cleric import run_ritual, list_rituals

    ritual_id = args.ritual

    # Check if ritual exists
    available = list_rituals()
    if ritual_id not in available:
        print(f"Unknown ritual: {ritual_id}")
        print(f"Available: {', '.join(sorted(available.keys()))}")
        return 1

    if not args.json:
        print(f"Running {ritual_id} ritual...")

    result = run_ritual(ritual_id)

    if args.json:
        output = {
            "ritual_id": result.ritual_id,
            "name": result.name,
            "status": result.status,
            "duration_ms": result.duration_ms(),
            "checks": [
                {
                    "id": c.id,
                    "name": c.name,
                    "status": c.status,
                    "details": c.details,
                    "remediation": c.remediation,
                }
                for c in result.checks
            ]
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_ritual_result(result, args.verbose)

    return 0 if result.status == "ok" else 1


def cmd_ceremony(args):
    """Run all rituals."""
    from temple.cleric import run_ceremony, get_ceremony_status

    if not args.json:
        print("=" * 60)
        print("FULL TEMPLE CEREMONY")
        print("=" * 60)

    results = run_ceremony()
    status = get_ceremony_status(results)

    if args.json:
        output = {
            "status": status,
            "rituals": {
                rid: {
                    "status": r.status,
                    "duration_ms": r.duration_ms(),
                    "checks_passed": sum(1 for c in r.checks if c.status == "ok"),
                    "checks_total": len(r.checks),
                }
                for rid, r in results.items()
            }
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        for rid, result in results.items():
            print_ritual_result(result, args.verbose)

        print("\n" + "=" * 60)
        counts = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
        for r in results.values():
            counts[r.status] = counts.get(r.status, 0) + 1

        print(f"CEREMONY COMPLETE: {format_status(status)}")
        print(f"  {counts['ok']} passed, {counts['warn']} warnings, {counts['fail']} failed")
        print("=" * 60)

    return 0 if status == "ok" else 1


def cmd_list(args):
    """List available rituals."""
    from temple.cleric import list_rituals

    rituals = list_rituals()

    if args.json:
        print(json.dumps(rituals, indent=2))
    else:
        print("Available Rituals:")
        print("-" * 60)

        # Group by kind
        by_kind: Dict[str, List] = {}
        for rid, meta in rituals.items():
            kind = meta.get("kind", "healthcheck")
            if kind not in by_kind:
                by_kind[kind] = []
            by_kind[kind].append((rid, meta))

        for kind in ["healthcheck", "eval", "training", "infra", "meta"]:
            if kind in by_kind:
                print(f"\n{kind.upper()}:")
                for rid, meta in sorted(by_kind[kind]):
                    icon = meta.get("icon", "")
                    print(f"  {icon} {rid:12} - {meta['description']}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Temple CLI - Run diagnostic rituals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command")

    # check command
    check_parser = subparsers.add_parser("check", help="Run critical healthchecks")
    check_parser.set_defaults(func=cmd_check)

    # ceremony command
    ceremony_parser = subparsers.add_parser("ceremony", help="Run all rituals")
    ceremony_parser.set_defaults(func=cmd_ceremony)

    # list command
    list_parser = subparsers.add_parser("list", help="List available rituals")
    list_parser.set_defaults(func=cmd_list)

    # Parse known args to handle ritual names
    args, remaining = parser.parse_known_args()

    # If no command but have remaining args, treat first as ritual name
    if args.command is None and remaining:
        args.ritual = remaining[0]
        args.command = "ritual"
        args.func = cmd_ritual
    elif args.command is None:
        # Default to check
        args.command = "check"
        args.func = cmd_check

    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
