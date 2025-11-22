#!/usr/bin/env python3
"""
Length-aware queue helper.

Polls status/training_status.json, inspects the `length_bin_staleness` field,
and optionally runs a user-specified command (e.g., enqueueing a dataset)
whenever a bin exceeds a chosen threshold.

Usage examples:

    # Dry-run mode, print warnings every minute
    python3 auto_queue_length_monitor.py --threshold 1800 --interval 60 --dry-run

    # Trigger a generator when the 300-500 bucket goes stale
    python3 auto_queue_length_monitor.py \
        --threshold 900 \
        --command "python3 training_queue.py add inbox/no_think_tags_latest/training_samples.jsonl normal"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

STATUS_PATH = Path("/path/to/training/status/training_status.json")


def load_status(path: Path) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Status file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}")


def find_stalest_bin(staleness: Dict[str, float]) -> Tuple[str, float]:
    if not staleness:
        return ("", 0.0)
    return max(staleness.items(), key=lambda item: item[1])


def run_command(cmd: str, dry_run: bool) -> int:
    if dry_run:
        print(f"[dry-run] Would run: {cmd}")
        return 0
    print(f"Running command: {cmd}")
    try:
        return subprocess.call(cmd, shell=True)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"Command failed: {exc}")
        return 1


def check_once(args) -> bool:
    data = load_status(args.status)
    staleness = data.get("length_bin_staleness") or {}

    if not staleness:
        print("[queue-monitor] No length coverage data yet.")
        return False

    bin_name, seconds = find_stalest_bin(staleness)
    print(f"[queue-monitor] Most stale bin: {bin_name or 'unknown'} ({seconds:.0f}s)")

    if seconds >= args.threshold:
        print(f"[queue-monitor] Threshold exceeded ({seconds:.0f}s >= {args.threshold}s)")
        if args.command:
            result = run_command(args.command, args.dry_run)
            if result == 0:
                print("[queue-monitor] Command finished successfully.")
            else:
                print(f"[queue-monitor] Command exited with status {result}.")
        else:
            print("[queue-monitor] No command configured; taking no action.")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Length-based queue helper")
    parser.add_argument("--status", type=Path, default=STATUS_PATH, help="Path to training_status.json")
    parser.add_argument("--threshold", type=int, default=1800, help="Seconds of inactivity before triggering")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--command", type=str, default="", help="Shell command to run when threshold is exceeded")
    parser.add_argument("--dry-run", action="store_true", help="Do not run the command; only log actions")
    parser.add_argument("--watch", action="store_true", help="Continue polling instead of exiting after first check")

    args = parser.parse_args()

    try:
        triggered = check_once(args)
        if not args.watch:
            return 0

        while True:
            time.sleep(args.interval)
            triggered = check_once(args) or triggered
    except KeyboardInterrupt:
        print("\n[queue-monitor] Exiting on user request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
