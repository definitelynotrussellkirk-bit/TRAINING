#!/usr/bin/env python3
"""
Verbose Monitor - Watch and analyze task lifecycle logs.

View real-time task execution with detailed timestamps and durations.

Usage:
    # Watch live verbose logs
    python3 scripts/verbose_monitor.py --watch

    # Show stats for all tasks
    python3 scripts/verbose_monitor.py --stats

    # Show stats for eval tasks only
    python3 scripts/verbose_monitor.py --stats --type eval

    # Show details for a specific task
    python3 scripts/verbose_monitor.py --task eval-183726-binary-1

    # Tail last N verbose log entries
    python3 scripts/verbose_monitor.py --tail 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.verbose_logger import (
    VerboseLogger,
    is_verbose_mode,
    get_verbose_file,
    print_task_summary,
    print_all_tasks_summary,
    format_duration
)


def tail_verbose_log(count: int = 20):
    """Print last N entries from verbose log."""
    log_file = get_verbose_file()

    if not log_file or not log_file.exists():
        print(f"No verbose log found at {log_file}")
        print("Enable verbose mode with: export VERBOSE=1")
        return

    with open(log_file) as f:
        lines = f.readlines()

    print(f"\n{'='*80}")
    print(f"Last {count} verbose log entries ({log_file})")
    print(f"{'='*80}\n")

    for line in lines[-count:]:
        try:
            entry = json.loads(line.strip())
            timestamp = entry.get("timestamp", "")
            event = entry.get("event", "")
            task_id = entry.get("task_id", "N/A")
            task_type = entry.get("task_type", "N/A")

            # Format based on event type
            if event == "task_queued":
                print(f"[{timestamp}] 📥 QUEUED  {task_type}/{task_id}")
                metadata = entry.get("metadata", {})
                if metadata:
                    print(f"             Metadata: {json.dumps(metadata, default=str)}")

            elif event == "task_started":
                queue_time = entry.get("time_in_queue_seconds")
                print(f"[{timestamp}] ▶️  STARTED {task_type}/{task_id}")
                if queue_time is not None:
                    print(f"             Queue time: {format_duration(queue_time)}")

            elif event == "task_progress":
                message = entry.get("message", "")
                print(f"[{timestamp}] ⚙️  PROGRESS {task_type}/{task_id}: {message}")

            elif event == "task_finished":
                success = entry.get("success", False)
                exec_time = entry.get("execution_time_seconds")
                total_time = entry.get("total_time_seconds")
                status_icon = "✅" if success else "❌"

                print(f"[{timestamp}] {status_icon} FINISHED {task_type}/{task_id}")
                if exec_time is not None:
                    print(f"             Execution: {format_duration(exec_time)}, Total: {format_duration(total_time)}")

                result = entry.get("result", {})
                if result:
                    print(f"             Result: {json.dumps(result, default=str)}")

                error = entry.get("error")
                if error:
                    print(f"             Error: {error}")

            else:
                print(f"[{timestamp}] {event}: {json.dumps(entry, default=str, indent=2)}")

            print()

        except json.JSONDecodeError:
            print(f"Invalid JSON: {line.strip()}")


def watch_verbose_log():
    """Watch verbose log in real-time (like tail -f)."""
    log_file = get_verbose_file()

    if not log_file:
        print("No verbose log file configured")
        print("Enable verbose mode with: export VERBOSE=1")
        return

    print(f"\nWatching {log_file} (Ctrl+C to stop)\n")
    print(f"{'='*80}\n")

    # Create file if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch()

    # Follow file
    with open(log_file) as f:
        # Go to end of file
        f.seek(0, 2)

        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue

                try:
                    entry = json.loads(line.strip())
                    timestamp = entry.get("timestamp", "")
                    event = entry.get("event", "")
                    task_id = entry.get("task_id", "N/A")
                    task_type = entry.get("task_type", "N/A")

                    # Colorful output for different events
                    if event == "task_queued":
                        print(f"\033[94m[{timestamp}] 📥 QUEUED  {task_type}/{task_id}\033[0m")
                    elif event == "task_started":
                        print(f"\033[92m[{timestamp}] ▶️  STARTED {task_type}/{task_id}\033[0m")
                    elif event == "task_progress":
                        message = entry.get("message", "")
                        print(f"\033[93m[{timestamp}] ⚙️  PROGRESS {task_type}/{task_id}: {message}\033[0m")
                    elif event == "task_finished":
                        success = entry.get("success", False)
                        exec_time = entry.get("execution_time_seconds")
                        total_time = entry.get("total_time_seconds")

                        if success:
                            print(f"\033[92m[{timestamp}] ✅ SUCCESS  {task_type}/{task_id} ({format_duration(total_time)})\033[0m")
                        else:
                            error = entry.get("error", "unknown")
                            print(f"\033[91m[{timestamp}] ❌ FAILED   {task_type}/{task_id}: {error}\033[0m")
                    else:
                        print(f"[{timestamp}] {event}: {task_id}")

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {line.strip()}")

        except KeyboardInterrupt:
            print("\n\nStopped watching log")


def main():
    parser = argparse.ArgumentParser(
        description="Verbose Monitor - Watch and analyze task lifecycle logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--watch", "-w", action="store_true", help="Watch verbose log in real-time")
    parser.add_argument("--stats", "-s", action="store_true", help="Show task statistics")
    parser.add_argument("--type", "-t", help="Filter by task type (e.g., 'eval')")
    parser.add_argument("--task", help="Show details for specific task ID")
    parser.add_argument("--tail", type=int, help="Show last N log entries")
    parser.add_argument("--cleanup", action="store_true", help="Clean up completed tasks older than 1 hour")

    args = parser.parse_args()

    # Enable verbose mode if not already set
    if not is_verbose_mode():
        print("⚠️  VERBOSE mode is not enabled")
        print("   Enable it with: export VERBOSE=1")
        print()

    if args.watch:
        watch_verbose_log()

    elif args.stats:
        print_all_tasks_summary(task_type=args.type)

        # Show active tasks
        tasks = VerboseLogger.get_all_tasks(task_type=args.type, incomplete_only=True)
        if tasks:
            print(f"\n{'='*80}")
            print(f"Active Tasks ({len(tasks)})")
            print(f"{'='*80}\n")

            for task in sorted(tasks, key=lambda t: t.created_at, reverse=True):
                age = time.time() - task.created_at
                if task.started_at:
                    duration = time.time() - task.started_at
                    print(f"⚙️  {task.task_type}/{task.task_id}")
                    print(f"   Running for: {format_duration(duration)}, Age: {format_duration(age)}")
                else:
                    print(f"📥 {task.task_type}/{task.task_id}")
                    print(f"   Queued for: {format_duration(age)}")
                print()

    elif args.task:
        print_task_summary(args.task)

    elif args.tail:
        tail_verbose_log(args.tail)

    elif args.cleanup:
        VerboseLogger.cleanup_completed(max_age_seconds=3600)
        print("Cleaned up completed tasks older than 1 hour")

    else:
        # Default: show recent activity
        tail_verbose_log(count=10)


if __name__ == "__main__":
    main()
