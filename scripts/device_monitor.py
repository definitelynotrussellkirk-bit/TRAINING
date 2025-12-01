#!/usr/bin/env python3
"""
Device Monitor - Track device status and task distribution.

Shows real-time status of all devices, active services, and task distribution
across the training infrastructure.

Usage:
    python3 scripts/device_monitor.py              # Live status
    python3 scripts/device_monitor.py --summary    # Last 24h summary
    python3 scripts/device_monitor.py --watch      # Continuous monitoring
    python3 scripts/device_monitor.py --json       # JSON output
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir


def check_process_alive(pid: int) -> bool:
    """Check if a process is running."""
    try:
        import os
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_local_services() -> Dict[str, dict]:
    """Get status of local services (4090)."""
    base_dir = get_base_dir()
    pids_dir = base_dir / ".pids"

    services = {
        "training_daemon": {
            "name": "Training Daemon",
            "pid_file": base_dir / ".daemon.pid",
        },
        "tavern": {
            "name": "Tavern Server",
            "pid_file": pids_dir / "tavern.pid",
        },
        "vault": {
            "name": "VaultKeeper",
            "pid_file": pids_dir / "vault.pid",
        },
        "eval_runner": {
            "name": "Eval Runner",
            "pid_file": pids_dir / "eval_runner.pid",
        },
        "weaver": {
            "name": "The Weaver",
            "pid_file": pids_dir / "weaver.pid",
        },
    }

    status = {}
    for service_id, info in services.items():
        pid_file = info["pid_file"]
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                alive = check_process_alive(pid)
                status[service_id] = {
                    "name": info["name"],
                    "status": "running" if alive else "dead",
                    "pid": pid if alive else None,
                }
            except (ValueError, OSError):
                status[service_id] = {
                    "name": info["name"],
                    "status": "unknown",
                    "pid": None,
                }
        else:
            status[service_id] = {
                "name": info["name"],
                "status": "stopped",
                "pid": None,
            }

    return status


def check_remote_service(host: str, port: int, path: str = "/health") -> bool:
    """Check if remote service is responsive."""
    import requests
    try:
        resp = requests.get(f"http://{host}:{port}{path}", timeout=2)
        return resp.status_code == 200
    except:
        return False


def get_remote_services() -> Dict[str, dict]:
    """Get status of remote services."""
    services = {
        "inference_3090": {
            "name": "Inference Server (3090)",
            "host": "192.168.x.x",
            "port": 8765,
        },
    }

    status = {}
    for service_id, info in services.items():
        alive = check_remote_service(info["host"], info["port"])
        status[service_id] = {
            "name": info["name"],
            "status": "running" if alive else "unreachable",
            "endpoint": f"http://{info['host']}:{info['port']}",
        }

    return status


def get_gpu_info() -> Optional[Dict]:
    """Get GPU utilization info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "utilization": int(parts[2]),
                            "memory_used_mb": int(parts[3]),
                            "memory_total_mb": int(parts[4]),
                        })
            return {"gpus": gpus}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def parse_task_log(hours: int = 24) -> Dict[str, any]:
    """Parse task distribution log for the last N hours."""
    base_dir = get_base_dir()
    log_file = base_dir / "logs" / "task_distribution.log"

    if not log_file.exists():
        return {
            "total_tasks": 0,
            "by_device": {},
            "by_type": {},
            "by_status": {},
            "failures": [],
        }

    cutoff = datetime.now() - timedelta(hours=hours)

    stats = {
        "total_tasks": 0,
        "by_device": defaultdict(lambda: {"started": 0, "completed": 0, "failed": 0}),
        "by_type": defaultdict(lambda: {"started": 0, "completed": 0, "failed": 0}),
        "by_status": {"success": 0, "failed": 0, "in_progress": 0},
        "failures": [],
        "avg_duration_ms": {},
    }

    task_starts = {}  # Track started tasks
    durations = defaultdict(list)

    try:
        with open(log_file) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_time = datetime.fromisoformat(event["timestamp"])

                    if event_time < cutoff:
                        continue

                    task_id = event["task_id"]
                    task_type = event["task_type"]
                    device_id = event["device_id"]
                    event_type = event["event_type"]

                    if event_type == "start":
                        stats["total_tasks"] += 1
                        stats["by_device"][device_id]["started"] += 1
                        stats["by_type"][task_type]["started"] += 1
                        task_starts[task_id] = event

                    elif event_type == "complete":
                        stats["by_device"][device_id]["completed"] += 1
                        stats["by_type"][task_type]["completed"] += 1

                        if event.get("success"):
                            stats["by_status"]["success"] += 1
                        else:
                            stats["by_status"]["failed"] += 1

                        # Track duration
                        if event.get("duration_ms"):
                            durations[task_type].append(event["duration_ms"])

                    elif event_type == "fail":
                        stats["by_device"][device_id]["failed"] += 1
                        stats["by_type"][task_type]["failed"] += 1
                        stats["by_status"]["failed"] += 1

                        stats["failures"].append({
                            "timestamp": event["timestamp"],
                            "task_type": task_type,
                            "device": device_id,
                            "error": event.get("error", "Unknown"),
                        })

                        if event.get("duration_ms"):
                            durations[task_type].append(event["duration_ms"])

                except (json.JSONDecodeError, KeyError):
                    pass

        # Calculate in-progress tasks
        stats["by_status"]["in_progress"] = (
            stats["total_tasks"] -
            stats["by_status"]["success"] -
            stats["by_status"]["failed"]
        )

        # Calculate average durations
        for task_type, times in durations.items():
            if times:
                stats["avg_duration_ms"][task_type] = sum(times) / len(times)

        # Convert defaultdicts to regular dicts
        stats["by_device"] = dict(stats["by_device"])
        stats["by_type"] = dict(stats["by_type"])

    except FileNotFoundError:
        pass

    return stats


def print_live_status():
    """Print live device status."""
    local = get_local_services()
    remote = get_remote_services()
    gpu = get_gpu_info()

    print("\n" + "=" * 70)
    print("  DEVICE STATUS - Realm of Training")
    print("=" * 70)

    # Local services (4090)
    print("\n📍 LOCAL (4090 - Training Server)")
    print("-" * 70)
    for service_id, info in local.items():
        status_icon = "✅" if info["status"] == "running" else "❌"
        pid_str = f"(PID {info['pid']})" if info['pid'] else ""
        print(f"  {status_icon} {info['name']:<30} {info['status']:>10} {pid_str}")

    # GPU info
    if gpu:
        print("\n🖥️  GPU Status:")
        for g in gpu["gpus"]:
            mem_pct = (g["memory_used_mb"] / g["memory_total_mb"]) * 100
            print(f"  GPU {g['index']}: {g['name']}")
            print(f"    Utilization: {g['utilization']}%")
            print(f"    Memory: {g['memory_used_mb']}MB / {g['memory_total_mb']}MB ({mem_pct:.0f}%)")

    # Remote services (3090)
    print("\n📡 REMOTE (3090 - Inference Server)")
    print("-" * 70)
    for service_id, info in remote.items():
        status_icon = "✅" if info["status"] == "running" else "❌"
        print(f"  {status_icon} {info['name']:<30} {info['status']:>10}")
        print(f"      Endpoint: {info['endpoint']}")

    print("\n" + "=" * 70)


def print_task_summary(hours: int = 24):
    """Print task distribution summary."""
    stats = parse_task_log(hours)

    print("\n" + "=" * 70)
    print(f"  TASK DISTRIBUTION - Last {hours} Hours")
    print("=" * 70)

    if stats["total_tasks"] == 0:
        print("\n  No tasks logged yet.")
        print("  Task logging will begin when tasks are executed.")
        print("\n" + "=" * 70)
        return

    print(f"\n📊 Overall Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Success: {stats['by_status']['success']} ({stats['by_status']['success']/stats['total_tasks']*100:.1f}%)")
    print(f"  Failed: {stats['by_status']['failed']} ({stats['by_status']['failed']/stats['total_tasks']*100:.1f}%)")
    print(f"  In Progress: {stats['by_status']['in_progress']}")

    # By device
    print(f"\n🖥️  By Device:")
    for device, counts in stats["by_device"].items():
        total = counts["started"]
        completed = counts["completed"]
        failed = counts["failed"]
        print(f"  {device}:")
        print(f"    Started: {total}, Completed: {completed}, Failed: {failed}")

    # By task type
    print(f"\n📋 By Task Type:")
    for task_type, counts in stats["by_type"].items():
        total = counts["started"]
        completed = counts["completed"]
        failed = counts["failed"]
        success_rate = (completed / total * 100) if total > 0 else 0
        print(f"  {task_type}:")
        print(f"    Started: {total}, Completed: {completed}, Failed: {failed}")
        print(f"    Success Rate: {success_rate:.1f}%")

    # Average durations
    if stats["avg_duration_ms"]:
        print(f"\n⏱️  Average Duration:")
        for task_type, avg_ms in stats["avg_duration_ms"].items():
            print(f"  {task_type}: {avg_ms:.0f}ms ({avg_ms/1000:.2f}s)")

    # Recent failures
    if stats["failures"]:
        print(f"\n❌ Recent Failures ({len(stats['failures'])} total):")
        for failure in stats["failures"][-5:]:  # Last 5
            ts = failure["timestamp"].split('T')[1].split('.')[0]
            print(f"  [{ts}] {failure['task_type']} on {failure['device']}")
            print(f"          Error: {failure['error']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Monitor device status and task distribution")
    parser.add_argument("--summary", action="store_true", help="Show 24h task summary")
    parser.add_argument("--hours", type=int, default=24, help="Hours of history (with --summary)")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring (Ctrl+C to stop)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval in seconds")

    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                print_live_status()
                if args.summary:
                    print_task_summary(args.hours)
                print(f"\n[Refreshing every {args.interval}s - Ctrl+C to stop]")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)

    elif args.json:
        data = {
            "local_services": get_local_services(),
            "remote_services": get_remote_services(),
            "gpu": get_gpu_info(),
        }
        if args.summary:
            data["task_stats"] = parse_task_log(args.hours)
        print(json.dumps(data, indent=2))

    elif args.summary:
        print_task_summary(args.hours)

    else:
        print_live_status()


if __name__ == "__main__":
    main()
