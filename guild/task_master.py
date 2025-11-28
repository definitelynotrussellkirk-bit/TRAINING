#!/usr/bin/env python3
"""
Task Master - GPU-aware task scheduler

Monitors GPU utilization and runs tasks when resources are available.
Integrates with task_registry.py for available tasks.

Features:
- Monitors 4090 (local) and 3090 (remote) GPU utilization
- Runs tasks when GPU utilization drops below threshold
- Respects task priorities, cooldowns, and GPU requirements
- Can run as daemon or one-shot

Usage:
    python3 guild/task_master.py --status           # Show GPU + task status
    python3 guild/task_master.py --once             # Check and run one task
    python3 guild/task_master.py --daemon           # Run continuously
    python3 guild/task_master.py --run sparring_binary  # Force run specific task
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from guild.task_registry import (
    get_available_tasks,
    get_task,
    run_task,
    get_status as get_registry_status,
    get_task_state,
    Task,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/path/to/training")


@dataclass
class GPUStatus:
    """Status of a GPU"""
    name: str
    host: str
    available: bool
    utilization: float  # 0-100
    memory_used_gb: float
    memory_total_gb: float
    memory_free_pct: float
    temperature: float
    is_idle: bool  # Below threshold


class TaskMaster:
    """
    GPU-aware task scheduler.

    Monitors 3090 GPU and runs tasks when resources are available.
    IMPORTANT: Never schedules tasks on 4090 - that's the trainer!
    """

    # Utilization thresholds (percent)
    IDLE_THRESHOLD = 40.0      # Below this = idle, can run tasks
    BUSY_THRESHOLD = 80.0      # Above this = busy, don't start new tasks

    # Memory thresholds (percent free)
    MIN_MEMORY_FREE = 30.0     # Need at least 30% free to start task

    def __init__(
        self,
        base_dir: Path = None,
        inference_url: str = "http://192.168.x.x:8765",
        check_interval: int = 60,
    ):
        self.base_dir = base_dir or BASE_DIR
        self.inference_url = inference_url
        self.check_interval = check_interval

        self.status_file = self.base_dir / "status" / "task_master.json"
        self.pid_file = self.base_dir / ".pids" / "task_master.pid"

        self.running = False
        self.stats = {
            "tasks_run": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "started_at": None,
            "last_check": None,
        }

    def get_4090_status(self) -> GPUStatus:
        """Get local 4090 GPU status via nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return GPUStatus(
                    name="4090", host="localhost", available=False,
                    utilization=0, memory_used_gb=0, memory_total_gb=24,
                    memory_free_pct=0, temperature=0, is_idle=False
                )

            parts = result.stdout.strip().split(", ")
            util = float(parts[0])
            mem_used = float(parts[1]) / 1024  # MB to GB
            mem_total = float(parts[2]) / 1024
            temp = float(parts[3])
            name = parts[4] if len(parts) > 4 else "4090"

            mem_free_pct = ((mem_total - mem_used) / mem_total) * 100

            return GPUStatus(
                name="4090",
                host="localhost",
                available=True,
                utilization=util,
                memory_used_gb=mem_used,
                memory_total_gb=mem_total,
                memory_free_pct=mem_free_pct,
                temperature=temp,
                is_idle=util < self.IDLE_THRESHOLD and mem_free_pct > self.MIN_MEMORY_FREE,
            )

        except Exception as e:
            logger.warning(f"Failed to get 4090 status: {e}")
            return GPUStatus(
                name="4090", host="localhost", available=False,
                utilization=0, memory_used_gb=0, memory_total_gb=24,
                memory_free_pct=0, temperature=0, is_idle=False
            )

    def get_3090_status(self) -> GPUStatus:
        """Get remote 3090 GPU status via inference API"""
        try:
            resp = requests.get(f"{self.inference_url}/health", timeout=5)
            data = resp.json()

            gpu = data.get("gpu", {})
            if not gpu.get("available", False):
                return GPUStatus(
                    name="3090", host="192.168.x.x", available=False,
                    utilization=0, memory_used_gb=0, memory_total_gb=24,
                    memory_free_pct=0, temperature=0, is_idle=False
                )

            # Inference API doesn't report utilization directly
            # Use memory as proxy, and check if worker is busy
            mem_used = gpu.get("memory_allocated_gb", 0)
            mem_total = 24  # RTX 3090 = 24GB
            mem_free_pct = ((mem_total - mem_used) / mem_total) * 100

            worker_busy = data.get("worker_busy", False)

            # Consider idle if worker not busy and memory < 50% used
            is_idle = not worker_busy and mem_free_pct > 50

            return GPUStatus(
                name="3090",
                host="192.168.x.x",
                available=True,
                utilization=50 if worker_busy else 10,  # Estimate
                memory_used_gb=mem_used,
                memory_total_gb=mem_total,
                memory_free_pct=mem_free_pct,
                temperature=0,  # Not reported
                is_idle=is_idle,
            )

        except Exception as e:
            logger.warning(f"Failed to get 3090 status: {e}")
            return GPUStatus(
                name="3090", host="192.168.x.x", available=False,
                utilization=0, memory_used_gb=0, memory_total_gb=24,
                memory_free_pct=0, temperature=0, is_idle=False
            )

    def get_all_gpu_status(self) -> Dict[str, GPUStatus]:
        """Get status of all GPUs (only 3090 - never use 4090 trainer!)"""
        return {
            # NOTE: 4090 is the TRAINER - never schedule tasks on it!
            # Only monitor 3090 for opportunistic tasks
            "3090": self.get_3090_status(),
        }

    def find_runnable_task(self, gpu_status: Dict[str, GPUStatus]) -> Optional[Task]:
        """
        Find the highest priority task that can run given current GPU status.

        Returns:
            Task to run, or None if nothing can run
        """
        # Check which GPUs are idle
        idle_gpus = [name for name, status in gpu_status.items() if status.is_idle]

        if not idle_gpus:
            return None

        # Get available tasks for each idle GPU
        for gpu in idle_gpus:
            tasks = get_available_tasks(gpu=gpu, check_cooldown=True)
            if tasks:
                # Return highest priority task (list is sorted)
                return tasks[0]

        # Also check tasks that don't need GPU
        tasks = get_available_tasks(gpu="none", check_cooldown=True)
        if tasks:
            return tasks[0]

        return None

    def check_and_run(self) -> Optional[Dict]:
        """
        Check GPU status and run a task if resources available.

        Returns:
            Task result dict, or None if nothing ran
        """
        self.stats["last_check"] = datetime.now().isoformat()

        # Get GPU status
        gpu_status = self.get_all_gpu_status()

        # Log status
        for name, status in gpu_status.items():
            if status.available:
                idle_str = "IDLE" if status.is_idle else "BUSY"
                logger.debug(f"GPU {name}: {status.utilization:.0f}% util, "
                           f"{status.memory_free_pct:.0f}% mem free [{idle_str}]")
            else:
                logger.debug(f"GPU {name}: UNAVAILABLE")

        # Find task to run
        task = self.find_runnable_task(gpu_status)

        if not task:
            logger.debug("No tasks available to run")
            return None

        logger.info(f"🎯 Found task to run: {task.name} (priority {task.priority}, gpu={task.gpu})")

        # Run the task
        result = run_task(task.id)

        self.stats["tasks_run"] += 1
        if result.get("success"):
            self.stats["tasks_succeeded"] += 1
            logger.info(f"✅ Task {task.id} completed in {result.get('duration', 0):.1f}s")
        else:
            self.stats["tasks_failed"] += 1
            logger.warning(f"❌ Task {task.id} failed: {result.get('error', 'Unknown')}")

        # Save status
        self._save_status(gpu_status, task, result)

        return result

    def _save_status(self, gpu_status: Dict[str, GPUStatus], task: Task = None, result: Dict = None):
        """Save current status to file"""
        status = {
            "gpus": {
                name: {
                    "available": s.available,
                    "utilization": s.utilization,
                    "memory_used_gb": s.memory_used_gb,
                    "memory_total_gb": s.memory_total_gb,
                    "memory_free_pct": s.memory_free_pct,
                    "is_idle": s.is_idle,
                }
                for name, s in gpu_status.items()
            },
            "last_task": {
                "id": task.id if task else None,
                "name": task.name if task else None,
                "success": result.get("success") if result else None,
                "duration": result.get("duration") if result else None,
                "timestamp": datetime.now().isoformat(),
            } if task else None,
            "stats": self.stats,
            "updated_at": datetime.now().isoformat(),
        }

        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def run_daemon(self):
        """Run as continuous daemon"""
        logger.info("🤖 Task Master starting...")
        logger.info(f"   Check interval: {self.check_interval}s")
        logger.info(f"   Idle threshold: <{self.IDLE_THRESHOLD}% utilization")

        # Write PID
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self.stats["started_at"] = datetime.now().isoformat()

        try:
            while self.running:
                try:
                    self.check_and_run()
                except Exception as e:
                    logger.error(f"Error in check cycle: {e}")

                # Sleep in small increments to respond to signals
                for _ in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)

        finally:
            logger.info("🤖 Task Master stopping...")
            if self.pid_file.exists():
                self.pid_file.unlink()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def get_status_report(self) -> str:
        """Get formatted status report"""
        gpu_status = self.get_all_gpu_status()
        registry_status = get_registry_status()

        lines = [
            "",
            "="*70,
            "TASK MASTER STATUS",
            "="*70,
            "",
            "GPUs:",
        ]

        for name, status in gpu_status.items():
            if status.available:
                idle = "✓ IDLE" if status.is_idle else "⏳ BUSY"
                lines.append(f"  {name}: {status.utilization:.0f}% util | "
                           f"{status.memory_used_gb:.1f}/{status.memory_total_gb:.0f}GB | "
                           f"{status.memory_free_pct:.0f}% free | {idle}")
            else:
                lines.append(f"  {name}: ❌ UNAVAILABLE")

        lines.extend([
            "",
            "Available Tasks:",
        ])

        # Show tasks that could run
        for gpu_name, gpu in gpu_status.items():
            if gpu.is_idle:
                tasks = get_available_tasks(gpu=gpu_name, check_cooldown=True)
                if tasks:
                    lines.append(f"  {gpu_name} (idle):")
                    for t in tasks[:3]:
                        lines.append(f"    [{t.priority}] {t.id}: {t.name}")
                    if len(tasks) > 3:
                        lines.append(f"    ... and {len(tasks)-3} more")

        # CPU tasks
        cpu_tasks = get_available_tasks(gpu="none", check_cooldown=True)
        if cpu_tasks:
            lines.append("  CPU (no GPU needed):")
            for t in cpu_tasks[:3]:
                lines.append(f"    [{t.priority}] {t.id}: {t.name}")

        lines.extend([
            "",
            f"Total registered tasks: {registry_status['total_tasks']}",
            f"Enabled tasks: {registry_status['enabled_tasks']}",
            "="*70,
            "",
        ])

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task Master - GPU-aware task scheduler")
    parser.add_argument("--status", action="store_true", help="Show GPU and task status")
    parser.add_argument("--once", action="store_true", help="Check once and run if possible")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--run", metavar="TASK", help="Force run specific task")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--base-dir", default="/path/to/training")

    args = parser.parse_args()

    master = TaskMaster(
        base_dir=Path(args.base_dir),
        check_interval=args.interval,
    )

    if args.status:
        print(master.get_status_report())

    elif args.run:
        task = get_task(args.run)
        if not task:
            print(f"Unknown task: {args.run}")
            sys.exit(1)

        print(f"Running task: {task.name}")
        result = run_task(args.run)

        if result.get("success"):
            print(f"✅ Completed in {result.get('duration', 0):.1f}s")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown')}")
            sys.exit(1)

    elif args.once:
        result = master.check_and_run()
        if result:
            if result.get("success"):
                print(f"✅ Ran task successfully")
            else:
                print(f"❌ Task failed: {result.get('error')}")
        else:
            print("No tasks available to run (GPUs busy or all on cooldown)")

    elif args.daemon:
        master.run_daemon()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
