#!/usr/bin/env python3
"""
Auto-Improve Daemon - Orchestrates the full self-improving training loop.

This daemon coordinates:
1. Periodic self_improve_cycle on 3090 (via scheduler)
2. Correction sync from 3090 to 4090
3. Impact tracking before/after training
4. Adaptive scheduling based on results

The full loop:
  3090: self_improve_cycle → generates corrections
  4090: correction_sync → pulls corrections to inbox
  4090: training_daemon → trains on corrections
  3090: next cycle → measures impact

Usage:
    # Start the daemon
    python3 auto_improve_daemon.py --daemon

    # One-shot cycle
    python3 auto_improve_daemon.py --cycle

    # Check status
    python3 auto_improve_daemon.py --status

Configuration:
    - improve_interval: How often to run self_improve_cycle (default: 4 hours)
    - sync_interval: How often to sync corrections (default: 5 minutes)
    - min_training_gap: Minimum training steps between cycles (default: 500)
"""

import argparse
import json
import logging
import os
import requests
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.paths import get_base_dir, get_status_dir
from core.hosts import get_service_url

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutoImprove")


@dataclass
class CycleRecord:
    """Record of an improvement cycle."""
    cycle_id: str
    started_at: str
    completed_at: Optional[str] = None
    baseline_accuracy: Optional[float] = None
    errors_found: int = 0
    corrections_generated: int = 0
    corrections_synced: int = 0
    corrections_trained: int = 0
    post_accuracy: Optional[float] = None
    improvement: Optional[float] = None
    status: str = "running"


class AutoImproveDaemon:
    """Orchestrates the self-improving training loop."""

    def __init__(
        self,
        base_dir: str = None,
        scheduler_url: str = None,
        improve_interval: int = 14400,  # 4 hours
        sync_interval: int = 300,       # 5 minutes
        min_training_gap: int = 500     # Minimum steps between cycles
    ):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.scheduler_url = scheduler_url if scheduler_url else get_service_url("scheduler")
        self.improve_interval = improve_interval
        self.sync_interval = sync_interval
        self.min_training_gap = min_training_gap

        # Status files
        self.status_dir = get_status_dir()
        self.status_file = self.status_dir / "auto_improve.json"
        self.pid_file = self.base_dir / ".pids" / "auto_improve.pid"

        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

        # State
        self.cycles: List[CycleRecord] = []
        self.last_improve_time: Optional[datetime] = None
        self.last_sync_time: Optional[datetime] = None
        self.last_training_step: int = 0
        self._load_state()

    def _load_state(self):
        """Load daemon state from file."""
        if self.status_file.exists():
            with open(self.status_file) as f:
                data = json.load(f)
                self.cycles = [CycleRecord(**c) for c in data.get("cycles", [])]
                if data.get("last_improve_time"):
                    self.last_improve_time = datetime.fromisoformat(data["last_improve_time"])
                self.last_training_step = data.get("last_training_step", 0)

    def _save_state(self):
        """Save daemon state to file."""
        with open(self.status_file, 'w') as f:
            json.dump({
                "cycles": [asdict(c) for c in self.cycles[-50:]],  # Keep last 50
                "last_improve_time": self.last_improve_time.isoformat() if self.last_improve_time else None,
                "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
                "last_training_step": self.last_training_step,
                "daemon_status": "running",
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

    def _write_pid(self):
        """Write PID file."""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))

    def get_current_step(self) -> int:
        """Get current training step."""
        status_file = self.status_dir / "training_status.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
                return data.get("current_step", 0)
        return 0

    def should_run_improve_cycle(self) -> bool:
        """Check if we should run an improvement cycle."""
        # Check time since last cycle
        if self.last_improve_time:
            elapsed = (datetime.now() - self.last_improve_time).total_seconds()
            if elapsed < self.improve_interval:
                return False

        # Check training progress
        current_step = self.get_current_step()
        if current_step - self.last_training_step < self.min_training_gap:
            logger.debug(f"Not enough training progress: {current_step - self.last_training_step} < {self.min_training_gap}")
            return False

        return True

    def submit_improve_cycle(self) -> Optional[str]:
        """Submit self_improve_cycle to scheduler."""
        try:
            response = requests.post(
                f"{self.scheduler_url}/api/tasks/submit",
                json={
                    "task_type": "self_improve_cycle",
                    "params": {
                        "corrections_per_error": 20
                    }
                },
                timeout=30
            )

            if response.ok:
                data = response.json()
                return data.get("task_id")
            else:
                logger.error(f"Failed to submit task: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Scheduler request failed: {e}")
            return None

    def wait_for_task(self, task_id: str, timeout: int = 120) -> Optional[Dict]:
        """Wait for a task to complete."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                response = requests.get(
                    f"{self.scheduler_url}/api/tasks/{task_id}",
                    timeout=10
                )

                if response.ok:
                    data = response.json()
                    if data.get("status") == "completed":
                        return data.get("result", {})
                    elif data.get("status") == "failed":
                        logger.error(f"Task failed: {data.get('error')}")
                        return None

            except Exception as e:
                logger.warning(f"Error checking task: {e}")

            time.sleep(5)

        logger.warning(f"Task {task_id} timed out")
        return None

    def run_correction_sync(self) -> Dict:
        """Run correction sync from 3090 to 4090."""
        try:
            cmd = [
                "python3",
                str(self.base_dir / "monitoring" / "correction_sync.py"),
                "--sync"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"Sync failed: {result.stderr}")
                return {"synced": 0, "error": result.stderr}

        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {"synced": 0, "error": str(e)}

    def run_improve_cycle(self) -> CycleRecord:
        """Run a complete improvement cycle."""
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_step = self.get_current_step()

        cycle = CycleRecord(
            cycle_id=cycle_id,
            started_at=datetime.now().isoformat()
        )
        self.cycles.append(cycle)

        logger.info(f"Starting improvement cycle: {cycle_id}")

        # Step 1: Submit self_improve_cycle to 3090
        logger.info("Step 1: Submitting self_improve_cycle to scheduler...")
        task_id = self.submit_improve_cycle()

        if not task_id:
            cycle.status = "failed"
            cycle.completed_at = datetime.now().isoformat()
            self._save_state()
            return cycle

        # Step 2: Wait for completion
        logger.info(f"Step 2: Waiting for task {task_id}...")
        result = self.wait_for_task(task_id)

        if result:
            summary = result.get("summary", {})
            cycle.baseline_accuracy = summary.get("baseline_accuracy", 0)
            cycle.errors_found = summary.get("errors_found", 0)
            cycle.corrections_generated = summary.get("corrections_queued", 0)

        # Step 3: Sync corrections
        logger.info("Step 3: Syncing corrections from 3090...")
        sync_result = self.run_correction_sync()
        cycle.corrections_synced = sync_result.get("synced", 0)

        # Step 4: Update state
        cycle.status = "completed"
        cycle.completed_at = datetime.now().isoformat()
        self.last_improve_time = datetime.now()
        self.last_training_step = current_step

        self._save_state()

        logger.info(f"Cycle {cycle_id} complete: "
                   f"accuracy={cycle.baseline_accuracy:.1%}, "
                   f"corrections={cycle.corrections_generated}, "
                   f"synced={cycle.corrections_synced}")

        return cycle

    def get_status(self) -> Dict:
        """Get daemon status."""
        scheduler_ok = False
        try:
            response = requests.get(f"{self.scheduler_url}/api/health", timeout=5)
            scheduler_ok = response.ok
        except:
            pass

        # Time until next cycle
        next_cycle_in = None
        if self.last_improve_time:
            elapsed = (datetime.now() - self.last_improve_time).total_seconds()
            remaining = self.improve_interval - elapsed
            if remaining > 0:
                next_cycle_in = int(remaining)

        return {
            "daemon_status": "running" if self.pid_file.exists() else "stopped",
            "scheduler_connected": scheduler_ok,
            "total_cycles": len(self.cycles),
            "last_cycle": asdict(self.cycles[-1]) if self.cycles else None,
            "next_cycle_in_seconds": next_cycle_in,
            "improve_interval": self.improve_interval,
            "sync_interval": self.sync_interval,
            "current_step": self.get_current_step(),
            "last_training_step": self.last_training_step
        }

    def run_daemon(self):
        """Run the auto-improve daemon."""
        logger.info("=" * 60)
        logger.info("Auto-Improve Daemon Starting")
        logger.info("=" * 60)
        logger.info(f"Improve interval: {self.improve_interval}s ({self.improve_interval/3600:.1f}h)")
        logger.info(f"Sync interval: {self.sync_interval}s")
        logger.info(f"Min training gap: {self.min_training_gap} steps")

        self._write_pid()
        last_sync = time.time()

        try:
            while True:
                # Check if we should run improvement cycle
                if self.should_run_improve_cycle():
                    try:
                        self.run_improve_cycle()
                    except Exception as e:
                        logger.error(f"Cycle failed: {e}")

                # Periodic sync
                if time.time() - last_sync >= self.sync_interval:
                    try:
                        result = self.run_correction_sync()
                        if result.get("synced", 0) > 0:
                            logger.info(f"Synced {result['synced']} corrections")
                        last_sync = time.time()
                        self.last_sync_time = datetime.now()
                    except Exception as e:
                        logger.error(f"Sync failed: {e}")

                self._save_state()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
        finally:
            if self.pid_file.exists():
                self.pid_file.unlink()


def main():
    parser = argparse.ArgumentParser(description="Auto-Improve Daemon")
    parser.add_argument('--base-dir', default=None,
                       help='Base directory (default: auto-detect)')
    parser.add_argument('--scheduler-url', default=None,
                       help='GPU scheduler URL (default: from hosts.json)')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon')
    parser.add_argument('--cycle', action='store_true',
                       help='Run one improvement cycle')
    parser.add_argument('--status', action='store_true',
                       help='Show daemon status')
    parser.add_argument('--improve-interval', type=int, default=14400,
                       help='Improve cycle interval in seconds (default: 4h)')
    parser.add_argument('--sync-interval', type=int, default=300,
                       help='Sync interval in seconds (default: 5m)')

    args = parser.parse_args()

    daemon = AutoImproveDaemon(
        base_dir=args.base_dir,
        scheduler_url=args.scheduler_url,
        improve_interval=args.improve_interval,
        sync_interval=args.sync_interval
    )

    if args.daemon:
        daemon.run_daemon()

    elif args.cycle:
        cycle = daemon.run_improve_cycle()
        print(json.dumps(asdict(cycle), indent=2))

    elif args.status:
        status = daemon.get_status()
        print(json.dumps(status, indent=2))

    else:
        # Default: show status
        status = daemon.get_status()
        print("\nAuto-Improve Status:")
        print(f"  Scheduler: {'Connected' if status['scheduler_connected'] else 'Disconnected'}")
        print(f"  Total cycles: {status['total_cycles']}")
        print(f"  Current step: {status['current_step']}")
        if status['next_cycle_in_seconds']:
            mins = status['next_cycle_in_seconds'] // 60
            print(f"  Next cycle in: {mins} minutes")
        print(f"\nUse --daemon to start, --cycle for one-shot")


if __name__ == "__main__":
    main()
