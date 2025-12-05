#!/usr/bin/env python3
"""
Checkpoint Sync Daemon - Intelligently syncs checkpoints from 4090 to 3090

DEPRECATED: This pull-based sync is superseded by DeploymentOrchestrator (push-based).

The push-based approach in monitoring/deployment_orchestrator.py is preferred:
- Runs on 4090 (trainer) and pushes to 3090 (inference)
- Uses smart push logic (push on improvement or after N steps)
- More reliable - doesn't depend on 3090 being up
- Single source of truth for what gets synced

This file is kept for backwards compatibility but should not be used.
Use `python3 monitoring/deployment_orchestrator.py` instead.

Original description:
Runs on 3090 inference machine:
- Discovers available checkpoints on 4090 training machine
- Prioritizes recent checkpoints
- Tracks which have been tested by monitoring systems
- Auto-syncs best untested checkpoint when idle
- Removes old checkpoints to save space
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

from core.paths import get_base_dir, get_status_dir
from core.hosts import get_host

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CheckpointSyncDaemon')


class CheckpointSyncDaemon:
    def __init__(
        self,
        remote_host: str = None,
        remote_dir: str = None,
        local_dir: str = None,
        base_dir: str = None,
        interval: int = 300,  # 5 minutes
        keep_recent: int = 3,  # Keep 3 most recent checkpoints
    ):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()

        # Get trainer host info for remote_host and remote_dir
        trainer = get_host("4090")
        self.remote_host = remote_host if remote_host else trainer.host
        self.remote_dir = remote_dir if remote_dir else str(Path(trainer.models_dir) / "current_model")

        self.local_dir = Path(local_dir) if local_dir else self.base_dir / "models" / "current_model"
        self.interval = interval
        self.keep_recent = keep_recent

        # Ensure local dir exists
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Track tested checkpoints - use base_dir for status
        self.status_file = get_status_dir() / "checkpoint_sync.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        self.tested_checkpoints = self.load_tested_checkpoints()

    def load_tested_checkpoints(self) -> set:
        """Load set of checkpoint steps that have been tested"""
        tested = set()

        # Check curriculum optimizer results
        curriculum_file = self.base_dir / "status" / "curriculum_optimization.json"
        if curriculum_file.exists():
            try:
                with open(curriculum_file) as f:
                    data = json.load(f)
                for eval in data.get('evaluations', []):
                    step = eval.get('step')
                    if step:
                        tested.add(step)
            except:
                pass

        # Check regression monitor results
        regression_file = self.base_dir / "status" / "regression_monitoring.json"
        if regression_file.exists():
            try:
                with open(regression_file) as f:
                    data = json.load(f)
                for check in data.get('checks', []):
                    step = check.get('step')
                    if step:
                        tested.add(step)
            except:
                pass

        logger.info(f"Loaded {len(tested)} tested checkpoints from monitoring results")
        return tested

    def get_remote_checkpoints(self) -> List[int]:
        """Get list of checkpoint steps available on remote 4090"""
        try:
            # SSH to remote and list checkpoint directories
            cmd = f"ssh {self.remote_host} 'ls -1d {self.remote_dir}/checkpoint-* 2>/dev/null'"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.warning(f"Failed to list remote checkpoints: {result.stderr}")
                return []

            # Extract step numbers
            checkpoints = []
            for line in result.stdout.strip().split('\n'):
                if line and 'checkpoint-' in line:
                    try:
                        step = int(line.split('checkpoint-')[-1])
                        checkpoints.append(step)
                    except ValueError:
                        continue

            checkpoints.sort(reverse=True)  # Most recent first
            logger.info(f"Found {len(checkpoints)} checkpoints on 4090: {checkpoints[:5]}...")
            return checkpoints

        except Exception as e:
            logger.error(f"Error getting remote checkpoints: {e}")
            return []

    def get_local_checkpoints(self) -> List[int]:
        """Get list of checkpoint steps available locally on 3090"""
        checkpoints = []
        for ckpt_dir in self.local_dir.glob('checkpoint-*'):
            if ckpt_dir.is_dir():
                try:
                    step = int(ckpt_dir.name.split('-')[-1])
                    checkpoints.append(step)
                except ValueError:
                    continue

        checkpoints.sort(reverse=True)
        return checkpoints

    def select_best_checkpoint(
        self,
        remote_checkpoints: List[int],
        local_checkpoints: List[int]
    ) -> Optional[int]:
        """
        Select best checkpoint to sync:
        1. Prioritize untested checkpoints
        2. Prefer most recent
        3. Skip if already local
        """
        # Filter to untested checkpoints
        untested = [
            step for step in remote_checkpoints
            if step not in self.tested_checkpoints
        ]

        if not untested:
            logger.info("All remote checkpoints have been tested")
            # Fall back to most recent
            untested = remote_checkpoints

        # Skip checkpoints already local
        need_sync = [
            step for step in untested
            if step not in local_checkpoints
        ]

        if not need_sync:
            logger.info("All untested checkpoints are already local")
            return None

        # Pick most recent
        selected = need_sync[0]
        logger.info(f"Selected checkpoint-{selected} for sync")
        return selected

    def sync_checkpoint(self, step: int) -> bool:
        """Sync checkpoint from 4090 to 3090"""
        remote_path = f"{self.remote_host}:{self.remote_dir}/checkpoint-{step}/"
        local_path = self.local_dir / f"checkpoint-{step}"

        logger.info(f"Syncing checkpoint-{step} from 4090...")
        logger.info(f"Remote: {remote_path}")
        logger.info(f"Local: {local_path}")

        try:
            # Use rsync for efficient transfer
            cmd = [
                'rsync',
                '-avz',
                '--progress',
                remote_path,
                str(local_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully synced checkpoint-{step}")
                return True
            else:
                logger.error(f"✗ Failed to sync checkpoint-{step}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error syncing checkpoint-{step}: {e}")
            return False

    def cleanup_old_checkpoints(self, keep_steps: List[int]):
        """Remove old checkpoints to save space, keeping N most recent"""
        local_checkpoints = self.get_local_checkpoints()

        # Keep only N most recent
        to_keep = set(keep_steps[:self.keep_recent])
        to_remove = [step for step in local_checkpoints if step not in to_keep]

        for step in to_remove:
            ckpt_path = self.local_dir / f"checkpoint-{step}"
            if ckpt_path.exists():
                try:
                    import shutil
                    shutil.rmtree(ckpt_path)
                    logger.info(f"Removed old checkpoint-{step}")
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint-{step}: {e}")

    def save_status(self, synced: Optional[int], available: List[int]):
        """Save sync status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'last_synced': synced,
            'remote_checkpoints': available[:10],  # Top 10
            'local_checkpoints': self.get_local_checkpoints()[:10],
            'tested_checkpoints': sorted(list(self.tested_checkpoints), reverse=True)[:10],
            'untested_count': len([s for s in available if s not in self.tested_checkpoints])
        }

        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)

    def _save_status(self):
        """Alias for save_status() - called by GPU scheduler"""
        remote = self.get_remote_checkpoints()
        local = self.get_local_checkpoints()
        self.save_status(local[0] if local else None, remote)

    def check_sync_status(self) -> Dict:
        """
        Check sync status between local and remote checkpoints.
        Called by GPU scheduler.

        Returns:
            Dict with sync status
        """
        logger.info("Checking checkpoint sync status...")

        # Refresh tested checkpoints
        self.tested_checkpoints = self.load_tested_checkpoints()

        # Get checkpoint lists
        remote_checkpoints = self.get_remote_checkpoints()
        local_checkpoints = self.get_local_checkpoints()

        # Check if in sync
        in_sync = False
        if remote_checkpoints and local_checkpoints:
            # In sync if latest remote is also local
            in_sync = remote_checkpoints[0] in local_checkpoints

        # Find what needs syncing
        need_sync = [
            step for step in remote_checkpoints
            if step not in local_checkpoints
        ][:5]  # Top 5 missing

        result = {
            "in_sync": in_sync,
            "local_checkpoint": f"checkpoint-{local_checkpoints[0]}" if local_checkpoints else "none",
            "remote_checkpoint": f"checkpoint-{remote_checkpoints[0]}" if remote_checkpoints else "none",
            "local_count": len(local_checkpoints),
            "remote_count": len(remote_checkpoints),
            "need_sync": need_sync,
            "last_sync": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat()
        }

        # Optionally sync if needed
        if need_sync:
            selected = self.select_best_checkpoint(remote_checkpoints, local_checkpoints)
            if selected:
                logger.info(f"Syncing checkpoint-{selected}...")
                if self.sync_checkpoint(selected):
                    result["synced"] = selected
                    result["in_sync"] = True

        # Save status
        self.save_status(
            result.get("synced"),
            remote_checkpoints
        )

        return result

    def run_iteration(self):
        """Run one sync iteration"""
        logger.info("=== Checkpoint Sync Iteration ===")

        # Refresh tested checkpoints
        self.tested_checkpoints = self.load_tested_checkpoints()

        # Get available checkpoints
        remote_checkpoints = self.get_remote_checkpoints()
        local_checkpoints = self.get_local_checkpoints()

        logger.info(f"Remote: {len(remote_checkpoints)} checkpoints")
        logger.info(f"Local: {len(local_checkpoints)} checkpoints")
        logger.info(f"Tested: {len(self.tested_checkpoints)} checkpoints")

        # Select best checkpoint to sync
        selected = self.select_best_checkpoint(remote_checkpoints, local_checkpoints)

        synced = None
        if selected:
            # Sync it
            if self.sync_checkpoint(selected):
                synced = selected

                # Cleanup old checkpoints
                all_local = self.get_local_checkpoints()
                self.cleanup_old_checkpoints(all_local)
        else:
            logger.info("No checkpoints need syncing")

        # Save status
        self.save_status(synced, remote_checkpoints)

        logger.info(f"Sleeping for {self.interval}s...\n")

    def run(self):
        """Run daemon loop"""
        logger.info("=== Checkpoint Sync Daemon Started ===")
        logger.info(f"Remote: {self.remote_host}:{self.remote_dir}")
        logger.info(f"Local: {self.local_dir}")
        logger.info(f"Interval: {self.interval}s")
        logger.info(f"Keep: {self.keep_recent} most recent checkpoints")
        logger.info("=" * 60)

        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"Iteration {iteration}")
                self.run_iteration()
                time.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in iteration: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute on error


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Checkpoint Sync Daemon')
    parser.add_argument('--remote-host', default=None,
                       help='Remote 4090 hostname/IP (default: from hosts.json)')
    parser.add_argument('--remote-dir', default=None,
                       help='Remote checkpoint directory (default: from hosts.json)')
    parser.add_argument('--local-dir', default=None,
                       help='Local checkpoint directory (default: base_dir/models/current_model)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300)')
    parser.add_argument('--keep', type=int, default=3,
                       help='Number of recent checkpoints to keep (default: 3)')

    args = parser.parse_args()

    daemon = CheckpointSyncDaemon(
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        local_dir=args.local_dir,
        interval=args.interval,
        keep_recent=args.keep
    )

    daemon.run()


if __name__ == '__main__':
    main()
