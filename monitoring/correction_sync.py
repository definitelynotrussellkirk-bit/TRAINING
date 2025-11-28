#!/usr/bin/env python3
"""
Correction Sync - Sync generated corrections from 3090 to 4090 training inbox.

Runs on the 4090 (training machine) and pulls correction files from 3090.
This closes the loop: 3090 generates corrections → 4090 trains on them.

The sync process:
1. Check 3090 inbox for corrections_*.jsonl files
2. Copy new files to 4090 inbox
3. Track synced files to avoid duplicates
4. Optionally move synced files on 3090 to archive

Usage:
    # One-time sync
    python3 correction_sync.py --sync

    # Continuous sync daemon
    python3 correction_sync.py --daemon --interval 300

    # Check status
    python3 correction_sync.py --status

Output:
    status/correction_sync.json - Sync history and status
"""

import argparse
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SyncRecord:
    """Record of a synced file."""
    filename: str
    synced_at: str
    source_path: str
    dest_path: str
    size_bytes: int
    examples_count: int


class CorrectionSync:
    """Sync corrections from remote 3090 to local 4090."""

    def __init__(
        self,
        base_dir: str = None,
        remote_host: str = None,
        remote_base: str = None
    ):
        if base_dir is None:
            from core.paths import require_base_dir
            base_dir = str(require_base_dir())
        if remote_host is None:
            try:
                from core.hosts import get_host
                remote_host = get_host("3090").host
            except (ImportError, Exception):
                remote_host = "192.168.x.x"
        if remote_base is None:
            from core.hosts import get_trainer_base_dir; remote_base = get_trainer_base_dir()
        self.base_dir = Path(base_dir)
        self.remote_host = remote_host
        self.remote_base = remote_base

        # Local paths
        self.inbox_dir = self.base_dir / "inbox"
        self.status_dir = self.base_dir / "status"
        self.status_file = self.status_dir / "correction_sync.json"

        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)

        # Load sync history
        self.synced_files: Set[str] = set()
        self.sync_records: List[SyncRecord] = []
        self._load_status()

    def _load_status(self):
        """Load sync status from file."""
        if self.status_file.exists():
            with open(self.status_file) as f:
                data = json.load(f)
                self.synced_files = set(data.get("synced_files", []))
                self.sync_records = [
                    SyncRecord(**r) for r in data.get("records", [])
                ]

    def _save_status(self):
        """Save sync status to file."""
        with open(self.status_file, 'w') as f:
            json.dump({
                "synced_files": list(self.synced_files),
                "records": [asdict(r) for r in self.sync_records[-100:]],  # Keep last 100
                "last_sync": datetime.now().isoformat(),
                "total_synced": len(self.synced_files)
            }, f, indent=2)

    def list_remote_corrections(self) -> List[Dict]:
        """List correction files on 3090."""
        try:
            cmd = [
                "ssh", self.remote_host,
                f"ls -la {self.remote_base}/inbox/corrections_*.jsonl 2>/dev/null || true"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            files = []
            for line in result.stdout.strip().split('\n'):
                if not line or 'corrections_' not in line:
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    filename = parts[-1].split('/')[-1]
                    size = int(parts[4])
                    files.append({
                        "filename": filename,
                        "size": size,
                        "path": f"{self.remote_base}/inbox/{filename}"
                    })

            return files

        except Exception as e:
            logger.error(f"Failed to list remote files: {e}")
            return []

    def count_examples(self, filepath: Path) -> int:
        """Count examples in a JSONL file."""
        try:
            with open(filepath) as f:
                return sum(1 for _ in f)
        except:
            return 0

    def sync_file(self, remote_file: Dict) -> Optional[SyncRecord]:
        """Sync a single file from 3090 to 4090."""
        filename = remote_file["filename"]

        # Skip if already synced
        if filename in self.synced_files:
            logger.debug(f"Skipping {filename} (already synced)")
            return None

        # Check if file already exists locally
        local_path = self.inbox_dir / filename
        if local_path.exists():
            logger.debug(f"Skipping {filename} (exists locally)")
            self.synced_files.add(filename)
            return None

        # Copy file
        try:
            cmd = [
                "scp",
                f"{self.remote_host}:{remote_file['path']}",
                str(local_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"SCP failed for {filename}: {result.stderr}")
                return None

            # Verify file
            if not local_path.exists():
                logger.error(f"File not found after copy: {filename}")
                return None

            # Create record
            record = SyncRecord(
                filename=filename,
                synced_at=datetime.now().isoformat(),
                source_path=remote_file['path'],
                dest_path=str(local_path),
                size_bytes=local_path.stat().st_size,
                examples_count=self.count_examples(local_path)
            )

            self.synced_files.add(filename)
            self.sync_records.append(record)

            logger.info(f"Synced: {filename} ({record.examples_count} examples)")
            return record

        except Exception as e:
            logger.error(f"Failed to sync {filename}: {e}")
            return None

    def sync_all(self) -> Dict:
        """Sync all new correction files."""
        remote_files = self.list_remote_corrections()
        logger.info(f"Found {len(remote_files)} correction files on 3090")

        synced = []
        skipped = 0

        for remote_file in remote_files:
            record = self.sync_file(remote_file)
            if record:
                synced.append(record)
            else:
                skipped += 1

        self._save_status()

        result = {
            "synced": len(synced),
            "skipped": skipped,
            "total_remote": len(remote_files),
            "files": [r.filename for r in synced],
            "timestamp": datetime.now().isoformat()
        }

        if synced:
            logger.info(f"Synced {len(synced)} new correction files")
        else:
            logger.info("No new files to sync")

        return result

    def archive_remote(self, filename: str) -> bool:
        """Move a synced file to archive on 3090."""
        try:
            archive_dir = f"{self.remote_base}/inbox/synced"
            cmd = [
                "ssh", self.remote_host,
                f"mkdir -p {archive_dir} && mv {self.remote_base}/inbox/{filename} {archive_dir}/"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to archive {filename}: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current sync status."""
        remote_files = self.list_remote_corrections()
        pending = [f for f in remote_files if f["filename"] not in self.synced_files]

        # Count local corrections
        local_corrections = list(self.inbox_dir.glob("corrections_*.jsonl"))

        return {
            "remote_total": len(remote_files),
            "remote_pending": len(pending),
            "local_corrections": len(local_corrections),
            "total_synced": len(self.synced_files),
            "recent_syncs": [asdict(r) for r in self.sync_records[-5:]],
            "last_check": datetime.now().isoformat()
        }

    def run_daemon(self, interval: int = 300):
        """Run continuous sync daemon."""
        logger.info(f"Starting correction sync daemon (interval: {interval}s)")

        while True:
            try:
                result = self.sync_all()
                if result["synced"] > 0:
                    logger.info(f"Daemon synced {result['synced']} files")
            except Exception as e:
                logger.error(f"Sync error: {e}")

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Correction Sync")
    parser.add_argument('--base-dir', default=None,
                       help='Local base directory (default: auto-detected)')
    parser.add_argument('--remote-host', default=None,
                       help='Remote host (3090) (default: auto-detected)')
    parser.add_argument('--sync', action='store_true',
                       help='Run one-time sync')
    parser.add_argument('--daemon', action='store_true',
                       help='Run continuous sync daemon')
    parser.add_argument('--interval', type=int, default=300,
                       help='Daemon sync interval in seconds')
    parser.add_argument('--status', action='store_true',
                       help='Show sync status')
    parser.add_argument('--archive', action='store_true',
                       help='Archive synced files on remote')

    args = parser.parse_args()

    syncer = CorrectionSync(
        base_dir=args.base_dir,
        remote_host=args.remote_host
    )

    if args.sync:
        result = syncer.sync_all()
        print(json.dumps(result, indent=2))

    elif args.daemon:
        syncer.run_daemon(args.interval)

    elif args.status:
        status = syncer.get_status()
        print(json.dumps(status, indent=2))

    elif args.archive:
        # Archive all synced files
        for filename in list(syncer.synced_files)[-10:]:  # Last 10
            if syncer.archive_remote(filename):
                print(f"Archived: {filename}")

    else:
        # Default: show status
        status = syncer.get_status()
        print(f"\nCorrection Sync Status:")
        print(f"  Remote files: {status['remote_total']}")
        print(f"  Pending sync: {status['remote_pending']}")
        print(f"  Local corrections: {status['local_corrections']}")
        print(f"  Total synced: {status['total_synced']}")
        print(f"\nUse --sync to sync, --daemon for continuous")


if __name__ == "__main__":
    main()
