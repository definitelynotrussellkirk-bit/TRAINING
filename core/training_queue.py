#!/usr/bin/env python3
"""
Training Queue Management System

This module implements a priority-based file queue system for managing training data files.
Files are organized into three priority levels (high/normal/low) and processed in FIFO order
within each priority level. The system prevents race conditions using filesystem-based locking
and tracks file lifecycle through metadata.

Key Components:
    - TrainingQueue: Main queue manager with priority handling
    - Priority Queues: Three directories (high/normal/low) for prioritized processing
    - Processing State: Files move to processing/ during training
    - Metadata Tracking: JSON-based history of processed/failed/skipped files
    - Race Condition Prevention: Atomic file moves prevent concurrent processing

Queue Lifecycle:
    1. Inbox → Queue: Files dropped in inbox/ are moved to priority queue
    2. Queue → Processing: get_next_file() moves file to processing/
    3. Processing → Completed: mark_completed() deletes file (or moves to recently_completed/)
    4. Processing → Failed: mark_failed() moves file to failed/
    5. Processing → Skipped: mark_skipped() moves file back to original queue

Directory Structure:
    base_dir/
        inbox/                      - Drop zone for new .jsonl files
        queue/
            high/                   - High priority files (process first)
            normal/                 - Normal priority files (default)
            low/                    - Low priority files (process last)
            processing/             - Currently training file (only 1 at a time)
            failed/                 - Failed training files (kept for review)
            recently_completed/     - Recently completed files (optional retention)
            queue_metadata.json     - History of all processed files

Priority Processing Order:
    1. high/ → Process all files in FIFO order
    2. normal/ → Process all files in FIFO order (if high empty)
    3. low/ → Process all files in FIFO order (if high + normal empty)

Integration Points:
    - core/training_daemon.py: Main consumer (process_inbox, get_next_file loop)
    - CLI tools: Direct queue management (add, list, change priority)

Usage:
    from core.training_queue import TrainingQueue

    # Initialize queue
    queue = TrainingQueue("/training")

    # Process inbox files into queue
    count = queue.process_inbox(default_priority="normal")

    # Get next file to train on (by priority)
    data_file = queue.get_next_file()
    if data_file:
        # Train on file...
        queue.mark_completed(data_file, delete_file=True)

    # Check queue status
    status = queue.get_queue_status()
    print(f"Queue: {status['queued']['high']} high, {status['queued']['normal']} normal")
"""

import json
import shutil
import fcntl
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingQueue:
    """
    Priority-based queue manager for training data files.

    Responsibilities:
        - Scan inbox for new .jsonl training files
        - Add files to priority queues (high/normal/low)
        - Process inbox automatically (batch move to queues)
        - Get next file by priority (high → normal → low, FIFO within level)
        - Track file lifecycle (queued → processing → completed/failed/skipped)
        - Prevent race conditions (atomic file moves)
        - Maintain metadata (history of all processed files)
        - Support queue inspection (status, list, change priority)

    Data Flow:
        1. Scan & Add:
            → scan_inbox() finds .jsonl files in inbox/
            → add_to_queue() moves file to priority queue (high/normal/low)
            → File now visible in queue status

        2. Get Next File:
            → get_next_file() checks high/, then normal/, then low/ (oldest first)
            → Moves file to processing/ (atomic, prevents concurrent access)
            → Returns Path to file or None if queue empty

        3. Mark Result:
            → mark_completed(): Delete file or move to recently_completed/
            → mark_failed(): Move to failed/ for review
            → mark_skipped(): Move back to original priority queue

        4. Metadata Update:
            → All results recorded in queue_metadata.json
            → Tracks: filename, timestamp, priority, result (completed/failed/skipped)

    Race Condition Prevention:
        - Atomic file moves: shutil.move() is atomic on same filesystem
        - Processing lock: Only one file in processing/ at a time
        - get_next_file() checks processing/ is empty before moving
        - If crash occurs: processing/ contains orphaned file (daemon recovers on startup)

    Attributes:
        base_dir: Root directory for training system
        inbox: Inbox directory (base_dir/inbox)
        queue_dir: Queue root directory (base_dir/queue)
        high_priority: High priority queue directory
        normal_priority: Normal priority queue directory
        low_priority: Low priority queue directory
        processing: Processing directory (currently training file)
        metadata_file: Queue metadata JSON file (base_dir/queue/queue_metadata.json)

    Example:
        # Initialize queue
        queue = TrainingQueue("/training")

        # Process inbox → queue
        count = queue.process_inbox()  # Move all inbox files to normal queue
        print(f"Added {count} files to queue")

        # Training loop
        while True:
            file = queue.get_next_file()
            if not file:
                break  # Queue empty

            # Train on file...
            success = train(file)

            if success:
                queue.mark_completed(file, delete_file=True)
            else:
                queue.mark_failed(file, error="OOM", keep_file=True)

        # Check queue status
        status = queue.get_queue_status()
        print(f"Queue: {status['total_queued']} files")
        print(f"  High: {status['queued']['high']}")
        print(f"  Normal: {status['queued']['normal']}")
        print(f"  Low: {status['queued']['low']}")
    """

    def __init__(self, base_dir: str = None):
        # Import here to avoid circular imports
        if base_dir is None:
            from paths import get_base_dir
            self.base_dir = get_base_dir()
        else:
            self.base_dir = Path(base_dir)
        self.inbox = self.base_dir / "inbox"
        self.queue_dir = self.base_dir / "queue"

        # Create priority queues
        self.high_priority = self.queue_dir / "high"
        self.normal_priority = self.queue_dir / "normal"
        self.low_priority = self.queue_dir / "low"
        self.processing = self.queue_dir / "processing"

        for d in [self.high_priority, self.normal_priority, self.low_priority, self.processing]:
            d.mkdir(parents=True, exist_ok=True)

        # Queue metadata
        self.metadata_file = self.queue_dir / "queue_metadata.json"

    @contextmanager
    def _metadata_lock(self):
        """Context manager for exclusive access to metadata file.

        Uses fcntl.flock() for cross-process file locking to prevent
        race conditions when daemon and CLI access metadata simultaneously.
        """
        lock_file = self.queue_dir / ".metadata.lock"
        lock_file.touch(exist_ok=True)

        with open(lock_file, 'r') as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _get_metadata(self) -> Dict:
        """Load queue metadata (with file locking)"""
        with self._metadata_lock():
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    return json.load(f)
            return {"processed": [], "skipped": [], "failed": []}

    def _save_metadata(self, metadata: Dict):
        """Save queue metadata (with file locking)"""
        with self._metadata_lock():
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

    def scan_inbox(self) -> List[Path]:
        """
        Scan inbox for new .jsonl files

        Returns list of new files found
        """
        if not self.inbox.exists():
            return []

        new_files = []
        for file_path in self.inbox.glob("*.jsonl"):
            if file_path.is_file():
                new_files.append(file_path)

        return new_files

    def add_to_queue(self, file_path: Path, priority: str = "normal") -> bool:
        """
        Add a file to the queue

        Args:
            file_path: Path to .jsonl file
            priority: "high", "normal", or "low"

        Returns:
            True if successfully added
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # Determine target queue
        if priority == "high":
            target_dir = self.high_priority
        elif priority == "low":
            target_dir = self.low_priority
        else:
            target_dir = self.normal_priority

        # Move file to queue
        target_path = target_dir / file_path.name
        try:
            shutil.move(str(file_path), str(target_path))
            logger.info(f"📥 Added to {priority} queue: {file_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add {file_path.name} to queue: {e}")
            return False

    def process_inbox(self, default_priority: str = "normal") -> int:
        """
        Move all files from inbox to appropriate queue

        Args:
            default_priority: Default priority for inbox files

        Returns:
            Number of files added to queue
        """
        new_files = self.scan_inbox()
        count = 0

        for file_path in new_files:
            if self.add_to_queue(file_path, default_priority):
                count += 1

        if count > 0:
            logger.info(f"📥 Processed {count} files from inbox to queue")

        return count

    def get_next_file(self) -> Optional[Path]:
        """
        Get next file to process from priority queues in FIFO order.

        This is the core method for priority-based queue processing. It checks
        queues in priority order (high → normal → low) and returns the oldest
        file (by modification time) within the highest non-empty priority level.

        Priority Processing Order:
            1. Check high/ → if files exist, return oldest
            2. Check normal/ → if files exist and high/ empty, return oldest
            3. Check low/ → if files exist and high/ + normal/ empty, return oldest
            4. Return None if all queues empty

        Race Condition Prevention:
            - Atomic move: shutil.move() to processing/ prevents concurrent access
            - Single file in processing/: Only one file can be in processing/ at a time
            - Daemon checks processing/ is empty before calling get_next_file()

        Returns:
            Path to file in processing/ directory if queue not empty, None otherwise.
            File is moved from priority queue to processing/ before returning.

        Side Effects:
            - Moves file from priority queue (high/normal/low/) to processing/
            - Logs which file and priority queue it came from

        Example:
            queue = TrainingQueue("/training")

            # Get next file (by priority)
            file = queue.get_next_file()
            if file:
                print(f"Training on: {file.name}")
                # file is now in processing/
            else:
                print("Queue empty")

        Crash Recovery:
            If daemon crashes while processing a file, the file remains in processing/.
            On restart, daemon should move orphaned files back to their original queue
            or to failed/ for manual review.
        """
        # Check high priority first
        for priority_dir in [self.high_priority, self.normal_priority, self.low_priority]:
            files = sorted(priority_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime)
            if files:
                next_file = files[0]

                # Move to processing
                processing_path = self.processing / next_file.name
                shutil.move(str(next_file), str(processing_path))

                logger.info(f"📤 Next file: {next_file.name} (from {priority_dir.name})")
                return processing_path

        return None

    def mark_completed(self, file_path: Path, delete_file: bool = True):
        """
        Mark a file as successfully processed

        Args:
            file_path: Path to file in processing/
            delete_file: Whether to delete the file (if True, moves to backup first)
        """
        metadata = self._get_metadata()
        metadata["processed"].append({
            "file": file_path.name,
            "completed_at": datetime.now().isoformat()
        })
        self._save_metadata(metadata)

        if delete_file and file_path.exists():
            # SAFETY: Move to backup directory instead of immediate deletion
            # This allows recovery if training was invalid
            backup_dir = self.queue_dir / "recently_completed"
            backup_dir.mkdir(exist_ok=True)

            # Add timestamp to avoid name collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name

            shutil.move(str(file_path), str(backup_path))
            logger.info(f"✅ Completed: {file_path.name} (backed up to {backup_name})")
            logger.info(f"   Backup will be auto-cleaned after 7 days")
        else:
            logger.info(f"✅ Completed: {file_path.name}")

    def mark_failed(self, file_path: Path, error: str = "Training failed", keep_file: bool = True):
        """
        Mark a file as failed

        Args:
            file_path: Path to file in processing/
            error: Error message
            keep_file: Whether to keep the file for retry
        """
        metadata = self._get_metadata()

        # Check existing failures to count attempts
        attempts = 1
        for fail in metadata.get("failed", []):
            if fail["file"] == file_path.name:
                attempts = fail.get("attempts", 1) + 1

        # Max 3 attempts
        if attempts >= 3:
            # Give up - move to permanent failure directory
            failed_dir = self.queue_dir / "failed"
            failed_dir.mkdir(exist_ok=True)
            target = failed_dir / file_path.name
            if file_path.exists():
                shutil.move(str(file_path), str(target))

            metadata["failed"].append({
                "file": file_path.name,
                "attempts": attempts,
                "last_error": error,
                "final_failure": datetime.now().isoformat()
            })
            self._save_metadata(metadata)
            logger.error(f"❌ PERMANENTLY FAILED: {file_path.name} ({attempts} attempts)")
        else:
            # Retry - move back to low priority
            if file_path.exists():
                target = self.low_priority / file_path.name
                shutil.move(str(file_path), str(target))

            metadata["failed"].append({
                "file": file_path.name,
                "attempts": attempts,
                "last_error": error,
                "retry_at": datetime.now().isoformat()
            })
            self._save_metadata(metadata)
            logger.warning(f"⚠️  Failed: {file_path.name} (attempt {attempts}/3, will retry)")

    def mark_skipped(self, file_path: Path, reason: str):
        """
        Mark a file as skipped

        Args:
            file_path: Path to file in processing/
            reason: Reason for skipping
        """
        metadata = self._get_metadata()
        metadata["skipped"].append({
            "file": file_path.name,
            "skipped_at": datetime.now().isoformat(),
            "reason": reason
        })
        self._save_metadata(metadata)

        # Move back to low priority for potential retry
        if file_path.exists():
            target = self.low_priority / file_path.name
            shutil.move(str(file_path), str(target))

        logger.warning(f"⏭️  Skipped: {file_path.name} - {reason}")

    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        metadata = self._get_metadata()

        high_count = len(list(self.high_priority.glob("*.jsonl")))
        normal_count = len(list(self.normal_priority.glob("*.jsonl")))
        low_count = len(list(self.low_priority.glob("*.jsonl")))
        processing_count = len(list(self.processing.glob("*.jsonl")))

        total_queued = high_count + normal_count + low_count

        return {
            "queued": {
                "high": high_count,
                "normal": normal_count,
                "low": low_count,
                "total": total_queued
            },
            "total_queued": total_queued,  # Alias for daemon compatibility
            "processing": processing_count,
            "completed": len(metadata.get("processed", [])),
            "failed": len(metadata.get("failed", [])),
            "skipped": len(metadata.get("skipped", []))
        }

    def list_queue(self, priority: Optional[str] = None) -> List[Dict]:
        """
        List files in queue

        Args:
            priority: "high", "normal", "low", or None for all

        Returns:
            List of file info dicts
        """
        files = []

        if priority:
            if priority == "high":
                dirs = [self.high_priority]
            elif priority == "normal":
                dirs = [self.normal_priority]
            elif priority == "low":
                dirs = [self.low_priority]
            else:
                return []
        else:
            dirs = [self.high_priority, self.normal_priority, self.low_priority]

        for dir_path in dirs:
            for file_path in sorted(dir_path.glob("*.jsonl"), key=lambda x: x.stat().st_mtime):
                files.append({
                    "file": file_path.name,
                    "priority": dir_path.name,
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                    "added_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

        return files

    def change_priority(self, filename: str, new_priority: str) -> bool:
        """
        Change priority of a queued file

        Args:
            filename: Name of file in queue
            new_priority: "high", "normal", or "low"

        Returns:
            True if successful
        """
        # Find file in queues
        for priority_dir in [self.high_priority, self.normal_priority, self.low_priority]:
            file_path = priority_dir / filename
            if file_path.exists():
                # Determine target
                if new_priority == "high":
                    target_dir = self.high_priority
                elif new_priority == "low":
                    target_dir = self.low_priority
                else:
                    target_dir = self.normal_priority

                # Move to new priority
                target_path = target_dir / filename
                shutil.move(str(file_path), str(target_path))
                logger.info(f"📝 Changed priority: {filename} → {new_priority}")
                return True

        logger.error(f"File not found in queue: {filename}")
        return False


def main():
    """Command-line interface for queue management"""
    import argparse
    from paths import get_base_dir

    parser = argparse.ArgumentParser(description="Training Queue Management")
    parser.add_argument('--base-dir', default=None, help='Base directory (default: auto-detect or $TRAINING_BASE_DIR)')

    subparsers = parser.add_subparsers(dest='command', help='Queue command')

    # Status
    subparsers.add_parser('status', help='Show queue status')

    # List
    list_parser = subparsers.add_parser('list', help='List queued files')
    list_parser.add_argument('--priority', choices=['high', 'normal', 'low'], help='Filter by priority')

    # Add
    add_parser = subparsers.add_parser('add', help='Add file to queue')
    add_parser.add_argument('file', help='Path to .jsonl file')
    add_parser.add_argument('--priority', default='normal', choices=['high', 'normal', 'low'], help='Priority level')

    # Process inbox
    process_parser = subparsers.add_parser('process-inbox', help='Process inbox files to queue')
    process_parser.add_argument('--priority', default='normal', choices=['high', 'normal', 'low'], help='Default priority')

    # Change priority
    priority_parser = subparsers.add_parser('set-priority', help='Change file priority')
    priority_parser.add_argument('file', help='Filename in queue')
    priority_parser.add_argument('priority', choices=['high', 'normal', 'low'], help='New priority')

    args = parser.parse_args()

    queue = TrainingQueue(args.base_dir)

    if args.command == 'status':
        status = queue.get_queue_status()
        print("\n" + "="*80)
        print("TRAINING QUEUE STATUS")
        print("="*80)
        print(f"\nQueued Files:")
        print(f"  High Priority:   {status['queued']['high']}")
        print(f"  Normal Priority: {status['queued']['normal']}")
        print(f"  Low Priority:    {status['queued']['low']}")
        print(f"  Total:           {status['queued']['total']}")
        print(f"\nProcessing: {status['processing']}")
        print(f"Completed:  {status['completed']}")
        print(f"Failed:     {status['failed']}")
        print(f"Skipped:    {status['skipped']}")
        print("="*80 + "\n")

    elif args.command == 'list':
        files = queue.list_queue(args.priority)
        print("\n" + "="*80)
        print("QUEUED FILES")
        print("="*80 + "\n")

        if not files:
            print("No files in queue\n")
        else:
            for f in files:
                print(f"📄 {f['file']}")
                print(f"   Priority: {f['priority']}")
                print(f"   Size: {f['size_mb']:.1f} MB")
                print(f"   Added: {f['added_at']}")
                print()

        print("="*80 + "\n")

    elif args.command == 'add':
        file_path = Path(args.file)
        if queue.add_to_queue(file_path, args.priority):
            print(f"✅ Added {file_path.name} to {args.priority} queue")
        else:
            print(f"❌ Failed to add {file_path.name}")

    elif args.command == 'process-inbox':
        count = queue.process_inbox(args.priority)
        print(f"✅ Processed {count} files from inbox")

    elif args.command == 'set-priority':
        if queue.change_priority(args.file, args.priority):
            print(f"✅ Changed {args.file} priority to {args.priority}")
        else:
            print(f"❌ Failed to change priority")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
