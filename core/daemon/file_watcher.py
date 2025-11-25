#!/usr/bin/env python3
"""
File Watcher - Monitor directory for new files.

This module provides a simple file watcher that tracks new files
appearing in a directory. Designed for the inbox monitoring use case
in the training daemon.

Usage:
    from daemon.file_watcher import FileWatcher

    watcher = FileWatcher(Path("inbox"), pattern="*.jsonl")

    # Get new files since last check
    new_files = watcher.get_new_files()
    for file in new_files:
        process(file)
"""

import logging
from pathlib import Path
from typing import List, Set, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FileWatcher:
    """
    Watches a directory for new files matching a pattern.

    Tracks files that have been seen and returns only new files
    on each check. Files are returned sorted by modification time
    (oldest first) for FIFO processing.

    Attributes:
        watch_dir: Directory to monitor
        pattern: Glob pattern for files to watch (default: "*.jsonl")
        _seen: Set of file paths that have been returned

    Example:
        watcher = FileWatcher(Path("inbox"))

        # First check - returns all existing files
        files = watcher.get_new_files()
        print(f"Found {len(files)} files")

        # Later check - returns only new files
        new_files = watcher.get_new_files()
        print(f"Found {len(new_files)} new files")

        # Mark file as processed (won't appear again)
        watcher.mark_processed(files[0])
    """

    def __init__(self, watch_dir: Path, pattern: str = "*.jsonl"):
        """
        Initialize file watcher.

        Args:
            watch_dir: Directory to monitor
            pattern: Glob pattern for files to watch
        """
        self.watch_dir = Path(watch_dir)
        self.pattern = pattern
        self._seen: Set[Path] = set()

    def get_new_files(self) -> List[Path]:
        """
        Get files that haven't been seen yet.

        Returns:
            List of new file paths, sorted by modification time (oldest first)

        Side Effects:
            - Updates internal _seen set with returned files
        """
        if not self.watch_dir.exists():
            return []

        current_files = set(self.watch_dir.glob(self.pattern))
        new_files = current_files - self._seen

        # Mark as seen
        self._seen.update(new_files)

        # Return sorted by modification time (oldest first = FIFO)
        return sorted(new_files, key=lambda p: p.stat().st_mtime)

    def get_all_files(self) -> List[Path]:
        """
        Get all files matching pattern (regardless of seen status).

        Returns:
            List of all file paths matching pattern, sorted by mtime
        """
        if not self.watch_dir.exists():
            return []

        files = list(self.watch_dir.glob(self.pattern))
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def mark_processed(self, file_path: Path) -> None:
        """
        Mark a file as processed (won't appear in get_new_files).

        Args:
            file_path: Path to mark as processed
        """
        self._seen.add(Path(file_path))

    def mark_removed(self, file_path: Path) -> None:
        """
        Remove a file from the seen set (e.g., after it's deleted).

        Args:
            file_path: Path to remove from seen set
        """
        self._seen.discard(Path(file_path))

    def reset(self) -> None:
        """Clear the seen set, treating all files as new on next check."""
        self._seen.clear()

    def count_unseen(self) -> int:
        """
        Count files that haven't been seen yet.

        Returns:
            Number of new files waiting to be processed
        """
        if not self.watch_dir.exists():
            return 0

        current_files = set(self.watch_dir.glob(self.pattern))
        return len(current_files - self._seen)

    def get_stats(self) -> dict:
        """
        Get watcher statistics.

        Returns:
            Dict with stats: total files, seen count, unseen count
        """
        if not self.watch_dir.exists():
            return {
                "watch_dir": str(self.watch_dir),
                "pattern": self.pattern,
                "total_files": 0,
                "seen_count": len(self._seen),
                "unseen_count": 0
            }

        current_files = set(self.watch_dir.glob(self.pattern))
        return {
            "watch_dir": str(self.watch_dir),
            "pattern": self.pattern,
            "total_files": len(current_files),
            "seen_count": len(self._seen),
            "unseen_count": len(current_files - self._seen)
        }


class InboxFlattener:
    """
    Flattens files from subdirectories to inbox root.

    Some systems drop files into subdirectories of inbox.
    This class moves them to the root for uniform processing.

    Example:
        flattener = InboxFlattener(Path("inbox"))
        moved = flattener.flatten()
        print(f"Moved {moved} files to inbox root")
    """

    def __init__(self, inbox_dir: Path, pattern: str = "*.jsonl"):
        """
        Initialize inbox flattener.

        Args:
            inbox_dir: Root inbox directory
            pattern: Pattern for files to flatten
        """
        self.inbox_dir = Path(inbox_dir)
        self.pattern = pattern

    def flatten(self) -> int:
        """
        Move files from subdirectories to inbox root.

        Files are renamed to include subdirectory name to avoid collisions:
            subdir/file.jsonl -> file_subdir.jsonl

        Returns:
            Number of files moved
        """
        if not self.inbox_dir.exists():
            return 0

        import shutil

        subdirs = [d for d in self.inbox_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return 0

        moved_count = 0

        for subdir in subdirs:
            # Find files recursively in subdir
            files = list(subdir.rglob(self.pattern))

            for file_path in files:
                # Construct new name at inbox root
                subdir_name = subdir.name
                original_name = file_path.stem
                new_name = f"{original_name}_{subdir_name}.jsonl"
                dest_path = self.inbox_dir / new_name

                # Handle collisions
                counter = 1
                while dest_path.exists():
                    new_name = f"{original_name}_{subdir_name}_{counter}.jsonl"
                    dest_path = self.inbox_dir / new_name
                    counter += 1

                try:
                    shutil.move(str(file_path), str(dest_path))
                    logger.info(f"Flattened: {file_path.name} -> {new_name}")
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Failed to flatten {file_path}: {e}")

            # Clean up empty subdir
            try:
                if subdir.exists() and not any(subdir.iterdir()):
                    subdir.rmdir()
                    logger.info(f"Removed empty subdir: {subdir.name}")
            except Exception as e:
                logger.warning(f"Could not remove subdir {subdir.name}: {e}")

        return moved_count


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        inbox = Path(tmpdir) / "inbox"
        inbox.mkdir()

        # Create test files
        (inbox / "file1.jsonl").write_text('{"test": 1}')
        (inbox / "file2.jsonl").write_text('{"test": 2}')

        print(f"Testing FileWatcher with {inbox}")

        watcher = FileWatcher(inbox)

        # First check - should find both files
        files = watcher.get_new_files()
        assert len(files) == 2, f"Expected 2 files, got {len(files)}"
        print(f"First check: found {len(files)} files")

        # Second check - should find nothing (already seen)
        files = watcher.get_new_files()
        assert len(files) == 0, f"Expected 0 files, got {len(files)}"
        print(f"Second check: found {len(files)} files (expected 0)")

        # Add new file
        (inbox / "file3.jsonl").write_text('{"test": 3}')

        # Third check - should find only new file
        files = watcher.get_new_files()
        assert len(files) == 1, f"Expected 1 file, got {len(files)}"
        assert files[0].name == "file3.jsonl"
        print(f"Third check: found {len(files)} new file(s)")

        # Test stats
        stats = watcher.get_stats()
        print(f"Stats: {stats}")

        # Test flattener
        subdir = inbox / "batch1"
        subdir.mkdir()
        (subdir / "nested.jsonl").write_text('{"nested": true}')

        flattener = InboxFlattener(inbox)
        moved = flattener.flatten()
        assert moved == 1, f"Expected 1 moved, got {moved}"
        assert (inbox / "nested_batch1.jsonl").exists()
        print(f"Flattener: moved {moved} file(s)")

        print("\nAll tests passed!")
