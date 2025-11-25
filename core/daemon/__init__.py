"""
Daemon services package.

Extracted components from training_daemon.py for better testability
and separation of concerns.

Components:
    - PIDManager: Single-instance enforcement via PID files
    - FileWatcher: Monitor directories for new files
    - InboxFlattener: Move files from subdirs to inbox root
    - SnapshotService: Create and verify model snapshots
"""

from .pid_manager import PIDManager
from .file_watcher import FileWatcher, InboxFlattener
from .snapshot_service import SnapshotService, SnapshotConfig, SnapshotResult

__all__ = [
    "PIDManager", "FileWatcher", "InboxFlattener",
    "SnapshotService", "SnapshotConfig", "SnapshotResult",
]
