"""
Daemon services package.

Extracted components from training_daemon.py for better testability
and separation of concerns.

Components:
    - PIDManager: Single-instance enforcement via PID files
    - FileWatcher: Monitor directories for new files
    - InboxFlattener: Move files from subdirs to inbox root
"""

from .pid_manager import PIDManager
from .file_watcher import FileWatcher, InboxFlattener

__all__ = ["PIDManager", "FileWatcher", "InboxFlattener"]
