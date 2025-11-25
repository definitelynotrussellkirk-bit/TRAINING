#!/usr/bin/env python3
"""
PID Manager - Single-instance enforcement for daemons.

This module provides PID file locking to ensure only one instance
of a daemon runs at a time. Supports context manager usage for
automatic cleanup.

Usage:
    from daemon.pid_manager import PIDManager

    # Context manager (recommended)
    with PIDManager(Path(".pids/daemon.pid")) as pm:
        # Daemon runs here
        pass  # Lock auto-released on exit

    # Manual usage
    pm = PIDManager(Path(".pids/daemon.pid"))
    if pm.acquire():
        try:
            # Daemon runs here
            pass
        finally:
            pm.release()
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PIDManager:
    """
    Manages PID file for single-instance enforcement.

    Attributes:
        pid_file: Path to the PID lock file
        _acquired: Whether lock is currently held

    Example:
        # Using context manager
        with PIDManager(Path("/var/run/myapp.pid")) as pm:
            run_daemon()  # Only one instance can reach here

        # Manual acquire/release
        pm = PIDManager(Path("/var/run/myapp.pid"))
        if pm.acquire():
            try:
                run_daemon()
            finally:
                pm.release()
    """

    def __init__(self, pid_file: Path):
        """
        Initialize PID manager.

        Args:
            pid_file: Path to the PID lock file
        """
        self.pid_file = Path(pid_file)
        self._acquired = False

    def acquire(self) -> bool:
        """
        Acquire PID lock.

        Returns:
            True if lock acquired, False if another instance is running

        Side Effects:
            - Writes current PID to pid_file
            - Removes stale PID files from dead processes
        """
        # Ensure parent directory exists
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

        if self.pid_file.exists():
            try:
                old_pid = int(self.pid_file.read_text().strip())
                if self._is_running(old_pid):
                    logger.error(f"Another instance is running (PID {old_pid})")
                    return False
                # Stale PID file from crashed process
                logger.warning(f"Removing stale PID file (old PID: {old_pid})")
                self.pid_file.unlink()
            except (ValueError, OSError) as e:
                logger.warning(f"Error reading PID file: {e}, removing it")
                try:
                    self.pid_file.unlink()
                except OSError:
                    pass

        # Write our PID
        try:
            self.pid_file.write_text(str(os.getpid()))
            self._acquired = True
            logger.info(f"Acquired daemon lock (PID: {os.getpid()})")
            return True
        except OSError as e:
            logger.error(f"Failed to write PID file: {e}")
            return False

    def release(self) -> None:
        """
        Release PID lock.

        Side Effects:
            - Deletes pid_file if it exists
        """
        if self.pid_file.exists():
            try:
                # Only delete if we own it
                current_pid = int(self.pid_file.read_text().strip())
                if current_pid == os.getpid():
                    self.pid_file.unlink()
                    logger.info("Released daemon lock")
            except (ValueError, OSError) as e:
                logger.warning(f"Error releasing PID file: {e}")
        self._acquired = False

    def _is_running(self, pid: int) -> bool:
        """
        Check if a process with given PID is running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running, False otherwise
        """
        try:
            os.kill(pid, 0)  # Signal 0 = check existence
            return True
        except OSError:
            return False

    def is_acquired(self) -> bool:
        """Check if lock is currently held by this instance."""
        return self._acquired

    def get_running_pid(self) -> Optional[int]:
        """
        Get PID of running instance if one exists.

        Returns:
            PID of running instance, or None if no instance running
        """
        if not self.pid_file.exists():
            return None

        try:
            pid = int(self.pid_file.read_text().strip())
            if self._is_running(pid):
                return pid
        except (ValueError, OSError):
            pass

        return None

    def __enter__(self) -> "PIDManager":
        """Context manager entry - acquire lock or raise error."""
        if not self.acquire():
            running_pid = self.get_running_pid()
            raise RuntimeError(
                f"Another daemon instance is running (PID {running_pid})"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release lock."""
        self.release()


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        pid_file = Path(tmpdir) / "test.pid"

        print(f"Testing PIDManager with {pid_file}")

        # Test basic acquire/release
        pm = PIDManager(pid_file)
        assert pm.acquire(), "Should acquire lock"
        assert pid_file.exists(), "PID file should exist"
        assert pm.is_acquired(), "Should report acquired"

        # Test double acquire fails
        pm2 = PIDManager(pid_file)
        assert not pm2.acquire(), "Should not acquire (already locked)"

        pm.release()
        assert not pid_file.exists(), "PID file should be deleted"

        # Test context manager
        with PIDManager(pid_file) as pm3:
            assert pm3.is_acquired()

        assert not pid_file.exists()

        print("All tests passed!")
