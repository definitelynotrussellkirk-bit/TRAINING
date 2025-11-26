#!/usr/bin/env python3
"""
Daemon Watchdog - Monitors training daemon health and auto-restarts on crash

This watchdog runs continuously and:
1. Checks if daemon is alive
2. Checks if training is progressing
3. Detects crashes and hangs
4. Auto-restarts daemon on failure
5. Logs all events for debugging

Usage:
    python3 daemon_watchdog.py

Or run as systemd service for automatic startup.
"""

import os
import sys
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Use centralized path resolution instead of hard-coded paths
try:
    from core.paths import get_base_dir
    BASE_DIR = get_base_dir()
except ImportError:
    BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of safety/

# Configuration
PID_FILE = BASE_DIR / ".daemon.pid"
STATUS_FILE = BASE_DIR / "status" / "training_status.json"
LOG_FILE = BASE_DIR / "logs" / "watchdog.log"
TRAINING_OUTPUT_LOG = BASE_DIR / "training_output.log"

# Monitoring thresholds
CHECK_INTERVAL = 30  # Check every 30 seconds
PROGRESS_TIMEOUT = 600  # 10 minutes without progress = hung
DAEMON_START_TIMEOUT = 60  # 60 seconds to start daemon
MAX_RESTART_ATTEMPTS = 3  # Max restarts in RESTART_WINDOW
RESTART_WINDOW = 300  # 5 minutes

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

class WatchdogState:
    """Track watchdog state across checks"""
    def __init__(self):
        self.last_step = None
        self.last_step_time = None
        self.restart_attempts = []
        self.consecutive_failures = 0

    def record_restart(self):
        """Record a restart attempt"""
        now = datetime.now()
        self.restart_attempts.append(now)
        # Remove old attempts outside window
        cutoff = now - timedelta(seconds=RESTART_WINDOW)
        self.restart_attempts = [t for t in self.restart_attempts if t > cutoff]

    def too_many_restarts(self):
        """Check if we've restarted too many times"""
        return len(self.restart_attempts) >= MAX_RESTART_ATTEMPTS

    def update_progress(self, step):
        """Update progress tracking"""
        if step != self.last_step:
            self.last_step = step
            self.last_step_time = datetime.now()
            self.consecutive_failures = 0

    def is_hung(self):
        """Check if training appears hung"""
        if self.last_step_time is None:
            return False
        elapsed = (datetime.now() - self.last_step_time).total_seconds()
        return elapsed > PROGRESS_TIMEOUT


def get_daemon_pid():
    """Get daemon PID from PID file"""
    try:
        if PID_FILE.exists():
            return int(PID_FILE.read_text().strip())
    except:
        pass
    return None


def is_process_alive(pid):
    """Check if process with given PID is alive"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def get_training_status():
    """Get current training status"""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                return json.load(f)
    except:
        pass
    return None


def kill_orphaned_processes():
    """Kill any orphaned training processes"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'train.py'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    logging.warning(f"Killing orphaned training process: {pid}")
                    subprocess.run(['kill', pid])
    except Exception as e:
        logging.error(f"Error killing orphaned processes: {e}")


def start_daemon():
    """Start the training daemon"""
    # Clean up first
    if PID_FILE.exists():
        PID_FILE.unlink()

    # Kill orphaned processes
    kill_orphaned_processes()
    time.sleep(2)

    # Start daemon
    cmd = [
        'nohup', 'python3', 'training_daemon.py',
        '--base-dir', str(BASE_DIR)
    ]

    with open(TRAINING_OUTPUT_LOG, 'a') as f:
        subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=BASE_DIR,
            start_new_session=True
        )

    logging.info("Daemon start command issued, waiting for startup...")

    # Wait for daemon to start
    start_time = time.time()
    while time.time() - start_time < DAEMON_START_TIMEOUT:
        pid = get_daemon_pid()
        if pid and is_process_alive(pid):
            logging.info(f"✅ Daemon started successfully (PID: {pid})")
            return True
        time.sleep(2)

    logging.error("❌ Daemon failed to start within timeout")
    return False


def check_daemon_health(state):
    """
    Check daemon health and take action if needed

    Returns:
        str: Status message ("healthy", "restarted", "failed", "skipped")
    """
    # Check if daemon process is alive
    pid = get_daemon_pid()
    if not pid or not is_process_alive(pid):
        logging.warning(f"⚠️  Daemon not running (PID: {pid})")

        # Check restart limits
        if state.too_many_restarts():
            logging.error(f"❌ Too many restarts ({len(state.restart_attempts)} in {RESTART_WINDOW}s) - stopping watchdog")
            return "failed"

        # Attempt restart
        logging.info("🔄 Attempting to restart daemon...")
        state.record_restart()

        if start_daemon():
            state.consecutive_failures = 0
            return "restarted"
        else:
            state.consecutive_failures += 1
            return "failed"

    # Daemon is alive, check if training is progressing and status is fresh
    status = get_training_status()
    if status:
        # Stale status check
        try:
            mtime = STATUS_FILE.stat().st_mtime
            age = time.time() - mtime
            if age > PROGRESS_TIMEOUT:
                logging.warning(f"⚠️  Status file stale for {int(age)}s; restarting daemon")
                if state.too_many_restarts():
                    logging.error(f"❌ Too many restarts - stopping watchdog")
                    return "failed"
                state.record_restart()
                subprocess.run(['kill', str(pid)])
                time.sleep(5)
                start_daemon()
                return "restarted"
        except Exception as e:
            logging.warning(f"Could not check status file age: {e}")

        current_step = status.get('current_step', 0)
        state.update_progress(current_step)

        if state.is_hung():
            logging.warning(f"⚠️  Training appears hung at step {current_step} for {PROGRESS_TIMEOUT}s")

            # Check restart limits
            if state.too_many_restarts():
                logging.error(f"❌ Too many restarts - stopping watchdog")
                return "failed"

            # Restart daemon
            logging.info("🔄 Restarting hung daemon...")
            subprocess.run(['kill', str(pid)])
            time.sleep(5)

            state.record_restart()
            if start_daemon():
                state.consecutive_failures = 0
                state.last_step_time = datetime.now()  # Reset hung timer
                return "restarted"
            else:
                state.consecutive_failures += 1
                return "failed"

    return "healthy"


def main():
    """Main watchdog loop"""
    logging.info("=" * 80)
    logging.info("🐕 Daemon Watchdog Starting")
    logging.info("=" * 80)
    logging.info(f"Base directory: {BASE_DIR}")
    logging.info(f"Check interval: {CHECK_INTERVAL}s")
    logging.info(f"Progress timeout: {PROGRESS_TIMEOUT}s")
    logging.info(f"Max restarts: {MAX_RESTART_ATTEMPTS} in {RESTART_WINDOW}s")
    logging.info("")

    state = WatchdogState()
    check_count = 0

    try:
        while True:
            check_count += 1

            # Run health check
            result = check_daemon_health(state)

            # Log status
            if result == "healthy":
                status = get_training_status()
                if status:
                    step = status.get('current_step', 0)
                    total = status.get('total_steps', 0)
                    loss = status.get('loss', 0)
                    logging.info(f"✅ Healthy - Step {step}/{total}, Loss: {loss:.4f}")
                else:
                    logging.info("✅ Daemon healthy (no training status yet)")

            elif result == "restarted":
                logging.info(f"🔄 Daemon restarted (attempt {len(state.restart_attempts)}/{MAX_RESTART_ATTEMPTS})")

            elif result == "failed":
                if state.too_many_restarts():
                    logging.error("❌ CRITICAL: Too many restart failures - watchdog exiting")
                    logging.error("   Manual intervention required!")
                    break
                else:
                    logging.error(f"❌ Restart failed (consecutive failures: {state.consecutive_failures})")

            # Periodic status summary
            if check_count % 20 == 0:  # Every 10 minutes
                logging.info(f"📊 Status: {len(state.restart_attempts)} restarts in last {RESTART_WINDOW}s")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        logging.info("\n⏹️  Watchdog stopped by user")
    except Exception as e:
        logging.error(f"❌ Watchdog error: {e}", exc_info=True)

    logging.info("🐕 Daemon Watchdog Exiting")


if __name__ == '__main__':
    main()
