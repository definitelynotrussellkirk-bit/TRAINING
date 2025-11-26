#!/usr/bin/env python3
"""
Anti-Stuck Monitor - Detects and recovers from training hangs

This specifically addresses:
1. Stuck at eval steps (like the 2500 crash)
2. Infinite loops or deadlocks
3. CUDA hangs
4. I/O hangs

Usage:
    python3 anti_stuck_monitor.py [--timeout 900]
"""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime

# Use centralized path resolution instead of hard-coded paths
try:
    from core.paths import get_base_dir
    BASE_DIR = get_base_dir()
except ImportError:
    BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of safety/

STATUS_FILE = BASE_DIR / "status" / "training_status.json"
PID_FILE = BASE_DIR / ".daemon.pid"
LOG_FILE = BASE_DIR / "logs" / "anti_stuck.log"

# Configuration
CHECK_INTERVAL = 60  # Check every 60 seconds
STUCK_TIMEOUT = 900  # 15 minutes without progress = stuck
EVAL_TIMEOUT = 1800  # 30 minutes at same eval step = stuck at eval
MAX_KILL_ATTEMPTS = 3

class StuckMonitor:
    def __init__(self, stuck_timeout=STUCK_TIMEOUT):
        self.stuck_timeout = stuck_timeout
        self.last_step = None
        self.last_step_time = None
        self.same_step_count = 0
        self.last_eval_step = None
        self.eval_start_time = None

        # Setup logging
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(LOG_FILE, 'a')

    def log(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"
        self.log_file.write(log_line)
        self.log_file.flush()
        print(log_line.strip())

    def get_status(self):
        """Get current training status"""
        try:
            if STATUS_FILE.exists():
                with open(STATUS_FILE) as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"⚠️ Error reading status: {e}")
        return None

    def get_training_pid(self):
        """Get training process PID"""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'train.py'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = [int(p) for p in result.stdout.strip().split('\n') if p]
                return pids[0] if pids else None
        except:
            pass
        return None

    def is_at_eval_step(self, step):
        """Check if step is an eval step (multiples of 500)"""
        return step % 500 == 0

    def kill_stuck_process(self):
        """Kill stuck training process"""
        pid = self.get_training_pid()
        if not pid:
            self.log("❌ No training process found to kill")
            return False

        self.log(f"🔪 Killing stuck training process (PID: {pid})")

        for attempt in range(MAX_KILL_ATTEMPTS):
            try:
                # Try SIGTERM first
                os.kill(pid, signal.SIGTERM)
                time.sleep(5)

                # Check if still alive
                try:
                    os.kill(pid, 0)
                    # Still alive, try SIGKILL
                    if attempt == MAX_KILL_ATTEMPTS - 1:
                        self.log(f"⚠️ SIGTERM failed, using SIGKILL")
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(2)
                except OSError:
                    # Process is dead
                    self.log(f"✅ Process killed successfully")
                    return True

            except Exception as e:
                self.log(f"❌ Kill attempt {attempt+1} failed: {e}")

        return False

    def check_for_stuck(self):
        """Check if training is stuck"""
        status = self.get_status()
        if not status:
            return False

        current_step = status.get('current_step', 0)
        current_time = datetime.now()

        # First time or step changed
        if self.last_step is None or current_step != self.last_step:
            self.last_step = current_step
            self.last_step_time = current_time
            self.same_step_count = 0

            # Check if at eval step
            if self.is_at_eval_step(current_step):
                if self.last_eval_step != current_step:
                    self.log(f"📊 Started eval at step {current_step}")
                    self.last_eval_step = current_step
                    self.eval_start_time = current_time
            else:
                self.last_eval_step = None
                self.eval_start_time = None

            return False

        # Step hasn't changed
        self.same_step_count += 1
        elapsed = (current_time - self.last_step_time).total_seconds()

        # Check if stuck at eval
        if self.is_at_eval_step(current_step) and self.eval_start_time:
            eval_elapsed = (current_time - self.eval_start_time).total_seconds()
            if eval_elapsed > EVAL_TIMEOUT:
                self.log(f"🚨 STUCK AT EVAL STEP {current_step} for {eval_elapsed:.0f}s")
                self.log(f"   This is likely the same issue that caused the previous crash!")
                return True
            elif eval_elapsed > EVAL_TIMEOUT / 2:
                self.log(f"⚠️ Eval taking long at step {current_step}: {eval_elapsed:.0f}s (timeout: {EVAL_TIMEOUT}s)")

        # Check if stuck in general
        if elapsed > self.stuck_timeout:
            self.log(f"🚨 STUCK at step {current_step} for {elapsed:.0f}s (timeout: {self.stuck_timeout}s)")
            self.log(f"   No progress for {elapsed/60:.1f} minutes")
            return True
        elif elapsed > self.stuck_timeout / 2:
            self.log(f"⚠️ Slow progress: {elapsed:.0f}s at step {current_step}")

        return False

    def monitor(self):
        """Main monitoring loop"""
        self.log("="*80)
        self.log("🚫 Anti-Stuck Monitor Starting")
        self.log("="*80)
        self.log(f"Check interval: {CHECK_INTERVAL}s")
        self.log(f"Stuck timeout: {self.stuck_timeout}s ({self.stuck_timeout/60:.0f} min)")
        self.log(f"Eval timeout: {EVAL_TIMEOUT}s ({EVAL_TIMEOUT/60:.0f} min)")
        self.log("")

        try:
            while True:
                if self.check_for_stuck():
                    self.log("💥 STUCK DETECTED - Taking action!")

                    # Kill stuck process
                    if self.kill_stuck_process():
                        self.log("✅ Stuck process killed")
                        self.log("⏳ Waiting 30s for watchdog to restart...")
                        time.sleep(30)

                        # Reset tracking
                        self.last_step = None
                        self.last_step_time = None
                        self.same_step_count = 0
                        self.last_eval_step = None
                        self.eval_start_time = None
                    else:
                        self.log("❌ Failed to kill stuck process - manual intervention needed")
                        break

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            self.log("\n⏹️ Monitor stopped by user")
        except Exception as e:
            self.log(f"❌ Monitor error: {e}")
        finally:
            self.log_file.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor for stuck training")
    parser.add_argument('--timeout', type=int, default=STUCK_TIMEOUT,
                        help=f"Stuck timeout in seconds (default: {STUCK_TIMEOUT})")
    args = parser.parse_args()

    monitor = StuckMonitor(stuck_timeout=args.timeout)
    monitor.monitor()

if __name__ == '__main__':
    main()
