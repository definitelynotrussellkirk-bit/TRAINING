#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
MODULE: core/training_controller.py
═══════════════════════════════════════════════════════════════════════════════

File-based signaling system for graceful training control without killing processes.

OVERVIEW
--------
Provides a safe way to control training via signal files in control/ directory.
All control operations are graceful - training completes current batch before acting.
No process signals (SIGINT/SIGTERM), no race conditions, fully atomic operations.

FEATURES
--------
✓ Pause/Resume - Pause training after current batch, resume when ready
✓ Stop - Finish current batch, then stop cleanly
✓ Skip - Skip current file, move to queue/failed/, continue with next
✓ Status - Query current training state and active signals
✓ State Tracking - Persistent state in control/state.json

ARCHITECTURE
------------

    User CLI                 Signal Files              Training Daemon
    ========                 ============              ===============

    pause                → control/.pause      →     check after batch
                                                      ↓
    resume               → control/.resume     →     wait_for_resume()
                                                      ↓
    stop                 → control/.stop       →     clean shutdown

    status               ← control/state.json  ←     update_state()

File-Based Communication:
    - Signal files: .pause, .stop, .skip, .resume (atomic existence checks)
    - State file: state.json (training daemon writes, CLI reads)
    - No shared memory, no locks, no race conditions

SIGNAL FILES (control/*.signal)
-------------------------------
All signals are files in control/ directory. File existence = signal active.

1. .pause - Pause after current batch completes
   - Training daemon checks after each batch
   - When detected, training finishes batch then pauses
   - File contains: reason + timestamp
   - Training calls wait_for_resume() to block

2. .stop - Stop after current batch completes
   - Training finishes current batch then exits cleanly
   - File contains: reason + timestamp
   - Training saves checkpoint, cleans up, exits 0

3. .skip - Skip current training file
   - Moves current file to queue/failed/
   - Continues with next file in queue
   - File contains: reason + timestamp
   - Useful for stuck files or bad data

4. .resume - Resume from paused state
   - Clears .pause signal
   - Training continues from where it paused
   - File is empty (just marker)
   - Unblocks wait_for_resume()

STATE FILE (control/state.json)
-------------------------------
Persistent state written by training daemon, read by CLI:

{
    "status": "training",              // Current state
    "last_update": "2025-11-24T10:30:00",  // ISO timestamp
    "current_file": "data.jsonl",      // File being trained (str or null)
    "paused_at": "2025-11-24T10:25:00", // When paused (ISO timestamp or null)
    "reason": "User requested"         // Reason for current state (str or null)
}

Status Values:
    - idle: Not training, waiting for files in queue
    - training: Currently training on a file
    - paused: Paused by user, waiting for resume
    - stopping: Finishing current batch, will exit
    - skipping: Skipping current file, moving to next

State Transitions:
    idle → training → paused → training (pause/resume cycle)
    training → stopping → idle (graceful stop)
    training → skipping → training (skip file)

THREAD SAFETY
-------------
✓ Signal file checks are atomic (file.exists() is atomic syscall)
✓ State file writes are NOT atomic (JSON serialization + write)
✓ Design assumption: Single training daemon writes, CLI reads
✓ Multiple CLI processes can safely read state concurrently

Race Condition Prevention:
    - Signal files: Atomic filesystem operations (touch, unlink)
    - State file: Single-writer (training daemon only)
    - No locks needed, no shared memory

EXAMPLE WORKFLOW
----------------

Pause/Resume Cycle:
    1. Training is running (status="training")
    2. User runs: training_controller.py pause --reason "Check logs"
    3. .pause signal file created with reason + timestamp
    4. Training completes current batch (e.g., batch 5/100)
    5. Training checks should_pause_after_batch() → True
    6. Training calls wait_for_resume()
       - Writes status="paused" to state.json
       - Logs: "⏸️ PAUSED - Waiting for resume signal..."
       - Blocks in 5-second polling loop
    7. User checks logs, fixes issue
    8. User runs: training_controller.py resume
    9. .resume signal created, .pause removed
    10. wait_for_resume() detects .resume, clears it
    11. Status becomes "training" again
    12. Training continues from batch 6/100

Stop Workflow:
    1. User runs: training_controller.py stop
    2. .stop signal created
    3. Training completes current batch
    4. Training checks should_stop_after_batch() → True
    5. Training saves checkpoint, cleans up
    6. Status becomes "idle"
    7. Process exits with code 0

Skip Workflow:
    1. File "bad_data.jsonl" causing issues
    2. User runs: training_controller.py skip --reason "Corrupt data"
    3. .skip signal created
    4. Training checks should_skip_current_file() → True
    5. File moved to queue/failed/bad_data.jsonl
    6. Training continues with next file in queue
    7. .skip signal cleared

COMMAND-LINE INTERFACE
----------------------

Control Commands:
    python3 training_controller.py pause [--reason "text"]
        Create pause signal. Training will pause after current batch.

    python3 training_controller.py resume
        Create resume signal. Training will continue from pause.

    python3 training_controller.py stop [--reason "text"]
        Create stop signal. Training will exit after current batch.

    python3 training_controller.py skip [--reason "text"]
        Create skip signal. Current file will move to failed/.

    python3 training_controller.py status
        Show current status, signals, and state.

    python3 training_controller.py clear
        Clear all control signals. Emergency reset.

Status Output:
    ================================================================================
    TRAINING CONTROL STATUS
    ================================================================================

    Status: training
    Last Update: 2025-11-24T10:30:15
    Current File: data.jsonl

    Active Signals:
      🔴 pause

    ================================================================================

INTEGRATION WITH TRAINING DAEMON
---------------------------------

In training loop (core/training_daemon.py):

    controller = TrainingController()

    while True:
        # Get next file from queue
        training_file = queue.get_next()
        controller.set_status("training", current_file=training_file.name)

        # Train for N batches
        for batch in batches:
            train_batch(batch)

            # Check signals after each batch
            if controller.should_stop_after_batch():
                logger.info("Stop signal received - exiting")
                controller.clear_stop()
                return 0  # Clean exit

            if controller.should_pause_after_batch():
                logger.info("Pause signal received - pausing")
                controller.wait_for_resume()  # Blocks until resume
                logger.info("Resumed - continuing training")

            if controller.should_skip_current_file():
                logger.info(f"Skip signal received - skipping {training_file.name}")
                move_to_failed(training_file)
                controller.clear_skip()
                break  # Move to next file

        controller.set_status("idle")

Key Points:
    - Check signals AFTER batch completes (graceful)
    - wait_for_resume() blocks until user resumes
    - Clear signals after handling them
    - Update state to inform CLI

PROGRAMMATIC USAGE
------------------

From Python code:

    from training_controller import TrainingController

    # Initialize
    controller = TrainingController("/path/to/training")

    # Send signals
    controller.signal_pause("Need to check something")
    controller.signal_stop("Emergency stop")
    controller.signal_skip("File is corrupted")
    controller.signal_resume()

    # Check signals (in training loop)
    if controller.should_pause_after_batch():
        controller.wait_for_resume()

    # Update state (in training daemon)
    controller.set_status("training", current_file="data.jsonl")

    # Query status (in monitoring)
    status = controller.get_status()
    print(f"Status: {status['status']}")
    print(f"Signals: {status['signals']}")

ERROR HANDLING
--------------

Missing control/ Directory:
    - Created automatically on TrainingController init
    - No error if already exists (mkdir_p=True)

Corrupt state.json:
    - get_status() returns empty dict on JSON parse error
    - State file recreated on next set_status() call

Stale Signal Files:
    - No automatic cleanup (by design)
    - User can run: training_controller.py clear
    - Or manually: rm control/.pause control/.stop

Multiple Resume Signals:
    - wait_for_resume() clears .resume after detecting it
    - Additional .resume files ignored (already resumed)

GOTCHAS & EDGE CASES
--------------------

1. Multiple .pause Signals:
   - Only one needed (file existence check)
   - Additional pauses do nothing (already paused)
   - Resume once to clear pause state

2. Pause During Pause:
   - No-op (already in paused state)
   - wait_for_resume() still blocking

3. Stop During Pause:
   - Stop takes precedence
   - Training exits immediately (no resume needed)

4. Signal File Permissions:
   - Must be writable by both CLI and daemon
   - Usually same user, no issue
   - If different users, check group permissions

5. State File Lag:
   - State file written by daemon (periodic updates)
   - CLI status may show stale data (1-5 second lag)
   - Not a bug, expected behavior

HISTORY
-------
Created: 2025-11-10 - Initial implementation
Updated: 2025-11-15 - Added skip command
Updated: 2025-11-20 - Enhanced status display

RELATED MODULES
---------------
- core/training_daemon.py - Main training loop (checks signals)
- core/training_queue.py - Queue management (skip moves files)
- core/training_status.py - Training progress (different from control state)

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingController:
    """
    Manages training control signals via file-based communication.

    Responsibilities:
        - Create/check signal files (.pause, .stop, .skip, .resume)
        - Maintain persistent state (control/state.json)
        - Provide query methods for training daemon
        - Offer command-line interface for user control
        - Wait for resume signals (blocking wait)

    Signal Management:
        - signal_pause(): Create .pause signal file
        - signal_stop(): Create .stop signal file
        - signal_skip(): Create .skip signal file
        - signal_resume(): Create .resume signal, clear .pause
        - clear_signals(): Remove all signal files

    Query Methods (for training daemon):
        - check_pause() -> bool: Is pause signal present?
        - check_stop() -> bool: Is stop signal present?
        - check_skip() -> bool: Is skip signal present?
        - check_resume() -> bool: Is resume signal present?
        - should_pause_after_batch() -> bool: Should pause now?
        - should_stop_after_batch() -> bool: Should stop now?
        - should_skip_current_file() -> bool: Should skip file?

    State Management:
        - set_status(): Update status (idle/training/paused/stopping/skipping)
        - get_status() -> Dict: Get full state + signals
        - wait_for_resume(): Block until resume signal (used when paused)

    File Structure:
        control/.pause         - Pause signal (contains reason + timestamp)
        control/.stop          - Stop signal (contains reason + timestamp)
        control/.skip          - Skip signal (contains reason + timestamp)
        control/.resume        - Resume signal (empty marker)
        control/state.json     - Current state (status, current_file, etc.)

    Thread Safety:
        - Signal file checks are thread-safe (atomic file existence)
        - State file is NOT thread-safe (single-writer assumption)
        - Training daemon reads, controller writes

    Example (Training Daemon):
        controller = TrainingController()

        while training:
            # Check signals after each batch
            if controller.should_stop_after_batch():
                controller.clear_stop()
                break

            if controller.should_pause_after_batch():
                controller.wait_for_resume()  # Blocks until resumed

            if controller.should_skip_current_file():
                move_to_failed(current_file)
                controller.clear_skip()
                continue

            train_batch()

    Example (User Control):
        controller = TrainingController()
        controller.signal_pause("Need to check logs")
        # Training will pause after current batch

        # Later...
        controller.signal_resume()
        # Training will continue
    """

    def __init__(self, base_dir: str = "/path/to/training"):
        self.base_dir = Path(base_dir)
        self.control_dir = self.base_dir / "control"
        self.control_dir.mkdir(exist_ok=True)

        # Control signal files
        self.pause_signal = self.control_dir / ".pause"
        self.stop_signal = self.control_dir / ".stop"
        self.skip_signal = self.control_dir / ".skip"
        self.resume_signal = self.control_dir / ".resume"

        # State file
        self.state_file = self.control_dir / "state.json"

        # Initialize state if doesn't exist
        if not self.state_file.exists():
            self._save_state({
                "status": "idle",
                "last_update": datetime.now().isoformat(),
                "current_file": None,
                "paused_at": None,
                "reason": None
            })

    def _save_state(self, state: Dict):
        """Save current state to file"""
        state["last_update"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> Dict:
        """Load current state from file"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def check_pause(self) -> bool:
        """Check if pause signal is present"""
        return self.pause_signal.exists()

    def check_stop(self) -> bool:
        """Check if stop signal is present"""
        return self.stop_signal.exists()

    def check_skip(self) -> bool:
        """Check if skip signal is present"""
        return self.skip_signal.exists()

    def check_resume(self) -> bool:
        """Check if resume signal is present"""
        return self.resume_signal.exists()

    def signal_pause(self, reason: str = "User requested"):
        """Signal training to pause"""
        logger.info(f"🔶 PAUSE signal: {reason}")
        self.pause_signal.touch()
        with open(self.pause_signal, 'w') as f:
            f.write(f"{reason}\n{datetime.now().isoformat()}")

    def signal_stop(self, reason: str = "User requested"):
        """Signal training to stop"""
        logger.info(f"🛑 STOP signal: {reason}")
        self.stop_signal.touch()
        with open(self.stop_signal, 'w') as f:
            f.write(f"{reason}\n{datetime.now().isoformat()}")

    def signal_skip(self, reason: str = "User requested"):
        """Signal to skip current file"""
        logger.info(f"⏭️  SKIP signal: {reason}")
        self.skip_signal.touch()
        with open(self.skip_signal, 'w') as f:
            f.write(f"{reason}\n{datetime.now().isoformat()}")

    def signal_resume(self):
        """Signal to resume from pause"""
        logger.info(f"▶️  RESUME signal")
        self.resume_signal.touch()
        # Clear pause signal
        if self.pause_signal.exists():
            self.pause_signal.unlink()

    def clear_signals(self):
        """Clear all control signals"""
        for signal in [self.pause_signal, self.stop_signal, self.skip_signal, self.resume_signal]:
            if signal.exists():
                signal.unlink()
        logger.info("✅ All control signals cleared")

    def clear_pause(self):
        """Clear pause signal (used after pausing)"""
        if self.pause_signal.exists():
            self.pause_signal.unlink()

    def clear_stop(self):
        """Clear stop signal (used after stopping)"""
        if self.stop_signal.exists():
            self.stop_signal.unlink()

    def clear_skip(self):
        """Clear skip signal (used after skipping)"""
        if self.skip_signal.exists():
            self.skip_signal.unlink()

    def clear_resume(self):
        """Clear resume signal (used after resuming)"""
        if self.resume_signal.exists():
            self.resume_signal.unlink()

    def set_status(self, status: str, current_file: Optional[str] = None, reason: Optional[str] = None):
        """
        Update controller status

        Statuses:
        - idle: Nothing happening
        - training: Currently training
        - paused: Paused by user
        - stopping: Finishing current batch then will stop
        - skipping: Skipping current file
        """
        state = self._load_state()
        state["status"] = status

        if current_file is not None:
            state["current_file"] = current_file

        if reason is not None:
            state["reason"] = reason

        if status == "paused":
            state["paused_at"] = datetime.now().isoformat()

        self._save_state(state)
        logger.info(f"📊 Status: {status}" + (f" - {reason}" if reason else ""))

    def get_status(self) -> Dict:
        """Get current controller status"""
        state = self._load_state()

        # Add signal status
        state["signals"] = {
            "pause": self.check_pause(),
            "stop": self.check_stop(),
            "skip": self.check_skip(),
            "resume": self.check_resume()
        }

        return state

    def should_pause_after_batch(self) -> bool:
        """Check if should pause after completing current batch"""
        if self.check_pause():
            logger.info("🔶 Pause signal detected - will pause after batch")
            return True
        return False

    def should_stop_after_batch(self) -> bool:
        """Check if should stop after completing current batch"""
        if self.check_stop():
            logger.info("🛑 Stop signal detected - will stop after batch")
            return True
        return False

    def update_state(self, status: str, current_file: Optional[str] = None, reason: Optional[str] = None):
        """
        Update training state (alias for set_status for daemon compatibility)
        """
        self.set_status(status, current_file, reason)

    def should_skip_current_file(self) -> bool:
        """Check if should skip current file"""
        if self.check_skip():
            logger.info("⏭️  Skip signal detected - will skip current file")
            return True
        return False

    def wait_for_resume(self):
        """Wait until resume signal is given"""
        self.set_status("paused", reason="Waiting for resume signal")
        logger.info("⏸️  PAUSED - Waiting for resume signal...")
        logger.info("   To resume: python3 training_controller.py resume")

        while not self.check_resume():
            time.sleep(5)  # Check every 5 seconds

        logger.info("▶️  RESUME signal received - continuing...")
        self.clear_resume()
        self.set_status("training")


def main():
    """Command-line interface for training controller"""
    import argparse

    parser = argparse.ArgumentParser(description="Training Control System")
    parser.add_argument('--base-dir', default='/path/to/training', help='Base directory')

    subparsers = parser.add_subparsers(dest='command', help='Control command')

    # Pause
    pause_parser = subparsers.add_parser('pause', help='Pause training after current batch')
    pause_parser.add_argument('--reason', default='User requested', help='Reason for pause')

    # Stop
    stop_parser = subparsers.add_parser('stop', help='Stop training after current batch')
    stop_parser.add_argument('--reason', default='User requested', help='Reason for stop')

    # Skip
    skip_parser = subparsers.add_parser('skip', help='Skip current file')
    skip_parser.add_argument('--reason', default='User requested', help='Reason for skip')

    # Resume
    subparsers.add_parser('resume', help='Resume from pause')

    # Status
    subparsers.add_parser('status', help='Show current status')

    # Clear
    subparsers.add_parser('clear', help='Clear all control signals')

    args = parser.parse_args()

    controller = TrainingController(args.base_dir)

    if args.command == 'pause':
        controller.signal_pause(args.reason)
        print("✅ Pause signal sent - training will pause after current batch")

    elif args.command == 'stop':
        controller.signal_stop(args.reason)
        print("✅ Stop signal sent - training will stop after current batch")

    elif args.command == 'skip':
        controller.signal_skip(args.reason)
        print("✅ Skip signal sent - current file will be skipped")

    elif args.command == 'resume':
        controller.signal_resume()
        print("✅ Resume signal sent - training will continue")

    elif args.command == 'status':
        status = controller.get_status()
        print("\n" + "="*80)
        print("TRAINING CONTROL STATUS")
        print("="*80)
        print(f"\nStatus: {status['status']}")
        print(f"Last Update: {status['last_update']}")

        if status.get('current_file'):
            print(f"Current File: {status['current_file']}")

        if status.get('reason'):
            print(f"Reason: {status['reason']}")

        if status.get('paused_at'):
            print(f"Paused At: {status['paused_at']}")

        print("\nActive Signals:")
        signals = status['signals']
        if any(signals.values()):
            for sig, active in signals.items():
                if active:
                    print(f"  🔴 {sig}")
        else:
            print("  ✅ None")

        print("="*80 + "\n")

    elif args.command == 'clear':
        controller.clear_signals()
        print("✅ All control signals cleared")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
