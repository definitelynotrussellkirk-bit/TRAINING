#!/usr/bin/env python3
"""
Training Control System

Provides graceful control over training operations:
- Pause training (finish current batch, then pause)
- Stop training (finish current batch, then stop)
- Skip current file (move to next in queue)
- Resume training
- Check status

Control via simple signal files:
- .pause    - Pause after current batch
- .stop     - Stop after current batch
- .skip     - Skip current file, move to next
- .resume   - Resume from pause

NO PROCESS KILLING - Everything is graceful!
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
    """Manages training control signals and state"""

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
