#!/usr/bin/env python3
"""
State Repair Tool - Fixes corrupted training daemon state

Handles:
- Stale control/state.json (paused from days ago)
- Missing status/training_daemon.json
- Queue inconsistencies
- Stuck Data Flow thread
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paths import get_base_dir


def reset_control_state():
    """Reset control/state.json to clean idle state"""
    base_dir = get_base_dir()
    control_state_path = base_dir / "control" / "state.json"

    print(f"Resetting {control_state_path}")

    clean_state = {
        "status": "idle",
        "last_update": datetime.utcnow().isoformat(),
        "current_file": None,
        "paused_at": None,
        "reason": None
    }

    # Backup old state
    if control_state_path.exists():
        backup_path = control_state_path.with_suffix('.json.bak')
        print(f"Backing up old state to {backup_path}")
        with open(control_state_path) as f:
            old_state = json.load(f)
        with open(backup_path, 'w') as f:
            json.dump(old_state, f, indent=2)

    # Write clean state
    with open(control_state_path, 'w') as f:
        json.dump(clean_state, f, indent=2)

    print(f"✅ Control state reset to clean idle")
    return clean_state


def force_daemon_status_write():
    """Send signal to training daemon to write its status file"""
    base_dir = get_base_dir()
    status_path = base_dir / "status" / "training_daemon.json"

    print(f"Checking {status_path}")

    if not status_path.exists():
        print(f"⚠️  Status file missing - daemon should create it on next cycle")
        # Create minimal status to bootstrap
        initial_status = {
            "status": "idle",
            "pid": None,
            "last_update": datetime.utcnow().isoformat(),
            "current_file": None,
            "note": "Reset by state repair tool"
        }
        with open(status_path, 'w') as f:
            json.dump(initial_status, f, indent=2)
        print(f"✅ Created initial status file")
    else:
        print(f"✅ Status file exists")

    return True


def clean_queue_inconsistencies():
    """Clean up .tmp files and check for stuck files"""
    base_dir = get_base_dir()
    queue_dir = base_dir / "queue" / "normal"

    print(f"Checking {queue_dir} for .tmp files")

    tmp_files = list(queue_dir.glob("*.tmp"))
    if tmp_files:
        print(f"Found {len(tmp_files)} .tmp files:")
        for tmp_file in tmp_files:
            print(f"  - {tmp_file.name}")
            # Don't delete - they might be actively being written
            # Just report them
            print(f"    ⚠️  Incomplete file (not deleting - may be in use)")
    else:
        print("✅ No .tmp files found")

    # Check processing directory
    processing_dir = base_dir / "queue" / "processing"
    processing_files = list(processing_dir.glob("*.jsonl"))
    if processing_files:
        print(f"\n⚠️  Found {len(processing_files)} stuck files in processing/:")
        for f in processing_files:
            print(f"  - {f.name}")
            # These are orphaned - move back to normal queue
            dest = queue_dir / f.name
            print(f"    Moving back to normal queue")
            f.rename(dest)
        print(f"✅ Moved {len(processing_files)} files back to queue")
    else:
        print("✅ Processing directory clean")

    return True


def check_daemon_running():
    """Check if training daemon is actually running"""
    import subprocess

    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )

    daemon_lines = [
        line for line in result.stdout.split('\n')
        if 'training_daemon.py' in line and 'grep' not in line
    ]

    if daemon_lines:
        print(f"✅ Training daemon is running:")
        for line in daemon_lines:
            print(f"  {line}")
        return True
    else:
        print(f"❌ Training daemon is NOT running")
        return False


def check_weaver_pid():
    """Check weaver PID file"""
    base_dir = get_base_dir()
    weaver_pid_path = base_dir / ".pids" / "weaver.pid"

    if weaver_pid_path.exists():
        with open(weaver_pid_path) as f:
            pid = f.read().strip()
        print(f"Weaver PID file exists: {pid}")

        # Check if process is running
        try:
            import psutil
            proc = psutil.Process(int(pid))
            print(f"  ✅ Weaver is running: {proc.cmdline()}")
            return True
        except:
            print(f"  ❌ Weaver PID file exists but process is dead")
            # Clean up stale PID file
            weaver_pid_path.unlink()
            print(f"  Removed stale PID file")
            return False
    else:
        print(f"❌ Weaver PID file does not exist")
        return False


def main():
    """Main state repair routine"""
    print("=" * 60)
    print("STATE REPAIR TOOL")
    print("=" * 60)

    print("\n1. Checking daemon status...")
    daemon_running = check_daemon_running()

    print("\n2. Checking Weaver...")
    weaver_running = check_weaver_pid()

    print("\n3. Resetting control state...")
    reset_control_state()

    print("\n4. Ensuring daemon status file exists...")
    force_daemon_status_write()

    print("\n5. Cleaning queue inconsistencies...")
    clean_queue_inconsistencies()

    print("\n" + "=" * 60)
    print("STATE REPAIR COMPLETE")
    print("=" * 60)

    print("\nNext steps:")
    if not daemon_running:
        print("  ⚠️  Training daemon not running - start it")
    else:
        print("  ✅ Training daemon should pick up clean state on next cycle")

    if not weaver_running:
        print("  ⚠️  Weaver not running - start it if needed")
    else:
        print("  ✅ Weaver should stabilize with clean state")

    print("\nRecommendation:")
    print("  1. Restart training daemon: python3 core/training_controller.py stop && python3 core/training_daemon.py")
    print("  2. Restart Weaver: python3 weaver/weaver.py --daemon")
    print("  3. Monitor: python3 -m training doctor")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
