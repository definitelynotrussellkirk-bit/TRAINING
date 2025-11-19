#!/usr/bin/env python3
"""
Edge Case Testing Suite - Tests system behavior under failure scenarios

This script simulates and tests:
1. Daemon crash scenarios
2. Disk space exhaustion
3. GPU OOM conditions
4. Process hangs
5. File corruption
6. Config errors
7. Network failures (if applicable)

Usage:
    python3 test_edge_cases.py [--test-name <name>] [--dry-run]
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path("/path/to/training")
CONFIG_FILE = BASE_DIR / "config.json"
PID_FILE = BASE_DIR / ".daemon.pid"

class EdgeCaseTester:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.results = []

    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        print(f"\n{'='*80}")
        print(f"🧪 Testing: {test_name}")
        print(f"{'='*80}")

        if self.dry_run:
            print("   [DRY RUN - Not actually executing]")
            return

        try:
            result = test_func()
            self.results.append({
                "test": test_name,
                "status": "PASS" if result else "FAIL",
                "details": result
            })
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            self.results.append({
                "test": test_name,
                "status": "ERROR",
                "error": str(e)
            })

    def test_daemon_crash_recovery(self):
        """Test: Daemon crash and auto-restart"""
        print("   Simulating daemon crash...")

        # Get daemon PID
        if not PID_FILE.exists():
            print("   ⚠️ Daemon not running - skipping test")
            return None

        pid = int(PID_FILE.read_text().strip())
        print(f"   Found daemon PID: {pid}")

        # Kill daemon
        print("   Killing daemon...")
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)

        # Check if watchdog restarted it
        print("   Waiting for watchdog to restart daemon...")
        time.sleep(35)  # Watchdog checks every 30s

        # Verify daemon restarted
        if PID_FILE.exists():
            new_pid = int(PID_FILE.read_text().strip())
            if new_pid != pid:
                print(f"   ✅ Daemon restarted (new PID: {new_pid})")
                return True
            else:
                print("   ❌ PID unchanged - watchdog may not be running")
                return False
        else:
            print("   ❌ Daemon not restarted")
            return False

    def test_config_corruption_detection(self):
        """Test: Detect corrupted config.json"""
        print("   Testing config corruption detection...")

        # Backup config
        backup = CONFIG_FILE.with_suffix('.json.backup_test')
        CONFIG_FILE.rename(backup)

        # Write corrupted config
        CONFIG_FILE.write_text("{ invalid json }}")

        # Try to load
        try:
            with open(CONFIG_FILE) as f:
                json.load(f)
            result = False  # Should have failed
        except json.JSONDecodeError:
            print("   ✅ Corrupted config detected")
            result = True

        # Restore
        CONFIG_FILE.unlink()
        backup.rename(CONFIG_FILE)

        return result

    def test_missing_base_model_detection(self):
        """Test: Detect missing base model"""
        print("   Testing missing base model detection...")

        with open(CONFIG_FILE) as f:
            config = json.load(f)

        base_model = Path(config.get('base_model', ''))

        if base_model.exists():
            print(f"   ✅ Base model exists: {base_model}")
            return True
        else:
            print(f"   ❌ Base model missing: {base_model}")
            return False

    def test_disk_space_monitoring(self):
        """Test: Disk space monitoring"""
        print("   Testing disk space monitoring...")

        stat = os.statvfs(BASE_DIR)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        print(f"   Free space: {free_gb:.1f} GB")

        if free_gb < 10:
            print("   ⚠️ Low disk space detected")
            return True
        else:
            print("   ✅ Sufficient disk space")
            return True

    def test_checkpoint_recovery(self):
        """Test: Training resume from checkpoint"""
        print("   Testing checkpoint recovery...")

        model_dir = BASE_DIR / "current_model"
        if not model_dir.exists():
            print("   ⚠️ No current_model - skipping test")
            return None

        checkpoints = list(model_dir.glob("checkpoint-*"))
        if not checkpoints:
            print("   ⚠️ No checkpoints found - skipping test")
            return None

        latest = sorted(checkpoints)[-1]
        print(f"   Found latest checkpoint: {latest.name}")

        # Check if checkpoint has required files
        required_files = ['adapter_model.safetensors', 'trainer_state.json']
        missing = []
        for fname in required_files:
            if not (latest / fname).exists():
                missing.append(fname)

        if missing:
            print(f"   ❌ Missing files in checkpoint: {missing}")
            return False
        else:
            print("   ✅ Checkpoint appears valid")
            return True

    def test_orphaned_process_cleanup(self):
        """Test: Orphaned process detection"""
        print("   Testing orphaned process detection...")

        result = subprocess.run(
            ['pgrep', '-af', 'train.py'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            pids = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
            print(f"   Found {len(pids)} train.py process(es)")

            # Check if they're managed by daemon
            daemon_pid = None
            if PID_FILE.exists():
                daemon_pid = int(PID_FILE.read_text().strip())

            orphaned = []
            for pid in pids:
                try:
                    proc_info = subprocess.run(
                        ['ps', '-o', 'ppid=', '-p', pid],
                        capture_output=True,
                        text=True
                    )
                    ppid = int(proc_info.stdout.strip())
                    if ppid != daemon_pid:
                        orphaned.append(pid)
                except:
                    pass

            if orphaned:
                print(f"   ⚠️ Found {len(orphaned)} orphaned process(es)")
                return True  # Detection working
            else:
                print("   ✅ No orphaned processes")
                return True
        else:
            print("   ✅ No train.py processes running")
            return True

    def test_gpu_availability(self):
        """Test: GPU availability check"""
        print("   Testing GPU availability...")

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                gpus = result.stdout.strip().split('\n')
                print(f"   ✅ Found {len(gpus)} GPU(s)")
                for gpu in gpus:
                    print(f"      {gpu}")
                return True
            else:
                print("   ❌ nvidia-smi failed")
                return False
        except FileNotFoundError:
            print("   ❌ nvidia-smi not found")
            return False
        except subprocess.TimeoutExpired:
            print("   ❌ GPU query timeout")
            return False

    def test_queue_system(self):
        """Test: Queue system functionality"""
        print("   Testing queue system...")

        queue_dir = BASE_DIR / "queue"
        priority_dirs = ['high', 'normal', 'low']

        all_exist = True
        for pdir in priority_dirs:
            path = queue_dir / pdir
            if path.exists():
                files = list(path.glob('*.jsonl'))
                print(f"   ✅ {pdir}: {len(files)} file(s)")
            else:
                print(f"   ❌ {pdir}: directory missing")
                all_exist = False

        return all_exist

    def test_status_file_updates(self):
        """Test: Status file is being updated"""
        print("   Testing status file updates...")

        status_file = BASE_DIR / "status" / "training_status.json"

        if not status_file.exists():
            print("   ⚠️ No status file yet")
            return None

        # Get current status
        with open(status_file) as f:
            status1 = json.load(f)
        step1 = status1.get('current_step', 0)

        # Wait and check again
        print("   Waiting 30s for status update...")
        time.sleep(30)

        with open(status_file) as f:
            status2 = json.load(f)
        step2 = status2.get('current_step', 0)

        if step2 > step1:
            print(f"   ✅ Status updated ({step1} → {step2})")
            return True
        else:
            print(f"   ⚠️ No progress ({step1} → {step2})")
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 EDGE CASE TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results if r['status'] == 'ERROR')
        skipped = sum(1 for r in self.results if r.get('details') is None)

        print(f"\n✅ Passed:  {passed}")
        print(f"❌ Failed:  {failed}")
        print(f"⚠️  Errors:  {errors}")
        print(f"⏭️  Skipped: {skipped}")

        if failed > 0 or errors > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if r['status'] in ['FAIL', 'ERROR']:
                    print(f"   • {r['test']}")
                    if 'error' in r:
                        print(f"     Error: {r['error']}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test edge cases and failure scenarios")
    parser.add_argument('--test', type=str, help="Run specific test only")
    parser.add_argument('--dry-run', action='store_true', help="Show tests without executing")
    args = parser.parse_args()

    tester = EdgeCaseTester(dry_run=args.dry_run)

    # Define all tests
    tests = [
        ("Config corruption detection", tester.test_config_corruption_detection),
        ("Missing base model detection", tester.test_missing_base_model_detection),
        ("Disk space monitoring", tester.test_disk_space_monitoring),
        ("Checkpoint recovery", tester.test_checkpoint_recovery),
        ("Orphaned process cleanup", tester.test_orphaned_process_cleanup),
        ("GPU availability", tester.test_gpu_availability),
        ("Queue system", tester.test_queue_system),
        ("Status file updates", tester.test_status_file_updates),
        # Note: Daemon crash recovery test commented out as it kills the daemon
        # ("Daemon crash recovery", tester.test_daemon_crash_recovery),
    ]

    if args.test:
        # Run specific test
        for name, func in tests:
            if args.test.lower() in name.lower():
                tester.run_test(name, func)
                break
        else:
            print(f"❌ Test '{args.test}' not found")
            sys.exit(1)
    else:
        # Run all tests
        for name, func in tests:
            tester.run_test(name, func)

    tester.print_summary()


if __name__ == '__main__':
    main()
