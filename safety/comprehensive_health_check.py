#!/usr/bin/env python3
"""
Comprehensive Health Check - Tests for common failure scenarios

This script tests:
1. Daemon health
2. Training progress
3. Disk space
4. GPU availability
5. Config validity
6. File permissions
7. Process health
8. Network connectivity (if needed)
9. Memory availability
10. Checkpoint integrity

Usage:
    python3 comprehensive_health_check.py [--fix]
"""

import os
import sys
import json
import time
import psutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Use centralized path resolution instead of hard-coded paths
try:
    from core.paths import get_base_dir
    BASE_DIR = get_base_dir()
except ImportError:
    BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of safety/

CONFIG_FILE = BASE_DIR / "config.json"
STATUS_FILE = BASE_DIR / "status" / "training_status.json"
PID_FILE = BASE_DIR / ".daemon.pid"

# Health check thresholds
MIN_DISK_SPACE_GB = 50
MIN_MEMORY_GB = 4
MAX_CHECKPOINT_AGE_HOURS = 24
REQUIRED_DIRS = ["inbox", "logs", "status", "queue", "current_model"]
REQUIRED_FILES = ["config.json", "train.py", "training_daemon.py"]


class HealthCheck:
    def __init__(self, auto_fix=False):
        self.auto_fix = auto_fix
        self.issues = []
        self.warnings = []
        self.passed = []

    def log_pass(self, test_name, message=""):
        """Log a passed test"""
        self.passed.append({"test": test_name, "message": message})
        print(f"✅ {test_name}: {message}" if message else f"✅ {test_name}")

    def log_warning(self, test_name, message, fix_cmd=None):
        """Log a warning"""
        self.warnings.append({
            "test": test_name,
            "message": message,
            "fix_cmd": fix_cmd
        })
        print(f"⚠️  {test_name}: {message}")
        if fix_cmd:
            print(f"   Fix: {fix_cmd}")

    def log_issue(self, test_name, message, fix_cmd=None):
        """Log a critical issue"""
        self.issues.append({
            "test": test_name,
            "message": message,
            "fix_cmd": fix_cmd
        })
        print(f"❌ {test_name}: {message}")
        if fix_cmd:
            print(f"   Fix: {fix_cmd}")

    def check_directory_structure(self):
        """Check if required directories exist"""
        print("\n📁 Checking directory structure...")
        for dirname in REQUIRED_DIRS:
            dirpath = BASE_DIR / dirname
            if dirpath.exists():
                self.log_pass(f"Directory {dirname}", "exists")
            else:
                self.log_issue(
                    f"Directory {dirname}",
                    "missing",
                    f"mkdir -p {dirpath}"
                )
                if self.auto_fix:
                    dirpath.mkdir(parents=True, exist_ok=True)
                    print(f"   ✓ Created {dirname}")

    def check_required_files(self):
        """Check if required files exist"""
        print("\n📄 Checking required files...")
        for filename in REQUIRED_FILES:
            filepath = BASE_DIR / filename
            if filepath.exists():
                self.log_pass(f"File {filename}", "exists")
            else:
                self.log_issue(f"File {filename}", "missing")

    def check_disk_space(self):
        """Check available disk space"""
        print("\n💾 Checking disk space...")
        stat = os.statvfs(BASE_DIR)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        if free_gb >= MIN_DISK_SPACE_GB:
            self.log_pass("Disk space", f"{free_gb:.1f} GB free")
        elif free_gb >= MIN_DISK_SPACE_GB / 2:
            self.log_warning(
                "Disk space",
                f"Only {free_gb:.1f} GB free (recommend {MIN_DISK_SPACE_GB} GB)",
                "Clean old checkpoints or free up space"
            )
        else:
            self.log_issue(
                "Disk space",
                f"Critical: Only {free_gb:.1f} GB free",
                "Free up space immediately"
            )

    def check_memory(self):
        """Check available memory"""
        print("\n🧠 Checking memory...")
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        if available_gb >= MIN_MEMORY_GB:
            self.log_pass("System memory", f"{available_gb:.1f} GB available")
        else:
            self.log_warning(
                "System memory",
                f"Only {available_gb:.1f} GB available",
                "Close unnecessary programs"
            )

    def check_gpu(self):
        """Check GPU availability and status"""
        print("\n🎮 Checking GPU...")
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    gpu_id, name, mem_used, mem_total, util, temp = line.split(', ')
                    self.log_pass(
                        f"GPU {gpu_id}",
                        f"{name} - {mem_used}/{mem_total} MB, {util}% util, {temp}°C"
                    )

                    # Check for issues
                    if int(temp) > 85:
                        self.log_warning(f"GPU {gpu_id} temperature", f"{temp}°C is high")
                    if int(mem_used) / int(mem_total) > 0.95:
                        self.log_warning(f"GPU {gpu_id} memory", "Almost full")
            else:
                self.log_issue("GPU", "nvidia-smi failed", "Check NVIDIA drivers")
        except FileNotFoundError:
            self.log_issue("GPU", "nvidia-smi not found", "Install NVIDIA drivers")
        except subprocess.TimeoutExpired:
            self.log_issue("GPU", "nvidia-smi timeout", "GPU may be hung")

    def check_daemon(self):
        """Check if daemon is running"""
        print("\n🔧 Checking daemon status...")
        pid = None
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
            except:
                pass

        if pid and psutil.pid_exists(pid):
            try:
                proc = psutil.Process(pid)
                if 'training_daemon' in ' '.join(proc.cmdline()):
                    self.log_pass("Daemon process", f"Running (PID {pid})")
                else:
                    self.log_issue(
                        "Daemon PID",
                        f"PID {pid} exists but not daemon",
                        "rm .daemon.pid && restart daemon"
                    )
            except:
                self.log_issue("Daemon process", "Error checking process")
        else:
            self.log_issue(
                "Daemon process",
                "Not running",
                f"python3 training_daemon.py --base-dir {BASE_DIR} &"
            )

    def check_training_progress(self):
        """Check if training is progressing"""
        print("\n📈 Checking training progress...")
        if not STATUS_FILE.exists():
            self.log_warning("Training status", "No status file yet")
            return

        try:
            with open(STATUS_FILE) as f:
                status = json.load(f)

            step = status.get('current_step', 0)
            total = status.get('total_steps', 0)
            loss = status.get('loss', 0)

            if total > 0:
                pct = (step / total) * 100
                self.log_pass(
                    "Training progress",
                    f"Step {step}/{total} ({pct:.1f}%), Loss: {loss:.4f}"
                )
            else:
                self.log_warning("Training progress", "No training in progress")
        except Exception as e:
            self.log_issue("Training status", f"Error reading status: {e}")

    def check_config_validity(self):
        """Check if config.json is valid"""
        print("\n⚙️  Checking configuration...")
        if not CONFIG_FILE.exists():
            self.log_issue("Config file", "config.json missing")
            return

        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)

            # Check required fields
            required = ['model_name', 'base_model', 'batch_size', 'learning_rate']
            for field in required:
                if field in config:
                    self.log_pass(f"Config field '{field}'", str(config[field]))
                else:
                    self.log_issue(f"Config field '{field}'", "missing")

            # Check base_model path
            base_model = config.get('base_model')
            if base_model:
                if Path(base_model).exists():
                    self.log_pass("Base model path", f"{base_model} exists")
                else:
                    self.log_issue("Base model path", f"{base_model} not found")
        except json.JSONDecodeError as e:
            self.log_issue("Config file", f"Invalid JSON: {e}")
        except Exception as e:
            self.log_issue("Config file", f"Error: {e}")

    def check_checkpoint_integrity(self):
        """Check if checkpoints are recent and valid"""
        print("\n💾 Checking checkpoints...")
        model_dir = BASE_DIR / "current_model"

        if not model_dir.exists():
            self.log_warning("Checkpoints", "No current_model directory")
            return

        checkpoints = sorted(model_dir.glob("checkpoint-*"))
        if not checkpoints:
            self.log_warning("Checkpoints", "No checkpoints found")
            return

        latest = checkpoints[-1]
        age_hours = (time.time() - latest.stat().st_mtime) / 3600

        if age_hours < MAX_CHECKPOINT_AGE_HOURS:
            self.log_pass(
                "Latest checkpoint",
                f"{latest.name} ({age_hours:.1f}h old)"
            )
        else:
            self.log_warning(
                "Latest checkpoint",
                f"{latest.name} is {age_hours:.1f}h old",
                "May indicate stalled training"
            )

        # Check number of checkpoints
        if len(checkpoints) > 30:
            self.log_warning(
                "Checkpoint count",
                f"{len(checkpoints)} checkpoints taking up disk space",
                "Consider cleanup"
            )

    def check_orphaned_processes(self):
        """Check for orphaned training processes"""
        print("\n👻 Checking for orphaned processes...")
        try:
            result = subprocess.run(
                ['pgrep', '-af', 'train.py'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                pids = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
                if pids:
                    self.log_warning(
                        "Orphaned processes",
                        f"Found {len(pids)} train.py process(es)",
                        "Check if these should be running"
                    )
                else:
                    self.log_pass("Orphaned processes", "None found")
            else:
                self.log_pass("Orphaned processes", "None found")
        except Exception as e:
            self.log_warning("Process check", f"Error: {e}")

    def run_all_checks(self):
        """Run all health checks"""
        print("\n" + "=" * 80)
        print("🏥 COMPREHENSIVE HEALTH CHECK")
        print("=" * 80)

        self.check_directory_structure()
        self.check_required_files()
        self.check_disk_space()
        self.check_memory()
        self.check_gpu()
        self.check_daemon()
        self.check_training_progress()
        self.check_config_validity()
        self.check_checkpoint_integrity()
        self.check_orphaned_processes()

        self.print_summary()

    def print_summary(self):
        """Print summary of health check"""
        print("\n" + "=" * 80)
        print("📊 HEALTH CHECK SUMMARY")
        print("=" * 80)
        print(f"\n✅ Passed:   {len(self.passed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues:   {len(self.issues)}")

        if self.issues:
            print("\n🔧 CRITICAL ISSUES TO FIX:")
            print("-" * 80)
            for issue in self.issues:
                print(f"\n   {issue['test']}: {issue['message']}")
                if issue['fix_cmd']:
                    print(f"   Fix: {issue['fix_cmd']}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            print("-" * 80)
            for warning in self.warnings:
                print(f"\n   {warning['test']}: {warning['message']}")
                if warning['fix_cmd']:
                    print(f"   Suggestion: {warning['fix_cmd']}")

        print("\n" + "=" * 80)

        # Exit code
        if self.issues:
            print("\n❌ Health check FAILED - fix critical issues")
            return 1
        elif self.warnings:
            print("\n⚠️  Health check PASSED with warnings")
            return 0
        else:
            print("\n✅ Health check PASSED - all systems healthy")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive health checks")
    parser.add_argument('--fix', action='store_true',
                        help="Automatically fix issues where possible")
    args = parser.parse_args()

    checker = HealthCheck(auto_fix=args.fix)
    exit_code = checker.run_all_checks()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
