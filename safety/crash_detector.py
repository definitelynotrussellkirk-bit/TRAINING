#!/usr/bin/env python3
"""
Crash Detector - Analyzes logs to detect and categorize crashes

This tool:
1. Scans training logs for crash patterns
2. Identifies crash type (OOM, CUDA, timeout, etc.)
3. Suggests recovery actions
4. Tracks crash history

Usage:
    python3 crash_detector.py [--last-n-lines 1000]
"""

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Use centralized path resolution instead of hard-coded paths
try:
    from core.paths import get_base_dir
    BASE_DIR = get_base_dir()
except ImportError:
    BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of safety/

LOGS_DIR = BASE_DIR / "logs"
CRASH_HISTORY_FILE = BASE_DIR / ".crash_history.json"

# Crash patterns to detect
CRASH_PATTERNS = {
    "cuda_oom": [
        r"CUDA out of memory",
        r"RuntimeError.*out of memory",
        r"torch.cuda.OutOfMemoryError"
    ],
    "cuda_error": [
        r"CUDA error",
        r"cudnn.*error",
        r"cublas.*error",
        r"RuntimeError.*CUDA"
    ],
    "process_killed": [
        r"Killed",
        r"signal.*SIGKILL",
        r"Process.*killed"
    ],
    "timeout": [
        r"TimeoutError",
        r"timed out",
        r"timeout exceeded"
    ],
    "import_error": [
        r"ImportError",
        r"ModuleNotFoundError",
        r"No module named"
    ],
    "file_not_found": [
        r"FileNotFoundError",
        r"No such file or directory",
        r"\[Errno 2\]"
    ],
    "permission_error": [
        r"PermissionError",
        r"Permission denied",
        r"\[Errno 13\]"
    ],
    "connection_error": [
        r"ConnectionError",
        r"Connection refused",
        r"Failed to connect"
    ],
    "assertion_error": [
        r"AssertionError",
        r"assert.*failed"
    ],
    "key_error": [
        r"KeyError",
        r"key not found"
    ],
    "value_error": [
        r"ValueError",
        r"invalid literal"
    ],
    "type_error": [
        r"TypeError",
        r"unsupported operand type"
    ],
    "multiprocessing_error": [
        r"Cannot re-initialize CUDA",
        r"forked subprocess.*CUDA",
        r"multiprocessing.*CUDA"
    ],
    "segfault": [
        r"Segmentation fault",
        r"SIGSEGV",
        r"segfault"
    ],
    "disk_full": [
        r"No space left on device",
        r"\[Errno 28\]"
    ]
}

# Recovery suggestions for each crash type
RECOVERY_SUGGESTIONS = {
    "cuda_oom": [
        "Reduce batch size in config.json",
        "Reduce max_length to decrease memory usage",
        "Check for memory leaks in training loop",
        "Use gradient checkpointing if not already enabled"
    ],
    "cuda_error": [
        "Reset GPU: sudo nvidia-smi -r",
        "Check GPU health: nvidia-smi",
        "Update CUDA drivers",
        "Check for hardware issues"
    ],
    "process_killed": [
        "Check system memory: free -h",
        "Check OOM killer logs: dmesg | grep -i oom",
        "Reduce memory usage or add swap space"
    ],
    "timeout": [
        "Increase timeout values in config",
        "Check network connectivity",
        "Check if remote services are responding"
    ],
    "import_error": [
        "Install missing package: pip install <package>",
        "Check Python environment is activated",
        "Verify dependencies in requirements.txt"
    ],
    "file_not_found": [
        "Check file paths in config.json",
        "Verify base_model path exists",
        "Check data files exist in inbox/"
    ],
    "permission_error": [
        "Check file permissions: ls -l <file>",
        "Fix permissions: chmod +rw <file>",
        "Check directory permissions"
    ],
    "multiprocessing_error": [
        "Set num_proc=None in dataset tokenization",
        "Disable multiprocessing in data loading",
        "Already fixed in train.py:414"
    ],
    "disk_full": [
        "Free up disk space: df -h",
        "Clean old checkpoints",
        "Run: python3 cleanup_old_checkpoints.py"
    ]
}


class CrashDetector:
    def __init__(self):
        self.crashes = []
        self.crash_counts = defaultdict(int)

    def detect_crashes(self, log_content, source_file=None):
        """Detect crashes in log content"""
        lines = log_content.split('\n')

        for i, line in enumerate(lines):
            for crash_type, patterns in CRASH_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Get context (5 lines before and after)
                        context_start = max(0, i - 5)
                        context_end = min(len(lines), i + 6)
                        context = '\n'.join(lines[context_start:context_end])

                        self.crashes.append({
                            "type": crash_type,
                            "line": line.strip(),
                            "line_number": i + 1,
                            "context": context,
                            "source_file": source_file,
                            "timestamp": datetime.now().isoformat()
                        })
                        self.crash_counts[crash_type] += 1
                        break

    def analyze_logs(self, last_n_lines=None):
        """Analyze all logs for crashes"""
        # Check training output log
        training_log = BASE_DIR / "training_output.log"
        if training_log.exists():
            content = training_log.read_text(errors='ignore')
            if last_n_lines:
                lines = content.split('\n')
                content = '\n'.join(lines[-last_n_lines:])
            self.detect_crashes(content, "training_output.log")

        # Check daemon logs
        today = datetime.now().strftime('%Y%m%d')
        daemon_log = LOGS_DIR / f"daemon_{today}.log"
        if daemon_log.exists():
            content = daemon_log.read_text(errors='ignore')
            if last_n_lines:
                lines = content.split('\n')
                content = '\n'.join(lines[-last_n_lines:])
            self.detect_crashes(content, f"daemon_{today}.log")

        # Check train logs
        for log_file in sorted(LOGS_DIR.glob("train_*.log"), reverse=True)[:3]:
            content = log_file.read_text(errors='ignore')
            if last_n_lines:
                lines = content.split('\n')
                content = '\n'.join(lines[-last_n_lines:])
            self.detect_crashes(content, log_file.name)

    def save_crash_history(self):
        """Save crash history to file"""
        history = {
            "last_update": datetime.now().isoformat(),
            "crash_counts": dict(self.crash_counts),
            "recent_crashes": self.crashes[-10:]  # Last 10 crashes
        }

        with open(CRASH_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    def print_report(self):
        """Print crash analysis report"""
        print("\n" + "=" * 80)
        print("🔍 CRASH DETECTION REPORT")
        print("=" * 80)

        if not self.crashes:
            print("\n✅ No crashes detected!")
            return

        print(f"\n⚠️  Found {len(self.crashes)} crash indicators")
        print("\n📊 CRASH SUMMARY:")
        print("-" * 80)
        for crash_type, count in sorted(self.crash_counts.items(), key=lambda x: -x[1]):
            print(f"   {crash_type:25s}: {count:3d} occurrences")

        print("\n🔍 RECENT CRASHES:")
        print("-" * 80)
        for crash in self.crashes[-5:]:  # Show last 5
            print(f"\n   Type: {crash['type']}")
            print(f"   Source: {crash['source_file']}:{crash['line_number']}")
            print(f"   Error: {crash['line']}")

            # Show recovery suggestions
            if crash['type'] in RECOVERY_SUGGESTIONS:
                print(f"\n   💡 RECOVERY SUGGESTIONS:")
                for suggestion in RECOVERY_SUGGESTIONS[crash['type']]:
                    print(f"      • {suggestion}")

        print("\n" + "=" * 80)

        # Actionable summary
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("-" * 80)
        top_crashes = sorted(self.crash_counts.items(), key=lambda x: -x[1])[:3]
        for crash_type, count in top_crashes:
            print(f"\n   {crash_type} ({count} times):")
            if crash_type in RECOVERY_SUGGESTIONS:
                for suggestion in RECOVERY_SUGGESTIONS[crash_type]:
                    print(f"   • {suggestion}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Detect and analyze training crashes")
    parser.add_argument('--last-n-lines', type=int, default=None,
                        help="Only analyze last N lines of each log")
    parser.add_argument('--quiet', action='store_true',
                        help="Only show summary, not details")
    args = parser.parse_args()

    detector = CrashDetector()
    detector.analyze_logs(args.last_n_lines)

    if not args.quiet:
        detector.print_report()
    else:
        if detector.crashes:
            print(f"⚠️  {len(detector.crashes)} crashes detected")
            for crash_type, count in detector.crash_counts.items():
                print(f"   {crash_type}: {count}")
        else:
            print("✅ No crashes detected")

    detector.save_crash_history()


if __name__ == '__main__':
    main()
