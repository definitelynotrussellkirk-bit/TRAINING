#!/usr/bin/env python3
"""
3090 Memory Guardian - VRAM leak detection and automatic cleanup
Prevents OOM failures by monitoring and clearing GPU memory
"""

import subprocess
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import logging

from core.paths import get_base_dir
from core.hosts import get_host

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryGuardian:
    """Monitors VRAM usage and prevents memory leaks"""

    def __init__(
        self,
        remote_host: str = None,
        vram_warning_threshold: float = 85.0,  # % VRAM usage
        vram_critical_threshold: float = 95.0,
        leak_detection_window: int = 10,  # samples
        base_dir: str = None
    ):
        # Get inference host info
        inference_host = get_host("3090")
        self.remote_host = remote_host if remote_host else inference_host.host

        self.vram_warning_threshold = vram_warning_threshold
        self.vram_critical_threshold = vram_critical_threshold
        self.leak_detection_window = leak_detection_window
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()

        # Memory usage history (last 60 samples = 1 hour)
        self.vram_history = deque(maxlen=60)
        self.leak_alerts = 0
        self.cleanup_count = 0

        # Logs
        self.logs_dir = self.base_dir / "logs" / "memory_guardian"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_vram_stats(self) -> Dict:
        """Get current VRAM usage"""
        try:
            cmd = [
                "ssh", self.remote_host,
                "nvidia-smi --query-gpu=memory.used,memory.total,memory.free "
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                used = float(parts[0])
                total = float(parts[1])
                free = float(parts[2])

                usage_pct = (used / total) * 100

                stats = {
                    "used_mb": used,
                    "total_mb": total,
                    "free_mb": free,
                    "usage_pct": usage_pct,
                    "timestamp": datetime.now().isoformat()
                }

                self.vram_history.append(stats)
                return stats
            else:
                return {"error": "nvidia-smi failed"}
        except Exception as e:
            logger.error(f"VRAM stats failed: {e}")
            return {"error": str(e)}

    def detect_memory_leak(self) -> Optional[Dict]:
        """Detect if VRAM usage is trending upward (potential leak)"""
        if len(self.vram_history) < self.leak_detection_window:
            return None

        # Get last N samples
        recent = list(self.vram_history)[-self.leak_detection_window:]

        # Check if usage is monotonically increasing
        usage_values = [s['usage_pct'] for s in recent]

        # Calculate trend (simple linear regression slope)
        n = len(usage_values)
        x = list(range(n))
        y = usage_values

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Leak if: slope > 1% per sample AND current usage > 70%
        is_leak = slope > 1.0 and usage_values[-1] > 70.0

        if is_leak:
            return {
                "detected": True,
                "slope": slope,
                "current_usage": usage_values[-1],
                "start_usage": usage_values[0],
                "increase": usage_values[-1] - usage_values[0],
                "window_size": n
            }
        else:
            return {"detected": False}

    def clear_gpu_cache(self) -> bool:
        """Clear GPU cache by calling API endpoint"""
        try:
            # Send cleanup request to API
            cmd = [
                "ssh", self.remote_host,
                "curl -X POST http://localhost:8765/clear_cache"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                logger.info("✅ GPU cache cleared successfully")
                self.cleanup_count += 1
                return True
            else:
                logger.error("❌ Failed to clear GPU cache")
                return False
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    def restart_api_server(self) -> bool:
        """Restart API server to fully clear memory"""
        try:
            logger.warning("🔄 Restarting API server to clear memory...")

            # Kill process
            kill_cmd = ["ssh", self.remote_host, "pkill -f 'python3 main.py'"]
            subprocess.run(kill_cmd, timeout=10)
            time.sleep(3)

            # Restart
            start_cmd = [
                "ssh", self.remote_host,
                "cd ~/llm && source venv/bin/activate && "
                "nohup python3 main.py > logs/api_server.log 2>&1 &"
            ]
            subprocess.run(start_cmd, timeout=15)
            time.sleep(5)

            logger.info("✅ API server restarted")
            return True
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            return False

    def handle_high_memory(self, stats: Dict, leak_info: Optional[Dict]):
        """Handle high memory usage scenarios"""
        usage = stats['usage_pct']

        # Critical threshold - restart immediately
        if usage >= self.vram_critical_threshold:
            logger.error(
                f"🚨 CRITICAL: VRAM usage at {usage:.1f}% (threshold: {self.vram_critical_threshold}%)"
            )
            logger.error("Restarting API server to prevent OOM...")
            self.restart_api_server()
            return

        # Warning threshold - try cache clear
        if usage >= self.vram_warning_threshold:
            logger.warning(
                f"⚠️  WARNING: VRAM usage at {usage:.1f}% (threshold: {self.vram_warning_threshold}%)"
            )

            # If leak detected, be more aggressive
            if leak_info and leak_info.get('detected'):
                logger.warning(
                    f"📈 Memory leak detected! Usage increased {leak_info['increase']:.1f}% "
                    f"over last {leak_info['window_size']} samples"
                )
                logger.warning("Attempting cache clear...")
                self.clear_gpu_cache()
                self.leak_alerts += 1
            else:
                logger.info("Attempting cache clear...")
                self.clear_gpu_cache()

    def get_process_memory_breakdown(self) -> List[Dict]:
        """Get memory usage breakdown by process"""
        try:
            cmd = [
                "ssh", self.remote_host,
                "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                processes = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    processes.append({
                        "pid": int(parts[0]),
                        "vram_mb": float(parts[1])
                    })
                return processes
            else:
                return []
        except Exception as e:
            logger.error(f"Process memory breakdown failed: {e}")
            return []

    def generate_memory_report(self) -> Dict:
        """Generate memory usage report"""
        stats = self.get_vram_stats()
        leak_info = self.detect_memory_leak()
        processes = self.get_process_memory_breakdown()

        # Calculate statistics
        if len(self.vram_history) > 0:
            recent_usage = [s['usage_pct'] for s in self.vram_history]
            avg_usage = sum(recent_usage) / len(recent_usage)
            max_usage = max(recent_usage)
            min_usage = min(recent_usage)
        else:
            avg_usage = max_usage = min_usage = 0

        return {
            "timestamp": datetime.now().isoformat(),
            "current": stats,
            "leak_detection": leak_info,
            "processes": processes,
            "statistics": {
                "avg_usage_pct": avg_usage,
                "max_usage_pct": max_usage,
                "min_usage_pct": min_usage,
                "samples": len(self.vram_history)
            },
            "guardian_stats": {
                "leak_alerts": self.leak_alerts,
                "cleanup_count": self.cleanup_count
            }
        }

    def save_report(self, report: Dict):
        """Save memory report to JSON"""
        report_file = self.logs_dir / f"memory_report_{datetime.now().strftime('%Y%m%d')}.json"

        # Load existing reports
        if report_file.exists():
            with report_file.open() as f:
                reports = json.load(f)
        else:
            reports = []

        reports.append(report)

        # Keep last 1440 reports (24 hours at 60s intervals)
        reports = reports[-1440:]

        with report_file.open('w') as f:
            json.dump(reports, f, indent=2)

    def run_guardian(self, interval: int = 60):
        """Main guardian loop"""
        logger.info("=" * 80)
        logger.info("🛡️  MEMORY GUARDIAN STARTED")
        logger.info("=" * 80)
        logger.info(f"Remote host: {self.remote_host}")
        logger.info(f"Warning threshold: {self.vram_warning_threshold}%")
        logger.info(f"Critical threshold: {self.vram_critical_threshold}%")
        logger.info(f"Check interval: {interval}s")
        logger.info("=" * 80)

        while True:
            try:
                # Get current stats
                stats = self.get_vram_stats()

                if 'error' not in stats:
                    # Check for memory leak
                    leak_info = self.detect_memory_leak()

                    # Log current usage
                    logger.info(
                        f"VRAM: {stats['used_mb']:.0f}MB / {stats['total_mb']:.0f}MB "
                        f"({stats['usage_pct']:.1f}%)"
                    )

                    # Handle high memory
                    self.handle_high_memory(stats, leak_info)

                    # Generate and save report
                    report = self.generate_memory_report()
                    self.save_report(report)
                else:
                    logger.error(f"Failed to get VRAM stats: {stats.get('error')}")

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("\n🛑 Memory Guardian stopped by user")
                break
            except Exception as e:
                logger.error(f"Guardian error: {e}")
                time.sleep(interval)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='3090 Memory Guardian')
    parser.add_argument('--remote-host', default=None,
                       help='Remote host (default: from hosts.json)')
    parser.add_argument('--warning-threshold', type=float, default=85.0,
                       help='VRAM warning threshold percentage')
    parser.add_argument('--critical-threshold', type=float, default=95.0,
                       help='VRAM critical threshold percentage')
    parser.add_argument('--leak-window', type=int, default=10,
                       help='Leak detection window size')
    parser.add_argument('--base-dir', default=None,
                       help='Base directory (default: auto-detect)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds')
    args = parser.parse_args()

    guardian = MemoryGuardian(
        remote_host=args.remote_host,
        vram_warning_threshold=args.warning_threshold,
        vram_critical_threshold=args.critical_threshold,
        leak_detection_window=args.leak_window,
        base_dir=args.base_dir
    )

    guardian.run_guardian(interval=args.interval)


if __name__ == "__main__":
    main()
