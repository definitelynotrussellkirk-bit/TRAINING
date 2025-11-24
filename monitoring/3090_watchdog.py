#!/usr/bin/env python3
"""
3090 Process Watchdog - Auto-restart and health monitoring
Ensures the API server on 192.168.x.x stays running 24/7
"""

import subprocess
import time
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Remote3090Watchdog:
    """Monitors and auto-restarts the 3090 API server"""

    def __init__(
        self,
        remote_host: str = "192.168.x.x",
        api_port: int = 8765,
        check_interval: int = 60,
        max_restart_attempts: int = 3,
        base_dir: str = "/path/to/training"
    ):
        self.remote_host = remote_host
        self.api_port = api_port
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.base_dir = Path(base_dir)

        # State tracking
        self.consecutive_failures = 0
        self.restart_count = 0
        self.last_restart_time = None
        self.uptime_start = datetime.now()

        # Logs directory
        self.logs_dir = self.base_dir / "logs" / "3090_watchdog"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Status file
        self.status_file = self.base_dir / "status" / "3090_watchdog_status.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def check_api_health(self) -> Dict:
        """Check if API is responding"""
        try:
            url = f"http://{self.remote_host}:{self.api_port}/health"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "error": "Request timeout (>10s)",
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "connection_refused",
                "error": "Cannot connect to API",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def check_process_running(self) -> bool:
        """Check if main.py process is running on remote host"""
        try:
            cmd = [
                "ssh", self.remote_host,
                "ps aux | grep 'python3 main.py' | grep -v grep"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except Exception as e:
            logger.error(f"Process check failed: {e}")
            return False

    def restart_api_server(self) -> bool:
        """Restart the API server on remote host"""
        try:
            logger.warning("🔄 Attempting to restart API server...")

            # Step 1: Kill existing processes
            kill_cmd = [
                "ssh", self.remote_host,
                "pkill -f 'python3 main.py'"
            ]
            subprocess.run(kill_cmd, timeout=10)
            time.sleep(2)

            # Step 2: Start new process
            start_cmd = [
                "ssh", self.remote_host,
                "cd ~/llm && source venv/bin/activate && "
                "nohup python3 main.py > logs/api_server.log 2>&1 &"
            ]
            subprocess.run(start_cmd, timeout=15)
            time.sleep(5)

            # Step 3: Verify it started
            if self.check_process_running():
                logger.info("✅ API server restarted successfully")
                self.restart_count += 1
                self.last_restart_time = datetime.now()
                self.consecutive_failures = 0
                return True
            else:
                logger.error("❌ API server failed to start")
                return False

        except Exception as e:
            logger.error(f"Restart failed: {e}")
            return False

    def get_gpu_health(self) -> Dict:
        """Get GPU health metrics from remote host"""
        try:
            cmd = [
                "ssh", self.remote_host,
                "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,"
                "utilization.gpu,utilization.memory --format=csv,noheader,nounits"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                temp, mem_used, mem_total, gpu_util, mem_util = map(float, parts)

                return {
                    "temperature_c": temp,
                    "vram_used_mb": mem_used,
                    "vram_total_mb": mem_total,
                    "vram_usage_pct": (mem_used / mem_total) * 100,
                    "gpu_utilization_pct": gpu_util,
                    "memory_utilization_pct": mem_util,
                    "status": "healthy" if temp < 85 else "thermal_warning",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"status": "error", "error": "nvidia-smi failed"}

        except Exception as e:
            logger.error(f"GPU health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def save_status(self, health_check: Dict, gpu_health: Dict):
        """Save current status to JSON file"""
        status = {
            "api_health": health_check,
            "gpu_health": gpu_health,
            "watchdog": {
                "uptime_start": self.uptime_start.isoformat(),
                "consecutive_failures": self.consecutive_failures,
                "restart_count": self.restart_count,
                "last_restart_time": self.last_restart_time.isoformat() if self.last_restart_time else None,
                "timestamp": datetime.now().isoformat()
            }
        }

        self.status_file.write_text(json.dumps(status, indent=2))

    def alert(self, message: str, severity: str = "warning"):
        """Log alert (can be extended to email/SMS/webhook)"""
        prefix = "🚨" if severity == "critical" else "⚠️"
        logger.warning(f"{prefix} ALERT [{severity.upper()}]: {message}")

        # Write to alert log
        alert_log = self.logs_dir / "alerts.log"
        with alert_log.open("a") as f:
            f.write(f"{datetime.now().isoformat()} [{severity.upper()}] {message}\n")

    def run_health_check(self) -> bool:
        """Run full health check and handle failures"""
        # Check API health
        api_health = self.check_api_health()
        gpu_health = self.get_gpu_health()

        # Save status
        self.save_status(api_health, gpu_health)

        # Handle API health
        if api_health["status"] == "healthy":
            logger.info(f"✅ API healthy (response time: {api_health['response_time']:.3f}s)")
            self.consecutive_failures = 0
            return True
        else:
            self.consecutive_failures += 1
            logger.warning(
                f"❌ API unhealthy: {api_health.get('error', 'unknown')} "
                f"(failures: {self.consecutive_failures})"
            )

            # Try to restart if failures persist
            if self.consecutive_failures >= 2:
                if self.restart_count < self.max_restart_attempts:
                    self.alert(
                        f"API unhealthy for {self.consecutive_failures} checks, attempting restart",
                        severity="warning"
                    )
                    success = self.restart_api_server()

                    if not success and self.consecutive_failures >= 5:
                        self.alert(
                            "API restart failed and failures persist",
                            severity="critical"
                        )
                else:
                    self.alert(
                        f"Max restart attempts ({self.max_restart_attempts}) reached",
                        severity="critical"
                    )

            return False

        # Check GPU health
        if gpu_health.get("status") == "thermal_warning":
            self.alert(
                f"GPU temperature high: {gpu_health['temperature_c']}°C",
                severity="warning"
            )

        if gpu_health.get("vram_usage_pct", 0) > 95:
            self.alert(
                f"VRAM usage critical: {gpu_health['vram_usage_pct']:.1f}%",
                severity="warning"
            )

    def run_forever(self):
        """Main watchdog loop"""
        logger.info("=" * 80)
        logger.info("🐕 3090 WATCHDOG STARTED")
        logger.info("=" * 80)
        logger.info(f"Remote host: {self.remote_host}:{self.api_port}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Max restart attempts: {self.max_restart_attempts}")
        logger.info(f"Status file: {self.status_file}")
        logger.info("=" * 80)

        while True:
            try:
                self.run_health_check()
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("\n🛑 Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(self.check_interval)


def main():
    watchdog = Remote3090Watchdog(
        remote_host="192.168.x.x",
        api_port=8765,
        check_interval=60,  # Check every 60 seconds
        max_restart_attempts=5,
        base_dir="/path/to/training"
    )

    watchdog.run_forever()


if __name__ == "__main__":
    main()
