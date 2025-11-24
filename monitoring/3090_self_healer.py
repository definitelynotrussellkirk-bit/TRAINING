#!/usr/bin/env python3
"""
3090 Self-Healing System - Automatic recovery from common failures
Implements recovery playbooks for known failure modes
"""

import subprocess
import time
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfHealer:
    """Automatic recovery from common 3090 failures"""

    def __init__(
        self,
        remote_host: str = "192.168.x.x",
        api_port: int = 8765,
        base_dir: str = "/path/to/training"
    ):
        self.remote_host = remote_host
        self.api_port = api_port
        self.base_dir = Path(base_dir)

        # Recovery stats
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0

        # Logs
        self.logs_dir = self.base_dir / "logs" / "self_healer"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def diagnose_failure(self) -> Dict:
        """Diagnose current system state"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "severity": "normal"
        }

        # Check 1: API responding?
        try:
            response = requests.get(
                f"http://{self.remote_host}:{self.api_port}/health",
                timeout=5
            )
            if response.status_code != 200:
                diagnosis["issues"].append({
                    "type": "api_unhealthy",
                    "severity": "critical",
                    "details": f"HTTP {response.status_code}"
                })
                diagnosis["severity"] = "critical"
        except requests.exceptions.ConnectionError:
            diagnosis["issues"].append({
                "type": "api_not_responding",
                "severity": "critical",
                "details": "Connection refused"
            })
            diagnosis["severity"] = "critical"
        except requests.exceptions.Timeout:
            diagnosis["issues"].append({
                "type": "api_slow",
                "severity": "warning",
                "details": "Response timeout >5s"
            })
            if diagnosis["severity"] != "critical":
                diagnosis["severity"] = "warning"

        # Check 2: Process running?
        try:
            cmd = ["ssh", self.remote_host, "pgrep -f 'python3 main.py'"]
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode != 0:
                diagnosis["issues"].append({
                    "type": "process_not_running",
                    "severity": "critical",
                    "details": "main.py process not found"
                })
                diagnosis["severity"] = "critical"
        except Exception as e:
            diagnosis["issues"].append({
                "type": "ssh_failure",
                "severity": "critical",
                "details": str(e)
            })
            diagnosis["severity"] = "critical"

        # Check 3: GPU accessible?
        try:
            cmd = ["ssh", self.remote_host, "nvidia-smi --query-gpu=name --format=csv,noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                diagnosis["issues"].append({
                    "type": "gpu_not_accessible",
                    "severity": "critical",
                    "details": "nvidia-smi failed"
                })
                diagnosis["severity"] = "critical"
        except Exception as e:
            diagnosis["issues"].append({
                "type": "gpu_check_failed",
                "severity": "warning",
                "details": str(e)
            })

        # Check 4: High VRAM usage?
        try:
            cmd = [
                "ssh", self.remote_host,
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(", "))
                usage_pct = (used / total) * 100
                if usage_pct > 95:
                    diagnosis["issues"].append({
                        "type": "vram_critical",
                        "severity": "critical",
                        "details": f"VRAM usage: {usage_pct:.1f}%"
                    })
                    diagnosis["severity"] = "critical"
                elif usage_pct > 85:
                    diagnosis["issues"].append({
                        "type": "vram_high",
                        "severity": "warning",
                        "details": f"VRAM usage: {usage_pct:.1f}%"
                    })
                    if diagnosis["severity"] == "normal":
                        diagnosis["severity"] = "warning"
        except Exception:
            pass

        # Check 5: Zombie processes?
        try:
            cmd = ["ssh", self.remote_host, "ps aux | grep 'python3 main.py' | grep -v grep | wc -l"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                count = int(result.stdout.strip())
                if count > 1:
                    diagnosis["issues"].append({
                        "type": "multiple_processes",
                        "severity": "warning",
                        "details": f"{count} main.py processes running"
                    })
                    if diagnosis["severity"] == "normal":
                        diagnosis["severity"] = "warning"
        except Exception:
            pass

        return diagnosis

    def playbook_restart_api(self) -> bool:
        """Recovery Playbook: Restart API server"""
        logger.info("📖 Running playbook: RESTART_API")

        try:
            # Step 1: Kill existing processes
            logger.info("  Step 1: Killing existing processes...")
            kill_cmd = ["ssh", self.remote_host, "pkill -f 'python3 main.py'"]
            subprocess.run(kill_cmd, timeout=10)
            time.sleep(3)

            # Step 2: Clear any GPU memory
            logger.info("  Step 2: Clearing GPU memory...")
            clear_cmd = ["ssh", self.remote_host, "nvidia-smi --gpu-reset || true"]
            subprocess.run(clear_cmd, timeout=10)
            time.sleep(2)

            # Step 3: Start fresh process
            logger.info("  Step 3: Starting fresh process...")
            start_cmd = [
                "ssh", self.remote_host,
                "cd ~/llm && source venv/bin/activate && "
                "nohup python3 main.py > logs/api_server.log 2>&1 &"
            ]
            subprocess.run(start_cmd, timeout=15)
            time.sleep(5)

            # Step 4: Verify
            logger.info("  Step 4: Verifying...")
            verify_cmd = ["ssh", self.remote_host, "pgrep -f 'python3 main.py'"]
            result = subprocess.run(verify_cmd, capture_output=True, timeout=5)

            if result.returncode == 0:
                logger.info("  ✅ Playbook RESTART_API succeeded")
                return True
            else:
                logger.error("  ❌ Playbook RESTART_API failed (process not started)")
                return False

        except Exception as e:
            logger.error(f"  ❌ Playbook RESTART_API failed: {e}")
            return False

    def playbook_clear_vram(self) -> bool:
        """Recovery Playbook: Clear VRAM without full restart"""
        logger.info("📖 Running playbook: CLEAR_VRAM")

        try:
            # Step 1: Call API clear cache endpoint
            logger.info("  Step 1: Calling API clear_cache...")
            try:
                response = requests.post(
                    f"http://{self.remote_host}:{self.api_port}/clear_cache",
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info("  ✅ Cache cleared via API")
                    return True
            except Exception:
                logger.warning("  API clear_cache failed, trying alternative...")

            # Step 2: If API fails, restart the server
            logger.info("  Step 2: Restarting API to clear memory...")
            return self.playbook_restart_api()

        except Exception as e:
            logger.error(f"  ❌ Playbook CLEAR_VRAM failed: {e}")
            return False

    def playbook_kill_zombies(self) -> bool:
        """Recovery Playbook: Kill zombie processes"""
        logger.info("📖 Running playbook: KILL_ZOMBIES")

        try:
            # Step 1: Find all main.py processes
            logger.info("  Step 1: Finding zombie processes...")
            find_cmd = ["ssh", self.remote_host, "pgrep -f 'python3 main.py'"]
            result = subprocess.run(find_cmd, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                logger.info("  No processes found")
                return True

            pids = result.stdout.strip().split('\n')
            logger.info(f"  Found {len(pids)} processes: {pids}")

            # Step 2: Kill all
            logger.info("  Step 2: Killing all processes...")
            kill_cmd = ["ssh", self.remote_host, "pkill -9 -f 'python3 main.py'"]
            subprocess.run(kill_cmd, timeout=10)
            time.sleep(2)

            # Step 3: Restart clean process
            logger.info("  Step 3: Starting clean process...")
            return self.playbook_restart_api()

        except Exception as e:
            logger.error(f"  ❌ Playbook KILL_ZOMBIES failed: {e}")
            return False

    def playbook_network_reset(self) -> bool:
        """Recovery Playbook: Reset network connections"""
        logger.info("📖 Running playbook: NETWORK_RESET")

        try:
            # Step 1: Test SSH connection
            logger.info("  Step 1: Testing SSH connection...")
            test_cmd = ["ssh", self.remote_host, "echo 'connection ok'"]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                logger.error("  ❌ SSH connection failed - cannot recover remotely")
                logger.error("  Manual intervention required")
                return False

            logger.info("  ✅ SSH connection OK")

            # Step 2: Restart API (network issue likely on API side)
            return self.playbook_restart_api()

        except Exception as e:
            logger.error(f"  ❌ Playbook NETWORK_RESET failed: {e}")
            return False

    def select_and_run_playbook(self, diagnosis: Dict) -> bool:
        """Select appropriate playbook based on diagnosis"""
        issues = diagnosis.get("issues", [])

        if not issues:
            logger.info("No issues detected - system healthy")
            return True

        # Priority order of playbooks
        issue_types = [issue["type"] for issue in issues]

        # Critical: Process not running or API not responding
        if "process_not_running" in issue_types or "api_not_responding" in issue_types:
            return self.playbook_restart_api()

        # Multiple processes (zombies)
        if "multiple_processes" in issue_types:
            return self.playbook_kill_zombies()

        # High VRAM usage
        if "vram_critical" in issue_types or "vram_high" in issue_types:
            return self.playbook_clear_vram()

        # API issues
        if "api_unhealthy" in issue_types or "api_slow" in issue_types:
            return self.playbook_restart_api()

        # SSH/network issues
        if "ssh_failure" in issue_types:
            return self.playbook_network_reset()

        # Default: try restart
        logger.warning("Unknown issue types, trying default restart...")
        return self.playbook_restart_api()

    def run_healing_cycle(self):
        """Run one healing cycle"""
        logger.info("=" * 80)
        logger.info("🔍 Running diagnostic scan...")

        # Diagnose
        diagnosis = self.diagnose_failure()

        if diagnosis["severity"] == "normal":
            logger.info("✅ System healthy - no action needed")
            return True

        # Log issues
        logger.warning(f"⚠️  Issues detected (severity: {diagnosis['severity']})")
        for issue in diagnosis["issues"]:
            logger.warning(f"   - {issue['type']}: {issue['details']} [{issue['severity']}]")

        # Attempt recovery
        logger.info("🔧 Attempting automatic recovery...")
        self.recovery_attempts += 1

        success = self.select_and_run_playbook(diagnosis)

        if success:
            self.successful_recoveries += 1
            logger.info("✅ Recovery successful!")
            return True
        else:
            self.failed_recoveries += 1
            logger.error("❌ Recovery failed - manual intervention required")
            return False

    def run_forever(self, interval: int = 300):
        """Run self-healing continuously"""
        logger.info("=" * 80)
        logger.info("🏥 SELF-HEALING SYSTEM STARTED")
        logger.info("=" * 80)
        logger.info(f"Remote host: {self.remote_host}:{self.api_port}")
        logger.info(f"Check interval: {interval}s")
        logger.info("=" * 80)

        while True:
            try:
                self.run_healing_cycle()

                logger.info(f"\nRecovery Stats:")
                logger.info(f"  Attempts: {self.recovery_attempts}")
                logger.info(f"  Successful: {self.successful_recoveries}")
                logger.info(f"  Failed: {self.failed_recoveries}")
                logger.info(f"\nNext check in {interval}s...\n")

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("\n🛑 Self-healer stopped by user")
                break
            except Exception as e:
                logger.error(f"Self-healer error: {e}")
                time.sleep(interval)


def main():
    healer = SelfHealer(
        remote_host="192.168.x.x",
        api_port=8765,
        base_dir="/path/to/training"
    )

    # Run healing check every 5 minutes
    healer.run_forever(interval=300)


if __name__ == "__main__":
    main()
