#!/usr/bin/env python3
"""
Garrison - Fleet Health Manager

The Garrison monitors and maintains the health of all distributed services:
- Inference servers (disk space, GPU health, model status)
- Workers (heartbeat, job processing)
- Training machines (disk space, GPU utilization)

RPG Flavor: The Garrison is where the army maintains its strength.
Soldiers rest, equipment is repaired, and supplies are managed.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("garrison")

# Project imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import get_base_dir


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"      # All good
    WARNING = "warning"      # Needs attention soon
    CRITICAL = "critical"    # Needs immediate attention
    UNKNOWN = "unknown"      # Can't determine


@dataclass
class HostHealth:
    """Health report for a single host."""
    host: str
    status: HealthStatus
    disk_percent: Optional[float] = None
    disk_free_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    last_check: Optional[str] = None
    services: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    maintenance_performed: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class FleetHealth:
    """Overall fleet health report."""
    timestamp: str
    overall_status: HealthStatus
    hosts: dict  # host -> HostHealth
    alerts: list = field(default_factory=list)
    maintenance_log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "hosts": {k: v.to_dict() for k, v in self.hosts.items()},
            "alerts": self.alerts,
            "maintenance_log": self.maintenance_log,
        }


class Garrison:
    """
    Fleet health manager.

    Monitors all hosts in the fleet and performs automatic maintenance.
    """

    # Thresholds
    DISK_WARNING_PERCENT = 80
    DISK_CRITICAL_PERCENT = 90
    CHECKPOINT_MAX_COUNT = 10

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.hosts_file = self.base_dir / "config" / "hosts.json"
        self.status_file = self.base_dir / "status" / "garrison.json"
        self.hosts_config = self._load_hosts()

    def _load_hosts(self) -> dict:
        """Load host configuration."""
        if self.hosts_file.exists():
            with open(self.hosts_file) as f:
                data = json.load(f)
            # Handle nested "hosts" structure
            return data.get("hosts", data)
        return {}

    def _ssh_command(self, host: str, command: str, timeout: int = 30) -> tuple[bool, str]:
        """Execute SSH command on remote host."""
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    def _get_disk_usage(self, host: str, path: str = "/") -> tuple[Optional[float], Optional[float]]:
        """Get disk usage percent and free GB for a path on host."""
        success, output = self._ssh_command(host, f"df -h {path} | tail -1")
        if not success:
            return None, None

        try:
            parts = output.split()
            # Format: Filesystem Size Used Avail Use% Mounted
            percent = float(parts[4].rstrip('%'))
            avail = parts[3]

            # Parse available space
            if avail.endswith('G'):
                free_gb = float(avail[:-1])
            elif avail.endswith('M'):
                free_gb = float(avail[:-1]) / 1024
            elif avail.endswith('T'):
                free_gb = float(avail[:-1]) * 1024
            else:
                free_gb = float(avail) / (1024**3)  # Assume bytes

            return percent, free_gb
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse disk usage for {host}: {e}")
            return None, None

    def _count_checkpoints(self, host: str, path: str) -> Optional[int]:
        """Count checkpoints on remote host."""
        success, output = self._ssh_command(host, f"ls {path} 2>/dev/null | wc -l")
        if success:
            try:
                return int(output.strip())
            except ValueError:
                pass
        return None

    def _cleanup_checkpoints(self, host: str, path: str, keep: int = 10) -> tuple[bool, int]:
        """Remove old checkpoints, keeping the most recent ones."""
        # First count what would be deleted
        success, output = self._ssh_command(
            host,
            f"cd {path} && ls -t 2>/dev/null | tail -n +{keep + 1} | wc -l"
        )

        if not success:
            return False, 0

        try:
            to_delete = int(output.strip())
        except ValueError:
            return False, 0

        if to_delete == 0:
            return True, 0

        # Perform cleanup
        cleanup_cmd = f"cd {path} && ls -t | tail -n +{keep + 1} | xargs -r rm -rf"
        success, _ = self._ssh_command(host, cleanup_cmd, timeout=120)

        if success:
            logger.info(f"Cleaned up {to_delete} checkpoints on {host}:{path}")
            return True, to_delete

        return False, 0

    def check_inference_server(self, host_key: str = "3090") -> HostHealth:
        """Check health of inference server."""
        config = self.hosts_config.get(host_key, {})
        host = config.get("host", config.get("ip", ""))
        user = config.get("ssh_user", config.get("user", "russ"))
        models_path = config.get("models_dir", config.get("checkpoints_dir", ""))

        if not host:
            return HostHealth(
                host=host_key,
                status=HealthStatus.UNKNOWN,
                issues=["No host configured"],
            )

        ssh_target = f"{user}@{host}"
        health = HostHealth(
            host=host,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check disk space
        percent, free_gb = self._get_disk_usage(ssh_target)
        if percent is not None:
            health.disk_percent = percent
            health.disk_free_gb = free_gb

            if percent >= self.DISK_CRITICAL_PERCENT:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"Disk critically full: {percent}%")
            elif percent >= self.DISK_WARNING_PERCENT:
                if health.status == HealthStatus.HEALTHY:
                    health.status = HealthStatus.WARNING
                health.issues.append(f"Disk usage high: {percent}%")
        else:
            health.issues.append("Could not check disk space")

        # Check checkpoint count
        checkpoint_count = self._count_checkpoints(ssh_target, models_path)
        if checkpoint_count is not None:
            health.services["checkpoints"] = checkpoint_count
            if checkpoint_count > self.CHECKPOINT_MAX_COUNT + 5:
                health.issues.append(f"Too many checkpoints: {checkpoint_count}")

        # Check inference server API
        services = config.get("services", {})
        port = services.get("inference", {}).get("port", config.get("port", 8765))
        api_url = f"http://{host}:{port}/health"
        try:
            import requests
            resp = requests.get(api_url, timeout=5)
            if resp.status_code == 200:
                health.services["inference_api"] = "up"
                data = resp.json()
                if "gpu" in data:
                    gpu_used = data["gpu"].get("memory_used_mb", 0)
                    gpu_total = data["gpu"].get("memory_total_mb", 1)
                    health.gpu_memory_percent = (gpu_used / gpu_total) * 100
            else:
                health.services["inference_api"] = "error"
                health.issues.append(f"Inference API returned {resp.status_code}")
        except Exception as e:
            health.services["inference_api"] = "down"
            health.issues.append(f"Inference API unreachable: {e}")
            if health.status == HealthStatus.HEALTHY:
                health.status = HealthStatus.WARNING

        return health

    def check_trainer(self, host_key: str = "trainer") -> HostHealth:
        """Check health of training machine."""
        health = HostHealth(
            host="localhost",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check local disk space
        try:
            result = subprocess.run(
                ["df", "-h", str(self.base_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split('\n')[-1].split()
                percent = float(parts[4].rstrip('%'))
                health.disk_percent = percent

                avail = parts[3]
                if avail.endswith('G'):
                    health.disk_free_gb = float(avail[:-1])
                elif avail.endswith('T'):
                    health.disk_free_gb = float(avail[:-1]) * 1024

                if percent >= self.DISK_CRITICAL_PERCENT:
                    health.status = HealthStatus.CRITICAL
                    health.issues.append(f"Disk critically full: {percent}%")
                elif percent >= self.DISK_WARNING_PERCENT:
                    health.status = HealthStatus.WARNING
                    health.issues.append(f"Disk usage high: {percent}%")
        except Exception as e:
            health.issues.append(f"Could not check disk: {e}")

        # Check key services
        services_to_check = [
            ("vault", 8767),
            ("tavern", 8888),
        ]

        for name, port in services_to_check:
            try:
                import requests
                resp = requests.get(f"http://localhost:{port}/health", timeout=3)
                health.services[name] = "up" if resp.status_code == 200 else "error"
            except:
                health.services[name] = "down"

        # Check training daemon
        try:
            result = subprocess.run(
                ["pgrep", "-f", "training_daemon"],
                capture_output=True,
            )
            health.services["training_daemon"] = "up" if result.returncode == 0 else "down"
        except:
            health.services["training_daemon"] = "unknown"

        # Check eval runner
        try:
            result = subprocess.run(
                ["pgrep", "-f", "eval_runner"],
                capture_output=True,
            )
            health.services["eval_runner"] = "up" if result.returncode == 0 else "down"
        except:
            health.services["eval_runner"] = "unknown"

        return health

    def check_worker(self, host_key: str, host_config: dict) -> HostHealth:
        """Check health of a worker machine."""
        host = host_config.get("host", host_config.get("ip", ""))
        user = host_config.get("user", "root")

        if not host:
            return HostHealth(
                host=host_key,
                status=HealthStatus.UNKNOWN,
                issues=["No host configured"],
            )

        ssh_target = f"{user}@{host}"
        health = HostHealth(
            host=host,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check disk space
        percent, free_gb = self._get_disk_usage(ssh_target)
        if percent is not None:
            health.disk_percent = percent
            health.disk_free_gb = free_gb

            if percent >= self.DISK_CRITICAL_PERCENT:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"Disk critically full: {percent}%")
            elif percent >= self.DISK_WARNING_PERCENT:
                if health.status == HealthStatus.HEALTHY:
                    health.status = HealthStatus.WARNING
                health.issues.append(f"Disk usage high: {percent}%")
        else:
            health.status = HealthStatus.UNKNOWN
            health.issues.append("Could not connect to host")

        # Check worker process
        success, output = self._ssh_command(ssh_target, "pgrep -f claiming_worker")
        health.services["worker"] = "up" if success and output else "down"

        return health

    def perform_maintenance(self, dry_run: bool = False) -> list[str]:
        """
        Perform automatic maintenance tasks.

        Returns list of actions taken.
        """
        actions = []

        # Maintenance: Clean up inference server checkpoints
        inference_config = self.hosts_config.get("3090", {})
        host = inference_config.get("host", "")
        user = inference_config.get("ssh_user", "russ")
        models_path = inference_config.get("models_dir", "")

        if host and models_path:
            ssh_target = f"{user}@{host}"
            checkpoint_count = self._count_checkpoints(ssh_target, models_path)

            if checkpoint_count and checkpoint_count > self.CHECKPOINT_MAX_COUNT:
                if dry_run:
                    actions.append(f"[DRY RUN] Would clean up {checkpoint_count - self.CHECKPOINT_MAX_COUNT} checkpoints on inference server")
                else:
                    success, deleted = self._cleanup_checkpoints(ssh_target, models_path, self.CHECKPOINT_MAX_COUNT)
                    if success and deleted > 0:
                        actions.append(f"Cleaned up {deleted} checkpoints on inference server")

        # Maintenance: Clean up local temp files
        temp_patterns = [
            self.base_dir / "logs" / "*.log.old",
            self.base_dir / ".pids" / "*.pid.old",
        ]

        for pattern in temp_patterns:
            try:
                import glob
                old_files = glob.glob(str(pattern))
                if old_files:
                    if dry_run:
                        actions.append(f"[DRY RUN] Would delete {len(old_files)} old files matching {pattern.name}")
                    else:
                        for f in old_files:
                            Path(f).unlink()
                        actions.append(f"Deleted {len(old_files)} old files matching {pattern.name}")
            except Exception as e:
                logger.warning(f"Maintenance error for {pattern}: {e}")

        return actions

    def get_fleet_health(self, perform_maintenance: bool = False) -> FleetHealth:
        """
        Get complete fleet health report.

        If perform_maintenance is True, also runs maintenance tasks.
        """
        hosts = {}
        alerts = []
        maintenance_log = []

        # Check trainer (local)
        trainer_health = self.check_trainer()
        hosts["trainer"] = trainer_health

        # Check inference server
        if "3090" in self.hosts_config:
            inference_health = self.check_inference_server("3090")
            hosts["inference"] = inference_health

            if inference_health.status == HealthStatus.CRITICAL:
                alerts.append(f"CRITICAL: Inference server {inference_health.host} - {', '.join(inference_health.issues)}")
            elif inference_health.status == HealthStatus.WARNING:
                alerts.append(f"WARNING: Inference server {inference_health.host} - {', '.join(inference_health.issues)}")

        # Check workers
        for key, config in self.hosts_config.items():
            if config.get("role") == "worker" or key.startswith("worker"):
                worker_health = self.check_worker(key, config)
                hosts[key] = worker_health

                if worker_health.status in (HealthStatus.CRITICAL, HealthStatus.WARNING):
                    alerts.append(f"{worker_health.status.value.upper()}: Worker {key} - {', '.join(worker_health.issues)}")

        # Perform maintenance if requested
        if perform_maintenance:
            maintenance_log = self.perform_maintenance()

        # Determine overall status
        statuses = [h.status for h in hosts.values()]
        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            overall = HealthStatus.WARNING
        else:
            overall = HealthStatus.HEALTHY

        report = FleetHealth(
            timestamp=datetime.now().isoformat(),
            overall_status=overall,
            hosts=hosts,
            alerts=alerts,
            maintenance_log=maintenance_log,
        )

        # Save status file
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report

    def run_daemon(self, interval: int = 300, maintenance_interval: int = 3600):
        """
        Run as daemon, checking health periodically.

        Args:
            interval: Health check interval in seconds (default 5 min)
            maintenance_interval: Maintenance interval in seconds (default 1 hour)
        """
        # Write PID file
        pid_file = self.base_dir / ".pids" / "garrison.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))

        logger.info(f"Garrison daemon starting (check every {interval}s, maintenance every {maintenance_interval}s)")

        last_maintenance = 0

        while True:
            try:
                now = time.time()
                do_maintenance = (now - last_maintenance) >= maintenance_interval

                report = self.get_fleet_health(perform_maintenance=do_maintenance)

                if do_maintenance:
                    last_maintenance = now
                    if report.maintenance_log:
                        logger.info(f"Maintenance: {', '.join(report.maintenance_log)}")

                # Log status
                status_icon = {
                    HealthStatus.HEALTHY: "✓",
                    HealthStatus.WARNING: "⚠",
                    HealthStatus.CRITICAL: "✗",
                    HealthStatus.UNKNOWN: "?",
                }

                logger.info(f"Fleet status: {status_icon[report.overall_status]} {report.overall_status.value}")

                for host, health in report.hosts.items():
                    icon = status_icon[health.status]
                    disk_info = f"disk={health.disk_percent:.0f}%" if health.disk_percent else ""
                    logger.info(f"  {icon} {host}: {health.status.value} {disk_info}")

                if report.alerts:
                    for alert in report.alerts:
                        logger.warning(f"  {alert}")

            except Exception as e:
                logger.error(f"Health check failed: {e}")

            time.sleep(interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Garrison - Fleet Health Manager")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=300, help="Check interval (seconds)")
    parser.add_argument("--maintenance-interval", type=int, default=3600, help="Maintenance interval (seconds)")
    parser.add_argument("--maintenance", action="store_true", help="Run maintenance now")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually perform maintenance")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    garrison = Garrison()

    if args.daemon:
        garrison.run_daemon(args.interval, args.maintenance_interval)
    elif args.maintenance:
        actions = garrison.perform_maintenance(dry_run=args.dry_run)
        if args.json:
            print(json.dumps({"actions": actions}, indent=2))
        else:
            if actions:
                for action in actions:
                    print(f"  {action}")
            else:
                print("No maintenance needed")
    else:
        report = garrison.get_fleet_health()

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            status_icon = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.WARNING: "⚠",
                HealthStatus.CRITICAL: "✗",
                HealthStatus.UNKNOWN: "?",
            }

            print(f"\n{'='*50}")
            print(f"  GARRISON - Fleet Health Report")
            print(f"  {report.timestamp}")
            print(f"{'='*50}")
            print(f"\nOverall: {status_icon[report.overall_status]} {report.overall_status.value.upper()}")

            print(f"\nHosts:")
            for host, health in report.hosts.items():
                icon = status_icon[health.status]
                print(f"\n  {icon} {host} ({health.host})")
                if health.disk_percent is not None:
                    print(f"    Disk: {health.disk_percent:.1f}% used ({health.disk_free_gb:.1f}GB free)")
                if health.gpu_memory_percent is not None:
                    print(f"    GPU: {health.gpu_memory_percent:.1f}% memory")
                if health.services:
                    svc_str = ", ".join(f"{k}={v}" for k, v in health.services.items())
                    print(f"    Services: {svc_str}")
                if health.issues:
                    for issue in health.issues:
                        print(f"    ⚠ {issue}")

            if report.alerts:
                print(f"\nAlerts:")
                for alert in report.alerts:
                    print(f"  • {alert}")

            print()


if __name__ == "__main__":
    main()
