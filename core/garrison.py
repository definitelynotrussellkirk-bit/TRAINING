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
    # Disk
    disk_percent: Optional[float] = None
    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    # Memory
    ram_percent: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    # GPU
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    gpu_name: Optional[str] = None
    # CPU
    cpu_temp_c: Optional[float] = None
    cpu_percent: Optional[float] = None
    # Meta
    last_check: Optional[str] = None
    uptime: Optional[str] = None
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

    def _get_disk_usage(self, host: str, path: str = "/") -> dict:
        """Get disk usage for a path on host. Returns dict with percent, free_gb, total_gb."""
        success, output = self._ssh_command(host, f"df -BG {path} | tail -1")
        if not success:
            return {}

        try:
            parts = output.split()
            # Format: Filesystem Size Used Avail Use% Mounted
            # With -BG: values are in GB with 'G' suffix
            total_gb = float(parts[1].rstrip('G'))
            used_gb = float(parts[2].rstrip('G'))
            free_gb = float(parts[3].rstrip('G'))
            percent = float(parts[4].rstrip('%'))

            return {
                "percent": percent,
                "free_gb": free_gb,
                "total_gb": total_gb,
                "used_gb": used_gb,
            }
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse disk usage for {host}: {e}")
            return {}

    def _get_ram_usage(self, host: str) -> dict:
        """Get RAM usage for a host. Returns dict with percent, used_gb, total_gb."""
        success, output = self._ssh_command(host, "free -g | grep Mem")
        if not success:
            return {}

        try:
            parts = output.split()
            # Format: Mem: total used free shared buff/cache available
            total_gb = float(parts[1])
            used_gb = float(parts[2])
            free_gb = float(parts[3])
            available_gb = float(parts[6]) if len(parts) > 6 else free_gb
            percent = (used_gb / total_gb * 100) if total_gb > 0 else 0

            return {
                "percent": percent,
                "used_gb": used_gb,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "available_gb": available_gb,
            }
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse RAM usage for {host}: {e}")
            return {}

    def _get_cpu_temp(self, host: str) -> Optional[float]:
        """Get CPU temperature for a host."""
        # Try sensors first
        success, output = self._ssh_command(host, "sensors 2>/dev/null | grep -i 'core 0' | head -1")
        if success and output:
            try:
                # Format: "Core 0:        +45.0°C  (high = +80.0°C, crit = +100.0°C)"
                temp_str = output.split('+')[1].split('°')[0]
                return float(temp_str)
            except (IndexError, ValueError):
                pass

        # Try thermal zone
        success, output = self._ssh_command(host, "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null")
        if success and output:
            try:
                return float(output) / 1000  # Convert millidegrees to degrees
            except ValueError:
                pass

        return None

    def _get_gpu_info(self, host: str) -> dict:
        """Get GPU info via nvidia-smi. Returns dict with name, memory, temp."""
        success, output = self._ssh_command(
            host,
            "nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null"
        )
        if not success:
            return {}

        try:
            parts = [p.strip() for p in output.split(',')]
            if len(parts) >= 4:
                name = parts[0]
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                temp = float(parts[3])
                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

                return {
                    "name": name,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "memory_percent": mem_percent,
                    "temp_c": temp,
                }
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse GPU info for {host}: {e}")

        return {}

    def _get_uptime(self, host: str) -> Optional[str]:
        """Get system uptime."""
        success, output = self._ssh_command(host, "uptime -p 2>/dev/null || uptime")
        if success and output:
            # Clean up output
            if output.startswith("up "):
                return output[3:]
            return output.split("up")[-1].split(",")[0].strip() if "up" in output else output
        return None

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
        name = config.get("name", host_key)

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
        health.services["name"] = name

        # Check disk space
        disk = self._get_disk_usage(ssh_target)
        if disk:
            health.disk_percent = disk.get("percent")
            health.disk_free_gb = disk.get("free_gb")
            health.disk_total_gb = disk.get("total_gb")

            if health.disk_percent and health.disk_percent >= self.DISK_CRITICAL_PERCENT:
                health.status = HealthStatus.CRITICAL
                health.issues.append(f"Disk critically full: {health.disk_percent}%")
            elif health.disk_percent and health.disk_percent >= self.DISK_WARNING_PERCENT:
                if health.status == HealthStatus.HEALTHY:
                    health.status = HealthStatus.WARNING
                health.issues.append(f"Disk usage high: {health.disk_percent}%")
        else:
            health.issues.append("Could not check disk space")

        # Check RAM
        ram = self._get_ram_usage(ssh_target)
        if ram:
            health.ram_percent = ram.get("percent")
            health.ram_used_gb = ram.get("used_gb")
            health.ram_total_gb = ram.get("total_gb")

        # Check GPU via SSH (nvidia-smi)
        gpu = self._get_gpu_info(ssh_target)
        if gpu:
            health.gpu_name = gpu.get("name")
            health.gpu_memory_used_mb = gpu.get("memory_used_mb")
            health.gpu_memory_total_mb = gpu.get("memory_total_mb")
            health.gpu_memory_percent = gpu.get("memory_percent")
            health.gpu_temp_c = gpu.get("temp_c")

        # Check CPU temp
        health.cpu_temp_c = self._get_cpu_temp(ssh_target)

        # Check uptime
        health.uptime = self._get_uptime(ssh_target)

        # Check checkpoint count
        if models_path:
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
                # API might have more accurate GPU info
                data = resp.json()
                if "gpu" in data and not health.gpu_memory_percent:
                    gpu_used = data["gpu"].get("memory_used_mb", 0)
                    gpu_total = data["gpu"].get("memory_total_mb", 1)
                    health.gpu_memory_percent = (gpu_used / gpu_total) * 100
                    health.gpu_memory_used_mb = gpu_used
                    health.gpu_memory_total_mb = gpu_total
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
        """Check health of training machine (local)."""
        config = self.hosts_config.get("4090", {})
        name = config.get("name", "Training Server")

        health = HostHealth(
            host="localhost",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )
        health.services["name"] = name

        # Check local disk space
        try:
            result = subprocess.run(
                ["df", "-BG", str(self.base_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split('\n')[-1].split()
                health.disk_total_gb = float(parts[1].rstrip('G'))
                health.disk_free_gb = float(parts[3].rstrip('G'))
                health.disk_percent = float(parts[4].rstrip('%'))

                if health.disk_percent >= self.DISK_CRITICAL_PERCENT:
                    health.status = HealthStatus.CRITICAL
                    health.issues.append(f"Disk critically full: {health.disk_percent}%")
                elif health.disk_percent >= self.DISK_WARNING_PERCENT:
                    health.status = HealthStatus.WARNING
                    health.issues.append(f"Disk usage high: {health.disk_percent}%")
        except Exception as e:
            health.issues.append(f"Could not check disk: {e}")

        # Check local RAM
        try:
            result = subprocess.run(["free", "-g"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Mem:'):
                        parts = line.split()
                        health.ram_total_gb = float(parts[1])
                        health.ram_used_gb = float(parts[2])
                        health.ram_percent = (health.ram_used_gb / health.ram_total_gb * 100) if health.ram_total_gb > 0 else 0
                        break
        except Exception:
            pass

        # Check local GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 4:
                    health.gpu_name = parts[0]
                    health.gpu_memory_used_mb = float(parts[1])
                    health.gpu_memory_total_mb = float(parts[2])
                    health.gpu_temp_c = float(parts[3])
                    health.gpu_memory_percent = (health.gpu_memory_used_mb / health.gpu_memory_total_mb * 100) if health.gpu_memory_total_mb > 0 else 0
        except Exception:
            pass

        # Check CPU temp
        try:
            result = subprocess.run(
                ["cat", "/sys/class/thermal/thermal_zone0/temp"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                health.cpu_temp_c = float(result.stdout.strip()) / 1000
        except Exception:
            pass

        # Check uptime
        try:
            result = subprocess.run(["uptime", "-p"], capture_output=True, text=True)
            if result.returncode == 0:
                health.uptime = result.stdout.strip().replace("up ", "")
        except Exception:
            pass

        # Check key services
        services_to_check = [
            ("vault", 8767),
            ("tavern", 8888),
        ]

        for svc_name, port in services_to_check:
            try:
                import requests
                resp = requests.get(f"http://localhost:{port}/health", timeout=3)
                health.services[svc_name] = "up" if resp.status_code == 200 else "error"
            except:
                health.services[svc_name] = "down"

        # Check training daemon
        try:
            result = subprocess.run(["pgrep", "-f", "training_daemon"], capture_output=True)
            health.services["training_daemon"] = "up" if result.returncode == 0 else "down"
        except:
            health.services["training_daemon"] = "unknown"

        # Check eval runner
        try:
            result = subprocess.run(["pgrep", "-f", "eval_runner"], capture_output=True)
            health.services["eval_runner"] = "up" if result.returncode == 0 else "down"
        except:
            health.services["eval_runner"] = "unknown"

        # Check garrison
        try:
            result = subprocess.run(["pgrep", "-f", "garrison"], capture_output=True)
            health.services["garrison"] = "up" if result.returncode == 0 else "down"
        except:
            health.services["garrison"] = "unknown"

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
