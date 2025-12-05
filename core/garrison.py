#!/usr/bin/env python3
"""
Garrison - Fleet Health Manager

The Garrison monitors and maintains the health of all distributed services:
- Training machines (GPU, disk, services)
- Inference servers (GPU, disk, API health)
- Storage hosts (NAS volumes, RAID, temps)
- Compute workers (CPU load, memory, job status)

RPG Flavor: The Garrison is where the army maintains its strength.
Soldiers rest, equipment is repaired, and supplies are managed.
"""

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
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
    OFFLINE = "offline"      # Host unreachable


class DeviceRole(Enum):
    """Device roles from devices.json."""
    TRAINER = "trainer"
    INFERENCE = "inference"
    EVAL_WORKER = "eval_worker"
    DATA_FORGE = "data_forge"
    VAULT_WORKER = "vault_worker"
    ANALYTICS = "analytics"
    STORAGE_HOT = "storage_hot"
    STORAGE_WARM = "storage_warm"
    STORAGE_COLD = "storage_cold"
    CONTROL_PLANE = "control_plane"


@dataclass
class VolumeInfo:
    """Info about a single storage volume."""
    path: str
    percent: float
    free_gb: float
    total_gb: float
    used_gb: float


@dataclass
class HostHealth:
    """Health report for a single host."""
    host: str
    status: HealthStatus
    device_id: str = ""
    name: str = ""
    roles: list = field(default_factory=list)
    resource_class: str = ""
    # Disk (primary volume)
    disk_percent: Optional[float] = None
    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    # Additional volumes (for NAS)
    volumes: list = field(default_factory=list)
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
    gpu_utilization: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    # CPU
    cpu_temp_c: Optional[float] = None
    cpu_percent: Optional[float] = None
    load_avg_1m: Optional[float] = None
    load_avg_5m: Optional[float] = None
    load_avg_15m: Optional[float] = None
    cpu_cores: Optional[int] = None
    # Storage-specific
    raid_status: Optional[str] = None
    # Meta
    last_check: Optional[str] = None
    uptime: Optional[str] = None
    services: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    maintenance_performed: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        # Convert VolumeInfo objects
        d["volumes"] = [v if isinstance(v, dict) else asdict(v) for v in self.volumes]
        return d


@dataclass
class FleetHealth:
    """Overall fleet health report."""
    timestamp: str
    overall_status: HealthStatus
    hosts: dict  # host -> HostHealth
    zones: dict = field(default_factory=dict)  # zone -> aggregate stats
    alerts: list = field(default_factory=list)
    maintenance_log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "hosts": {k: v.to_dict() for k, v in self.hosts.items()},
            "zones": self.zones,
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
    SSH_TIMEOUT = 15
    MAX_WORKERS = 8

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.devices_file = self.base_dir / "config" / "devices.json"
        self.hosts_file = self.base_dir / "config" / "hosts.json"
        self.status_file = self.base_dir / "status" / "garrison.json"
        self.devices = self._load_devices()
        self.hosts_config = self._load_hosts()
        # Determine local device
        self.local_device_id = self._detect_local_device()

    def _load_devices(self) -> dict:
        """Load device configuration from devices.json."""
        if self.devices_file.exists():
            with open(self.devices_file) as f:
                data = json.load(f)
            return data.get("devices", {})
        return {}

    def _load_hosts(self) -> dict:
        """Load host configuration (legacy, for additional info)."""
        if self.hosts_file.exists():
            with open(self.hosts_file) as f:
                data = json.load(f)
            return data.get("hosts", data)
        return {}

    def _detect_local_device(self) -> Optional[str]:
        """Detect which device we're running on."""
        import socket
        local_ips = set()
        try:
            # Get all local IPs
            hostname = socket.gethostname()
            local_ips.add(socket.gethostbyname(hostname))
            # Also check common interface
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass

        # Match against devices
        for device_id, config in self.devices.items():
            if config.get("hostname") in local_ips:
                return device_id
        return None

    def _get_ssh_user(self, device_id: str, config: dict) -> str:
        """Get SSH user for a device."""
        # Check hosts.json for SSH user override
        for host_key, host_config in self.hosts_config.items():
            if host_config.get("host") == config.get("hostname"):
                return host_config.get("ssh_user", "russ")
        # Default by device type
        if device_id == "r730xd":
            return "root"
        return "russ"

    def _ssh_command(self, host: str, command: str, timeout: int = None) -> tuple[bool, str]:
        """Execute SSH command on remote host."""
        timeout = timeout or self.SSH_TIMEOUT
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "-o", "StrictHostKeyChecking=no", host, command],
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
        success, output = self._ssh_command(host, f"df -BG {path} 2>/dev/null | tail -1")
        if not success:
            return {}

        try:
            parts = output.split()
            if len(parts) < 5:
                return {}
            # Format: Filesystem Size Used Avail Use% Mounted
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

    def _get_multiple_volumes(self, host: str, paths: list[str]) -> list[VolumeInfo]:
        """Get disk usage for multiple volumes."""
        volumes = []
        # Build command to check all paths at once
        paths_str = " ".join(paths)
        success, output = self._ssh_command(host, f"df -BG {paths_str} 2>/dev/null | tail -n +2")
        if not success:
            return volumes

        for line in output.strip().split('\n'):
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 6:
                    total_gb = float(parts[1].rstrip('G'))
                    used_gb = float(parts[2].rstrip('G'))
                    free_gb = float(parts[3].rstrip('G'))
                    percent = float(parts[4].rstrip('%'))
                    mount = parts[5]
                    volumes.append(VolumeInfo(
                        path=mount,
                        percent=percent,
                        free_gb=free_gb,
                        total_gb=total_gb,
                        used_gb=used_gb,
                    ))
            except (IndexError, ValueError):
                continue
        return volumes

    def _get_ram_usage(self, host: str) -> dict:
        """Get RAM usage for a host."""
        success, output = self._ssh_command(host, "free -g 2>/dev/null | grep Mem")
        if not success:
            return {}

        try:
            parts = output.split()
            total_gb = float(parts[1])
            used_gb = float(parts[2])
            percent = (used_gb / total_gb * 100) if total_gb > 0 else 0

            return {
                "percent": percent,
                "used_gb": used_gb,
                "total_gb": total_gb,
            }
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse RAM usage for {host}: {e}")
            return {}

    def _get_load_average(self, host: str) -> dict:
        """Get CPU load averages."""
        success, output = self._ssh_command(host, "cat /proc/loadavg 2>/dev/null")
        if not success:
            return {}

        try:
            parts = output.split()
            return {
                "load_1m": float(parts[0]),
                "load_5m": float(parts[1]),
                "load_15m": float(parts[2]),
            }
        except (IndexError, ValueError):
            return {}

    def _get_cpu_temp(self, host: str) -> Optional[float]:
        """Get CPU temperature for a host."""
        # Try sensors first (Linux with lm-sensors)
        success, output = self._ssh_command(host, "sensors 2>/dev/null | grep -i 'core 0' | head -1")
        if success and output:
            try:
                temp_str = output.split('+')[1].split('°')[0]
                return float(temp_str)
            except (IndexError, ValueError):
                pass

        # Try thermal zone (Linux generic)
        success, output = self._ssh_command(host, "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null")
        if success and output:
            try:
                return float(output) / 1000
            except ValueError:
                pass

        # Try Synology specific
        success, output = self._ssh_command(host, "cat /proc/synotemp 2>/dev/null | head -1")
        if success and output:
            try:
                # Format varies, try to extract number
                import re
                match = re.search(r'(\d+)', output)
                if match:
                    return float(match.group(1))
            except ValueError:
                pass

        return None

    def _get_gpu_info(self, host: str) -> dict:
        """Get GPU info via nvidia-smi."""
        success, output = self._ssh_command(
            host,
            "nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw "
            "--format=csv,noheader,nounits 2>/dev/null"
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

                result = {
                    "name": name,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "memory_percent": mem_percent,
                    "temp_c": temp,
                }

                # Optional fields
                if len(parts) >= 5 and parts[4] != '[N/A]':
                    try:
                        result["utilization"] = float(parts[4])
                    except ValueError:
                        pass
                if len(parts) >= 6 and parts[5] != '[N/A]':
                    try:
                        result["power_watts"] = float(parts[5])
                    except ValueError:
                        pass

                return result
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse GPU info for {host}: {e}")

        return {}

    def _get_uptime(self, host: str) -> Optional[str]:
        """Get system uptime."""
        success, output = self._ssh_command(host, "uptime -p 2>/dev/null || uptime")
        if success and output:
            if output.startswith("up "):
                return output[3:]
            return output.split("up")[-1].split(",")[0].strip() if "up" in output else output
        return None

    def _get_raid_status(self, host: str) -> Optional[str]:
        """Get RAID status (for NAS/servers with software RAID)."""
        # Try mdstat (Linux software RAID)
        success, output = self._ssh_command(host, "cat /proc/mdstat 2>/dev/null | grep -E '^md|blocks'")
        if success and output:
            lines = output.strip().split('\n')
            if lines:
                # Check for degraded state
                if 'degraded' in output.lower() or '_' in output:
                    return "DEGRADED"
                return "HEALTHY"

        # Try Synology RAID status
        success, output = self._ssh_command(host, "cat /proc/mdstat 2>/dev/null")
        if success and "md" in output:
            if '[U' in output and '_' in output:  # [UU_] means degraded
                return "DEGRADED"
            elif '[U' in output:
                return "HEALTHY"

        return None

    def _count_checkpoints(self, host: str, path: str) -> Optional[int]:
        """Count checkpoints on remote host."""
        success, output = self._ssh_command(host, f"ls -1 {path} 2>/dev/null | wc -l")
        if success:
            try:
                return int(output.strip())
            except ValueError:
                pass
        return None

    def _cleanup_checkpoints(self, host: str, path: str, keep: int = 10) -> tuple[bool, int]:
        """Remove old checkpoints, keeping the most recent ones.

        Only deletes directories matching checkpoint-* pattern.
        Base models (Qwen*, etc.) are preserved.
        """
        # Only count checkpoint-* directories, not base models
        success, output = self._ssh_command(
            host,
            f"cd {path} && ls -dt checkpoint-* 2>/dev/null | tail -n +{keep + 1} | wc -l"
        )

        if not success:
            return False, 0

        try:
            to_delete = int(output.strip())
        except ValueError:
            return False, 0

        if to_delete == 0:
            return True, 0

        # Only delete checkpoint-* directories, preserving base models
        cleanup_cmd = f"cd {path} && ls -dt checkpoint-* 2>/dev/null | tail -n +{keep + 1} | xargs -r rm -rf"
        success, _ = self._ssh_command(host, cleanup_cmd, timeout=120)

        if success:
            logger.info(f"Cleaned up {to_delete} checkpoints on {host}:{path}")
            return True, to_delete

        return False, 0

    def _check_service_http(self, host: str, port: int, path: str = "/health") -> str:
        """Check HTTP service health."""
        try:
            import requests
            url = f"http://{host}:{port}{path}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return "up"
            return "error"
        except Exception:
            return "down"

    def _check_process(self, host: str, pattern: str) -> str:
        """Check if a process is running."""
        success, output = self._ssh_command(host, f"pgrep -f '{pattern}' 2>/dev/null")
        return "up" if success and output else "down"

    # =========================================================================
    # Host Type Checkers
    # =========================================================================

    def check_local_trainer(self) -> HostHealth:
        """Check health of local training machine."""
        config = self.devices.get(self.local_device_id, {})
        name = config.get("description", "Training Server")

        health = HostHealth(
            host="localhost",
            device_id=self.local_device_id or "trainer",
            name=name,
            roles=config.get("roles", ["trainer"]),
            resource_class=config.get("resource_class", "gpu_heavy"),
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check local disk space
        try:
            result = subprocess.run(
                ["df", "-BG", str(self.base_dir)],
                capture_output=True, text=True,
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

        # Check load average
        try:
            with open('/proc/loadavg') as f:
                parts = f.read().split()
                health.load_avg_1m = float(parts[0])
                health.load_avg_5m = float(parts[1])
                health.load_avg_15m = float(parts[2])
        except Exception:
            pass

        # Get CPU cores for load context
        try:
            result = subprocess.run(["nproc"], capture_output=True, text=True)
            if result.returncode == 0:
                health.cpu_cores = int(result.stdout.strip())
        except Exception:
            pass

        # Check local GPU
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
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
                    if len(parts) >= 5 and parts[4] != '[N/A]':
                        try:
                            health.gpu_utilization = float(parts[4])
                        except ValueError:
                            pass
                    if len(parts) >= 6 and parts[5] != '[N/A]':
                        try:
                            health.gpu_power_watts = float(parts[5])
                        except ValueError:
                            pass
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
            health.services[svc_name] = self._check_service_http("localhost", port)

        # Check daemons
        daemon_checks = [
            ("hero_loop", "hero_loop"),
            ("eval_runner", "eval_runner"),
            ("garrison", "garrison"),
        ]

        for name, pattern in daemon_checks:
            try:
                result = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
                health.services[name] = "up" if result.returncode == 0 else "down"
            except:
                health.services[name] = "unknown"

        return health

    def check_gpu_host(self, device_id: str, config: dict) -> HostHealth:
        """Check health of a remote GPU host (inference server, etc.)."""
        hostname = config.get("hostname", "")
        ssh_user = self._get_ssh_user(device_id, config)

        if not hostname:
            return HostHealth(
                host=device_id,
                device_id=device_id,
                status=HealthStatus.UNKNOWN,
                issues=["No hostname configured"],
            )

        ssh_target = f"{ssh_user}@{hostname}"
        health = HostHealth(
            host=hostname,
            device_id=device_id,
            name=config.get("description", device_id),
            roles=config.get("roles", []),
            resource_class=config.get("resource_class", ""),
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check if host is reachable
        success, _ = self._ssh_command(ssh_target, "echo ok")
        if not success:
            health.status = HealthStatus.OFFLINE
            health.issues.append("Host unreachable via SSH")
            return health

        # Disk
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

        # RAM
        ram = self._get_ram_usage(ssh_target)
        if ram:
            health.ram_percent = ram.get("percent")
            health.ram_used_gb = ram.get("used_gb")
            health.ram_total_gb = ram.get("total_gb")

        # Load average
        load = self._get_load_average(ssh_target)
        if load:
            health.load_avg_1m = load.get("load_1m")
            health.load_avg_5m = load.get("load_5m")
            health.load_avg_15m = load.get("load_15m")

        # GPU
        gpu = self._get_gpu_info(ssh_target)
        if gpu:
            health.gpu_name = gpu.get("name")
            health.gpu_memory_used_mb = gpu.get("memory_used_mb")
            health.gpu_memory_total_mb = gpu.get("memory_total_mb")
            health.gpu_memory_percent = gpu.get("memory_percent")
            health.gpu_temp_c = gpu.get("temp_c")
            health.gpu_utilization = gpu.get("utilization")
            health.gpu_power_watts = gpu.get("power_watts")

        # CPU temp
        health.cpu_temp_c = self._get_cpu_temp(ssh_target)

        # Uptime
        health.uptime = self._get_uptime(ssh_target)

        # Check inference API if this is an inference server
        if "inference" in config.get("roles", []):
            host_config = self.hosts_config.get("3090", {})
            services = host_config.get("services", {})
            port = services.get("inference", {}).get("port", 8765)
            health.services["inference_api"] = self._check_service_http(hostname, port)

            if health.services["inference_api"] == "down":
                health.issues.append("Inference API unreachable")
                if health.status == HealthStatus.HEALTHY:
                    health.status = HealthStatus.WARNING

            # Count checkpoints
            models_dir = host_config.get("models_dir", "")
            if models_dir:
                count = self._count_checkpoints(ssh_target, models_dir)
                if count is not None:
                    health.services["checkpoints"] = count
                    if count > self.CHECKPOINT_MAX_COUNT + 5:
                        health.issues.append(f"Too many checkpoints: {count}")

        return health

    def check_storage_host(self, device_id: str, config: dict) -> HostHealth:
        """Check health of a storage host (NAS)."""
        hostname = config.get("hostname", "")
        ssh_user = self._get_ssh_user(device_id, config)

        if not hostname:
            return HostHealth(
                host=device_id,
                device_id=device_id,
                status=HealthStatus.UNKNOWN,
                issues=["No hostname configured"],
            )

        ssh_target = f"{ssh_user}@{hostname}"
        health = HostHealth(
            host=hostname,
            device_id=device_id,
            name=config.get("description", device_id),
            roles=config.get("roles", []),
            resource_class=config.get("resource_class", "storage"),
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check if host is reachable
        success, _ = self._ssh_command(ssh_target, "echo ok")
        if not success:
            health.status = HealthStatus.OFFLINE
            health.issues.append("Host unreachable via SSH")
            return health

        # Check multiple volumes (Synology typically has /volume1, /volume2, etc.)
        volume_paths = ["/volume1", "/volume2", "/volume3", "/volume4"]
        volumes = self._get_multiple_volumes(ssh_target, volume_paths)

        if volumes:
            health.volumes = volumes
            # Use largest volume as primary
            primary = max(volumes, key=lambda v: v.total_gb)
            health.disk_percent = primary.percent
            health.disk_free_gb = primary.free_gb
            health.disk_total_gb = primary.total_gb

            # Check if any volume is critical
            for vol in volumes:
                if vol.percent >= self.DISK_CRITICAL_PERCENT:
                    health.status = HealthStatus.CRITICAL
                    health.issues.append(f"{vol.path} critically full: {vol.percent}%")
                elif vol.percent >= self.DISK_WARNING_PERCENT:
                    if health.status == HealthStatus.HEALTHY:
                        health.status = HealthStatus.WARNING
                    health.issues.append(f"{vol.path} usage high: {vol.percent}%")
        else:
            # Fallback to root
            disk = self._get_disk_usage(ssh_target)
            if disk:
                health.disk_percent = disk.get("percent")
                health.disk_free_gb = disk.get("free_gb")
                health.disk_total_gb = disk.get("total_gb")

        # RAM
        ram = self._get_ram_usage(ssh_target)
        if ram:
            health.ram_percent = ram.get("percent")
            health.ram_used_gb = ram.get("used_gb")
            health.ram_total_gb = ram.get("total_gb")

        # CPU temp (Synology specific)
        health.cpu_temp_c = self._get_cpu_temp(ssh_target)

        # RAID status
        health.raid_status = self._get_raid_status(ssh_target)
        if health.raid_status == "DEGRADED":
            health.status = HealthStatus.CRITICAL
            health.issues.append("RAID is DEGRADED!")

        # Uptime
        health.uptime = self._get_uptime(ssh_target)

        return health

    def check_compute_worker(self, device_id: str, config: dict) -> HostHealth:
        """Check health of a compute worker (CPU-only machines like R730xd, Mac Minis)."""
        hostname = config.get("hostname", "")
        ssh_user = self._get_ssh_user(device_id, config)

        if not hostname:
            return HostHealth(
                host=device_id,
                device_id=device_id,
                status=HealthStatus.UNKNOWN,
                issues=["No hostname configured"],
            )

        ssh_target = f"{ssh_user}@{hostname}"
        health = HostHealth(
            host=hostname,
            device_id=device_id,
            name=config.get("description", device_id),
            roles=config.get("roles", []),
            resource_class=config.get("resource_class", ""),
            status=HealthStatus.HEALTHY,
            last_check=datetime.now().isoformat(),
        )

        # Check if host is reachable
        success, _ = self._ssh_command(ssh_target, "echo ok")
        if not success:
            health.status = HealthStatus.OFFLINE
            health.issues.append("Host unreachable via SSH")
            return health

        # Disk
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

        # RAM
        ram = self._get_ram_usage(ssh_target)
        if ram:
            health.ram_percent = ram.get("percent")
            health.ram_used_gb = ram.get("used_gb")
            health.ram_total_gb = ram.get("total_gb")

        # Load average (critical for CPU workers)
        load = self._get_load_average(ssh_target)
        if load:
            health.load_avg_1m = load.get("load_1m")
            health.load_avg_5m = load.get("load_5m")
            health.load_avg_15m = load.get("load_15m")

        # Get CPU cores for load context
        success, output = self._ssh_command(ssh_target, "nproc 2>/dev/null")
        if success:
            try:
                health.cpu_cores = int(output.strip())
            except ValueError:
                pass

        # CPU temp
        health.cpu_temp_c = self._get_cpu_temp(ssh_target)

        # Uptime
        health.uptime = self._get_uptime(ssh_target)

        # Check worker process
        worker_status = self._check_process(ssh_target, "claiming_worker")
        health.services["worker"] = worker_status

        return health

    def _check_device(self, device_id: str, config: dict) -> HostHealth:
        """Route to appropriate checker based on device roles."""
        # Handle both "role" (string) and "roles" (list) formats
        roles_config = config.get("roles", config.get("role", []))
        if isinstance(roles_config, str):
            roles = {roles_config}
        else:
            roles = set(roles_config)

        # Skip if disabled
        if not config.get("enabled", True):
            return HostHealth(
                host=config.get("hostname", device_id),
                device_id=device_id,
                name=config.get("description", device_id),
                status=HealthStatus.UNKNOWN,
                issues=["Device disabled"],
            )

        # Local trainer (this machine)
        if device_id == self.local_device_id:
            return self.check_local_trainer()

        # GPU hosts (inference, trainer with GPU)
        if "inference" in roles or ("trainer" in roles and config.get("gpus")):
            return self.check_gpu_host(device_id, config)

        # Storage hosts (NAS)
        if "storage" in roles or "storage_warm" in roles or "storage_cold" in roles:
            return self.check_storage_host(device_id, config)

        # Compute workers (CPU-based)
        if any(r in roles for r in ["eval_worker", "data_forge", "vault_worker", "analytics"]):
            return self.check_compute_worker(device_id, config)

        # Unknown type - try basic check
        return self.check_compute_worker(device_id, config)

    def _compute_zone_stats(self, hosts: dict) -> dict:
        """Compute aggregate stats per storage zone."""
        zones = {
            "hot": {"total_gb": 0, "used_gb": 0, "devices": []},
            "warm": {"total_gb": 0, "used_gb": 0, "devices": []},
            "cold": {"total_gb": 0, "used_gb": 0, "devices": []},
        }

        for device_id, health in hosts.items():
            config = self.devices.get(device_id, {})
            storage_zones = config.get("storage_zones", [])

            for zone in storage_zones:
                if zone in zones and health.disk_total_gb:
                    zones[zone]["total_gb"] += health.disk_total_gb
                    used = health.disk_total_gb - (health.disk_free_gb or 0)
                    zones[zone]["used_gb"] += used
                    zones[zone]["devices"].append(device_id)

        # Compute percentages
        for zone, stats in zones.items():
            if stats["total_gb"] > 0:
                stats["percent"] = (stats["used_gb"] / stats["total_gb"]) * 100
            else:
                stats["percent"] = 0

        return zones

    def perform_maintenance(self, dry_run: bool = False) -> list[str]:
        """Perform automatic maintenance tasks."""
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
        """Get complete fleet health report with parallel checking."""
        hosts = {}
        alerts = []
        maintenance_log = []

        # Check all devices in parallel
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {}
            for device_id, config in self.devices.items():
                future = executor.submit(self._check_device, device_id, config)
                futures[future] = device_id

            for future in as_completed(futures, timeout=90):
                device_id = futures[future]
                try:
                    health = future.result()
                    hosts[device_id] = health

                    # Generate alerts
                    if health.status == HealthStatus.CRITICAL:
                        alerts.append(f"CRITICAL: {health.name} ({health.host}) - {', '.join(health.issues)}")
                    elif health.status == HealthStatus.WARNING:
                        alerts.append(f"WARNING: {health.name} ({health.host}) - {', '.join(health.issues)}")
                    elif health.status == HealthStatus.OFFLINE:
                        alerts.append(f"OFFLINE: {health.name} ({health.host})")

                except Exception as e:
                    logger.error(f"Failed to check {device_id}: {e}")
                    hosts[device_id] = HostHealth(
                        host=device_id,
                        device_id=device_id,
                        status=HealthStatus.UNKNOWN,
                        issues=[f"Check failed: {str(e)}"],
                    )

        # Compute zone stats
        zones = self._compute_zone_stats(hosts)

        # Perform maintenance if requested
        if perform_maintenance:
            maintenance_log = self.perform_maintenance()

        # Determine overall status
        statuses = [h.status for h in hosts.values()]
        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.OFFLINE in statuses:
            overall = HealthStatus.WARNING
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses and len(statuses) > 1:
            overall = HealthStatus.WARNING
        else:
            overall = HealthStatus.HEALTHY

        report = FleetHealth(
            timestamp=datetime.now().isoformat(),
            overall_status=overall,
            hosts=hosts,
            zones=zones,
            alerts=alerts,
            maintenance_log=maintenance_log,
        )

        # Save status file
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report

    def run_daemon(self, interval: int = 300, maintenance_interval: int = 3600):
        """Run as daemon, checking health periodically."""
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
                    HealthStatus.OFFLINE: "○",
                }

                logger.info(f"Fleet status: {status_icon[report.overall_status]} {report.overall_status.value}")

                for device_id, health in report.hosts.items():
                    icon = status_icon[health.status]
                    disk_info = f"disk={health.disk_percent:.0f}%" if health.disk_percent else ""
                    logger.info(f"  {icon} {device_id}: {health.status.value} {disk_info}")

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
                HealthStatus.OFFLINE: "○",
            }

            print(f"\n{'='*60}")
            print(f"  GARRISON - Fleet Health Report")
            print(f"  {report.timestamp}")
            print(f"{'='*60}")
            print(f"\nOverall: {status_icon[report.overall_status]} {report.overall_status.value.upper()}")

            # Zone summary
            if report.zones:
                print(f"\nStorage Zones:")
                for zone, stats in report.zones.items():
                    if stats["total_gb"] > 0:
                        print(f"  {zone.upper()}: {stats['percent']:.1f}% ({stats['used_gb']:.0f}GB / {stats['total_gb']:.0f}GB)")

            print(f"\nHosts ({len(report.hosts)}):")
            for device_id, health in report.hosts.items():
                icon = status_icon[health.status]
                print(f"\n  {icon} {device_id} - {health.name}")
                print(f"    Host: {health.host}")
                if health.roles:
                    print(f"    Roles: {', '.join(health.roles)}")
                if health.disk_percent is not None:
                    print(f"    Disk: {health.disk_percent:.1f}% used ({health.disk_free_gb:.1f}GB free)")
                if health.ram_percent is not None:
                    print(f"    RAM: {health.ram_percent:.1f}% ({health.ram_used_gb:.0f}GB / {health.ram_total_gb:.0f}GB)")
                if health.load_avg_1m is not None:
                    cores = f"/{health.cpu_cores} cores" if health.cpu_cores else ""
                    print(f"    Load: {health.load_avg_1m:.2f} / {health.load_avg_5m:.2f} / {health.load_avg_15m:.2f}{cores}")
                if health.gpu_memory_percent is not None:
                    util = f", {health.gpu_utilization:.0f}% util" if health.gpu_utilization else ""
                    print(f"    GPU: {health.gpu_memory_percent:.1f}% memory{util}")
                if health.gpu_temp_c is not None:
                    print(f"    GPU Temp: {health.gpu_temp_c:.0f}°C")
                if health.volumes:
                    print(f"    Volumes:")
                    for vol in health.volumes:
                        v = vol if isinstance(vol, dict) else asdict(vol)
                        print(f"      {v['path']}: {v['percent']:.1f}% ({v['free_gb']:.0f}GB free)")
                if health.raid_status:
                    print(f"    RAID: {health.raid_status}")
                if health.services:
                    svc_str = ", ".join(f"{k}={v}" for k, v in health.services.items())
                    print(f"    Services: {svc_str}")
                if health.uptime:
                    print(f"    Uptime: {health.uptime}")
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
