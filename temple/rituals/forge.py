"""
Ritual of the Forge - GPU, hardware, and resource diagnostics.

This ritual checks the health of the training infrastructure:
- GPU VRAM usage and availability
- GPU temperature
- Disk space on training directories
- System memory
"""

import os
import subprocess
from datetime import datetime
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("forge", "Ritual of the Forge", "GPU, hardware, and resource diagnostics")
def run() -> List[RitualCheckResult]:
    """Execute all forge ritual checks."""
    results = []
    results.append(_check_gpu_vram())
    results.append(_check_gpu_temperature())
    results.append(_check_disk_space())
    results.append(_check_system_memory())
    return results


def _check_gpu_vram() -> RitualCheckResult:
    """Check GPU VRAM usage."""
    start = datetime.utcnow()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise Exception(f"nvidia-smi failed: {result.stderr}")

        lines = result.stdout.strip().split('\n')
        gpus = []
        total_used = 0
        total_available = 0

        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                used = int(parts[0])
                total = int(parts[1])
                name = parts[2]
                pct = (used / total * 100) if total > 0 else 0
                gpus.append({
                    "gpu": i,
                    "name": name,
                    "used_mb": used,
                    "total_mb": total,
                    "used_pct": round(pct, 1),
                })
                total_used += used
                total_available += total

        overall_pct = (total_used / total_available * 100) if total_available > 0 else 0

        # Status based on usage
        if overall_pct > 95:
            status = "fail"
        elif overall_pct > 85:
            status = "warn"
        else:
            status = "ok"

        return RitualCheckResult(
            id="gpu_vram",
            name="GPU VRAM Usage",
            description="Check GPU memory usage across all GPUs",
            status=status,
            category="hardware",
            details={
                "gpus": gpus,
                "total_used_mb": total_used,
                "total_available_mb": total_available,
                "overall_pct": round(overall_pct, 1),
            },
            remediation="Free GPU memory by stopping unused processes or reducing batch size",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except FileNotFoundError:
        return RitualCheckResult(
            id="gpu_vram",
            name="GPU VRAM Usage",
            description="Check GPU memory usage across all GPUs",
            status="skip",
            category="hardware",
            details={"error": "nvidia-smi not found - no NVIDIA GPU?"},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="gpu_vram",
            name="GPU VRAM Usage",
            description="Check GPU memory usage across all GPUs",
            status="fail",
            category="hardware",
            details={"error": str(e)},
            remediation="Ensure NVIDIA drivers are installed and GPU is accessible",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_gpu_temperature() -> RitualCheckResult:
    """Check GPU temperature."""
    start = datetime.utcnow()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise Exception(f"nvidia-smi failed: {result.stderr}")

        lines = result.stdout.strip().split('\n')
        gpus = []
        max_temp = 0

        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                temp = int(parts[0])
                name = parts[1]
                gpus.append({
                    "gpu": i,
                    "name": name,
                    "temp_c": temp,
                })
                max_temp = max(max_temp, temp)

        # Status based on temperature
        if max_temp > 85:
            status = "fail"
        elif max_temp > 75:
            status = "warn"
        else:
            status = "ok"

        return RitualCheckResult(
            id="gpu_temperature",
            name="GPU Temperature",
            description="Check GPU temperature is within safe range",
            status=status,
            category="hardware",
            details={
                "gpus": gpus,
                "max_temp_c": max_temp,
            },
            remediation="Improve cooling or reduce GPU load. Check fans and airflow.",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except FileNotFoundError:
        return RitualCheckResult(
            id="gpu_temperature",
            name="GPU Temperature",
            description="Check GPU temperature is within safe range",
            status="skip",
            category="hardware",
            details={"error": "nvidia-smi not found"},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="gpu_temperature",
            name="GPU Temperature",
            description="Check GPU temperature is within safe range",
            status="warn",
            category="hardware",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_disk_space() -> RitualCheckResult:
    """Check disk space on training directories."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        base_dir = get_base_dir()

        # Get disk stats for base directory
        stat = os.statvfs(base_dir)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb
        used_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0

        # Check key directories
        dirs_info = {}
        for dirname in ["models", "campaigns", "queue", "data"]:
            dir_path = base_dir / dirname
            if dir_path.exists():
                # Get directory size (quick estimate)
                try:
                    result = subprocess.run(
                        ["du", "-sh", str(dir_path)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        size = result.stdout.split()[0]
                        dirs_info[dirname] = size
                except:
                    dirs_info[dirname] = "?"

        # Status based on free space
        if free_gb < 10:
            status = "fail"
        elif free_gb < 50:
            status = "warn"
        else:
            status = "ok"

        return RitualCheckResult(
            id="disk_space",
            name="Disk Space",
            description="Check available disk space for training",
            status=status,
            category="storage",
            details={
                "path": str(base_dir),
                "total_gb": round(total_gb, 1),
                "free_gb": round(free_gb, 1),
                "used_pct": round(used_pct, 1),
                "directories": dirs_info,
            },
            remediation="Free disk space by removing old checkpoints: /vault → delete unused",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="disk_space",
            name="Disk Space",
            description="Check available disk space for training",
            status="fail",
            category="storage",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_system_memory() -> RitualCheckResult:
    """Check system RAM usage."""
    start = datetime.utcnow()
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = int(val)

        total_kb = meminfo.get('MemTotal', 0)
        free_kb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
        total_gb = total_kb / (1024**2)
        free_gb = free_kb / (1024**2)
        used_gb = total_gb - free_gb
        used_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0

        # Status based on usage
        if used_pct > 95:
            status = "fail"
        elif used_pct > 85:
            status = "warn"
        else:
            status = "ok"

        return RitualCheckResult(
            id="system_memory",
            name="System Memory",
            description="Check system RAM usage",
            status=status,
            category="hardware",
            details={
                "total_gb": round(total_gb, 1),
                "free_gb": round(free_gb, 1),
                "used_pct": round(used_pct, 1),
            },
            remediation="Free memory by stopping unused processes or adding swap",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="system_memory",
            name="System Memory",
            description="Check system RAM usage",
            status="warn",
            category="hardware",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
