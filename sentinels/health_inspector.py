"""
Health Inspector - Comprehensive system health assessment.

The Health Inspector performs thorough patrols of all system domains,
checking GPU, disk, queue, daemon, and configuration health.

RPG Flavor:
    The Health Inspector is a meticulous examiner who checks every
    aspect of the training realm's wellbeing. Their patrol reports
    are comprehensive and actionable.

Domains Checked:
    - Daemon: Training daemon running and responsive
    - GPU: VRAM available, temperature safe
    - Disk: Free space sufficient
    - Queue: Training queue not stuck
    - Config: Configuration valid

This module wraps safety/comprehensive_health_check.py.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentinels.types import (
    ThreatLevel,
    HealthDomain,
    HealthReport,
    PatrolReport,
)


class HealthInspector:
    """
    Comprehensive health inspector for the training system.

    Performs full patrol of all domains.

    Usage:
        inspector = HealthInspector(base_dir)

        # Full patrol
        report = inspector.full_patrol()
        if not report.is_all_clear:
            for threat in report.threats_detected:
                print(f"[{threat.status.value}] {threat.domain.value}: {threat.message}")

        # Check specific domain
        gpu_health = inspector.inspect_gpu()
    """

    def __init__(self, base_dir: str | Path = None):
        """Initialize the Health Inspector."""
        if base_dir is None:
            try:
                from core.paths import get_base_dir
                self.base_dir = get_base_dir()
            except ImportError:
                self.base_dir = Path(__file__).parent.parent  # Fallback
        else:
            self.base_dir = Path(base_dir)

        # Thresholds
        self.disk_warning_gb = 50
        self.disk_critical_gb = 20
        self.gpu_temp_warning = 80
        self.gpu_temp_critical = 90

    def full_patrol(self) -> PatrolReport:
        """
        Perform a full patrol of all domains.

        Returns:
            PatrolReport with all domain assessments
        """
        start_time = datetime.now()

        reports = [
            self.inspect_daemon(),
            self.inspect_disk(),
            self.inspect_queue(),
            self.inspect_config(),
        ]

        # Try GPU if available
        try:
            reports.append(self.inspect_gpu())
        except Exception:
            pass  # GPU check optional

        # Calculate overall status
        worst_status = ThreatLevel.CLEAR
        for report in reports:
            if report.status.value > worst_status.value:
                worst_status = report.status

        duration = (datetime.now() - start_time).total_seconds() * 1000

        return PatrolReport(
            overall_status=worst_status,
            domain_reports=reports,
            patrol_time=start_time,
            patrol_duration_ms=duration,
        )

    def inspect_daemon(self) -> HealthReport:
        """Inspect training daemon health."""
        pid_file = self.base_dir / ".pids" / "training_daemon.pid"
        checks_passed = 0
        checks_failed = 0
        details = {}

        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True
                )
                if result.returncode == 0:
                    checks_passed += 1
                    details["daemon_running"] = True
                else:
                    checks_failed += 1
                    details["daemon_running"] = False
            except Exception:
                checks_failed += 1
        else:
            details["pid_file_missing"] = True

        if checks_failed > 0:
            return HealthReport(
                domain=HealthDomain.DAEMON,
                status=ThreatLevel.SEVERE,
                message="Training daemon not running",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                details=details,
                checked_at=datetime.now(),
            )

        return HealthReport(
            domain=HealthDomain.DAEMON,
            status=ThreatLevel.CLEAR,
            message="Daemon healthy",
            checks_passed=checks_passed,
            details=details,
            checked_at=datetime.now(),
        )

    def inspect_disk(self) -> HealthReport:
        """Inspect disk space."""
        usage = shutil.disk_usage(self.base_dir)
        free_gb = usage.free / (1024 ** 3)
        used_pct = (usage.used / usage.total) * 100

        details = {
            "free_gb": round(free_gb, 2),
            "used_percent": round(used_pct, 1),
        }

        if free_gb < self.disk_critical_gb:
            return HealthReport(
                domain=HealthDomain.DISK,
                status=ThreatLevel.CRITICAL,
                message=f"Critical: Only {free_gb:.1f}GB free",
                checks_failed=1,
                details=details,
                checked_at=datetime.now(),
            )
        elif free_gb < self.disk_warning_gb:
            return HealthReport(
                domain=HealthDomain.DISK,
                status=ThreatLevel.MODERATE,
                message=f"Low disk: {free_gb:.1f}GB free",
                checks_passed=1,
                details=details,
                checked_at=datetime.now(),
            )

        return HealthReport(
            domain=HealthDomain.DISK,
            status=ThreatLevel.CLEAR,
            message=f"Disk healthy: {free_gb:.1f}GB free",
            checks_passed=1,
            details=details,
            checked_at=datetime.now(),
        )

    def inspect_gpu(self) -> HealthReport:
        """Inspect GPU health."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return HealthReport(
                    domain=HealthDomain.GPU,
                    status=ThreatLevel.MINOR,
                    message="Could not query GPU",
                    checked_at=datetime.now(),
                )

            parts = result.stdout.strip().split(",")
            temp = int(parts[0].strip())
            mem_used = int(parts[1].strip())
            mem_total = int(parts[2].strip())
            mem_pct = (mem_used / mem_total) * 100

            details = {
                "temperature_c": temp,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "memory_percent": round(mem_pct, 1),
            }

            if temp >= self.gpu_temp_critical:
                return HealthReport(
                    domain=HealthDomain.GPU,
                    status=ThreatLevel.CRITICAL,
                    message=f"GPU overheating: {temp}C",
                    checks_failed=1,
                    details=details,
                    checked_at=datetime.now(),
                )
            elif temp >= self.gpu_temp_warning:
                return HealthReport(
                    domain=HealthDomain.GPU,
                    status=ThreatLevel.MODERATE,
                    message=f"GPU hot: {temp}C",
                    details=details,
                    checked_at=datetime.now(),
                )

            return HealthReport(
                domain=HealthDomain.GPU,
                status=ThreatLevel.CLEAR,
                message=f"GPU healthy: {temp}C, {mem_pct:.0f}% VRAM",
                checks_passed=1,
                details=details,
                checked_at=datetime.now(),
            )

        except FileNotFoundError:
            return HealthReport(
                domain=HealthDomain.GPU,
                status=ThreatLevel.MINOR,
                message="nvidia-smi not found",
                checked_at=datetime.now(),
            )

    def inspect_queue(self) -> HealthReport:
        """Inspect training queue health."""
        queue_dir = self.base_dir / "queue"
        details = {}

        # Count files in each queue
        for priority in ["high", "normal", "low"]:
            subdir = queue_dir / priority
            if subdir.exists():
                details[priority] = len(list(subdir.glob("*.jsonl")))

        # Check processing directory
        processing = queue_dir / "processing"
        if processing.exists():
            stuck_files = list(processing.glob("*.jsonl"))
            details["processing"] = len(stuck_files)

            # Check if files are old (stuck)
            for f in stuck_files:
                age_hours = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).total_seconds() / 3600
                if age_hours > 2:
                    return HealthReport(
                        domain=HealthDomain.QUEUE,
                        status=ThreatLevel.MODERATE,
                        message=f"File stuck in processing for {age_hours:.1f}h",
                        details=details,
                        checked_at=datetime.now(),
                    )

        return HealthReport(
            domain=HealthDomain.QUEUE,
            status=ThreatLevel.CLEAR,
            message="Queue healthy",
            checks_passed=1,
            details=details,
            checked_at=datetime.now(),
        )

    def inspect_config(self) -> HealthReport:
        """Inspect configuration validity."""
        config_file = self.base_dir / "config.json"

        if not config_file.exists():
            return HealthReport(
                domain=HealthDomain.CONFIG,
                status=ThreatLevel.SEVERE,
                message="config.json missing",
                checks_failed=1,
                checked_at=datetime.now(),
            )

        try:
            import json
            with open(config_file) as f:
                config = json.load(f)

            # Basic validation
            required = ["model_name", "hyperparams"]
            missing = [k for k in required if k not in config]

            if missing:
                return HealthReport(
                    domain=HealthDomain.CONFIG,
                    status=ThreatLevel.MODERATE,
                    message=f"Config missing: {missing}",
                    details={"missing_keys": missing},
                    checked_at=datetime.now(),
                )

            return HealthReport(
                domain=HealthDomain.CONFIG,
                status=ThreatLevel.CLEAR,
                message="Config valid",
                checks_passed=1,
                checked_at=datetime.now(),
            )

        except json.JSONDecodeError as e:
            return HealthReport(
                domain=HealthDomain.CONFIG,
                status=ThreatLevel.SEVERE,
                message=f"Config parse error: {e}",
                checks_failed=1,
                checked_at=datetime.now(),
            )

    def quick_check(self) -> Dict[str, Any]:
        """Quick health check returning simple status."""
        report = self.full_patrol()
        return {
            "healthy": report.is_all_clear,
            "status": report.overall_status.value,
            "threats": len(report.threats_detected),
            "checked_at": datetime.now().isoformat(),
        }


def get_health_inspector(base_dir: str | Path = None) -> HealthInspector:
    """Get a HealthInspector instance."""
    return HealthInspector(base_dir)
