"""
Guardian - The watchdog sentinel protecting the training daemon.

The Guardian watches over the training daemon, ensuring it stays alive
and responsive. If the daemon falls, the Guardian raises it again.

RPG Flavor:
    The Guardian is an eternal watcher at the Arena gates. When the
    training daemon falls in battle (crashes), the Guardian revives
    it and sends it back into the fight.

Capabilities:
    - Monitor daemon health
    - Detect crashes and hangs
    - Auto-restart on failure
    - Track uptime and incidents

This module wraps safety/daemon_watchdog.py with RPG-themed naming.
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sentinels.types import (
    SentinelState,
    ThreatLevel,
    HealthDomain,
    HealthReport,
    AlertRecord,
)


class Guardian:
    """
    The Guardian - watchdog for the training daemon.

    Monitors daemon health and auto-restarts on failure.

    Usage:
        guardian = Guardian(base_dir)

        # Check daemon health
        report = guardian.check_vitals()
        if not report.is_healthy:
            guardian.revive_daemon()

        # Run continuous patrol
        guardian.stand_watch(check_interval=60)
    """

    def __init__(
        self,
        base_dir: str | Path = "/path/to/training",
        daemon_script: str = "core/training_daemon.py",
    ):
        """
        Initialize the Guardian.

        Args:
            base_dir: Base training directory
            daemon_script: Path to daemon script relative to base_dir
        """
        self.base_dir = Path(base_dir)
        self.daemon_script = daemon_script
        self.pid_file = self.base_dir / ".pids" / "training_daemon.pid"
        self.log_file = self.base_dir / "logs" / "training_output.log"

        self.state = SentinelState.DORMANT
        self._incidents: list[AlertRecord] = []

    def check_vitals(self) -> HealthReport:
        """
        Check if the training daemon is alive and healthy.

        Returns:
            HealthReport with daemon status
        """
        checks_passed = 0
        checks_failed = 0
        details = {}

        # Check 1: PID file exists
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                details["pid"] = pid

                # Check 2: Process is running
                result = subprocess.run(
                    ["ps", "-p", str(pid)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    checks_passed += 1
                    details["process_running"] = True
                else:
                    checks_failed += 1
                    details["process_running"] = False
            except (ValueError, FileNotFoundError):
                checks_failed += 1
                details["pid_valid"] = False
        else:
            checks_failed += 1
            details["pid_file_exists"] = False

        # Check 3: Recent log activity
        if self.log_file.exists():
            mtime = datetime.fromtimestamp(self.log_file.stat().st_mtime)
            age_seconds = (datetime.now() - mtime).total_seconds()
            details["log_age_seconds"] = age_seconds

            if age_seconds < 300:  # 5 minutes
                checks_passed += 1
            else:
                checks_failed += 1
                details["log_stale"] = True

        # Determine status
        if checks_failed == 0:
            status = ThreatLevel.CLEAR
            message = "Daemon is healthy and responsive"
        elif checks_passed > checks_failed:
            status = ThreatLevel.MINOR
            message = "Daemon running but some concerns"
        else:
            status = ThreatLevel.SEVERE
            message = "Daemon appears to be down or unresponsive"

        return HealthReport(
            domain=HealthDomain.DAEMON,
            status=status,
            message=message,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details=details,
            checked_at=datetime.now(),
        )

    def is_daemon_alive(self) -> bool:
        """Quick check if daemon is running."""
        report = self.check_vitals()
        return report.is_healthy

    def revive_daemon(self, reason: str = "Guardian revival") -> bool:
        """
        Revive (restart) the training daemon.

        Args:
            reason: Why revival was needed

        Returns:
            True if successfully started
        """
        # Record incident
        incident = AlertRecord(
            alert_id=f"revival_{int(time.time())}",
            threat_level=ThreatLevel.SEVERE,
            domain=HealthDomain.DAEMON,
            title="Daemon Revival",
            description=reason,
            auto_response=True,
            response_action="restart_daemon",
            detected_at=datetime.now(),
        )
        self._incidents.append(incident)

        try:
            # Start daemon
            cmd = [
                "nohup", "python3",
                str(self.base_dir / self.daemon_script),
                "--base-dir", str(self.base_dir)
            ]

            with open(self.log_file, "a") as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.base_dir),
                    start_new_session=True,
                )

            # Write PID
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.pid_file.write_text(str(process.pid))

            # Mark resolved
            incident.resolved = True
            incident.resolved_at = datetime.now()

            return True

        except Exception as e:
            incident.response_action = f"failed: {e}"
            return False

    def stand_watch(
        self,
        check_interval: int = 60,
        auto_revive: bool = True,
    ):
        """
        Stand continuous watch over the daemon.

        Args:
            check_interval: Seconds between checks
            auto_revive: Automatically restart if down
        """
        self.state = SentinelState.PATROLLING

        while self.state == SentinelState.PATROLLING:
            report = self.check_vitals()

            if not report.is_healthy:
                self.state = SentinelState.ALERTED

                if auto_revive:
                    self.state = SentinelState.RESPONDING
                    self.revive_daemon("Auto-revival: daemon not healthy")
                    self.state = SentinelState.PATROLLING

            time.sleep(check_interval)

    def get_incident_log(self) -> list[Dict[str, Any]]:
        """Get log of all incidents."""
        return [i.to_dict() for i in self._incidents]

    def get_status(self) -> Dict[str, Any]:
        """Get Guardian status."""
        return {
            "state": self.state.value,
            "daemon_healthy": self.is_daemon_alive(),
            "incidents_count": len(self._incidents),
            "last_check": datetime.now().isoformat(),
        }


def get_guardian(base_dir: str | Path = "/path/to/training") -> Guardian:
    """Get a Guardian instance."""
    return Guardian(base_dir)
