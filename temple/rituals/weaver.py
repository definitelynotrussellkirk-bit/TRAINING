"""
Ritual of the Weaver - Daemon and process health diagnostics.

This ritual checks the health of background processes:
- Weaver orchestrator status
- Hero loop status
- PID file validity
- Process health via PID files
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("weaver", "Ritual of the Weaver", "Daemon and process health diagnostics")
def run() -> List[RitualCheckResult]:
    """Execute all weaver ritual checks."""
    results = []
    results.append(_check_weaver_status())
    results.append(_check_pid_files())
    results.append(_check_hero_loop())
    results.append(_check_eval_runner())
    return results


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_weaver_status() -> RitualCheckResult:
    """Check Weaver orchestrator status."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        pid_file = get_base_dir() / ".pids" / "weaver.pid"

        if not pid_file.exists():
            return RitualCheckResult(
                id="weaver_status",
                name="Weaver Orchestrator",
                description="Check if Weaver daemon is running",
                status="warn",
                category="daemon",
                details={"error": "No weaver.pid file found"},
                remediation="Start Weaver: python3 -m training start-all",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        pid = int(pid_file.read_text().strip())
        running = _is_process_running(pid)

        return RitualCheckResult(
            id="weaver_status",
            name="Weaver Orchestrator",
            description="Check if Weaver daemon is running",
            status="ok" if running else "fail",
            category="daemon",
            details={
                "pid": pid,
                "running": running,
                "pid_file": str(pid_file),
            },
            remediation="Start Weaver: python3 -m training start-all" if not running else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="weaver_status",
            name="Weaver Orchestrator",
            description="Check if Weaver daemon is running",
            status="fail",
            category="daemon",
            details={"error": str(e)},
            remediation="Check logs: tail -f logs/weaver.log",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_pid_files() -> RitualCheckResult:
    """Check all PID files for validity."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        pid_dir = get_base_dir() / ".pids"

        if not pid_dir.exists():
            return RitualCheckResult(
                id="pid_files",
                name="PID Files Health",
                description="Validate all PID files point to running processes",
                status="warn",
                category="daemon",
                details={"error": "No .pids directory found"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        pid_files = list(pid_dir.glob("*.pid"))
        results = {}
        alive_count = 0
        stale_count = 0

        for pf in pid_files:
            name = pf.stem
            try:
                pid = int(pf.read_text().strip())
                running = _is_process_running(pid)
                results[name] = {
                    "pid": pid,
                    "running": running,
                }
                if running:
                    alive_count += 1
                else:
                    stale_count += 1
            except:
                results[name] = {"error": "Invalid PID file"}
                stale_count += 1

        # Status based on stale PIDs
        if stale_count > 2:
            status = "fail"
        elif stale_count > 0:
            status = "warn"
        else:
            status = "ok"

        return RitualCheckResult(
            id="pid_files",
            name="PID Files Health",
            description="Validate all PID files point to running processes",
            status=status,
            category="daemon",
            details={
                "total": len(pid_files),
                "alive": alive_count,
                "stale": stale_count,
                "files": results,
            },
            remediation="Clean stale PIDs: rm .pids/*.pid && python3 -m training start-all" if stale_count > 0 else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="pid_files",
            name="PID Files Health",
            description="Validate all PID files point to running processes",
            status="fail",
            category="daemon",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_hero_loop() -> RitualCheckResult:
    """Check hero loop status."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        pid_file = get_base_dir() / ".pids" / "hero_loop.pid"

        if not pid_file.exists():
            return RitualCheckResult(
                id="hero_loop",
                name="Hero Loop",
                description="Check if hero loop is running",
                status="warn",
                category="daemon",
                details={"error": "No hero_loop.pid file"},
                remediation="Hero loop starts automatically with Weaver",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        pid = int(pid_file.read_text().strip())
        running = _is_process_running(pid)

        # Also check if actively training
        status_file = get_base_dir() / "status" / "training_status.json"
        training_active = False
        if status_file.exists():
            import json
            try:
                with open(status_file) as f:
                    data = json.load(f)
                    training_active = data.get("status") == "training"
            except:
                pass

        return RitualCheckResult(
            id="hero_loop",
            name="Hero Loop",
            description="Check if hero loop is running",
            status="ok" if running else "fail",
            category="daemon",
            details={
                "pid": pid,
                "running": running,
                "training_active": training_active,
            },
            remediation="Restart: python3 -m training start-all" if not running else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="hero_loop",
            name="Hero Loop",
            description="Check if hero loop is running",
            status="fail",
            category="daemon",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_eval_runner() -> RitualCheckResult:
    """Check eval runner daemon status."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        pid_file = get_base_dir() / ".pids" / "eval_runner.pid"

        if not pid_file.exists():
            return RitualCheckResult(
                id="eval_runner",
                name="Eval Runner",
                description="Check if evaluation runner is running",
                status="warn",
                category="daemon",
                details={"error": "No eval_runner.pid file"},
                remediation="Start eval runner: python3 core/eval_runner.py --daemon",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        pid = int(pid_file.read_text().strip())
        running = _is_process_running(pid)

        return RitualCheckResult(
            id="eval_runner",
            name="Eval Runner",
            description="Check if evaluation runner is running",
            status="ok" if running else "warn",
            category="daemon",
            details={
                "pid": pid,
                "running": running,
            },
            remediation="Start: python3 core/eval_runner.py --daemon" if not running else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="eval_runner",
            name="Eval Runner",
            description="Check if evaluation runner is running",
            status="warn",
            category="daemon",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
