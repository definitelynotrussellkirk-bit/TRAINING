"""
Ritual of the Scribe - Evaluation system diagnostics.

This ritual checks the health of evaluation infrastructure:
- Eval runner daemon status
- SSH connectivity to inference host
- Remote checkpoint directory access
- Eval queue status
- Evaluation ledger health
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("scribe", "Ritual of the Scribe", "Evaluation system diagnostics")
def run() -> List[RitualCheckResult]:
    """Execute all scribe ritual checks."""
    results = []
    results.append(_check_eval_runner_daemon())
    results.append(_check_ssh_connectivity())
    results.append(_check_remote_checkpoint_dir())
    results.append(_check_eval_queue())
    results.append(_check_eval_ledger())
    results.append(_check_eval_data_flow())
    return results


def _get_base_dir() -> Path:
    """Get base directory."""
    try:
        from core.paths import get_base_dir
        return get_base_dir()
    except:
        return Path(__file__).parent.parent.parent


def _get_inference_host() -> dict:
    """Get inference host info from hosts.json."""
    try:
        hosts_file = _get_base_dir() / "config" / "hosts.json"
        if hosts_file.exists():
            with open(hosts_file) as f:
                hosts = json.load(f)
            return hosts.get("hosts", {}).get("3090", {})
    except:
        pass
    return {"host": "localhost", "ssh_user": os.environ.get("USER", "user")}


def _check_eval_runner_daemon() -> RitualCheckResult:
    """Check if eval runner daemon is running."""
    start = datetime.utcnow()
    base_dir = _get_base_dir()
    pid_file = base_dir / ".pids" / "eval_runner.pid"

    details = {"pid_file": str(pid_file)}

    try:
        if not pid_file.exists():
            return RitualCheckResult(
                id="eval_runner_daemon",
                name="Eval Runner Daemon",
                description="Check if eval runner daemon is running",
                status="warn",
                category="eval",
                details={**details, "status": "no_pid_file"},
                remediation="Start eval runner: python3 core/eval_runner.py --daemon",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        with open(pid_file) as f:
            pid = int(f.read().strip())

        details["pid"] = pid

        # Check if process is running
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            process_name = result.stdout.strip()
            details["process"] = process_name

            # Check recent log activity
            log_file = base_dir / "logs" / "eval_runner.log"
            if log_file.exists():
                stat = log_file.stat()
                age_seconds = (datetime.now().timestamp() - stat.st_mtime)
                details["log_age_seconds"] = int(age_seconds)
                details["log_size_kb"] = int(stat.st_size / 1024)

            return RitualCheckResult(
                id="eval_runner_daemon",
                name="Eval Runner Daemon",
                description="Check if eval runner daemon is running",
                status="ok",
                category="eval",
                details=details,
                started_at=start,
                finished_at=datetime.utcnow(),
            )
        else:
            return RitualCheckResult(
                id="eval_runner_daemon",
                name="Eval Runner Daemon",
                description="Check if eval runner daemon is running",
                status="fail",
                category="eval",
                details={**details, "status": "process_not_found"},
                remediation=f"PID {pid} not running. Restart with: python3 core/eval_runner.py --daemon",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

    except Exception as e:
        return RitualCheckResult(
            id="eval_runner_daemon",
            name="Eval Runner Daemon",
            description="Check if eval runner daemon is running",
            status="fail",
            category="eval",
            details={**details, "error": str(e)},
            remediation="Check .pids/eval_runner.pid and process status",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_ssh_connectivity() -> RitualCheckResult:
    """Check SSH connectivity to inference host."""
    start = datetime.utcnow()
    host_info = _get_inference_host()
    host = host_info.get("host", "localhost")
    user = host_info.get("ssh_user", os.environ.get("USER", "user"))
    ssh_target = f"{user}@{host}" if user else host

    details = {"host": host, "user": user}

    try:
        # Quick SSH check with timeout
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_target, "echo ok"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and "ok" in result.stdout:
            details["response"] = "ok"
            return RitualCheckResult(
                id="ssh_connectivity",
                name="SSH to Inference Host",
                description="Check SSH connectivity to inference server",
                status="ok",
                category="network",
                details=details,
                started_at=start,
                finished_at=datetime.utcnow(),
            )
        else:
            details["returncode"] = result.returncode
            details["stderr"] = result.stderr[:200] if result.stderr else None
            return RitualCheckResult(
                id="ssh_connectivity",
                name="SSH to Inference Host",
                description="Check SSH connectivity to inference server",
                status="fail",
                category="network",
                details=details,
                remediation=f"Check SSH key setup and network to {host}",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

    except subprocess.TimeoutExpired:
        return RitualCheckResult(
            id="ssh_connectivity",
            name="SSH to Inference Host",
            description="Check SSH connectivity to inference server",
            status="fail",
            category="network",
            details={**details, "error": "timeout"},
            remediation=f"SSH connection to {host} timed out. Check network/firewall.",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="ssh_connectivity",
            name="SSH to Inference Host",
            description="Check SSH connectivity to inference server",
            status="fail",
            category="network",
            details={**details, "error": str(e)},
            remediation="Check SSH client installation and configuration",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_remote_checkpoint_dir() -> RitualCheckResult:
    """Check access to remote checkpoint directory."""
    start = datetime.utcnow()
    host_info = _get_inference_host()
    host = host_info.get("host", "localhost")
    user = host_info.get("ssh_user", os.environ.get("USER", "user"))
    models_dir = host_info.get("models_dir", "~/llm/models")
    ssh_target = f"{user}@{host}" if user else host

    details = {"host": host, "models_dir": models_dir}

    try:
        # Check if directory exists and list checkpoint count
        result = subprocess.run(
            [
                "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_target,
                f"ls -d {models_dir}/checkpoint-* 2>/dev/null | wc -l"
            ],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0:
            checkpoint_count = int(result.stdout.strip() or "0")
            details["checkpoint_count"] = checkpoint_count

            # Get disk space
            df_result = subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_target,
                    f"df -h {models_dir} | tail -1 | awk '{{print $4}}'"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if df_result.returncode == 0:
                details["free_space"] = df_result.stdout.strip()

            return RitualCheckResult(
                id="remote_checkpoint_dir",
                name="Remote Checkpoint Directory",
                description="Check access to checkpoint storage on inference host",
                status="ok" if checkpoint_count >= 0 else "warn",
                category="storage",
                details=details,
                started_at=start,
                finished_at=datetime.utcnow(),
            )
        else:
            details["stderr"] = result.stderr[:200] if result.stderr else None
            return RitualCheckResult(
                id="remote_checkpoint_dir",
                name="Remote Checkpoint Directory",
                description="Check access to checkpoint storage on inference host",
                status="fail",
                category="storage",
                details=details,
                remediation=f"Check if {models_dir} exists on {host}",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

    except subprocess.TimeoutExpired:
        return RitualCheckResult(
            id="remote_checkpoint_dir",
            name="Remote Checkpoint Directory",
            description="Check access to checkpoint storage on inference host",
            status="fail",
            category="storage",
            details={**details, "error": "timeout"},
            remediation="SSH command timed out. Check network/storage.",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="remote_checkpoint_dir",
            name="Remote Checkpoint Directory",
            description="Check access to checkpoint storage on inference host",
            status="fail",
            category="storage",
            details={**details, "error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_eval_queue() -> RitualCheckResult:
    """Check evaluation queue status."""
    start = datetime.utcnow()
    base_dir = _get_base_dir()

    details = {}

    try:
        from core.evaluation_ledger import get_pending_evaluations
        from core.passives import get_pending_passives

        skill_pending = get_pending_evaluations()
        passive_pending = get_pending_passives()

        details["skill_queue"] = len(skill_pending)
        details["passive_queue"] = len(passive_pending)
        details["total_pending"] = len(skill_pending) + len(passive_pending)

        # Show next few items
        if skill_pending:
            next_items = []
            for entry in skill_pending[:3]:
                level_str = f"L{entry.level}" if entry.level else "ALL"
                next_items.append(f"checkpoint-{entry.checkpoint_step} {entry.skill} {level_str}")
            details["next_skill_evals"] = next_items

        # Determine status
        status = "ok"
        if details["total_pending"] > 50:
            status = "warn"  # Large backlog

        return RitualCheckResult(
            id="eval_queue",
            name="Evaluation Queue",
            description="Check pending evaluations in queue",
            status=status,
            category="eval",
            details=details,
            remediation="Large queue backlog. Ensure eval_runner is processing." if status == "warn" else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )

    except Exception as e:
        return RitualCheckResult(
            id="eval_queue",
            name="Evaluation Queue",
            description="Check pending evaluations in queue",
            status="fail",
            category="eval",
            details={"error": str(e)},
            remediation="Check evaluation_ledger module and status files",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_eval_ledger() -> RitualCheckResult:
    """Check evaluation ledger health."""
    start = datetime.utcnow()
    base_dir = _get_base_dir()

    details = {}

    try:
        from core.evaluation_ledger import get_eval_ledger

        ledger = get_eval_ledger(base_dir)
        summary = ledger.summary()

        details["total_evaluations"] = summary.get("total_evaluations", 0)
        details["by_skill"] = {}

        for skill, info in summary.get("by_skill", {}).items():
            details["by_skill"][skill] = {
                "count": info.get("count", 0),
                "best_accuracy": f"{info.get('best_accuracy', 0):.0%}",
                "best_checkpoint": info.get("best_checkpoint"),
            }

        # Check if we have any evaluations
        status = "ok"
        if details["total_evaluations"] == 0:
            status = "warn"

        return RitualCheckResult(
            id="eval_ledger",
            name="Evaluation Ledger",
            description="Check evaluation ledger health and records",
            status=status,
            category="eval",
            details=details,
            remediation="No evaluations recorded. Run: python3 core/eval_runner.py --once" if status == "warn" else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )

    except Exception as e:
        return RitualCheckResult(
            id="eval_ledger",
            name="Evaluation Ledger",
            description="Check evaluation ledger health and records",
            status="fail",
            category="eval",
            details={"error": str(e)},
            remediation="Check evaluation_ledger.py and status/eval_ledger.json",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_eval_data_flow() -> RitualCheckResult:
    """
    Check end-to-end evaluation data flow.

    Validates that:
    1. Passive evals in ledger are visible in Tavern API
    2. Skill pages show eval counts when passives exist
    3. accuracy_history is populated when eval data exists

    This diagnostic would have caught the "0 evals" bug where passives_ledger.json
    had 2600+ results but the UI showed 0 evals.
    """
    start = datetime.utcnow()
    base_dir = _get_base_dir()
    details = {}
    issues = []

    try:
        # Step 1: Read passives_ledger.json directly
        passives_file = base_dir / "status" / "passives_ledger.json"
        ledger_count = 0
        if passives_file.exists():
            with open(passives_file) as f:
                ledger = json.load(f)
                ledger_count = ledger.get("count", len(ledger.get("results", [])))
        details["passives_ledger_count"] = ledger_count

        # Step 2: Check curriculum_state.json
        curriculum_file = base_dir / "data_manager" / "curriculum_state.json"
        curriculum_evals = 0
        if curriculum_file.exists():
            with open(curriculum_file) as f:
                curriculum = json.load(f)
                # Count total evals across all skills
                for skill_id, skill_data in curriculum.items():
                    if isinstance(skill_data, dict):
                        evals = skill_data.get("total_evaluations", 0)
                        curriculum_evals += evals
        details["curriculum_state_evals"] = curriculum_evals

        # Step 3: Check Tavern API response (if running)
        api_total = 0
        api_passives = 0
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                "http://localhost:8888/api/skills",
                headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                api_data = json.loads(resp.read().decode())
                api_total = api_data.get("total_passive_evals", 0)
                api_passives = api_data.get("passive_count", 0)
                details["api_total_passive_evals"] = api_total
                details["api_passive_count"] = api_passives

                # Check individual skills
                skill_issues = []
                for skill in api_data.get("skills", []):
                    skill_id = skill.get("id", "unknown")
                    eval_count = skill.get("eval_count", 0)
                    from_passives = skill.get("from_passives", False)

                    # If skill has no evals but passives exist, flag it
                    if eval_count == 0 and ledger_count > 0:
                        skill_issues.append(f"{skill_id}: 0 evals but passives exist")

                    details[f"skill_{skill_id}_evals"] = eval_count
                    details[f"skill_{skill_id}_from_passives"] = from_passives

                if skill_issues:
                    issues.extend(skill_issues)

        except urllib.error.URLError:
            details["api_status"] = "tavern_not_running"
        except Exception as e:
            details["api_error"] = str(e)

        # Step 4: Check for data flow issues

        # Issue: Passives exist in ledger but API returns 0
        if ledger_count > 0 and api_total == 0 and "api_status" not in details:
            issues.append(f"DATA_FLOW_BROKEN: {ledger_count} passives in ledger but API returns 0")

        # Issue: curriculum_state is empty but passives exist
        if curriculum_evals == 0 and ledger_count > 0:
            # This is OK now since we fallback to passives, but note it
            details["fallback_to_passives"] = True

        # Issue: API running but returning much less than ledger
        if api_total > 0 and ledger_count > 0:
            ratio = api_total / ledger_count
            if ratio < 0.5:  # Less than 50% surfacing
                issues.append(f"DATA_LOSS: Only {ratio:.0%} of ledger passives in API ({api_total}/{ledger_count})")

        # Determine status
        if issues:
            status = "fail"
            remediation = "Eval data not flowing to UI. Check:\n" + "\n".join([f"  - {i}" for i in issues])
        elif ledger_count == 0:
            status = "warn"
            remediation = "No passive evaluations recorded yet. Run some evals."
        else:
            status = "ok"
            remediation = None

        details["issues"] = issues if issues else []

        return RitualCheckResult(
            id="eval_data_flow",
            name="Evaluation Data Flow",
            description="Check end-to-end flow from ledger to UI",
            status=status,
            category="eval",
            details=details,
            remediation=remediation,
            started_at=start,
            finished_at=datetime.utcnow(),
        )

    except Exception as e:
        return RitualCheckResult(
            id="eval_data_flow",
            name="Evaluation Data Flow",
            description="Check end-to-end flow from ledger to UI",
            status="fail",
            category="eval",
            details={"error": str(e)},
            remediation="Check passives_ledger.json and tavern/api/skills.py",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
