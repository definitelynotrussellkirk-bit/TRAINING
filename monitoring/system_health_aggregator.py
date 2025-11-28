#!/usr/bin/env python3
"""
System Health Aggregator - Unified health check across all machines and daemons

Provides a single status/system_health.json that shows whether all expected
processes are running across both machines (4090 and 3090).

Usage:
    # One-shot check
    python3 system_health_aggregator.py

    # Continuous mode (updates every 60s)
    python3 system_health_aggregator.py --continuous --interval 60

    # Check specific machine
    python3 system_health_aggregator.py --machine 4090
"""

import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import argparse
import time
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.paths import get_base_dir, get_status_dir, REMOTE_HOST
except ImportError:
    def get_base_dir():
        return Path(__file__).parent.parent
    def get_status_dir():
        return get_base_dir() / "status"
    REMOTE_HOST = "192.168.x.x"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessStatus:
    """Status of a single expected process"""
    name: str
    pattern: str
    machine: str
    running: bool
    pid: Optional[int] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    uptime: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MachineHealth:
    """Health summary for a machine"""
    machine: str
    hostname: str
    reachable: bool
    processes_expected: int
    processes_running: int
    processes_missing: List[str]
    gpu_available: bool
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health"""
    timestamp: str
    overall_status: str  # healthy, degraded, critical
    machines: Dict[str, MachineHealth]
    processes: List[ProcessStatus]
    summary: Dict[str, Any]


# Expected processes by machine
EXPECTED_PROCESSES = {
    "4090": [
        {"name": "training_daemon", "pattern": "training_daemon.py", "critical": True},
        {"name": "deployment_orchestrator", "pattern": "deployment_orchestrator.py", "critical": False},
    ],
    "3090": [
        # Note: inference server runs as main.py in /home/user/llm
        {"name": "inference_server", "pattern": "python3 main.py", "critical": True},
        {"name": "gpu_task_scheduler", "pattern": "gpu_task_scheduler.py", "critical": False},
        {"name": "self_correction_loop", "pattern": "self_correction_loop.py", "critical": False},
        {"name": "automated_testing_daemon", "pattern": "automated_testing_daemon.py", "critical": False},
        {"name": "curriculum_eval_loop", "pattern": "curriculum_eval_loop.py", "critical": False},
    ]
}


class SystemHealthAggregator:
    """
    Aggregates health information from all machines and daemons.

    Checks:
    - Process existence via ps/pgrep
    - GPU availability via nvidia-smi
    - Network reachability via ping
    """

    def __init__(self, base_dir: Optional[Path] = None, remote_host: str = None):
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.status_dir = self.base_dir / "status"
        self.status_file = self.status_dir / "system_health.json"
        self.remote_host = remote_host or REMOTE_HOST

        # Ensure status dir exists
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def check_local_process(self, pattern: str) -> Optional[Dict]:
        """Check if a process matching pattern is running locally"""
        try:
            # Use pgrep to find matching processes
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return None

            pids = result.stdout.strip().split('\n')
            if not pids or not pids[0]:
                return None

            pid = int(pids[0])

            # Get process info via ps
            ps_result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "pid,rss,%cpu,etime", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if ps_result.returncode == 0 and ps_result.stdout.strip():
                parts = ps_result.stdout.strip().split()
                if len(parts) >= 4:
                    return {
                        "pid": pid,
                        "memory_mb": int(parts[1]) / 1024,
                        "cpu_percent": float(parts[2]),
                        "uptime": parts[3]
                    }

            return {"pid": pid}

        except Exception as e:
            logger.debug(f"Error checking process {pattern}: {e}")
            return None

    def check_remote_process(self, pattern: str) -> Optional[Dict]:
        """Check if a process matching pattern is running on remote machine"""
        try:
            cmd = f'ssh -o ConnectTimeout=5 {self.remote_host} "pgrep -f \'{pattern}\'"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            pids = result.stdout.strip().split('\n')
            if not pids or not pids[0]:
                return None

            pid = int(pids[0])

            # Get process info
            ps_cmd = f'ssh -o ConnectTimeout=5 {self.remote_host} "ps -p {pid} -o pid,rss,%cpu,etime --no-headers"'
            ps_result = subprocess.run(
                ps_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            if ps_result.returncode == 0 and ps_result.stdout.strip():
                parts = ps_result.stdout.strip().split()
                if len(parts) >= 4:
                    return {
                        "pid": pid,
                        "memory_mb": int(parts[1]) / 1024,
                        "cpu_percent": float(parts[2]),
                        "uptime": parts[3]
                    }

            return {"pid": pid}

        except Exception as e:
            logger.debug(f"Error checking remote process {pattern}: {e}")
            return None

    def check_machine_reachable(self, machine: str) -> bool:
        """Check if a machine is reachable"""
        if machine == "4090":
            return True  # Local machine

        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", self.remote_host],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def check_local_gpu(self) -> Dict:
        """Check local GPU status"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    return {
                        "available": True,
                        "utilization": float(parts[0].strip()),
                        "memory_used_gb": float(parts[1].strip()) / 1024,
                        "memory_total_gb": float(parts[2].strip()) / 1024
                    }

            return {"available": False, "error": "nvidia-smi failed"}

        except Exception as e:
            return {"available": False, "error": str(e)}

    def check_remote_gpu(self) -> Dict:
        """Check remote GPU status"""
        try:
            cmd = f'ssh -o ConnectTimeout=5 {self.remote_host} "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    return {
                        "available": True,
                        "utilization": float(parts[0].strip()),
                        "memory_used_gb": float(parts[1].strip()) / 1024,
                        "memory_total_gb": float(parts[2].strip()) / 1024
                    }

            return {"available": False, "error": "nvidia-smi failed"}

        except Exception as e:
            return {"available": False, "error": str(e)}

    def collect_health(self) -> SystemHealth:
        """Collect health information from all sources"""
        processes = []
        machines = {}

        # Check 4090 (local)
        logger.info("Checking 4090 (local)...")
        local_gpu = self.check_local_gpu()
        local_missing = []
        local_running = 0

        for proc_info in EXPECTED_PROCESSES["4090"]:
            proc_status = self.check_local_process(proc_info["pattern"])

            if proc_status:
                local_running += 1
                processes.append(ProcessStatus(
                    name=proc_info["name"],
                    pattern=proc_info["pattern"],
                    machine="4090",
                    running=True,
                    pid=proc_status.get("pid"),
                    memory_mb=proc_status.get("memory_mb"),
                    cpu_percent=proc_status.get("cpu_percent"),
                    uptime=proc_status.get("uptime")
                ))
            else:
                local_missing.append(proc_info["name"])
                processes.append(ProcessStatus(
                    name=proc_info["name"],
                    pattern=proc_info["pattern"],
                    machine="4090",
                    running=False
                ))

        machines["4090"] = MachineHealth(
            machine="4090",
            hostname="localhost",
            reachable=True,
            processes_expected=len(EXPECTED_PROCESSES["4090"]),
            processes_running=local_running,
            processes_missing=local_missing,
            gpu_available=local_gpu.get("available", False),
            gpu_utilization=local_gpu.get("utilization"),
            gpu_memory_used_gb=local_gpu.get("memory_used_gb")
        )

        # Check 3090 (remote)
        logger.info(f"Checking 3090 ({self.remote_host})...")
        remote_reachable = self.check_machine_reachable("3090")

        if remote_reachable:
            remote_gpu = self.check_remote_gpu()
            remote_missing = []
            remote_running = 0

            for proc_info in EXPECTED_PROCESSES["3090"]:
                proc_status = self.check_remote_process(proc_info["pattern"])

                if proc_status:
                    remote_running += 1
                    processes.append(ProcessStatus(
                        name=proc_info["name"],
                        pattern=proc_info["pattern"],
                        machine="3090",
                        running=True,
                        pid=proc_status.get("pid"),
                        memory_mb=proc_status.get("memory_mb"),
                        cpu_percent=proc_status.get("cpu_percent"),
                        uptime=proc_status.get("uptime")
                    ))
                else:
                    remote_missing.append(proc_info["name"])
                    processes.append(ProcessStatus(
                        name=proc_info["name"],
                        pattern=proc_info["pattern"],
                        machine="3090",
                        running=False
                    ))

            machines["3090"] = MachineHealth(
                machine="3090",
                hostname=self.remote_host,
                reachable=True,
                processes_expected=len(EXPECTED_PROCESSES["3090"]),
                processes_running=remote_running,
                processes_missing=remote_missing,
                gpu_available=remote_gpu.get("available", False),
                gpu_utilization=remote_gpu.get("utilization"),
                gpu_memory_used_gb=remote_gpu.get("memory_used_gb")
            )
        else:
            # Remote not reachable
            for proc_info in EXPECTED_PROCESSES["3090"]:
                processes.append(ProcessStatus(
                    name=proc_info["name"],
                    pattern=proc_info["pattern"],
                    machine="3090",
                    running=False,
                    error="Machine unreachable"
                ))

            machines["3090"] = MachineHealth(
                machine="3090",
                hostname=self.remote_host,
                reachable=False,
                processes_expected=len(EXPECTED_PROCESSES["3090"]),
                processes_running=0,
                processes_missing=[p["name"] for p in EXPECTED_PROCESSES["3090"]],
                gpu_available=False,
                error="Machine unreachable"
            )

        # Determine overall status
        total_expected = sum(len(EXPECTED_PROCESSES[m]) for m in EXPECTED_PROCESSES)
        total_running = sum(1 for p in processes if p.running)

        # Check critical processes
        critical_missing = []
        for machine, procs in EXPECTED_PROCESSES.items():
            for proc in procs:
                if proc.get("critical", False):
                    matching = [p for p in processes if p.name == proc["name"] and p.machine == machine]
                    if matching and not matching[0].running:
                        critical_missing.append(f"{machine}:{proc['name']}")

        if critical_missing:
            overall_status = "critical"
        elif total_running < total_expected:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Build summary
        summary = {
            "total_expected": total_expected,
            "total_running": total_running,
            "total_missing": total_expected - total_running,
            "critical_missing": critical_missing,
            "machines_reachable": sum(1 for m in machines.values() if m.reachable),
            "machines_total": len(machines),
            "gpus_available": sum(1 for m in machines.values() if m.gpu_available)
        }

        return SystemHealth(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            machines={k: asdict(v) for k, v in machines.items()},
            processes=[asdict(p) for p in processes],
            summary=summary
        )

    def save_health(self, health: SystemHealth):
        """Save health data to status file"""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": health.timestamp,
            "overall_status": health.overall_status,
            "machines": health.machines,
            "processes": health.processes,
            "summary": health.summary
        }

        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Health status saved to {self.status_file}")

    def print_status(self, health: SystemHealth):
        """Print human-readable status"""
        status_emoji = {
            "healthy": "\u2705",
            "degraded": "\u26a0\ufe0f",
            "critical": "\u274c"
        }

        print("\n" + "="*70)
        print(f"SYSTEM HEALTH: {status_emoji.get(health.overall_status, '?')} {health.overall_status.upper()}")
        print("="*70)
        print(f"Timestamp: {health.timestamp}")
        print(f"Processes: {health.summary['total_running']}/{health.summary['total_expected']} running")

        if health.summary.get('critical_missing'):
            print(f"\nCRITICAL MISSING: {', '.join(health.summary['critical_missing'])}")

        print("\n" + "-"*70)
        print("MACHINES:")
        print("-"*70)

        for name, machine in health.machines.items():
            status = "\u2705" if machine['reachable'] else "\u274c"
            gpu_status = "\u2705" if machine['gpu_available'] else "\u274c"

            print(f"\n{name} ({machine['hostname']}):")
            print(f"  Reachable: {status}")
            print(f"  Processes: {machine['processes_running']}/{machine['processes_expected']}")
            if machine['processes_missing']:
                print(f"  Missing: {', '.join(machine['processes_missing'])}")
            print(f"  GPU: {gpu_status}", end="")
            if machine['gpu_utilization'] is not None:
                print(f" ({machine['gpu_utilization']:.0f}% util, {machine['gpu_memory_used_gb']:.1f}GB used)")
            else:
                print()

        print("\n" + "-"*70)
        print("PROCESSES:")
        print("-"*70)

        for proc in health.processes:
            status = "\u2705" if proc['running'] else "\u274c"
            print(f"  {status} {proc['machine']}:{proc['name']}", end="")
            if proc['running'] and proc.get('pid'):
                info_parts = []
                if proc.get('memory_mb'):
                    info_parts.append(f"{proc['memory_mb']:.0f}MB")
                if proc.get('uptime'):
                    info_parts.append(f"up {proc['uptime']}")
                if info_parts:
                    print(f" (PID {proc['pid']}, {', '.join(info_parts)})")
                else:
                    print(f" (PID {proc['pid']})")
            elif proc.get('error'):
                print(f" - {proc['error']}")
            else:
                print()

        print("\n" + "="*70)

    def run_continuous(self, interval: int = 60):
        """Run health checks continuously"""
        logger.info(f"Starting continuous health checks (interval: {interval}s)")

        while True:
            try:
                health = self.collect_health()
                self.save_health(health)
                self.print_status(health)
            except Exception as e:
                logger.error(f"Health check failed: {e}")

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="System Health Aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--base-dir', help='Base directory (auto-detected if not specified)')
    parser.add_argument('--remote-host', default=REMOTE_HOST, help=f'Remote 3090 host (default: {REMOTE_HOST})')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')
    parser.add_argument('--quiet', action='store_true', help='Only output JSON')
    parser.add_argument('--machine', choices=['4090', '3090'], help='Check only specific machine')

    args = parser.parse_args()

    aggregator = SystemHealthAggregator(
        base_dir=args.base_dir,
        remote_host=args.remote_host
    )

    if args.continuous:
        aggregator.run_continuous(args.interval)
    else:
        health = aggregator.collect_health()
        aggregator.save_health(health)

        if args.quiet:
            print(json.dumps({
                "timestamp": health.timestamp,
                "overall_status": health.overall_status,
                "machines": health.machines,
                "processes": health.processes,
                "summary": health.summary
            }, indent=2))
        else:
            aggregator.print_status(health)


if __name__ == "__main__":
    main()
