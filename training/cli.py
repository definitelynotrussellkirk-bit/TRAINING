#!/usr/bin/env python3
"""
CLI command implementations for the training module.
"""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_base_dir() -> Path:
    """Get base directory using core.paths or fallback."""
    try:
        from core.paths import get_base_dir as _get_base_dir
        return _get_base_dir()
    except ImportError:
        # Fallback: find CLAUDE.md
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            if (parent / "CLAUDE.md").exists():
                return parent
        return Path.cwd()


def run_command(command: str, args) -> int:
    """Dispatch to the appropriate command handler."""
    handlers = {
        "play": cmd_play,
        "doctor": cmd_doctor,
        "start-all": cmd_start_all,
        "stop-all": cmd_stop_all,
        "status": cmd_status,
        "reset": cmd_reset,
    }
    handler = handlers.get(command)
    if handler:
        return handler(args)
    print(f"Unknown command: {command}")
    return 1


# =============================================================================
# PLAY COMMAND - The main game entry point
# =============================================================================

def cmd_play(args) -> int:
    """
    Enter the Realm of Training.

    This starts everything needed to "play" - services, heroes, and opens the UI.
    """
    import webbrowser
    import threading

    base_dir = get_base_dir()
    tavern_port = int(os.environ.get("TAVERN_PORT", 8888))

    print()
    print("=" * 60)
    print("     THE REALM OF TRAINING")
    print("=" * 60)
    print()

    processes: List[subprocess.Popen] = []
    hero_processes: List[subprocess.Popen] = []

    def cleanup():
        """Clean up all processes on exit."""
        print("\n\nLeaving the realm...")
        for proc in hero_processes + processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        print("Farewell, adventurer.")

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Step 1: Start core services
    print("Starting services...")

    # VaultKeeper
    vault_port = int(os.environ.get("VAULTKEEPER_PORT", 8767))
    vault_server = base_dir / "vault" / "server.py"
    if vault_server.exists():
        proc = subprocess.Popen(
            [sys.executable, str(vault_server), "--port", str(vault_port)],
            cwd=str(base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(proc)
        print(f"  [+] VaultKeeper (port {vault_port})")

    # Tavern
    tavern_server = base_dir / "tavern" / "server.py"
    if tavern_server.exists():
        proc = subprocess.Popen(
            [sys.executable, str(tavern_server), "--port", str(tavern_port)],
            cwd=str(base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(proc)
        print(f"  [+] Tavern (port {tavern_port})")

    # Wait for Tavern to be ready
    time.sleep(2)

    # Step 2: Find and awaken heroes (start hero loops)
    print("\nAwakening heroes...")

    campaigns_dir = base_dir / "campaigns"
    campaigns_to_start = []

    if hasattr(args, 'hero') and args.hero:
        # Specific hero requested
        campaign_path = campaigns_dir / args.hero
        if campaign_path.exists():
            campaigns_to_start.append(args.hero)
        else:
            print(f"  [!] Campaign not found: {args.hero}")
    else:
        # Find all active campaigns
        for hero_dir in campaigns_dir.iterdir():
            if not hero_dir.is_dir():
                continue
            for campaign_dir in hero_dir.iterdir():
                if not campaign_dir.is_dir():
                    continue
                config_file = campaign_dir / "config.json"
                if config_file.exists():
                    rel_path = f"{hero_dir.name}/{campaign_dir.name}"
                    campaigns_to_start.append(rel_path)

    for campaign in campaigns_to_start:
        campaign_path = campaigns_dir / campaign
        try:
            # Load hero name from config
            config_file = campaign_path / "config.json"
            hero_name = campaign
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    hero_name = config.get("hero_name", campaign)

            # Start hero loop
            proc = subprocess.Popen(
                [sys.executable, "-m", "arena.hero_loop", campaign],
                cwd=str(base_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            hero_processes.append(proc)
            print(f"  [+] {hero_name} awakened")
        except Exception as e:
            print(f"  [!] Failed to awaken {campaign}: {e}")

    if not hero_processes:
        print("  [!] No heroes to awaken. Create a campaign first.")

    # Step 3: Open browser
    if not (hasattr(args, 'no_browser') and args.no_browser):
        print(f"\nOpening Tavern...")
        time.sleep(1)
        webbrowser.open(f"http://localhost:{tavern_port}")

    # Step 4: Show status and wait
    print()
    print("=" * 60)
    print(f"  Tavern: http://localhost:{tavern_port}")
    print(f"  Heroes: {len(hero_processes)} active")
    print()
    print("  The heroes are training. Press Ctrl+C to leave.")
    print("=" * 60)
    print()

    # Monitor processes
    try:
        while True:
            time.sleep(5)

            # Check if any service died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"[!] Service (PID {proc.pid}) exited unexpectedly")

            # Check if any hero died
            dead_heroes = []
            for i, proc in enumerate(hero_processes):
                if proc.poll() is not None:
                    dead_heroes.append(i)

            # Clean up dead heroes
            for i in reversed(dead_heroes):
                hero_processes.pop(i)

    except KeyboardInterrupt:
        cleanup()

    return 0


# =============================================================================
# DOCTOR COMMAND
# =============================================================================

class DoctorCheck:
    """A single doctor check result."""
    def __init__(self, name: str, status: str, message: str = "", fix: str = ""):
        self.name = name
        self.status = status  # "pass", "warn", "fail"
        self.message = message
        self.fix = fix

    def __str__(self):
        icons = {"pass": "\u2705", "warn": "\u26a0\ufe0f ", "fail": "\u274c"}
        icon = icons.get(self.status, "?")
        result = f"{icon} {self.name}"
        if self.message:
            result += f": {self.message}"
        return result


def cmd_doctor(args) -> int:
    """Run diagnostic checks on the training environment."""
    print("=" * 60)
    print("REALM OF TRAINING - Doctor")
    print("=" * 60)

    checks: List[DoctorCheck] = []
    base_dir = get_base_dir()

    # 1. Base directory
    print("\n[Environment]")
    checks.append(DoctorCheck(
        "TRAINING_BASE_DIR",
        "pass",
        str(base_dir)
    ))
    print(checks[-1])

    # 2. .env file
    env_file = base_dir / ".env"
    env_example = base_dir / ".env.example"
    if env_file.exists():
        checks.append(DoctorCheck("Environment file", "pass", ".env found"))
    elif env_example.exists():
        checks.append(DoctorCheck(
            "Environment file",
            "warn",
            ".env missing, .env.example exists",
            "cp .env.example .env"
        ))
    else:
        checks.append(DoctorCheck(
            "Environment file",
            "warn",
            "No .env or .env.example"
        ))
    print(checks[-1])

    # 3. Config files
    print("\n[Configuration]")
    config_checks = [
        ("config/devices.json", "Device registry"),
        ("config/hosts.json", "Host registry"),
        ("config/storage_zones.json", "Storage zones"),
        ("config.json", "Training config"),
    ]
    for filename, desc in config_checks:
        filepath = base_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                count = ""
                if "devices" in data:
                    count = f" ({len(data['devices'])} devices)"
                elif "hosts" in data:
                    count = f" ({len(data['hosts'])} hosts)"
                elif "zones" in data:
                    count = f" ({len(data['zones'])} zones)"
                checks.append(DoctorCheck(desc, "pass", f"found{count}"))
            except json.JSONDecodeError as e:
                checks.append(DoctorCheck(desc, "fail", f"invalid JSON: {e}"))
        else:
            example = filepath.with_suffix(".example.json")
            if example.exists() or (base_dir / f"config/{filename.split('/')[-1].replace('.json', '.example.json')}").exists():
                checks.append(DoctorCheck(desc, "warn", "missing, example exists"))
            else:
                checks.append(DoctorCheck(desc, "warn", "not found"))
        print(checks[-1])

    # 4. Required directories
    print("\n[Directories]")
    required_dirs = ["models", "data", "logs", "status", "queue", "vault"]
    for dirname in required_dirs:
        dirpath = base_dir / dirname
        if dirpath.exists():
            checks.append(DoctorCheck(f"Directory: {dirname}", "pass", "exists"))
        else:
            checks.append(DoctorCheck(
                f"Directory: {dirname}",
                "warn",
                "missing",
                f"mkdir -p {dirpath}"
            ))
        print(checks[-1])

    # 5. Database files
    print("\n[Databases]")
    db_files = [
        ("vault/catalog.db", "VaultKeeper catalog"),
        ("vault/jobs.db", "Job queue"),
    ]
    for filename, desc in db_files:
        filepath = base_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            checks.append(DoctorCheck(desc, "pass", f"{size_mb:.1f} MB"))
        else:
            checks.append(DoctorCheck(desc, "warn", "not initialized"))
        print(checks[-1])

    # 6. Service connectivity
    print("\n[Services]")
    services = [
        ("VaultKeeper", "localhost", int(os.environ.get("VAULTKEEPER_PORT", 8767)), "/health"),
        ("Tavern", "localhost", int(os.environ.get("TAVERN_PORT", 8888)), "/health"),
    ]
    for name, host, port, path in services:
        status, msg = check_service(host, port, path)
        checks.append(DoctorCheck(name, status, msg))
        print(checks[-1])

    # 7. GPU availability
    print("\n[Hardware]")
    gpu_status, gpu_msg = check_gpu()
    checks.append(DoctorCheck("GPU", gpu_status, gpu_msg))
    print(checks[-1])

    # 8. Disk space
    disk_status, disk_msg = check_disk(base_dir)
    checks.append(DoctorCheck("Disk space", disk_status, disk_msg))
    print(checks[-1])

    # 9. Hardcode check (optional)
    if args.check_hardcodes:
        print("\n[Hardcode Audit]")
        hardcode_status, hardcode_msg = check_hardcodes(base_dir)
        checks.append(DoctorCheck("Hardcoded paths", hardcode_status, hardcode_msg))
        print(checks[-1])

    # Summary
    print("\n" + "=" * 60)
    passes = sum(1 for c in checks if c.status == "pass")
    warns = sum(1 for c in checks if c.status == "warn")
    fails = sum(1 for c in checks if c.status == "fail")

    print(f"Doctor finished: {passes} passed, {warns} warnings, {fails} failures")

    if fails > 0:
        print("\nCritical issues need attention before proceeding.")
        return 1
    elif warns > 0:
        print("\nWarnings found but system should work. Run with --fix to auto-resolve.")
        return 0
    else:
        print("\nAll checks passed! Run 'python -m training start-all' to begin.")
        return 0


def check_service(host: str, port: int, path: str) -> Tuple[str, str]:
    """Check if a service is reachable."""
    try:
        import urllib.request
        url = f"http://{host}:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return "pass", f"http://{host}:{port} responding"
    except Exception as e:
        pass
    return "warn", f"not reachable on port {port}"


def check_gpu() -> Tuple[str, str]:
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            return "pass", f"{count} GPU(s): {name}"
        else:
            return "warn", "CUDA not available"
    except ImportError:
        return "warn", "PyTorch not installed"
    except Exception as e:
        return "warn", str(e)


def check_disk(base_dir: Path) -> Tuple[str, str]:
    """Check disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(base_dir)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        pct = (used / total) * 100
        if free_gb < 20:
            return "warn", f"{free_gb:.1f} GB free ({pct:.0f}% used)"
        return "pass", f"{free_gb:.1f} GB free of {total_gb:.0f} GB"
    except Exception as e:
        return "warn", str(e)


def check_hardcodes(base_dir: Path) -> Tuple[str, str]:
    """Check for hardcoded paths."""
    import re
    patterns = [
        re.compile(r"/home/[^/\s\"']+/"),
        re.compile(r"C:\\Users\\[^\\]+\\"),
    ]
    bad_files = []

    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".sh", ".json", ".yaml", ".yml"}:
            continue
        if "archive" in str(path) or ".git" in str(path):
            continue
        try:
            text = path.read_text(errors="ignore")
            for pat in patterns:
                if pat.search(text):
                    bad_files.append(path.relative_to(base_dir))
                    break
        except Exception:
            pass

    if bad_files:
        return "warn", f"{len(bad_files)} files with hardcoded paths"
    return "pass", "no hardcoded user paths found"


# =============================================================================
# START-ALL COMMAND
# =============================================================================

def cmd_start_all(args) -> int:
    """Start all services for development."""
    base_dir = get_base_dir()

    print("=" * 60)
    print("REALM OF TRAINING - Starting Services")
    print("=" * 60)

    # Check if Weaver exists and use it
    weaver_path = base_dir / "weaver" / "weaver.py"
    start_all_script = base_dir / "scripts" / "start_all.sh"

    if start_all_script.exists():
        print("\nUsing scripts/start_all.sh (The Weaver)...")
        result = subprocess.run(
            ["bash", str(start_all_script)],
            cwd=str(base_dir)
        )
        return result.returncode

    # Fallback: start services manually
    print("\nStarting services manually...")

    processes: List[subprocess.Popen] = []

    # Get ports from environment
    vault_port = int(os.environ.get("VAULTKEEPER_PORT", 8767))
    tavern_port = int(os.environ.get("TAVERN_PORT", 8888))

    # Start VaultKeeper
    vault_server = base_dir / "vault" / "server.py"
    if vault_server.exists():
        print(f"Starting VaultKeeper on port {vault_port}...")
        proc = subprocess.Popen(
            [sys.executable, str(vault_server), "--port", str(vault_port)],
            cwd=str(base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(proc)
        time.sleep(1)

    # Start Tavern
    tavern_server = base_dir / "tavern" / "server.py"
    if tavern_server.exists():
        print(f"Starting Tavern on port {tavern_port}...")
        proc = subprocess.Popen(
            [sys.executable, str(tavern_server), "--port", str(tavern_port)],
            cwd=str(base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(proc)
        time.sleep(1)

    if not processes:
        print("No services to start. Check your installation.")
        return 1

    print("\n" + "=" * 60)
    print("Services started!")
    print(f"  Tavern UI: http://localhost:{tavern_port}")
    print(f"  VaultKeeper: http://localhost:{vault_port}")
    print("\nPress Ctrl+C to stop all services...")
    print("=" * 60)

    # Wait for Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping services...")
        for proc in processes:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            time.sleep(1)
            # Check if any process died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"Process {proc.pid} exited unexpectedly")
    except KeyboardInterrupt:
        signal_handler(None, None)

    return 0


# =============================================================================
# STOP-ALL COMMAND
# =============================================================================

def cmd_stop_all(args) -> int:
    """Stop all services."""
    base_dir = get_base_dir()

    print("=" * 60)
    print("REALM OF TRAINING - Stopping Services")
    print("=" * 60)

    # Use stop_all.sh if available
    stop_all_script = base_dir / "scripts" / "stop_all.sh"
    if stop_all_script.exists():
        print("\nUsing scripts/stop_all.sh...")
        result = subprocess.run(
            ["bash", str(stop_all_script)],
            cwd=str(base_dir)
        )
        return result.returncode

    # Fallback: kill by PID files
    print("\nStopping services via PID files...")
    pids_dir = base_dir / ".pids"

    if not pids_dir.exists():
        print("No .pids directory found")
        return 0

    for pid_file in pids_dir.glob("*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            print(f"Stopping {pid_file.stem} (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink()
        except (ValueError, ProcessLookupError, FileNotFoundError):
            pass

    print("Services stopped.")
    return 0


# =============================================================================
# STATUS COMMAND
# =============================================================================

def cmd_status(args) -> int:
    """Show status of all services."""
    base_dir = get_base_dir()

    print("=" * 60)
    print("REALM OF TRAINING - Status")
    print("=" * 60)

    # Check services
    print("\n[Services]")
    services = [
        ("VaultKeeper", "localhost", int(os.environ.get("VAULTKEEPER_PORT", 8767)), "/health"),
        ("Tavern", "localhost", int(os.environ.get("TAVERN_PORT", 8888)), "/health"),
    ]

    for name, host, port, path in services:
        status, msg = check_service(host, port, path)
        icon = "\u2705" if status == "pass" else "\u274c"
        print(f"  {icon} {name}: {msg}")

    # Check PID files
    print("\n[Daemons]")
    pids_dir = base_dir / ".pids"
    if pids_dir.exists():
        for pid_file in sorted(pids_dir.glob("*.pid")):
            try:
                pid = int(pid_file.read_text().strip())
                # Check if process is running
                os.kill(pid, 0)
                print(f"  \u2705 {pid_file.stem}: running (PID {pid})")
            except (ValueError, ProcessLookupError):
                print(f"  \u274c {pid_file.stem}: stale PID file")
            except PermissionError:
                print(f"  \u2705 {pid_file.stem}: running (PID {pid})")
    else:
        print("  No PID files found")

    # Training status
    print("\n[Training]")
    status_file = base_dir / "status" / "training_status.json"
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)
            state = status.get("state", "unknown")
            step = status.get("global_step", 0)
            print(f"  State: {state}")
            print(f"  Step: {step:,}")
        except Exception as e:
            print(f"  Unable to read: {e}")
    else:
        print("  No training status file")

    return 0


# =============================================================================
# RESET COMMAND - Clear stale state while preserving models
# =============================================================================

def cmd_reset(args) -> int:
    """
    Reset training environment.

    Clears runtime state that may be stale or corrupted:
    - .pids/ (daemon PID files)
    - control/state.json (training state)
    - status/training_status.json (last training status)
    - status/events.jsonl (event log)

    Preserves:
    - campaigns/ (all hero data and checkpoints)
    - models/ (all model files)
    - vault/jobs.db (unless --clear-jobs)
    - config.json (root configuration)
    """
    base_dir = get_base_dir()

    print("=" * 60)
    print("REALM OF TRAINING - Environment Reset")
    print("=" * 60)
    print()
    print("This will clear runtime state while preserving models and campaigns.")
    print()
    print("Will clear:")
    print("  - .pids/ (daemon PID files)")
    print("  - control/state.json (training state)")
    print("  - status/training_status.json (last status)")
    print("  - status/events.jsonl (event log)")
    if not getattr(args, 'keep_jobs', False):
        print("  - Pending/running jobs in vault/jobs.db")
    print()
    print("Will preserve:")
    print("  - campaigns/ (all hero data and checkpoints)")
    print("  - models/ (all model files)")
    print("  - control/active_campaign.json (campaign selection)")
    if getattr(args, 'keep_jobs', False):
        print("  - vault/jobs.db (job database)")
    print()

    # Confirm unless --yes
    if not getattr(args, 'yes', False):
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return 0

    # Step 1: Stop all daemons
    print("\n[1/4] Stopping daemons...")
    stopped = 0
    pids_dir = base_dir / ".pids"
    if pids_dir.exists():
        for pid_file in pids_dir.glob("*.pid"):
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                print(f"  Stopped {pid_file.stem} (PID {pid})")
                stopped += 1
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    print(f"  {stopped} daemon(s) stopped")

    # Step 2: Clear PID files
    print("\n[2/4] Clearing PID files...")
    cleared = 0
    if pids_dir.exists():
        for pid_file in pids_dir.glob("*.pid"):
            try:
                pid_file.unlink()
                cleared += 1
            except Exception:
                pass
    print(f"  {cleared} PID file(s) cleared")

    # Step 3: Clear state files
    print("\n[3/4] Clearing state files...")
    state_files = [
        base_dir / "control" / "state.json",
        base_dir / "status" / "training_status.json",
        base_dir / "status" / "events.jsonl",
    ]
    cleared_state = 0
    for state_file in state_files:
        if state_file.exists():
            try:
                state_file.unlink()
                print(f"  Cleared {state_file.name}")
                cleared_state += 1
            except Exception as e:
                print(f"  Failed to clear {state_file.name}: {e}")
    print(f"  {cleared_state} state file(s) cleared")

    # Step 4: Clear pending jobs (unless --keep-jobs)
    if not getattr(args, 'keep_jobs', False):
        print("\n[4/4] Clearing pending jobs...")
        try:
            from jobs.store import get_store
            store = get_store()
            # Cancel all pending/claimed/running jobs
            from guild.job_types import JobStatus
            jobs = store.list_jobs(status=JobStatus.PENDING, limit=1000)
            jobs += store.list_jobs(status=JobStatus.CLAIMED, limit=1000)
            jobs += store.list_jobs(status=JobStatus.RUNNING, limit=1000)
            cancelled = 0
            for job in jobs:
                try:
                    store.cancel(job.job_id, actor="reset")
                    cancelled += 1
                except Exception:
                    pass
            print(f"  {cancelled} job(s) cancelled")
        except ImportError:
            print("  Job store not available (skipped)")
        except Exception as e:
            print(f"  Error clearing jobs: {e}")
    else:
        print("\n[4/4] Keeping jobs (--keep-jobs)")

    print()
    print("=" * 60)
    print("Reset complete!")
    print()
    print("Next steps:")
    print("  1. python -m training play    # Start fresh")
    print("  2. python -m training status  # Verify clean state")
    print("=" * 60)

    return 0
