#!/usr/bin/env python3
"""
THE SUMMONER - Skill Trainer Server Manager

Summons skill trainers (API servers) into the realm. Reads skill configs
and automatically starts, monitors, and restarts their API servers.

Usage:
    python guild/summoner.py --status          # Check all skill server status
    python guild/summoner.py --summon          # Start all configured servers
    python guild/summoner.py --summon bin sy   # Start specific skills
    python guild/summoner.py --dismiss         # Stop all servers
    python guild/summoner.py --dismiss bin     # Stop specific skill
    python guild/summoner.py --daemon          # Run as persistent daemon
    python guild/summoner.py --pull-evals      # Pull eval sets from running servers
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv

# Load environment
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Get singleSKILL path from env or auto-detect
def _get_skill_servers_path() -> str:
    """Get path to singleSKILL servers."""
    env_path = os.getenv("SKILL_SERVERS_PATH") or os.getenv("SINGLESKILL_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    try:
        from core.paths import get_external_tool_path
        return str(get_external_tool_path("singleSKILL"))
    except Exception:
        # Last resort: try sibling directory
        sibling = BASE_DIR.parent / "singleSKILL"
        if sibling.exists():
            return str(sibling)
        raise RuntimeError(
            "singleSKILL not found. Set SKILL_SERVERS_PATH or SINGLESKILL_PATH env var."
        )

SKILL_SERVERS_PATH = _get_skill_servers_path()
SKILLS_CONFIG_DIR = BASE_DIR / "configs" / "skills"
PIDS_DIR = BASE_DIR / ".pids"
STATUS_DIR = BASE_DIR / "status"
VALIDATION_DIR = BASE_DIR / "data" / "validation"

# Ensure directories exist
PIDS_DIR.mkdir(exist_ok=True)
STATUS_DIR.mkdir(exist_ok=True)


@dataclass
class SkillServerConfig:
    """Configuration for a skill's API server."""
    skill_id: str
    name: str
    port: int
    url: str
    source_dir: str
    working_dir: str
    start_command: str
    health_endpoint: str
    startup_timeout: int
    max_level: int
    eval_cache_dir: str


def load_skill_configs() -> Dict[str, SkillServerConfig]:
    """Load all skill configurations from YAML files."""
    configs = {}

    for yaml_file in SKILLS_CONFIG_DIR.glob("*.yaml"):
        if yaml_file.name.startswith("_"):
            continue  # Skip template

        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            if not data or "api" not in data:
                continue

            api = data["api"]
            skill_id = data.get("id", yaml_file.stem)

            # Resolve working directory
            working_dir = api.get("working_dir", SKILL_SERVERS_PATH)
            if "${SKILL_SERVERS_PATH}" in working_dir:
                working_dir = working_dir.replace("${SKILL_SERVERS_PATH}", SKILL_SERVERS_PATH)

            # Resolve source directory (can be relative or absolute)
            source_dir = api.get("source_dir", "")
            if not source_dir.startswith("/"):
                source_dir = os.path.join(SKILL_SERVERS_PATH, source_dir)

            # Get eval cache location
            eval_config = data.get("eval", {})
            eval_cache = eval_config.get("local_cache", f"data/validation/{skill_id}/")

            configs[skill_id] = SkillServerConfig(
                skill_id=skill_id,
                name=data.get("name", skill_id),
                port=api.get("port", 8080),
                url=api.get("url", f"http://localhost:{api.get('port', 8080)}"),
                source_dir=source_dir,
                working_dir=working_dir,
                start_command=api.get("start_command", ""),
                health_endpoint=api.get("health_endpoint", "/health"),
                startup_timeout=api.get("startup_timeout", 10),
                max_level=data.get("max_level", 10),
                eval_cache_dir=str(BASE_DIR / eval_cache),
            )
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file.name}: {e}")

    return configs


def check_server_health(config: SkillServerConfig) -> bool:
    """Check if a skill server is healthy."""
    try:
        url = f"{config.url}{config.health_endpoint}"
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False


def get_server_pid(skill_id: str) -> Optional[int]:
    """Get the PID of a running skill server."""
    pid_file = PIDS_DIR / f"skill_{skill_id}.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)
    return None


def summon_server(config: SkillServerConfig) -> bool:
    """Start a skill server."""
    # Check if already running
    if check_server_health(config):
        print(f"  {config.skill_id}: Already running on port {config.port}")
        return True

    if not config.start_command:
        print(f"  {config.skill_id}: No start_command configured")
        return False

    # Ensure working directory exists
    if not os.path.isdir(config.working_dir):
        print(f"  {config.skill_id}: Working directory not found: {config.working_dir}")
        return False

    print(f"  {config.skill_id}: Summoning on port {config.port}...")

    # Start the server
    log_file = BASE_DIR / "logs" / f"skill_{config.skill_id}.log"
    log_file.parent.mkdir(exist_ok=True)

    with open(log_file, "a") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Command: {config.start_command}\n")
        log.write(f"Working dir: {config.working_dir}\n")
        log.write(f"{'='*60}\n\n")

        process = subprocess.Popen(
            config.start_command,
            shell=True,
            cwd=config.working_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Save PID
    pid_file = PIDS_DIR / f"skill_{config.skill_id}.pid"
    pid_file.write_text(str(process.pid))

    # Wait for startup
    for i in range(config.startup_timeout):
        time.sleep(1)
        if check_server_health(config):
            print(f"  {config.skill_id}: Summoned successfully!")
            return True

    print(f"  {config.skill_id}: Failed to start (timeout after {config.startup_timeout}s)")
    return False


def dismiss_server(config: SkillServerConfig) -> bool:
    """Stop a skill server."""
    pid = get_server_pid(config.skill_id)

    if pid:
        print(f"  {config.skill_id}: Dismissing (PID {pid})...")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(1)
            # Force kill if still running
            try:
                os.kill(pid, 0)
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        except Exception as e:
            print(f"  {config.skill_id}: Error stopping: {e}")

        # Clean up PID file
        pid_file = PIDS_DIR / f"skill_{config.skill_id}.pid"
        pid_file.unlink(missing_ok=True)
        return True
    else:
        # Try to find by port
        try:
            result = subprocess.run(
                f"lsof -ti:{config.port}",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for p in pids:
                    try:
                        os.kill(int(p), signal.SIGTERM)
                        print(f"  {config.skill_id}: Killed process on port {config.port}")
                    except:
                        pass
                return True
        except:
            pass

    print(f"  {config.skill_id}: Not running")
    return False


def pull_eval_sets(config: SkillServerConfig) -> int:
    """Pull evaluation sets from a running skill server."""
    if not check_server_health(config):
        print(f"  {config.skill_id}: Server not running, skipping")
        return 0

    # Create eval cache directory
    cache_dir = Path(config.eval_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pulled = 0
    for level in range(1, config.max_level + 1):
        try:
            # API uses /eval/{level} format
            url = f"{config.url}/eval/{level}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Save to cache
                eval_file = cache_dir / f"level_{level:02d}.json"
                with open(eval_file, "w") as f:
                    json.dump(data, f, indent=2)

                sample_count = len(data.get("samples", data.get("problems", [])))
                pulled += 1
                print(f"  {config.skill_id} L{level}: {sample_count} samples")
            else:
                print(f"  {config.skill_id} L{level}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  {config.skill_id} L{level}: Error - {e}")

    return pulled


def show_status(configs: Dict[str, SkillServerConfig]):
    """Show status of all skill servers."""
    print("\n  THE SUMMONER - Skill Trainer Status")
    print("  " + "=" * 50)

    for skill_id, config in sorted(configs.items()):
        healthy = check_server_health(config)
        pid = get_server_pid(skill_id)

        if healthy:
            status = f"ONLINE (port {config.port})"
            icon = "+"
        elif pid:
            status = f"STARTING (PID {pid})"
            icon = "~"
        else:
            status = "OFFLINE"
            icon = "-"

        print(f"  [{icon}] {config.name:15} {status}")

    print()


def write_status_file(configs: Dict[str, SkillServerConfig]):
    """Write status to JSON file for other services."""
    status = {
        "timestamp": time.time(),
        "skills": {}
    }

    for skill_id, config in configs.items():
        healthy = check_server_health(config)
        status["skills"][skill_id] = {
            "name": config.name,
            "port": config.port,
            "url": config.url,
            "healthy": healthy,
            "pid": get_server_pid(skill_id),
        }

    status_file = STATUS_DIR / "summoner.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)


def daemon_loop(configs: Dict[str, SkillServerConfig], interval: int = 30):
    """Run as a persistent daemon, keeping servers alive."""
    print("\n  THE SUMMONER - Daemon Mode")
    print("  " + "=" * 50)
    print(f"  Monitoring {len(configs)} skill servers")
    print(f"  Check interval: {interval}s")
    print("  Press Ctrl+C to stop\n")

    # Initial summon
    for skill_id, config in configs.items():
        summon_server(config)

    try:
        while True:
            time.sleep(interval)

            for skill_id, config in configs.items():
                if not check_server_health(config):
                    print(f"  [{time.strftime('%H:%M:%S')}] {skill_id}: Resummoning...")
                    summon_server(config)

            write_status_file(configs)
    except KeyboardInterrupt:
        print("\n  Summoner daemon stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="The Summoner - Skill Trainer Server Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python guild/summoner.py --status          # Check all servers
  python guild/summoner.py --summon          # Start all servers
  python guild/summoner.py --summon bin      # Start only Binary skill
  python guild/summoner.py --dismiss         # Stop all servers
  python guild/summoner.py --daemon          # Run as persistent daemon
  python guild/summoner.py --pull-evals      # Download eval sets
        """
    )

    parser.add_argument("--status", action="store_true", help="Show server status")
    parser.add_argument("--summon", nargs="*", metavar="SKILL", help="Start servers (all if no skills specified)")
    parser.add_argument("--dismiss", nargs="*", metavar="SKILL", help="Stop servers (all if no skills specified)")
    parser.add_argument("--daemon", action="store_true", help="Run as persistent daemon")
    parser.add_argument("--interval", type=int, default=30, help="Daemon check interval (seconds)")
    parser.add_argument("--pull-evals", action="store_true", help="Pull eval sets from running servers")

    args = parser.parse_args()

    # Load configs
    configs = load_skill_configs()

    if not configs:
        print("No skill configurations found in configs/skills/")
        return 1

    # Filter skills if specified
    def filter_configs(skill_list: Optional[List[str]]) -> Dict[str, SkillServerConfig]:
        if skill_list is None or len(skill_list) == 0:
            return configs
        return {k: v for k, v in configs.items() if k in skill_list}

    # Execute command
    if args.status or (args.summon is None and args.dismiss is None and not args.daemon and not args.pull_evals):
        show_status(configs)
        write_status_file(configs)

    elif args.summon is not None:
        target_configs = filter_configs(args.summon)
        print(f"\n  Summoning {len(target_configs)} skill trainer(s)...\n")
        for skill_id, config in target_configs.items():
            summon_server(config)
        print()
        write_status_file(configs)

    elif args.dismiss is not None:
        target_configs = filter_configs(args.dismiss)
        print(f"\n  Dismissing {len(target_configs)} skill trainer(s)...\n")
        for skill_id, config in target_configs.items():
            dismiss_server(config)
        print()
        write_status_file(configs)

    elif args.daemon:
        daemon_loop(configs, args.interval)

    elif args.pull_evals:
        print("\n  Pulling evaluation sets from skill servers...\n")
        total = 0
        for skill_id, config in configs.items():
            count = pull_eval_sets(config)
            total += count
        print(f"\n  Pulled {total} eval sets total.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
