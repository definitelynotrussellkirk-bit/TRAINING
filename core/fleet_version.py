"""
Fleet Version System - Multi-machine deployment version tracking.

This system ensures all machines in the fleet are running compatible code.
When code changes on the trainer, remote services know they need to update.

Architecture:
- Trainer maintains /status/fleet_manifest.json with version info
- Remote machines check this manifest on startup and periodically
- Garrison includes fleet version checks in health monitoring
- Temple Cleric can run "fleet_vitals" ritual to verify consistency

Usage:
    from core.fleet_version import FleetVersionManager, get_fleet_manager

    manager = get_fleet_manager()

    # Check if local service needs update
    if manager.is_stale("eval_runner"):
        print("eval_runner needs restart - code changed")

    # Get version info
    manifest = manager.get_manifest()
    print(f"Fleet version: {manifest.version}")
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceVersionInfo:
    """Version info for a single service."""
    name: str
    critical_files: List[str]  # Files this service depends on
    current_hash: str = ""  # Hash of critical files
    last_updated: str = ""
    required_restart_after: str = ""  # If manifest version > this, restart needed


@dataclass
class FleetManifest:
    """Master manifest tracking fleet-wide versions."""
    version: str  # Semantic version or timestamp-based
    code_hash: str  # Combined hash of all critical code
    config_hash: str  # Hash of critical configs
    updated_at: str
    trainer_hostname: str

    services: Dict[str, ServiceVersionInfo] = field(default_factory=dict)

    # Files that affect ALL services
    global_critical_files: List[str] = field(default_factory=list)

    # Minimum compatible versions for remote machines
    min_inference_version: str = "0.0.0"
    min_eval_version: str = "0.0.0"


# Critical files per service - when these change, service needs restart
SERVICE_DEPENDENCIES = {
    "inference": {
        "critical_files": [
            "inference/main.py",
            "inference/inference_worker.py",
            "inference/model_pool.py",
        ],
        "description": "Remote inference server on 3090",
    },
    "eval_runner": {
        "critical_files": [
            "core/eval_runner.py",
            "core/eval_dynamics.py",
            "core/checkpoint_ledger.py",
            "guild/skills/__init__.py",
        ],
        "description": "Evaluation processor daemon",
    },
    "deployment_orchestrator": {
        "critical_files": [
            "monitoring/deployment_orchestrator.py",
            "core/checkpoint_ledger.py",
            "vault/device_mapping.py",
        ],
        "description": "Checkpoint deployment to inference",
    },
    "hero_loop": {
        "critical_files": [
            "arena/hero_loop.py",
            "core/eval_dynamics.py",
        ],
        "description": "Main training orchestration loop",
    },
    "tavern": {
        "critical_files": [
            "tavern/server.py",
            "tavern/api/*.py",
        ],
        "description": "Game UI server",
    },
}

# Files that affect ALL services - changes here require full fleet restart
GLOBAL_CRITICAL_FILES = [
    "core/paths.py",
    "config/hosts.json",
    "config/device_mapping.json",
    "configs/services.json",
]


class FleetVersionManager:
    """Manages fleet version tracking and staleness detection."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "status" / "fleet_manifest.json"
        self._manifest: Optional[FleetManifest] = None
        self._local_hashes: Dict[str, str] = {}

    def _hash_file(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        if not filepath.exists():
            return "MISSING"
        try:
            with open(filepath, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not hash {filepath}: {e}")
            return "ERROR"

    def _hash_files(self, files: List[str]) -> str:
        """Compute combined hash of multiple files."""
        hashes = []
        for file_pattern in files:
            # Handle glob patterns
            if "*" in file_pattern:
                matches = list(self.base_dir.glob(file_pattern))
                for match in sorted(matches):
                    hashes.append(self._hash_file(match))
            else:
                filepath = self.base_dir / file_pattern
                hashes.append(self._hash_file(filepath))

        combined = "|".join(hashes)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _get_version_string(self) -> str:
        """Generate version string based on timestamp."""
        now = datetime.now()
        return now.strftime("%Y.%m.%d.%H%M")

    def generate_manifest(self) -> FleetManifest:
        """Generate a new fleet manifest from current code state."""
        now = datetime.now().isoformat()

        # Hash global critical files
        global_hash = self._hash_files(GLOBAL_CRITICAL_FILES)

        # Hash all config files
        config_files = list(self.base_dir.glob("configs/**/*.yaml"))
        config_files.extend(self.base_dir.glob("configs/**/*.json"))
        config_files.extend(self.base_dir.glob("config/*.json"))
        config_hash = self._hash_files([str(f.relative_to(self.base_dir)) for f in config_files])

        # Generate per-service version info
        services = {}
        all_hashes = [global_hash]

        for service_name, info in SERVICE_DEPENDENCIES.items():
            service_hash = self._hash_files(info["critical_files"])
            all_hashes.append(service_hash)

            services[service_name] = ServiceVersionInfo(
                name=service_name,
                critical_files=info["critical_files"],
                current_hash=service_hash,
                last_updated=now,
            )

        # Combined code hash
        code_hash = hashlib.sha256("|".join(all_hashes).encode()).hexdigest()[:16]

        import socket
        hostname = socket.gethostname()

        manifest = FleetManifest(
            version=self._get_version_string(),
            code_hash=code_hash,
            config_hash=config_hash,
            updated_at=now,
            trainer_hostname=hostname,
            services=services,
            global_critical_files=GLOBAL_CRITICAL_FILES,
        )

        return manifest

    def save_manifest(self, manifest: Optional[FleetManifest] = None) -> FleetManifest:
        """Generate and save fleet manifest."""
        if manifest is None:
            manifest = self.generate_manifest()

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        data = asdict(manifest)
        # Convert ServiceVersionInfo objects
        data["services"] = {k: asdict(v) for k, v in manifest.services.items()}

        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

        self._manifest = manifest
        logger.info(f"Saved fleet manifest v{manifest.version} (code={manifest.code_hash})")
        return manifest

    def load_manifest(self) -> Optional[FleetManifest]:
        """Load fleet manifest from disk."""
        if not self.manifest_path.exists():
            return None

        try:
            with open(self.manifest_path) as f:
                data = json.load(f)

            # Reconstruct ServiceVersionInfo objects
            services = {}
            for k, v in data.get("services", {}).items():
                services[k] = ServiceVersionInfo(**v)
            data["services"] = services

            self._manifest = FleetManifest(**data)
            return self._manifest
        except Exception as e:
            logger.error(f"Failed to load fleet manifest: {e}")
            return None

    def get_manifest(self) -> FleetManifest:
        """Get current manifest, loading or generating as needed."""
        if self._manifest is None:
            self._manifest = self.load_manifest()
        if self._manifest is None:
            self._manifest = self.save_manifest()
        return self._manifest

    def is_stale(self, service_name: str) -> bool:
        """Check if a service's code has changed since manifest was created."""
        manifest = self.get_manifest()

        if service_name not in SERVICE_DEPENDENCIES:
            logger.warning(f"Unknown service: {service_name}")
            return False

        # Check global files first
        current_global = self._hash_files(GLOBAL_CRITICAL_FILES)
        if current_global != self._hash_files(manifest.global_critical_files):
            logger.info(f"Global critical files changed - {service_name} is stale")
            return True

        # Check service-specific files
        info = SERVICE_DEPENDENCIES[service_name]
        current_hash = self._hash_files(info["critical_files"])

        if service_name in manifest.services:
            manifest_hash = manifest.services[service_name].current_hash
            if current_hash != manifest_hash:
                logger.info(f"{service_name} is stale: {manifest_hash} -> {current_hash}")
                return True

        return False

    def get_stale_services(self) -> List[str]:
        """Get list of all services that need restart."""
        stale = []
        for service_name in SERVICE_DEPENDENCIES:
            if self.is_stale(service_name):
                stale.append(service_name)
        return stale

    def check_fleet_health(self) -> Dict:
        """Check overall fleet version health."""
        manifest = self.get_manifest()
        stale = self.get_stale_services()

        # Generate current hashes for comparison
        current = self.generate_manifest()

        return {
            "manifest_version": manifest.version,
            "manifest_updated": manifest.updated_at,
            "code_hash_match": manifest.code_hash == current.code_hash,
            "config_hash_match": manifest.config_hash == current.config_hash,
            "stale_services": stale,
            "all_services_current": len(stale) == 0,
            "current_code_hash": current.code_hash,
            "manifest_code_hash": manifest.code_hash,
        }

    def update_and_report(self) -> Dict:
        """Update manifest and report what changed."""
        old_manifest = self.load_manifest()
        new_manifest = self.generate_manifest()

        changes = {
            "version_changed": False,
            "code_changed": False,
            "config_changed": False,
            "services_changed": [],
        }

        if old_manifest:
            changes["version_changed"] = old_manifest.version != new_manifest.version
            changes["code_changed"] = old_manifest.code_hash != new_manifest.code_hash
            changes["config_changed"] = old_manifest.config_hash != new_manifest.config_hash

            # Find which services changed
            for service_name, new_info in new_manifest.services.items():
                if service_name in old_manifest.services:
                    old_info = old_manifest.services[service_name]
                    if old_info.current_hash != new_info.current_hash:
                        changes["services_changed"].append(service_name)
                else:
                    changes["services_changed"].append(service_name)

        # Save new manifest
        self.save_manifest(new_manifest)

        return changes


# Singleton instance
_manager: Optional[FleetVersionManager] = None


def get_fleet_manager() -> FleetVersionManager:
    """Get the singleton FleetVersionManager instance."""
    global _manager
    if _manager is None:
        _manager = FleetVersionManager()
    return _manager


def check_and_update_manifest() -> Dict:
    """Convenience function to update manifest and check for changes."""
    return get_fleet_manager().update_and_report()


def is_service_stale(service_name: str) -> bool:
    """Convenience function to check if a service needs restart."""
    return get_fleet_manager().is_stale(service_name)


def get_stale_services() -> List[str]:
    """Convenience function to get all stale services."""
    return get_fleet_manager().get_stale_services()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fleet Version Manager")
    parser.add_argument("command", choices=["update", "check", "stale", "info"])
    parser.add_argument("--service", help="Service name for stale check")
    args = parser.parse_args()

    manager = get_fleet_manager()

    if args.command == "update":
        changes = manager.update_and_report()
        print(f"Manifest updated")
        if changes["code_changed"]:
            print(f"  Code changed!")
        if changes["config_changed"]:
            print(f"  Config changed!")
        if changes["services_changed"]:
            print(f"  Services changed: {', '.join(changes['services_changed'])}")

    elif args.command == "check":
        health = manager.check_fleet_health()
        print(f"Fleet Version: {health['manifest_version']}")
        print(f"Updated: {health['manifest_updated']}")
        print(f"Code Match: {'✓' if health['code_hash_match'] else '✗'}")
        print(f"Config Match: {'✓' if health['config_hash_match'] else '✗'}")
        if health["stale_services"]:
            print(f"Stale Services: {', '.join(health['stale_services'])}")
        else:
            print("All services current ✓")

    elif args.command == "stale":
        if args.service:
            if manager.is_stale(args.service):
                print(f"{args.service}: STALE - needs restart")
            else:
                print(f"{args.service}: current")
        else:
            stale = manager.get_stale_services()
            if stale:
                print(f"Stale services: {', '.join(stale)}")
            else:
                print("No stale services")

    elif args.command == "info":
        manifest = manager.get_manifest()
        print(f"Version: {manifest.version}")
        print(f"Code Hash: {manifest.code_hash}")
        print(f"Config Hash: {manifest.config_hash}")
        print(f"Updated: {manifest.updated_at}")
        print(f"Trainer: {manifest.trainer_hostname}")
        print(f"Services:")
        for name, info in manifest.services.items():
            print(f"  {name}: {info.current_hash}")
