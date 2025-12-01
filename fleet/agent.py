#!/usr/bin/env python3
"""
Fleet Node Agent - Health monitoring and command execution for a single node.

The agent runs on each node in the fleet and:
1. Collects health metrics (disk, GPU, memory, processes)
2. Serves an HTTP API for health queries and commands
3. Executes retention and other maintenance tasks

SELF-CONTAINED: This agent can run standalone without external dependencies.
All configuration can be passed via command line or local config file.

Usage:
    # Run with command-line config (recommended for remote nodes)
    python3 fleet_agent.py \\
        --host-id 3090 \\
        --device-id inference3090 \\
        --hostname inference.local \\
        --checkpoints-dir ~/llm/models \\
        --max-checkpoints 3 \\
        --max-gb 150

    # Run with hosts.json (for nodes that have it)
    python3 -m fleet.agent --host-id 4090

    # Run with local config file
    python3 fleet_agent.py --config ~/llm/fleet_agent.json

API Endpoints:
    GET  /health          - Basic health check
    GET  /api/status      - Full health snapshot
    POST /api/retention   - Trigger retention (body: {"dry_run": bool})
    GET  /api/checkpoints - List checkpoints on this node
"""

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Import types - these are bundled with the agent
try:
    from fleet.types import (
        GPUHealth,
        NodeHealth,
        NodeStatus,
        ProcessInfo,
        RetentionPolicy,
        RetentionResult,
        StorageHealth,
        StorageZone,
    )
except ImportError:
    # Standalone mode - types defined inline
    from dataclasses import dataclass, field
    from enum import Enum

    class NodeStatus(str, Enum):
        HEALTHY = "healthy"
        WARNING = "warning"
        CRITICAL = "critical"
        OFFLINE = "offline"
        UNKNOWN = "unknown"

    class StorageZone(str, Enum):
        HOT = "hot"
        WARM = "warm"
        COLD = "cold"

    @dataclass
    class StorageHealth:
        path: str
        zone: StorageZone
        total_bytes: int
        used_bytes: int
        free_bytes: int
        checkpoint_count: int = 0
        oldest_checkpoint: Optional[str] = None
        newest_checkpoint: Optional[str] = None

        @property
        def used_pct(self) -> float:
            return (self.used_bytes / self.total_bytes * 100) if self.total_bytes else 0

        @property
        def free_gb(self) -> float:
            return self.free_bytes / (1024 ** 3)

        @property
        def used_gb(self) -> float:
            return self.used_bytes / (1024 ** 3)

        @property
        def total_gb(self) -> float:
            return self.total_bytes / (1024 ** 3)

        def to_dict(self) -> Dict:
            return {
                "path": self.path, "zone": self.zone.value,
                "total_gb": round(self.total_gb, 2), "used_gb": round(self.used_gb, 2),
                "free_gb": round(self.free_gb, 2), "used_pct": round(self.used_pct, 1),
                "checkpoint_count": self.checkpoint_count,
                "oldest_checkpoint": self.oldest_checkpoint,
                "newest_checkpoint": self.newest_checkpoint,
            }

    @dataclass
    class GPUHealth:
        index: int
        name: str
        vram_total_mb: int
        vram_used_mb: int
        vram_free_mb: int
        utilization_pct: float
        temperature_c: int
        power_draw_w: float

        @property
        def vram_used_pct(self) -> float:
            return (self.vram_used_mb / self.vram_total_mb * 100) if self.vram_total_mb else 0

        def to_dict(self) -> Dict:
            return {
                "index": self.index, "name": self.name,
                "vram_total_mb": self.vram_total_mb, "vram_used_mb": self.vram_used_mb,
                "vram_free_mb": self.vram_free_mb, "vram_used_pct": round(self.vram_used_pct, 1),
                "utilization_pct": round(self.utilization_pct, 1),
                "temperature_c": self.temperature_c, "power_draw_w": round(self.power_draw_w, 1),
            }

    @dataclass
    class ProcessInfo:
        pid: int
        name: str
        cpu_pct: float
        memory_mb: float
        started_at: Optional[str] = None

        def to_dict(self) -> Dict:
            return {"pid": self.pid, "name": self.name, "cpu_pct": round(self.cpu_pct, 1),
                    "memory_mb": round(self.memory_mb, 1), "started_at": self.started_at}

    @dataclass
    class RetentionPolicy:
        max_checkpoints: Optional[int]
        max_gb: Optional[float]
        keep_strategy: str
        is_vault: bool
        cleanup_threshold_pct: float = 90.0

        def to_dict(self) -> Dict:
            return {"max_checkpoints": self.max_checkpoints, "max_gb": self.max_gb,
                    "keep_strategy": self.keep_strategy, "is_vault": self.is_vault,
                    "cleanup_threshold_pct": self.cleanup_threshold_pct}

    @dataclass
    class RetentionResult:
        host_id: str
        device_id: str
        timestamp: str
        dry_run: bool
        checkpoints_before: int
        checkpoints_after: int
        deleted_count: int
        deleted_steps: List[int]
        freed_bytes: int
        errors: List[str]

        @property
        def freed_gb(self) -> float:
            return self.freed_bytes / (1024 ** 3)

        def to_dict(self) -> Dict:
            return {
                "host_id": self.host_id, "device_id": self.device_id,
                "timestamp": self.timestamp, "dry_run": self.dry_run,
                "checkpoints_before": self.checkpoints_before,
                "checkpoints_after": self.checkpoints_after,
                "deleted_count": self.deleted_count, "deleted_steps": self.deleted_steps,
                "freed_gb": round(self.freed_gb, 2), "errors": self.errors,
            }

    @dataclass
    class NodeHealth:
        host_id: str
        device_id: str
        hostname: str
        status: NodeStatus
        timestamp: str
        uptime_seconds: int
        cpu_pct: float
        memory_total_mb: int
        memory_used_mb: int
        memory_free_mb: int
        storage: List[StorageHealth] = field(default_factory=list)
        gpus: List[GPUHealth] = field(default_factory=list)
        processes: List[ProcessInfo] = field(default_factory=list)
        retention_policy: Optional[RetentionPolicy] = None
        alerts: List[str] = field(default_factory=list)
        last_retention_run: Optional[str] = None
        checkpoints_deleted_today: int = 0
        bytes_freed_today: int = 0

        @property
        def memory_used_pct(self) -> float:
            return (self.memory_used_mb / self.memory_total_mb * 100) if self.memory_total_mb else 0

        @property
        def needs_retention(self) -> bool:
            if self.retention_policy and self.retention_policy.is_vault:
                return False
            threshold = self.retention_policy.cleanup_threshold_pct if self.retention_policy else 90
            return any(s.used_pct > threshold for s in self.storage)

        @property
        def checkpoint_count(self) -> int:
            return sum(s.checkpoint_count for s in self.storage)

        def to_dict(self) -> Dict:
            return {
                "host_id": self.host_id, "device_id": self.device_id,
                "hostname": self.hostname, "status": self.status.value,
                "timestamp": self.timestamp, "uptime_seconds": self.uptime_seconds,
                "cpu_pct": round(self.cpu_pct, 1),
                "memory": {"total_mb": self.memory_total_mb, "used_mb": self.memory_used_mb,
                           "free_mb": self.memory_free_mb, "used_pct": round(self.memory_used_pct, 1)},
                "storage": [s.to_dict() for s in self.storage],
                "gpus": [g.to_dict() for g in self.gpus],
                "processes": [p.to_dict() for p in self.processes],
                "retention_policy": self.retention_policy.to_dict() if self.retention_policy else None,
                "alerts": self.alerts, "needs_retention": self.needs_retention,
                "checkpoint_count": self.checkpoint_count,
                "last_retention_run": self.last_retention_run,
                "checkpoints_deleted_today": self.checkpoints_deleted_today,
                "bytes_freed_today": self.bytes_freed_today,
            }


logger = logging.getLogger("fleet.agent")

# Default agent port
DEFAULT_PORT = 8769


class NodeAgent:
    """
    Agent that runs on each node to collect health and execute commands.

    Can be configured via:
    1. Command-line arguments (fully self-contained)
    2. Local config file (fleet_agent.json)
    3. Central hosts.json (if available)
    """

    def __init__(
        self,
        host_id: Optional[str] = None,
        device_id: Optional[str] = None,
        hostname: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        max_checkpoints: Optional[int] = None,
        max_gb: Optional[float] = None,
        is_vault: bool = False,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the agent.

        Args:
            host_id: Host ID (e.g., "3090")
            device_id: Device ID (e.g., "inference3090")
            hostname: IP or hostname
            checkpoints_dir: Directory containing checkpoints
            models_dir: Models directory to monitor
            max_checkpoints: Maximum checkpoints to keep
            max_gb: Maximum GB of checkpoints to keep
            is_vault: If True, never delete checkpoints
            config_path: Path to local config file
        """
        # Store CLI args
        self._cli_host_id = host_id
        self._cli_device_id = device_id
        self._cli_hostname = hostname
        self._cli_checkpoints_dir = checkpoints_dir
        self._cli_models_dir = models_dir
        self._cli_max_checkpoints = max_checkpoints
        self._cli_max_gb = max_gb
        self._cli_is_vault = is_vault

        # Will be populated
        self.host_id: str = ""
        self.device_id: str = ""
        self.hostname: str = ""
        self.checkpoints_dir: Optional[Path] = None
        self.retention_policy: Optional[RetentionPolicy] = None
        self.storage_paths: List[Tuple[str, StorageZone]] = []

        # Runtime state
        self.start_time = time.time()
        self.last_retention_run: Optional[str] = None
        self.checkpoints_deleted_today: int = 0
        self.bytes_freed_today: int = 0
        self._lock = threading.Lock()

        # Load configuration (priority: CLI > local config > hosts.json)
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path] = None):
        """Load configuration from CLI, local config, or hosts.json."""

        # Try local config file first
        if config_path and config_path.exists():
            self._load_from_file(config_path)
            return

        # Try auto-detect local config
        auto_configs = [
            Path("fleet_agent.json"),
            Path.home() / ".fleet_agent.json",
            Path("/etc/fleet/agent.json"),
        ]
        for cfg in auto_configs:
            if cfg.exists():
                self._load_from_file(cfg)
                return

        # Try hosts.json (central config)
        hosts_json = self._find_hosts_json()
        if hosts_json and hosts_json.exists():
            try:
                self._load_from_hosts_json(hosts_json)
                return
            except Exception as e:
                logger.warning(f"Failed to load hosts.json: {e}")

        # Fall back to CLI args only
        self._load_from_cli()

    def _load_from_cli(self):
        """Configure from command-line arguments only."""
        if not self._cli_host_id:
            raise ValueError("--host-id required when no config file available")

        self.host_id = self._cli_host_id
        self.device_id = self._cli_device_id or self._cli_host_id
        self.hostname = self._cli_hostname or "localhost"

        if self._cli_checkpoints_dir:
            self.checkpoints_dir = Path(self._cli_checkpoints_dir)

        models_dir = self._cli_models_dir or self._cli_checkpoints_dir
        if models_dir:
            self.storage_paths.append((models_dir, StorageZone.HOT))

        self.retention_policy = RetentionPolicy(
            max_checkpoints=self._cli_max_checkpoints,
            max_gb=self._cli_max_gb,
            keep_strategy="recently_used",
            is_vault=self._cli_is_vault,
            cleanup_threshold_pct=90.0,
        )

        logger.info(f"Configured from CLI: {self.host_id} ({self.device_id})")

    def _load_from_file(self, path: Path):
        """Load from local JSON config file."""
        with open(path) as f:
            config = json.load(f)

        self.host_id = self._cli_host_id or config.get("host_id", "unknown")
        self.device_id = self._cli_device_id or config.get("device_id", self.host_id)
        self.hostname = self._cli_hostname or config.get("hostname", "localhost")

        ckpt_dir = self._cli_checkpoints_dir or config.get("checkpoints_dir")
        if ckpt_dir:
            self.checkpoints_dir = Path(ckpt_dir)

        models_dir = self._cli_models_dir or config.get("models_dir") or ckpt_dir
        if models_dir:
            self.storage_paths.append((models_dir, StorageZone.HOT))

        retention = config.get("retention", {})
        self.retention_policy = RetentionPolicy(
            max_checkpoints=self._cli_max_checkpoints or retention.get("max_checkpoints"),
            max_gb=self._cli_max_gb or retention.get("max_gb"),
            keep_strategy=retention.get("keep_strategy", "recently_used"),
            is_vault=self._cli_is_vault or retention.get("is_vault", False),
            cleanup_threshold_pct=retention.get("cleanup_threshold_pct", 90.0),
        )

        logger.info(f"Configured from {path}: {self.host_id} ({self.device_id})")

    def _find_hosts_json(self) -> Optional[Path]:
        """Find hosts.json config file."""
        candidates = [
            Path(__file__).parent.parent / "config" / "hosts.json",
            Path.home() / "Desktop" / "TRAINING" / "config" / "hosts.json",
            Path.home() / "llm" / "config" / "hosts.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_from_hosts_json(self, hosts_json: Path):
        """Load from central hosts.json."""
        with open(hosts_json) as f:
            config = json.load(f)

        hosts = config.get("hosts", {})

        # Determine host ID
        host_id = self._cli_host_id or self._detect_host_id(hosts)
        if host_id not in hosts:
            raise ValueError(f"Host '{host_id}' not found in hosts.json")

        host_config = hosts[host_id]

        self.host_id = host_id
        self.device_id = self._cli_device_id or host_config.get("device_id", host_id)
        self.hostname = self._cli_hostname or host_config.get("host", "localhost")

        ckpt_dir = self._cli_checkpoints_dir or host_config.get("checkpoints_dir")
        if ckpt_dir:
            self.checkpoints_dir = Path(ckpt_dir)

        models_dir = self._cli_models_dir or host_config.get("models_dir") or ckpt_dir
        if models_dir:
            self.storage_paths.append((models_dir, StorageZone.HOT))

        retention = host_config.get("checkpoint_retention", {})
        self.retention_policy = RetentionPolicy(
            max_checkpoints=self._cli_max_checkpoints or retention.get("max_checkpoints"),
            max_gb=self._cli_max_gb or retention.get("max_gb"),
            keep_strategy=retention.get("keep_strategy", "recently_used"),
            is_vault=self._cli_is_vault or retention.get("is_vault", False),
            cleanup_threshold_pct=retention.get("cleanup_threshold_pct", 90.0),
        )

        logger.info(f"Configured from hosts.json: {self.host_id} ({self.device_id})")

    def _detect_host_id(self, hosts: Dict) -> str:
        """Auto-detect which host we're running on."""
        import socket

        local_ips = set()
        try:
            hostname = socket.gethostname()
            local_ips.add(socket.gethostbyname(hostname))
            for info in socket.getaddrinfo(hostname, None):
                local_ips.add(info[4][0])
        except Exception:
            pass
        local_ips.add("127.0.0.1")

        for host_id, host_config in hosts.items():
            if host_config.get("host") in local_ips:
                return host_id

        raise ValueError(f"Cannot detect host ID. Local IPs: {local_ips}")

    def get_health(self) -> NodeHealth:
        """Collect current health snapshot."""
        alerts = []
        status = NodeStatus.HEALTHY

        # System metrics
        cpu_pct = self._get_cpu_percent()
        mem_total, mem_used, mem_free = self._get_memory()

        # Storage health
        storage_health = []
        for path, zone in self.storage_paths:
            sh = self._get_storage_health(path, zone)
            if sh:
                storage_health.append(sh)
                # Check thresholds
                if sh.used_pct > 95:
                    alerts.append(f"CRITICAL: {path} at {sh.used_pct:.1f}% capacity")
                    status = NodeStatus.CRITICAL
                elif self.retention_policy and sh.used_pct > self.retention_policy.cleanup_threshold_pct:
                    alerts.append(f"WARNING: {path} at {sh.used_pct:.1f}% capacity")
                    if status == NodeStatus.HEALTHY:
                        status = NodeStatus.WARNING

        # GPU health
        gpus = self._get_gpu_health()

        # Running processes
        processes = self._get_processes()

        # Memory threshold
        if mem_total > 0 and mem_used / mem_total > 0.9:
            alerts.append(f"WARNING: Memory at {mem_used/mem_total*100:.1f}%")
            if status == NodeStatus.HEALTHY:
                status = NodeStatus.WARNING

        return NodeHealth(
            host_id=self.host_id,
            device_id=self.device_id,
            hostname=self.hostname,
            status=status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=int(time.time() - self.start_time),
            cpu_pct=cpu_pct,
            memory_total_mb=mem_total,
            memory_used_mb=mem_used,
            memory_free_mb=mem_free,
            storage=storage_health,
            gpus=gpus,
            processes=processes,
            retention_policy=self.retention_policy,
            alerts=alerts,
            last_retention_run=self.last_retention_run,
            checkpoints_deleted_today=self.checkpoints_deleted_today,
            bytes_freed_today=self.bytes_freed_today,
        )

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percent."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            idle = int(parts[4])
            total = sum(int(p) for p in parts[1:])
            return max(0, min(100, 100 - (idle * 100 / total)))
        except Exception:
            return 0.0

    def _get_memory(self) -> Tuple[int, int, int]:
        """Get memory stats in MB: (total, used, free)."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem = {}
            for line in lines:
                parts = line.split()
                key = parts[0].rstrip(":")
                value = int(parts[1])
                mem[key] = value

            total = mem.get("MemTotal", 0) // 1024
            free = mem.get("MemAvailable", mem.get("MemFree", 0)) // 1024
            used = total - free
            return total, used, free
        except Exception:
            return 0, 0, 0

    def _get_storage_health(self, path: str, zone: StorageZone) -> Optional[StorageHealth]:
        """Get storage health for a path."""
        try:
            p = Path(path)
            if not p.exists():
                return None

            usage = shutil.disk_usage(path)

            # Count checkpoints
            checkpoint_count = 0
            oldest = None
            newest = None

            ckpt_dir = self.checkpoints_dir or p
            if ckpt_dir.exists():
                checkpoints = sorted(
                    [d for d in ckpt_dir.iterdir() if d.is_dir() and "checkpoint" in d.name],
                    key=lambda x: x.stat().st_mtime,
                )
                checkpoint_count = len(checkpoints)
                if checkpoints:
                    oldest = checkpoints[0].name
                    newest = checkpoints[-1].name

            return StorageHealth(
                path=path,
                zone=zone,
                total_bytes=usage.total,
                used_bytes=usage.used,
                free_bytes=usage.free,
                checkpoint_count=checkpoint_count,
                oldest_checkpoint=oldest,
                newest_checkpoint=newest,
            )
        except Exception as e:
            logger.warning(f"Failed to get storage health for {path}: {e}")
            return None

    def _get_gpu_health(self) -> List[GPUHealth]:
        """Get GPU health metrics using nvidia-smi."""
        gpus = []
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return []

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    gpus.append(GPUHealth(
                        index=int(parts[0]), name=parts[1],
                        vram_total_mb=int(parts[2]), vram_used_mb=int(parts[3]),
                        vram_free_mb=int(parts[4]), utilization_pct=float(parts[5]),
                        temperature_c=int(parts[6]),
                        power_draw_w=float(parts[7]) if parts[7] != "[N/A]" else 0,
                    ))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to get GPU health: {e}")
        return gpus

    def _get_processes(self) -> List[ProcessInfo]:
        """Get processes of interest."""
        processes = []
        interesting = ["python", "train", "inference", "vllm", "eval"]

        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n")[1:]:
                parts = line.split(None, 10)
                if len(parts) < 11:
                    continue
                cmd = parts[10].lower()
                if any(term in cmd for term in interesting):
                    processes.append(ProcessInfo(
                        pid=int(parts[1]), name=parts[10][:50],
                        cpu_pct=float(parts[2]), memory_mb=float(parts[5]) / 1024,
                    ))
        except Exception as e:
            logger.warning(f"Failed to get processes: {e}")

        return processes[:10]

    def run_retention(self, dry_run: bool = False) -> RetentionResult:
        """
        Run checkpoint retention on this node.

        SELF-CONTAINED: Does not require external modules.
        """
        logger.info(f"Running retention (dry_run={dry_run})")

        errors = []
        deleted_steps = []
        freed_bytes = 0

        if not self.checkpoints_dir or not self.checkpoints_dir.exists():
            return RetentionResult(
                host_id=self.host_id, device_id=self.device_id,
                timestamp=datetime.now().isoformat(), dry_run=dry_run,
                checkpoints_before=0, checkpoints_after=0,
                deleted_count=0, deleted_steps=[], freed_bytes=0,
                errors=["No checkpoints directory configured"],
            )

        if self.retention_policy and self.retention_policy.is_vault:
            return RetentionResult(
                host_id=self.host_id, device_id=self.device_id,
                timestamp=datetime.now().isoformat(), dry_run=dry_run,
                checkpoints_before=0, checkpoints_after=0,
                deleted_count=0, deleted_steps=[], freed_bytes=0,
                errors=["This node is a vault - no retention performed"],
            )

        # Find checkpoints sorted by modification time (oldest first)
        checkpoints = []
        for item in self.checkpoints_dir.iterdir():
            if not item.is_dir() or "checkpoint" not in item.name:
                continue
            match = re.search(r"checkpoint[_-]?(\d+)", item.name)
            step = int(match.group(1)) if match else 0
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            checkpoints.append({
                "path": item, "step": step, "size": size,
                "mtime": item.stat().st_mtime,
            })

        checkpoints.sort(key=lambda x: x["mtime"])
        checkpoints_before = len(checkpoints)

        # Determine how many to keep
        max_keep = self.retention_policy.max_checkpoints if self.retention_policy else None
        max_bytes = int(self.retention_policy.max_gb * 1024**3) if self.retention_policy and self.retention_policy.max_gb else None

        # Calculate which to delete
        to_delete = []

        if max_keep and len(checkpoints) > max_keep:
            # Delete oldest beyond limit
            to_delete = checkpoints[:-max_keep]

        if max_bytes:
            # Also check size limit
            total_size = sum(c["size"] for c in checkpoints)
            if total_size > max_bytes:
                # Delete oldest until under limit
                remaining = [c for c in checkpoints if c not in to_delete]
                cumulative = 0
                keep_from = 0
                for i, c in enumerate(reversed(remaining)):
                    cumulative += c["size"]
                    if cumulative > max_bytes:
                        keep_from = len(remaining) - i
                        break
                to_delete.extend(remaining[:keep_from])

        # Remove duplicates preserving order
        seen = set()
        to_delete = [c for c in to_delete if not (c["step"] in seen or seen.add(c["step"]))]

        # Delete checkpoints
        for ckpt in to_delete:
            step = ckpt["step"]
            path = ckpt["path"]

            if dry_run:
                logger.info(f"[DRY RUN] Would delete: {path}")
                deleted_steps.append(step)
                freed_bytes += ckpt["size"]
            else:
                try:
                    shutil.rmtree(path)
                    deleted_steps.append(step)
                    freed_bytes += ckpt["size"]
                    logger.info(f"Deleted: {path} ({ckpt['size'] / 1e9:.2f} GB)")
                except Exception as e:
                    errors.append(f"Failed to delete {path}: {e}")
                    logger.error(f"Failed to delete {path}: {e}")

        # Update tracking
        with self._lock:
            self.last_retention_run = datetime.now().isoformat()
            if not dry_run:
                self.checkpoints_deleted_today += len(deleted_steps)
                self.bytes_freed_today += freed_bytes

        return RetentionResult(
            host_id=self.host_id, device_id=self.device_id,
            timestamp=datetime.now().isoformat(), dry_run=dry_run,
            checkpoints_before=checkpoints_before,
            checkpoints_after=checkpoints_before - len(deleted_steps),
            deleted_count=len(deleted_steps), deleted_steps=deleted_steps,
            freed_bytes=freed_bytes, errors=errors,
        )

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List checkpoints on this node."""
        checkpoints = []

        if not self.checkpoints_dir or not self.checkpoints_dir.exists():
            return checkpoints

        for item in self.checkpoints_dir.iterdir():
            if not item.is_dir() or "checkpoint" not in item.name:
                continue

            match = re.search(r"checkpoint[_-]?(\d+)", item.name)
            step = int(match.group(1)) if match else 0
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            stat = item.stat()

            checkpoints.append({
                "name": item.name, "step": step,
                "size_bytes": size, "size_gb": round(size / (1024**3), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return sorted(checkpoints, key=lambda x: x["step"])


class AgentRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the agent API."""

    agent: NodeAgent

    def log_message(self, format, *args):
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400):
        self._send_json({"error": message}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json({"status": "ok", "host_id": self.agent.host_id})
        elif path == "/api/status":
            health = self.agent.get_health()
            self._send_json(health.to_dict())
        elif path == "/api/checkpoints":
            checkpoints = self.agent.list_checkpoints()
            self._send_json({"checkpoints": checkpoints, "count": len(checkpoints)})
        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/retention":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {}

            dry_run = data.get("dry_run", False)
            result = self.agent.run_retention(dry_run=dry_run)
            self._send_json(result.to_dict())
        else:
            self._send_error("Not found", 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_agent(
    host_id: Optional[str] = None,
    device_id: Optional[str] = None,
    hostname: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    max_checkpoints: Optional[int] = None,
    max_gb: Optional[float] = None,
    is_vault: bool = False,
    config_path: Optional[str] = None,
    port: int = DEFAULT_PORT,
):
    """Run the agent HTTP server."""
    agent = NodeAgent(
        host_id=host_id,
        device_id=device_id,
        hostname=hostname,
        checkpoints_dir=checkpoints_dir,
        models_dir=models_dir,
        max_checkpoints=max_checkpoints,
        max_gb=max_gb,
        is_vault=is_vault,
        config_path=Path(config_path) if config_path else None,
    )

    class Handler(AgentRequestHandler):
        pass
    Handler.agent = agent

    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info(f"Fleet Agent starting on port {port} for host {agent.host_id}")
    logger.info(f"  Device ID: {agent.device_id}")
    logger.info(f"  Checkpoints: {agent.checkpoints_dir}")
    logger.info(f"  Retention: max_ckpt={agent.retention_policy.max_checkpoints}, max_gb={agent.retention_policy.max_gb}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down agent")
        server.shutdown()


def create_local_config(output_path: str, **kwargs):
    """Create a local config file for standalone deployment."""
    config = {
        "host_id": kwargs.get("host_id", "unknown"),
        "device_id": kwargs.get("device_id", kwargs.get("host_id", "unknown")),
        "hostname": kwargs.get("hostname", "localhost"),
        "checkpoints_dir": kwargs.get("checkpoints_dir", ""),
        "models_dir": kwargs.get("models_dir", ""),
        "retention": {
            "max_checkpoints": kwargs.get("max_checkpoints"),
            "max_gb": kwargs.get("max_gb"),
            "keep_strategy": "recently_used",
            "is_vault": kwargs.get("is_vault", False),
            "cleanup_threshold_pct": 90.0,
        },
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created config: {output_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fleet Node Agent")
    parser.add_argument("--host-id", type=str, help="Host ID (e.g., '3090')")
    parser.add_argument("--device-id", type=str, help="Device ID (e.g., 'inference3090')")
    parser.add_argument("--hostname", type=str, help="IP or hostname")
    parser.add_argument("--checkpoints-dir", type=str, help="Checkpoints directory")
    parser.add_argument("--models-dir", type=str, help="Models directory to monitor")
    parser.add_argument("--max-checkpoints", type=int, help="Max checkpoints to keep")
    parser.add_argument("--max-gb", type=float, help="Max GB of checkpoints")
    parser.add_argument("--is-vault", action="store_true", help="Never delete checkpoints")
    parser.add_argument("--config", type=str, help="Path to local config file")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="API port")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--create-config", type=str, help="Create config file and exit")

    args = parser.parse_args()

    if args.create_config:
        create_local_config(
            args.create_config,
            host_id=args.host_id,
            device_id=args.device_id,
            hostname=args.hostname,
            checkpoints_dir=args.checkpoints_dir,
            models_dir=args.models_dir,
            max_checkpoints=args.max_checkpoints,
            max_gb=args.max_gb,
            is_vault=args.is_vault,
        )
    elif args.status:
        agent = NodeAgent(
            host_id=args.host_id,
            device_id=args.device_id,
            hostname=args.hostname,
            checkpoints_dir=args.checkpoints_dir,
            models_dir=args.models_dir,
            max_checkpoints=args.max_checkpoints,
            max_gb=args.max_gb,
            is_vault=args.is_vault,
            config_path=Path(args.config) if args.config else None,
        )
        health = agent.get_health()
        print(json.dumps(health.to_dict(), indent=2))
    else:
        run_agent(
            host_id=args.host_id,
            device_id=args.device_id,
            hostname=args.hostname,
            checkpoints_dir=args.checkpoints_dir,
            models_dir=args.models_dir,
            max_checkpoints=args.max_checkpoints,
            max_gb=args.max_gb,
            is_vault=args.is_vault,
            config_path=args.config,
            port=args.port,
        )
