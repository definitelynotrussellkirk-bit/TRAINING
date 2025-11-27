"""
Vault Location Handlers - Handle operations across different storage types.

Location handlers abstract the differences between storage types:
    - LocalHandler: Local filesystem operations
    - RemoteHandler: SSH/rsync operations for remote servers
    - NASHandler: Synology NAS operations

Each handler can:
    - Check if a path exists
    - Verify file integrity (checksum)
    - Transfer files to/from the location
    - List contents at a path
    - Get file/directory size

RPG Flavor:
    Each stronghold has a Guardian that knows how to access its treasures.
    The LocalGuardian walks the local filesystem.
    The RemoteGuardian teleports via SSH.
    The NASGuardian speaks the ancient protocol of SMB.
"""

import hashlib
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vault.storage_registry import Stronghold, StrongholdType, StrongholdStatus


@dataclass
class PathInfo:
    """Information about a path at a location."""
    exists: bool
    is_file: bool
    is_dir: bool
    size_bytes: int
    modified_at: Optional[datetime]
    checksum: Optional[str] = None  # MD5 hash for verification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exists": self.exists,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "checksum": self.checksum,
        }


@dataclass
class TransferResult:
    """Result of a file transfer operation."""
    success: bool
    source: str
    destination: str
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "destination": self.destination,
            "bytes_transferred": self.bytes_transferred,
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }


class LocationHandler(ABC):
    """
    Abstract base class for location handlers.

    Each storage type (local, remote, NAS) has a handler that knows
    how to access and manipulate files at that location.
    """

    def __init__(self, stronghold: Stronghold):
        """
        Initialize handler with stronghold configuration.

        Args:
            stronghold: The stronghold this handler manages
        """
        self.stronghold = stronghold

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists at this location."""
        pass

    @abstractmethod
    def get_info(self, path: str) -> PathInfo:
        """Get information about a path."""
        pass

    @abstractmethod
    def list_dir(self, path: str) -> List[str]:
        """List contents of a directory."""
        pass

    @abstractmethod
    def calculate_checksum(self, path: str) -> Optional[str]:
        """Calculate MD5 checksum of a file."""
        pass

    @abstractmethod
    def fetch(self, remote_path: str, local_path: str) -> TransferResult:
        """
        Fetch a file/directory from this location to local.

        Args:
            remote_path: Path at this location
            local_path: Local destination path

        Returns:
            TransferResult
        """
        pass

    @abstractmethod
    def push(self, local_path: str, remote_path: str) -> TransferResult:
        """
        Push a file/directory from local to this location.

        Args:
            local_path: Local source path
            remote_path: Destination path at this location

        Returns:
            TransferResult
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete a file/directory at this location."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this location is currently accessible."""
        pass

    def verify(self, path: str, expected_checksum: Optional[str] = None) -> Tuple[bool, PathInfo]:
        """
        Verify a file/directory exists and optionally matches checksum.

        Args:
            path: Path to verify
            expected_checksum: Expected MD5 checksum (optional)

        Returns:
            Tuple of (verified, PathInfo)
        """
        info = self.get_info(path)

        if not info.exists:
            return False, info

        if expected_checksum and info.is_file:
            actual_checksum = self.calculate_checksum(path)
            info.checksum = actual_checksum
            if actual_checksum != expected_checksum:
                return False, info

        return True, info


class LocalHandler(LocationHandler):
    """
    Handler for local filesystem operations.

    The LocalGuardian walks the local filesystem directly.
    """

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def get_info(self, path: str) -> PathInfo:
        p = Path(path)

        if not p.exists():
            return PathInfo(
                exists=False,
                is_file=False,
                is_dir=False,
                size_bytes=0,
                modified_at=None,
            )

        stat = p.stat()

        if p.is_file():
            size = stat.st_size
        else:
            # Directory - sum all files
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

        return PathInfo(
            exists=True,
            is_file=p.is_file(),
            is_dir=p.is_dir(),
            size_bytes=size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

    def list_dir(self, path: str) -> List[str]:
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return []
        return [str(item) for item in p.iterdir()]

    def calculate_checksum(self, path: str) -> Optional[str]:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None

        hash_md5 = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def fetch(self, remote_path: str, local_path: str) -> TransferResult:
        # For local handler, fetch is just a copy
        return self._copy(remote_path, local_path)

    def push(self, local_path: str, remote_path: str) -> TransferResult:
        # For local handler, push is just a copy
        return self._copy(local_path, remote_path)

    def _copy(self, src: str, dst: str) -> TransferResult:
        start = datetime.now()
        src_path = Path(src)
        dst_path = Path(dst)

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

            duration = (datetime.now() - start).total_seconds()
            size = self.get_info(dst).size_bytes

            return TransferResult(
                success=True,
                source=src,
                destination=dst,
                bytes_transferred=size,
                duration_seconds=duration,
            )

        except Exception as e:
            return TransferResult(
                success=False,
                source=src,
                destination=dst,
                error=str(e),
            )

    def delete(self, path: str) -> bool:
        p = Path(path)
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return Path(self.stronghold.base_path).exists()


class RemoteHandler(LocationHandler):
    """
    Handler for remote server operations via SSH/rsync.

    The RemoteGuardian teleports via SSH to access distant strongholds.
    Requires SSH key-based authentication configured.
    """

    def __init__(self, stronghold: Stronghold):
        super().__init__(stronghold)
        self.host = stronghold.host
        self.base_path = stronghold.base_path
        # SSH user - defaults to current user
        self.user = stronghold.metadata.get("user", os.getenv("USER", "user")) if hasattr(stronghold, 'metadata') else os.getenv("USER", "user")

    def _ssh_cmd(self, cmd: str) -> Tuple[bool, str]:
        """Execute a command via SSH."""
        full_cmd = ["ssh", f"{self.user}@{self.host}", cmd]
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "SSH command timed out"
        except Exception as e:
            return False, str(e)

    def _full_path(self, path: str) -> str:
        """Get full remote path."""
        if path.startswith("/"):
            return path
        return f"{self.base_path}/{path}"

    def exists(self, path: str) -> bool:
        full_path = self._full_path(path)
        success, _ = self._ssh_cmd(f'test -e "{full_path}" && echo yes || echo no')
        return success and "yes" in _

    def get_info(self, path: str) -> PathInfo:
        full_path = self._full_path(path)

        # Check existence and type
        success, output = self._ssh_cmd(
            f'if [ -e "{full_path}" ]; then '
            f'if [ -f "{full_path}" ]; then echo "file"; else echo "dir"; fi; '
            f'stat -c "%s %Y" "{full_path}" 2>/dev/null || du -sb "{full_path}" | cut -f1; '
            f'else echo "missing"; fi'
        )

        if not success or "missing" in output:
            return PathInfo(
                exists=False,
                is_file=False,
                is_dir=False,
                size_bytes=0,
                modified_at=None,
            )

        lines = output.strip().split("\n")
        file_type = lines[0] if lines else "dir"

        # Parse size and time
        size = 0
        mtime = None
        if len(lines) > 1:
            parts = lines[1].split()
            if parts:
                try:
                    size = int(parts[0])
                except ValueError:
                    pass
                if len(parts) > 1:
                    try:
                        mtime = datetime.fromtimestamp(int(parts[1]))
                    except (ValueError, OSError):
                        pass

        return PathInfo(
            exists=True,
            is_file=(file_type == "file"),
            is_dir=(file_type == "dir"),
            size_bytes=size,
            modified_at=mtime,
        )

    def list_dir(self, path: str) -> List[str]:
        full_path = self._full_path(path)
        success, output = self._ssh_cmd(f'ls -1 "{full_path}" 2>/dev/null')
        if not success:
            return []
        return [f"{full_path}/{name}" for name in output.strip().split("\n") if name]

    def calculate_checksum(self, path: str) -> Optional[str]:
        full_path = self._full_path(path)
        success, output = self._ssh_cmd(f'md5sum "{full_path}" 2>/dev/null | cut -d" " -f1')
        return output if success else None

    def fetch(self, remote_path: str, local_path: str) -> TransferResult:
        """Fetch from remote to local using rsync."""
        start = datetime.now()
        full_remote = self._full_path(remote_path)
        remote_spec = f"{self.user}@{self.host}:{full_remote}"

        # Ensure local parent exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use rsync for efficient transfer
            cmd = [
                "rsync", "-avz", "--progress",
                remote_spec,
                local_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            duration = (datetime.now() - start).total_seconds()

            if result.returncode == 0:
                # Get size of transferred data
                local_handler = LocalHandler(self.stronghold)
                size = local_handler.get_info(local_path).size_bytes

                return TransferResult(
                    success=True,
                    source=remote_spec,
                    destination=local_path,
                    bytes_transferred=size,
                    duration_seconds=duration,
                )
            else:
                return TransferResult(
                    success=False,
                    source=remote_spec,
                    destination=local_path,
                    error=result.stderr,
                )

        except Exception as e:
            return TransferResult(
                success=False,
                source=remote_spec,
                destination=local_path,
                error=str(e),
            )

    def push(self, local_path: str, remote_path: str) -> TransferResult:
        """Push from local to remote using rsync."""
        start = datetime.now()
        full_remote = self._full_path(remote_path)
        remote_spec = f"{self.user}@{self.host}:{full_remote}"

        # Ensure remote parent exists
        parent_dir = str(Path(full_remote).parent)
        self._ssh_cmd(f'mkdir -p "{parent_dir}"')

        try:
            cmd = [
                "rsync", "-avz", "--progress",
                local_path,
                remote_spec,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            duration = (datetime.now() - start).total_seconds()

            if result.returncode == 0:
                local_handler = LocalHandler(self.stronghold)
                size = local_handler.get_info(local_path).size_bytes

                return TransferResult(
                    success=True,
                    source=local_path,
                    destination=remote_spec,
                    bytes_transferred=size,
                    duration_seconds=duration,
                )
            else:
                return TransferResult(
                    success=False,
                    source=local_path,
                    destination=remote_spec,
                    error=result.stderr,
                )

        except Exception as e:
            return TransferResult(
                success=False,
                source=local_path,
                destination=remote_spec,
                error=str(e),
            )

    def delete(self, path: str) -> bool:
        full_path = self._full_path(path)
        success, _ = self._ssh_cmd(f'rm -rf "{full_path}"')
        return success

    def is_available(self) -> bool:
        """Check if remote host is reachable via SSH."""
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", f"{self.user}@{self.host}", "echo ok"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


class NASHandler(LocationHandler):
    """
    Handler for NAS operations (Synology).

    The NASGuardian can access the NAS via:
    1. Local mount point (if mounted)
    2. SSH/rsync (if SSH enabled)
    3. Synology API (future)

    Prefers local mount > SSH > API for performance.
    """

    def __init__(self, stronghold: Stronghold):
        super().__init__(stronghold)
        self.host = stronghold.host
        self.share = stronghold.share
        self.base_path = stronghold.base_path
        self.mount_point = stronghold.mount_point

        # Determine best access method
        self._method = self._determine_access_method()

    def _determine_access_method(self) -> str:
        """Determine the best way to access this NAS."""
        # Check mount point first
        if self.mount_point and Path(self.mount_point).exists():
            # Verify it's actually mounted (not empty)
            try:
                contents = list(Path(self.mount_point).iterdir())
                if contents:
                    return "mount"
            except PermissionError:
                pass

        # Try SSH
        if self.host:
            try:
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=3", f"admin@{self.host}", "echo ok"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return "ssh"
            except Exception:
                pass

        # Fallback to rsync over SSH
        return "rsync"

    def _get_full_path(self, path: str) -> str:
        """Get full path depending on access method."""
        if self._method == "mount":
            mount_base = Path(self.mount_point) / self.base_path
            if path.startswith("/"):
                return str(mount_base / path.lstrip("/"))
            return str(mount_base / path)
        else:
            # SSH/rsync path
            if path.startswith("/"):
                return f"/volume1/{self.share}/{self.base_path}{path}"
            return f"/volume1/{self.share}/{self.base_path}/{path}"

    def _local_handler(self) -> Optional[LocalHandler]:
        """Get a local handler for mounted NAS."""
        if self._method == "mount":
            return LocalHandler(self.stronghold)
        return None

    def _remote_handler(self) -> Optional[RemoteHandler]:
        """Get a remote handler for SSH access."""
        if self._method in ("ssh", "rsync"):
            # Create a pseudo-stronghold for remote handler
            remote_sh = Stronghold(
                name=self.stronghold.name,
                stronghold_type=StrongholdType.REMOTE,
                host=self.host,
                base_path=f"/volume1/{self.share}/{self.base_path}",
            )
            return RemoteHandler(remote_sh)
        return None

    def exists(self, path: str) -> bool:
        full_path = self._get_full_path(path)

        if self._method == "mount":
            return Path(full_path).exists()
        else:
            handler = self._remote_handler()
            return handler.exists(full_path) if handler else False

    def get_info(self, path: str) -> PathInfo:
        full_path = self._get_full_path(path)

        if self._method == "mount":
            handler = self._local_handler()
            return handler.get_info(full_path) if handler else PathInfo(
                exists=False, is_file=False, is_dir=False, size_bytes=0, modified_at=None
            )
        else:
            handler = self._remote_handler()
            return handler.get_info(full_path) if handler else PathInfo(
                exists=False, is_file=False, is_dir=False, size_bytes=0, modified_at=None
            )

    def list_dir(self, path: str) -> List[str]:
        full_path = self._get_full_path(path)

        if self._method == "mount":
            handler = self._local_handler()
            return handler.list_dir(full_path) if handler else []
        else:
            handler = self._remote_handler()
            return handler.list_dir(full_path) if handler else []

    def calculate_checksum(self, path: str) -> Optional[str]:
        full_path = self._get_full_path(path)

        if self._method == "mount":
            handler = self._local_handler()
            return handler.calculate_checksum(full_path) if handler else None
        else:
            handler = self._remote_handler()
            return handler.calculate_checksum(full_path) if handler else None

    def fetch(self, remote_path: str, local_path: str) -> TransferResult:
        full_path = self._get_full_path(remote_path)

        if self._method == "mount":
            # Direct copy from mount
            handler = self._local_handler()
            return handler.fetch(full_path, local_path) if handler else TransferResult(
                success=False, source=full_path, destination=local_path, error="No handler"
            )
        else:
            # rsync from NAS
            handler = self._remote_handler()
            return handler.fetch(full_path, local_path) if handler else TransferResult(
                success=False, source=full_path, destination=local_path, error="No handler"
            )

    def push(self, local_path: str, remote_path: str) -> TransferResult:
        full_path = self._get_full_path(remote_path)

        if self._method == "mount":
            # Direct copy to mount
            handler = self._local_handler()
            return handler.push(local_path, full_path) if handler else TransferResult(
                success=False, source=local_path, destination=full_path, error="No handler"
            )
        else:
            # rsync to NAS
            handler = self._remote_handler()
            return handler.push(local_path, full_path) if handler else TransferResult(
                success=False, source=local_path, destination=full_path, error="No handler"
            )

    def delete(self, path: str) -> bool:
        full_path = self._get_full_path(path)

        if self._method == "mount":
            handler = self._local_handler()
            return handler.delete(full_path) if handler else False
        else:
            handler = self._remote_handler()
            return handler.delete(full_path) if handler else False

    def is_available(self) -> bool:
        """Check if NAS is accessible."""
        if self._method == "mount":
            try:
                mount_path = Path(self.mount_point)
                return mount_path.exists() and any(mount_path.iterdir())
            except Exception:
                return False
        else:
            handler = self._remote_handler()
            return handler.is_available() if handler else False


def get_handler(stronghold: Stronghold) -> LocationHandler:
    """
    Get the appropriate handler for a stronghold.

    Args:
        stronghold: The stronghold configuration

    Returns:
        LocationHandler for this stronghold type
    """
    if stronghold.stronghold_type == StrongholdType.LOCAL:
        return LocalHandler(stronghold)
    elif stronghold.stronghold_type == StrongholdType.NAS:
        return NASHandler(stronghold)
    elif stronghold.stronghold_type == StrongholdType.REMOTE:
        return RemoteHandler(stronghold)
    elif stronghold.stronghold_type == StrongholdType.CLOUD:
        # Future: CloudHandler
        raise NotImplementedError(f"Cloud storage not yet implemented")
    else:
        raise ValueError(f"Unknown stronghold type: {stronghold.stronghold_type}")
