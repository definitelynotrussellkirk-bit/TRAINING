#!/usr/bin/env python3
"""
Storage Manager - Synology NAS Integration

Monitors storage health, capacity, and can sync checkpoints to NAS.

Features:
- Connect to Synology DSM API
- Monitor disk usage and health
- List available shares
- Sync checkpoints/backups to NAS
- Write status to storage_status.json

Usage:
    # As module
    from monitoring.storage_manager import StorageManager
    mgr = StorageManager(host="192.168.x.x", username="user", password="...")
    status = mgr.get_status()

    # As daemon
    python3 storage_manager.py --host 192.168.x.x --interval 300
"""

import json
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings for self-signed certs
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - StorageManager - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages connection to Synology NAS and monitors storage.

    Supports two modes:
    1. DSM API (recommended) - Full access to storage info
    2. SSH fallback - Basic disk usage via df command
    """

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        port: int = 5001,  # HTTPS port for DSM
        base_dir: str = None,
        use_https: bool = True
    ):
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = str(get_base_dir())
        self.base_dir = Path(base_dir)
        self.use_https = use_https

        self.protocol = "https" if use_https else "http"
        self.base_url = f"{self.protocol}://{host}:{port}/webapi"

        # Session management
        self.sid = None  # Session ID from DSM
        self.session = requests.Session()
        self.session.verify = False  # Allow self-signed certs

        # Status file
        self.status_file = self.base_dir / "status" / "storage_status.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Results cache
        self.last_status = None
        self.last_check = None

    def _api_call(self, api: str, method: str, version: int = 1, **params) -> Optional[Dict]:
        """Make a DSM API call"""
        try:
            url = f"{self.base_url}/entry.cgi"
            data = {
                "api": api,
                "method": method,
                "version": version,
                **params
            }

            if self.sid:
                data["_sid"] = self.sid

            resp = self.session.get(url, params=data, timeout=30)
            result = resp.json()

            if result.get("success"):
                return result.get("data", {})
            else:
                error = result.get("error", {})
                logger.warning(f"API error: {api}.{method} - code {error.get('code')}")
                return None

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    def login(self) -> bool:
        """Authenticate with DSM API"""
        try:
            result = self._api_call(
                "SYNO.API.Auth",
                "login",
                version=3,
                account=self.username,
                passwd=self.password,
                session="StorageManager",
                format="sid"
            )

            if result and "sid" in result:
                self.sid = result["sid"]
                logger.info(f"Logged in to Synology DSM at {self.host}")
                return True
            else:
                logger.warning("DSM login failed - trying SSH fallback")
                return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def logout(self):
        """End DSM session"""
        if self.sid:
            self._api_call("SYNO.API.Auth", "logout", session="StorageManager")
            self.sid = None

    def get_system_info(self) -> Optional[Dict]:
        """Get NAS system information"""
        return self._api_call("SYNO.DSM.Info", "getinfo", version=2)

    def get_storage_info(self) -> Optional[Dict]:
        """Get storage volume information"""
        return self._api_call("SYNO.Storage.CGI.Storage", "load_info", version=1)

    def get_disk_info(self) -> Optional[Dict]:
        """Get individual disk information"""
        return self._api_call("SYNO.Storage.CGI.Storage", "load_info", version=1)

    def get_share_info(self) -> Optional[List[Dict]]:
        """Get shared folder information"""
        result = self._api_call("SYNO.FileStation.List", "list_share", version=2)
        if result:
            return result.get("shares", [])
        return None

    def _get_status_via_ssh(self) -> Optional[Dict]:
        """Fallback: Get basic status via SSH"""
        try:
            # Get disk usage
            cmd = f"sshpass -p '{self.password}' ssh -o StrictHostKeyChecking=no {self.username}@{self.host} 'df -h && cat /proc/meminfo | head -3'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                volumes = []

                for line in lines[1:]:  # Skip header
                    if line.startswith('/') or 'volume' in line.lower():
                        parts = line.split()
                        if len(parts) >= 6:
                            volumes.append({
                                "filesystem": parts[0],
                                "size": parts[1],
                                "used": parts[2],
                                "available": parts[3],
                                "use_percent": parts[4].rstrip('%'),
                                "mount": parts[5]
                            })

                return {
                    "method": "ssh",
                    "volumes": volumes,
                    "raw_output": result.stdout[:1000]
                }
        except Exception as e:
            logger.error(f"SSH fallback failed: {e}")

        return None

    def get_status(self) -> Dict:
        """
        Get comprehensive storage status.
        Called by GPU scheduler.

        Returns:
            Dict with storage status for dashboard
        """
        status = {
            "host": self.host,
            "connected": False,
            "method": None,
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "volumes": [],
            "disks": [],
            "shares": [],
            "health": "unknown",
            "total_capacity_tb": 0,
            "used_capacity_tb": 0,
            "free_capacity_tb": 0,
            "usage_percent": 0
        }

        # Try DSM API first
        if self.login():
            status["connected"] = True
            status["method"] = "dsm_api"

            # System info
            sys_info = self.get_system_info()
            if sys_info:
                status["system"] = {
                    "model": sys_info.get("model", "Unknown"),
                    "serial": sys_info.get("serial", "Unknown"),
                    "version": sys_info.get("version_string", "Unknown"),
                    "uptime": sys_info.get("uptime", 0),
                    "temperature": sys_info.get("temperature", 0)
                }

            # Storage info
            storage_info = self.get_storage_info()
            if storage_info:
                # Parse volumes
                for vol in storage_info.get("volumes", []):
                    size_info = vol.get("size", {})
                    vol_info = {
                        "id": vol.get("id", ""),
                        "status": vol.get("status", "unknown"),
                        "size_total_bytes": int(size_info.get("total", 0) or 0),
                        "size_used_bytes": int(size_info.get("used", 0) or 0),
                    }

                    total = vol_info["size_total_bytes"]
                    used = vol_info["size_used_bytes"]

                    if total > 0:
                        vol_info["size_total_tb"] = round(total / (1024**4), 2)
                        vol_info["size_used_tb"] = round(used / (1024**4), 2)
                        vol_info["size_free_tb"] = round((total - used) / (1024**4), 2)
                        vol_info["usage_percent"] = round((used / total) * 100, 1)

                        status["total_capacity_tb"] += vol_info["size_total_tb"]
                        status["used_capacity_tb"] += vol_info["size_used_tb"]
                        status["free_capacity_tb"] += vol_info["size_free_tb"]

                    status["volumes"].append(vol_info)

                # Parse disks
                for disk in storage_info.get("disks", []):
                    disk_size = int(disk.get("size_total", 0) or 0)
                    status["disks"].append({
                        "id": disk.get("id", ""),
                        "model": disk.get("model", "Unknown"),
                        "status": disk.get("status", "unknown"),
                        "size_tb": round(disk_size / (1024**4), 2) if disk_size else 0,
                        "temperature": disk.get("temp", 0),
                        "smart_status": disk.get("smart_status", "unknown")
                    })

            # Shares
            shares = self.get_share_info()
            if shares:
                status["shares"] = [
                    {"name": s.get("name", ""), "path": s.get("path", "")}
                    for s in shares[:10]  # Limit to 10
                ]

            # Calculate overall usage
            if status["total_capacity_tb"] > 0:
                status["usage_percent"] = round(
                    (status["used_capacity_tb"] / status["total_capacity_tb"]) * 100, 1
                )

            # Determine health
            disk_health = all(d.get("status") == "normal" for d in status["disks"])
            vol_health = all(v.get("status") == "normal" for v in status["volumes"])

            if disk_health and vol_health:
                status["health"] = "healthy"
            elif any(d.get("status") == "crashed" for d in status["disks"]):
                status["health"] = "critical"
            else:
                status["health"] = "degraded"

            self.logout()

        else:
            # Try SSH fallback
            ssh_status = self._get_status_via_ssh()
            if ssh_status:
                status["connected"] = True
                status["method"] = "ssh"
                status["volumes"] = ssh_status.get("volumes", [])
                status["health"] = "unknown"  # Can't determine via SSH

        self.last_status = status
        self.last_check = datetime.now()

        return status

    def check_connection(self) -> Dict:
        """
        Quick connection check.
        Called by GPU scheduler for checkpoint_sync-like tasks.
        """
        try:
            # Try a simple ping first
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", self.host],
                capture_output=True,
                timeout=5
            )

            reachable = result.returncode == 0

            return {
                "host": self.host,
                "reachable": reachable,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "host": self.host,
                "reachable": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _save_status(self):
        """Save status to JSON file for dashboard"""
        if self.last_status:
            self.last_status["last_updated"] = datetime.now().isoformat()
            with open(self.status_file, 'w') as f:
                json.dump(self.last_status, f, indent=2)
            logger.info(f"Status saved to {self.status_file}")

    def sync_to_nas(self, local_path: str, remote_share: str, remote_path: str = "") -> Dict:
        """
        Sync files to NAS via rsync over SSH.

        Args:
            local_path: Local directory/file to sync
            remote_share: NAS share name (e.g., "backups")
            remote_path: Path within the share

        Returns:
            Dict with sync result
        """
        try:
            # Build rsync command
            remote_dest = f"{self.username}@{self.host}:/volume1/{remote_share}/{remote_path}"

            cmd = [
                "rsync", "-avz", "--progress",
                str(local_path),
                remote_dest
            ]

            logger.info(f"Syncing {local_path} to {remote_dest}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env={"RSYNC_PASSWORD": self.password}
            )

            return {
                "success": result.returncode == 0,
                "local_path": str(local_path),
                "remote_dest": remote_dest,
                "stdout": result.stdout[-500:] if result.stdout else "",
                "stderr": result.stderr[-200:] if result.stderr else "",
                "timestamp": datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Sync timeout after 1 hour"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def monitor_loop(self, interval: int = 300):
        """
        Run continuous monitoring loop.

        Args:
            interval: Seconds between checks
        """
        logger.info(f"Starting storage monitor (interval: {interval}s)")

        while True:
            try:
                status = self.get_status()
                self._save_status()

                # Log summary
                if status["connected"]:
                    logger.info(
                        f"Storage: {status['used_capacity_tb']:.1f}TB / "
                        f"{status['total_capacity_tb']:.1f}TB "
                        f"({status['usage_percent']}%) - {status['health']}"
                    )
                else:
                    logger.warning(f"Storage disconnected: {self.host}")

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Stopping storage monitor")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(60)


def load_storage_config(base_dir: str = None) -> Dict:
    """Load storage configuration"""
    if base_dir is None:
        from core.paths import get_base_dir
        base_dir = str(get_base_dir())
    config_file = Path(base_dir) / "config" / "storage.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def get_nas_path(folder_type: str, base_dir: str = None) -> str:
    """Get full NAS path for a folder type"""
    config = load_storage_config(base_dir)
    nas = config.get("nas", {})
    folders = config.get("folders", {})

    folder_config = folders.get(folder_type, {})
    folder_path = folder_config.get("path", folder_type)

    return f"/volume1/{nas.get('share', 'data')}/{nas.get('base_path', 'llm_training')}/{folder_path}"


def sync_to_nas(
    local_path: str,
    folder_type: str,
    subfolder: str = "",
    base_dir: str = None
) -> Dict:
    """
    Sync files to NAS using Synology FileStation API.

    Args:
        local_path: Local file/directory to sync
        folder_type: One of: checkpoints, models, backups, logs, training_data, snapshots, exports
        subfolder: Optional subfolder within the folder type
        base_dir: Base training directory

    Returns:
        Dict with sync result
    """
    creds = load_credentials()
    if not creds.get("password"):
        return {"success": False, "error": "No credentials"}

    config = load_storage_config(base_dir)
    nas = config.get("nas", {})
    folders = config.get("folders", {})
    folder_config = folders.get(folder_type, {})
    folder_path = folder_config.get("path", folder_type)

    # Build remote path
    share = nas.get('share', 'data')
    base_path = nas.get('base_path', 'llm_training')

    if subfolder:
        dest_folder = f"/{share}/{base_path}/{folder_path}/{subfolder}"
    else:
        dest_folder = f"/{share}/{base_path}/{folder_path}"

    local = Path(local_path)
    if not local.exists():
        return {"success": False, "error": f"Path not found: {local_path}"}

    host = creds['host']
    port = creds.get('port', 5001)

    try:
        session = requests.Session()
        session.verify = False

        # Login
        resp = session.get(f'https://{host}:{port}/webapi/entry.cgi', params={
            'api': 'SYNO.API.Auth',
            'method': 'login',
            'version': 3,
            'account': creds['username'],
            'passwd': creds['password'],
            'session': 'FileStation',
            'format': 'sid'
        }, timeout=30)
        sid = resp.json().get('data', {}).get('sid')
        if not sid:
            return {"success": False, "error": "Login failed"}

        # Create destination folder if needed
        session.get(f'https://{host}:{port}/webapi/entry.cgi', params={
            'api': 'SYNO.FileStation.CreateFolder',
            'method': 'create',
            'version': 2,
            'folder_path': '/'.join(dest_folder.rsplit('/', 1)[:-1]) or '/',
            'name': dest_folder.rsplit('/', 1)[-1],
            '_sid': sid
        }, timeout=30)

        uploaded = []
        errors = []

        # Upload files
        if local.is_file():
            files_to_upload = [local]
        else:
            files_to_upload = list(local.rglob('*'))
            files_to_upload = [f for f in files_to_upload if f.is_file()]

        for file_path in files_to_upload[:100]:  # Limit to 100 files
            try:
                # Calculate relative path for subdirs
                if local.is_dir():
                    rel_path = file_path.relative_to(local)
                    file_dest = f"{dest_folder}/{rel_path.parent}" if rel_path.parent != Path('.') else dest_folder
                else:
                    file_dest = dest_folder

                with open(file_path, 'rb') as f:
                    resp = session.post(
                        f'https://{host}:{port}/webapi/entry.cgi',
                        data={
                            'api': 'SYNO.FileStation.Upload',
                            'method': 'upload',
                            'version': 2,
                            'path': file_dest,
                            'create_parents': 'true',
                            'overwrite': 'true',
                            '_sid': sid
                        },
                        files={'file': (file_path.name, f)},
                        timeout=300
                    )

                result = resp.json()
                if result.get('success'):
                    uploaded.append(str(file_path))
                    logger.info(f"Uploaded: {file_path.name} -> {file_dest}")
                else:
                    errors.append(f"{file_path.name}: {result.get('error', {}).get('code', 'unknown')}")

            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")

        # Logout
        session.get(f'https://{host}:{port}/webapi/entry.cgi', params={
            'api': 'SYNO.API.Auth', 'method': 'logout', 'session': 'FileStation'
        })

        return {
            "success": len(errors) == 0,
            "local_path": str(local_path),
            "remote_path": dest_folder,
            "folder_type": folder_type,
            "uploaded": len(uploaded),
            "errors": errors[:5] if errors else [],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return {"success": False, "error": str(e)}


def load_credentials(creds_file: str = None) -> Dict:
    """Load credentials from file or environment"""
    import os

    # Try environment variables first
    host = os.environ.get("SYNOLOGY_HOST")
    user = os.environ.get("SYNOLOGY_USER")
    passwd = os.environ.get("SYNOLOGY_PASS")

    if host and user and passwd:
        return {"host": host, "username": user, "password": passwd}

    # Try credentials file
    if creds_file:
        creds_path = Path(creds_file)
        if creds_path.exists():
            with open(creds_path) as f:
                return json.load(f)

    # Default credentials file location
    try:
        from core.paths import get_base_dir
        default_creds = get_base_dir() / ".secrets" / "synology.json"
    except Exception:
        default_creds = Path.cwd() / ".secrets" / "synology.json"
    if default_creds.exists():
        with open(default_creds) as f:
            return json.load(f)

    return {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Synology Storage Manager")
    parser.add_argument("--host", default="192.168.x.x", help="Synology NAS IP")
    parser.add_argument("--username", default="user", help="DSM username")
    parser.add_argument("--password", help="DSM password (or use --creds-file)")
    parser.add_argument("--creds-file", help="JSON file with credentials")
    parser.add_argument("--port", type=int, default=5001, help="DSM port (default: 5001)")
    parser.add_argument("--base-dir", default=None, help="Base directory (auto-detect if not set)")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--status", action="store_true", help="Print status and exit")

    args = parser.parse_args()

    # Get credentials
    password = args.password
    if not password:
        creds = load_credentials(args.creds_file)
        password = creds.get("password")
        if not password:
            print("Error: Password required. Use --password or --creds-file")
            return 1

    mgr = StorageManager(
        host=args.host,
        username=args.username,
        password=password,
        port=args.port,
        base_dir=args.base_dir
    )

    if args.status or args.once:
        status = mgr.get_status()
        mgr._save_status()

        print(f"\nSynology NAS Status: {mgr.host}")
        print("=" * 50)
        print(f"Connected: {status['connected']} ({status['method']})")
        print(f"Health: {status['health']}")
        print(f"Capacity: {status['used_capacity_tb']:.1f}TB / {status['total_capacity_tb']:.1f}TB ({status['usage_percent']}%)")

        if status['disks']:
            print(f"\nDisks ({len(status['disks'])}):")
            for disk in status['disks']:
                print(f"  - {disk['id']}: {disk['model']} ({disk['size_tb']}TB) - {disk['status']}")

        if status['volumes']:
            print(f"\nVolumes ({len(status['volumes'])}):")
            for vol in status['volumes']:
                print(f"  - {vol['id']}: {vol.get('size_used_tb', 0):.1f}TB / {vol.get('size_total_tb', 0):.1f}TB")
    else:
        mgr.monitor_loop(args.interval)


if __name__ == "__main__":
    main()
