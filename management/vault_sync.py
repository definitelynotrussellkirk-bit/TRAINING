#!/usr/bin/env python3
"""
Vault Sync - Sync checkpoints to NAS before edge device cleanup.

The NAS is the canonical long-term archive (the "vault"). Edge devices
(trainer, inference) have limited storage and can only delete checkpoints
that exist on the NAS.

Flow:
    1. Find checkpoints on edge device not yet on NAS
    2. Rsync them to NAS
    3. Update ledger with NAS location
    4. Edge device retention can now safely delete

Usage:
    # Sync all unvaulted checkpoints from local device
    python3 -m management.vault_sync

    # Sync specific device
    python3 -m management.vault_sync --source inference3090

    # Dry run
    python3 -m management.vault_sync --dry-run

    # From code
    from management.vault_sync import sync_to_vault
    result = sync_to_vault(source_device="trainer4090")
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a vault sync operation."""
    source_device: str
    checkpoints_synced: int = 0
    checkpoints_failed: int = 0
    synced_steps: List[int] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    bytes_transferred: int = 0
    dry_run: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def gb_transferred(self) -> float:
        return self.bytes_transferred / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_device": self.source_device,
            "checkpoints_synced": self.checkpoints_synced,
            "checkpoints_failed": self.checkpoints_failed,
            "synced_steps": self.synced_steps,
            "failed_steps": self.failed_steps,
            "gb_transferred": round(self.gb_transferred, 2),
            "dry_run": self.dry_run,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat(),
        }


def get_nas_config() -> Optional[Dict[str, Any]]:
    """Get NAS configuration from hosts.json."""
    try:
        from core.paths import get_base_dir
        hosts_file = get_base_dir() / "config" / "hosts.json"
        if hosts_file.exists():
            with open(hosts_file) as f:
                data = json.load(f)
            return data.get("hosts", {}).get("nas")
    except Exception as e:
        logger.error(f"Failed to get NAS config: {e}")
    return None


def get_unvaulted_checkpoints(source_device: str) -> List[Dict[str, Any]]:
    """
    Get checkpoints on source device that aren't on NAS.

    Returns:
        List of dicts with step, path, size_bytes
    """
    from core.checkpoint_ledger import get_ledger

    ledger = get_ledger()
    records = ledger.list_by_device(source_device)

    # Filter to those without NAS copy
    unvaulted = []
    for record in records:
        # Check if synology_data (NAS device_id) is in locations
        if "synology_data" not in record.locations:
            unvaulted.append({
                "step": record.step,
                "path": record.path,
                "size_bytes": record.size_bytes or 0,
                "canonical_name": record.canonical_name,
            })

    return unvaulted


@dataclass
class TransferResult:
    """Result of a single checkpoint transfer."""
    success: bool
    bytes_transferred: int
    attempts: int
    error: Optional[str] = None
    verified: bool = False
    duration_seconds: float = 0.0


def get_remote_size(
    nas_host: str,
    nas_user: str,
    remote_path: str,
) -> Optional[int]:
    """Get total size of remote directory via SSH."""
    ssh_key = os.path.expanduser("~/.ssh/nas_key")
    cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{nas_user}@{nas_host}",
        f"du -sb '{remote_path}' 2>/dev/null | cut -f1",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception as e:
        logger.debug(f"Could not get remote size: {e}")
    return None


def get_remote_file_count(
    nas_host: str,
    nas_user: str,
    remote_path: str,
) -> Optional[int]:
    """Get count of files in remote directory via SSH."""
    ssh_key = os.path.expanduser("~/.ssh/nas_key")
    cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{nas_user}@{nas_host}",
        f"find '{remote_path}' -type f 2>/dev/null | wc -l",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception as e:
        logger.debug(f"Could not get remote file count: {e}")
    return None


def delete_remote_dir(
    nas_host: str,
    nas_user: str,
    remote_path: str,
) -> bool:
    """Delete remote directory (cleanup failed partial transfer)."""
    ssh_key = os.path.expanduser("~/.ssh/nas_key")
    cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        f"{nas_user}@{nas_host}",
        f"rm -rf '{remote_path}'",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def scp_checkpoint(
    source_path: str,
    nas_host: str,
    nas_checkpoints_dir: str,
    nas_user: str = "user",
    dry_run: bool = False,
    max_retries: int = 3,
    verify: bool = True,
) -> tuple[bool, int]:
    """
    Copy a checkpoint to NAS using SCP with retries and verification.

    Synology NAS doesn't support SFTP subsystem by default, so we use
    scp -O which uses the legacy SCP protocol over SSH.

    Features:
        - Retry with exponential backoff (3 attempts default)
        - Size verification after transfer
        - File count verification
        - Cleanup of partial transfers on failure
        - Progress reporting

    Args:
        source_path: Local path to checkpoint directory
        nas_host: NAS hostname/IP
        nas_checkpoints_dir: Remote directory on NAS
        nas_user: SSH user for NAS
        dry_run: If True, only show what would happen
        max_retries: Number of retry attempts (default: 3)
        verify: Whether to verify transfer (default: True)

    Returns:
        (success, bytes_transferred)
    """
    import time as time_module

    source = Path(source_path)
    if not source.exists():
        logger.warning(f"Source path does not exist: {source_path}")
        return False, 0

    # Calculate local size and file count
    local_files = list(source.rglob("*"))
    local_file_count = sum(1 for f in local_files if f.is_file())
    bytes_to_transfer = sum(f.stat().st_size for f in local_files if f.is_file())

    checkpoint_name = source.name
    remote_path = f"{nas_checkpoints_dir}/{checkpoint_name}"
    dest = f"{nas_user}@{nas_host}:{remote_path}"

    # Build scp command with legacy protocol (-O) and SSH key
    ssh_key = os.path.expanduser("~/.ssh/nas_key")
    cmd = [
        "scp",
        "-O",  # Legacy SCP protocol (Synology compatibility)
        "-r",  # Recursive
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=3",
        source_path,
        dest,
    ]

    logger.info(
        f"Syncing {checkpoint_name} to NAS "
        f"({bytes_to_transfer / 1e9:.2f} GB, {local_file_count} files)..."
    )

    if dry_run:
        logger.info(f"[DRY RUN] Would copy {source_path} to {dest}")
        return True, bytes_to_transfer

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(1, max_retries + 1):
        start_time = time_module.time()

        try:
            # Clean up any partial transfer from previous attempt
            if attempt > 1:
                logger.info(f"  Cleaning up partial transfer before retry...")
                delete_remote_dir(nas_host, nas_user, remote_path)
                time_module.sleep(2)

            logger.info(f"  Attempt {attempt}/{max_retries}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for large checkpoints
            )

            duration = time_module.time() - start_time

            if result.returncode == 0:
                # Verify transfer if requested
                if verify:
                    logger.info(f"  Verifying transfer...")

                    # Check file count
                    remote_file_count = get_remote_file_count(nas_host, nas_user, remote_path)
                    if remote_file_count is not None:
                        if remote_file_count != local_file_count:
                            logger.warning(
                                f"  File count mismatch: local={local_file_count}, "
                                f"remote={remote_file_count}"
                            )
                            last_error = f"File count mismatch: {local_file_count} vs {remote_file_count}"
                            if attempt < max_retries:
                                backoff = 2 ** attempt
                                logger.info(f"  Retrying in {backoff}s...")
                                time_module.sleep(backoff)
                                continue
                            else:
                                return False, 0

                    # Check total size (with 1% tolerance for filesystem differences)
                    remote_size = get_remote_size(nas_host, nas_user, remote_path)
                    if remote_size is not None:
                        size_diff_pct = abs(remote_size - bytes_to_transfer) / bytes_to_transfer * 100
                        if size_diff_pct > 1.0:  # Allow 1% variance
                            logger.warning(
                                f"  Size mismatch: local={bytes_to_transfer}, "
                                f"remote={remote_size} ({size_diff_pct:.1f}% diff)"
                            )
                            last_error = f"Size mismatch: {size_diff_pct:.1f}% difference"
                            if attempt < max_retries:
                                backoff = 2 ** attempt
                                logger.info(f"  Retrying in {backoff}s...")
                                time_module.sleep(backoff)
                                continue
                            else:
                                return False, 0

                    logger.info(f"  ✓ Verified: {local_file_count} files, sizes match")

                speed_mbps = (bytes_to_transfer / 1e6) / duration if duration > 0 else 0
                logger.info(
                    f"✓ Synced {checkpoint_name} in {duration:.1f}s "
                    f"({speed_mbps:.1f} MB/s)"
                )
                return True, bytes_to_transfer

            else:
                last_error = result.stderr.strip() or "Unknown error"
                logger.warning(f"  Attempt {attempt} failed: {last_error}")

        except subprocess.TimeoutExpired:
            last_error = "Transfer timed out (2 hours)"
            logger.warning(f"  Attempt {attempt} timed out")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"  Attempt {attempt} error: {e}")

        # Exponential backoff before retry
        if attempt < max_retries:
            backoff = 2 ** attempt
            logger.info(f"  Retrying in {backoff}s...")
            time_module.sleep(backoff)

    # All attempts failed
    logger.error(f"✗ Failed to sync {checkpoint_name} after {max_retries} attempts: {last_error}")

    # Clean up partial transfer
    delete_remote_dir(nas_host, nas_user, remote_path)

    return False, 0


# Alias for backwards compatibility
rsync_checkpoint = scp_checkpoint


def sync_to_vault(
    source_device: Optional[str] = None,
    dry_run: bool = False,
    max_count: Optional[int] = None,
) -> SyncResult:
    """
    Sync unvaulted checkpoints from source device to NAS.

    Args:
        source_device: Device ID to sync from (default: local device)
        dry_run: If True, only show what would happen
        max_count: Maximum number of checkpoints to sync (for testing)

    Returns:
        SyncResult with details
    """
    from core.checkpoint_ledger import get_ledger

    # Determine source device
    if source_device is None:
        from management.device_retention import get_local_device_id
        source_device = get_local_device_id()

    result = SyncResult(source_device=source_device, dry_run=dry_run)

    # Get NAS config
    nas_config = get_nas_config()
    if not nas_config:
        result.errors.append("NAS not configured in hosts.json")
        logger.error("NAS not configured")
        return result

    nas_host = nas_config.get("host")
    nas_user = nas_config.get("ssh_user", "admin")
    nas_checkpoints_dir = nas_config.get("checkpoints_dir", "/volume1/data/llm_training/checkpoints")

    if not nas_host:
        result.errors.append("NAS host not configured")
        return result

    # Get checkpoints that need syncing
    unvaulted = get_unvaulted_checkpoints(source_device)

    if not unvaulted:
        logger.info(f"All checkpoints on {source_device} are already vaulted")
        return result

    logger.info(f"Found {len(unvaulted)} unvaulted checkpoints on {source_device}")

    # Limit for testing
    if max_count:
        unvaulted = unvaulted[:max_count]
        logger.info(f"Limiting to {max_count} checkpoints")

    # Get ledger for updating locations
    ledger = get_ledger()

    # Sync each checkpoint
    for checkpoint in unvaulted:
        step = checkpoint["step"]
        path = checkpoint["path"]

        success, bytes_transferred = rsync_checkpoint(
            source_path=path,
            nas_host=nas_host,
            nas_checkpoints_dir=nas_checkpoints_dir,
            nas_user=nas_user,
            dry_run=dry_run,
        )

        if success:
            result.synced_steps.append(step)
            result.checkpoints_synced += 1
            result.bytes_transferred += bytes_transferred

            # Update ledger with NAS location
            if not dry_run:
                ledger.record_usage(step, "synology_data")
                logger.debug(f"Added synology_data to locations for step {step}")
        else:
            result.failed_steps.append(step)
            result.checkpoints_failed += 1

    # Summary
    if result.checkpoints_synced > 0:
        logger.info(
            f"Synced {result.checkpoints_synced} checkpoints, "
            f"{result.gb_transferred:.2f} GB"
        )

    if result.checkpoints_failed > 0:
        logger.warning(f"Failed to sync {result.checkpoints_failed} checkpoints")

    return result


def check_nas_connectivity() -> bool:
    """Check if NAS is reachable via SSH."""
    nas_config = get_nas_config()
    if not nas_config:
        return False

    nas_host = nas_config.get("host")
    nas_user = nas_config.get("ssh_user", "user")
    ssh_key = os.path.expanduser("~/.ssh/nas_key")

    try:
        result = subprocess.run(
            ["ssh", "-i", ssh_key, "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             f"{nas_user}@{nas_host}", "echo", "ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


def bootstrap_device_locations(
    device_id: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Bootstrap locations in ledger by scanning what's on disk.

    For checkpoints that exist on disk but don't have this device in their
    locations list, add the device.

    Args:
        device_id: Device ID to add (default: local device)
        checkpoints_dir: Directory to scan (default: from hosts.json)
        dry_run: If True, only show what would happen

    Returns:
        Dict with counts
    """
    from core.checkpoint_ledger import get_ledger, extract_step
    from pathlib import Path

    # Determine device
    if device_id is None:
        from management.device_retention import get_local_device_id
        device_id = get_local_device_id()

    # Get checkpoints directory
    if checkpoints_dir is None:
        from management.device_retention import get_checkpoints_dir
        checkpoints_dir = get_checkpoints_dir(device_id)

    if not checkpoints_dir:
        logger.error(f"No checkpoints directory for {device_id}")
        return {"error": "No checkpoints directory"}

    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        logger.error(f"Directory does not exist: {checkpoints_path}")
        return {"error": f"Directory does not exist: {checkpoints_path}"}

    ledger = get_ledger()
    result = {
        "device_id": device_id,
        "scanned": 0,
        "added": 0,
        "already_present": 0,
        "not_in_ledger": 0,
        "dry_run": dry_run,
    }

    # Scan checkpoints on disk
    for item in checkpoints_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            result["scanned"] += 1
            step = extract_step(item.name)

            if step == 0:
                continue

            record = ledger.get(step)
            if not record:
                result["not_in_ledger"] += 1
                continue

            if device_id in record.locations:
                result["already_present"] += 1
            else:
                result["added"] += 1
                if not dry_run:
                    ledger.record_usage(step, device_id)
                    logger.debug(f"Added {device_id} to step {step}")

    logger.info(
        f"Bootstrap {device_id}: scanned={result['scanned']}, "
        f"added={result['added']}, already={result['already_present']}"
    )

    return result


def verify_vaulted_checkpoint(
    step: int,
    nas_host: str,
    nas_checkpoints_dir: str,
    nas_user: str = "user",
) -> Dict[str, Any]:
    """
    Verify a vaulted checkpoint matches the local copy.

    Args:
        step: Checkpoint step number
        nas_host: NAS hostname
        nas_checkpoints_dir: Remote checkpoints directory
        nas_user: SSH user

    Returns:
        Dict with verification results
    """
    from core.checkpoint_ledger import get_ledger

    ledger = get_ledger()
    record = ledger.get(step)

    result = {
        "step": step,
        "verified": False,
        "local_exists": False,
        "remote_exists": False,
        "local_size": 0,
        "remote_size": 0,
        "local_files": 0,
        "remote_files": 0,
        "error": None,
    }

    if not record:
        result["error"] = f"Step {step} not in ledger"
        return result

    local_path = Path(record.path)
    if not local_path.exists():
        result["error"] = f"Local path does not exist: {local_path}"
        return result

    result["local_exists"] = True

    # Get local stats
    local_files = list(local_path.rglob("*"))
    result["local_files"] = sum(1 for f in local_files if f.is_file())
    result["local_size"] = sum(f.stat().st_size for f in local_files if f.is_file())

    # Get remote stats
    remote_path = f"{nas_checkpoints_dir}/{local_path.name}"

    remote_files = get_remote_file_count(nas_host, nas_user, remote_path)
    remote_size = get_remote_size(nas_host, nas_user, remote_path)

    if remote_files is not None:
        result["remote_exists"] = True
        result["remote_files"] = remote_files
    if remote_size is not None:
        result["remote_size"] = remote_size

    if not result["remote_exists"]:
        result["error"] = "Remote checkpoint not found"
        return result

    # Compare
    files_match = result["local_files"] == result["remote_files"]
    size_diff_pct = abs(result["remote_size"] - result["local_size"]) / result["local_size"] * 100 if result["local_size"] > 0 else 0
    sizes_match = size_diff_pct < 1.0  # 1% tolerance

    if files_match and sizes_match:
        result["verified"] = True
    else:
        issues = []
        if not files_match:
            issues.append(f"files: local={result['local_files']} vs remote={result['remote_files']}")
        if not sizes_match:
            issues.append(f"size: {size_diff_pct:.1f}% difference")
        result["error"] = "; ".join(issues)

    return result


def verify_all_vaulted(dry_run: bool = False) -> Dict[str, Any]:
    """
    Verify all vaulted checkpoints match local copies.

    Returns:
        Dict with verification summary
    """
    from core.checkpoint_ledger import get_ledger
    from management.device_retention import get_local_device_id

    nas_config = get_nas_config()
    if not nas_config:
        return {"error": "NAS not configured"}

    nas_host = nas_config.get("host")
    nas_user = nas_config.get("ssh_user", "user")
    nas_checkpoints_dir = nas_config.get("checkpoints_dir", "/volume1/data/llm_training/checkpoints")

    ledger = get_ledger()
    local_device = get_local_device_id()

    # Get local checkpoints that claim to be vaulted
    all_records = ledger.list_all()
    vaulted = [
        r for r in all_records
        if local_device in r.locations and "synology_data" in r.locations
    ]

    results = {
        "total_vaulted": len(vaulted),
        "verified": 0,
        "failed": 0,
        "errors": [],
    }

    for record in vaulted:
        verification = verify_vaulted_checkpoint(
            step=record.step,
            nas_host=nas_host,
            nas_checkpoints_dir=nas_checkpoints_dir,
            nas_user=nas_user,
        )

        if verification["verified"]:
            results["verified"] += 1
            logger.info(f"✓ Step {record.step}: verified")
        else:
            results["failed"] += 1
            results["errors"].append({
                "step": record.step,
                "error": verification["error"],
            })
            logger.warning(f"✗ Step {record.step}: {verification['error']}")

    return results


def get_vault_status() -> Dict[str, Any]:
    """Get current vault sync status."""
    from core.checkpoint_ledger import get_ledger
    from management.device_retention import get_local_device_id

    ledger = get_ledger()
    local_device = get_local_device_id()

    # Count checkpoints by location status
    all_records = ledger.list_all()
    local_records = [r for r in all_records if local_device in r.locations]

    vaulted = [r for r in local_records if "synology_data" in r.locations]
    unvaulted = [r for r in local_records if "synology_data" not in r.locations]

    return {
        "local_device": local_device,
        "total_local": len(local_records),
        "vaulted": len(vaulted),
        "unvaulted": len(unvaulted),
        "nas_reachable": check_nas_connectivity(),
        "unvaulted_steps": [r.step for r in unvaulted],
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Sync checkpoints to NAS vault")
    parser.add_argument("--source", type=str, help="Source device ID (default: local)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--max-count", type=int, help="Max checkpoints to sync")
    parser.add_argument("--status", action="store_true", help="Show vault status")
    parser.add_argument("--check-nas", action="store_true", help="Check NAS connectivity")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap locations from disk")
    parser.add_argument("--verify", action="store_true", help="Verify all vaulted checkpoints")
    parser.add_argument("--verify-step", type=int, help="Verify a specific checkpoint step")

    args = parser.parse_args()

    if args.check_nas:
        if check_nas_connectivity():
            print("✓ NAS is reachable")
            sys.exit(0)
        else:
            print("✗ NAS is not reachable")
            sys.exit(1)

    if args.bootstrap:
        result = bootstrap_device_locations(dry_run=args.dry_run)
        print(f"\nBootstrap Results for {result.get('device_id', 'unknown')}:")
        print(f"  Scanned: {result.get('scanned', 0)}")
        print(f"  Added: {result.get('added', 0)}")
        print(f"  Already present: {result.get('already_present', 0)}")
        print(f"  Not in ledger: {result.get('not_in_ledger', 0)}")
        if args.dry_run:
            print("  (Dry run - no changes made)")
        sys.exit(0)

    if args.status:
        status = get_vault_status()
        print(f"\nVault Status for {status['local_device']}:")
        print(f"  Total local checkpoints: {status['total_local']}")
        print(f"  Vaulted (on NAS): {status['vaulted']}")
        print(f"  Unvaulted: {status['unvaulted']}")
        print(f"  NAS reachable: {'Yes' if status['nas_reachable'] else 'No'}")
        if status['unvaulted_steps']:
            print(f"\n  Unvaulted steps: {status['unvaulted_steps'][:10]}...")
        sys.exit(0)

    if args.verify:
        print("\nVerifying all vaulted checkpoints...")
        results = verify_all_vaulted()
        print(f"\nVerification Results:")
        print(f"  Total vaulted: {results['total_vaulted']}")
        print(f"  Verified: {results['verified']}")
        print(f"  Failed: {results['failed']}")
        if results.get('errors'):
            print(f"\nFailed checkpoints:")
            for err in results['errors'][:10]:
                print(f"  Step {err['step']}: {err['error']}")
        sys.exit(0 if results['failed'] == 0 else 1)

    if args.verify_step:
        nas_config = get_nas_config()
        if not nas_config:
            print("✗ NAS not configured")
            sys.exit(1)
        result = verify_vaulted_checkpoint(
            step=args.verify_step,
            nas_host=nas_config["host"],
            nas_checkpoints_dir=nas_config.get("checkpoints_dir", "/volume1/data/llm_training/checkpoints"),
            nas_user=nas_config.get("ssh_user", "user"),
        )
        print(json.dumps(result, indent=2))
        sys.exit(0 if result['verified'] else 1)

    result = sync_to_vault(
        source_device=args.source,
        dry_run=args.dry_run,
        max_count=args.max_count,
    )

    print(json.dumps(result.to_dict(), indent=2))
