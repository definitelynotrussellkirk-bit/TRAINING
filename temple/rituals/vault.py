"""
Vault Ritual - Tests for the unified Vault system.

This ritual verifies that the Ledger and VaultKeeper are properly synchronized:
- Device mapping configuration is valid
- Ledger is accessible and has checkpoints
- VaultKeeper catalog is accessible
- Ledger and VaultKeeper are in sync
- New checkpoint sync works correctly
"""

from datetime import datetime
from pathlib import Path
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("vault", "Ritual of the Vault", "Verify Ledger-VaultKeeper unification and remote ops")
def run() -> List[RitualCheckResult]:
    """Execute all vault ritual checks."""
    results = []
    # Core vault checks
    results.append(_check_device_mapping())
    results.append(_check_ledger_access())
    results.append(_check_vault_catalog())
    results.append(_check_ledger_vault_sync())
    results.append(_check_bidirectional_mapping())
    results.append(_check_latest_checkpoint_synced())
    # Remote operations checks
    results.append(_check_remote_ssh_access())
    results.append(_check_remote_disk_space())
    results.append(_check_remote_write_delete())
    results.append(_check_sync_speed())
    return results


def _check_device_mapping() -> RitualCheckResult:
    """Check that device mapping configuration is valid and loaded."""
    start = datetime.utcnow()
    try:
        from vault.device_mapping import get_mapping

        mapping = get_mapping()
        devices = mapping.list_devices()
        strongholds = mapping.list_strongholds()

        if not devices:
            return RitualCheckResult(
                id="vault_device_mapping",
                name="Device mapping loaded",
                description="Check that device_mapping.json is valid and loaded",
                status="fail",
                category="vault",
                details={"error": "No devices configured"},
                remediation="Create config/device_mapping.json with device mappings",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        local_device = mapping.get_local_device_id()
        local_stronghold = mapping.get_local_stronghold()

        return RitualCheckResult(
            id="vault_device_mapping",
            name="Device mapping loaded",
            description="Check that device_mapping.json is valid and loaded",
            status="ok",
            category="vault",
            details={
                "devices": devices,
                "strongholds": strongholds,
                "local_device": local_device,
                "local_stronghold": local_stronghold,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_device_mapping",
            name="Device mapping loaded",
            description="Check that device_mapping.json is valid and loaded",
            status="fail",
            category="vault",
            details={"error": str(e)},
            remediation="Check config/device_mapping.json exists and is valid JSON",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_ledger_access() -> RitualCheckResult:
    """Check that the Checkpoint Ledger is accessible."""
    start = datetime.utcnow()
    try:
        from core.checkpoint_ledger import get_ledger

        ledger = get_ledger()
        all_records = ledger.list_all()
        count = len(all_records)

        if count == 0:
            return RitualCheckResult(
                id="vault_ledger_access",
                name="Checkpoint Ledger accessible",
                description="Check that checkpoint_ledger.json is accessible and has data",
                status="warn",
                category="vault",
                details={"checkpoint_count": 0},
                remediation="No checkpoints in ledger - training may not have started yet",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        latest = ledger.get_latest()

        return RitualCheckResult(
            id="vault_ledger_access",
            name="Checkpoint Ledger accessible",
            description="Check that checkpoint_ledger.json is accessible and has data",
            status="ok",
            category="vault",
            details={
                "checkpoint_count": count,
                "latest_step": latest.step if latest else None,
                "latest_canonical": latest.canonical_name if latest else None,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_ledger_access",
            name="Checkpoint Ledger accessible",
            description="Check that checkpoint_ledger.json is accessible and has data",
            status="fail",
            category="vault",
            details={"error": str(e)},
            remediation="Check status/checkpoint_ledger.json exists and is valid",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_vault_catalog() -> RitualCheckResult:
    """Check that the VaultKeeper catalog is accessible."""
    start = datetime.utcnow()
    try:
        from vault.keeper import get_vault_keeper
        from vault.assets import AssetType, AssetQuery

        keeper = get_vault_keeper()

        # Count checkpoints in catalog
        checkpoints = list(keeper.search(AssetQuery(asset_type=AssetType.CHECKPOINT)))
        count = len(checkpoints)

        # Check catalog file exists
        catalog_exists = keeper.catalog_path.exists()

        return RitualCheckResult(
            id="vault_catalog_access",
            name="VaultKeeper catalog accessible",
            description="Check that vault/catalog.db is accessible",
            status="ok",
            category="vault",
            details={
                "catalog_path": str(keeper.catalog_path),
                "catalog_exists": catalog_exists,
                "checkpoint_count": count,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_catalog_access",
            name="VaultKeeper catalog accessible",
            description="Check that vault/catalog.db is accessible",
            status="fail",
            category="vault",
            details={"error": str(e)},
            remediation="Check vault/catalog.db exists - VaultKeeper may need restart",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_ledger_vault_sync() -> RitualCheckResult:
    """Check that Ledger and VaultKeeper are in sync."""
    start = datetime.utcnow()
    try:
        from core.checkpoint_ledger import get_ledger
        from vault.keeper import get_vault_keeper
        from vault.assets import AssetType, AssetQuery

        ledger = get_ledger()
        keeper = get_vault_keeper()

        # Get counts
        ledger_count = len(ledger.list_all())
        vault_checkpoints = list(keeper.search(AssetQuery(asset_type=AssetType.CHECKPOINT)))
        vault_count = len(vault_checkpoints)

        # Check if latest ledger entry is in vault
        latest = ledger.get_latest()
        latest_in_vault = False
        if latest:
            asset_id = f"checkpoint_{latest.step}"
            asset = keeper.get(asset_id)
            latest_in_vault = asset is not None

        # Calculate sync ratio
        sync_ratio = vault_count / ledger_count if ledger_count > 0 else 0

        # Determine status
        if latest_in_vault and sync_ratio >= 0.3:
            status = "ok"
            remediation = None
        elif latest_in_vault:
            status = "warn"
            remediation = "Many older checkpoints missing from catalog - consider running full sync"
        else:
            status = "fail"
            remediation = "Latest checkpoint not in VaultKeeper - sync may be broken"

        return RitualCheckResult(
            id="vault_ledger_sync",
            name="Ledger-VaultKeeper sync",
            description="Check that Ledger and VaultKeeper are synchronized",
            status=status,
            category="vault",
            details={
                "ledger_count": ledger_count,
                "vault_count": vault_count,
                "sync_ratio": round(sync_ratio, 2),
                "latest_step": latest.step if latest else None,
                "latest_in_vault": latest_in_vault,
            },
            remediation=remediation,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_ledger_sync",
            name="Ledger-VaultKeeper sync",
            description="Check that Ledger and VaultKeeper are synchronized",
            status="fail",
            category="vault",
            details={"error": str(e)},
            remediation="Check both Ledger and VaultKeeper are accessible",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_bidirectional_mapping() -> RitualCheckResult:
    """Check that device-stronghold mapping is bidirectional."""
    start = datetime.utcnow()
    try:
        from vault.device_mapping import get_mapping

        mapping = get_mapping()
        devices = mapping.list_devices()

        failed_mappings = []
        successful_mappings = []

        for device_id in devices:
            try:
                stronghold = mapping.device_to_stronghold(device_id)
                back = mapping.stronghold_to_device(stronghold)
                if back == device_id:
                    successful_mappings.append(f"{device_id} <-> {stronghold}")
                else:
                    failed_mappings.append(f"{device_id} -> {stronghold} -> {back} (mismatch)")
            except KeyError as e:
                failed_mappings.append(f"{device_id}: {e}")

        if failed_mappings:
            return RitualCheckResult(
                id="vault_bidirectional_mapping",
                name="Bidirectional device mapping",
                description="Check that device-stronghold mappings are reversible",
                status="fail",
                category="vault",
                details={
                    "successful": successful_mappings,
                    "failed": failed_mappings,
                },
                remediation="Fix device_mapping.json to ensure all mappings are bidirectional",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        return RitualCheckResult(
            id="vault_bidirectional_mapping",
            name="Bidirectional device mapping",
            description="Check that device-stronghold mappings are reversible",
            status="ok",
            category="vault",
            details={
                "mappings": successful_mappings,
                "count": len(successful_mappings),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_bidirectional_mapping",
            name="Bidirectional device mapping",
            description="Check that device-stronghold mappings are reversible",
            status="fail",
            category="vault",
            details={"error": str(e)},
            remediation="Check device_mapping.json configuration",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_latest_checkpoint_synced() -> RitualCheckResult:
    """Check that the latest checkpoint has matching locations in both systems."""
    start = datetime.utcnow()
    try:
        from core.checkpoint_ledger import get_ledger
        from vault.keeper import get_vault_keeper
        from vault.device_mapping import get_mapping

        ledger = get_ledger()
        keeper = get_vault_keeper()
        mapping = get_mapping()

        latest = ledger.get_latest()
        if not latest:
            return RitualCheckResult(
                id="vault_latest_synced",
                name="Latest checkpoint synced",
                description="Check that latest checkpoint has matching locations",
                status="skip",
                category="vault",
                details={"reason": "No checkpoints in ledger"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Get ledger locations (device_ids)
        ledger_locations = latest.locations

        # Get vault locations (strongholds)
        asset_id = f"checkpoint_{latest.step}"
        asset = keeper.get(asset_id)

        if not asset:
            return RitualCheckResult(
                id="vault_latest_synced",
                name="Latest checkpoint synced",
                description="Check that latest checkpoint has matching locations",
                status="fail",
                category="vault",
                details={
                    "step": latest.step,
                    "ledger_locations": ledger_locations,
                    "vault_asset": None,
                },
                remediation="Latest checkpoint not in VaultKeeper - restart VaultKeeper to trigger sync",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        vault_strongholds = [loc.stronghold for loc in asset.locations]

        # Convert ledger device_ids to strongholds for comparison
        expected_strongholds = []
        for device_id in ledger_locations:
            try:
                expected_strongholds.append(mapping.device_to_stronghold(device_id))
            except KeyError:
                pass

        # Check if locations match
        missing_in_vault = set(expected_strongholds) - set(vault_strongholds)
        extra_in_vault = set(vault_strongholds) - set(expected_strongholds)

        if missing_in_vault or extra_in_vault:
            status = "warn"
            remediation = "Location mismatch - sync may need to catch up"
        else:
            status = "ok"
            remediation = None

        return RitualCheckResult(
            id="vault_latest_synced",
            name="Latest checkpoint synced",
            description="Check that latest checkpoint has matching locations",
            status=status,
            category="vault",
            details={
                "step": latest.step,
                "ledger_devices": ledger_locations,
                "vault_strongholds": vault_strongholds,
                "expected_strongholds": expected_strongholds,
                "missing_in_vault": list(missing_in_vault) if missing_in_vault else None,
                "extra_in_vault": list(extra_in_vault) if extra_in_vault else None,
            },
            remediation=remediation,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_latest_synced",
            name="Latest checkpoint synced",
            description="Check that latest checkpoint has matching locations",
            status="fail",
            category="vault",
            details={"error": str(e)},
            remediation="Check Ledger, VaultKeeper, and device mapping configuration",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


# =============================================================================
# REMOTE OPERATIONS CHECKS
# =============================================================================


def _get_remote_hosts():
    """Get remote host configurations from hosts.json."""
    try:
        from core.hosts import get_host
        hosts = {}
        for host_id in ["3090", "nas"]:
            host = get_host(host_id)
            if host:
                hosts[host_id] = {
                    "host": host.host,
                    "user": host.ssh_user or "russ",
                    "name": host.name,
                }
        return hosts
    except Exception:
        return {}


def _ssh_command(host: str, user: str, cmd: str, timeout: int = 10):
    """Execute SSH command and return (success, output)."""
    import subprocess
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}", cmd],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _check_remote_ssh_access() -> RitualCheckResult:
    """Check SSH access to remote hosts (3090, NAS)."""
    start = datetime.utcnow()
    try:
        hosts = _get_remote_hosts()
        if not hosts:
            return RitualCheckResult(
                id="vault_remote_ssh",
                name="Remote SSH access",
                description="Check SSH connectivity to remote hosts",
                status="skip",
                category="network",
                details={"reason": "No remote hosts configured in hosts.json"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        results = {}
        all_ok = True

        for host_id, config in hosts.items():
            success, output = _ssh_command(config["host"], config["user"], "echo ok")
            results[host_id] = {
                "host": config["host"],
                "name": config["name"],
                "connected": success,
                "output": output if not success else "ok",
            }
            if not success:
                all_ok = False

        return RitualCheckResult(
            id="vault_remote_ssh",
            name="Remote SSH access",
            description="Check SSH connectivity to remote hosts",
            status="ok" if all_ok else "fail",
            category="network",
            details={"hosts": results},
            remediation="Check SSH keys and network connectivity" if not all_ok else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_remote_ssh",
            name="Remote SSH access",
            description="Check SSH connectivity to remote hosts",
            status="fail",
            category="network",
            details={"error": str(e)},
            remediation="Check hosts.json configuration and SSH setup",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_remote_disk_space() -> RitualCheckResult:
    """Check disk space on remote hosts."""
    start = datetime.utcnow()
    try:
        hosts = _get_remote_hosts()
        if not hosts:
            return RitualCheckResult(
                id="vault_remote_disk",
                name="Remote disk space",
                description="Check available disk space on remote hosts",
                status="skip",
                category="storage",
                details={"reason": "No remote hosts configured"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        results = {}
        warnings = []

        for host_id, config in hosts.items():
            # Get disk usage for relevant paths
            cmd = "df -h / 2>/dev/null | tail -1 | awk '{print $4, $5}'"
            success, output = _ssh_command(config["host"], config["user"], cmd)

            if success and output:
                parts = output.split()
                if len(parts) >= 2:
                    free = parts[0]
                    used_pct = int(parts[1].rstrip('%')) if parts[1].rstrip('%').isdigit() else 0
                    results[host_id] = {
                        "host": config["host"],
                        "free": free,
                        "used_percent": used_pct,
                    }
                    if used_pct > 90:
                        warnings.append(f"{host_id}: {used_pct}% full")
                else:
                    results[host_id] = {"host": config["host"], "error": output}
            else:
                results[host_id] = {"host": config["host"], "error": output or "failed"}

        status = "warn" if warnings else "ok"

        return RitualCheckResult(
            id="vault_remote_disk",
            name="Remote disk space",
            description="Check available disk space on remote hosts",
            status=status,
            category="storage",
            details={"hosts": results, "warnings": warnings if warnings else None},
            remediation="Free up disk space on remote hosts" if warnings else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_remote_disk",
            name="Remote disk space",
            description="Check available disk space on remote hosts",
            status="fail",
            category="storage",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_remote_write_delete() -> RitualCheckResult:
    """Check that we can write and delete files on remote hosts."""
    start = datetime.utcnow()
    try:
        hosts = _get_remote_hosts()
        if not hosts:
            return RitualCheckResult(
                id="vault_remote_write_delete",
                name="Remote write/delete",
                description="Check write and delete permissions on remote hosts",
                status="skip",
                category="storage",
                details={"reason": "No remote hosts configured"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        import time
        results = {}
        all_ok = True
        # Use timestamp for unique filename
        test_file = f"/tmp/.vault_ritual_test_{int(time.time())}"

        for host_id, config in hosts.items():
            host = config["host"]
            user = config["user"]

            # Create test file
            create_ok, create_out = _ssh_command(host, user, f"echo test > {test_file} && echo created")

            # Verify it exists
            exists_ok, exists_out = _ssh_command(host, user, f"test -f {test_file} && echo exists")
            file_exists = exists_out == "exists"

            # Delete it
            delete_ok, _ = _ssh_command(host, user, f"rm -f {test_file}")

            # Verify it's gone
            gone_ok, gone_out = _ssh_command(host, user, f"test -f {test_file} && echo exists || echo gone")
            really_gone = gone_out == "gone"

            ok = create_ok and file_exists and delete_ok and really_gone
            results[host_id] = {
                "host": host,
                "create": create_ok,
                "exists_after_create": file_exists,
                "delete": delete_ok,
                "gone_after_delete": really_gone,
                "ok": ok,
            }
            if not ok:
                all_ok = False

        return RitualCheckResult(
            id="vault_remote_write_delete",
            name="Remote write/delete",
            description="Check write and delete permissions on remote hosts",
            status="ok" if all_ok else "fail",
            category="storage",
            details={"hosts": results},
            remediation="Check file permissions on remote hosts" if not all_ok else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_remote_write_delete",
            name="Remote write/delete",
            description="Check write and delete permissions on remote hosts",
            status="fail",
            category="storage",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_sync_speed() -> RitualCheckResult:
    """Check sync speed to remote hosts by transferring a test file."""
    start = datetime.utcnow()
    try:
        import subprocess
        import tempfile
        import time
        import os

        hosts = _get_remote_hosts()
        # Only test 3090 for speed (NAS is expected to be slower)
        if "3090" not in hosts:
            return RitualCheckResult(
                id="vault_sync_speed",
                name="Sync speed test",
                description="Test rsync transfer speed to inference server",
                status="skip",
                category="network",
                details={"reason": "Inference server (3090) not configured"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        config = hosts["3090"]
        host = config["host"]
        user = config["user"]

        # Create 10MB test file
        test_size_mb = 10
        test_size_bytes = test_size_mb * 1024 * 1024

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(os.urandom(test_size_bytes))
            local_file = f.name

        remote_file = "/tmp/.vault_speed_test"

        try:
            # Time the transfer
            transfer_start = time.time()

            result = subprocess.run(
                ["rsync", "-avz", "--progress",
                 local_file, f"{user}@{host}:{remote_file}"],
                capture_output=True,
                text=True,
                timeout=60
            )

            transfer_time = time.time() - transfer_start

            # Cleanup remote
            _ssh_command(host, user, f"rm -f {remote_file}")

            if result.returncode != 0:
                return RitualCheckResult(
                    id="vault_sync_speed",
                    name="Sync speed test",
                    description="Test rsync transfer speed to inference server",
                    status="fail",
                    category="network",
                    details={
                        "error": result.stderr,
                        "host": host,
                    },
                    remediation="Check rsync connectivity to inference server",
                    started_at=start,
                    finished_at=datetime.utcnow(),
                )

            # Calculate speed
            speed_mbps = test_size_mb / transfer_time

            # Determine status based on speed
            if speed_mbps >= 50:
                status = "ok"
                remediation = None
            elif speed_mbps >= 10:
                status = "warn"
                remediation = f"Transfer speed ({speed_mbps:.1f} MB/s) is below optimal"
            else:
                status = "fail"
                remediation = f"Transfer speed ({speed_mbps:.1f} MB/s) is very slow"

            return RitualCheckResult(
                id="vault_sync_speed",
                name="Sync speed test",
                description="Test rsync transfer speed to inference server",
                status=status,
                category="network",
                details={
                    "host": host,
                    "test_size_mb": test_size_mb,
                    "transfer_time_sec": round(transfer_time, 2),
                    "speed_mbps": round(speed_mbps, 1),
                    "speed_rating": "fast" if speed_mbps >= 50 else "moderate" if speed_mbps >= 10 else "slow",
                },
                remediation=remediation,
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        finally:
            # Cleanup local
            os.unlink(local_file)

    except subprocess.TimeoutExpired:
        return RitualCheckResult(
            id="vault_sync_speed",
            name="Sync speed test",
            description="Test rsync transfer speed to inference server",
            status="fail",
            category="network",
            details={"error": "Transfer timed out after 60 seconds"},
            remediation="Network connection to inference server is too slow or blocked",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="vault_sync_speed",
            name="Sync speed test",
            description="Test rsync transfer speed to inference server",
            status="fail",
            category="network",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
