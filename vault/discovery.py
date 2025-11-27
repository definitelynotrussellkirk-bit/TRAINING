"""
VaultKeeper Discovery - Auto-discover and register assets.

The Discovery service scans directories to find and register assets:
    - Checkpoints in models/ and current_model/
    - Training data in data/ and inbox/
    - Configurations
    - Backups and snapshots

It also provides the "ask vault first" pattern - before loading anything,
check if the VaultKeeper knows where it is and get it from the best source.

Usage:
    from vault.discovery import VaultDiscovery, ask_vault_first

    # Scan and register all assets
    discovery = VaultDiscovery()
    discovery.scan_all()

    # Ask vault before loading
    checkpoint_path = ask_vault_first("checkpoint_175000", fallback="/models/checkpoint-175000")

RPG Flavor:
    The Discovery service is the Seeker - a magical entity that wanders
    the realm cataloging all treasures. When deployed, it leaves no
    stone unturned, finding and recording every artifact.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vault.keeper import VaultKeeper, get_vault_keeper
from vault.assets import Asset, AssetType, AssetLocation, LocationStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vault_discovery")


class VaultDiscovery:
    """
    Discovery service for finding and registering assets.

    Scans configured directories and registers found assets
    with the VaultKeeper.
    """

    # Patterns for identifying assets
    CHECKPOINT_PATTERN = re.compile(r"checkpoint[-_](\d+)")
    MODEL_DIRS = ["models", "current_model", "current_model_small"]
    DATA_DIRS = ["data", "inbox", "queue"]
    BACKUP_DIRS = ["backups"]

    def __init__(
        self,
        keeper: Optional[VaultKeeper] = None,
        base_dir: str | Path = "/path/to/training",
    ):
        """
        Initialize the discovery service.

        Args:
            keeper: VaultKeeper instance (or create new)
            base_dir: Base training directory
        """
        self.base_dir = Path(base_dir)
        self.keeper = keeper or get_vault_keeper(base_dir)

    def scan_all(self) -> Dict[str, int]:
        """
        Scan all known directories for assets.

        Returns:
            Dict mapping asset_type -> count registered
        """
        counts = {
            "checkpoints": 0,
            "models": 0,
            "training_data": 0,
            "validation_data": 0,
            "configs": 0,
            "backups": 0,
        }

        logger.info(f"Starting full discovery scan in {self.base_dir}")

        # Scan for checkpoints
        counts["checkpoints"] = self.scan_checkpoints()

        # Scan for models
        counts["models"] = self.scan_models()

        # Scan for training data
        counts["training_data"] = self.scan_training_data()

        # Scan for validation data
        counts["validation_data"] = self.scan_validation_data()

        # Scan for configs
        counts["configs"] = self.scan_configs()

        # Scan for backups
        counts["backups"] = self.scan_backups()

        total = sum(counts.values())
        logger.info(f"Discovery complete: {total} assets registered")
        logger.info(f"Breakdown: {counts}")

        return counts

    def scan_checkpoints(self) -> int:
        """Scan for checkpoint directories."""
        count = 0

        for model_dir in self.MODEL_DIRS:
            dir_path = self.base_dir / model_dir
            if not dir_path.exists():
                continue

            # Look for checkpoint directories
            for item in dir_path.iterdir():
                if item.is_dir() and self.CHECKPOINT_PATTERN.match(item.name):
                    try:
                        self.keeper.register_from_path(str(item))
                        count += 1
                        logger.debug(f"Registered checkpoint: {item.name}")
                    except Exception as e:
                        logger.warning(f"Failed to register {item}: {e}")

        logger.info(f"Found {count} checkpoints")
        return count

    def scan_models(self) -> int:
        """Scan for model directories (non-checkpoint)."""
        count = 0

        models_dir = self.base_dir / "models"
        if not models_dir.exists():
            return count

        for item in models_dir.iterdir():
            if item.is_dir() and not self.CHECKPOINT_PATTERN.match(item.name):
                # Check if it looks like a model (has model files)
                if self._is_model_dir(item):
                    try:
                        # Determine if it's a base model
                        is_base = any(
                            x in item.name.lower()
                            for x in ["qwen", "llama", "mistral", "base"]
                        )
                        asset = self.keeper.register_from_path(str(item))
                        if is_base:
                            # Update type to base_model
                            asset.asset_type = AssetType.BASE_MODEL
                            self.keeper.register(asset)
                        count += 1
                        logger.debug(f"Registered model: {item.name}")
                    except Exception as e:
                        logger.warning(f"Failed to register {item}: {e}")

        logger.info(f"Found {count} models")
        return count

    def _is_model_dir(self, path: Path) -> bool:
        """Check if a directory looks like a model."""
        model_files = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        return any((path / f).exists() for f in model_files)

    def scan_training_data(self) -> int:
        """Scan for training data files."""
        count = 0

        for data_dir in self.DATA_DIRS:
            dir_path = self.base_dir / data_dir
            if not dir_path.exists():
                continue

            # Look for JSONL files
            for jsonl_file in dir_path.rglob("*.jsonl"):
                # Skip validation files
                if "validation" in str(jsonl_file).lower() or "val" in jsonl_file.stem.lower():
                    continue

                try:
                    self.keeper.register_from_path(str(jsonl_file))
                    count += 1
                    logger.debug(f"Registered training data: {jsonl_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to register {jsonl_file}: {e}")

        logger.info(f"Found {count} training data files")
        return count

    def scan_validation_data(self) -> int:
        """Scan for validation data files."""
        count = 0

        # Check data/validation specifically
        val_dir = self.base_dir / "data" / "validation"
        if val_dir.exists():
            for jsonl_file in val_dir.rglob("*.jsonl"):
                try:
                    self.keeper.register_from_path(str(jsonl_file))
                    count += 1
                    logger.debug(f"Registered validation data: {jsonl_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to register {jsonl_file}: {e}")

        logger.info(f"Found {count} validation data files")
        return count

    def scan_configs(self) -> int:
        """Scan for configuration files."""
        count = 0

        # Main config
        main_config = self.base_dir / "config.json"
        if main_config.exists():
            try:
                self.keeper.register_from_path(str(main_config))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register config.json: {e}")

        # Config directory
        config_dir = self.base_dir / "config"
        if config_dir.exists():
            for json_file in config_dir.glob("*.json"):
                try:
                    self.keeper.register_from_path(str(json_file))
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to register {json_file}: {e}")

        logger.info(f"Found {count} config files")
        return count

    def scan_backups(self) -> int:
        """Scan for backup archives."""
        count = 0

        for backup_dir in self.BACKUP_DIRS:
            dir_path = self.base_dir / backup_dir
            if not dir_path.exists():
                continue

            # Look for archives
            for archive in dir_path.rglob("*"):
                if archive.is_file() and archive.suffix in [".tar", ".tar.gz", ".zip", ".tgz"]:
                    try:
                        self.keeper.register_from_path(str(archive))
                        count += 1
                        logger.debug(f"Registered backup: {archive.name}")
                    except Exception as e:
                        logger.warning(f"Failed to register {archive}: {e}")

        logger.info(f"Found {count} backups")
        return count

    def scan_remote(
        self,
        stronghold: str,
        paths: Optional[List[str]] = None,
    ) -> int:
        """
        Scan a remote stronghold for assets.

        Args:
            stronghold: Stronghold name
            paths: Specific paths to scan (or use defaults)

        Returns:
            Number of assets registered
        """
        handler = self.keeper._get_handler(stronghold)
        if not handler or not handler.is_available():
            logger.error(f"Stronghold not available: {stronghold}")
            return 0

        count = 0
        sh = self.keeper.storage.get_stronghold(stronghold)

        # Default paths to scan
        if not paths:
            paths = ["checkpoint", "model", "training_data", "backup"]

        for path in paths:
            try:
                items = handler.list_dir(path)
                for item in items:
                    try:
                        # Get info about the item
                        info = handler.get_info(item)
                        if not info.exists:
                            continue

                        # Create asset from path info
                        name = Path(item).name

                        # Determine type
                        if self.CHECKPOINT_PATTERN.match(name):
                            asset_type = AssetType.CHECKPOINT
                        elif "model" in path.lower():
                            asset_type = AssetType.MODEL
                        elif item.endswith(".jsonl"):
                            asset_type = AssetType.TRAINING_DATA
                        elif any(item.endswith(ext) for ext in [".tar", ".zip", ".tgz"]):
                            asset_type = AssetType.BACKUP
                        else:
                            continue

                        # Create minimal asset
                        from vault.assets import generate_asset_id
                        asset_id = generate_asset_id(asset_type, name, item)

                        # Check if already registered
                        existing = self.keeper.get(asset_id)
                        if existing:
                            # Just add location
                            self.keeper.add_location(asset_id, stronghold, item)
                        else:
                            # Create new asset
                            location = AssetLocation(
                                stronghold=stronghold,
                                path=item,
                                status=LocationStatus.VERIFIED,
                                verified_at=datetime.now(),
                                size_bytes=info.size_bytes,
                                is_primary=False,  # Remote is not primary
                            )

                            asset = Asset(
                                asset_id=asset_id,
                                asset_type=asset_type,
                                name=name,
                                size_bytes=info.size_bytes,
                                modified_at=info.modified_at,
                                locations=[location],
                            )
                            self.keeper.register(asset)

                        count += 1
                        logger.debug(f"Registered from {stronghold}: {name}")

                    except Exception as e:
                        logger.warning(f"Failed to register {item}: {e}")

            except Exception as e:
                logger.warning(f"Failed to scan {path} in {stronghold}: {e}")

        logger.info(f"Found {count} assets in {stronghold}")
        return count


# =============================================================================
# ASK VAULT FIRST PATTERN
# =============================================================================

def ask_vault_first(
    asset_id: str,
    fallback: Optional[str] = None,
    keeper: Optional[VaultKeeper] = None,
    fetch_if_remote: bool = True,
    local_dir: str = "/tmp/vault_cache",
) -> Optional[str]:
    """
    Ask the VaultKeeper for an asset before using a fallback path.

    This is the "check vault first" pattern - before loading any asset,
    ask the keeper where it is. The keeper may know of a better/faster
    location than your default.

    Args:
        asset_id: Asset identifier (e.g., "checkpoint_175000", "model_qwen3_0.6b")
        fallback: Fallback path if not in vault or vault unavailable
        keeper: VaultKeeper instance (or use default)
        fetch_if_remote: If best location is remote, fetch to local
        local_dir: Directory to fetch remote assets to

    Returns:
        Path to the asset (local), or None if not found

    Example:
        # Old way (hardcoded path):
        model_path = "/path/to/training/models/checkpoint-175000"

        # New way (ask vault first):
        model_path = ask_vault_first(
            "checkpoint_175000",
            fallback="/path/to/training/models/checkpoint-175000"
        )

        # The vault might return:
        # - The same path (if that's the best location)
        # - A different local path (if model was moved)
        # - A fetched copy (if best source was remote NAS)
    """
    keeper = keeper or get_vault_keeper()

    # Ask the vault
    result = keeper.locate(asset_id)

    if not result.found:
        logger.debug(f"Asset {asset_id} not in vault, using fallback")
        return fallback

    # Find best location
    best = result.best_location
    if not best:
        logger.debug(f"No available location for {asset_id}, using fallback")
        return fallback

    # If best is local, return directly
    if best.stronghold == "local_vault":
        if Path(best.path).exists():
            return best.path
        else:
            logger.warning(f"Local path missing: {best.path}")
            # Try to fetch from another location
            for loc in result.locations:
                if loc.stronghold != "local_vault":
                    best = loc
                    break
            else:
                return fallback

    # Best is remote - decide whether to fetch
    if not fetch_if_remote:
        # Return the remote path info for caller to handle
        logger.info(f"Best location for {asset_id} is remote: {best.stronghold}:{best.path}")
        return fallback

    # Fetch to local
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(local_dir) / Path(best.path).name

    logger.info(f"Fetching {asset_id} from {best.stronghold} to {dest}")
    fetch_result = keeper.fetch(asset_id, str(dest), from_stronghold=best.stronghold)

    if fetch_result.success:
        return fetch_result.local_path
    else:
        logger.warning(f"Fetch failed: {fetch_result.error}, using fallback")
        return fallback


def ensure_asset(
    asset_id: str,
    keeper: Optional[VaultKeeper] = None,
    local_dir: str = "/tmp/vault_cache",
) -> Optional[str]:
    """
    Ensure an asset is available locally.

    Similar to ask_vault_first but without fallback - either returns
    a valid local path or None.

    Args:
        asset_id: Asset identifier
        keeper: VaultKeeper instance
        local_dir: Directory to fetch to if needed

    Returns:
        Local path to asset, or None if unavailable
    """
    return ask_vault_first(
        asset_id,
        fallback=None,
        keeper=keeper,
        fetch_if_remote=True,
        local_dir=local_dir,
    )


def register_and_locate(
    path: str | Path,
    keeper: Optional[VaultKeeper] = None,
    stronghold: str = "local_vault",
) -> Tuple[str, Asset]:
    """
    Register an asset from path and return its ID.

    Convenience function for registering an asset and immediately
    using it via the vault system.

    Args:
        path: Path to the asset
        keeper: VaultKeeper instance
        stronghold: Which stronghold this path is in

    Returns:
        Tuple of (asset_id, Asset)
    """
    keeper = keeper or get_vault_keeper()
    asset = keeper.register_from_path(path, stronghold)
    return asset.asset_id, asset


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run discovery from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="VaultKeeper Discovery Service")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/path/to/training",
        help="Base training directory",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan all directories for assets",
    )
    parser.add_argument(
        "--scan-type",
        type=str,
        choices=["checkpoints", "models", "data", "configs", "backups"],
        help="Scan specific asset type",
    )
    parser.add_argument(
        "--scan-remote",
        type=str,
        help="Scan a remote stronghold",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    discovery = VaultDiscovery(base_dir=args.base_dir)

    if args.scan_all:
        counts = discovery.scan_all()
        print(f"\nDiscovery Results:")
        for asset_type, count in counts.items():
            print(f"  {asset_type}: {count}")

    elif args.scan_type:
        type_map = {
            "checkpoints": discovery.scan_checkpoints,
            "models": discovery.scan_models,
            "data": discovery.scan_training_data,
            "configs": discovery.scan_configs,
            "backups": discovery.scan_backups,
        }
        count = type_map[args.scan_type]()
        print(f"\nFound {count} {args.scan_type}")

    elif args.scan_remote:
        count = discovery.scan_remote(args.scan_remote)
        print(f"\nFound {count} assets in {args.scan_remote}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
