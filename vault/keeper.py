"""
VaultKeeper - Central asset registry and retrieval service.

The VaultKeeper is the master custodian of all treasures in the realm.
It knows where every asset is located and can retrieve it on demand:

    - Track assets across all strongholds (4090, 3090, Synology, etc.)
    - Know what exists where
    - Retrieve assets from the best available source
    - Coordinate syncing between locations

When any device needs something, they ask the VaultKeeper first:

    keeper = VaultKeeper()

    # Where is checkpoint 175000?
    locations = keeper.locate("checkpoint_175000")

    # Get it for me
    local_path = keeper.fetch("checkpoint_175000", "/tmp/checkpoint")

    # Or just tell me the best place to get it
    best_location = keeper.best_location("checkpoint_175000")

RPG Flavor:
    The VaultKeeper is an ancient entity who guards the realm's treasures.
    It maintains the Great Ledger (catalog) - a magical tome that records
    every treasure's location across all strongholds. Ask the Keeper where
    something is, and it will divine the answer.

Storage:
    Uses SQLite for the catalog (fast, portable, no server needed).
    The catalog can be synced across devices for distributed access.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from vault.assets import (
    Asset,
    AssetLocation,
    AssetQuery,
    AssetStatus,
    AssetType,
    LocationStatus,
    asset_from_path,
    generate_asset_id,
)
from vault.handlers import (
    LocationHandler,
    TransferResult,
    get_handler,
)
from vault.storage_registry import (
    StorageRegistry,
    Stronghold,
    StrongholdStatus,
    StrongholdType,
)


@dataclass
class LookupResult:
    """Result of a locate/lookup operation."""
    asset_id: str
    found: bool
    locations: List[AssetLocation]
    best_location: Optional[AssetLocation] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "found": self.found,
            "locations": [loc.to_dict() for loc in self.locations],
            "best_location": self.best_location.to_dict() if self.best_location else None,
            "error": self.error,
        }


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    asset_id: str
    success: bool
    local_path: Optional[str] = None
    source_location: Optional[AssetLocation] = None
    transfer: Optional[TransferResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "success": self.success,
            "local_path": self.local_path,
            "source_location": self.source_location.to_dict() if self.source_location else None,
            "transfer": self.transfer.to_dict() if self.transfer else None,
            "error": self.error,
        }


class VaultKeeper:
    """
    Central asset registry and retrieval service.

    The VaultKeeper maintains a catalog of all assets across all strongholds
    and coordinates lookup and retrieval operations.

    Usage:
        keeper = VaultKeeper()

        # Register an asset
        keeper.register(asset)

        # Find where an asset is
        result = keeper.locate("checkpoint_175000")
        print(f"Found at: {result.best_location.path}")

        # Get the asset
        result = keeper.fetch("checkpoint_175000", "/local/path")
        print(f"Downloaded to: {result.local_path}")

        # Search for assets
        checkpoints = keeper.search(AssetQuery(asset_type=AssetType.CHECKPOINT))
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(
        self,
        base_dir: Optional[str | Path] = None,
        catalog_path: Optional[str | Path] = None,
    ):
        """
        Initialize the VaultKeeper.

        Args:
            base_dir: Base training directory (default: auto-detect)
            catalog_path: Path to catalog database (defaults to vault/catalog.db)
        """
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.catalog_path = Path(catalog_path) if catalog_path else self.base_dir / "vault" / "catalog.db"

        # Storage registry for stronghold management
        self.storage = StorageRegistry(base_dir)

        # Location handlers cache
        self._handlers: Dict[str, LocationHandler] = {}

        # Thread lock for database access
        self._lock = threading.Lock()

        # Initialize database
        self._init_catalog()

    # =========================================================================
    # DATABASE MANAGEMENT
    # =========================================================================

    def _init_catalog(self):
        """Initialize the catalog database."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_conn() as conn:
            # Create tables
            conn.executescript("""
                -- Assets table
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    size_bytes INTEGER DEFAULT 0,
                    created_at TEXT,
                    modified_at TEXT,
                    status TEXT DEFAULT 'active',
                    metadata TEXT DEFAULT '{}'
                );

                -- Locations table (where each asset exists)
                CREATE TABLE IF NOT EXISTS locations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL,
                    stronghold TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT DEFAULT 'unverified',
                    verified_at TEXT,
                    checksum TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    synced_at TEXT,
                    is_primary INTEGER DEFAULT 0,
                    FOREIGN KEY (asset_id) REFERENCES assets(asset_id),
                    UNIQUE(asset_id, stronghold, path)
                );

                -- Transfer history
                CREATE TABLE IF NOT EXISTS transfers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL,
                    from_stronghold TEXT NOT NULL,
                    to_stronghold TEXT NOT NULL,
                    from_path TEXT NOT NULL,
                    to_path TEXT NOT NULL,
                    bytes_transferred INTEGER DEFAULT 0,
                    duration_seconds REAL DEFAULT 0,
                    success INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
                );

                -- Schema version
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);
                CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
                CREATE INDEX IF NOT EXISTS idx_locations_asset ON locations(asset_id);
                CREATE INDEX IF NOT EXISTS idx_locations_stronghold ON locations(stronghold);
            """)

            # Set schema version
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION))
            )

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        with self._lock:
            conn = sqlite3.connect(str(self.catalog_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # =========================================================================
    # HANDLER MANAGEMENT
    # =========================================================================

    def _get_handler(self, stronghold_name: str) -> Optional[LocationHandler]:
        """Get or create a handler for a stronghold."""
        if stronghold_name in self._handlers:
            return self._handlers[stronghold_name]

        stronghold = self.storage.get_stronghold(stronghold_name)
        if not stronghold:
            return None

        try:
            handler = get_handler(stronghold)
            self._handlers[stronghold_name] = handler
            return handler
        except Exception:
            return None

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(self, asset: Asset) -> bool:
        """
        Register an asset in the catalog.

        Args:
            asset: Asset to register

        Returns:
            True if registered successfully
        """
        with self._get_conn() as conn:
            # Insert or update asset
            conn.execute(
                """
                INSERT OR REPLACE INTO assets
                (asset_id, asset_type, name, description, size_bytes,
                 created_at, modified_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset.asset_id,
                    asset.asset_type.value,
                    asset.name,
                    asset.description,
                    asset.size_bytes,
                    asset.created_at.isoformat() if asset.created_at else None,
                    asset.modified_at.isoformat() if asset.modified_at else None,
                    asset.status.value,
                    json.dumps(asset.metadata),
                )
            )

            # Insert locations
            for loc in asset.locations:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO locations
                    (asset_id, stronghold, path, status, verified_at,
                     checksum, size_bytes, synced_at, is_primary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset.asset_id,
                        loc.stronghold,
                        loc.path,
                        loc.status.value,
                        loc.verified_at.isoformat() if loc.verified_at else None,
                        loc.checksum,
                        loc.size_bytes,
                        loc.synced_at.isoformat() if loc.synced_at else None,
                        1 if loc.is_primary else 0,
                    )
                )

        return True

    def register_from_path(
        self,
        path: str | Path,
        stronghold: str = "local_vault",
        is_primary: bool = True,
    ) -> Asset:
        """
        Register an asset from a filesystem path.

        Args:
            path: Path to the asset
            stronghold: Which stronghold this path is in
            is_primary: Whether this is the primary copy

        Returns:
            The registered Asset
        """
        asset = asset_from_path(path, stronghold, is_primary)
        self.register(asset)
        return asset

    def add_location(
        self,
        asset_id: str,
        stronghold: str,
        path: str,
        is_primary: bool = False,
    ) -> bool:
        """
        Add a location for an existing asset.

        Args:
            asset_id: Asset ID
            stronghold: Stronghold name
            path: Path within stronghold
            is_primary: Whether this is the primary copy

        Returns:
            True if added successfully
        """
        with self._get_conn() as conn:
            # Verify asset exists
            row = conn.execute(
                "SELECT asset_id FROM assets WHERE asset_id = ?",
                (asset_id,)
            ).fetchone()

            if not row:
                return False

            # Add location
            conn.execute(
                """
                INSERT OR REPLACE INTO locations
                (asset_id, stronghold, path, status, is_primary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (asset_id, stronghold, path, "unverified", 1 if is_primary else 0)
            )

        return True

    def remove_location(self, asset_id: str, stronghold: str, path: str) -> bool:
        """Remove a location from an asset."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM locations WHERE asset_id = ? AND stronghold = ? AND path = ?",
                (asset_id, stronghold, path)
            )
        return True

    # =========================================================================
    # LOOKUP OPERATIONS
    # =========================================================================

    def get(self, asset_id: str) -> Optional[Asset]:
        """
        Get an asset by ID.

        Args:
            asset_id: Asset identifier

        Returns:
            Asset or None if not found
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?",
                (asset_id,)
            ).fetchone()

            if not row:
                return None

            # Get locations
            loc_rows = conn.execute(
                "SELECT * FROM locations WHERE asset_id = ?",
                (asset_id,)
            ).fetchall()

            locations = [
                AssetLocation(
                    stronghold=r["stronghold"],
                    path=r["path"],
                    status=LocationStatus(r["status"]),
                    verified_at=datetime.fromisoformat(r["verified_at"]) if r["verified_at"] else None,
                    checksum=r["checksum"],
                    size_bytes=r["size_bytes"] or 0,
                    synced_at=datetime.fromisoformat(r["synced_at"]) if r["synced_at"] else None,
                    is_primary=bool(r["is_primary"]),
                )
                for r in loc_rows
            ]

            return Asset(
                asset_id=row["asset_id"],
                asset_type=AssetType(row["asset_type"]),
                name=row["name"],
                description=row["description"] or "",
                size_bytes=row["size_bytes"] or 0,
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                modified_at=datetime.fromisoformat(row["modified_at"]) if row["modified_at"] else None,
                status=AssetStatus(row["status"]),
                locations=locations,
                metadata=json.loads(row["metadata"] or "{}"),
            )

    def locate(self, asset_id: str) -> LookupResult:
        """
        Locate an asset across all strongholds.

        Args:
            asset_id: Asset identifier

        Returns:
            LookupResult with all known locations
        """
        asset = self.get(asset_id)

        if not asset:
            return LookupResult(
                asset_id=asset_id,
                found=False,
                locations=[],
                error="Asset not found in catalog",
            )

        # Rank locations by preference
        available = asset.available_locations
        best = self._pick_best_location(available)

        return LookupResult(
            asset_id=asset_id,
            found=True,
            locations=available,
            best_location=best,
        )

    def _pick_best_location(self, locations: List[AssetLocation]) -> Optional[AssetLocation]:
        """Pick the best location to fetch from."""
        if not locations:
            return None

        # Priority: primary > local > verified remote > unverified
        def score(loc: AssetLocation) -> Tuple[int, int, int]:
            primary = 0 if loc.is_primary else 1
            local = 0 if loc.stronghold == "local_vault" else 1
            verified = 0 if loc.status == LocationStatus.VERIFIED else 1
            return (primary, local, verified)

        return min(locations, key=score)

    def search(self, query: AssetQuery) -> List[Asset]:
        """
        Search for assets matching a query.

        Args:
            query: Search criteria

        Returns:
            List of matching assets
        """
        with self._get_conn() as conn:
            # Build query
            sql = "SELECT DISTINCT asset_id FROM assets WHERE 1=1"
            params: List[Any] = []

            if query.asset_type:
                sql += " AND asset_type = ?"
                params.append(query.asset_type.value)

            if query.status:
                sql += " AND status = ?"
                params.append(query.status.value)

            if query.name_pattern:
                sql += " AND name LIKE ?"
                params.append(f"%{query.name_pattern}%")

            if query.min_size_bytes:
                sql += " AND size_bytes >= ?"
                params.append(query.min_size_bytes)

            if query.max_size_bytes:
                sql += " AND size_bytes <= ?"
                params.append(query.max_size_bytes)

            if query.stronghold:
                sql += " AND asset_id IN (SELECT asset_id FROM locations WHERE stronghold = ?)"
                params.append(query.stronghold)

            rows = conn.execute(sql, params).fetchall()

        # Get full assets
        assets = []
        for row in rows:
            asset = self.get(row["asset_id"])
            if asset and (not query.metadata_filter or query.matches(asset)):
                assets.append(asset)

        return assets

    def list_by_type(self, asset_type: AssetType) -> List[Asset]:
        """List all assets of a given type."""
        return self.search(AssetQuery(asset_type=asset_type))

    def list_checkpoints(self) -> List[Asset]:
        """List all checkpoints, sorted by step number."""
        checkpoints = self.list_by_type(AssetType.CHECKPOINT)
        return sorted(
            checkpoints,
            key=lambda a: a.metadata.get("step_number", 0),
            reverse=True,
        )

    def list_in_stronghold(self, stronghold: str) -> List[Asset]:
        """List all assets in a specific stronghold."""
        return self.search(AssetQuery(stronghold=stronghold))

    # =========================================================================
    # RETRIEVAL OPERATIONS
    # =========================================================================

    def fetch(
        self,
        asset_id: str,
        destination: str | Path,
        from_stronghold: Optional[str] = None,
    ) -> FetchResult:
        """
        Fetch an asset to a local path.

        Args:
            asset_id: Asset to fetch
            destination: Local destination path
            from_stronghold: Specific stronghold to fetch from (or best available)

        Returns:
            FetchResult with transfer details
        """
        # Locate the asset
        lookup = self.locate(asset_id)

        if not lookup.found:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error=lookup.error or "Asset not found",
            )

        # Pick source location
        if from_stronghold:
            source = next(
                (loc for loc in lookup.locations if loc.stronghold == from_stronghold),
                None
            )
            if not source:
                return FetchResult(
                    asset_id=asset_id,
                    success=False,
                    error=f"Asset not found in stronghold: {from_stronghold}",
                )
        else:
            source = lookup.best_location

        if not source:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error="No available location for asset",
            )

        # Get handler for source stronghold
        handler = self._get_handler(source.stronghold)
        if not handler:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error=f"No handler for stronghold: {source.stronghold}",
            )

        # Check if source stronghold is available
        if not handler.is_available():
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error=f"Stronghold not available: {source.stronghold}",
            )

        # Special case: if source is local_vault, it's already local
        if source.stronghold == "local_vault":
            # Just verify it exists and return the path
            if Path(source.path).exists():
                return FetchResult(
                    asset_id=asset_id,
                    success=True,
                    local_path=source.path,
                    source_location=source,
                )
            else:
                return FetchResult(
                    asset_id=asset_id,
                    success=False,
                    error=f"Local file not found: {source.path}",
                )

        # Fetch from remote
        destination = Path(destination)
        transfer = handler.fetch(source.path, str(destination))

        # Record transfer
        self._record_transfer(
            asset_id=asset_id,
            from_stronghold=source.stronghold,
            to_stronghold="local_vault",
            from_path=source.path,
            to_path=str(destination),
            result=transfer,
        )

        if transfer.success:
            # Register the new local location
            self.add_location(asset_id, "local_vault", str(destination))

            return FetchResult(
                asset_id=asset_id,
                success=True,
                local_path=str(destination),
                source_location=source,
                transfer=transfer,
            )
        else:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                source_location=source,
                transfer=transfer,
                error=transfer.error,
            )

    def push(
        self,
        asset_id: str,
        to_stronghold: str,
        destination_path: Optional[str] = None,
    ) -> FetchResult:
        """
        Push an asset to a remote stronghold.

        Args:
            asset_id: Asset to push
            to_stronghold: Target stronghold
            destination_path: Path in target stronghold (defaults to organized path)

        Returns:
            FetchResult with transfer details
        """
        # Get the asset
        asset = self.get(asset_id)
        if not asset:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error="Asset not found",
            )

        # Find local copy
        local = asset.get_location("local_vault")
        if not local or not Path(local.path).exists():
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error="No local copy available to push",
            )

        # Get handler for destination
        handler = self._get_handler(to_stronghold)
        if not handler:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error=f"No handler for stronghold: {to_stronghold}",
            )

        if not handler.is_available():
            return FetchResult(
                asset_id=asset_id,
                success=False,
                error=f"Stronghold not available: {to_stronghold}",
            )

        # Build destination path
        if not destination_path:
            # Organize by type
            destination_path = f"{asset.asset_type.value}/{asset.name}"

        # Push the asset
        transfer = handler.push(local.path, destination_path)

        # Record transfer
        self._record_transfer(
            asset_id=asset_id,
            from_stronghold="local_vault",
            to_stronghold=to_stronghold,
            from_path=local.path,
            to_path=destination_path,
            result=transfer,
        )

        if transfer.success:
            # Register the new location
            self.add_location(asset_id, to_stronghold, destination_path)

            return FetchResult(
                asset_id=asset_id,
                success=True,
                local_path=local.path,
                transfer=transfer,
            )
        else:
            return FetchResult(
                asset_id=asset_id,
                success=False,
                transfer=transfer,
                error=transfer.error,
            )

    def _record_transfer(
        self,
        asset_id: str,
        from_stronghold: str,
        to_stronghold: str,
        from_path: str,
        to_path: str,
        result: TransferResult,
    ):
        """Record a transfer in the history."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO transfers
                (asset_id, from_stronghold, to_stronghold, from_path, to_path,
                 bytes_transferred, duration_seconds, success, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    from_stronghold,
                    to_stronghold,
                    from_path,
                    to_path,
                    result.bytes_transferred,
                    result.duration_seconds,
                    1 if result.success else 0,
                    result.error,
                )
            )

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify(self, asset_id: str, stronghold: Optional[str] = None) -> Dict[str, bool]:
        """
        Verify an asset exists at its registered locations.

        Args:
            asset_id: Asset to verify
            stronghold: Specific stronghold to verify (or all)

        Returns:
            Dict mapping stronghold -> verified
        """
        asset = self.get(asset_id)
        if not asset:
            return {}

        results = {}
        locations = asset.locations

        if stronghold:
            locations = [loc for loc in locations if loc.stronghold == stronghold]

        for loc in locations:
            handler = self._get_handler(loc.stronghold)
            if not handler:
                results[loc.stronghold] = False
                continue

            verified, info = handler.verify(loc.path, loc.checksum)
            results[loc.stronghold] = verified

            # Update location status in database
            with self._get_conn() as conn:
                status = "verified" if verified else "missing"
                conn.execute(
                    """
                    UPDATE locations
                    SET status = ?, verified_at = ?, size_bytes = ?, checksum = ?
                    WHERE asset_id = ? AND stronghold = ? AND path = ?
                    """,
                    (
                        status,
                        datetime.now().isoformat() if verified else None,
                        info.size_bytes if verified else 0,
                        info.checksum,
                        asset_id,
                        loc.stronghold,
                        loc.path,
                    )
                )

        return results

    def verify_all(self, stronghold: Optional[str] = None) -> Dict[str, Dict[str, bool]]:
        """
        Verify all assets (optionally in a specific stronghold).

        Args:
            stronghold: Specific stronghold to verify (or all)

        Returns:
            Dict mapping asset_id -> {stronghold -> verified}
        """
        if stronghold:
            assets = self.list_in_stronghold(stronghold)
        else:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT asset_id FROM assets").fetchall()
            assets = [self.get(row["asset_id"]) for row in rows]
            assets = [a for a in assets if a]

        results = {}
        for asset in assets:
            results[asset.asset_id] = self.verify(asset.asset_id, stronghold)

        return results

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        with self._get_conn() as conn:
            # Asset counts by type
            type_counts = {}
            for row in conn.execute(
                "SELECT asset_type, COUNT(*) as count FROM assets GROUP BY asset_type"
            ):
                type_counts[row["asset_type"]] = row["count"]

            # Location counts by stronghold
            stronghold_counts = {}
            for row in conn.execute(
                "SELECT stronghold, COUNT(*) as count FROM locations GROUP BY stronghold"
            ):
                stronghold_counts[row["stronghold"]] = row["count"]

            # Total size
            total_size = conn.execute(
                "SELECT SUM(size_bytes) as total FROM assets"
            ).fetchone()["total"] or 0

            # Recent transfers
            recent_transfers = conn.execute(
                """
                SELECT * FROM transfers
                ORDER BY created_at DESC
                LIMIT 10
                """
            ).fetchall()

        return {
            "total_assets": sum(type_counts.values()),
            "assets_by_type": type_counts,
            "locations_by_stronghold": stronghold_counts,
            "total_size_gb": round(total_size / (1024 ** 3), 2),
            "recent_transfers": [
                {
                    "asset_id": t["asset_id"],
                    "from": t["from_stronghold"],
                    "to": t["to_stronghold"],
                    "success": bool(t["success"]),
                    "timestamp": t["created_at"],
                }
                for t in recent_transfers
            ],
        }

    def export_catalog(self) -> Dict[str, Any]:
        """Export the entire catalog as JSON."""
        with self._get_conn() as conn:
            assets = []
            for row in conn.execute("SELECT asset_id FROM assets"):
                asset = self.get(row["asset_id"])
                if asset:
                    assets.append(asset.to_dict())

        return {
            "version": self.SCHEMA_VERSION,
            "exported_at": datetime.now().isoformat(),
            "assets": assets,
            "stats": self.get_stats(),
        }

    def import_catalog(self, data: Dict[str, Any], merge: bool = True):
        """
        Import catalog from JSON export.

        Args:
            data: Exported catalog data
            merge: If True, merge with existing. If False, replace.
        """
        if not merge:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM locations")
                conn.execute("DELETE FROM assets")

        for asset_data in data.get("assets", []):
            asset = Asset.from_dict(asset_data)
            self.register(asset)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_keeper_instance: Optional[VaultKeeper] = None


def get_vault_keeper(base_dir: Optional[str | Path] = None) -> VaultKeeper:
    """Get or create a VaultKeeper instance."""
    global _keeper_instance
    if _keeper_instance is None:
        _keeper_instance = VaultKeeper(base_dir)
    return _keeper_instance


def locate(asset_id: str) -> LookupResult:
    """Convenience function to locate an asset."""
    return get_vault_keeper().locate(asset_id)


def fetch(asset_id: str, destination: str | Path) -> FetchResult:
    """Convenience function to fetch an asset."""
    return get_vault_keeper().fetch(asset_id, destination)
