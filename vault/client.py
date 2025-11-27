"""
VaultKeeper Client - Client library for querying the VaultKeeper.

Use this client from any device to query the VaultKeeper:
    - Find where assets are located
    - Request asset retrieval
    - Search the catalog

Usage:
    from vault.client import VaultKeeperClient

    # Connect to keeper (on 4090)
    client = VaultKeeperClient("192.168.x.x:8767")

    # Find checkpoint
    result = client.locate("checkpoint_175000")
    if result.found:
        print(f"Found at: {result.best_location['path']}")

    # Get it locally
    fetch_result = client.fetch("checkpoint_175000", "/local/models/ckpt")

    # Search for all models
    models = client.search(type="model")

RPG Flavor:
    The VaultKeeper Client is a Sending Stone - a magical item that
    allows communication with the VaultKeeper across any distance.
    Hold it and speak your query, and the Keeper shall respond.
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vault_client")


@dataclass
class LocationInfo:
    """Information about an asset location."""
    stronghold: str
    path: str
    status: str
    verified_at: Optional[str]
    checksum: Optional[str]
    size_bytes: int
    is_primary: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocationInfo":
        return cls(
            stronghold=data.get("stronghold", ""),
            path=data.get("path", ""),
            status=data.get("status", "unknown"),
            verified_at=data.get("verified_at"),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
            is_primary=data.get("is_primary", False),
        )


@dataclass
class LocateResult:
    """Result of a locate operation."""
    asset_id: str
    found: bool
    locations: List[LocationInfo]
    best_location: Optional[LocationInfo]
    error: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocateResult":
        locations = [
            LocationInfo.from_dict(loc)
            for loc in data.get("locations", [])
        ]
        best = None
        if data.get("best_location"):
            best = LocationInfo.from_dict(data["best_location"])

        return cls(
            asset_id=data.get("asset_id", ""),
            found=data.get("found", False),
            locations=locations,
            best_location=best,
            error=data.get("error"),
        )


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    asset_id: str
    success: bool
    local_path: Optional[str]
    source_location: Optional[LocationInfo]
    bytes_transferred: int
    duration_seconds: float
    error: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FetchResult":
        source = None
        if data.get("source_location"):
            source = LocationInfo.from_dict(data["source_location"])

        transfer = data.get("transfer", {})

        return cls(
            asset_id=data.get("asset_id", ""),
            success=data.get("success", False),
            local_path=data.get("local_path"),
            source_location=source,
            bytes_transferred=transfer.get("bytes_transferred", 0),
            duration_seconds=transfer.get("duration_seconds", 0),
            error=data.get("error"),
        )


@dataclass
class AssetInfo:
    """Information about an asset."""
    asset_id: str
    asset_type: str
    name: str
    description: str
    size_bytes: int
    size_gb: float
    created_at: Optional[str]
    status: str
    locations: List[LocationInfo]
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetInfo":
        locations = [
            LocationInfo.from_dict(loc)
            for loc in data.get("locations", [])
        ]

        return cls(
            asset_id=data.get("asset_id", ""),
            asset_type=data.get("asset_type", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            size_bytes=data.get("size_bytes", 0),
            size_gb=data.get("size_gb", 0),
            created_at=data.get("created_at"),
            status=data.get("status", "unknown"),
            locations=locations,
            metadata=data.get("metadata", {}),
        )

    @property
    def step_number(self) -> Optional[int]:
        """Get step number for checkpoints."""
        return self.metadata.get("step_number")


class VaultKeeperClient:
    """
    Client for querying the VaultKeeper service.

    Usage:
        client = VaultKeeperClient("192.168.x.x:8767")

        # Check health
        if client.is_healthy():
            print("Keeper is available")

        # Find an asset
        result = client.locate("checkpoint_175000")

        # Get it locally
        fetch_result = client.fetch("checkpoint_175000", "/tmp/checkpoint")
    """

    def __init__(
        self,
        host: str = "localhost:8767",
        timeout: int = 30,
    ):
        """
        Initialize the client.

        Args:
            host: VaultKeeper host:port (e.g., "192.168.x.x:8767")
            timeout: Request timeout in seconds
        """
        # Normalize host
        if not host.startswith("http"):
            host = f"http://{host}"
        self.base_url = host.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the keeper."""
        url = f"{self.base_url}{endpoint}"

        if params:
            url = f"{url}?{urlencode(params)}"

        data = None
        if body:
            data = json.dumps(body).encode("utf-8")

        request = urllib.request.Request(
            url,
            method=method,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                return error_body
            except Exception:
                return {"error": f"HTTP {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"Connection failed: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}

    def _get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request."""
        return self._request("POST", endpoint, body=body)

    # =========================================================================
    # HEALTH & STATUS
    # =========================================================================

    def is_healthy(self) -> bool:
        """Check if the VaultKeeper is available."""
        result = self._get("/health")
        return result.get("status") == "healthy"

    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return self._get("/api/stats")

    def get_strongholds(self) -> List[Dict[str, Any]]:
        """List all registered strongholds."""
        result = self._get("/api/strongholds")
        return result.get("strongholds", [])

    # =========================================================================
    # LOOKUP OPERATIONS
    # =========================================================================

    def locate(self, asset_id: str) -> LocateResult:
        """
        Locate an asset across all strongholds.

        Args:
            asset_id: Asset identifier (e.g., "checkpoint_175000")

        Returns:
            LocateResult with all known locations
        """
        result = self._get(f"/api/locate/{asset_id}")
        return LocateResult.from_dict(result)

    def get_asset(self, asset_id: str) -> Optional[AssetInfo]:
        """
        Get full details about an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            AssetInfo or None if not found
        """
        result = self._get(f"/api/asset/{asset_id}")
        if "error" in result:
            return None
        return AssetInfo.from_dict(result)

    def search(
        self,
        type: Optional[str] = None,
        name: Optional[str] = None,
        stronghold: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AssetInfo]:
        """
        Search for assets.

        Args:
            type: Asset type (checkpoint, model, training_data, etc.)
            name: Name pattern to match
            stronghold: Only assets in this stronghold
            status: Asset status (active, archived, etc.)

        Returns:
            List of matching assets
        """
        params = {}
        if type:
            params["type"] = type
        if name:
            params["name"] = name
        if stronghold:
            params["stronghold"] = stronghold
        if status:
            params["status"] = status

        result = self._get("/api/search", params)
        return [AssetInfo.from_dict(a) for a in result.get("assets", [])]

    def list_checkpoints(self) -> List[AssetInfo]:
        """List all checkpoints sorted by step number."""
        result = self._get("/api/checkpoints")
        return [AssetInfo.from_dict(c) for c in result.get("checkpoints", [])]

    def list_by_type(self, asset_type: str) -> List[AssetInfo]:
        """List all assets of a given type."""
        result = self._get("/api/list", {"type": asset_type})
        return [AssetInfo.from_dict(a) for a in result.get("assets", [])]

    def list_in_stronghold(self, stronghold: str) -> List[AssetInfo]:
        """List all assets in a specific stronghold."""
        result = self._get("/api/list", {"stronghold": stronghold})
        return [AssetInfo.from_dict(a) for a in result.get("assets", [])]

    # =========================================================================
    # RETRIEVAL OPERATIONS
    # =========================================================================

    def fetch(
        self,
        asset_id: str,
        destination: str,
        from_stronghold: Optional[str] = None,
    ) -> FetchResult:
        """
        Fetch an asset to a local path.

        Args:
            asset_id: Asset to fetch
            destination: Local destination path
            from_stronghold: Specific stronghold to fetch from (optional)

        Returns:
            FetchResult with transfer details
        """
        body = {
            "asset_id": asset_id,
            "destination": destination,
        }
        if from_stronghold:
            body["from_stronghold"] = from_stronghold

        result = self._post("/api/fetch", body)
        return FetchResult.from_dict(result)

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
            destination_path: Path in target (optional)

        Returns:
            FetchResult with transfer details
        """
        body = {
            "asset_id": asset_id,
            "to_stronghold": to_stronghold,
        }
        if destination_path:
            body["destination_path"] = destination_path

        result = self._post("/api/push", body)
        return FetchResult.from_dict(result)

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(
        self,
        path: str,
        stronghold: str = "local_vault",
        is_primary: bool = True,
    ) -> Optional[AssetInfo]:
        """
        Register an asset from a path.

        Args:
            path: Path to the asset
            stronghold: Which stronghold this path is in
            is_primary: Whether this is the primary copy

        Returns:
            AssetInfo or None if failed
        """
        result = self._post("/api/register", {
            "path": path,
            "stronghold": stronghold,
            "is_primary": is_primary,
        })

        if result.get("success"):
            return AssetInfo.from_dict(result.get("asset", {}))
        return None

    def scan(
        self,
        path: str,
        stronghold: str = "local_vault",
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan a directory and register all assets.

        Args:
            path: Directory to scan
            stronghold: Which stronghold this is
            recursive: Scan subdirectories

        Returns:
            Dict with scan results
        """
        return self._post("/api/scan", {
            "path": path,
            "stronghold": stronghold,
            "recursive": recursive,
        })

    def verify(
        self,
        asset_id: str,
        stronghold: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Verify an asset exists at its locations.

        Args:
            asset_id: Asset to verify
            stronghold: Specific stronghold to verify (optional)

        Returns:
            Dict mapping stronghold -> verified
        """
        body = {"asset_id": asset_id}
        if stronghold:
            body["stronghold"] = stronghold

        result = self._post("/api/verify", body)
        return result.get("verification_results", {})

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def find_checkpoint(self, step: int) -> Optional[AssetInfo]:
        """
        Find a checkpoint by step number.

        Args:
            step: Training step number

        Returns:
            AssetInfo or None
        """
        asset_id = f"checkpoint_{step}"
        return self.get_asset(asset_id)

    def get_latest_checkpoint(self) -> Optional[AssetInfo]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def ensure_local(self, asset_id: str, local_dir: str = "/tmp") -> Optional[str]:
        """
        Ensure an asset is available locally.

        If the asset has a local copy, returns that path.
        Otherwise fetches it to local_dir.

        Args:
            asset_id: Asset to ensure locally
            local_dir: Directory to fetch to if needed

        Returns:
            Local path or None if failed
        """
        result = self.locate(asset_id)

        if not result.found:
            logger.error(f"Asset not found: {asset_id}")
            return None

        # Check if we have a local copy
        for loc in result.locations:
            if loc.stronghold == "local_vault":
                if Path(loc.path).exists():
                    return loc.path

        # Need to fetch
        if not result.best_location:
            logger.error(f"No available location for: {asset_id}")
            return None

        # Build destination path
        dest = Path(local_dir) / Path(result.best_location.path).name
        fetch_result = self.fetch(asset_id, str(dest))

        if fetch_result.success:
            return fetch_result.local_path
        else:
            logger.error(f"Fetch failed: {fetch_result.error}")
            return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Default client instance
_default_client: Optional[VaultKeeperClient] = None


def get_client(host: str = "localhost:8767") -> VaultKeeperClient:
    """Get or create a default client."""
    global _default_client
    if _default_client is None:
        _default_client = VaultKeeperClient(host)
    return _default_client


def locate(asset_id: str, host: str = "localhost:8767") -> LocateResult:
    """Convenience function to locate an asset."""
    return get_client(host).locate(asset_id)


def fetch(asset_id: str, destination: str, host: str = "localhost:8767") -> FetchResult:
    """Convenience function to fetch an asset."""
    return get_client(host).fetch(asset_id, destination)


def ensure_local(asset_id: str, host: str = "localhost:8767") -> Optional[str]:
    """Convenience function to ensure an asset is local."""
    return get_client(host).ensure_local(asset_id)
