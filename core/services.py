"""
Service Client - Unified abstraction for remote HTTP service communication.

This module provides a standardized way to communicate with remote services
(inference, vault, monitoring, data_manager, etc.) with:
- Centralized URL construction (via core.hosts)
- Configurable timeouts, retries, and backoff
- Standard exception hierarchy
- Structured logging and metrics
- API key authentication support

Usage:
    from core.services import get_service_client, ServiceError

    # Get a client for a service
    client = get_service_client("inference")

    # Make requests
    try:
        result = client.post_json("/v1/chat/completions", json=payload)
    except ServiceUnavailable:
        # Service is down, degrade gracefully
        pass
    except ServiceHttpError as e:
        # HTTP 4xx/5xx error
        logger.error(f"HTTP {e.status}: {e.body}")

Domain-specific clients should wrap ServiceClient:

    class InferenceClient:
        def __init__(self):
            self._client = get_service_client("inference")

        def chat(self, messages: list, **kwargs) -> dict:
            return self._client.post_json("/v1/chat/completions", json={
                "messages": messages, **kwargs
            })

Design:
    - ServiceClient is the low-level HTTP pipe
    - All network/timeout/retry behavior lives here
    - Domain clients add schema validation and high-level methods
    - Exceptions never leak raw requests errors
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import requests

logger = logging.getLogger("services")


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ServiceError(Exception):
    """Base error for any remote service problem."""
    pass


class ServiceUnavailable(ServiceError):
    """Service not reachable - connection error, timeout, or all retries exhausted."""

    def __init__(self, service: str, url: str, cause: Optional[Exception] = None):
        self.service = service
        self.url = url
        self.cause = cause
        message = f"Service '{service}' at {url} is unavailable"
        if cause:
            message += f": {cause}"
        super().__init__(message)


class ServiceHttpError(ServiceError):
    """HTTP 4xx/5xx error from the service."""

    def __init__(self, status: int, body: Optional[str] = None, url: Optional[str] = None):
        self.status = status
        self.body = body
        self.url = url
        truncated = body[:200] if body else ""
        super().__init__(f"HTTP {status}: {truncated}")


class ServiceDecodeError(ServiceError):
    """Response wasn't valid JSON or missing expected fields."""

    def __init__(self, message: str, url: Optional[str] = None):
        self.url = url
        super().__init__(message)


class ServiceAuthError(ServiceError):
    """Authentication failed (401/403)."""

    def __init__(self, service: str, message: str = "Authentication required"):
        self.service = service
        super().__init__(f"Auth error for '{service}': {message}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ServiceId:
    """Logical identity of a remote service."""
    name: str  # e.g., "inference", "vault", "data_manager", "scheduler"

    def __str__(self) -> str:
        return self.name


@dataclass
class ServiceConfig:
    """Configuration for a remote service."""
    id: ServiceId
    base_url: str
    timeout_s: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    api_key: Optional[str] = None
    admin_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def with_auth(self, api_key: Optional[str] = None, admin_key: Optional[str] = None) -> "ServiceConfig":
        """Return a new config with updated auth keys."""
        return ServiceConfig(
            id=self.id,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
            api_key=api_key or self.api_key,
            admin_key=admin_key or self.admin_key,
            headers=self.headers.copy(),
        )


# =============================================================================
# SERVICE CLIENT
# =============================================================================

class ServiceClient:
    """
    Low-level HTTP client for remote services.

    Provides:
    - Retry with exponential backoff
    - Configurable timeouts
    - Standard exception handling
    - Structured logging
    - API key authentication

    Usage:
        client = ServiceClient(config)
        result = client.get_json("/health")
        result = client.post_json("/predict", json={"prompt": "Hello"})
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._session = requests.Session()

        # Set default headers
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._session.headers.update(config.headers)

    @property
    def service_name(self) -> str:
        """Name of the service this client connects to."""
        return self.config.id.name

    @property
    def base_url(self) -> str:
        """Base URL of the service."""
        return self.config.base_url

    def _full_url(self, path: str) -> str:
        """Build full URL for a path."""
        if not path.startswith("/"):
            path = "/" + path
        return self.config.base_url.rstrip("/") + path

    def _get_headers(self, require_admin: bool = False) -> Dict[str, str]:
        """Get headers with appropriate API key."""
        headers = {}

        # Add API key if configured
        api_key = self.config.admin_key if require_admin else self.config.api_key
        if api_key:
            headers["X-API-Key"] = api_key

        return headers

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def get_json(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """
        Make a GET request and return JSON response.

        Args:
            path: URL path (e.g., "/health")
            params: Query parameters
            timeout: Request timeout (overrides config)

        Returns:
            Response JSON as dict

        Raises:
            ServiceUnavailable: Connection failed or timeout
            ServiceHttpError: HTTP 4xx/5xx
            ServiceDecodeError: Invalid JSON response
        """
        return self._request_json("GET", path, params=params, timeout=timeout)

    def post_json(
        self,
        path: str,
        json: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        require_admin: bool = False,
    ) -> dict:
        """
        Make a POST request with JSON body and return JSON response.

        Args:
            path: URL path
            json: Request body
            params: Query parameters
            timeout: Request timeout (overrides config)
            require_admin: Use admin API key

        Returns:
            Response JSON as dict
        """
        return self._request_json(
            "POST", path, json=json, params=params,
            timeout=timeout, require_admin=require_admin
        )

    def put_json(
        self,
        path: str,
        json: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        require_admin: bool = False,
    ) -> dict:
        """Make a PUT request with JSON body."""
        return self._request_json(
            "PUT", path, json=json,
            timeout=timeout, require_admin=require_admin
        )

    def delete(
        self,
        path: str,
        timeout: Optional[float] = None,
        require_admin: bool = False,
    ) -> dict:
        """Make a DELETE request."""
        return self._request_json(
            "DELETE", path,
            timeout=timeout, require_admin=require_admin
        )

    def health_check(self, path: str = "/health", timeout: float = 5.0) -> bool:
        """
        Quick health check - returns True if service responds.

        Does NOT retry - single attempt with short timeout.
        """
        try:
            url = self._full_url(path)
            resp = self._session.get(url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # =========================================================================
    # CORE REQUEST LOGIC
    # =========================================================================

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        require_admin: bool = False,
    ) -> dict:
        """
        Make HTTP request with retry logic.

        Retries on:
        - Connection errors
        - Timeouts
        - HTTP 5xx (server errors)

        Does NOT retry on:
        - HTTP 4xx (client errors)
        - JSON decode errors
        """
        url = self._full_url(path)
        timeout = timeout or self.config.timeout_s
        headers = self._get_headers(require_admin)

        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                # Make request
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                )

                # Check for HTTP errors
                if resp.status_code >= 500:
                    # Server error - retryable
                    last_exception = ServiceHttpError(resp.status_code, resp.text, url)
                    self._log_retry(attempt, url, f"HTTP {resp.status_code}")
                    continue

                if resp.status_code == 401 or resp.status_code == 403:
                    # Auth error - not retryable
                    raise ServiceAuthError(self.service_name, resp.text)

                if resp.status_code >= 400:
                    # Client error - not retryable
                    raise ServiceHttpError(resp.status_code, resp.text, url)

                # Success - decode JSON
                try:
                    return resp.json()
                except ValueError as e:
                    raise ServiceDecodeError(f"Invalid JSON from {url}: {e}", url)

            except (requests.ConnectionError, requests.Timeout) as e:
                last_exception = e
                self._log_retry(attempt, url, str(e))
                continue

            except (ServiceHttpError, ServiceDecodeError, ServiceAuthError):
                # Re-raise without retry
                raise

        # All retries exhausted
        logger.error(
            f"Service '{self.service_name}' unavailable after {self.config.max_retries} attempts: {url}"
        )
        raise ServiceUnavailable(self.service_name, url, last_exception)

    def _log_retry(self, attempt: int, url: str, reason: str):
        """Log retry attempt and sleep."""
        if attempt < self.config.max_retries - 1:
            sleep_time = self.config.backoff_factor ** attempt
            logger.warning(
                f"[{self.service_name}] Attempt {attempt + 1}/{self.config.max_retries} "
                f"failed ({reason}), retrying in {sleep_time:.1f}s..."
            )
            time.sleep(sleep_time)


# =============================================================================
# SERVICE REGISTRY & FACTORY
# =============================================================================

# Well-known services and their default ports (for fallback only)
_SERVICE_DEFAULTS = {
    "inference": {"port": 8765, "timeout": 120},
    "scheduler": {"port": 8766, "timeout": 30},
    "vault": {"port": 8767, "timeout": 30},
    "tavern": {"port": 8888, "timeout": 10},
    "monitor": {"port": 8081, "timeout": 10},
    "data_manager": {"port": 8080, "timeout": 60},
}


def get_service_config(service_name: str) -> ServiceConfig:
    """
    Get configuration for a service.

    Resolves URL from core.hosts registry, with fallback to defaults.
    Reads per-service env vars for customization:
        {SERVICE}_TIMEOUT_S
        {SERVICE}_MAX_RETRIES
        {SERVICE}_API_KEY
        {SERVICE}_ADMIN_KEY

    Args:
        service_name: Service identifier (e.g., "inference", "vault")

    Returns:
        ServiceConfig ready for use
    """
    # Try to get URL from host registry
    base_url = None
    try:
        from core.hosts import get_service_url
        base_url = get_service_url(service_name)
    except Exception:
        pass

    # Fallback to defaults if registry unavailable
    if not base_url:
        defaults = _SERVICE_DEFAULTS.get(service_name, {"port": 8080})
        port = defaults.get("port", 8080)
        base_url = f"http://localhost:{port}"
        logger.debug(f"Using fallback URL for {service_name}: {base_url}")

    # Get defaults for this service type
    defaults = _SERVICE_DEFAULTS.get(service_name, {})
    default_timeout = defaults.get("timeout", 30)

    # Read env overrides
    env_prefix = service_name.upper()
    timeout_s = float(os.getenv(f"{env_prefix}_TIMEOUT_S", str(default_timeout)))
    max_retries = int(os.getenv(f"{env_prefix}_MAX_RETRIES", "3"))
    api_key = os.getenv(f"{env_prefix}_API_KEY") or os.getenv(f"{env_prefix}_READ_KEY")
    admin_key = os.getenv(f"{env_prefix}_ADMIN_KEY")

    return ServiceConfig(
        id=ServiceId(name=service_name),
        base_url=base_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
        api_key=api_key,
        admin_key=admin_key,
    )


def get_service_client(service_name: str) -> ServiceClient:
    """
    Get a ServiceClient for a named service.

    This is the main entry point for making service calls.

    Args:
        service_name: Service identifier (e.g., "inference", "vault")

    Returns:
        ServiceClient configured for the service

    Example:
        client = get_service_client("inference")
        result = client.post_json("/v1/chat/completions", json={...})
    """
    config = get_service_config(service_name)
    return ServiceClient(config)


# Cached clients for singleton pattern
_clients: Dict[str, ServiceClient] = {}


def get_cached_client(service_name: str) -> ServiceClient:
    """
    Get a cached ServiceClient (singleton per service).

    Use this when you want to reuse connections.
    """
    if service_name not in _clients:
        _clients[service_name] = get_service_client(service_name)
    return _clients[service_name]


def clear_client_cache():
    """Clear the client cache (useful for testing)."""
    _clients.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def service_health(service_name: str, timeout: float = 5.0) -> bool:
    """
    Quick health check for a service.

    Args:
        service_name: Service to check
        timeout: Health check timeout

    Returns:
        True if service is healthy
    """
    try:
        client = get_service_client(service_name)
        return client.health_check(timeout=timeout)
    except Exception:
        return False


def service_status(service_name: str) -> Dict[str, Any]:
    """
    Get detailed status for a service.

    Returns:
        Dict with 'available', 'url', 'error' fields
    """
    try:
        config = get_service_config(service_name)
        client = ServiceClient(config)
        healthy = client.health_check(timeout=5.0)
        return {
            "service": service_name,
            "available": healthy,
            "url": config.base_url,
            "error": None,
        }
    except Exception as e:
        return {
            "service": service_name,
            "available": False,
            "url": None,
            "error": str(e),
        }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    service = sys.argv[1] if len(sys.argv) > 1 else "inference"

    print(f"\n=== Testing ServiceClient for '{service}' ===\n")

    # Get config
    config = get_service_config(service)
    print(f"URL: {config.base_url}")
    print(f"Timeout: {config.timeout_s}s")
    print(f"Retries: {config.max_retries}")
    print(f"API Key: {'configured' if config.api_key else 'not set'}")

    # Health check
    print(f"\nHealth check...")
    client = ServiceClient(config)
    if client.health_check():
        print(f"  {service} is healthy")
    else:
        print(f"  {service} is NOT responding")
        sys.exit(1)

    # Try a GET request
    print(f"\nTesting GET /health...")
    try:
        result = client.get_json("/health")
        print(f"  Response: {result}")
    except ServiceError as e:
        print(f"  Error: {e}")

    print("\n=== Done ===")
