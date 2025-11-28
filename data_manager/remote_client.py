#!/usr/bin/env python3
"""
Remote GPU Client - Interface to RTX 3090 inference server
Handles all communication with inference server

This client is built on core.services.ServiceClient for standardized
network handling, retries, and error semantics.
"""

import logging
from typing import Dict, List, Any, Optional

from core.services import (
    ServiceClient,
    ServiceConfig,
    ServiceId,
    ServiceError,
    ServiceHttpError,
    ServiceUnavailable,
    ServiceDecodeError,
    get_service_config,
)

logger = logging.getLogger(__name__)


class RemoteGPUClient:
    """
    Client for remote RTX 3090 inference server.

    Built on ServiceClient for consistent retry, timeout, and error handling.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, timeout: int = 300):
        """
        Initialize RemoteGPUClient.

        Args:
            host: Remote host (defaults to host registry)
            port: Remote port (defaults to host registry)
            timeout: Request timeout in seconds
        """
        # Get base config from service registry
        config = get_service_config("inference")

        # Override with constructor parameters if provided
        if host is not None or port is not None:
            # Use host registry for missing values
            try:
                from core.hosts import get_host
                inference_host = get_host("3090")
                host = host or inference_host.host
                port = port or inference_host.services.get("inference", {}).get("port", 8765)
            except Exception:
                host = host or "inference.local"
                port = port or 8765
            config.base_url = f"http://{host}:{port}"

        config.timeout_s = float(timeout)

        # Store for backward compatibility
        self._host = host
        self._port = port
        self.timeout = timeout

        # Create underlying ServiceClient
        self._client = ServiceClient(config)

        logger.info(f"RemoteGPUClient initialized: {config.base_url}")

    @property
    def host(self) -> str:
        """Host of the remote server."""
        # Extract host from base_url
        url = self._client.base_url
        # http://host:port -> host
        return url.replace("http://", "").replace("https://", "").split(":")[0]

    @property
    def port(self) -> int:
        """Port of the remote server."""
        url = self._client.base_url
        try:
            return int(url.split(":")[-1])
        except (ValueError, IndexError):
            return 8765

    @property
    def base_url(self) -> str:
        """Base URL of the remote server."""
        return self._client.base_url

    def health_check(self) -> Dict[str, Any]:
        """Check if remote server is alive and get GPU stats"""
        try:
            return self._client.get_json("/health", timeout=10)
        except ServiceError as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get detailed GPU statistics"""
        try:
            return self._client.get_json("/gpu", timeout=10)
        except ServiceError as e:
            logger.error(f"GPU stats failed: {e}")
            return {"error": str(e)}

    def list_models(self) -> List[str]:
        """List available models on remote server"""
        try:
            data = self._client.get_json("/models", timeout=10)
            return data.get("models", [])
        except ServiceError as e:
            logger.error(f"List models failed: {e}")
            return []

    def generate_data(self, count: int, seed: Optional[int] = None,
                     payload: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Generate training data via remote server

        Args:
            count: Number of examples to generate
            seed: Random seed for reproducibility
            payload: Additional generation parameters

        Returns:
            List of generated examples
        """
        request_data = {"count": count}
        if seed is not None:
            request_data["seed"] = seed
        if payload:
            request_data.update(payload)

        try:
            result = self._client.post_json("/generate", json=request_data)

            # Handle different response formats
            if isinstance(result, dict):
                if "examples" in result:
                    return result["examples"]
                elif "puzzles" in result:
                    return result["puzzles"]
                elif "data" in result:
                    return result["data"]
                else:
                    # Assume the dict itself is the data
                    return [result]
            elif isinstance(result, list):
                return result
            else:
                logger.warning(f"Unexpected response format: {type(result)}")
                return []

        except ServiceUnavailable as e:
            logger.error(f"Data generation failed: {e}")
            raise RuntimeError(f"Remote generation failed: {e}") from e
        except ServiceDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise RuntimeError(f"Invalid response from server: {e}") from e
        except ServiceError as e:
            logger.error(f"Data generation failed: {e}")
            raise RuntimeError(f"Remote generation failed: {e}") from e

    def inference(self, prompt: str, max_tokens: int = 512,
                  temperature: float = 0.7) -> Dict[str, Any]:
        """
        Run inference on remote GPU

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Inference result
        """
        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            return self._client.post_json("/inference", json=request_data)
        except ServiceError as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Remote inference failed: {e}") from e

    def is_available(self) -> bool:
        """Check if remote server is available"""
        health = self.health_check()
        return health.get("status") == "ok"

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB"""
        stats = self.get_gpu_stats()
        if "error" in stats:
            return {"allocated": 0.0, "reserved": 0.0, "total": 24.0}

        gpu_info = stats.get("gpu", {})
        return {
            "allocated": gpu_info.get("memory_allocated_gb", 0.0),
            "reserved": gpu_info.get("memory_reserved_gb", 0.0),
            "total": 24.0  # RTX 3090
        }

    def __repr__(self):
        return f"RemoteGPUClient({self.host}:{self.port})"


# Re-export exceptions for convenience
__all__ = [
    "RemoteGPUClient",
    "ServiceError",
    "ServiceHttpError",
    "ServiceUnavailable",
    "ServiceDecodeError",
]
