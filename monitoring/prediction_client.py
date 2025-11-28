#!/usr/bin/env python3
"""
PredictionClient - Standardized client for 3090 inference API.

All monitoring scripts should use this instead of raw requests to ensure:
- Consistent retry logic
- Proper error handling
- Unified logging
- Standard request/response format

This client is built on core.services.ServiceClient for standardized
network handling, retries, and error semantics.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from core.services import (
    ServiceClient,
    ServiceError,
    ServiceHttpError,
    ServiceUnavailable,
    ServiceDecodeError,
    ServiceAuthError,
    get_service_config,
)

logger = logging.getLogger(__name__)

# Default API key from environment
DEFAULT_READ_KEY = os.environ.get("INFERENCE_READ_KEY", "")
DEFAULT_ADMIN_KEY = os.environ.get("INFERENCE_ADMIN_KEY", "")


class PredictionClient:
    """
    Standardized client for 3090 inference server.

    Built on ServiceClient for consistent retry, timeout, and error handling.

    Example usage:
        client = PredictionClient()

        # Chat completion
        response = client.chat(messages=[
            {"role": "user", "content": "Hello"}
        ])

        # Check model info
        info = client.get_model_info()

        # Reload model
        client.reload_model()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        api_key: Optional[str] = None,
        admin_key: Optional[str] = None
    ):
        """
        Initialize prediction client.

        Args:
            base_url: API base URL (defaults to host registry)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_backoff: Exponential backoff multiplier
            api_key: Read-level API key (defaults to INFERENCE_READ_KEY env var)
            admin_key: Admin-level API key (defaults to INFERENCE_ADMIN_KEY env var)
        """
        # Get base config from service registry
        config = get_service_config("inference")

        # Override with constructor parameters
        if base_url:
            config.base_url = base_url
        config.timeout_s = float(timeout)
        config.max_retries = max_retries
        config.backoff_factor = retry_backoff
        config.api_key = api_key or DEFAULT_READ_KEY or config.api_key
        config.admin_key = admin_key or DEFAULT_ADMIN_KEY or config.admin_key

        # Create underlying ServiceClient
        self._client = ServiceClient(config)

        logger.info(f"PredictionClient initialized: {config.base_url}")
        if config.api_key:
            logger.info("  Read API key: configured")
        if config.admin_key:
            logger.info("  Admin API key: configured")

    @property
    def base_url(self) -> str:
        """Base URL of the inference server."""
        return self._client.base_url

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (if None, queries /models/info for currently loaded model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            {
                'id': 'chatcmpl-...',
                'choices': [{
                    'message': {'role': 'assistant', 'content': '...'},
                    'finish_reason': 'stop'
                }],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 50}
            }

        Raises:
            ServiceUnavailable: Server not reachable
            ServiceHttpError: HTTP error from server
        """
        # If model not specified, get currently loaded model
        if model is None:
            try:
                info = self.get_model_info()
                model = info.get('model_id', 'unknown')
            except ServiceError:
                model = "unknown"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        return self._client.post_json("/v1/chat/completions", json=payload)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get currently loaded model information.

        Returns:
            {
                'loaded': bool,
                'model_id': str,
                'checkpoint_step': Optional[int],
                'loaded_at': str (ISO timestamp),
                'vram_usage_gb': float
            }
        """
        return self._client.get_json("/models/info")

    def reload_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a specific checkpoint by path.

        Requires admin API key (INFERENCE_ADMIN_KEY).

        Args:
            model_path: Full path to checkpoint

        Returns:
            {
                'status': 'reloaded',
                'model_id': str,
                'checkpoint_step': Optional[int],
                'loaded_from': str,
                'loaded_at': str (ISO timestamp),
                'vram_usage_gb': float
            }
        """
        return self._client.post_json(
            "/models/reload",
            json={"model_path": model_path},
            require_admin=True
        )

    def health_check(self) -> bool:
        """
        Check if server is responding.

        Returns:
            True if server is healthy
        """
        return self._client.health_check()

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information.

        Returns:
            {
                'status': 'ok',
                'gpu': {...},
                'active_model': str,
                'worker_busy': bool
            }
        """
        return self._client.get_json("/health")

    def health(self) -> Dict[str, Any]:
        """Alias for get_health_details (for backward compatibility)."""
        return self.get_health_details()

    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics (if available)."""
        return self._client.get_json("/metrics")


# Global client instance
_client: Optional[PredictionClient] = None


def get_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    admin_key: Optional[str] = None
) -> PredictionClient:
    """
    Get global PredictionClient instance (singleton pattern).

    Args:
        base_url: API base URL (defaults to host registry; only used on first call)
        api_key: Read API key (defaults to INFERENCE_READ_KEY env var)
        admin_key: Admin API key (defaults to INFERENCE_ADMIN_KEY env var)

    Returns:
        PredictionClient instance
    """
    global _client
    if _client is None:
        _client = PredictionClient(
            base_url=base_url,
            api_key=api_key,
            admin_key=admin_key
        )
    return _client


# Re-export exceptions for convenience
__all__ = [
    "PredictionClient",
    "get_client",
    "ServiceError",
    "ServiceHttpError",
    "ServiceUnavailable",
    "ServiceDecodeError",
    "ServiceAuthError",
]


if __name__ == "__main__":
    # Test client
    logging.basicConfig(level=logging.INFO)

    client = PredictionClient()

    print("Testing PredictionClient...\n")

    # Test 1: Health check
    print("1. Health check:")
    if client.health_check():
        print("  Server is healthy")
        health = client.get_health_details()
        print(f"  GPU: {health.get('gpu', {}).get('device_name', 'unknown')}")
        print(f"  Active model: {health.get('active_model', 'unknown')}")
    else:
        print("  Server is down")

    # Test 2: Model info
    print("\n2. Model info:")
    try:
        info = client.get_model_info()
        print(f"  Loaded: {info.get('loaded', False)}")
        if info.get('loaded'):
            print(f"  Model ID: {info.get('model_id')}")
            print(f"  Step: {info.get('checkpoint_step')}")
            print(f"  VRAM: {info.get('vram_usage_gb')}GB")
    except ServiceError as e:
        print(f"  Error: {e}")

    # Test 3: Chat completion (optional - can be slow)
    print("\n3. Chat completion:")
    try:
        response = client.chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=20
        )
        content = response['choices'][0]['message']['content']
        print(f"  Response: {content[:50]}")
    except ServiceError as e:
        print(f"  Error: {e}")

    print("\nClient tests complete")
