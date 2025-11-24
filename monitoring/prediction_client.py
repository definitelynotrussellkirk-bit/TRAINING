#!/usr/bin/env python3
"""
PredictionClient - Standardized client for 3090 inference API

All monitoring scripts should use this instead of raw requests to ensure:
- Consistent retry logic
- Proper error handling
- Unified logging
- Standard request/response format
"""

import requests
import time
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class PredictionClient:
    """
    Standardized client for 3090 inference server.

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
        base_url: str = "http://192.168.x.x:8765",
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0
    ):
        """
        Initialize prediction client.

        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_backoff: Exponential backoff multiplier
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        logger.info(f"PredictionClient initialized: {self.base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "deployed",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
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
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        return self._request_with_retry("POST", url, json=payload)

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
        url = f"{self.base_url}/models/info"
        return self._request_with_retry("GET", url)

    def reload_model(self) -> Dict[str, Any]:
        """
        Trigger model reload from deployed checkpoint.

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
        url = f"{self.base_url}/models/reload"
        return self._request_with_retry("POST", url)

    def health_check(self) -> bool:
        """
        Check if server is responding.

        Returns:
            True if server is healthy
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

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
        url = f"{self.base_url}/health"
        return self._request_with_retry("GET", url)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL
            **kwargs: Passed to requests.request()

        Returns:
            Response JSON

        Raises:
            requests.exceptions.RequestException: After all retries exhausted
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Set timeout if not specified
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = self.timeout

                # Make request
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()

                # Parse JSON
                return response.json()

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    f"Request timeout (attempt {attempt+1}/{self.max_retries}): {url}"
                )

            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.warning(
                    f"HTTP error {response.status_code} (attempt {attempt+1}/{self.max_retries}): {url}"
                )

                # Don't retry 4xx errors (client errors)
                if 400 <= response.status_code < 500:
                    raise

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(
                    f"Request failed (attempt {attempt+1}/{self.max_retries}): {url} - {e}"
                )

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                sleep_time = self.retry_backoff ** attempt
                logger.debug(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        # All retries exhausted
        logger.error(f"All {self.max_retries} attempts failed for {url}")
        raise last_error


# Global client instance
_client = None


def get_client(base_url: str = "http://192.168.x.x:8765") -> PredictionClient:
    """
    Get global PredictionClient instance (singleton pattern).

    Args:
        base_url: API base URL (only used on first call)

    Returns:
        PredictionClient instance
    """
    global _client
    if _client is None:
        _client = PredictionClient(base_url=base_url)
    return _client


if __name__ == "__main__":
    # Test client
    logging.basicConfig(level=logging.INFO)

    client = PredictionClient()

    print("Testing PredictionClient...\n")

    # Test 1: Health check
    print("1. Health check:")
    if client.health_check():
        print("  ✅ Server is healthy")
        health = client.get_health_details()
        print(f"  GPU: {health['gpu']['device_name']}")
        print(f"  Active model: {health['active_model']}")
    else:
        print("  ❌ Server is down")

    # Test 2: Model info
    print("\n2. Model info:")
    info = client.get_model_info()
    print(f"  Loaded: {info['loaded']}")
    if info['loaded']:
        print(f"  Model ID: {info['model_id']}")
        print(f"  Step: {info['checkpoint_step']}")
        print(f"  VRAM: {info['vram_usage_gb']}GB")

    # Test 3: Chat completion (optional - can be slow)
    print("\n3. Chat completion:")
    try:
        response = client.chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=20
        )
        content = response['choices'][0]['message']['content']
        print(f"  Response: {content[:50]}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n✅ Client tests complete")
