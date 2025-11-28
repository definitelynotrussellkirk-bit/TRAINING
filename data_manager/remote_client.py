#!/usr/bin/env python3
"""
Remote GPU Client - Interface to RTX 3090 inference server
Handles all communication with inference server
"""

import json
import logging
from typing import Dict, List, Any, Optional
from urllib import request, error
from datetime import datetime

logger = logging.getLogger(__name__)


class RemoteGPUClient:
    """Client for remote RTX 3090 inference server"""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, timeout: int = 300):
        # Use host registry for defaults
        if host is None or port is None:
            try:
                from core.hosts import get_host
                inference_host = get_host("3090")
                if host is None:
                    host = inference_host.host
                if port is None:
                    port = inference_host.services.get("inference", {}).get("port", 8765)
            except Exception:
                # Fallback if host registry unavailable
                host = host or "192.168.x.x"
                port = port or 8765

        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def health_check(self) -> Dict[str, Any]:
        """Check if remote server is alive and get GPU stats"""
        try:
            url = f"{self.base_url}/health"
            with request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get detailed GPU statistics"""
        try:
            url = f"{self.base_url}/gpu"
            with request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as e:
            logger.error(f"GPU stats failed: {e}")
            return {"error": str(e)}

    def list_models(self) -> List[str]:
        """List available models on remote server"""
        try:
            url = f"{self.base_url}/models"
            with request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("models", [])
        except error.URLError as e:
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
        url = f"{self.base_url}/generate"

        request_data = {"count": count}
        if seed is not None:
            request_data["seed"] = seed
        if payload:
            request_data.update(payload)

        data = json.dumps(request_data).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        try:
            req = request.Request(url, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                result = json.loads(body)

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

        except error.URLError as e:
            logger.error(f"Data generation failed: {e}")
            raise RuntimeError(f"Remote generation failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise RuntimeError(f"Invalid response from server: {e}") from e

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
        url = f"{self.base_url}/inference"

        request_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        data = json.dumps(request_data).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        try:
            req = request.Request(url, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as e:
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
