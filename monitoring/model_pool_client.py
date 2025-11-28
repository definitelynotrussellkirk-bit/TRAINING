#!/usr/bin/env python3
"""
Model Pool Client - Shared helper for accessing the multi-model inference pool

Usage:
    from monitoring.model_pool_client import get_active_model, call_inference

    # Get first available model
    model_id = get_active_model()

    # Call inference with explicit model
    result = call_inference(prompt, model_id=model_id)
"""

import os
import requests
from typing import Dict, Optional, List, Any

DEFAULT_API_URL = "http://inference.local:8765"
DEFAULT_API_KEY = "admin123"


def get_api_key() -> str:
    """Get API key from environment or default"""
    return os.environ.get("INFERENCE_ADMIN_KEY", DEFAULT_API_KEY)


def get_api_url() -> str:
    """Get API URL from environment or default"""
    return os.environ.get("INFERENCE_API_URL", DEFAULT_API_URL)


def get_pool_status(api_url: str = None) -> Dict[str, Any]:
    """
    Get full status of the model pool.

    Returns:
        {
            "loaded_count": 1,
            "max_models": 3,
            "total_vram_mb": 1192.1,
            "models": [{"model_id": "checkpoint-175000", ...}]
        }
    """
    api_url = api_url or get_api_url()
    try:
        response = requests.get(
            f"{api_url}/models/pool",
            headers={"X-API-Key": get_api_key()},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}", "models": []}
    except Exception as e:
        return {"error": str(e), "models": []}


def get_active_model(api_url: str = None) -> Optional[str]:
    """
    Get the first available model from the pool.

    Returns:
        Model ID (e.g., "checkpoint-175000") or None if no models loaded
    """
    status = get_pool_status(api_url)
    if status.get("models"):
        return status["models"][0]["model_id"]
    return None


def get_all_models(api_url: str = None) -> List[str]:
    """
    Get list of all loaded model IDs.

    Returns:
        List of model IDs (e.g., ["checkpoint-175000", "Qwen3-0.6B"])
    """
    status = get_pool_status(api_url)
    return [m["model_id"] for m in status.get("models", [])]


def load_model(model_id: str, path: str = None, api_url: str = None) -> Dict[str, Any]:
    """
    Load a model into the pool.

    Args:
        model_id: Model identifier
        path: Full path to model (optional, uses models_dir/model_id if not specified)

    Returns:
        Load result with pool status
    """
    api_url = api_url or get_api_url()
    try:
        params = {"model_id": model_id}
        if path:
            params["path"] = path

        response = requests.post(
            f"{api_url}/models/pool/load",
            params=params,
            headers={"X-API-Key": get_api_key()},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def call_inference(
    prompt: str,
    model_id: str = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    api_url: str = None
) -> Dict[str, Any]:
    """
    Call the inference API with explicit model selection.

    Args:
        prompt: Text prompt
        model_id: Model to use (REQUIRED - will fail if not specified and no models loaded)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        {
            "generated_text": "...",
            "model_id": "checkpoint-175000",
            "model_path": "/path/to/model",
            ...
        }
        or {"error": "..."} on failure
    """
    api_url = api_url or get_api_url()

    # Get model if not specified
    if model_id is None:
        model_id = get_active_model(api_url)
        if model_id is None:
            return {"error": "No model loaded in pool and no model_id specified"}

    try:
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            headers={"X-API-Key": get_api_key()},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            # Extract from OpenAI format
            text = ""
            if "choices" in result and result["choices"]:
                text = result["choices"][0]["message"]["content"]

            return {
                "generated_text": text,
                "model_id": result.get("model"),
                "model_path": result.get("model_path"),
                "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": result.get("usage", {}).get("total_tokens", 0)
            }
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Quick test
    print("Model Pool Client Test")
    print("=" * 40)

    status = get_pool_status()
    print(f"Pool status: {status}")

    model = get_active_model()
    print(f"Active model: {model}")

    if model:
        result = call_inference("What is 2+2?", model_id=model, max_tokens=50)
        print(f"Inference result: {result}")
