"""
Client library for RTX 3090 Inference API

Provides Python interface to all remote API endpoints
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore


class InferenceAPIClient:
    """Client for RTX 3090 Inference API"""

    def __init__(self, base_url: Optional[str] = None):
        if requests is None:
            raise ImportError("requests library required. Install: pip install requests")

        # Use host registry for default URL
        if base_url is None:
            try:
                from core.hosts import get_service_url
                base_url = get_service_url("inference")
            except Exception:
                # Fallback if host registry unavailable
                base_url = "http://192.168.x.x:8765"

        self.base_url = base_url

    # ===== Health & Info =====

    def health(self) -> Dict[str, Any]:
        """Get system health and status"""
        return requests.get(f"{self.base_url}/health").json()

    def info(self) -> Dict[str, Any]:
        """Get system information"""
        return requests.get(f"{self.base_url}/info").json()

    def version(self) -> Dict[str, Any]:
        """Get version information"""
        return requests.get(f"{self.base_url}/version").json()

    def is_healthy(self) -> bool:
        """Quick health check"""
        try:
            resp = self.health()
            return resp.get("status") == "ok"
        except Exception:
            return False

    # ===== GPU & System Stats =====

    def gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        return requests.get(f"{self.base_url}/gpu").json()

    def system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return requests.get(f"{self.base_url}/system").json()

    # ===== Model Management =====

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        resp = requests.get(f"{self.base_url}/models").json()
        return resp.get("models", [])

    def register_model(
        self,
        model_id: str,
        source: str = "4090",
        tags: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new model checkpoint"""
        return requests.post(
            f"{self.base_url}/models/register",
            json={"id": model_id, "source": source, "tags": tags}
        ).json()

    def set_active_model(self, model_id: str) -> Dict[str, Any]:
        """Set active model for inference"""
        return requests.post(
            f"{self.base_url}/models/set_active",
            json={"id": model_id}
        ).json()

    def get_active_model(self) -> Dict[str, Any]:
        """Get currently active model"""
        return requests.get(f"{self.base_url}/models/active").json()

    # ===== Inference =====

    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        mode: str = "normal"
    ) -> Dict[str, Any]:
        """Queue text generation job"""
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "prompt": prompt,
                "model_id": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "mode": mode
            }
        ).json()

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "Qwen3-0.6B",
        temperature: float = 0.7,
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """OpenAI-compatible chat completion"""
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        ).json()

    # ===== Evaluation Jobs =====

    def create_eval_job(
        self,
        model_id: str,
        name: str,
        dataset_ref: str,
        metrics: Optional[List[str]] = None,
        max_samples: int = 1000,
        per_example_logging: bool = True
    ) -> Dict[str, Any]:
        """Create evaluation job"""
        return requests.post(
            f"{self.base_url}/eval/jobs",
            json={
                "model_id": model_id,
                "name": name,
                "dataset_ref": dataset_ref,
                "metrics": metrics or ["accuracy"],
                "max_samples": max_samples,
                "per_example_logging": per_example_logging
            }
        ).json()

    def get_eval_job(self, job_id: str) -> Dict[str, Any]:
        """Get evaluation job status and results"""
        return requests.get(f"{self.base_url}/eval/jobs/{job_id}").json()

    def wait_for_eval(
        self,
        job_id: str,
        poll_interval: int = 10,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Wait for evaluation job to complete"""
        start_time = datetime.now()
        while True:
            result = self.get_eval_job(job_id)
            status = result.get("status")

            if status in ["done", "failed"]:
                return result

            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Eval job {job_id} timed out after {timeout}s")

            time.sleep(poll_interval)

    # ===== Data Generation Jobs =====

    def create_data_gen_job(
        self,
        model_id: str,
        strategy: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create data generation job"""
        return requests.post(
            f"{self.base_url}/data_gen/jobs",
            json={
                "model_id": model_id,
                "strategy": strategy,
                "config": config
            }
        ).json()

    def get_data_gen_job(self, job_id: str) -> Dict[str, Any]:
        """Get data generation job status"""
        return requests.get(f"{self.base_url}/data_gen/jobs/{job_id}").json()

    # ===== Job Management =====

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filters"""
        params = {"limit": limit}
        if job_type:
            params["type"] = job_type
        if status:
            params["status"] = status

        resp = requests.get(f"{self.base_url}/jobs", params=params).json()
        return resp.get("jobs", [])

    def job_stats(self) -> Dict[str, Any]:
        """Get job queue statistics"""
        return requests.get(f"{self.base_url}/jobs/stats").json()

    # ===== Power Management =====

    def get_power_profile(self) -> Dict[str, Any]:
        """Get current power profile"""
        return requests.get(f"{self.base_url}/settings/power_profile").json()

    def set_power_profile(self, profile: str) -> Dict[str, Any]:
        """Set power profile (quiet/normal/max)"""
        return requests.post(
            f"{self.base_url}/settings/power_profile",
            params={"profile": profile}
        ).json()

    # ===== Logs =====

    def get_logs(self, component: str, lines: int = 100) -> List[str]:
        """Get recent logs for a component"""
        resp = requests.get(
            f"{self.base_url}/logs/{component}",
            params={"lines": lines}
        ).json()
        return resp.get("lines", [])

    # ===== Helper Methods =====

    def print_status(self) -> None:
        """Print comprehensive system status"""
        print("=== RTX 3090 Inference Server Status ===\n")

        # Health
        health = self.health()
        print(f"Status: {health.get('status')}")
        print(f"Active Model: {health.get('active_model')}")
        print(f"Worker Busy: {health.get('worker_busy')}\n")

        # GPU
        gpu = self.gpu_stats()
        print("GPU:")
        print(f"  Temperature: {gpu.get('temperature_gpu')}°C")
        print(f"  Power: {gpu.get('power_draw_w')}W / {gpu.get('power_limit_w')}W")
        print(f"  VRAM: {gpu.get('memory_used_mb')}MB / {gpu.get('memory_total_mb')}MB")
        print(f"  Utilization: {gpu.get('utilization_gpu')}%\n")

        # Jobs
        stats = self.job_stats()
        print("Job Queue:")
        print(f"  Pending: {stats.get('pending')}")
        print(f"  Running: {stats.get('running')}")
        print(f"  Done: {stats.get('done')}")
        print(f"  Failed: {stats.get('failed')}")
        print(f"  Last Hour: {stats.get('jobs_last_hour')}\n")


if __name__ == "__main__":
    # Demo usage
    client = InferenceAPIClient()

    if client.is_healthy():
        print("✓ API is healthy\n")
        client.print_status()
    else:
        print("✗ API is not responding")
