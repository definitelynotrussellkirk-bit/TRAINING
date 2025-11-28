"""
Job Store Client - HTTP client for the job store API.

Use this client to interact with the job store from any machine:
- Submit jobs
- Query job status
- Claim jobs (for workers)

Usage:
    from jobs.client import JobStoreClient, get_client

    client = get_client()

    # Submit a job
    from jobs import eval_job
    job_id = client.submit(eval_job("bin", level=5))

    # Check status
    job = client.get(job_id)
    print(f"Status: {job['status']}")

    # For workers - claim a job
    job = client.claim("worker_1", ["eval_worker"])
    if job:
        # execute...
        client.complete(job["job_id"], {"accuracy": 0.95})
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from guild.job_types import Job, JobSpec

logger = logging.getLogger("job_client")


class JobStoreClient:
    """
    HTTP client for the job store API.

    Provides a simple interface to the job store running on VaultKeeper.
    """

    def __init__(self, server_url: Optional[str] = None):
        """
        Initialize job store client.

        Args:
            server_url: Job server URL. Defaults to JOB_SERVER_URL env
                       or http://localhost:8767
        """
        self.server_url = (
            server_url or
            os.environ.get("JOB_SERVER_URL") or
            self._auto_detect_url()
        ).rstrip("/")

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _auto_detect_url(self) -> str:
        """Auto-detect job server URL based on environment."""
        try:
            from core.hosts import get_service_url
            url = get_service_url("vault")
            if url:
                return url.replace("/api", "")
        except ImportError:
            pass
        return "http://localhost:8767"

    # =========================================================================
    # SUBMISSION
    # =========================================================================

    def submit(self, spec: JobSpec) -> str:
        """
        Submit a job to the store.

        Args:
            spec: Job specification

        Returns:
            Job ID

        Raises:
            RuntimeError: If submission fails
        """
        response = self.session.post(
            f"{self.server_url}/api/jobs",
            json=spec.to_dict(),
            timeout=10,
        )

        if response.status_code not in (200, 201):
            raise RuntimeError(f"Submit failed: {response.text}")

        data = response.json()
        return data["job_id"]

    def submit_and_wait(
        self,
        spec: JobSpec,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Submit a job and wait for completion.

        Args:
            spec: Job specification
            timeout: Maximum wait time
            poll_interval: Polling interval

        Returns:
            Job data including result

        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If job failed
        """
        import time

        job_id = self.submit(spec)
        start_time = time.time()

        while time.time() - start_time < timeout:
            job = self.get(job_id)
            if not job:
                raise RuntimeError(f"Job {job_id} not found")

            status = job.get("status")
            if status == "completed":
                return job
            elif status in ("failed", "cancelled", "timeout"):
                error = job.get("result", {}).get("error", "Unknown error")
                raise RuntimeError(f"Job {job_id} {status}: {error}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        response = self.session.get(
            f"{self.server_url}/api/jobs/{job_id}",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()
        return None

    def list(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filters."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if job_type:
            params["type"] = job_type

        response = self.session.get(
            f"{self.server_url}/api/jobs",
            params=params,
            timeout=10,
        )

        if response.status_code == 200:
            return response.json().get("jobs", [])
        return []

    def stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        response = self.session.get(
            f"{self.server_url}/api/jobs/stats",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()
        return {}

    # =========================================================================
    # WORKER OPERATIONS
    # =========================================================================

    def claim(
        self,
        device_id: str,
        roles: List[str],
        lease_duration: int = 300,
    ) -> Optional[Dict[str, Any]]:
        """
        Claim the next available job.

        Args:
            device_id: ID of claiming device
            roles: Roles this device can handle
            lease_duration: Lease duration in seconds

        Returns:
            Job data if claimed, None if no jobs available
        """
        response = self.session.post(
            f"{self.server_url}/api/jobs/claim",
            json={
                "device_id": device_id,
                "roles": roles,
                "lease_duration": lease_duration,
            },
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("claimed"):
                return data["job"]
        return None

    def running(self, job_id: str, device_id: str) -> bool:
        """Mark a job as running."""
        response = self.session.post(
            f"{self.server_url}/api/jobs/{job_id}/running",
            json={"device_id": device_id},
            timeout=5,
        )
        return response.status_code == 200

    def complete(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Mark a job as completed."""
        response = self.session.post(
            f"{self.server_url}/api/jobs/{job_id}/complete",
            json={"result": result},
            timeout=10,
        )
        return response.status_code == 200

    def failed(self, job_id: str, error: str) -> bool:
        """Mark a job as failed."""
        response = self.session.post(
            f"{self.server_url}/api/jobs/{job_id}/failed",
            json={"error": error},
            timeout=5,
        )
        return response.status_code == 200

    def release(self, job_id: str) -> bool:
        """Release a claimed job back to queue."""
        response = self.session.post(
            f"{self.server_url}/api/jobs/{job_id}/release",
            timeout=5,
        )
        return response.status_code == 200

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        response = self.session.post(
            f"{self.server_url}/api/jobs/{job_id}/cancel",
            timeout=5,
        )
        return response.status_code == 200

    # =========================================================================
    # HEALTH
    # =========================================================================

    def health_check(self) -> bool:
        """Check if server is reachable."""
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def is_available(self) -> bool:
        """Alias for health_check."""
        return self.health_check()


# =============================================================================
# SINGLETON
# =============================================================================

_client: Optional[JobStoreClient] = None


def get_client(server_url: Optional[str] = None) -> JobStoreClient:
    """Get or create the job store client singleton."""
    global _client
    if _client is None:
        _client = JobStoreClient(server_url)
    return _client


def reset_client():
    """Reset the client singleton."""
    global _client
    _client = None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Job Store Client")
    parser.add_argument(
        "command",
        choices=["stats", "list", "get", "submit", "health"],
        help="Command to run",
    )
    parser.add_argument("--server", help="Server URL")
    parser.add_argument("--job-id", help="Job ID (for get)")
    parser.add_argument("--skill", default="bin", help="Skill for eval")
    parser.add_argument("--level", type=int, default=5, help="Level for eval")

    args = parser.parse_args()

    client = JobStoreClient(args.server)

    if args.command == "health":
        ok = client.health_check()
        print(f"Server: {client.server_url}")
        print(f"Status: {'OK' if ok else 'UNREACHABLE'}")

    elif args.command == "stats":
        stats = client.stats()
        print(json.dumps(stats, indent=2))

    elif args.command == "list":
        jobs = client.list(limit=20)
        for job in jobs:
            print(f"{job['job_id']}: {job['spec']['job_type']} [{job['status']}]")

    elif args.command == "get":
        if not args.job_id:
            print("Error: --job-id required")
        else:
            job = client.get(args.job_id)
            print(json.dumps(job, indent=2) if job else "Not found")

    elif args.command == "submit":
        from jobs import eval_job
        spec = eval_job(args.skill, args.level)
        job_id = client.submit(spec)
        print(f"Submitted: {job_id}")
