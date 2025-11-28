"""
Claiming Worker - Worker that claims jobs from central store (pull model).

Unlike push-based workers that receive jobs via HTTP, ClaimingWorker
actively polls the job server and claims work. This is ideal for:
- Mac minis that may come and go
- Workers behind NAT/firewalls
- Environments where the dispatcher can't reach workers directly

Usage:
    # Start on a Mac mini
    python3 -m workers.claiming_worker \
        --device macmini_eval_1 \
        --server http://trainer.local:8767

    # With specific roles
    python3 -m workers.claiming_worker \
        --device macmini_eval_1 \
        --roles eval_worker,data_forge

Features:
- Atomic job claiming with lease-based locking
- Automatic retry on failure
- Graceful shutdown
- Heartbeat/status reporting
"""

import json
import logging
import os
import signal
import socket
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from guild.job_types import JobError, JobErrorCode

logger = logging.getLogger("claiming_worker")

# Version for registration
VERSION = "2025.11.28"


@dataclass
class ClaimingWorkerConfig:
    """Configuration for claiming worker."""
    device_id: str
    roles: List[str]
    server_url: str = "http://localhost:8767"
    claim_interval: float = 5.0         # Seconds between claim attempts
    lease_duration: int = 300           # Lease duration in seconds
    max_consecutive_errors: int = 10    # Shutdown after this many errors
    inference_url: Optional[str] = None  # For eval workers


class JobStoreClient:
    """Client for the job store API."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()

    def register(
        self,
        device_id: str,
        worker_kind: str,
        roles: List[str],
        version: str,
        hostname: str,
    ) -> Dict[str, Any]:
        """Register worker with server."""
        try:
            response = self.session.post(
                f"{self.server_url}/api/jobs/workers/register",
                json={
                    "device_id": device_id,
                    "worker_kind": worker_kind,
                    "roles": roles,
                    "version": version,
                    "hostname": hostname,
                },
                timeout=10,
            )
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Registration failed: {e}")
            return {"registered": False, "error": str(e)}

    def heartbeat(self, worker_id: str, active_jobs: int = 0) -> bool:
        """Send heartbeat to server."""
        try:
            response = self.session.post(
                f"{self.server_url}/api/jobs/workers/heartbeat",
                json={
                    "worker_id": worker_id,
                    "active_jobs": active_jobs,
                    "status": "online",
                },
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def claim_next(
        self,
        device_id: str,
        roles: List[str],
        lease_duration: int = 300,
    ) -> Optional[Dict[str, Any]]:
        """Claim the next available job."""
        try:
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

        except requests.RequestException as e:
            logger.warning(f"Claim failed: {e}")
            return None

    def mark_running(self, job_id: str, device_id: str) -> bool:
        """Mark job as running."""
        try:
            response = self.session.post(
                f"{self.server_url}/api/jobs/{job_id}/running",
                json={"device_id": device_id},
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Mark running failed: {e}")
            return False

    def mark_complete(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Mark job as completed."""
        try:
            response = self.session.post(
                f"{self.server_url}/api/jobs/{job_id}/complete",
                json={"result": result},
                timeout=10,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Mark complete failed: {e}")
            return False

    def mark_failed(
        self,
        job_id: str,
        error: str,
        error_code: Optional[JobErrorCode] = None,
    ) -> bool:
        """Mark job as failed with structured error code."""
        try:
            payload = {
                "error": error,
                "error_code": (error_code or JobErrorCode.UNKNOWN).value,
            }
            response = self.session.post(
                f"{self.server_url}/api/jobs/{job_id}/failed",
                json=payload,
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Mark failed failed: {e}")
            return False

    def release(self, job_id: str) -> bool:
        """Release job back to queue."""
        try:
            response = self.session.post(
                f"{self.server_url}/api/jobs/{job_id}/release",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logger.warning(f"Release failed: {e}")
            return False

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


class ClaimingWorker:
    """
    Worker that claims jobs from central store.

    Runs an infinite loop:
    1. Register with server
    2. Claim next job from server
    3. Execute job
    4. Report result
    5. Sleep and repeat

    Handles graceful shutdown via SIGTERM/SIGINT.
    """

    def __init__(self, config: ClaimingWorkerConfig):
        self.config = config
        self.client = JobStoreClient(config.server_url)

        # Worker identity
        self.worker_id = f"{config.device_id}.claiming"
        self.allowed_job_types: List[str] = []

        self._running = False
        self._current_job: Optional[Dict] = None
        self._start_time = 0
        self._consecutive_errors = 0
        self._last_heartbeat = 0

        # Statistics
        self._stats = {
            "jobs_claimed": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "total_duration": 0,
        }

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

        # Release current job if any
        if self._current_job:
            job_id = self._current_job.get("job_id")
            logger.info(f"Releasing job {job_id}")
            self.client.release(job_id)

    def run(self):
        """Run the worker loop."""
        self._running = True
        self._start_time = time.time()
        self._last_heartbeat = time.time()

        logger.info(f"ClaimingWorker starting")
        logger.info(f"  Worker ID: {self.worker_id}")
        logger.info(f"  Device: {self.config.device_id}")
        logger.info(f"  Roles: {self.config.roles}")
        logger.info(f"  Server: {self.config.server_url}")
        logger.info(f"  Claim interval: {self.config.claim_interval}s")
        logger.info(f"  Lease duration: {self.config.lease_duration}s")

        # Check server connectivity
        if not self.client.health_check():
            logger.error(f"Cannot reach server at {self.config.server_url}")
            return

        logger.info("Connected to job server")

        # Register with server
        if not self._register():
            logger.error("Failed to register with server")
            return

        while self._running:
            try:
                # Send heartbeat every 30 seconds
                if time.time() - self._last_heartbeat > 30:
                    self._send_heartbeat()

                # Try to claim a job
                job = self.client.claim_next(
                    self.config.device_id,
                    self.config.roles,
                    self.config.lease_duration,
                )

                if job:
                    self._consecutive_errors = 0
                    self._execute_job(job)
                else:
                    # No job available, wait
                    time.sleep(self.config.claim_interval)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                self._consecutive_errors += 1

                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    logger.error(f"Too many consecutive errors, shutting down")
                    break

                time.sleep(self.config.claim_interval)

        logger.info("Worker stopped")
        self._print_stats()

    def _register(self) -> bool:
        """Register worker with server."""
        result = self.client.register(
            device_id=self.config.device_id,
            worker_kind="claiming",
            roles=self.config.roles,
            version=VERSION,
            hostname=socket.gethostname(),
        )

        if result.get("registered"):
            self.allowed_job_types = result.get("allowed_job_types", [])
            logger.info(f"Registered as {self.worker_id}")
            logger.info(f"  Allowed job types: {self.allowed_job_types}")
            return True
        else:
            logger.error(f"Registration failed: {result.get('error')}")
            return False

    def _send_heartbeat(self):
        """Send heartbeat to server."""
        active_jobs = 1 if self._current_job else 0
        if self.client.heartbeat(self.worker_id, active_jobs):
            self._last_heartbeat = time.time()
        else:
            logger.warning("Heartbeat failed - may need to re-register")

    def _execute_job(self, job: Dict):
        """Execute a claimed job."""
        self._current_job = job
        job_id = job.get("job_id")
        spec = job.get("spec", {})
        job_type = spec.get("job_type")
        payload = spec.get("payload", {})

        self._stats["jobs_claimed"] += 1
        logger.info(f"Executing job {job_id}: {job_type}")

        # Mark as running
        self.client.mark_running(job_id, self.config.device_id)

        start_time = time.time()

        try:
            # Execute based on job type
            if job_type == "eval":
                result = self._execute_eval(payload)
            elif job_type == "sparring":
                result = self._execute_sparring(payload)
            elif job_type == "data_gen":
                result = self._execute_data_gen(payload)
            elif job_type == "inference":
                result = self._execute_inference(payload)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            # Report success
            duration = time.time() - start_time
            result["duration_seconds"] = duration
            self.client.mark_complete(job_id, result)

            self._stats["jobs_completed"] += 1
            self._stats["total_duration"] += duration
            logger.info(f"Job {job_id} completed in {duration:.1f}s")

        except Exception as e:
            # Classify the error
            error_code = self._classify_error(e, job_type)
            error_msg = str(e)

            logger.error(f"Job {job_id} failed [{error_code.value}]: {error_msg}")
            self.client.mark_failed(job_id, error_msg, error_code)
            self._stats["jobs_failed"] += 1

        finally:
            self._current_job = None

    def _classify_error(self, e: Exception, job_type: str) -> JobErrorCode:
        """Classify an exception into a structured error code."""
        exc_type = type(e).__name__
        exc_str = str(e).lower()

        # Connection errors
        if isinstance(e, requests.exceptions.ConnectionError):
            return JobErrorCode.CONNECTION_REFUSED
        if isinstance(e, requests.exceptions.Timeout):
            return JobErrorCode.TIMEOUT
        if isinstance(e, requests.exceptions.RequestException):
            return JobErrorCode.TRANSPORT_ERROR

        # Resource errors
        if "cuda" in exc_str or "out of memory" in exc_str:
            return JobErrorCode.RESOURCE_UNAVAILABLE
        if isinstance(e, FileNotFoundError) or "not found" in exc_str:
            if "model" in exc_str or "checkpoint" in exc_str:
                return JobErrorCode.MODEL_NOT_FOUND
            return JobErrorCode.WORKER_SETUP

        # Inference errors (from our _generate method)
        if "inference failed" in exc_str:
            return JobErrorCode.INFERENCE_ERROR

        # Validation errors
        if "validation" in exc_str:
            return JobErrorCode.VALIDATION_ERROR

        # Generator errors (for data_gen jobs)
        if job_type == "data_gen":
            return JobErrorCode.GENERATOR_ERROR

        # Generic execution error
        return JobErrorCode.EXECUTION_ERROR

    def _execute_eval(self, payload: Dict) -> Dict:
        """Execute an eval job."""
        skill_id = payload.get("skill_id", "bin")
        level = payload.get("level", 1)
        batch_size = payload.get("batch_size", 100)

        logger.info(f"Running eval: skill={skill_id}, level={level}, batch={batch_size}")

        # Import skill engine
        from guild.skills import get_engine

        engine = get_engine()

        # Generate eval batch
        batch = engine.generate_eval_batch(skill_id, level=level, count=batch_size)

        if not batch or not batch.problems:
            return {
                "success": False,
                "error": f"Could not generate eval batch for {skill_id}",
            }

        # Get model answers via inference
        answers = []
        for problem in batch.problems:
            try:
                answer = self._generate(problem.prompt)
                answers.append(answer)
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                answers.append("")

        # Score results
        result, state = engine.run_eval(skill_id, answers, level=level)

        return {
            "success": True,
            "skill_id": skill_id,
            "level": level,
            "accuracy": result.accuracy,
            "correct": result.correct_count,
            "total": result.total_count,
            "per_primitive": result.per_primitive_accuracy,
        }

    def _execute_sparring(self, payload: Dict) -> Dict:
        """Execute a sparring job."""
        skill_id = payload.get("skill_id", "binary")
        count = payload.get("count", 100)
        checkpoint = payload.get("checkpoint")

        logger.info(f"Running sparring: skill={skill_id}, count={count}")

        from guild.sparring import run_sparring_session

        result = run_sparring_session(
            skill=skill_id,
            count=count,
            checkpoint=checkpoint,
            inference_url=self.config.inference_url,
        )

        return {
            "success": True,
            "skill_id": skill_id,
            "problems_attempted": result.get("attempted", 0),
            "correct": result.get("correct", 0),
            "training_examples": result.get("examples_generated", 0),
            "output_file": result.get("output_file"),
        }

    def _execute_data_gen(self, payload: Dict) -> Dict:
        """Execute a data generation job."""
        generator = payload.get("generator", "")
        count = payload.get("count", 1000)

        logger.info(f"Running data gen: generator={generator}, count={count}")

        # TODO: Implement data generation
        # For now, return a placeholder
        return {
            "success": True,
            "generator": generator,
            "count": count,
            "message": "Data generation not yet implemented",
        }

    def _execute_inference(self, payload: Dict) -> Dict:
        """Execute an inference job."""
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 100)

        if not prompt:
            return {"success": False, "error": "No prompt provided"}

        response = self._generate(prompt, max_tokens=max_tokens)

        return {
            "success": True,
            "prompt": prompt,
            "response": response,
        }

    def _generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text via inference server."""
        inference_url = self.config.inference_url or os.environ.get(
            "INFERENCE_URL", "http://localhost:8765"
        )

        response = requests.post(
            f"{inference_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            headers={"X-API-Key": "admin123"},
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Inference failed: {response.status_code}")

        data = response.json()
        return data.get("text", data.get("response", ""))

    def _print_stats(self):
        """Print worker statistics."""
        uptime = time.time() - self._start_time
        logger.info(f"Worker Statistics:")
        logger.info(f"  Uptime: {uptime:.0f}s")
        logger.info(f"  Jobs claimed: {self._stats['jobs_claimed']}")
        logger.info(f"  Jobs completed: {self._stats['jobs_completed']}")
        logger.info(f"  Jobs failed: {self._stats['jobs_failed']}")
        if self._stats['jobs_completed'] > 0:
            avg = self._stats['total_duration'] / self._stats['jobs_completed']
            logger.info(f"  Avg job duration: {avg:.1f}s")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Claiming Worker - Pull jobs from central store"
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("TRAINING_DEVICE_ID", "worker"),
        help="Device ID (default: from TRAINING_DEVICE_ID env)",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("JOB_SERVER_URL", "http://localhost:8767"),
        help="Job server URL",
    )
    parser.add_argument(
        "--roles",
        default="eval_worker",
        help="Comma-separated roles (default: eval_worker)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Claim interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--lease",
        type=int,
        default=300,
        help="Lease duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--inference-url",
        default=os.environ.get("INFERENCE_URL"),
        help="Inference server URL",
    )

    args = parser.parse_args()

    config = ClaimingWorkerConfig(
        device_id=args.device,
        roles=args.roles.split(","),
        server_url=args.server,
        claim_interval=args.interval,
        lease_duration=args.lease,
        inference_url=args.inference_url,
    )

    worker = ClaimingWorker(config)
    worker.run()


if __name__ == "__main__":
    main()
