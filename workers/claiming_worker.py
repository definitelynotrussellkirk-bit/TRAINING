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
from core.cluster_state import (
    register_host as cluster_register_host,
    heartbeat as cluster_heartbeat,
    probe_local_resources,
    update_host_resources,
    set_host_status,
)

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
        worker_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Claim the next available job.

        Args:
            device_id: Device ID (used in legacy mode)
            roles: Roles (used in legacy mode)
            lease_duration: Lease duration in seconds
            worker_id: Worker ID for preferred mode (server looks up roles)

        If worker_id is provided, server uses roles from registration.
        """
        try:
            # Prefer worker_id mode for role trust
            if worker_id:
                payload = {
                    "worker_id": worker_id,
                    "lease_duration": lease_duration,
                }
            else:
                payload = {
                    "device_id": device_id,
                    "roles": roles,
                    "lease_duration": lease_duration,
                }

            response = self.session.post(
                f"{self.server_url}/api/jobs/claim",
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("claimed"):
                    return data["job"]
                # Check if we need to re-register
                if data.get("should_register"):
                    logger.warning("Server says we need to re-register")
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
        self.host_id = config.device_id  # ClusterState host ID
        self.allowed_job_types: List[str] = []

        self._running = False
        self._current_job: Optional[Dict] = None
        self._start_time = 0
        self._consecutive_errors = 0
        self._last_heartbeat = 0
        self._last_resource_update = 0

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

        # Mark host as offline in ClusterState
        try:
            set_host_status(self.host_id, "offline", error="Worker shutdown")
            logger.info(f"Marked {self.host_id} offline in ClusterState")
        except Exception as e:
            logger.warning(f"Failed to update ClusterState on shutdown: {e}")

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

        # Register with job server
        if not self._register():
            logger.error("Failed to register with server")
            return

        # Register with ClusterState for cluster visibility
        self._register_with_cluster()

        while self._running:
            try:
                # Send heartbeat every 30 seconds
                if time.time() - self._last_heartbeat > 30:
                    self._send_heartbeat()

                # Try to claim a job using worker_id mode (server looks up roles)
                job = self.client.claim_next(
                    self.config.device_id,
                    self.config.roles,
                    self.config.lease_duration,
                    worker_id=self.worker_id,  # Preferred: server-side role lookup
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

    def _register_with_cluster(self):
        """Register this worker with ClusterState for cluster-wide visibility."""
        try:
            # Map worker roles to ClusterState roles
            # ClusterState uses: trainer, oracle, forge, monitor, mixed
            cluster_roles = []
            if any(r in self.config.roles for r in ["eval_worker", "sparring"]):
                cluster_roles.append("forge")
            if "inference" in self.config.roles:
                cluster_roles.append("oracle")
            if "trainer" in self.config.roles:
                cluster_roles.append("trainer")
            if not cluster_roles:
                cluster_roles = ["forge"]  # Default for job workers

            # Get IP address
            ip_address = None
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]
                s.close()
            except Exception:
                pass

            cluster_register_host(
                host_id=self.host_id,
                name=f"Worker: {self.config.device_id}",
                roles=cluster_roles,
                ip_address=ip_address,
            )
            logger.info(f"Registered {self.host_id} with ClusterState (roles: {cluster_roles})")

            # Initial resource update
            self._update_cluster_resources()

        except Exception as e:
            logger.warning(f"Failed to register with ClusterState: {e}")

    def _update_cluster_resources(self):
        """Update resource metrics in ClusterState."""
        try:
            resources = probe_local_resources()
            update_host_resources(self.host_id, resources)
            self._last_resource_update = time.time()
        except Exception as e:
            logger.debug(f"Failed to update cluster resources: {e}")

    def _send_heartbeat(self):
        """Send heartbeat to job server and ClusterState."""
        active_jobs = 1 if self._current_job else 0

        # Job server heartbeat
        if self.client.heartbeat(self.worker_id, active_jobs):
            self._last_heartbeat = time.time()
        else:
            logger.warning("Heartbeat failed - may need to re-register")

        # ClusterState heartbeat
        try:
            cluster_heartbeat(self.host_id, extra={"running_jobs": active_jobs})
        except Exception as e:
            logger.debug(f"ClusterState heartbeat failed: {e}")

        # Update resources every 5 minutes
        if time.time() - self._last_resource_update > 300:
            self._update_cluster_resources()

    def _execute_job(self, job: Dict):
        """Execute a claimed job."""
        self._current_job = job
        job_id = job.get("job_id")
        spec = job.get("spec", {})
        job_type = spec.get("job_type")
        payload = spec.get("payload", {})

        self._stats["jobs_claimed"] += 1

        # Guard: check if job type is allowed for this worker
        if self.allowed_job_types and job_type not in self.allowed_job_types:
            logger.error(
                f"Job {job_id} has type '{job_type}' not allowed for this worker. "
                f"Allowed: {self.allowed_job_types}"
            )
            self.client.mark_failed(
                job_id,
                f"Job type '{job_type}' not allowed for worker {self.worker_id}",
                JobErrorCode.WORKER_SETUP,
            )
            self._stats["jobs_failed"] += 1
            self._current_job = None
            return

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
            elif job_type == "layer_stats":
                result = self._execute_layer_stats(payload)
            elif job_type == "layer_drift":
                result = self._execute_layer_drift(payload)
            elif job_type == "data_validate":
                result = self._execute_data_validate(payload)
            elif job_type == "data_profile":
                result = self._execute_data_profile(payload)
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
        result, state = engine.run_eval(skill_id, answers, level=level, count=batch_size)

        # Log to battle log for UI visibility
        try:
            from core.battle_log import log_eval

            pct = result.accuracy * 100
            severity = "success" if pct >= 80 else "warning" if pct >= 50 else "error"
            msg = f"Eval {skill_id} L{level}: {pct:.1f}% ({result.num_correct}/{result.num_examples})"

            log_eval(
                msg,
                severity=severity,
                details={
                    "skill": skill_id,
                    "level": level,
                    "accuracy": result.accuracy,
                    "correct": result.num_correct,
                    "total": result.num_examples,
                    "per_primitive": result.per_primitive_accuracy,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to log to battle log: {e}")

        return {
            "success": True,
            "skill_id": skill_id,
            "level": level,
            "accuracy": result.accuracy,
            "correct": result.num_correct,
            "total": result.num_examples,
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
        if self.config.inference_url:
            inference_url = self.config.inference_url
        else:
            # Use service discovery from hosts.json
            from core.hosts import get_service_url
            inference_url = get_service_url("inference") or os.environ.get(
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

    def _execute_layer_stats(self, payload: Dict) -> Dict:
        """
        Execute a layer_stats job - Model Archaeology.

        Computes weight norms, activation stats, and drift for a checkpoint.
        """
        from analysis import run_layer_stats_analysis
        from analysis.probe_datasets import get_default_probes

        campaign_id = payload["campaign_id"]
        hero_id = payload["hero_id"]
        checkpoint_path = payload["checkpoint_path"]
        model_ref = payload.get("model_ref", "qwen3-0.6b")
        reference_path = payload.get("reference_checkpoint_path")
        probe_dataset = payload.get("probe_dataset", "default")
        max_tokens = payload.get("max_probe_tokens", 4096)
        compute_act = payload.get("compute_activations", True)

        logger.info(f"Running layer_stats: campaign={campaign_id}, hero={hero_id}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        if reference_path:
            logger.info(f"  Reference: {reference_path}")

        # Load probes
        if probe_dataset == "default":
            probes = get_default_probes()
        else:
            from analysis.probe_datasets import load_probe_dataset
            probes = load_probe_dataset(probe_dataset)

        # Run analysis
        result = run_layer_stats_analysis(
            checkpoint_path=checkpoint_path,
            campaign_id=campaign_id,
            hero_id=hero_id,
            model_ref=model_ref,
            reference_checkpoint_path=reference_path,
            probe_sequences=probes if compute_act else None,
            max_probe_tokens=max_tokens,
            compute_activations=compute_act,
        )

        # Save to campaign analysis dir
        output_path = payload.get("output_path")
        if not output_path:
            output_path = self._get_analysis_path(campaign_id, hero_id, "layer_stats")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Filename based on checkpoint step
        filename = f"ckpt-{result.checkpoint_step:06d}.layer_stats.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            f.write(result.to_json())

        logger.info(f"Saved layer stats to {filepath}")

        return {
            "success": True,
            "output_path": str(filepath),
            "checkpoint_step": result.checkpoint_step,
            "num_layers": len(result.weight_stats),
            "has_activations": bool(result.activation_stats),
            "has_drift": bool(result.drift_stats),
            "most_changed_layer": (
                result.global_drift_stats.get("most_changed_layer")
                if result.global_drift_stats else None
            ),
            "duration_sec": result.compute_duration_sec,
        }

    def _execute_layer_drift(self, payload: Dict) -> Dict:
        """
        Execute a layer_drift job - compare two checkpoints.

        Lighter-weight than layer_stats - only computes weight drift.
        """
        from analysis.model_loader import load_reference_state_dict
        import torch

        campaign_id = payload["campaign_id"]
        hero_id = payload["hero_id"]
        base_path = payload["base_checkpoint_path"]
        target_path = payload["target_checkpoint_path"]

        logger.info(f"Computing layer drift: {base_path} -> {target_path}")

        # Load both state dicts (CPU to save VRAM)
        base_state = load_reference_state_dict(base_path, device="cpu")
        target_state = load_reference_state_dict(target_path, device="cpu")

        # Compute drift per layer
        drift_stats = {}
        all_l2 = []

        for name in base_state:
            if name not in target_state:
                continue

            base_param = base_state[name]
            target_param = target_state[name]

            l2 = (target_param.float() - base_param.float()).norm(2).item()
            all_l2.append(l2)

            # Group by layer
            parts = name.split(".")
            layer_name = ".".join(parts[:-1]) if len(parts) > 1 else name

            if layer_name not in drift_stats:
                drift_stats[layer_name] = {"l2_total": 0, "params": {}}

            drift_stats[layer_name]["params"][name] = l2
            drift_stats[layer_name]["l2_total"] += l2

        # Find extremes
        sorted_layers = sorted(
            drift_stats.items(),
            key=lambda x: x[1]["l2_total"],
            reverse=True
        )

        result = {
            "success": True,
            "campaign_id": campaign_id,
            "hero_id": hero_id,
            "base_checkpoint": base_path,
            "target_checkpoint": target_path,
            "layer_count": len(drift_stats),
            "avg_drift_l2": sum(all_l2) / len(all_l2) if all_l2 else 0,
            "max_drift_l2": max(all_l2) if all_l2 else 0,
            "most_changed_layer": sorted_layers[0][0] if sorted_layers else None,
            "least_changed_layer": sorted_layers[-1][0] if sorted_layers else None,
        }

        # Save if output path specified
        output_path = payload.get("output_path")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({**result, "layer_drift": drift_stats}, f, indent=2)
            result["output_path"] = str(output_path)

        return result

    def _execute_data_validate(self, payload: Dict) -> Dict:
        """Execute a data_validate job using the Forge validator."""
        from pathlib import Path

        file_path = Path(payload["file_path"])
        skill_id = payload.get("skill_id")
        deep = payload.get("deep", True)
        check_leakage = payload.get("check_leakage", True)
        add_to_queue = payload.get("add_to_queue", False)
        priority = payload.get("priority", "normal")

        logger.info(f"Validating: {file_path} (skill={skill_id}, deep={deep})")

        # Run Forge validation
        from forge.validator import ForgeValidator

        validator = ForgeValidator()
        result = validator.validate(
            file_path,
            skill_id=skill_id,
            deep=deep,
            check_leakage=check_leakage,
        )

        # Log to Battle Log
        try:
            from core.battle_log import log_jobs
            if result.passed:
                log_jobs(
                    f"✅ Validated {file_path.name}: {result.summary}",
                    source="forge.data_validate",
                    severity="success",
                    details={"file": file_path.name, "skill_id": skill_id}
                )
            else:
                log_jobs(
                    f"🚫 Validation failed {file_path.name}: {result.summary}",
                    source="forge.data_validate",
                    severity="warning",
                    details={
                        "file": file_path.name,
                        "errors": result.errors[:5],
                        "leakage_count": result.leakage_count,
                    }
                )
        except Exception:
            pass

        # Optionally add to queue if passed
        if add_to_queue and result.passed:
            try:
                from core.training_queue import TrainingQueue
                from core.paths import get_base_dir

                queue = TrainingQueue(str(get_base_dir()))
                if queue.add_to_queue(file_path, priority):
                    logger.info(f"Added {file_path.name} to {priority} queue")
            except Exception as e:
                logger.warning(f"Failed to add to queue: {e}")

        return result.to_dict()

    def _execute_data_profile(self, payload: Dict) -> Dict:
        """Execute a data_profile job using the Forge profiler."""
        from pathlib import Path
        import json

        file_path = Path(payload["file_path"])
        max_samples = payload.get("max_samples", 10000)
        output_path = payload.get("output_path")
        dataset_id = payload.get("dataset_id")

        logger.info(f"Profiling: {file_path} (max_samples={max_samples})")

        # Run profiler
        from forge.profiler import profile_shard

        profile = profile_shard(file_path, max_samples=max_samples)

        # Save report if output path specified
        if output_path:
            output_path = Path(output_path)
        else:
            # Default: save to data/reports/
            try:
                from core.paths import get_base_dir
                reports_dir = get_base_dir() / "data" / "reports"
            except ImportError:
                reports_dir = Path("data/reports")

            if dataset_id:
                reports_dir = reports_dir / dataset_id

            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / f"{file_path.stem}.profile.json"

        with open(output_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

        logger.info(f"Profile saved to: {output_path}")

        # Log to Battle Log
        try:
            from core.battle_log import log_jobs
            log_jobs(
                f"Profiled {file_path.name}: {profile.rows_total} rows, "
                f"avg {profile.char_lengths.get('mean', 0):.0f} chars",
                source="forge.data_profile",
                severity="info",
                details={
                    "file": file_path.name,
                    "rows": profile.rows_total,
                    "output": str(output_path),
                }
            )
        except Exception:
            pass

        return {
            "file_path": str(file_path),
            "rows_total": profile.rows_total,
            "output_path": str(output_path),
            "summary": profile.summary(),
        }

    def _get_analysis_path(self, campaign_id: str, hero_id: str, analysis_type: str) -> Path:
        """Get the analysis directory path for a campaign."""
        try:
            from core.paths import get_base_dir
            base = get_base_dir()
        except ImportError:
            base = Path(os.environ.get("TRAINING_BASE_DIR", "."))
        return base / "campaigns" / hero_id / campaign_id / "analysis" / analysis_type

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
