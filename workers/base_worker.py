"""
Base Worker - Abstract base class for distributed workers.

Workers are HTTP servers that accept and execute jobs.
This base class provides:
- HTTP server with job submission endpoints
- Job queue management
- Status reporting and heartbeats
- Graceful shutdown handling

Usage:
    from workers.base_worker import BaseWorker

    class MyWorker(BaseWorker):
        def handle_job(self, job_id, spec):
            # Execute the job
            return {"result": "success"}

    worker = MyWorker(device_id="my_device")
    worker.run(port=8900)

API Endpoints:
    POST /jobs           - Submit a job
    GET  /jobs/{id}      - Get job status
    POST /jobs/{id}/cancel - Cancel a job
    GET  /health         - Health check
    GET  /status         - Worker status
"""

import json
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("worker")


@dataclass
class WorkerConfig:
    """Configuration for a worker."""
    device_id: str
    max_concurrent: int = 1
    heartbeat_interval: int = 30  # Seconds between heartbeats
    job_timeout: int = 3600  # Default job timeout
    log_dir: Optional[str] = None


@dataclass
class WorkerJob:
    """A job being processed by the worker."""
    job_id: str
    spec: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed, cancelled
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None


class WorkerRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for worker API."""

    def log_message(self, format, *args):
        """Override to use Python logging."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict]:
        """Read JSON from request body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            return json.loads(body)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json({
                "status": "healthy",
                "device_id": self.server.worker.config.device_id,
                "uptime": time.time() - self.server.worker._start_time,
            })

        elif path == "/status":
            self._send_json(self.server.worker.get_status())

        elif path.startswith("/jobs/"):
            job_id = path.split("/")[2]
            job = self.server.worker.get_job(job_id)
            if job:
                self._send_json({
                    "job_id": job.job_id,
                    "status": job.status,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "result": job.result,
                    "error": job.error,
                })
            else:
                self._send_json({"error": "Job not found"}, 404)

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/jobs":
            data = self._read_json()
            if not data:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            job_id = data.get("job_id")
            spec = data.get("spec")

            if not job_id or not spec:
                self._send_json({"error": "Missing job_id or spec"}, 400)
                return

            success, message = self.server.worker.submit_job(job_id, spec)
            if success:
                self._send_json({"status": "accepted", "job_id": job_id}, 202)
            else:
                self._send_json({"error": message}, 503)

        elif path.endswith("/cancel"):
            job_id = path.split("/")[2]
            if self.server.worker.cancel_job(job_id):
                self._send_json({"status": "cancelled"})
            else:
                self._send_json({"error": "Cannot cancel"}, 400)

        else:
            self._send_json({"error": "Not found"}, 404)


class WorkerHTTPServer(HTTPServer):
    """HTTP server with worker reference."""

    def __init__(self, server_address, handler_class, worker):
        self.worker = worker
        super().__init__(server_address, handler_class)


class BaseWorker(ABC):
    """
    Abstract base class for workers.

    Subclasses must implement handle_job() to process jobs.
    """

    def __init__(self, config: Optional[WorkerConfig] = None, device_id: Optional[str] = None):
        """
        Initialize the worker.

        Args:
            config: Worker configuration
            device_id: Override device ID (from TRAINING_DEVICE_ID env if not set)
        """
        if config:
            self.config = config
        else:
            device_id = device_id or os.environ.get("TRAINING_DEVICE_ID", "worker")
            self.config = WorkerConfig(device_id=device_id)

        self._jobs: Dict[str, WorkerJob] = {}
        self._lock = threading.Lock()
        self._running = False
        self._start_time = 0
        self._server: Optional[WorkerHTTPServer] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Statistics
        self._stats = {
            "jobs_received": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "total_duration": 0,
        }

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================

    @abstractmethod
    def handle_job(self, job_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a job. Override in subclass.

        Args:
            job_id: Job identifier
            spec: Job specification (from JobSpec.to_dict())

        Returns:
            Result dict with output data

        Raises:
            Exception: Job failed
        """
        pass

    def get_supported_job_types(self) -> List[str]:
        """Get job types this worker supports. Override in subclass."""
        return []

    # =========================================================================
    # JOB MANAGEMENT
    # =========================================================================

    def submit_job(self, job_id: str, spec: Dict[str, Any]) -> tuple[bool, str]:
        """
        Submit a job for execution.

        Args:
            job_id: Job identifier
            spec: Job specification

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            # Check capacity
            running = sum(1 for j in self._jobs.values() if j.status == "running")
            if running >= self.config.max_concurrent:
                return False, f"At capacity ({running}/{self.config.max_concurrent})"

            # Check job type
            job_type = spec.get("job_type")
            supported = self.get_supported_job_types()
            if supported and job_type not in supported:
                return False, f"Unsupported job type: {job_type}"

            # Create job
            job = WorkerJob(job_id=job_id, spec=spec)
            self._jobs[job_id] = job
            self._stats["jobs_received"] += 1

        # Start job in thread
        thread = threading.Thread(target=self._run_job, args=(job,))
        job.thread = thread
        thread.start()

        logger.info(f"Accepted job {job_id}: {job_type}")
        return True, "accepted"

    def _run_job(self, job: WorkerJob):
        """Run a job in a thread."""
        job.status = "running"
        job.started_at = datetime.now()

        logger.info(f"Starting job {job.job_id}")

        try:
            result = self.handle_job(job.job_id, job.spec)
            job.status = "completed"
            job.result = result
            self._stats["jobs_completed"] += 1
            logger.info(f"Completed job {job.job_id}")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            self._stats["jobs_failed"] += 1
            logger.error(f"Job {job.job_id} failed: {e}")

        finally:
            job.completed_at = datetime.now()
            if job.started_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                self._stats["total_duration"] += duration

    def get_job(self, job_id: str) -> Optional[WorkerJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in ("completed", "failed", "cancelled"):
            return False

        job.status = "cancelled"
        job.completed_at = datetime.now()
        logger.info(f"Cancelled job {job_id}")
        return True

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        with self._lock:
            running = [j for j in self._jobs.values() if j.status == "running"]
            pending = [j for j in self._jobs.values() if j.status == "pending"]

        return {
            "device_id": self.config.device_id,
            "status": "running" if self._running else "stopped",
            "uptime": time.time() - self._start_time if self._running else 0,
            "max_concurrent": self.config.max_concurrent,
            "running_jobs": len(running),
            "pending_jobs": len(pending),
            "supported_types": self.get_supported_job_types(),
            "stats": dict(self._stats),
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status,
                    "type": j.spec.get("job_type"),
                }
                for j in list(self._jobs.values())[-10:]  # Last 10 jobs
            ],
        }

    # =========================================================================
    # SERVER
    # =========================================================================

    def run(self, port: int = 8900, host: str = "0.0.0.0"):
        """
        Run the worker HTTP server.

        Args:
            port: Port to listen on
            host: Host to bind to
        """
        self._running = True
        self._start_time = time.time()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # Start HTTP server
        self._server = WorkerHTTPServer((host, port), WorkerRequestHandler, self)

        logger.info(f"Worker {self.config.device_id} starting on {host}:{port}")
        logger.info(f"Supported job types: {self.get_supported_job_types()}")

        try:
            self._server.serve_forever()
        finally:
            self._running = False
            logger.info("Worker stopped")

    def stop(self):
        """Stop the worker."""
        self._running = False
        if self._server:
            self._server.shutdown()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while self._running:
            try:
                self._send_heartbeat()
            except Exception as e:
                logger.debug(f"Heartbeat failed: {e}")

            time.sleep(self.config.heartbeat_interval)

    def _send_heartbeat(self):
        """Send heartbeat to coordinator. Override to implement."""
        pass


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("BaseWorker is an abstract class.")
    print("Use EvalWorker or DataForgeWorker instead:")
    print("  python3 -m workers.eval_worker --port 8900")
    print("  python3 -m workers.data_forge_worker --port 8900")
