#!/usr/bin/env python3
"""
Task Client - Client library for submitting tasks to the GPU Task Scheduler

Usage:
    from monitoring.task_client import TaskClient

    client = TaskClient()

    # Submit a task and wait for result
    result = client.submit_and_wait("curriculum_eval", {"skill": "syllo"})

    # Submit a task asynchronously
    task_id = client.submit("baseline_test", {"skill": "primitives"})
    status = client.get_status(task_id)

    # Check if scheduler is healthy
    if client.is_healthy():
        client.submit("live_prediction", {"prompts": ["Hello"]})

This client is built on core.services.ServiceClient for standardized
network handling, retries, and error semantics.
"""

import logging
import time
from typing import Any, Dict, Optional

from core.services import (
    ServiceClient,
    ServiceConfig,
    ServiceId,
    ServiceError,
    ServiceHttpError,
    ServiceUnavailable,
    get_service_config,
)

logger = logging.getLogger(__name__)


class TaskClient:
    """
    Client for interacting with the GPU Task Scheduler.

    Built on ServiceClient for consistent retry, timeout, and error handling.

    Example usage:
        client = TaskClient()

        # Submit a task and wait
        result = client.submit_and_wait("curriculum_eval", {"skill": "syllo"})

        # Check scheduler health
        if client.is_healthy():
            client.submit("baseline_test", {"skill": "binary"})
    """

    def __init__(
        self,
        scheduler_url: str = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize TaskClient.

        Args:
            scheduler_url: URL of the GPU Task Scheduler (defaults to host registry)
            timeout: Default request timeout in seconds
            max_retries: Max retries for failed requests
        """
        # Get base config from service registry
        config = get_service_config("scheduler")

        # Override with constructor parameters
        if scheduler_url:
            config.base_url = scheduler_url
        config.timeout_s = float(timeout)
        config.max_retries = max_retries

        # Create underlying ServiceClient
        self._client = ServiceClient(config)

        logger.info(f"TaskClient initialized: {config.base_url}")

    @property
    def scheduler_url(self) -> str:
        """URL of the scheduler (for backward compatibility)."""
        return self._client.base_url

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> Optional[Dict]:
        """Make a request to the scheduler API."""
        try:
            if method == "GET":
                return self._client.get_json(endpoint, timeout=timeout)
            elif method == "POST":
                return self._client.post_json(endpoint, json=data, timeout=timeout)
            elif method == "DELETE":
                return self._client.delete(endpoint, timeout=timeout)
            else:
                raise ValueError(f"Unknown method: {method}")
        except ServiceUnavailable as e:
            logger.warning(f"Scheduler unavailable: {e}")
            return None
        except ServiceHttpError as e:
            logger.warning(f"Scheduler HTTP error: {e}")
            return None
        except ServiceError as e:
            logger.warning(f"Scheduler error: {e}")
            return None

    def is_healthy(self) -> bool:
        """Check if the scheduler is healthy and running"""
        result = self._request("GET", "/api/health")
        return result and result.get("status") == "ok" and result.get("scheduler") == "running"

    def submit(
        self,
        task_type: str,
        params: Optional[Dict] = None,
        priority: Optional[int] = None,
        callback_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Submit a task to the scheduler.

        Args:
            task_type: Type of task (e.g., "curriculum_eval", "baseline_test")
            params: Task-specific parameters
            priority: Optional priority (0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW, 4=IDLE)
            callback_url: Optional URL to call when task completes

        Returns:
            task_id if successful, None otherwise
        """
        data = {
            "task_type": task_type,
            "params": params or {}
        }

        if priority is not None:
            data["priority"] = priority
        if callback_url:
            data["callback_url"] = callback_url

        result = self._request("POST", "/api/tasks/submit", data)

        if result:
            task_id = result.get("task_id")
            logger.info(f"Submitted task {task_id}: {task_type}")
            return task_id

        logger.error(f"Failed to submit task: {task_type}")
        return None

    def get_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task"""
        return self._request("GET", f"/api/tasks/{task_id}")

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task"""
        result = self._request("DELETE", f"/api/tasks/{task_id}")
        return result and result.get("status") == "cancelled"

    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            poll_interval: Seconds between status checks
            timeout: Max seconds to wait (None = forever)

        Returns:
            Task result dict or None if timeout/error
        """
        start = time.time()

        while True:
            status = self.get_status(task_id)

            if not status:
                logger.error(f"Failed to get status for task {task_id}")
                return None

            if status.get("status") in ("completed", "failed", "cancelled"):
                return status

            if timeout and (time.time() - start) > timeout:
                logger.warning(f"Timeout waiting for task {task_id}")
                return status

            time.sleep(poll_interval)

    def submit_and_wait(
        self,
        task_type: str,
        params: Optional[Dict] = None,
        priority: Optional[int] = None,
        timeout: Optional[float] = 300.0
    ) -> Optional[Dict]:
        """
        Submit a task and wait for completion.

        Args:
            task_type: Type of task
            params: Task parameters
            priority: Optional priority
            timeout: Max seconds to wait

        Returns:
            Task result dict or None if failed
        """
        task_id = self.submit(task_type, params, priority)

        if not task_id:
            return None

        return self.wait_for_completion(task_id, timeout=timeout)

    def get_queue(self) -> Optional[Dict]:
        """Get current queue status"""
        return self._request("GET", "/api/tasks/queue")

    def get_active_task(self) -> Optional[Dict]:
        """Get currently running task"""
        return self._request("GET", "/api/tasks/active")

    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU statistics"""
        return self._request("GET", "/api/gpu/stats")

    def get_metrics(self) -> Optional[Dict]:
        """Get scheduler metrics"""
        return self._request("GET", "/api/metrics")

    def get_task_types(self) -> Optional[Dict]:
        """Get available task types and priorities"""
        return self._request("GET", "/api/task-types")


class SchedulerAwareDaemon:
    """
    Base class for daemons that use the GPU Task Scheduler.

    Subclass this to create daemons that submit tasks to the scheduler
    instead of executing directly.
    """

    def __init__(
        self,
        scheduler_url: str = None,
        fallback_to_local: bool = True
    ):
        """
        Initialize daemon.

        Args:
            scheduler_url: URL of the GPU Task Scheduler (defaults to host registry)
            fallback_to_local: If True, run locally when scheduler unavailable
        """
        self.client = TaskClient(scheduler_url)
        self.fallback_to_local = fallback_to_local
        self._scheduler_available = None
        self._last_health_check = 0

    def check_scheduler(self) -> bool:
        """Check if scheduler is available (cached for 30s)"""
        now = time.time()

        if now - self._last_health_check > 30:
            self._scheduler_available = self.client.is_healthy()
            self._last_health_check = now
            if self._scheduler_available:
                logger.info("GPU Task Scheduler is available")
            else:
                logger.warning("GPU Task Scheduler unavailable")

        return self._scheduler_available

    def submit_task(
        self,
        task_type: str,
        params: Optional[Dict] = None,
        priority: Optional[int] = None,
        wait: bool = False,
        timeout: float = 300.0
    ) -> Optional[Dict]:
        """
        Submit a task to the scheduler or run locally.

        Args:
            task_type: Type of task
            params: Task parameters
            priority: Optional priority
            wait: If True, wait for completion
            timeout: Max wait time

        Returns:
            Task result if wait=True, otherwise submission result
        """
        if self.check_scheduler():
            if wait:
                return self.client.submit_and_wait(task_type, params, priority, timeout)
            else:
                task_id = self.client.submit(task_type, params, priority)
                return {"task_id": task_id, "status": "submitted"}

        elif self.fallback_to_local:
            logger.info(f"Running task locally: {task_type}")
            return self.run_local(task_type, params)

        else:
            logger.error("Scheduler unavailable and fallback disabled")
            return None

    def run_local(self, task_type: str, params: Optional[Dict] = None) -> Dict:
        """
        Run a task locally (override in subclass).

        Args:
            task_type: Type of task
            params: Task parameters

        Returns:
            Task result
        """
        raise NotImplementedError("Subclass must implement run_local()")


# Convenience functions for direct use
_default_client: Optional[TaskClient] = None


def get_client(scheduler_url: str = None) -> TaskClient:
    """Get or create the default TaskClient"""
    global _default_client
    if _default_client is None:
        _default_client = TaskClient(scheduler_url)
    return _default_client


def submit_task(
    task_type: str,
    params: Optional[Dict] = None,
    priority: Optional[int] = None
) -> Optional[str]:
    """Submit a task using the default client"""
    return get_client().submit(task_type, params, priority)


def submit_and_wait(
    task_type: str,
    params: Optional[Dict] = None,
    priority: Optional[int] = None,
    timeout: float = 300.0
) -> Optional[Dict]:
    """Submit a task and wait for completion using the default client"""
    return get_client().submit_and_wait(task_type, params, priority, timeout)


# Re-export exceptions for convenience
__all__ = [
    "TaskClient",
    "SchedulerAwareDaemon",
    "get_client",
    "submit_task",
    "submit_and_wait",
    "ServiceError",
    "ServiceHttpError",
    "ServiceUnavailable",
]


if __name__ == "__main__":
    # Simple test
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Task Client Test")
    parser.add_argument("--url", default=None,
                        help="Scheduler URL (defaults to host registry)")
    parser.add_argument("--task", default="live_prediction",
                        help="Task type to submit")

    args = parser.parse_args()

    client = TaskClient(args.url)

    print(f"Checking scheduler at {client.scheduler_url}...")
    if client.is_healthy():
        print("Scheduler is healthy!")

        print(f"\nSubmitting task: {args.task}")
        result = client.submit_and_wait(args.task, {}, timeout=60)
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("Scheduler is not available")
