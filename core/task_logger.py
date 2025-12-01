"""
Task Distribution Logger - Track task assignments and outcomes by device/role.

Provides centralized logging for:
- Task routing decisions (which device got which task)
- Success/failure tracking
- Performance metrics
- Device utilization patterns

Usage:
    from core.task_logger import TaskLogger, log_task

    # Start a task
    task_id = log_task(
        task_type="training",
        device_id="trainer4090",
        details={"model": "qwen3", "steps": 1000}
    )

    # Complete the task
    log_task_complete(task_id, success=True, metrics={"loss": 0.5})

    # Or fail
    log_task_failed(task_id, error="CUDA OOM")
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("task_logger")


@dataclass
class TaskEvent:
    """A task event (start, complete, fail)."""
    task_id: str
    event_type: str  # start, complete, fail
    timestamp: str
    task_type: str  # training, eval, inference, storage
    device_id: str
    device_role: str  # trainer, inference, storage
    details: Dict[str, Any]
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class TaskLogger:
    """
    Centralized task distribution logger.

    Logs all task assignments and outcomes to:
    - logs/task_distribution.log (JSON lines)
    - logs/task_distribution_human.log (human-readable)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()

        self.base_dir = Path(base_dir)
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # JSON log for machine parsing
        self.json_log = self.log_dir / "task_distribution.log"

        # Human-readable log
        self.human_log = self.log_dir / "task_distribution_human.log"

        # In-memory task tracking
        self._active_tasks: Dict[str, Dict[str, Any]] = {}

        # Load device info from hosts.json
        self._device_roles = self._load_device_roles()

    def _load_device_roles(self) -> Dict[str, str]:
        """Load device role mapping from hosts.json."""
        try:
            from core.hosts import get_all_hosts
            hosts = get_all_hosts()
            return {host.device_id: host.role for host in hosts}
        except:
            # Fallback defaults
            return {
                "trainer4090": "trainer",
                "inference3090": "inference",
                "synology_data": "storage",
            }

    def start_task(
        self,
        task_type: str,
        device_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log task start.

        Args:
            task_type: Type of task (training, eval, inference, storage)
            device_id: Device executing the task
            details: Additional task details

        Returns:
            task_id for completion tracking
        """
        task_id = str(uuid.uuid4())
        device_role = self._device_roles.get(device_id, "unknown")

        event = TaskEvent(
            task_id=task_id,
            event_type="start",
            timestamp=datetime.now().isoformat(),
            task_type=task_type,
            device_id=device_id,
            device_role=device_role,
            details=details or {},
        )

        # Track start time
        self._active_tasks[task_id] = {
            "start_time": time.time(),
            "task_type": task_type,
            "device_id": device_id,
            "details": details,
        }

        self._write_event(event)
        return task_id

    def complete_task(
        self,
        task_id: str,
        success: bool = True,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log task completion.

        Args:
            task_id: Task ID from start_task()
            success: Whether task succeeded
            metrics: Performance metrics
        """
        if task_id not in self._active_tasks:
            logger.warning(f"Completing unknown task: {task_id}")
            return

        task_info = self._active_tasks.pop(task_id)
        duration_ms = (time.time() - task_info["start_time"]) * 1000

        event = TaskEvent(
            task_id=task_id,
            event_type="complete",
            timestamp=datetime.now().isoformat(),
            task_type=task_info["task_type"],
            device_id=task_info["device_id"],
            device_role=self._device_roles.get(task_info["device_id"], "unknown"),
            details=task_info.get("details", {}),
            duration_ms=duration_ms,
            success=success,
            metrics=metrics,
        )

        self._write_event(event)

    def fail_task(
        self,
        task_id: str,
        error: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log task failure.

        Args:
            task_id: Task ID from start_task()
            error: Error description
            metrics: Partial metrics if available
        """
        if task_id not in self._active_tasks:
            logger.warning(f"Failing unknown task: {task_id}")
            return

        task_info = self._active_tasks.pop(task_id)
        duration_ms = (time.time() - task_info["start_time"]) * 1000

        event = TaskEvent(
            task_id=task_id,
            event_type="fail",
            timestamp=datetime.now().isoformat(),
            task_type=task_info["task_type"],
            device_id=task_info["device_id"],
            device_role=self._device_roles.get(task_info["device_id"], "unknown"),
            details=task_info.get("details", {}),
            duration_ms=duration_ms,
            success=False,
            error=error,
            metrics=metrics,
        )

        self._write_event(event)

    def _write_event(self, event: TaskEvent):
        """Write event to both logs."""
        # JSON log
        with open(self.json_log, 'a') as f:
            f.write(json.dumps(asdict(event)) + '\n')

        # Human-readable log
        human_msg = self._format_human(event)
        with open(self.human_log, 'a') as f:
            f.write(human_msg + '\n')

        # Also log to Python logger
        if event.event_type == "fail":
            logger.error(human_msg)
        else:
            logger.info(human_msg)

    def _format_human(self, event: TaskEvent) -> str:
        """Format event for human reading."""
        ts = event.timestamp.split('T')[1].split('.')[0]  # HH:MM:SS

        if event.event_type == "start":
            return (
                f"[{ts}] START {event.task_type} on {event.device_id} ({event.device_role}) "
                f"- {event.details}"
            )
        elif event.event_type == "complete":
            status = "SUCCESS" if event.success else "FAILED"
            duration = f"{event.duration_ms:.0f}ms" if event.duration_ms else "?"
            metrics_str = f" | {event.metrics}" if event.metrics else ""
            return (
                f"[{ts}] {status} {event.task_type} on {event.device_id} "
                f"in {duration}{metrics_str}"
            )
        elif event.event_type == "fail":
            duration = f"{event.duration_ms:.0f}ms" if event.duration_ms else "?"
            return (
                f"[{ts}] FAILED {event.task_type} on {event.device_id} "
                f"after {duration} - {event.error}"
            )
        else:
            return f"[{ts}] UNKNOWN EVENT: {event}"


# Global logger instance
_task_logger: Optional[TaskLogger] = None


def get_task_logger(base_dir: Optional[Path] = None) -> TaskLogger:
    """Get or create global task logger."""
    global _task_logger
    if _task_logger is None:
        _task_logger = TaskLogger(base_dir)
    return _task_logger


# Convenience functions

def log_task_start(task_type: str, device_id: str, details: Optional[Dict] = None) -> str:
    """Start logging a task. Returns task_id for completion."""
    return get_task_logger().start_task(task_type, device_id, details)


def log_task_complete(task_id: str, success: bool = True, metrics: Optional[Dict] = None):
    """Mark task as complete."""
    get_task_logger().complete_task(task_id, success, metrics)


def log_task_failed(task_id: str, error: str, metrics: Optional[Dict] = None):
    """Mark task as failed."""
    get_task_logger().fail_task(task_id, error, metrics)


# Context manager for easy task logging

class TaskContext:
    """Context manager for automatic task logging."""

    def __init__(self, task_type: str, device_id: str, details: Optional[Dict] = None):
        self.task_type = task_type
        self.device_id = device_id
        self.details = details
        self.task_id: Optional[str] = None
        self.metrics: Dict[str, Any] = {}

    def __enter__(self):
        self.task_id = log_task_start(self.task_type, self.device_id, self.details)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            log_task_complete(self.task_id, success=True, metrics=self.metrics)
        else:
            # Failure
            error = f"{exc_type.__name__}: {exc_val}"
            log_task_failed(self.task_id, error, self.metrics)
        return False  # Don't suppress exception

    def add_metric(self, key: str, value: Any):
        """Add a metric to be logged at completion."""
        self.metrics[key] = value


# Example usage in code:
if __name__ == "__main__":
    # Example 1: Manual logging
    task_id = log_task_start("training", "trainer4090", {"model": "qwen3", "steps": 1000})
    time.sleep(0.1)  # Simulate work
    log_task_complete(task_id, success=True, metrics={"loss": 0.5, "acc": 0.8})

    # Example 2: Context manager
    try:
        with TaskContext("eval", "inference3090", {"skill": "bin", "level": 1}) as ctx:
            time.sleep(0.05)  # Simulate work
            ctx.add_metric("accuracy", 0.75)
            ctx.add_metric("latency_ms", 50)
            # Success logged automatically
    except Exception as e:
        pass  # Failure logged automatically

    # Example 3: Failed task
    task_id = log_task_start("inference", "inference3090", {"prompt_tokens": 100})
    time.sleep(0.02)
    log_task_failed(task_id, "CUDA OOM", {"partial_tokens": 50})

    print("\nCheck logs/task_distribution.log and logs/task_distribution_human.log")
