"""
Verbose Logger - Track task lifecycle with detailed timestamps.

Provides structured logging for tracking tasks (evals, training runs, etc.)
through their complete lifecycle:
- Task created/queued
- Task started
- Task finished
- Duration calculations

Usage:
    from core.verbose_logger import VerboseLogger, is_verbose_mode

    if is_verbose_mode():
        VerboseLogger.task_queued("eval", task_id, {"skill": "bin", "level": 1})
        VerboseLogger.task_started("eval", task_id)
        # ... do work ...
        VerboseLogger.task_finished("eval", task_id, success=True)

Environment variables:
    VERBOSE=1 - Enable verbose mode
    VERBOSE_FILE=/path/to/log - Write verbose logs to file (default: status/verbose.log)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)

# Global verbose mode flag
_VERBOSE_MODE: Optional[bool] = None
_VERBOSE_FILE: Optional[Path] = None
_VERBOSE_LOCK = Lock()
_VERBOSE_TASKS: Dict[str, 'TaskLifecycle'] = {}


@dataclass
class TaskLifecycle:
    """Track a task's lifecycle timestamps."""
    task_type: str  # "eval", "training", "sync", etc.
    task_id: str
    created_at: float  # Unix timestamp
    metadata: Dict[str, Any]

    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    @property
    def time_in_queue(self) -> Optional[float]:
        """Seconds from creation to start."""
        if self.started_at:
            return self.started_at - self.created_at
        return None

    @property
    def execution_time(self) -> Optional[float]:
        """Seconds from start to finish."""
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    @property
    def total_time(self) -> Optional[float]:
        """Seconds from creation to finish."""
        if self.finished_at:
            return self.finished_at - self.created_at
        return None

    @property
    def is_complete(self) -> bool:
        """Has the task finished?"""
        return self.finished_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with calculated fields."""
        d = asdict(self)
        d['time_in_queue'] = self.time_in_queue
        d['execution_time'] = self.execution_time
        d['total_time'] = self.total_time
        d['is_complete'] = self.is_complete
        return d


def is_verbose_mode() -> bool:
    """Check if verbose mode is enabled."""
    global _VERBOSE_MODE

    if _VERBOSE_MODE is None:
        _VERBOSE_MODE = os.environ.get("VERBOSE", "0") == "1"

    return _VERBOSE_MODE


def get_verbose_file() -> Optional[Path]:
    """Get path to verbose log file."""
    global _VERBOSE_FILE

    if _VERBOSE_FILE is None:
        env_file = os.environ.get("VERBOSE_FILE")
        if env_file:
            _VERBOSE_FILE = Path(env_file)
        else:
            # Default to status/verbose.log
            from core.paths import get_base_dir
            _VERBOSE_FILE = get_base_dir() / "status" / "verbose.log"

    return _VERBOSE_FILE


def _log_verbose(event: str, data: Dict[str, Any]):
    """Write a verbose log entry."""
    if not is_verbose_mode():
        return

    timestamp = datetime.now()
    iso_time = timestamp.isoformat()
    unix_time = timestamp.timestamp()

    entry = {
        "timestamp": iso_time,
        "unix_timestamp": unix_time,
        "event": event,
        **data
    }

    # Log to console
    logger.info(f"[VERBOSE] {event}: {json.dumps(data, default=str)}")

    # Write to file
    log_file = get_verbose_file()
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with _VERBOSE_LOCK:
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write verbose log: {e}")


class VerboseLogger:
    """Verbose logger for task lifecycle tracking."""

    @staticmethod
    def task_queued(task_type: str, task_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Log when a task is created/queued."""
        if not is_verbose_mode():
            return

        now = time.time()
        lifecycle = TaskLifecycle(
            task_type=task_type,
            task_id=task_id,
            created_at=now,
            metadata=metadata or {}
        )

        with _VERBOSE_LOCK:
            _VERBOSE_TASKS[task_id] = lifecycle

        _log_verbose("task_queued", {
            "task_type": task_type,
            "task_id": task_id,
            "created_at": datetime.fromtimestamp(now).isoformat(),
            "metadata": metadata or {}
        })

    @staticmethod
    def task_started(task_type: str, task_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Log when a task starts execution."""
        if not is_verbose_mode():
            return

        now = time.time()

        with _VERBOSE_LOCK:
            lifecycle = _VERBOSE_TASKS.get(task_id)
            if lifecycle:
                lifecycle.started_at = now
                if metadata:
                    lifecycle.metadata.update(metadata)
                time_in_queue = lifecycle.time_in_queue
            else:
                # Task not found in tracking - create it now
                lifecycle = TaskLifecycle(
                    task_type=task_type,
                    task_id=task_id,
                    created_at=now,
                    started_at=now,
                    metadata=metadata or {}
                )
                _VERBOSE_TASKS[task_id] = lifecycle
                time_in_queue = 0.0

        _log_verbose("task_started", {
            "task_type": task_type,
            "task_id": task_id,
            "started_at": datetime.fromtimestamp(now).isoformat(),
            "time_in_queue_seconds": time_in_queue,
            "metadata": metadata or {}
        })

    @staticmethod
    def task_finished(
        task_type: str,
        task_id: str,
        success: bool = True,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ):
        """Log when a task finishes (success or failure)."""
        if not is_verbose_mode():
            return

        now = time.time()

        with _VERBOSE_LOCK:
            lifecycle = _VERBOSE_TASKS.get(task_id)
            if lifecycle:
                lifecycle.finished_at = now
                lifecycle.success = success
                lifecycle.error = error
                lifecycle.result = result
                time_in_queue = lifecycle.time_in_queue
                execution_time = lifecycle.execution_time
                total_time = lifecycle.total_time
            else:
                time_in_queue = None
                execution_time = None
                total_time = None

        _log_verbose("task_finished", {
            "task_type": task_type,
            "task_id": task_id,
            "finished_at": datetime.fromtimestamp(now).isoformat(),
            "success": success,
            "error": error,
            "time_in_queue_seconds": time_in_queue,
            "execution_time_seconds": execution_time,
            "total_time_seconds": total_time,
            "result": result or {}
        })

    @staticmethod
    def task_progress(task_type: str, task_id: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log task progress update."""
        if not is_verbose_mode():
            return

        _log_verbose("task_progress", {
            "task_type": task_type,
            "task_id": task_id,
            "message": message,
            **(data or {})
        })

    @staticmethod
    def get_task_lifecycle(task_id: str) -> Optional[TaskLifecycle]:
        """Get lifecycle tracking for a task."""
        with _VERBOSE_LOCK:
            return _VERBOSE_TASKS.get(task_id)

    @staticmethod
    def get_all_tasks(task_type: Optional[str] = None, incomplete_only: bool = False) -> List[TaskLifecycle]:
        """Get all tracked tasks, optionally filtered."""
        with _VERBOSE_LOCK:
            tasks = list(_VERBOSE_TASKS.values())

        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        if incomplete_only:
            tasks = [t for t in tasks if not t.is_complete]

        return tasks

    @staticmethod
    def cleanup_completed(max_age_seconds: float = 3600):
        """Remove completed tasks older than max_age_seconds."""
        now = time.time()

        with _VERBOSE_LOCK:
            to_remove = []
            for task_id, lifecycle in _VERBOSE_TASKS.items():
                if lifecycle.is_complete and lifecycle.finished_at:
                    age = now - lifecycle.finished_at
                    if age > max_age_seconds:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del _VERBOSE_TASKS[task_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed tasks")

    @staticmethod
    def get_stats(task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about tracked tasks."""
        tasks = VerboseLogger.get_all_tasks(task_type=task_type)

        if not tasks:
            return {"count": 0}

        completed = [t for t in tasks if t.is_complete]
        successful = [t for t in completed if t.success]
        failed = [t for t in completed if not t.success]
        in_progress = [t for t in tasks if t.started_at and not t.is_complete]
        queued = [t for t in tasks if not t.started_at]

        # Calculate average times for completed tasks
        queue_times = [t.time_in_queue for t in completed if t.time_in_queue]
        exec_times = [t.execution_time for t in completed if t.execution_time]
        total_times = [t.total_time for t in completed if t.total_time]

        return {
            "total": len(tasks),
            "completed": len(completed),
            "successful": len(successful),
            "failed": len(failed),
            "in_progress": len(in_progress),
            "queued": len(queued),
            "avg_queue_time": sum(queue_times) / len(queue_times) if queue_times else None,
            "avg_execution_time": sum(exec_times) / len(exec_times) if exec_times else None,
            "avg_total_time": sum(total_times) / len(total_times) if total_times else None,
        }


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if seconds is None:
        return "N/A"

    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_task_summary(task_id: str):
    """Print a human-readable summary of a task's lifecycle."""
    lifecycle = VerboseLogger.get_task_lifecycle(task_id)
    if not lifecycle:
        print(f"Task {task_id} not found")
        return

    print(f"\n{'='*70}")
    print(f"Task Lifecycle: {task_id}")
    print(f"{'='*70}")
    print(f"Type:          {lifecycle.task_type}")
    print(f"Metadata:      {json.dumps(lifecycle.metadata, indent=2)}")
    print(f"")
    print(f"Created:       {datetime.fromtimestamp(lifecycle.created_at).isoformat()}")

    if lifecycle.started_at:
        print(f"Started:       {datetime.fromtimestamp(lifecycle.started_at).isoformat()}")
        print(f"Queue time:    {format_duration(lifecycle.time_in_queue)}")

    if lifecycle.finished_at:
        print(f"Finished:      {datetime.fromtimestamp(lifecycle.finished_at).isoformat()}")
        print(f"Execution:     {format_duration(lifecycle.execution_time)}")
        print(f"Total time:    {format_duration(lifecycle.total_time)}")
        print(f"Status:        {'SUCCESS' if lifecycle.success else 'FAILED'}")

        if lifecycle.error:
            print(f"Error:         {lifecycle.error}")

        if lifecycle.result:
            print(f"Result:        {json.dumps(lifecycle.result, indent=2)}")
    else:
        print(f"Status:        IN PROGRESS")

    print(f"{'='*70}\n")


def print_all_tasks_summary(task_type: Optional[str] = None):
    """Print summary of all tracked tasks."""
    stats = VerboseLogger.get_stats(task_type=task_type)

    print(f"\n{'='*70}")
    print(f"Task Statistics" + (f" - {task_type}" if task_type else ""))
    print(f"{'='*70}")
    print(f"Total tasks:        {stats['total']}")
    print(f"Queued:             {stats['queued']}")
    print(f"In progress:        {stats['in_progress']}")
    print(f"Completed:          {stats['completed']}")
    print(f"  Successful:       {stats['successful']}")
    print(f"  Failed:           {stats['failed']}")
    print(f"")

    if stats['avg_queue_time']:
        print(f"Avg queue time:     {format_duration(stats['avg_queue_time'])}")
    if stats['avg_execution_time']:
        print(f"Avg execution time: {format_duration(stats['avg_execution_time'])}")
    if stats['avg_total_time']:
        print(f"Avg total time:     {format_duration(stats['avg_total_time'])}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    """Test verbose logger."""
    import sys

    # Enable verbose mode
    os.environ["VERBOSE"] = "1"

    # Test task lifecycle
    task_id = "test-eval-12345"

    print("Testing verbose logger...")
    print(f"Verbose mode: {is_verbose_mode()}")
    print(f"Log file: {get_verbose_file()}\n")

    # Simulate task lifecycle
    VerboseLogger.task_queued("eval", task_id, {"skill": "bin", "level": 1, "checkpoint": 175000})
    time.sleep(0.5)

    VerboseLogger.task_started("eval", task_id)
    time.sleep(1.0)

    VerboseLogger.task_progress("eval", task_id, "Loaded checkpoint", {"checkpoint_loaded": True})
    time.sleep(0.5)

    VerboseLogger.task_progress("eval", task_id, "Running evaluation", {"problems_evaluated": 3})
    time.sleep(1.0)

    VerboseLogger.task_finished("eval", task_id, success=True, result={"accuracy": 0.8, "correct": 4, "total": 5})

    # Print summary
    print_task_summary(task_id)
    print_all_tasks_summary("eval")
