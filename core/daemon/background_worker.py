#!/usr/bin/env python3
"""
Background Worker - Run heavy tasks without blocking the main loop.

Provides a thread-based task queue for offloading:
- Snapshot creation
- Model consolidation
- Retention cleanup
- Data validation

Usage:
    from daemon.background_worker import BackgroundWorker, Task

    worker = BackgroundWorker()
    worker.start()

    # Submit tasks
    worker.submit(Task("snapshot", create_snapshot))

    # Check status
    status = worker.get_status()

    # Cleanup
    worker.stop()
"""

import logging
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any, Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    A task to run in the background.

    IMPORTANT: The `timeout` parameter is metadata only and is NOT enforced.
    Tasks will run to completion regardless of timeout value. If a task hangs,
    it cannot be forcibly terminated - the daemon would need to be restarted.

    For critical operations that must respect timeouts, implement timeout logic
    within the task function itself (e.g., using signal.alarm or threading.Timer).

    Attributes:
        name: Human-readable task name for logging/status
        func: Callable to execute
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        timeout: Advisory timeout in seconds (not enforced)
    """
    name: str
    func: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None  # Advisory only - NOT enforced by worker


@dataclass
class TaskStatus:
    """Status of a background task."""
    name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None

    @property
    def is_running(self) -> bool:
        return self.completed_at is None

    @property
    def duration_seconds(self) -> float:
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


class BackgroundWorker:
    """
    Runs heavy tasks in background threads.

    Features:
    - Thread-safe task queue
    - Task status tracking
    - Graceful shutdown
    - Status reporting for monitoring

    Example:
        worker = BackgroundWorker(max_concurrent=1)
        worker.start()

        # Submit work
        worker.submit(Task("cleanup", cleanup_old_files))

        # Check status
        print(worker.get_status())

        # Stop when done
        worker.stop()
    """

    def __init__(self, max_concurrent: int = 1, history_size: int = 50):
        """
        Initialize background worker.

        Args:
            max_concurrent: Max concurrent tasks (default: 1)
            history_size: Max completed tasks to remember
        """
        self.max_concurrent = max_concurrent
        self.history_size = history_size

        self.task_queue: queue.Queue[Task] = queue.Queue()
        self.workers: List[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.running_tasks: Dict[str, TaskStatus] = {}
        self.task_history: List[TaskStatus] = []

    def start(self) -> None:
        """Start worker threads."""
        self._stop.clear()

        for i in range(self.max_concurrent):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BackgroundWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.max_concurrent} background worker(s)")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop worker threads gracefully."""
        self._stop.set()

        for worker in self.workers:
            worker.join(timeout=timeout)

        self.workers.clear()
        logger.info("Background workers stopped")

    def submit(self, task: Task) -> bool:
        """
        Submit a task to the queue.

        Args:
            task: Task to run

        Returns:
            True if submitted, False if worker stopped
        """
        if self._stop.is_set():
            logger.warning(f"Cannot submit task {task.name}: worker stopped")
            return False

        self.task_queue.put(task)
        logger.info(f"Queued background task: {task.name}")
        return True

    def _worker_loop(self) -> None:
        """Main worker loop."""
        while not self._stop.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            status = TaskStatus(name=task.name, started_at=datetime.now())

            with self._lock:
                self.running_tasks[task.name] = status

            try:
                logger.info(f"Starting background task: {task.name}")
                result = task.func(*task.args, **task.kwargs)
                status.result = result
                logger.info(f"Completed background task: {task.name}")

            except Exception as e:
                status.error = str(e)
                logger.error(f"Background task {task.name} failed: {e}")

            finally:
                status.completed_at = datetime.now()

                with self._lock:
                    self.running_tasks.pop(task.name, None)
                    self.task_history.append(status)

                    # Trim history
                    if len(self.task_history) > self.history_size:
                        self.task_history = self.task_history[-self.history_size:]

                self.task_queue.task_done()

    def get_status(self) -> Dict[str, Any]:
        """
        Get worker status for monitoring.

        Returns:
            Dict with queue size, running tasks, recent history
        """
        with self._lock:
            running = [
                {
                    "name": s.name,
                    "started_at": s.started_at.isoformat(),
                    "duration_seconds": s.duration_seconds
                }
                for s in self.running_tasks.values()
            ]

            recent = [
                {
                    "name": s.name,
                    "started_at": s.started_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "duration_seconds": s.duration_seconds,
                    "error": s.error
                }
                for s in self.task_history[-10:]
            ]

        return {
            "queue_size": self.task_queue.qsize(),
            "running": running,
            "recent": recent,
            "workers": len(self.workers)
        }

    def is_running(self) -> bool:
        """Check if any workers are alive."""
        return any(w.is_alive() for w in self.workers)

    def queue_size(self) -> int:
        """Get number of pending tasks."""
        return self.task_queue.qsize()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all queued tasks to complete.

        Args:
            timeout: Max seconds to wait (None = forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        try:
            self.task_queue.join()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Quick test
    import time

    logging.basicConfig(level=logging.INFO)

    def slow_task(name: str, duration: float) -> str:
        logger.info(f"Task {name} starting, will take {duration}s")
        time.sleep(duration)
        return f"{name} completed"

    def failing_task():
        raise ValueError("Intentional failure")

    print("Testing BackgroundWorker...")

    worker = BackgroundWorker(max_concurrent=2)
    worker.start()

    # Submit tasks
    worker.submit(Task("task1", slow_task, args=("Task 1", 1.0)))
    worker.submit(Task("task2", slow_task, args=("Task 2", 0.5)))
    worker.submit(Task("failing", failing_task))

    # Check status while running
    time.sleep(0.2)
    status = worker.get_status()
    print(f"\nStatus while running:")
    print(f"  Queue: {status['queue_size']}")
    print(f"  Running: {len(status['running'])}")

    # Wait for completion
    time.sleep(2)

    # Final status
    status = worker.get_status()
    print(f"\nFinal status:")
    print(f"  Queue: {status['queue_size']}")
    print(f"  Recent tasks: {len(status['recent'])}")
    for task in status['recent']:
        print(f"    - {task['name']}: {'ERROR' if task['error'] else 'OK'} ({task['duration_seconds']:.2f}s)")

    worker.stop()
    print("\nBackgroundWorker ready for use!")
