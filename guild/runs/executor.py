"""Run executor - orchestrates run lifecycle and execution."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any, Dict, Protocol

from guild.types import Status
from guild.runs.types import RunConfig, RunState, RunType
from guild.runs.state_manager import RunStateManager, get_state_manager


logger = logging.getLogger(__name__)


class RunCallback(Protocol):
    """Protocol for run lifecycle callbacks."""

    def on_run_start(self, run: RunState) -> None:
        """Called when run starts."""
        ...

    def on_run_pause(self, run: RunState) -> None:
        """Called when run is paused."""
        ...

    def on_run_resume(self, run: RunState) -> None:
        """Called when run resumes."""
        ...

    def on_run_complete(self, run: RunState) -> None:
        """Called when run completes successfully."""
        ...

    def on_run_fail(self, run: RunState, error: str) -> None:
        """Called when run fails."""
        ...

    def on_step(self, run: RunState, step: int) -> None:
        """Called after each step."""
        ...

    def on_checkpoint(self, run: RunState, checkpoint_path: str) -> None:
        """Called when checkpoint is saved."""
        ...


@dataclass
class RunCallbackAdapter:
    """
    Adapter that provides no-op defaults for RunCallback.

    Subclass and override only the methods you need.
    """

    def on_run_start(self, run: RunState) -> None:
        pass

    def on_run_pause(self, run: RunState) -> None:
        pass

    def on_run_resume(self, run: RunState) -> None:
        pass

    def on_run_complete(self, run: RunState) -> None:
        pass

    def on_run_fail(self, run: RunState, error: str) -> None:
        pass

    def on_step(self, run: RunState, step: int) -> None:
        pass

    def on_checkpoint(self, run: RunState, checkpoint_path: str) -> None:
        pass


@dataclass
class StepResult:
    """Result of executing a single step."""

    step: int
    quests_attempted: int = 0
    quests_succeeded: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False
    stop_reason: Optional[str] = None


class RunHandler(ABC):
    """
    Abstract handler for run execution logic.

    Subclass to implement specific run type behaviors
    (training, evaluation, audit, etc.)
    """

    @abstractmethod
    def initialize(self, run: RunState) -> None:
        """Initialize resources for the run (load model, etc.)."""
        pass

    @abstractmethod
    def execute_step(self, run: RunState) -> StepResult:
        """Execute a single step of the run."""
        pass

    @abstractmethod
    def save_checkpoint(self, run: RunState) -> str:
        """Save a checkpoint, return the path."""
        pass

    @abstractmethod
    def cleanup(self, run: RunState) -> None:
        """Cleanup resources after run."""
        pass

    def should_checkpoint(self, run: RunState) -> bool:
        """Check if checkpoint should be saved."""
        if run.config.checkpoint_every_steps <= 0:
            return False
        steps_since = run.current_step - run.last_checkpoint_step
        return steps_since >= run.config.checkpoint_every_steps


class RunExecutor:
    """
    Orchestrates run execution with lifecycle management.

    Responsibilities:
    - Create and manage runs through their lifecycle
    - Coordinate with RunHandler for actual execution
    - Track progress and metrics
    - Handle pause/resume/cancel signals
    - Fire callbacks at lifecycle events

    Example:
        executor = RunExecutor(state_manager, handler)

        # Create and start a run
        run_id = executor.create_run(config)
        executor.start(run_id)

        # Execute steps
        while executor.is_running(run_id):
            executor.step(run_id)
            if should_pause:
                executor.pause(run_id)

        # Or run to completion
        executor.run_to_completion(run_id)
    """

    def __init__(
        self,
        state_manager: Optional[RunStateManager] = None,
        handler: Optional[RunHandler] = None,
        callbacks: Optional[list[RunCallback]] = None,
    ):
        self.state_manager = state_manager or get_state_manager()
        self.handler = handler
        self.callbacks = callbacks or []

        # Control signals
        self._pause_requested: Dict[str, bool] = {}
        self._cancel_requested: Dict[str, bool] = {}

    def add_callback(self, callback: RunCallback):
        """Add a lifecycle callback."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: RunCallback):
        """Remove a lifecycle callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _fire_callback(self, method: str, *args, **kwargs):
        """Fire a callback method on all registered callbacks."""
        for cb in self.callbacks:
            try:
                getattr(cb, method)(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback {method} failed: {e}")

    # --- Run Creation ---

    def create_run(
        self,
        config: RunConfig,
        run_id: Optional[str] = None,
    ) -> str:
        """
        Create a new run.

        Returns:
            Run ID
        """
        run = self.state_manager.create_run(config, run_id)
        self._pause_requested[run.run_id] = False
        self._cancel_requested[run.run_id] = False
        return run.run_id

    # --- Lifecycle Control ---

    def start(self, run_id: str) -> RunState:
        """
        Start a run.

        Initializes the handler and transitions to ACTIVE.
        """
        run = self.state_manager.start_run(run_id)

        if self.handler:
            try:
                self.handler.initialize(run)
            except Exception as e:
                self.state_manager.fail_run(run_id, str(e))
                raise

        self._fire_callback("on_run_start", run)
        return run

    def pause(self, run_id: str) -> RunState:
        """Pause a running run."""
        run = self.state_manager.pause_run(run_id)
        self._fire_callback("on_run_pause", run)
        return run

    def resume(self, run_id: str) -> RunState:
        """Resume a paused run."""
        run = self.state_manager.resume_run(run_id)
        self._fire_callback("on_run_resume", run)
        return run

    def cancel(self, run_id: str, reason: Optional[str] = None) -> RunState:
        """Cancel a run."""
        run = self.state_manager.get_run_or_raise(run_id)

        if self.handler:
            try:
                self.handler.cleanup(run)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        run = self.state_manager.cancel_run(run_id, reason)
        return run

    def complete(
        self,
        run_id: str,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """Complete a run successfully."""
        run = self.state_manager.get_run_or_raise(run_id)

        if self.handler:
            try:
                self.handler.cleanup(run)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        run = self.state_manager.complete_run(run_id, final_metrics)
        self._fire_callback("on_run_complete", run)
        return run

    def fail(
        self,
        run_id: str,
        error: str,
        incident_id: Optional[str] = None,
    ) -> RunState:
        """Mark a run as failed."""
        run = self.state_manager.get_run_or_raise(run_id)

        if self.handler:
            try:
                self.handler.cleanup(run)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        run = self.state_manager.fail_run(run_id, error, incident_id)
        self._fire_callback("on_run_fail", run, error)
        return run

    # --- Execution ---

    def step(self, run_id: str) -> StepResult:
        """
        Execute a single step.

        Returns:
            StepResult with metrics and control signals
        """
        run = self.state_manager.get_run_or_raise(run_id)

        if run.status != Status.ACTIVE:
            raise ValueError(f"Cannot step run in status {run.status.value}")

        if not self.handler:
            raise RuntimeError("No handler configured")

        # Execute step
        result = self.handler.execute_step(run)

        # Update progress
        self.state_manager.increment_progress(
            run_id,
            steps=1,
            quests_completed=result.quests_attempted,
            quests_succeeded=result.quests_succeeded,
        )

        if result.metrics:
            self.state_manager.update_progress(run_id, metrics=result.metrics)

        # Refresh run state
        run = self.state_manager.get_run_or_raise(run_id)

        # Fire step callback
        self._fire_callback("on_step", run, run.current_step)

        # Check for checkpoint
        if self.handler.should_checkpoint(run):
            checkpoint_path = self.handler.save_checkpoint(run)
            self.state_manager.record_checkpoint(run_id, checkpoint_path)
            self._fire_callback("on_checkpoint", run, checkpoint_path)

        return result

    def request_pause(self, run_id: str):
        """Request run to pause at next opportunity."""
        self._pause_requested[run_id] = True

    def request_cancel(self, run_id: str):
        """Request run to cancel at next opportunity."""
        self._cancel_requested[run_id] = True

    def is_pause_requested(self, run_id: str) -> bool:
        """Check if pause has been requested."""
        return self._pause_requested.get(run_id, False)

    def is_cancel_requested(self, run_id: str) -> bool:
        """Check if cancel has been requested."""
        return self._cancel_requested.get(run_id, False)

    def is_running(self, run_id: str) -> bool:
        """Check if run is currently active."""
        run = self.state_manager.get_run(run_id)
        return run is not None and run.status == Status.ACTIVE

    def should_continue(self, run_id: str) -> bool:
        """
        Check if run should continue executing.

        Considers:
        - Run status
        - Pause/cancel requests
        - Step/quest/duration limits
        """
        run = self.state_manager.get_run(run_id)
        if run is None or run.status != Status.ACTIVE:
            return False

        # Check control signals
        if self._pause_requested.get(run_id, False):
            return False
        if self._cancel_requested.get(run_id, False):
            return False

        # Check limits
        config = run.config

        if config.max_steps and run.current_step >= config.max_steps:
            return False

        if config.max_quests and run.quests_completed >= config.max_quests:
            return False

        if config.max_duration_seconds:
            if run.duration_seconds >= config.max_duration_seconds:
                return False

        return True

    def run_to_completion(
        self,
        run_id: str,
        step_delay: float = 0.0,
    ) -> RunState:
        """
        Run until completion, failure, pause, or cancel.

        Args:
            run_id: Run to execute
            step_delay: Optional delay between steps (for rate limiting)

        Returns:
            Final RunState
        """
        run = self.state_manager.get_run_or_raise(run_id)

        # Start if pending
        if run.status == Status.PENDING:
            run = self.start(run_id)

        try:
            while self.should_continue(run_id):
                result = self.step(run_id)

                if result.should_stop:
                    logger.info(f"Run stopping: {result.stop_reason}")
                    break

                if step_delay > 0:
                    time.sleep(step_delay)

            # Handle final state
            run = self.state_manager.get_run_or_raise(run_id)

            if self._cancel_requested.get(run_id, False):
                return self.cancel(run_id, "Cancelled by request")

            if self._pause_requested.get(run_id, False):
                self._pause_requested[run_id] = False
                return self.pause(run_id)

            if run.status == Status.ACTIVE:
                return self.complete(run_id)

        except Exception as e:
            logger.exception(f"Run {run_id} failed")
            return self.fail(run_id, str(e))

        return self.state_manager.get_run_or_raise(run_id)

    # --- Queries ---

    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get run state."""
        return self.state_manager.get_run(run_id)

    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get run status summary."""
        run = self.state_manager.get_run(run_id)
        if run is None:
            return {"exists": False}

        return {
            "exists": True,
            "run_id": run.run_id,
            "config_id": run.config.id,
            "type": run.config.type.value,
            "status": run.status.value,
            "step": run.current_step,
            "quests_completed": run.quests_completed,
            "quests_succeeded": run.quests_succeeded,
            "success_rate": run.success_rate,
            "duration_seconds": run.duration_seconds,
            "pause_requested": self._pause_requested.get(run_id, False),
            "cancel_requested": self._cancel_requested.get(run_id, False),
        }


# Global executor
_executor: Optional[RunExecutor] = None


def init_executor(
    state_manager: Optional[RunStateManager] = None,
    handler: Optional[RunHandler] = None,
) -> RunExecutor:
    """Initialize the global executor."""
    global _executor
    _executor = RunExecutor(state_manager, handler)
    return _executor


def get_executor() -> RunExecutor:
    """Get the global executor."""
    global _executor
    if _executor is None:
        _executor = RunExecutor()
    return _executor


def reset_executor():
    """Reset the global executor (for testing)."""
    global _executor
    _executor = None
