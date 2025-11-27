"""Run state management and persistence."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from guild.types import Status
from guild.runs.types import RunConfig, RunState, RunType


logger = logging.getLogger(__name__)


class RunStateManager:
    """
    Manages run state persistence and lifecycle.

    Stores state for all runs (active and historical):
    - Active runs (PENDING, ACTIVE, PAUSED)
    - Completed runs (COMPLETED, FAILED, CANCELLED)

    State is persisted to a JSON file in the status directory.
    """

    def __init__(
        self,
        state_dir: Path,
        state_file: str = "run_states.json",
        history_limit: int = 100,
    ):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for state files
            state_file: Name of state file
            history_limit: Max completed runs to keep in history
        """
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / state_file
        self.history_limit = history_limit

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._runs: Dict[str, RunState] = {}
        self._metadata: Dict[str, Any] = {}

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if not self.state_file.exists():
            self._metadata = {
                "created_at": datetime.now().isoformat(),
            }
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse run state file: {e}")
            return

        self._metadata = data.get("metadata", {})

        for run_id, run_data in data.get("runs", {}).items():
            try:
                self._runs[run_id] = RunState.from_dict(run_data)
            except Exception as e:
                logger.warning(f"Failed to load run '{run_id}': {e}")

        logger.debug(f"Loaded state for {len(self._runs)} runs")

    def _save_state(self):
        """Save state to disk."""
        data = {
            "metadata": {
                **self._metadata,
                "last_updated": datetime.now().isoformat(),
            },
            "runs": {
                run_id: run_state.to_dict()
                for run_id, run_state in self._runs.items()
            },
        }

        # Atomic write
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(self.state_file)

        logger.debug("Run state saved")

    def _trim_history(self):
        """Trim old completed runs if over limit."""
        completed = [
            r for r in self._runs.values()
            if r.status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]
        ]

        if len(completed) <= self.history_limit:
            return

        # Sort by completion time, oldest first
        completed.sort(key=lambda r: r.completed_at or datetime.min)

        # Remove oldest
        to_remove = len(completed) - self.history_limit
        for run_state in completed[:to_remove]:
            del self._runs[run_state.run_id]
            logger.debug(f"Trimmed old run from history: {run_state.run_id}")

    # --- Run Creation ---

    def create_run(
        self,
        config: RunConfig,
        run_id: Optional[str] = None,
    ) -> RunState:
        """
        Create a new run from a config.

        Args:
            config: Run configuration
            run_id: Optional explicit run ID (generates UUID if not provided)

        Returns:
            New RunState in PENDING status
        """
        if run_id is None:
            run_id = f"{config.id}_{uuid.uuid4().hex[:8]}"

        if run_id in self._runs:
            raise ValueError(f"Run already exists: {run_id}")

        state = RunState(
            run_id=run_id,
            config=config,
            status=Status.PENDING,
        )

        self._runs[run_id] = state
        self._save_state()

        logger.info(f"Created run: {run_id} (type={config.type.value})")
        return state

    def create_run_from_dict(
        self,
        config_dict: dict,
        run_id: Optional[str] = None,
    ) -> RunState:
        """Create a run from a config dict (useful for ad-hoc runs)."""
        config = RunConfig.from_dict(config_dict)
        return self.create_run(config, run_id)

    # --- Run Lifecycle ---

    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get a run by ID."""
        return self._runs.get(run_id)

    def get_run_or_raise(self, run_id: str) -> RunState:
        """Get a run by ID, raising if not found."""
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run: {run_id}")
        return run

    def start_run(self, run_id: str) -> RunState:
        """
        Start a pending run.

        Transitions: PENDING -> ACTIVE
        """
        run = self.get_run_or_raise(run_id)

        if run.status != Status.PENDING:
            raise ValueError(
                f"Cannot start run in status {run.status.value}. "
                f"Must be PENDING."
            )

        run.status = Status.ACTIVE
        run.started_at = datetime.now()

        self._save_state()
        logger.info(f"Started run: {run_id}")
        return run

    def pause_run(self, run_id: str) -> RunState:
        """
        Pause an active run.

        Transitions: ACTIVE -> PAUSED
        """
        run = self.get_run_or_raise(run_id)

        if run.status != Status.ACTIVE:
            raise ValueError(
                f"Cannot pause run in status {run.status.value}. "
                f"Must be ACTIVE."
            )

        run.status = Status.PAUSED
        run.paused_at = datetime.now()

        self._save_state()
        logger.info(f"Paused run: {run_id}")
        return run

    def resume_run(self, run_id: str) -> RunState:
        """
        Resume a paused run.

        Transitions: PAUSED -> ACTIVE
        """
        run = self.get_run_or_raise(run_id)

        if run.status != Status.PAUSED:
            raise ValueError(
                f"Cannot resume run in status {run.status.value}. "
                f"Must be PAUSED."
            )

        run.status = Status.ACTIVE
        run.paused_at = None  # Clear pause timestamp

        self._save_state()
        logger.info(f"Resumed run: {run_id}")
        return run

    def complete_run(
        self,
        run_id: str,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """
        Mark a run as completed.

        Transitions: ACTIVE -> COMPLETED
        """
        run = self.get_run_or_raise(run_id)

        if run.status != Status.ACTIVE:
            raise ValueError(
                f"Cannot complete run in status {run.status.value}. "
                f"Must be ACTIVE."
            )

        run.status = Status.COMPLETED
        run.completed_at = datetime.now()

        if final_metrics:
            run.metrics.update(final_metrics)

        self._save_state()
        self._trim_history()

        logger.info(f"Completed run: {run_id}")
        return run

    def fail_run(
        self,
        run_id: str,
        error: Optional[str] = None,
        incident_id: Optional[str] = None,
    ) -> RunState:
        """
        Mark a run as failed.

        Transitions: PENDING|ACTIVE|PAUSED -> FAILED
        """
        run = self.get_run_or_raise(run_id)

        if run.status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]:
            raise ValueError(
                f"Cannot fail run in status {run.status.value}. "
                f"Already terminal."
            )

        run.status = Status.FAILED
        run.completed_at = datetime.now()

        if error:
            run.metrics["error"] = error

        if incident_id:
            run.incident_ids.append(incident_id)

        self._save_state()
        self._trim_history()

        logger.error(f"Failed run: {run_id} - {error or 'Unknown error'}")
        return run

    def cancel_run(self, run_id: str, reason: Optional[str] = None) -> RunState:
        """
        Cancel a run.

        Transitions: PENDING|ACTIVE|PAUSED -> CANCELLED
        """
        run = self.get_run_or_raise(run_id)

        if run.status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]:
            raise ValueError(
                f"Cannot cancel run in status {run.status.value}. "
                f"Already terminal."
            )

        run.status = Status.CANCELLED
        run.completed_at = datetime.now()

        if reason:
            run.metrics["cancel_reason"] = reason

        self._save_state()
        self._trim_history()

        logger.info(f"Cancelled run: {run_id}")
        return run

    # --- Progress Updates ---

    def update_progress(
        self,
        run_id: str,
        step: Optional[int] = None,
        quests_completed: Optional[int] = None,
        quests_succeeded: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> RunState:
        """
        Update run progress.

        Args:
            run_id: Run identifier
            step: Current training step
            quests_completed: Total quests attempted
            quests_succeeded: Total quests passed
            metrics: Additional metrics to merge
        """
        run = self.get_run_or_raise(run_id)

        if step is not None:
            run.current_step = step

        if quests_completed is not None:
            run.quests_completed = quests_completed

        if quests_succeeded is not None:
            run.quests_succeeded = quests_succeeded

        if metrics:
            run.metrics.update(metrics)

        self._save_state()
        return run

    def increment_progress(
        self,
        run_id: str,
        steps: int = 0,
        quests_completed: int = 0,
        quests_succeeded: int = 0,
    ) -> RunState:
        """Increment progress counters."""
        run = self.get_run_or_raise(run_id)

        run.current_step += steps
        run.quests_completed += quests_completed
        run.quests_succeeded += quests_succeeded

        self._save_state()
        return run

    def record_checkpoint(
        self,
        run_id: str,
        checkpoint_path: str,
    ) -> RunState:
        """Record a checkpoint save."""
        run = self.get_run_or_raise(run_id)

        run.last_checkpoint_step = run.current_step
        run.checkpoint_paths.append(checkpoint_path)

        self._save_state()
        logger.info(f"[{run_id}] Checkpoint at step {run.current_step}: {checkpoint_path}")
        return run

    def record_incident(
        self,
        run_id: str,
        incident_id: str,
    ) -> RunState:
        """Record an incident ID associated with this run."""
        run = self.get_run_or_raise(run_id)

        if incident_id not in run.incident_ids:
            run.incident_ids.append(incident_id)
            self._save_state()

        return run

    # --- Queries ---

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        return list(self._runs.keys())

    def list_active_runs(self) -> list[RunState]:
        """List runs that are currently active (PENDING, ACTIVE, PAUSED)."""
        return [
            r for r in self._runs.values()
            if r.status in [Status.PENDING, Status.ACTIVE, Status.PAUSED]
        ]

    def list_by_status(self, status: Status) -> list[RunState]:
        """List runs by status."""
        return [r for r in self._runs.values() if r.status == status]

    def list_by_type(self, run_type: RunType) -> list[RunState]:
        """List runs by type."""
        return [r for r in self._runs.values() if r.config.type == run_type]

    def get_current_run(self) -> Optional[RunState]:
        """Get the currently active run (if exactly one)."""
        active = self.list_by_status(Status.ACTIVE)
        if len(active) == 1:
            return active[0]
        return None

    def get_latest_run(self, run_type: Optional[RunType] = None) -> Optional[RunState]:
        """Get the most recently started run."""
        runs = list(self._runs.values())

        if run_type:
            runs = [r for r in runs if r.config.type == run_type]

        if not runs:
            return None

        # Sort by started_at, most recent first
        runs.sort(key=lambda r: r.started_at or datetime.min, reverse=True)
        return runs[0]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all runs."""
        by_status = {}
        for status in Status:
            count = len(self.list_by_status(status))
            if count > 0:
                by_status[status.value] = count

        active = self.get_current_run()

        return {
            "total_runs": len(self._runs),
            "by_status": by_status,
            "active_run_id": active.run_id if active else None,
            "active_run_type": active.config.type.value if active else None,
        }

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run from state.

        Only allows deletion of terminal runs (COMPLETED, FAILED, CANCELLED).
        """
        run = self.get_run(run_id)
        if run is None:
            return False

        if run.status not in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]:
            raise ValueError(
                f"Cannot delete run in status {run.status.value}. "
                f"Must be terminal (COMPLETED, FAILED, CANCELLED)."
            )

        del self._runs[run_id]
        self._save_state()

        logger.info(f"Deleted run: {run_id}")
        return True


# Global state manager
_manager: Optional[RunStateManager] = None


def init_state_manager(
    state_dir: Path,
    state_file: str = "run_states.json",
) -> RunStateManager:
    """Initialize the global run state manager."""
    global _manager
    _manager = RunStateManager(state_dir, state_file)
    return _manager


def get_state_manager() -> RunStateManager:
    """Get the global run state manager."""
    global _manager
    if _manager is None:
        raise RuntimeError(
            "Run state manager not initialized. "
            "Call init_state_manager() first."
        )
    return _manager


def reset_state_manager():
    """Reset the global run state manager (for testing)."""
    global _manager
    _manager = None


# Convenience functions

def create_run(config: RunConfig, run_id: Optional[str] = None) -> RunState:
    """Create a new run."""
    return get_state_manager().create_run(config, run_id)


def get_run(run_id: str) -> Optional[RunState]:
    """Get a run by ID."""
    return get_state_manager().get_run(run_id)


def start_run(run_id: str) -> RunState:
    """Start a pending run."""
    return get_state_manager().start_run(run_id)


def pause_run(run_id: str) -> RunState:
    """Pause an active run."""
    return get_state_manager().pause_run(run_id)


def resume_run(run_id: str) -> RunState:
    """Resume a paused run."""
    return get_state_manager().resume_run(run_id)


def complete_run(run_id: str, final_metrics: Optional[Dict[str, Any]] = None) -> RunState:
    """Complete a run."""
    return get_state_manager().complete_run(run_id, final_metrics)


def get_current_run() -> Optional[RunState]:
    """Get the currently active run."""
    return get_state_manager().get_current_run()
