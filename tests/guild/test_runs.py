"""Tests for run loading, registry, state management, and execution."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import tempfile
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from guild.types import Status
from guild.runs.types import (
    RunType,
    RunConfig,
    RunState,
)
from guild.runs.loader import (
    load_run_config,
    discover_runs,
    load_all_runs,
    RunLoader,
    _dict_to_run_config,
)
from guild.runs.registry import (
    RunRegistry,
    init_registry,
    get_registry,
    reset_registry,
    get_run,
    list_runs,
    runs_by_type,
)
from guild.runs.state_manager import (
    RunStateManager,
    init_state_manager,
    get_state_manager,
    reset_state_manager,
    create_run as create_run_func,
    start_run as start_run_func,
)
from guild.runs.executor import (
    RunCallback,
    RunCallbackAdapter,
    StepResult,
    RunHandler,
    RunExecutor,
    init_executor,
    get_executor,
    reset_executor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        runs_dir = config_dir / "runs"
        runs_dir.mkdir(parents=True)

        # Create test run configs
        training_run = """
id: test_training
type: training
name: Test Training Run
description: A test training run

facility_id: local
hero_id: test_hero

quest_filters:
  skills:
    - logic
  min_difficulty: 1
  max_difficulty: 3

max_steps: 1000
max_quests: 100
max_duration_seconds: 3600

hyperparams:
  learning_rate: 0.001
  batch_size: 16

log_level: INFO
checkpoint_every_steps: 100

tags:
  - test
  - training
"""
        (runs_dir / "test_training.yaml").write_text(training_run)

        eval_run = """
id: test_eval
type: evaluation
name: Test Evaluation Run
description: A test evaluation run

facility_id: local
hero_id: test_hero

quest_filters:
  skills:
    - logic
    - math

max_quests: 50

hyperparams:
  temperature: 0.0

tags:
  - test
  - evaluation
"""
        (runs_dir / "test_eval.yaml").write_text(eval_run)

        audit_run = """
id: test_audit
type: audit
name: Test Audit Run
description: An audit run

facility_id: local

tags:
  - test
  - audit
"""
        (runs_dir / "test_audit.yaml").write_text(audit_run)

        yield config_dir


@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before and after each test."""
    reset_registry()
    reset_state_manager()
    reset_executor()
    yield
    reset_registry()
    reset_state_manager()
    reset_executor()


@pytest.fixture
def sample_config():
    """Create a sample RunConfig."""
    return RunConfig(
        id="sample_run",
        type=RunType.TRAINING,
        name="Sample Run",
        description="A sample run for testing",
        facility_id="local",
        hero_id="test_hero",
        max_steps=100,
        max_quests=50,
    )


# =============================================================================
# Type Tests
# =============================================================================

class TestRunTypes:
    """Tests for run type definitions."""

    def test_run_type_enum(self):
        assert RunType.TRAINING.value == "training"
        assert RunType.EVALUATION.value == "evaluation"
        assert RunType.AUDIT.value == "audit"
        assert RunType.EXPERIMENT.value == "experiment"
        assert RunType.GENERATION.value == "generation"

    def test_run_config_creation(self):
        config = RunConfig(
            id="test",
            type=RunType.TRAINING,
            name="Test Run",
            facility_id="local",
        )

        assert config.id == "test"
        assert config.type == RunType.TRAINING
        assert config.name == "Test Run"
        assert config.facility_id == "local"
        assert config.max_steps is None
        assert config.checkpoint_every_steps == 1000

    def test_run_config_with_all_fields(self):
        config = RunConfig(
            id="full_test",
            type=RunType.EVALUATION,
            name="Full Test",
            description="Full config test",
            facility_id="local",
            hero_id="hero1",
            quest_filters={"skills": ["logic"]},
            max_steps=1000,
            max_quests=100,
            max_duration_seconds=3600,
            hyperparams={"lr": 0.001},
            log_level="DEBUG",
            log_facility_id="logs",
            checkpoint_every_steps=500,
            checkpoint_facility_id="checkpoints",
            tags=["test", "full"],
        )

        assert config.max_steps == 1000
        assert config.hyperparams["lr"] == 0.001
        assert "test" in config.tags

    def test_run_config_serialization(self):
        config = RunConfig(
            id="test",
            type=RunType.TRAINING,
            name="Test",
            tags=["a", "b"],
        )

        data = config.to_dict()
        assert data["id"] == "test"
        assert data["type"] == "training"
        assert data["tags"] == ["a", "b"]

        restored = RunConfig.from_dict(data)
        assert restored.id == "test"
        assert restored.type == RunType.TRAINING
        assert restored.tags == ["a", "b"]

    def test_run_state_creation(self):
        config = RunConfig(id="test", type=RunType.TRAINING)
        state = RunState(run_id="run_001", config=config)

        assert state.run_id == "run_001"
        assert state.status == Status.PENDING
        assert state.current_step == 0
        assert state.quests_completed == 0

    def test_run_state_duration(self):
        config = RunConfig(id="test", type=RunType.TRAINING)
        state = RunState(run_id="run_001", config=config)

        # No start time
        assert state.duration_seconds == 0.0

        # With start time
        state.started_at = datetime.now() - timedelta(seconds=100)
        assert state.duration_seconds >= 100.0

        # With completion
        state.completed_at = state.started_at + timedelta(seconds=50)
        assert state.duration_seconds == 50.0

    def test_run_state_success_rate(self):
        config = RunConfig(id="test", type=RunType.TRAINING)
        state = RunState(run_id="run_001", config=config)

        assert state.success_rate == 0.0

        state.quests_completed = 10
        state.quests_succeeded = 7

        assert state.success_rate == 0.7

    def test_run_state_serialization(self):
        config = RunConfig(id="test", type=RunType.TRAINING)
        state = RunState(
            run_id="run_001",
            config=config,
            status=Status.ACTIVE,
            current_step=500,
            quests_completed=50,
            quests_succeeded=40,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            metrics={"loss": 0.5},
            checkpoint_paths=["/path/to/ckpt"],
            incident_ids=["inc_001"],
        )

        data = state.to_dict()
        assert data["run_id"] == "run_001"
        assert data["status"] == "active"
        assert data["current_step"] == 500

        restored = RunState.from_dict(data)
        assert restored.run_id == "run_001"
        assert restored.status == Status.ACTIVE
        assert restored.quests_succeeded == 40


# =============================================================================
# Loader Tests
# =============================================================================

class TestRunLoader:
    """Tests for run loading from YAML."""

    def test_load_run_config(self, temp_config_dir):
        config = load_run_config("test_training", temp_config_dir)

        assert config.id == "test_training"
        assert config.type == RunType.TRAINING
        assert config.name == "Test Training Run"
        assert config.facility_id == "local"
        assert config.max_steps == 1000
        assert "test" in config.tags

    def test_load_run_not_found(self, temp_config_dir):
        with pytest.raises(FileNotFoundError):
            load_run_config("nonexistent", temp_config_dir)

    def test_discover_runs(self, temp_config_dir):
        run_ids = discover_runs(temp_config_dir)

        assert "test_training" in run_ids
        assert "test_eval" in run_ids
        assert "test_audit" in run_ids
        assert len(run_ids) == 3

    def test_load_all_runs(self, temp_config_dir):
        runs = load_all_runs(temp_config_dir)

        assert len(runs) == 3
        assert "test_training" in runs
        assert runs["test_training"].type == RunType.TRAINING
        assert runs["test_eval"].type == RunType.EVALUATION

    def test_run_loader_caching(self, temp_config_dir):
        loader = RunLoader(temp_config_dir)

        run1 = loader.load("test_training")
        run2 = loader.load("test_training")
        assert run1 is run2

        loader.clear_cache()
        run3 = loader.load("test_training")
        assert run1 is not run3

    def test_run_loader_exists(self, temp_config_dir):
        loader = RunLoader(temp_config_dir)

        assert loader.exists("test_training") is True
        assert loader.exists("nonexistent") is False

    def test_dict_to_run_config_missing_id(self):
        with pytest.raises(ValueError, match="missing 'id'"):
            _dict_to_run_config({"type": "training"})

    def test_dict_to_run_config_missing_type(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            _dict_to_run_config({"id": "test"})

    def test_dict_to_run_config_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid run type"):
            _dict_to_run_config({"id": "test", "type": "invalid"})


# =============================================================================
# Registry Tests
# =============================================================================

class TestRunRegistry:
    """Tests for run registry."""

    def test_registry_get(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        run = registry.get("test_training")
        assert run.id == "test_training"

    def test_registry_get_unknown(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        with pytest.raises(KeyError, match="Unknown run"):
            registry.get("nonexistent")

    def test_registry_get_or_none(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        run = registry.get_or_none("test_training")
        assert run is not None

        run = registry.get_or_none("nonexistent")
        assert run is None

    def test_registry_exists(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        assert registry.exists("test_training") is True
        assert registry.exists("nonexistent") is False
        assert "test_training" in registry

    def test_registry_list_ids(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        ids = registry.list_ids()
        assert "test_training" in ids
        assert "test_eval" in ids

    def test_registry_all(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        all_runs = registry.all()
        assert len(all_runs) == 3

    def test_registry_iteration(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        runs = list(registry)
        assert len(runs) == 3

    def test_registry_by_type(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        training_runs = registry.by_type(RunType.TRAINING)
        assert len(training_runs) == 1
        assert training_runs[0].id == "test_training"

        eval_runs = registry.by_type(RunType.EVALUATION)
        assert len(eval_runs) == 1

    def test_registry_by_tag(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        test_runs = registry.by_tag("test")
        assert len(test_runs) == 3

        training_runs = registry.by_tag("training")
        assert len(training_runs) == 1

    def test_registry_search(self, temp_config_dir):
        registry = RunRegistry(temp_config_dir)

        results = registry.search(run_type=RunType.TRAINING)
        assert len(results) == 1

        results = registry.search(tags=["test", "training"])
        assert len(results) == 1

        results = registry.search(name_contains="eval")
        assert len(results) == 1


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_init_and_get_registry(self, temp_config_dir):
        init_registry(temp_config_dir)

        registry = get_registry()
        assert registry is not None
        assert "test_training" in registry

    def test_get_run_convenience(self, temp_config_dir):
        init_registry(temp_config_dir)

        run = get_run("test_training")
        assert run.id == "test_training"

    def test_list_runs_convenience(self, temp_config_dir):
        init_registry(temp_config_dir)

        ids = list_runs()
        assert "test_training" in ids

    def test_runs_by_type_convenience(self, temp_config_dir):
        init_registry(temp_config_dir)

        runs = runs_by_type(RunType.TRAINING)
        assert len(runs) == 1


# =============================================================================
# State Manager Tests
# =============================================================================

class TestRunStateManager:
    """Tests for run state management."""

    def test_create_manager(self, temp_state_dir):
        manager = RunStateManager(temp_state_dir)
        assert manager.state_file.parent.exists()

    def test_create_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        run = manager.create_run(sample_config)

        assert run.run_id.startswith("sample_run_")
        assert run.status == Status.PENDING
        assert run.config.id == "sample_run"

    def test_create_run_with_explicit_id(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        run = manager.create_run(sample_config, run_id="explicit_id")

        assert run.run_id == "explicit_id"

    def test_create_run_duplicate_id(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="dup_id")

        with pytest.raises(ValueError, match="already exists"):
            manager.create_run(sample_config, run_id="dup_id")

    def test_get_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        created = manager.create_run(sample_config, run_id="test_run")
        retrieved = manager.get_run("test_run")

        assert retrieved is not None
        assert retrieved.run_id == "test_run"

    def test_get_run_not_found(self, temp_state_dir):
        manager = RunStateManager(temp_state_dir)

        run = manager.get_run("nonexistent")
        assert run is None

    def test_start_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        run = manager.start_run("test_run")

        assert run.status == Status.ACTIVE
        assert run.started_at is not None

    def test_start_run_wrong_status(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")

        with pytest.raises(ValueError, match="Must be PENDING"):
            manager.start_run("test_run")

    def test_pause_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")
        run = manager.pause_run("test_run")

        assert run.status == Status.PAUSED
        assert run.paused_at is not None

    def test_resume_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")
        manager.pause_run("test_run")
        run = manager.resume_run("test_run")

        assert run.status == Status.ACTIVE
        assert run.paused_at is None

    def test_complete_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")
        run = manager.complete_run("test_run", {"final_loss": 0.1})

        assert run.status == Status.COMPLETED
        assert run.completed_at is not None
        assert run.metrics["final_loss"] == 0.1

    def test_fail_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")
        run = manager.fail_run("test_run", "OOM Error", "inc_001")

        assert run.status == Status.FAILED
        assert run.metrics["error"] == "OOM Error"
        assert "inc_001" in run.incident_ids

    def test_cancel_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")
        run = manager.cancel_run("test_run", "User requested")

        assert run.status == Status.CANCELLED
        assert run.metrics["cancel_reason"] == "User requested"

    def test_update_progress(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")

        run = manager.update_progress(
            "test_run",
            step=100,
            quests_completed=50,
            quests_succeeded=40,
            metrics={"loss": 0.5},
        )

        assert run.current_step == 100
        assert run.quests_completed == 50
        assert run.quests_succeeded == 40
        assert run.metrics["loss"] == 0.5

    def test_increment_progress(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")

        manager.increment_progress("test_run", steps=10, quests_completed=5, quests_succeeded=3)
        manager.increment_progress("test_run", steps=10, quests_completed=5, quests_succeeded=4)

        run = manager.get_run("test_run")
        assert run.current_step == 20
        assert run.quests_completed == 10
        assert run.quests_succeeded == 7

    def test_record_checkpoint(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")
        manager.start_run("test_run")
        manager.update_progress("test_run", step=100)

        run = manager.record_checkpoint("test_run", "/path/to/ckpt-100")

        assert run.last_checkpoint_step == 100
        assert "/path/to/ckpt-100" in run.checkpoint_paths

    def test_record_incident(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="test_run")

        run = manager.record_incident("test_run", "inc_001")
        run = manager.record_incident("test_run", "inc_002")
        run = manager.record_incident("test_run", "inc_001")  # Duplicate

        assert len(run.incident_ids) == 2

    def test_list_runs(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="run1")
        manager.create_run(sample_config, run_id="run2")

        runs = manager.list_runs()
        assert "run1" in runs
        assert "run2" in runs

    def test_list_active_runs(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="pending")
        manager.create_run(sample_config, run_id="active")
        manager.start_run("active")
        manager.create_run(sample_config, run_id="completed")
        manager.start_run("completed")
        manager.complete_run("completed")

        active = manager.list_active_runs()

        run_ids = [r.run_id for r in active]
        assert "pending" in run_ids
        assert "active" in run_ids
        assert "completed" not in run_ids

    def test_list_by_status(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="pending")
        manager.create_run(sample_config, run_id="active")
        manager.start_run("active")

        pending = manager.list_by_status(Status.PENDING)
        active = manager.list_by_status(Status.ACTIVE)

        assert len(pending) == 1
        assert len(active) == 1

    def test_get_current_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        # No active runs
        assert manager.get_current_run() is None

        manager.create_run(sample_config, run_id="run1")
        manager.start_run("run1")

        current = manager.get_current_run()
        assert current.run_id == "run1"

    def test_persistence(self, temp_state_dir, sample_config):
        # First manager
        manager1 = RunStateManager(temp_state_dir)
        manager1.create_run(sample_config, run_id="persist_test")
        manager1.start_run("persist_test")
        manager1.update_progress("persist_test", step=100)

        # New manager loads from disk
        manager2 = RunStateManager(temp_state_dir)

        run = manager2.get_run("persist_test")
        assert run is not None
        assert run.status == Status.ACTIVE
        assert run.current_step == 100

    def test_history_limit(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir, history_limit=2)

        # Create and complete several runs
        for i in range(5):
            manager.create_run(sample_config, run_id=f"run{i}")
            manager.start_run(f"run{i}")
            manager.complete_run(f"run{i}")

        # Should only keep latest 2 completed
        completed = manager.list_by_status(Status.COMPLETED)
        assert len(completed) == 2

    def test_delete_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="to_delete")
        manager.start_run("to_delete")
        manager.complete_run("to_delete")

        result = manager.delete_run("to_delete")
        assert result is True
        assert manager.get_run("to_delete") is None

    def test_delete_run_active_fails(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)

        manager.create_run(sample_config, run_id="active_run")
        manager.start_run("active_run")

        with pytest.raises(ValueError, match="Cannot delete"):
            manager.delete_run("active_run")


class TestGlobalStateManager:
    """Tests for global state manager functions."""

    def test_init_and_get_state_manager(self, temp_state_dir):
        init_state_manager(temp_state_dir)

        manager = get_state_manager()
        assert manager is not None

    def test_get_state_manager_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_state_manager()

    def test_convenience_functions(self, temp_state_dir):
        init_state_manager(temp_state_dir)

        config = RunConfig(id="test", type=RunType.TRAINING)
        run = create_run_func(config, "conv_test")

        assert run.run_id == "conv_test"

        started = start_run_func("conv_test")
        assert started.status == Status.ACTIVE


# =============================================================================
# Executor Tests
# =============================================================================

class MockHandler(RunHandler):
    """Mock handler for testing."""

    def __init__(self, max_steps: int = 10):
        self.initialized = False
        self.cleaned_up = False
        self.steps_executed = 0
        self.max_steps = max_steps
        self.checkpoints_saved = []

    def initialize(self, run: RunState) -> None:
        self.initialized = True

    def execute_step(self, run: RunState) -> StepResult:
        self.steps_executed += 1
        return StepResult(
            step=run.current_step + 1,
            quests_attempted=1,
            quests_succeeded=1 if self.steps_executed % 2 == 0 else 0,
            metrics={"step_loss": 1.0 / self.steps_executed},
            should_stop=self.steps_executed >= self.max_steps,
            stop_reason="Max steps reached" if self.steps_executed >= self.max_steps else None,
        )

    def save_checkpoint(self, run: RunState) -> str:
        path = f"/ckpt/step-{run.current_step}"
        self.checkpoints_saved.append(path)
        return path

    def cleanup(self, run: RunState) -> None:
        self.cleaned_up = True


class MockCallback(RunCallbackAdapter):
    """Mock callback for testing."""

    def __init__(self):
        self.events = []

    def on_run_start(self, run: RunState) -> None:
        self.events.append(("start", run.run_id))

    def on_run_pause(self, run: RunState) -> None:
        self.events.append(("pause", run.run_id))

    def on_run_resume(self, run: RunState) -> None:
        self.events.append(("resume", run.run_id))

    def on_run_complete(self, run: RunState) -> None:
        self.events.append(("complete", run.run_id))

    def on_run_fail(self, run: RunState, error: str) -> None:
        self.events.append(("fail", run.run_id, error))

    def on_step(self, run: RunState, step: int) -> None:
        self.events.append(("step", run.run_id, step))

    def on_checkpoint(self, run: RunState, checkpoint_path: str) -> None:
        self.events.append(("checkpoint", run.run_id, checkpoint_path))


class TestRunExecutor:
    """Tests for run executor."""

    def test_create_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        executor = RunExecutor(manager)

        run_id = executor.create_run(sample_config)

        assert run_id.startswith("sample_run_")
        assert manager.get_run(run_id) is not None

    def test_start_run(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        handler = MockHandler()
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(sample_config)
        run = executor.start(run_id)

        assert run.status == Status.ACTIVE
        assert handler.initialized is True

    def test_pause_resume(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        executor = RunExecutor(manager)

        run_id = executor.create_run(sample_config)
        executor.start(run_id)

        run = executor.pause(run_id)
        assert run.status == Status.PAUSED

        run = executor.resume(run_id)
        assert run.status == Status.ACTIVE

    def test_cancel(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        handler = MockHandler()
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(sample_config)
        executor.start(run_id)
        run = executor.cancel(run_id, "User cancelled")

        assert run.status == Status.CANCELLED
        assert handler.cleaned_up is True

    def test_step(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        handler = MockHandler()
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(sample_config)
        executor.start(run_id)

        result = executor.step(run_id)

        assert result.step == 1
        assert handler.steps_executed == 1

        run = manager.get_run(run_id)
        assert run.current_step == 1

    def test_callbacks(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        handler = MockHandler()
        callback = MockCallback()
        executor = RunExecutor(manager, handler, [callback])

        run_id = executor.create_run(sample_config)
        executor.start(run_id)
        executor.step(run_id)
        executor.pause(run_id)
        executor.resume(run_id)
        executor.complete(run_id)

        event_types = [e[0] for e in callback.events]
        assert "start" in event_types
        assert "step" in event_types
        assert "pause" in event_types
        assert "resume" in event_types
        assert "complete" in event_types

    def test_should_continue(self, temp_state_dir):
        config = RunConfig(
            id="limited",
            type=RunType.TRAINING,
            max_steps=5,
        )

        manager = RunStateManager(temp_state_dir)
        executor = RunExecutor(manager)

        run_id = executor.create_run(config)
        executor.start(run_id)

        # Within limits
        assert executor.should_continue(run_id) is True

        # Exceed step limit
        manager.update_progress(run_id, step=5)
        assert executor.should_continue(run_id) is False

    def test_request_pause(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        executor = RunExecutor(manager)

        run_id = executor.create_run(sample_config)
        executor.start(run_id)

        assert executor.is_pause_requested(run_id) is False

        executor.request_pause(run_id)

        assert executor.is_pause_requested(run_id) is True
        assert executor.should_continue(run_id) is False

    def test_run_to_completion(self, temp_state_dir):
        config = RunConfig(
            id="auto",
            type=RunType.TRAINING,
        )

        manager = RunStateManager(temp_state_dir)
        handler = MockHandler(max_steps=5)
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(config)
        run = executor.run_to_completion(run_id)

        assert run.status == Status.COMPLETED
        assert handler.steps_executed == 5
        assert run.current_step == 5

    def test_checkpoint_during_execution(self, temp_state_dir):
        config = RunConfig(
            id="ckpt_test",
            type=RunType.TRAINING,
            checkpoint_every_steps=2,
        )

        manager = RunStateManager(temp_state_dir)
        handler = MockHandler(max_steps=5)
        callback = MockCallback()
        executor = RunExecutor(manager, handler, [callback])

        run_id = executor.create_run(config)
        executor.run_to_completion(run_id)

        # Should have saved checkpoints at steps 2, 4
        checkpoint_events = [e for e in callback.events if e[0] == "checkpoint"]
        assert len(checkpoint_events) == 2

    def test_get_status(self, temp_state_dir, sample_config):
        manager = RunStateManager(temp_state_dir)
        handler = MockHandler()
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(sample_config)
        executor.start(run_id)
        executor.step(run_id)
        executor.step(run_id)

        status = executor.get_status(run_id)

        assert status["exists"] is True
        assert status["step"] == 2
        assert status["status"] == "active"


class TestGlobalExecutor:
    """Tests for global executor functions."""

    def test_init_and_get_executor(self, temp_state_dir):
        manager = RunStateManager(temp_state_dir)
        init_executor(manager)

        executor = get_executor()
        assert executor is not None

    def test_get_executor_auto_init(self):
        # Should auto-init with default state manager
        # This will fail because state manager isn't initialized
        reset_executor()
        with pytest.raises(RuntimeError):
            executor = get_executor()
            executor.get_run("test")  # This needs state manager


# =============================================================================
# Integration Tests
# =============================================================================

class TestRunsIntegration:
    """Integration tests for the runs module."""

    def test_full_workflow(self, temp_config_dir, temp_state_dir):
        """Test complete workflow: load config, create run, execute."""
        # Initialize systems
        init_registry(temp_config_dir)
        init_state_manager(temp_state_dir)

        # Load config from registry
        config = get_run("test_training")
        assert config.type == RunType.TRAINING

        # Create and start run
        manager = get_state_manager()
        handler = MockHandler(max_steps=10)
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(config)
        run = executor.run_to_completion(run_id)

        assert run.status == Status.COMPLETED
        assert run.current_step == 10

    def test_pause_and_resume_workflow(self, temp_state_dir, sample_config):
        """Test pausing and resuming runs."""
        manager = RunStateManager(temp_state_dir)
        handler = MockHandler(max_steps=100)
        executor = RunExecutor(manager, handler)

        run_id = executor.create_run(sample_config)
        executor.start(run_id)

        # Run some steps
        for _ in range(5):
            executor.step(run_id)

        # Pause
        executor.pause(run_id)
        run = manager.get_run(run_id)
        assert run.status == Status.PAUSED
        assert run.current_step == 5

        # Resume
        executor.resume(run_id)

        # Run more steps
        for _ in range(5):
            executor.step(run_id)

        run = manager.get_run(run_id)
        assert run.status == Status.ACTIVE
        assert run.current_step == 10

    def test_multiple_runs(self, temp_state_dir):
        """Test managing multiple concurrent runs."""
        manager = RunStateManager(temp_state_dir)

        # Create different types of runs
        configs = [
            RunConfig(id="train1", type=RunType.TRAINING, tags=["batch1"]),
            RunConfig(id="eval1", type=RunType.EVALUATION, tags=["batch1"]),
            RunConfig(id="train2", type=RunType.TRAINING, tags=["batch2"]),
        ]

        run_ids = []
        for config in configs:
            run = manager.create_run(config)
            run_ids.append(run.run_id)

        # Query by type
        training = manager.list_by_type(RunType.TRAINING)
        assert len(training) == 2

        evaluation = manager.list_by_type(RunType.EVALUATION)
        assert len(evaluation) == 1


# =============================================================================
# Test with Real Configs
# =============================================================================

class TestRealConfigs:
    """Tests with real config files."""

    def test_load_real_runs(self):
        """Test loading real run configs."""
        real_config_dir = project_root / "configs"

        if not (real_config_dir / "runs").exists():
            pytest.skip("No real run configs found")

        runs = discover_runs(real_config_dir)

        if not runs:
            pytest.skip("No run configs in configs/runs/")

        # Load first run
        run = load_run_config(runs[0], real_config_dir)
        assert run.id is not None
        assert run.type is not None
