"""
Runs system - unified execution management.

The runs module provides:
- RunConfig/RunState: Configuration and state for runs
- RunRegistry: Central access to run configurations
- RunStateManager: State persistence and lifecycle management
- RunExecutor: Orchestrates run execution with handlers and callbacks

Example:
    from guild.runs import (
        RunConfig, RunType, RunState,
        init_run_registry, get_run_config,
        init_run_state_manager, create_run, start_run,
    )

    # Initialize systems
    init_run_registry("/path/to/configs")
    init_run_state_manager("/path/to/status")

    # Create and run
    config = get_run_config("daily_training")
    run = create_run(config)
    start_run(run.run_id)
"""

# Types
from guild.runs.types import (
    RunType,
    RunConfig,
    RunState,
)

# Loader
from guild.runs.loader import (
    load_run_config,
    discover_runs,
    load_all_runs,
    RunLoader,
)

# Registry
from guild.runs.registry import (
    RunRegistry,
    init_registry as init_run_registry,
    get_registry as get_run_registry,
    reset_registry as reset_run_registry,
    get_run as get_run_config,
    list_runs as list_run_configs,
    runs_by_type as run_configs_by_type,
    runs_by_tag as run_configs_by_tag,
)

# State Management
from guild.runs.state_manager import (
    RunStateManager,
    init_state_manager as init_run_state_manager,
    get_state_manager as get_run_state_manager,
    reset_state_manager as reset_run_state_manager,
    create_run,
    get_run,
    start_run,
    pause_run,
    resume_run,
    complete_run,
    get_current_run,
)

# Executor
from guild.runs.executor import (
    RunCallback,
    RunCallbackAdapter,
    StepResult,
    RunHandler,
    RunExecutor,
    init_executor as init_run_executor,
    get_executor as get_run_executor,
    reset_executor as reset_run_executor,
)

__all__ = [
    # Types
    "RunType",
    "RunConfig",
    "RunState",
    # Loader
    "load_run_config",
    "discover_runs",
    "load_all_runs",
    "RunLoader",
    # Registry
    "RunRegistry",
    "init_run_registry",
    "get_run_registry",
    "reset_run_registry",
    "get_run_config",
    "list_run_configs",
    "run_configs_by_type",
    "run_configs_by_tag",
    # State Management
    "RunStateManager",
    "init_run_state_manager",
    "get_run_state_manager",
    "reset_run_state_manager",
    "create_run",
    "get_run",
    "start_run",
    "pause_run",
    "resume_run",
    "complete_run",
    "get_current_run",
    # Executor
    "RunCallback",
    "RunCallbackAdapter",
    "StepResult",
    "RunHandler",
    "RunExecutor",
    "init_run_executor",
    "get_run_executor",
    "reset_run_executor",
]
