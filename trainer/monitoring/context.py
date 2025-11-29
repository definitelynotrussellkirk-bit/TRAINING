"""Monitoring context dataclasses for training engine.

Split into sub-contexts for cleaner organization:
- ProgressContext: Batch/file progress tracking
- EvalContext: Validation and micro-eval settings
- ControlContext: Training control signals
- MonitorContext: Combined context (backwards compatible)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProgressContext:
    """Context for progress/batch tracking."""
    current_file: Optional[str] = None
    batch_number: Optional[int] = None
    batch_queue_size: Optional[int] = None


@dataclass
class EvalContext:
    """Context for validation and micro-evaluation."""
    fixed_val_dataset: Any = None
    micro_eval_inputs: Any = None
    micro_eval_interval: int = 500


@dataclass
class ControlContext:
    """Context for training control signals."""
    controller: Any = None  # TrainingController (pause/stop)


@dataclass
class MonitorContext:
    """
    Combined context for training monitors.

    Maintains backwards compatibility while supporting sub-contexts.
    Used by callbacks to access all monitoring components.
    """
    # Sub-contexts
    progress: ProgressContext = field(default_factory=ProgressContext)
    eval: EvalContext = field(default_factory=EvalContext)
    control: ControlContext = field(default_factory=ControlContext)

    # Core monitors
    live_monitor: Any = None  # LiveInferenceMonitor
    evolution_tracker: Any = None  # EvolutionTracker
    layer_monitor: Any = None  # LayerMonitor

    # Training data context
    raw_train_examples: List[Dict] = field(default_factory=list)
    logits_processor: Any = None  # LogitsProcessorList for generation

    # Remote evaluation
    remote_eval_config: Dict[str, Any] = field(default_factory=dict)

    # Status writer (optional - engine has its own, but can be overridden)
    status_writer: Any = None

    # Backwards-compatible property accessors for flat structure
    @property
    def current_file(self) -> Optional[str]:
        return self.progress.current_file

    @current_file.setter
    def current_file(self, value: Optional[str]):
        self.progress.current_file = value

    @property
    def batch_number(self) -> Optional[int]:
        return self.progress.batch_number

    @batch_number.setter
    def batch_number(self, value: Optional[int]):
        self.progress.batch_number = value

    @property
    def batch_queue_size(self) -> Optional[int]:
        return self.progress.batch_queue_size

    @batch_queue_size.setter
    def batch_queue_size(self, value: Optional[int]):
        self.progress.batch_queue_size = value

    @property
    def fixed_val_dataset(self) -> Any:
        return self.eval.fixed_val_dataset

    @fixed_val_dataset.setter
    def fixed_val_dataset(self, value: Any):
        self.eval.fixed_val_dataset = value

    @property
    def micro_eval_inputs(self) -> Any:
        return self.eval.micro_eval_inputs

    @micro_eval_inputs.setter
    def micro_eval_inputs(self, value: Any):
        self.eval.micro_eval_inputs = value

    @property
    def micro_eval_interval(self) -> int:
        return self.eval.micro_eval_interval

    @micro_eval_interval.setter
    def micro_eval_interval(self, value: int):
        self.eval.micro_eval_interval = value

    @property
    def controller(self) -> Any:
        return self.control.controller

    @controller.setter
    def controller(self, value: Any):
        self.control.controller = value
