#!/usr/bin/env python3
"""
Monitoring Bundle - Orchestrates training monitoring components.

This module bundles together various monitoring components:
- TrainingStatusWriter: Real-time status JSON updates
- LiveInferenceMonitor: Inference monitoring during training
- EvolutionTracker: Track training evolution over time

Usage:
    from training.monitoring_bundle import MonitoringBundle, MonitoringConfig

    config = MonitoringConfig(
        status_dir=Path("status"),
        enable_live_monitor=True
    )

    bundle = MonitoringBundle(config)
    bundle.start(model, tokenizer, val_examples)

    # During training
    bundle.update_progress(step, loss, metrics)

    # Cleanup
    bundle.stop()
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """
    Configuration for training monitoring.

    Attributes:
        status_dir: Directory for status files
        status_filename: Name of status JSON file
        enable_live_monitor: Whether to enable live inference monitoring
        enable_evolution_tracker: Whether to track training evolution
        num_eval_samples: Number of samples for live monitoring
        max_eval_tokens: Max tokens for live inference
        eval_steps: Run evaluation every N steps
    """
    status_dir: Path = field(default_factory=lambda: Path("status"))
    status_filename: str = "training_status.json"
    enable_live_monitor: bool = True
    enable_evolution_tracker: bool = True
    num_eval_samples: int = 5
    max_eval_tokens: int = 2048
    eval_steps: int = 100


@dataclass
class MonitoringState:
    """
    Current monitoring state.

    Attributes:
        is_running: Whether monitoring is active
        current_step: Current training step
        total_steps: Total expected steps
        start_time: Training start time
        last_loss: Last recorded loss
        last_metrics: Last recorded metrics
    """
    is_running: bool = False
    current_step: int = 0
    total_steps: int = 0
    start_time: Optional[datetime] = None
    last_loss: Optional[float] = None
    last_metrics: Dict[str, Any] = field(default_factory=dict)


class MonitoringBundle:
    """
    Orchestrates training monitoring components.

    Manages lifecycle of:
    - Status writer (JSON updates)
    - Live inference monitor (optional)
    - Evolution tracker (optional)

    Example:
        config = MonitoringConfig(
            status_dir=Path("status"),
            enable_live_monitor=True
        )

        bundle = MonitoringBundle(config)
        bundle.start(model, tokenizer, val_examples, total_steps=1000)

        # During training callback
        bundle.update_progress(step=100, loss=0.5)

        # End of training
        bundle.stop()
    """

    def __init__(self, config: MonitoringConfig):
        """
        Initialize monitoring bundle.

        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.state = MonitoringState()

        # Component references (set in start())
        self.status_writer = None
        self.live_monitor = None
        self.evolution_tracker = None

        # Ensure status directory exists
        self.config.status_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        model: Any,
        tokenizer: Any,
        val_examples: List[Dict],
        total_steps: int,
        dataset_name: str = "training",
        base_dir: Optional[Path] = None,
        logits_processor: Any = None
    ) -> None:
        """
        Start monitoring.

        Args:
            model: The model being trained
            tokenizer: Tokenizer for the model
            val_examples: Validation examples for live monitoring
            total_steps: Total expected training steps
            dataset_name: Name of dataset (for evolution tracking)
            base_dir: Base directory (for evolution tracking)
            logits_processor: Optional logits processor for generation
        """
        self.state = MonitoringState(
            is_running=True,
            total_steps=total_steps,
            start_time=datetime.now()
        )

        # Initialize status writer
        status_path = self.config.status_dir / self.config.status_filename
        self.status_writer = StatusWriter(str(status_path))
        self.status_writer.start(total_steps)
        logger.info(f"Status writer initialized: {status_path}")

        # Initialize live monitor if enabled
        if self.config.enable_live_monitor and val_examples:
            try:
                # Import here to avoid circular imports
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / "monitoring" / "servers"))
                from live_monitor import LiveInferenceMonitor

                self.live_monitor = LiveInferenceMonitor(
                    model=model,
                    tokenizer=tokenizer,
                    val_examples=val_examples,
                    num_samples=self.config.num_eval_samples,
                    max_new_tokens=self.config.max_eval_tokens,
                    logits_processor=logits_processor
                )
                logger.info("Live inference monitor initialized")
            except ImportError as e:
                logger.warning(f"Could not initialize live monitor: {e}")

        # Initialize evolution tracker if enabled
        if self.config.enable_evolution_tracker and base_dir:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / "monitoring" / "servers"))
                from evolution_tracker import EvolutionTracker

                self.evolution_tracker = EvolutionTracker(base_dir, dataset_name)
                logger.info(f"Evolution tracker initialized for: {dataset_name}")
            except ImportError as e:
                logger.warning(f"Could not initialize evolution tracker: {e}")

    def update_progress(
        self,
        step: int,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        run_eval: bool = False
    ) -> None:
        """
        Update training progress.

        Args:
            step: Current training step
            loss: Current loss value
            metrics: Additional metrics dict
            run_eval: Whether to run live evaluation
        """
        self.state.current_step = step
        self.state.last_loss = loss
        if metrics:
            self.state.last_metrics.update(metrics)

        # Update status writer
        if self.status_writer:
            self.status_writer.update(
                step=step,
                loss=loss,
                **self.state.last_metrics
            )

        # Run live evaluation if requested
        if run_eval and self.live_monitor:
            try:
                results = self.live_monitor.run_inference()
                if self.status_writer:
                    self.status_writer.update_inference(results)
            except Exception as e:
                logger.warning(f"Live evaluation failed: {e}")

    def update_inference(self, results: Dict[str, Any]) -> None:
        """Update with inference results."""
        if self.status_writer:
            self.status_writer.update_inference(results)

    def mark_completed(self) -> None:
        """Mark training as completed."""
        if self.status_writer:
            self.status_writer.mark_completed(
                self.state.current_step,
                self.state.total_steps
            )

    def mark_crashed(self, error: str, error_type: str = "Unknown") -> None:
        """Mark training as crashed."""
        if self.status_writer:
            self.status_writer.mark_crashed(error, error_type)

    def stop(self) -> None:
        """Stop monitoring and cleanup."""
        self.state.is_running = False

        # Cleanup components
        if self.live_monitor:
            self.live_monitor = None

        if self.evolution_tracker:
            self.evolution_tracker = None

        logger.info("Monitoring stopped")

    def get_state(self) -> Dict[str, Any]:
        """Get current monitoring state."""
        return {
            "is_running": self.state.is_running,
            "current_step": self.state.current_step,
            "total_steps": self.state.total_steps,
            "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
            "last_loss": self.state.last_loss,
            "elapsed_seconds": (datetime.now() - self.state.start_time).total_seconds() if self.state.start_time else 0
        }


class StatusWriter:
    """
    Simple status writer for training status JSON.

    This is a simplified version - the full TrainingStatusWriter
    from core/training_status.py has more features.
    """

    def __init__(self, status_file: str):
        """Initialize status writer."""
        self.status_file = Path(status_file)
        self.status = {
            "status": "initializing",
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "started_at": None,
            "updated_at": None
        }

    def start(self, total_steps: int) -> None:
        """Mark training as started."""
        self.status["status"] = "training"
        self.status["total_steps"] = total_steps
        self.status["started_at"] = datetime.now().isoformat()
        self._write()

    def update(self, step: int, loss: Optional[float] = None, **metrics) -> None:
        """Update training progress."""
        self.status["current_step"] = step
        self.status["loss"] = loss
        self.status["updated_at"] = datetime.now().isoformat()
        self.status.update(metrics)
        self._write()

    def update_inference(self, results: Dict[str, Any]) -> None:
        """Update with inference results."""
        self.status["last_inference"] = results
        self.status["updated_at"] = datetime.now().isoformat()
        self._write()

    def mark_completed(self, step: int, total: int) -> None:
        """Mark training as completed."""
        self.status["status"] = "completed"
        self.status["current_step"] = step
        self.status["total_steps"] = total
        self.status["completed_at"] = datetime.now().isoformat()
        self._write()

    def mark_crashed(self, error: str, error_type: str) -> None:
        """Mark training as crashed."""
        self.status["status"] = "crashed"
        self.status["error"] = error
        self.status["error_type"] = error_type
        self.status["crashed_at"] = datetime.now().isoformat()
        self._write()

    def _write(self) -> None:
        """Write status to file."""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write status: {e}")


if __name__ == "__main__":
    # Quick test
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = MonitoringConfig(
            status_dir=Path(tmpdir),
            enable_live_monitor=False,  # Skip for test
            enable_evolution_tracker=False  # Skip for test
        )

        bundle = MonitoringBundle(config)

        # Simulate training
        bundle.start(
            model=None,  # Would be actual model
            tokenizer=None,
            val_examples=[],
            total_steps=100
        )

        # Simulate progress
        for step in [10, 20, 30]:
            bundle.update_progress(step=step, loss=1.0 - step/100)

        # Check status file
        status_file = Path(tmpdir) / "training_status.json"
        with open(status_file) as f:
            status = json.load(f)

        print(f"\nStatus file contents:")
        print(json.dumps(status, indent=2))

        # Mark completed
        bundle.mark_completed()
        bundle.stop()

        print("\nMonitoringBundle ready for use!")
