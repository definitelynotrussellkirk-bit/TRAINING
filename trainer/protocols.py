"""
Trainer Protocols - Formal interfaces for training backends.

This module defines the Protocol interfaces that training engines must implement,
enabling swappable backends (HuggingFace Trainer, Muon, custom loops, etc.).

Design Philosophy:
    - TrainerEngine is the facade (what Temple calls)
    - TrainerBackend is the implementation (HF, Muon, custom)
    - Config objects are immutable snapshots
    - Results are structured and serializable

Backend Support:
    - HuggingFaceBackend: Standard transformers.Trainer
    - MuonBackend: Orthogonalized momentum optimizer
    - Future: DeepSpeed, FSDP, custom loops
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

from trainer.config.schema import TrainerConfig


class BackendType(str, Enum):
    """
    Available training backend types.

    Each backend implements TrainerBackend protocol but may have
    different capabilities and requirements.
    """
    HUGGINGFACE = "huggingface"  # transformers.Trainer
    MUON = "muon"                # Orthogonalized momentum
    CUSTOM = "custom"           # User-defined loop


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Declares what a backend can do.

    Used by Temple to:
    - Validate config against backend capabilities
    - Show available options in UI
    - Skip unsupported features gracefully
    """
    supports_fp16: bool = True
    supports_bf16: bool = True
    supports_packing: bool = True
    supports_gradient_checkpointing: bool = True
    supports_flash_attention: bool = True
    supports_lora: bool = False
    supports_deepspeed: bool = False
    supports_fsdp: bool = False
    max_batch_size: Optional[int] = None
    recommended_batch_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "supports_fp16": self.supports_fp16,
            "supports_bf16": self.supports_bf16,
            "supports_packing": self.supports_packing,
            "supports_gradient_checkpointing": self.supports_gradient_checkpointing,
            "supports_flash_attention": self.supports_flash_attention,
            "supports_lora": self.supports_lora,
            "supports_deepspeed": self.supports_deepspeed,
            "supports_fsdp": self.supports_fsdp,
            "max_batch_size": self.max_batch_size,
            "recommended_batch_size": self.recommended_batch_size,
        }


@dataclass
class TrainingProgress:
    """
    Progress update during training.

    Emitted by backends during training for:
    - Live UI updates
    - Logging
    - Early stopping decisions
    """
    global_step: int
    epoch: float
    loss: float
    learning_rate: float
    samples_per_second: float = 0.0
    grad_norm: Optional[float] = None
    memory_allocated_gb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "samples_per_second": self.samples_per_second,
            "grad_norm": self.grad_norm,
            "memory_allocated_gb": self.memory_allocated_gb,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrainingResult:
    """
    Result of a completed training run.

    Returned by TrainerEngine.train() and TrainerBackend.train().
    Contains all information needed to:
    - Record in campaign history
    - Display in UI
    - Trigger post-training rituals
    """
    success: bool
    global_step: int
    runtime_seconds: float
    final_loss: float
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: str) -> 'TrainingResult':
        """Create a failed result from error message."""
        return cls(
            success=False,
            global_step=0,
            runtime_seconds=0.0,
            final_loss=0.0,
            error_message=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "success": self.success,
            "global_step": self.global_step,
            "runtime_seconds": self.runtime_seconds,
            "final_loss": self.final_loss,
            "checkpoint_path": self.checkpoint_path,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "config_snapshot": self.config_snapshot,
        }


@dataclass
class DryRunResult:
    """
    Result of a dry-run validation.

    Returned by TrainerEngine.dry_run() to validate config
    without actually loading models or starting training.
    """
    valid: bool
    error_message: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_examples: int = 0
    estimated_vram_gb: float = 0.0
    estimated_steps: int = 0
    warnings: List[str] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, error: str) -> 'DryRunResult':
        """Create a failed result from error message."""
        return cls(valid=False, error_message=error)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "valid": self.valid,
            "error_message": self.error_message,
            "dataset_path": self.dataset_path,
            "dataset_examples": self.dataset_examples,
            "estimated_vram_gb": self.estimated_vram_gb,
            "estimated_steps": self.estimated_steps,
            "warnings": self.warnings,
            "config_summary": self.config_summary,
        }


# Callback type for progress updates
ProgressCallback = Callable[[TrainingProgress], None]


@runtime_checkable
class TrainerBackend(Protocol):
    """
    Protocol for training backend implementations.

    A backend handles the actual training loop:
    - Model loading
    - Data loading
    - Optimization
    - Checkpointing

    Implementations:
    - HuggingFaceBackend: Uses transformers.Trainer
    - MuonBackend: Custom loop with Muon optimizer
    """

    @property
    def backend_type(self) -> BackendType:
        """Return the backend type identifier."""
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return what this backend supports."""
        ...

    def validate_config(self, config: TrainerConfig) -> List[str]:
        """
        Validate config against backend capabilities.

        Returns:
            List of error messages (empty if valid)
        """
        ...

    def train(
        self,
        config: TrainerConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """
        Execute training with the given config.

        Args:
            config: Complete training configuration
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with success status and metrics
        """
        ...

    def resume(
        self,
        checkpoint_path: str,
        config: TrainerConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            config: Training configuration (may override some settings)
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with success status and metrics
        """
        ...


@runtime_checkable
class TrainerEngineProtocol(Protocol):
    """
    Protocol for the main training engine facade.

    This is what Temple calls. It:
    - Selects the appropriate backend
    - Manages the training lifecycle
    - Handles errors and recovery
    - Reports to campaign system

    The engine is backend-agnostic: it delegates actual training
    to a TrainerBackend implementation.
    """

    def dry_run(self, config: TrainerConfig) -> DryRunResult:
        """
        Validate config without starting training.

        Checks:
        - Config validity
        - Dataset accessibility
        - VRAM requirements
        - Backend compatibility

        Returns:
            DryRunResult with validation status
        """
        ...

    def train(
        self,
        config: TrainerConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """
        Execute a complete training run.

        This is the main entry point for training. It:
        1. Validates config
        2. Selects backend
        3. Executes training
        4. Records results

        Args:
            config: Complete training configuration
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with success status and metrics
        """
        ...

    def resume(
        self,
        run_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """
        Resume a previous training run.

        Args:
            run_id: ID of the run to resume
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with success status and metrics
        """
        ...

    def stop(self) -> bool:
        """
        Request graceful stop of current training.

        Returns:
            True if stop was requested, False if no training running
        """
        ...

    @property
    def is_training(self) -> bool:
        """Return True if training is in progress."""
        ...

    @property
    def current_progress(self) -> Optional[TrainingProgress]:
        """Return current training progress if training."""
        ...


@dataclass
class BackendRegistry:
    """
    Registry of available training backends.

    Used to:
    - List available backends in UI
    - Select backend based on config
    - Validate backend compatibility
    """
    _backends: Dict[BackendType, type] = field(default_factory=dict)

    def register(self, backend_type: BackendType, backend_class: type):
        """Register a backend implementation."""
        self._backends[backend_type] = backend_class

    def get(self, backend_type: BackendType) -> Optional[type]:
        """Get backend class by type."""
        return self._backends.get(backend_type)

    def list_available(self) -> List[BackendType]:
        """List available backend types."""
        return list(self._backends.keys())

    def create(self, backend_type: BackendType, **kwargs) -> TrainerBackend:
        """Create a backend instance."""
        backend_class = self._backends.get(backend_type)
        if backend_class is None:
            raise ValueError(f"Unknown backend type: {backend_type}")
        return backend_class(**kwargs)


# Global backend registry
BACKEND_REGISTRY = BackendRegistry()


def register_backend(backend_type: BackendType):
    """
    Decorator to register a backend class.

    Usage:
        @register_backend(BackendType.MUON)
        class MuonBackend:
            ...
    """
    def decorator(cls):
        BACKEND_REGISTRY.register(backend_type, cls)
        return cls
    return decorator
