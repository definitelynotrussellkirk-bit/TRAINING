#!/usr/bin/env python3
"""
First-class Job Abstraction

Unified representation of a training job that provides a single source of truth
for job identity and state across the system.

Previously, job concepts were scattered across:
- TrainingQueue entries (file path + metadata)
- JobLogger events (job_id, timestamps, state)
- TrainingResult (success, loss, global_step)

This module consolidates into a single Job dataclass used by all systems.

Usage:
    from core.job import Job

    # Create from file path
    job = Job.from_file("/path/to/data.jsonl", priority="high")

    # Track lifecycle
    job.start(step=84000)
    # ... training runs ...
    job.complete(final_step=85000, final_loss=0.44)

    # Serialize for storage
    job_dict = job.to_dict()
    restored = Job.from_dict(job_dict)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import hashlib
import subprocess

# Type alias for job status
# State machine: queued → processing → (paused ↔ processing) → (stopped | completed | failed | skipped)
JobStatus = Literal['queued', 'processing', 'paused', 'stopped', 'completed', 'failed', 'skipped']


@dataclass
class Job:
    """
    Complete training job representation.

    Single source of truth for job identity and state.
    Used by TrainingQueue, JobLogger, and monitoring API.

    Lifecycle States:
        - queued: In queue, waiting for training
        - processing: Training in progress
        - completed: Training finished successfully
        - failed: Training failed (see last_error)
        - skipped: Skipped by user or system

    Attributes:
        id: Unique identifier (YYYY-MM-DDTHH:MM:SS_filename)
        dataset_path: Full path to .jsonl training file
        hero_id: Hero identifier (for campaign-based training)
        campaign_id: Campaign identifier
        priority: Queue priority (high, normal, low)
        status: Current lifecycle state
        attempts: Number of training attempts
        created_at: Timestamp when queued
        started_at: Timestamp when training started
        finished_at: Timestamp when training finished
        start_step: Global step when training started
        final_step: Global step when training finished
        final_loss: Final training loss
        final_val_loss: Final validation loss
        last_error: Error message if failed/skipped
        num_examples: Number of training examples
        estimated_tokens: Estimated total tokens
        config_hash: Hash of training config for reproducibility
        git_commit: Git commit for reproducibility
        metadata: Extensible metadata dict
    """
    # Identity
    id: str
    dataset_path: str

    # Context (for campaign-based training)
    hero_id: Optional[str] = None
    campaign_id: Optional[str] = None

    # Queue management
    priority: str = 'normal'
    status: JobStatus = 'queued'
    attempts: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # Training metrics
    start_step: Optional[int] = None
    final_step: Optional[int] = None
    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    tokens_per_sec_avg: Optional[float] = None

    # Error info
    last_error: Optional[str] = None

    # Dataset info
    num_examples: Optional[int] = None
    estimated_tokens: Optional[int] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None

    # Reproducibility
    config_hash: Optional[str] = None
    git_commit: Optional[str] = None
    checkpoint_used: Optional[str] = None
    best_checkpoint_after: Optional[str] = None

    # GPU info
    gpu_type: Optional[str] = None
    gpu_memory_gb: Optional[float] = None

    # Alerting
    regression_flag: bool = False
    alert_count: int = 0

    # Skill context (optional - only for skill-based jobs)
    skill_id: Optional[str] = None
    skill_level: Optional[int] = None
    skill_target_accuracy: Optional[float] = None
    skill_accuracy: Optional[float] = None  # Set after eval

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ========== Factory Methods ==========

    @classmethod
    def from_file(
        cls,
        path: str,
        priority: str = 'normal',
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> 'Job':
        """
        Create job from dataset path.

        Args:
            path: Path to .jsonl training file
            priority: Queue priority (high, normal, low)
            hero_id: Optional hero identifier
            campaign_id: Optional campaign identifier

        Returns:
            New Job instance in 'queued' state
        """
        filename = Path(path).name
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        job_id = f"{timestamp}_{filename}"

        return cls(
            id=job_id,
            dataset_path=str(path),
            priority=priority,
            hero_id=hero_id,
            campaign_id=campaign_id,
            git_commit=cls._get_git_commit(),
        )

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    # ========== Lifecycle Methods ==========

    def start(
        self,
        step: Optional[int] = None,
        checkpoint: Optional[str] = None,
        gpu_type: Optional[str] = None,
        gpu_memory_gb: Optional[float] = None,
    ) -> 'Job':
        """
        Mark job as started.

        Args:
            step: Current global step
            checkpoint: Checkpoint being resumed from
            gpu_type: GPU type (e.g., "RTX 4090")
            gpu_memory_gb: GPU VRAM

        Returns:
            self (for chaining)
        """
        self.status = 'processing'
        self.started_at = datetime.now().isoformat()
        self.start_step = step
        self.checkpoint_used = checkpoint
        self.gpu_type = gpu_type
        self.gpu_memory_gb = gpu_memory_gb
        self.attempts += 1
        return self

    def complete(
        self,
        final_step: int,
        final_loss: float,
        final_val_loss: Optional[float] = None,
        tokens_per_sec_avg: Optional[float] = None,
        best_checkpoint: Optional[str] = None,
        regression_flag: bool = False,
        alert_count: int = 0,
    ) -> 'Job':
        """
        Mark job as completed.

        Args:
            final_step: Final global step
            final_loss: Final training loss
            final_val_loss: Final validation loss
            tokens_per_sec_avg: Average throughput
            best_checkpoint: Best checkpoint produced
            regression_flag: Whether regression was detected
            alert_count: Number of alerts during training

        Returns:
            self (for chaining)
        """
        self.status = 'completed'
        self.finished_at = datetime.now().isoformat()
        self.final_step = final_step
        self.final_loss = final_loss
        self.final_val_loss = final_val_loss
        self.tokens_per_sec_avg = tokens_per_sec_avg
        self.best_checkpoint_after = best_checkpoint
        self.regression_flag = regression_flag
        self.alert_count = alert_count
        self._calculate_actual_hours()
        return self

    def fail(
        self,
        error: str,
        final_step: Optional[int] = None,
        final_loss: Optional[float] = None,
    ) -> 'Job':
        """
        Mark job as failed.

        Args:
            error: Error message
            final_step: Step when failure occurred
            final_loss: Last known loss

        Returns:
            self (for chaining)
        """
        self.status = 'failed'
        self.finished_at = datetime.now().isoformat()
        self.last_error = error
        self.final_step = final_step
        self.final_loss = final_loss
        self._calculate_actual_hours()
        return self

    def skip(self, reason: str) -> 'Job':
        """
        Mark job as skipped.

        Args:
            reason: Reason for skipping

        Returns:
            self (for chaining)
        """
        self.status = 'skipped'
        self.finished_at = datetime.now().isoformat()
        self.last_error = reason
        return self

    def pause(self, reason: Optional[str] = None) -> 'Job':
        """
        Mark job as paused.

        Only valid when status is 'processing'.

        Args:
            reason: Optional reason for pausing

        Returns:
            self (for chaining)
        """
        if self.status != 'processing':
            raise ValueError(f"Cannot pause job in state '{self.status}', must be 'processing'")
        self.status = 'paused'
        if reason:
            self.metadata['pause_reason'] = reason
        self.metadata['paused_at'] = datetime.now().isoformat()
        return self

    def resume(self) -> 'Job':
        """
        Resume a paused job.

        Only valid when status is 'paused'.

        Returns:
            self (for chaining)
        """
        if self.status != 'paused':
            raise ValueError(f"Cannot resume job in state '{self.status}', must be 'paused'")
        self.status = 'processing'
        self.metadata['resumed_at'] = datetime.now().isoformat()
        return self

    def stop(self, reason: str = "User requested stop") -> 'Job':
        """
        Stop job gracefully (user-initiated).

        Different from fail() - stopped is a clean exit, not an error.

        Args:
            reason: Reason for stopping

        Returns:
            self (for chaining)
        """
        self.status = 'stopped'
        self.finished_at = datetime.now().isoformat()
        self.last_error = reason
        self._calculate_actual_hours()
        return self

    def _calculate_actual_hours(self):
        """Calculate actual duration from timestamps."""
        if self.started_at and self.finished_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                end = datetime.fromisoformat(self.finished_at)
                self.actual_hours = (end - start).total_seconds() / 3600
            except Exception:
                pass

    # ========== Serialization ==========

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Deserialize from dictionary."""
        # Handle metadata separately to avoid issues with default_factory
        metadata = data.pop('metadata', {})
        job = cls(**data)
        job.metadata = metadata
        return job

    # ========== Properties ==========

    @property
    def duration_hours(self) -> Optional[float]:
        """Calculate duration if finished."""
        if self.actual_hours:
            return self.actual_hours
        if self.started_at and self.finished_at:
            try:
                start = datetime.fromisoformat(self.started_at)
                end = datetime.fromisoformat(self.finished_at)
                return (end - start).total_seconds() / 3600
            except Exception:
                pass
        return None

    @property
    def val_train_gap(self) -> Optional[float]:
        """Calculate validation/training loss gap."""
        if self.final_val_loss is not None and self.final_loss is not None:
            return self.final_val_loss - self.final_loss
        return None

    @property
    def filename(self) -> str:
        """Get just the filename from dataset_path."""
        return Path(self.dataset_path).name

    @property
    def is_finished(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in ('completed', 'failed', 'skipped', 'stopped')

    @property
    def is_paused(self) -> bool:
        """Check if job is currently paused."""
        return self.status == 'paused'

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == 'processing'

    # ========== String Representation ==========

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [f"Job({self.filename}"]
        parts.append(f"status={self.status}")
        if self.hero_id:
            parts.append(f"hero={self.hero_id}")
        if self.final_loss:
            parts.append(f"loss={self.final_loss:.4f}")
        return " ".join(parts) + ")"


# ========== Compatibility with JobRecord ==========

def job_to_record_dict(job: Job) -> Dict[str, Any]:
    """
    Convert Job to JobRecord-compatible dict.

    For backwards compatibility during migration.
    """
    return {
        'job_id': job.id,
        'file_name': job.filename,
        'file_path': job.dataset_path,
        'priority': job.priority,
        'status': job.status,
        'created_at': job.created_at,
        'started_at': job.started_at,
        'finished_at': job.finished_at,
        'reason': job.last_error,
        'num_examples': job.num_examples,
        'estimated_tokens': job.estimated_tokens,
        'estimated_hours': job.estimated_hours,
        'actual_hours': job.actual_hours,
        'start_step': job.start_step,
        'final_step': job.final_step,
        'final_loss': job.final_loss,
        'final_val_loss': job.final_val_loss,
        'val_train_gap': job.val_train_gap,
        'tokens_per_sec_avg': job.tokens_per_sec_avg,
        'gpu_type': job.gpu_type,
        'gpu_memory_gb': job.gpu_memory_gb,
        'checkpoint_used': job.checkpoint_used,
        'best_checkpoint_after': job.best_checkpoint_after,
        'config_hash': job.config_hash,
        'git_commit': job.git_commit,
        'regression_flag': job.regression_flag,
        'alert_count': job.alert_count,
        'skill_id': job.skill_id,
        'skill_level': job.skill_level,
        'skill_target_accuracy': job.skill_target_accuracy,
        'skill_accuracy': job.skill_accuracy,
        'metadata': job.metadata,
    }


if __name__ == "__main__":
    # Test job lifecycle
    print("Testing Job abstraction...\n")

    # Create job
    job = Job.from_file(
        "/path/to/test_data.jsonl",
        priority="high",
        hero_id="titan-qwen3-4b",
        campaign_id="campaign-001",
    )
    print(f"Created: {job}")
    print(f"  ID: {job.id}")
    print(f"  Status: {job.status}")

    # Start job
    job.start(step=84000, checkpoint="checkpoint-84000", gpu_type="RTX 4090")
    print(f"\nStarted: {job}")
    print(f"  Status: {job.status}")
    print(f"  Attempts: {job.attempts}")

    # Complete job
    job.complete(
        final_step=85000,
        final_loss=0.44,
        final_val_loss=0.52,
        tokens_per_sec_avg=2400.0,
    )
    print(f"\nCompleted: {job}")
    print(f"  Status: {job.status}")
    print(f"  Loss: {job.final_loss}")
    print(f"  Val/Train Gap: {job.val_train_gap}")

    # Serialize
    job_dict = job.to_dict()
    restored = Job.from_dict(job_dict)
    print(f"\nRestored: {restored}")

    print("\nJob abstraction tests passed!")
