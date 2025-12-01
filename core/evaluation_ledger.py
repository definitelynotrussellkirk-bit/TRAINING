"""
Evaluation Ledger - Permanent storage for checkpoint evaluation results.

Each checkpoint gets evaluated ONCE against the static validation set for
the current curriculum level. Results are stored forever.

Storage:
- status/evaluation_ledger.json - Master index (append-only)
- current_model/checkpoint-XXXXX/.evals.json - Per-checkpoint sidecar

Usage:
    from core.evaluation_ledger import get_eval_ledger, record_evaluation

    # Record an evaluation
    record_evaluation(
        checkpoint_step=190000,
        skill="bin",
        level=5,
        accuracy=0.8,
        correct=4,
        total=5,
        problems=[...]  # Optional detailed results
    )

    # Query evaluations
    ledger = get_eval_ledger()
    evals = ledger.get_by_checkpoint(190000)
    best = ledger.get_best(skill="bin", level=5)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

# Singleton instance
_eval_ledger: Optional['EvaluationLedger'] = None
_eval_ledger_lock = Lock()


@dataclass
class EvalRecord:
    """Single evaluation result for a checkpoint+skill+level."""
    checkpoint_step: int
    skill: str
    level: int
    accuracy: float  # 0.0 - 1.0
    correct: int
    total: int
    timestamp: str  # ISO format
    validation_type: str = "static"  # "static" (fixed 5 problems) or "dynamic"
    eval_type: str = "quick"  # "quick" (5 problems) or "full" (all levels)
    problems: List[Dict[str, Any]] = field(default_factory=list)  # Individual results

    # === Model Identity Fields (for job system integration) ===
    # These are optional for backwards compatibility with existing records
    hero_id: Optional[str] = None  # e.g., "dio-qwen3-0.6b"
    campaign_id: Optional[str] = None  # e.g., "campaign-001"
    checkpoint_id: Optional[str] = None  # e.g., "checkpoint-175000"
    checkpoint_path: Optional[str] = None  # Full path for model loading
    context_hash: Optional[str] = None  # Hash for drift detection

    # === Job System Fields ===
    job_id: Optional[str] = None  # Job ID from job store
    worker_id: Optional[str] = None  # Worker that executed the eval

    # === Extended Metrics ===
    metrics: Dict[str, Any] = field(default_factory=dict)  # Additional metrics
    started_at: Optional[float] = None  # Unix timestamp
    ended_at: Optional[float] = None  # Unix timestamp
    error: Optional[str] = None  # Error message if failed

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalRecord':
        return cls(
            checkpoint_step=data["checkpoint_step"],
            skill=data["skill"],
            level=data["level"],
            accuracy=data["accuracy"],
            correct=data["correct"],
            total=data["total"],
            timestamp=data["timestamp"],
            validation_type=data.get("validation_type", "static"),
            eval_type=data.get("eval_type", "quick"),
            problems=data.get("problems", []),
            # Model identity fields
            hero_id=data.get("hero_id"),
            campaign_id=data.get("campaign_id"),
            checkpoint_id=data.get("checkpoint_id"),
            checkpoint_path=data.get("checkpoint_path"),
            context_hash=data.get("context_hash"),
            # Job system fields
            job_id=data.get("job_id"),
            worker_id=data.get("worker_id"),
            # Extended metrics
            metrics=data.get("metrics", {}),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            error=data.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        """Unique key for this evaluation."""
        return f"{self.checkpoint_step}:{self.skill}:{self.level}"

    @property
    def identity_key(self) -> str:
        """Full identity key including hero/campaign (for job-based queries)."""
        return f"{self.hero_id or 'unknown'}:{self.campaign_id or 'unknown'}:{self.checkpoint_step}:{self.skill}:{self.level}"

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration from timestamps."""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None

    @property
    def display_summary(self) -> str:
        """Human-readable one-line summary."""
        return f"{self.skill} L{self.level}: {self.accuracy*100:.1f}% ({self.correct}/{self.total})"


@dataclass
class EvalQueueEntry:
    """Entry in the evaluation queue with priority."""
    checkpoint_step: int
    skill: str
    level: Optional[int]  # None means all levels (for FULL eval)
    queued_at: str
    eval_type: str = "quick"  # "quick" or "full"
    priority: int = 10  # 1-10, higher = processed sooner

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalQueueEntry':
        # Normalize checkpoint_step: handle both int and "checkpoint-XXXXX" string
        checkpoint_step_raw = data["checkpoint_step"]
        if isinstance(checkpoint_step_raw, str):
            checkpoint_step = extract_checkpoint_step(checkpoint_step_raw)
        else:
            checkpoint_step = int(checkpoint_step_raw)

        return cls(
            checkpoint_step=checkpoint_step,
            skill=data["skill"],
            level=data.get("level"),
            queued_at=data["queued_at"],
            eval_type=data.get("eval_type", "quick"),
            priority=data.get("priority", 10),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationLedger:
    """
    Append-only ledger for evaluation results.

    Stores evaluations in:
    1. Central index: status/evaluation_ledger.json
    2. Sidecar files: current_model/checkpoint-XXXXX/.evals.json
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.ledger_file = self.base_dir / "status" / "evaluation_ledger.json"
        self.models_dir = self.base_dir / "models" / "current_model"
        self._lock = Lock()
        self._cache: Dict[str, EvalRecord] = {}
        self._loaded = False
        self._last_mtime: float = 0.0  # Track file modification time

    def _ensure_loaded(self):
        """Load ledger from disk if not already loaded or if file has been updated."""
        # Check if file exists and get its modification time
        if self.ledger_file.exists():
            current_mtime = self.ledger_file.stat().st_mtime

            # Reload if file has been modified since last load
            if self._loaded and current_mtime <= self._last_mtime:
                return  # File hasn't changed, use cached data
        elif self._loaded:
            return  # No file exists, use cached data if we have it

        with self._lock:
            # Double-check after acquiring lock
            if self.ledger_file.exists():
                current_mtime = self.ledger_file.stat().st_mtime
                if self._loaded and current_mtime <= self._last_mtime:
                    return

                try:
                    self._cache.clear()  # Clear old data before reloading
                    with open(self.ledger_file) as f:
                        data = json.load(f)
                    for record_data in data.get("evaluations", []):
                        record = EvalRecord.from_dict(record_data)
                        self._cache[record.key] = record
                    self._last_mtime = current_mtime
                    logger.info(f"Loaded {len(self._cache)} evaluation records")
                except Exception as e:
                    logger.error(f"Failed to load evaluation ledger: {e}")

            self._loaded = True

    def reload(self):
        """Force reload ledger from disk (clears cache and reloads)."""
        with self._lock:
            self._loaded = False
            self._cache.clear()
        self._ensure_loaded()

    def _save(self):
        """Save ledger to disk."""
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "count": len(self._cache),
            "evaluations": [r.to_dict() for r in sorted(
                self._cache.values(),
                key=lambda x: (x.checkpoint_step, x.skill, x.level)
            )]
        }

        with open(self.ledger_file, "w") as f:
            json.dump(data, f, indent=2)

    def _save_sidecar(self, record: EvalRecord):
        """Save/update sidecar file for checkpoint using Ledger as source of truth."""
        # Use Checkpoint Ledger to find the path (single source of truth!)
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            checkpoint_record = ledger.get(record.checkpoint_step)

            if not checkpoint_record:
                logger.warning(f"Checkpoint {record.checkpoint_step} not in ledger, skipping sidecar")
                return

            checkpoint_dir = Path(checkpoint_record.path)
            if not checkpoint_dir.exists():
                logger.warning(f"Checkpoint path from ledger doesn't exist: {checkpoint_dir}")
                return

        except ImportError:
            # Fallback to glob if ledger not available
            checkpoint_dirs = list(self.models_dir.glob(f"checkpoint-{record.checkpoint_step}*"))
            if not checkpoint_dirs:
                logger.warning(f"No checkpoint dir found for step {record.checkpoint_step}")
                return
            checkpoint_dir = checkpoint_dirs[0]

        sidecar_file = checkpoint_dir / ".evals.json"

        # Load existing sidecar or create new
        if sidecar_file.exists():
            with open(sidecar_file) as f:
                sidecar_data = json.load(f)
        else:
            sidecar_data = {"evaluations": []}

        # Add/update this evaluation
        evals = sidecar_data["evaluations"]
        # Remove existing eval for same skill+level if any
        evals = [e for e in evals if not (e["skill"] == record.skill and e["level"] == record.level)]
        evals.append(record.to_dict())

        sidecar_data["evaluations"] = evals
        sidecar_data["updated_at"] = datetime.now().isoformat()

        with open(sidecar_file, "w") as f:
            json.dump(sidecar_data, f, indent=2)

        logger.info(f"Updated sidecar: {sidecar_file}")

    def record(self, record: EvalRecord) -> bool:
        """
        Record an evaluation result.

        Returns True if this is a new record, False if already exists.
        """
        self._ensure_loaded()

        with self._lock:
            # Check if already evaluated
            if record.key in self._cache:
                existing = self._cache[record.key]
                logger.info(f"Checkpoint {record.checkpoint_step} already evaluated for {record.skill} L{record.level} "
                           f"(accuracy={existing.accuracy:.1%})")
                return False

            # Store in cache
            self._cache[record.key] = record

            # Save to disk
            self._save()
            self._save_sidecar(record)

            logger.info(f"Recorded evaluation: checkpoint-{record.checkpoint_step} {record.skill} L{record.level} "
                       f"accuracy={record.accuracy:.1%} ({record.correct}/{record.total})")
            return True

    def has_evaluation(self, checkpoint_step: int, skill: str, level: int) -> bool:
        """Check if a checkpoint+skill+level has already been evaluated."""
        self._ensure_loaded()
        key = f"{checkpoint_step}:{skill}:{level}"
        return key in self._cache

    def get(self, checkpoint_step: int, skill: str, level: int) -> Optional[EvalRecord]:
        """Get a specific evaluation."""
        self._ensure_loaded()
        key = f"{checkpoint_step}:{skill}:{level}"
        return self._cache.get(key)

    def get_by_checkpoint(self, checkpoint_step: int) -> List[EvalRecord]:
        """Get all evaluations for a checkpoint."""
        self._ensure_loaded()
        return [r for r in self._cache.values() if r.checkpoint_step == checkpoint_step]

    def get_by_skill(self, skill: str, level: Optional[int] = None) -> List[EvalRecord]:
        """Get all evaluations for a skill (optionally filtered by level)."""
        self._ensure_loaded()
        results = [r for r in self._cache.values() if r.skill == skill]
        if level is not None:
            results = [r for r in results if r.level == level]
        return sorted(results, key=lambda x: x.checkpoint_step)

    def get_best(self, skill: str, level: int, metric: str = "accuracy") -> Optional[EvalRecord]:
        """Get the best checkpoint for a skill+level by metric."""
        evals = self.get_by_skill(skill, level)
        if not evals:
            return None
        return max(evals, key=lambda x: getattr(x, metric, 0))

    def get_latest(self, skill: Optional[str] = None) -> Optional[EvalRecord]:
        """Get the most recent evaluation."""
        self._ensure_loaded()
        records = list(self._cache.values())
        if skill:
            records = [r for r in records if r.skill == skill]
        if not records:
            return None
        return max(records, key=lambda x: x.timestamp)

    def list_all(self, limit: int = 100) -> List[EvalRecord]:
        """List all evaluations, newest first."""
        self._ensure_loaded()
        records = sorted(self._cache.values(), key=lambda x: x.timestamp, reverse=True)
        return records[:limit]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        self._ensure_loaded()

        by_skill: Dict[str, Dict[str, Any]] = {}
        for record in self._cache.values():
            if record.skill not in by_skill:
                by_skill[record.skill] = {
                    "count": 0,
                    "levels_evaluated": set(),
                    "best_accuracy": 0,
                    "best_checkpoint": None,
                }
            s = by_skill[record.skill]
            s["count"] += 1
            s["levels_evaluated"].add(record.level)
            if record.accuracy > s["best_accuracy"]:
                s["best_accuracy"] = record.accuracy
                s["best_checkpoint"] = record.checkpoint_step

        # Convert sets to lists for JSON
        for skill in by_skill:
            by_skill[skill]["levels_evaluated"] = sorted(by_skill[skill]["levels_evaluated"])

        return {
            "total_evaluations": len(self._cache),
            "by_skill": by_skill,
            "latest": self.get_latest().to_dict() if self.get_latest() else None,
        }

    # =========================================================================
    # CANONICAL INTERFACE - Use these for stable, documented access
    # =========================================================================

    def get_eval_count(
        self,
        skill: Optional[str] = None,
        level: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """
        Get canonical evaluation count.

        This is THE single source of truth for eval counts.
        All UI elements should use this method.

        Definition:
            eval_count = number of unique (checkpoint_step, skill, level) tuples
            successfully evaluated and recorded in the ledger.

        Args:
            skill: Filter by skill ID (e.g., "bin", "sy")
            level: Filter by level (requires skill)
            run_id: Filter by campaign run_id (not yet implemented)

        Returns:
            Integer count of matching evaluations
        """
        self._ensure_loaded()

        records = list(self._cache.values())

        if skill is not None:
            records = [r for r in records if r.skill == skill]
        if level is not None:
            records = [r for r in records if r.level == level]
        # run_id filtering would go here when implemented

        return len(records)

    def get_eval_breakdown(
        self,
        run_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Get evaluation count breakdown by skill.

        Args:
            run_id: Filter by campaign run_id (not yet implemented)

        Returns:
            Dict mapping skill_id to count
        """
        self._ensure_loaded()

        breakdown: Dict[str, int] = {}
        for record in self._cache.values():
            breakdown[record.skill] = breakdown.get(record.skill, 0) + 1

        return breakdown

    def get_unique_key(self, checkpoint_step: int, skill: str, level: int) -> str:
        """
        Get the unique key for an evaluation.

        This is the canonical uniqueness definition:
            key = f"{checkpoint_step}:{skill}:{level}"

        Two evaluations with the same key are considered the same evaluation.
        """
        return f"{checkpoint_step}:{skill}:{level}"

    def is_idempotent_write(self, checkpoint_step: int, skill: str, level: int) -> bool:
        """
        Check if writing this evaluation would be idempotent.

        Returns True if the evaluation already exists (write would be a no-op).
        This is useful for verifying ledger consistency.
        """
        return self.has_evaluation(checkpoint_step, skill, level)

    # =========================================================================
    # JOB SYSTEM INTEGRATION - Query by job/hero/campaign identity
    # =========================================================================

    def get_by_job(self, job_id: str) -> Optional[EvalRecord]:
        """Get evaluation result for a specific job ID."""
        self._ensure_loaded()
        for record in self._cache.values():
            if record.job_id == job_id:
                return record
        return None

    def get_by_hero_campaign(
        self,
        hero_id: str,
        campaign_id: Optional[str] = None,
        skill: Optional[str] = None,
        level: Optional[int] = None,
    ) -> List[EvalRecord]:
        """
        Get all evaluations for a hero (optionally filtered by campaign/skill/level).

        Useful for campaign-scoped views and analytics.
        """
        self._ensure_loaded()
        results = []
        for record in self._cache.values():
            if record.hero_id != hero_id:
                continue
            if campaign_id and record.campaign_id != campaign_id:
                continue
            if skill and record.skill != skill:
                continue
            if level is not None and record.level != level:
                continue
            results.append(record)
        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def get_checkpoint_skills(
        self,
        checkpoint_step: int,
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> Dict[str, EvalRecord]:
        """
        Get latest eval for each skill at a checkpoint.

        Returns dict mapping "skill:level" to EvalRecord.
        Useful for displaying skill cards on checkpoint detail page.
        """
        evals = self.get_by_checkpoint(checkpoint_step)

        # Filter by hero/campaign if specified
        if hero_id:
            evals = [e for e in evals if e.hero_id == hero_id]
        if campaign_id:
            evals = [e for e in evals if e.campaign_id == campaign_id]

        # Group by skill:level, keep latest
        result: Dict[str, EvalRecord] = {}
        for record in sorted(evals, key=lambda x: x.timestamp, reverse=True):
            key = f"{record.skill}:{record.level}"
            if key not in result:
                result[key] = record

        return result

    def get_skill_trend(
        self,
        skill: str,
        level: int,
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[EvalRecord]:
        """
        Get eval history for a skill/level to show trend.

        Returns list of records ordered by timestamp (oldest first).
        """
        evals = self.get_by_skill(skill, level)

        if hero_id:
            evals = [e for e in evals if e.hero_id == hero_id]
        if campaign_id:
            evals = [e for e in evals if e.campaign_id == campaign_id]

        # Sort by timestamp oldest first (for trend chart)
        evals = sorted(evals, key=lambda x: x.timestamp)
        return evals[-limit:]  # Return most recent N

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    def delete_for_checkpoint(self, checkpoint_step: int) -> int:
        """
        Delete all evaluation records for a given checkpoint.

        Returns the number of records removed.
        """
        self._ensure_loaded()

        with self._lock:
            before = len(self._cache)
            keys_to_remove = [
                key for key, record in self._cache.items()
                if record.checkpoint_step == checkpoint_step
            ]

            for key in keys_to_remove:
                del self._cache[key]

            removed = before - len(self._cache)

            if removed > 0:
                self._save()
                self._update_checkpoint_sidecar(checkpoint_step)
                logger.info(f"Deleted {removed} eval records for checkpoint {checkpoint_step}")

            return removed

    def delete_for_checkpoint_and_skill(
        self,
        checkpoint_step: int,
        skill: Optional[str] = None,
        level: Optional[int] = None,
    ) -> int:
        """
        Delete evaluation records for a given checkpoint, optionally
        filtered by skill and/or level.

        Returns the number of records removed.
        """
        self._ensure_loaded()

        with self._lock:
            before = len(self._cache)

            keys_to_remove = []
            for key, record in self._cache.items():
                if record.checkpoint_step != checkpoint_step:
                    continue
                if skill is not None and record.skill != skill:
                    continue
                if level is not None and record.level != level:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

            removed = before - len(self._cache)

            if removed > 0:
                self._save()
                self._update_checkpoint_sidecar(checkpoint_step)
                logger.info(
                    f"Deleted {removed} eval records for checkpoint {checkpoint_step} "
                    f"(skill={skill}, level={level})"
                )

            return removed

    def _update_checkpoint_sidecar(self, checkpoint_step: int):
        """Update the sidecar file for a checkpoint after deletions."""
        try:
            from core.checkpoint_ledger import get_ledger
            ledger = get_ledger()
            checkpoint_record = ledger.get(checkpoint_step)

            if not checkpoint_record:
                return

            checkpoint_dir = Path(checkpoint_record.path)
            if not checkpoint_dir.exists():
                return

            sidecar_file = checkpoint_dir / ".evals.json"

            # Get remaining evals for this checkpoint
            remaining = self.get_by_checkpoint(checkpoint_step)

            if remaining:
                sidecar_data = {
                    "evaluations": [r.to_dict() for r in remaining],
                    "updated_at": datetime.now().isoformat(),
                }
                with open(sidecar_file, "w") as f:
                    json.dump(sidecar_data, f, indent=2)
            else:
                # No evals left - remove sidecar
                if sidecar_file.exists():
                    sidecar_file.unlink()

        except Exception as e:
            logger.warning(f"Failed to update checkpoint sidecar: {e}")


def get_eval_ledger(base_dir: Optional[Path] = None) -> EvaluationLedger:
    """Get the singleton evaluation ledger instance."""
    global _eval_ledger

    with _eval_ledger_lock:
        if _eval_ledger is None:
            _eval_ledger = EvaluationLedger(base_dir)
        return _eval_ledger


def record_evaluation(
    checkpoint_step: int,
    skill: str,
    level: int,
    accuracy: float,
    correct: int,
    total: int,
    problems: Optional[List[Dict]] = None,
    validation_type: str = "static",
    eval_type: str = "quick",
    base_dir: Optional[Path] = None,
) -> bool:
    """
    Convenience function to record an evaluation.

    Args:
        checkpoint_step: The checkpoint step
        skill: Skill ID
        level: Skill level
        accuracy: Accuracy (0.0 - 1.0)
        correct: Number correct
        total: Total problems
        problems: Optional detailed results
        validation_type: "static" or "dynamic"
        eval_type: "quick" or "full"
        base_dir: Base directory (optional)

    Returns True if recorded, False if already exists.
    """
    ledger = get_eval_ledger(base_dir)

    record = EvalRecord(
        checkpoint_step=checkpoint_step,
        skill=skill,
        level=level,
        accuracy=accuracy,
        correct=correct,
        total=total,
        timestamp=datetime.now().isoformat(),
        validation_type=validation_type,
        eval_type=eval_type,
        problems=problems or [],
    )

    recorded = ledger.record(record)

    # Update campaign peak metrics if recorded successfully
    if recorded:
        _update_campaign_peak_accuracy(accuracy)

        # Also record to curriculum manager for progression tracking
        try:
            from data_manager.curriculum_manager import CurriculumManager
            from core.paths import get_base_dir as paths_get_base_dir

            cm = CurriculumManager(paths_get_base_dir(), {})
            # Map skill names (binary -> bin, syllo -> sy)
            skill_map = {"binary": "bin", "bin": "bin", "syllo": "sy", "sy": "sy"}
            skill_id = skill_map.get(skill, skill)

            cm.record_accuracy(
                skill=skill_id,
                accuracy=accuracy,
                step=checkpoint_step,
                metadata={"level": level, "problems": total, "correct": correct}
            )

            # Check and trigger progression if ready
            should_progress, reason = cm.should_progress(skill_id)
            if should_progress:
                result = cm.progress_to_next_level(skill_id)
                logger.info(f"🎉 [{skill_id}] Progressed to level {result['new_level']}! {reason}")

        except Exception as e:
            logger.warning(f"Failed to sync eval to curriculum: {e}")

    return recorded


def _update_campaign_peak_accuracy(accuracy: float):
    """
    Update campaign peak accuracy metric if this is a new personal best.

    Safe to call - never throws, just logs warnings.
    """
    try:
        from guild.campaigns.loader import load_active_campaign

        campaign = load_active_campaign()
        if campaign is None:
            return

        is_new_peak = campaign.update_peak_metric(
            "highest_accuracy", accuracy, lower_is_better=False
        )
        if is_new_peak:
            logger.info(
                f"[Campaign {campaign.id}] New best accuracy: {accuracy:.3f}"
            )
    except Exception as e:
        # Don't crash evaluation because peak tracking failed
        logger.warning(f"Failed to update peak accuracy: {e}")


def record_job_eval(
    job_id: str,
    skill_id: str,
    level: int,
    accuracy: float,
    correct: int,
    total: int,
    # Model identity
    hero_id: str,
    campaign_id: str,
    checkpoint_step: int,
    checkpoint_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    context_hash: Optional[str] = None,
    # Job info
    worker_id: Optional[str] = None,
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    # Additional metrics
    per_primitive: Optional[Dict[str, float]] = None,
    problems: Optional[List[Dict]] = None,
    error: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> bool:
    """
    Record an evaluation result from a job worker.

    This is the primary entry point for eval_worker to write results.
    Includes full model identity for job system integration.

    Args:
        job_id: Job ID from job store
        skill_id: Skill that was evaluated
        level: Skill level
        accuracy: Overall accuracy (0.0 - 1.0)
        correct: Number correct
        total: Total problems
        hero_id: Hero identifier
        campaign_id: Campaign identifier
        checkpoint_step: Step number
        checkpoint_id: Checkpoint ID string (e.g., "checkpoint-175000")
        checkpoint_path: Full path to checkpoint
        context_hash: Hash for drift detection
        worker_id: Worker that executed the eval
        started_at: Unix timestamp when eval started
        ended_at: Unix timestamp when eval ended
        per_primitive: Per-primitive accuracy breakdown
        problems: Individual problem results
        error: Error message if failed

    Returns:
        True if recorded, False if already exists
    """
    ledger = get_eval_ledger(base_dir)

    # Build metrics dict
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    if per_primitive:
        metrics["per_primitive"] = per_primitive

    record = EvalRecord(
        checkpoint_step=checkpoint_step,
        skill=skill_id,
        level=level,
        accuracy=accuracy,
        correct=correct,
        total=total,
        timestamp=datetime.now().isoformat(),
        validation_type="dynamic",  # Job-based evals are dynamic
        eval_type="job",  # Mark as job-originated
        problems=problems or [],
        # Model identity
        hero_id=hero_id,
        campaign_id=campaign_id,
        checkpoint_id=checkpoint_id or f"checkpoint-{checkpoint_step}",
        checkpoint_path=checkpoint_path,
        context_hash=context_hash,
        # Job system
        job_id=job_id,
        worker_id=worker_id,
        # Extended metrics
        metrics=metrics,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
    )

    recorded = ledger.record(record)

    # Update campaign peak metrics if recorded successfully
    if recorded and not error:
        _update_campaign_peak_accuracy(accuracy)

    return recorded


def extract_checkpoint_step(checkpoint_path: str) -> int:
    """
    Extract step number from checkpoint path.

    Examples:
        "checkpoint-175000" -> 175000
        "campaigns/dio/campaign-001/checkpoints/checkpoint-175000-20251129-1430" -> 175000
    """
    import re
    # Find checkpoint-XXXXX pattern
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0


# Eval Queue for checkpoint saves
_eval_queue: List[EvalQueueEntry] = []
_eval_queue_lock = Lock()


def queue_evaluation(
    checkpoint_step: int,
    skill: str,
    level: Optional[int] = None,
    eval_type: str = "quick",
    priority: int = 10,
) -> bool:
    """
    Queue an evaluation to run for a checkpoint.

    Args:
        checkpoint_step: The checkpoint step to evaluate
        skill: Skill ID (e.g., "bin", "sy")
        level: Level to evaluate (None = all levels for FULL eval)
        eval_type: "quick" (5 problems at level) or "full" (all levels)
        priority: 1-10, higher = processed sooner

    Called when a checkpoint is saved. The eval runner will pick this up
    and run the evaluation against the static validation set.
    """
    global _eval_queue

    # For quick evals, check if already evaluated
    if eval_type == "quick" and level is not None:
        ledger = get_eval_ledger()
        if ledger.has_evaluation(checkpoint_step, skill, level):
            logger.info(f"Checkpoint {checkpoint_step} already evaluated for {skill} L{level}, skipping queue")
            return False

    with _eval_queue_lock:
        _load_eval_queue_locked()

        # Check if already queued (same checkpoint, skill, level, type)
        for entry in _eval_queue:
            if (entry.checkpoint_step == checkpoint_step and
                entry.skill == skill and
                entry.level == level and
                entry.eval_type == eval_type):
                logger.info(f"Evaluation already queued for checkpoint-{checkpoint_step} {skill} L{level} ({eval_type})")
                return False

        entry = EvalQueueEntry(
            checkpoint_step=checkpoint_step,
            skill=skill,
            level=level,
            queued_at=datetime.now().isoformat(),
            eval_type=eval_type,
            priority=priority,
        )
        _eval_queue.append(entry)

        level_str = f"L{level}" if level else "ALL"
        logger.info(f"Queued {eval_type} evaluation: checkpoint-{checkpoint_step} {skill} {level_str} (priority={priority})")

        # Persist queue to disk
        _save_eval_queue()

        # Verbose logging
        try:
            from core.verbose_logger import VerboseLogger
            task_id = f"eval-{checkpoint_step}-{skill}-{level or 'all'}"
            VerboseLogger.task_queued("eval", task_id, {
                "checkpoint_step": checkpoint_step,
                "skill": skill,
                "level": level,
                "eval_type": eval_type,
                "priority": priority
            })
        except:
            pass  # Don't fail if verbose logging is unavailable

        return True


def queue_full_evaluation(checkpoint_step: int, skill: str, priority: int = 6) -> bool:
    """Queue a FULL evaluation (all levels) for a checkpoint."""
    return queue_evaluation(
        checkpoint_step=checkpoint_step,
        skill=skill,
        level=None,  # All levels
        eval_type="full",
        priority=priority,
    )


def get_pending_evaluations(
    sorted_by_priority: bool = True,
    eval_type: Optional[str] = None,
) -> List[EvalQueueEntry]:
    """
    Get list of pending evaluations.

    Args:
        sorted_by_priority: If True, sort by priority DESC, checkpoint DESC, quick before full
        eval_type: Filter by eval type ("quick" or "full")

    Returns:
        List of EvalQueueEntry objects
    """
    global _eval_queue
    _load_eval_queue()

    with _eval_queue_lock:
        entries = list(_eval_queue)

    # Filter by type if specified
    if eval_type:
        entries = [e for e in entries if e.eval_type == eval_type]

    # Sort by priority
    if sorted_by_priority:
        entries.sort(key=lambda e: (
            -e.priority,              # Higher priority first
            -e.checkpoint_step,       # Newer checkpoints first
            e.eval_type == "full",    # Quick before full (False < True)
            e.queued_at,              # Older queued items first
        ))

    return entries


def pop_evaluation(eval_type: Optional[str] = None) -> Optional[EvalQueueEntry]:
    """
    Pop the highest-priority evaluation from the queue.

    Args:
        eval_type: Only pop items of this type ("quick" or "full")

    Returns:
        EvalQueueEntry or None if queue is empty
    """
    global _eval_queue
    _load_eval_queue()

    with _eval_queue_lock:
        if not _eval_queue:
            return None

        # Sort queue by priority
        _eval_queue.sort(key=lambda e: (
            -e.priority,
            -e.checkpoint_step,
            e.eval_type == "full",
            e.queued_at,
        ))

        # Find matching item
        for i, entry in enumerate(_eval_queue):
            if eval_type is None or entry.eval_type == eval_type:
                _eval_queue.pop(i)
                _save_eval_queue()

                # Verbose logging - task being picked up from queue
                try:
                    from core.verbose_logger import VerboseLogger
                    task_id = f"eval-{entry.checkpoint_step}-{entry.skill}-{entry.level or 'all'}"
                    VerboseLogger.task_progress("eval", task_id, "Dequeued for execution", {
                        "priority": entry.priority,
                        "eval_type": entry.eval_type
                    })
                except:
                    pass

                return entry

        return None


def _get_queue_file() -> Path:
    """Get path to queue file."""
    base_dir = Path(__file__).parent.parent
    return base_dir / "status" / "eval_queue.json"


def _save_eval_queue():
    """Save queue to disk."""
    queue_file = _get_queue_file()
    queue_file.parent.mkdir(parents=True, exist_ok=True)

    with open(queue_file, "w") as f:
        json.dump({
            "queue": [e.to_dict() for e in _eval_queue],
            "updated_at": datetime.now().isoformat(),
        }, f, indent=2)


def _load_eval_queue():
    """Load queue from disk (acquires lock)."""
    with _eval_queue_lock:
        _load_eval_queue_locked()


def _load_eval_queue_locked():
    """Load queue from disk (must hold lock)."""
    global _eval_queue

    queue_file = _get_queue_file()
    if not queue_file.exists():
        return

    try:
        with open(queue_file) as f:
            data = json.load(f)
        queue_data = data.get("queue", [])

        # Convert to EvalQueueEntry objects
        _eval_queue = []
        for item in queue_data:
            if isinstance(item, dict):
                _eval_queue.append(EvalQueueEntry.from_dict(item))
    except Exception as e:
        logger.error(f"Failed to load eval queue: {e}")
