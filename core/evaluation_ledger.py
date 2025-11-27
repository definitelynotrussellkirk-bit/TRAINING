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
    problems: List[Dict[str, Any]] = field(default_factory=list)  # Individual results

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
            problems=data.get("problems", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        """Unique key for this evaluation."""
        return f"{self.checkpoint_step}:{self.skill}:{self.level}"


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

    def _ensure_loaded(self):
        """Load ledger from disk if not already loaded."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if self.ledger_file.exists():
                try:
                    with open(self.ledger_file) as f:
                        data = json.load(f)
                    for record_data in data.get("evaluations", []):
                        record = EvalRecord.from_dict(record_data)
                        self._cache[record.key] = record
                    logger.info(f"Loaded {len(self._cache)} evaluation records")
                except Exception as e:
                    logger.error(f"Failed to load evaluation ledger: {e}")

            self._loaded = True

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
    base_dir: Optional[Path] = None,
) -> bool:
    """
    Convenience function to record an evaluation.

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
        problems=problems or [],
    )

    return ledger.record(record)


# Eval Queue for checkpoint saves
_eval_queue: List[Dict[str, Any]] = []
_eval_queue_lock = Lock()


def queue_evaluation(checkpoint_step: int, skill: str, level: int):
    """
    Queue an evaluation to run for a checkpoint.

    Called when a checkpoint is saved. The eval runner will pick this up
    and run the evaluation against the static validation set.
    """
    global _eval_queue

    # Check if already evaluated
    ledger = get_eval_ledger()
    if ledger.has_evaluation(checkpoint_step, skill, level):
        logger.info(f"Checkpoint {checkpoint_step} already evaluated for {skill} L{level}, skipping queue")
        return False

    with _eval_queue_lock:
        # Check if already queued
        for item in _eval_queue:
            if (item["checkpoint_step"] == checkpoint_step and
                item["skill"] == skill and
                item["level"] == level):
                logger.info(f"Evaluation already queued for checkpoint-{checkpoint_step} {skill} L{level}")
                return False

        _eval_queue.append({
            "checkpoint_step": checkpoint_step,
            "skill": skill,
            "level": level,
            "queued_at": datetime.now().isoformat(),
        })

        logger.info(f"Queued evaluation: checkpoint-{checkpoint_step} {skill} L{level}")

        # Also persist queue to disk for daemon restart recovery
        _save_eval_queue()

        return True


def get_pending_evaluations() -> List[Dict[str, Any]]:
    """Get list of pending evaluations."""
    global _eval_queue
    _load_eval_queue()  # Reload from disk
    with _eval_queue_lock:
        return list(_eval_queue)


def pop_evaluation() -> Optional[Dict[str, Any]]:
    """Pop the next evaluation from the queue."""
    global _eval_queue
    _load_eval_queue()

    with _eval_queue_lock:
        if not _eval_queue:
            return None
        item = _eval_queue.pop(0)
        _save_eval_queue()
        return item


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
            "queue": _eval_queue,
            "updated_at": datetime.now().isoformat(),
        }, f, indent=2)


def _load_eval_queue():
    """Load queue from disk."""
    global _eval_queue

    queue_file = _get_queue_file()
    if not queue_file.exists():
        return

    with _eval_queue_lock:
        try:
            with open(queue_file) as f:
                data = json.load(f)
            _eval_queue = data.get("queue", [])
        except Exception as e:
            logger.error(f"Failed to load eval queue: {e}")
