#!/usr/bin/env python3
"""Generator statistics tracking with rolling window accuracy."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Deque


@dataclass
class EvalResult:
    """Single evaluation result."""
    generator_id: str
    difficulty_level: int
    timestamp: str
    num_examples: int
    correct_count: int
    accuracy: float
    toggles: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratorStats:
    """Rolling window statistics for a single generator.

    Maintains accuracy history per difficulty level with configurable
    window size for recent performance tracking.
    """
    generator_id: str
    target_accuracy: float = 0.8
    window_size: int = 200

    # Per-difficulty tracking: difficulty_level -> deque of 0/1 outcomes
    _history: Dict[int, Deque[int]] = field(default_factory=lambda: defaultdict(lambda: deque()))

    # Eval results log
    _eval_log: List[EvalResult] = field(default_factory=list)

    def update(self, difficulty: int, correct_count: int, total_count: int,
               toggles: Optional[Dict[str, any]] = None) -> None:
        """Update stats with new evaluation results.

        Args:
            difficulty: Difficulty level (0, 1, 2, ...)
            correct_count: Number of correct predictions
            total_count: Total number of examples evaluated
            toggles: Optional dict of generator toggle values
        """
        # Expand into individual 0/1 outcomes
        outcomes = [1] * correct_count + [0] * (total_count - correct_count)

        if difficulty not in self._history:
            self._history[difficulty] = deque()

        self._history[difficulty].extend(outcomes)

        # Maintain sliding window
        if len(self._history[difficulty]) > self.window_size:
            # Keep only most recent window_size outcomes
            excess = len(self._history[difficulty]) - self.window_size
            for _ in range(excess):
                self._history[difficulty].popleft()

        # Log this evaluation
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        result = EvalResult(
            generator_id=self.generator_id,
            difficulty_level=difficulty,
            timestamp=datetime.now().isoformat(),
            num_examples=total_count,
            correct_count=correct_count,
            accuracy=accuracy,
            toggles=toggles or {}
        )
        self._eval_log.append(result)

    def accuracy(self, difficulty: int) -> Optional[float]:
        """Get current rolling accuracy for a difficulty level.

        Returns:
            Accuracy (0.0-1.0) or None if no data available
        """
        if difficulty not in self._history or not self._history[difficulty]:
            return None

        outcomes = list(self._history[difficulty])
        return sum(outcomes) / len(outcomes)

    def sample_count(self, difficulty: int) -> int:
        """Get number of samples in rolling window for difficulty."""
        if difficulty not in self._history:
            return 0
        return len(self._history[difficulty])

    def all_accuracies(self) -> Dict[int, float]:
        """Get accuracies for all difficulty levels with data."""
        return {
            diff: self.accuracy(diff)
            for diff in self._history.keys()
            if self.accuracy(diff) is not None
        }

    def recent_evals(self, limit: int = 10) -> List[EvalResult]:
        """Get N most recent evaluations."""
        return self._eval_log[-limit:]

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "generator_id": self.generator_id,
            "target_accuracy": self.target_accuracy,
            "window_size": self.window_size,
            "current_accuracies": self.all_accuracies(),
            "sample_counts": {
                diff: self.sample_count(diff)
                for diff in self._history.keys()
            },
            "recent_evals": [r.to_dict() for r in self.recent_evals(20)]
        }


class StatsManager:
    """Manages statistics for all generators."""

    def __init__(self, storage_path: Optional[Path] = None,
                 target_accuracy: float = 0.8,
                 window_size: int = 200):
        """Initialize stats manager.

        Args:
            storage_path: Path to persist stats (JSONL format)
            target_accuracy: Target accuracy for all generators
            window_size: Rolling window size for accuracy tracking
        """
        self.storage_path = storage_path
        self.target_accuracy = target_accuracy
        self.window_size = window_size
        self._stats: Dict[str, GeneratorStats] = {}

        if storage_path and storage_path.exists():
            self._load()

    def get_stats(self, generator_id: str) -> GeneratorStats:
        """Get or create stats for a generator."""
        if generator_id not in self._stats:
            self._stats[generator_id] = GeneratorStats(
                generator_id=generator_id,
                target_accuracy=self.target_accuracy,
                window_size=self.window_size
            )
        return self._stats[generator_id]

    def update(self, generator_id: str, difficulty: int,
               correct_count: int, total_count: int,
               toggles: Optional[Dict[str, any]] = None) -> None:
        """Update stats for a generator."""
        stats = self.get_stats(generator_id)
        stats.update(difficulty, correct_count, total_count, toggles)

        if self.storage_path:
            self._save()

    def _save(self) -> None:
        """Persist stats to disk."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as single JSON file with all generator stats
        data = {
            "updated_at": datetime.now().isoformat(),
            "generators": {
                gen_id: stats.to_dict()
                for gen_id, stats in self._stats.items()
            }
        }

        with self.storage_path.open("w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load stats from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with self.storage_path.open("r") as f:
            data = json.load(f)

        # Recreate stats objects (history will be empty, but eval log preserved)
        for gen_id, gen_data in data.get("generators", {}).items():
            stats = GeneratorStats(
                generator_id=gen_id,
                target_accuracy=gen_data.get("target_accuracy", self.target_accuracy),
                window_size=gen_data.get("window_size", self.window_size)
            )

            # Restore recent evals (used for reporting, not for accuracy calc)
            for eval_data in gen_data.get("recent_evals", []):
                result = EvalResult(**eval_data)
                stats._eval_log.append(result)

            self._stats[gen_id] = stats

    def summary(self) -> dict:
        """Get summary of all generator stats."""
        return {
            gen_id: stats.to_dict()
            for gen_id, stats in self._stats.items()
        }
