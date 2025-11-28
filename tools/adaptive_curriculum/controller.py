#!/usr/bin/env python3
"""Difficulty controller that adjusts generator difficulty to maintain target accuracy."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from .stats import GeneratorStats


@dataclass
class DifficultyConfig:
    """Configuration for difficulty levels."""
    min_level: int = 0
    max_level: int = 3
    target_accuracy: float = 0.8
    accuracy_band: float = 0.05  # ±5% tolerance
    min_samples_required: int = 20  # Minimum samples before adjusting


class DifficultyController:
    """Adaptive difficulty controller.

    Maintains each generator at ~80% accuracy by adjusting difficulty:
    - If accuracy > 85%: increase difficulty
    - If accuracy < 75%: decrease difficulty
    - If 75-85%: stay at current level
    """

    def __init__(self, config: Optional[DifficultyConfig] = None):
        """Initialize controller.

        Args:
            config: Difficulty configuration (uses defaults if None)
        """
        self.config = config or DifficultyConfig()

        # Track current difficulty per generator
        self._current_level: Dict[str, int] = defaultdict(lambda: self.config.min_level)

        # Track adjustment history
        self._history: Dict[str, list] = defaultdict(list)

    def choose_difficulty(self, generator_id: str, stats: GeneratorStats) -> int:
        """Choose difficulty level for next generation batch.

        Args:
            generator_id: Generator identifier
            stats: Current statistics for this generator

        Returns:
            Difficulty level (0-max_level)
        """
        current_level = self._current_level[generator_id]
        accuracy = stats.accuracy(current_level)
        sample_count = stats.sample_count(current_level)

        # Not enough data yet? Stay at current level
        if accuracy is None or sample_count < self.config.min_samples_required:
            self._log_decision(
                generator_id, current_level, accuracy, sample_count,
                action="hold", reason="insufficient_data"
            )
            return current_level

        # Calculate bounds
        upper_bound = self.config.target_accuracy + self.config.accuracy_band
        lower_bound = self.config.target_accuracy - self.config.accuracy_band

        new_level = current_level
        action = "hold"
        reason = "in_range"

        # Too easy? Go harder
        if accuracy > upper_bound and current_level < self.config.max_level:
            new_level = current_level + 1
            action = "increase"
            reason = f"accuracy {accuracy:.2%} > {upper_bound:.2%}"

        # Too hard? Go easier
        elif accuracy < lower_bound and current_level > self.config.min_level:
            new_level = current_level - 1
            action = "decrease"
            reason = f"accuracy {accuracy:.2%} < {lower_bound:.2%}"

        # Update current level
        self._current_level[generator_id] = new_level

        # Log decision
        self._log_decision(
            generator_id, current_level, accuracy, sample_count,
            action=action, reason=reason, new_level=new_level
        )

        return new_level

    def get_current_level(self, generator_id: str) -> int:
        """Get current difficulty level for a generator."""
        return self._current_level[generator_id]

    def set_level(self, generator_id: str, level: int) -> None:
        """Manually set difficulty level (e.g., for bootstrapping)."""
        if not (self.config.min_level <= level <= self.config.max_level):
            raise ValueError(
                f"Level {level} outside valid range "
                f"[{self.config.min_level}, {self.config.max_level}]"
            )
        self._current_level[generator_id] = level
        self._log_decision(
            generator_id, level, None, 0,
            action="manual_set", reason="user_override"
        )

    def _log_decision(self, generator_id: str, level: int,
                      accuracy: Optional[float], sample_count: int,
                      action: str, reason: str,
                      new_level: Optional[int] = None) -> None:
        """Log difficulty adjustment decision."""
        from datetime import datetime

        entry = {
            "timestamp": datetime.now().isoformat(),
            "generator_id": generator_id,
            "current_level": level,
            "accuracy": accuracy,
            "sample_count": sample_count,
            "action": action,
            "reason": reason,
            "new_level": new_level or level
        }
        self._history[generator_id].append(entry)

    def get_history(self, generator_id: str, limit: int = 20) -> list:
        """Get recent adjustment history for a generator."""
        return self._history[generator_id][-limit:]

    def summary(self) -> dict:
        """Get summary of all generators' current states."""
        return {
            "current_levels": dict(self._current_level),
            "config": {
                "min_level": self.config.min_level,
                "max_level": self.config.max_level,
                "target_accuracy": self.config.target_accuracy,
                "accuracy_band": self.config.accuracy_band,
                "min_samples_required": self.config.min_samples_required
            },
            "recent_history": {
                gen_id: self.get_history(gen_id, 10)
                for gen_id in self._current_level.keys()
            }
        }


class MixedDifficultyController(DifficultyController):
    """Advanced controller that can choose difficulty mixtures.

    Instead of single difficulty level, can generate batches with
    multiple difficulty levels (e.g., 60% medium, 40% hard).
    """

    def __init__(self, config: Optional[DifficultyConfig] = None):
        super().__init__(config)
        # Track mixture ratios: generator_id -> {difficulty: weight}
        self._mixtures: Dict[str, Dict[int, float]] = {}

    def choose_mixture(self, generator_id: str, stats: GeneratorStats) -> Dict[int, float]:
        """Choose difficulty mixture for next batch.

        Args:
            generator_id: Generator identifier
            stats: Current statistics

        Returns:
            Dict mapping difficulty -> proportion (e.g., {0: 0.6, 1: 0.4})
        """
        current_level = self._current_level[generator_id]
        accuracy = stats.accuracy(current_level)

        # Not enough data? Pure current level
        if accuracy is None or stats.sample_count(current_level) < self.config.min_samples_required:
            return {current_level: 1.0}

        upper_bound = self.config.target_accuracy + self.config.accuracy_band
        lower_bound = self.config.target_accuracy - self.config.accuracy_band

        # In target range? Pure current level
        if lower_bound <= accuracy <= upper_bound:
            return {current_level: 1.0}

        # Too easy? Mix current + harder
        if accuracy > upper_bound and current_level < self.config.max_level:
            # Start introducing harder examples
            overshoot = (accuracy - upper_bound) / (1.0 - upper_bound)
            harder_ratio = min(0.5, overshoot * 0.7)  # Cap at 50% harder
            return {
                current_level: 1.0 - harder_ratio,
                current_level + 1: harder_ratio
            }

        # Too hard? Mix current + easier
        if accuracy < lower_bound and current_level > self.config.min_level:
            # Introduce easier examples
            undershoot = (lower_bound - accuracy) / lower_bound
            easier_ratio = min(0.5, undershoot * 0.7)  # Cap at 50% easier
            return {
                current_level: 1.0 - easier_ratio,
                current_level - 1: easier_ratio
            }

        # Default: pure current level
        return {current_level: 1.0}

    def apply_mixture(self, total_count: int, mixture: Dict[int, float]) -> Dict[int, int]:
        """Convert mixture proportions to actual counts.

        Args:
            total_count: Total number of examples to generate
            mixture: Difficulty proportions

        Returns:
            Dict mapping difficulty -> count
        """
        counts = {}
        remaining = total_count

        # Sort by difficulty to handle rounding consistently
        for difficulty in sorted(mixture.keys()):
            proportion = mixture[difficulty]
            count = int(total_count * proportion)
            counts[difficulty] = count
            remaining -= count

        # Distribute remainder to highest difficulty
        if remaining > 0:
            max_diff = max(mixture.keys())
            counts[max_diff] = counts.get(max_diff, 0) + remaining

        return counts
