"""Skill/Discipline type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from guild.types import SerializableMixin


class SkillCategory(Enum):
    """Categories of skills."""
    REASONING = "reasoning"
    COMPRESSION = "compression"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    TOOL_USE = "tool_use"
    INSTRUCTION = "instruction"
    MATH = "math"
    CODE = "code"


@dataclass
class MetricDefinition:
    """Definition of a measurable metric."""
    id: str
    name: str
    description: str = ""
    higher_is_better: bool = True
    range_min: float = 0.0
    range_max: float = 1.0
    format_string: str = "{:.2%}"


@dataclass
class SkillConfig:
    """
    Configuration for a trainable skill/discipline.
    Loaded from configs/skills/{id}.yaml
    """
    id: str
    name: str
    description: str
    category: SkillCategory

    tags: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    primary_metric: str = "accuracy"

    # Level -> required accuracy
    accuracy_thresholds: dict[int, float] = field(default_factory=dict)
    xp_multiplier: float = 1.0

    # RPG flavor
    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def get_threshold(self, level: int) -> float:
        """Get accuracy threshold for a level."""
        if level in self.accuracy_thresholds:
            return self.accuracy_thresholds[level]
        if not self.accuracy_thresholds:
            return 0.6 + (level - 1) * 0.03
        max_defined = max(self.accuracy_thresholds.keys())
        if level > max_defined:
            return self.accuracy_thresholds[max_defined]
        return 0.6


@dataclass
class SkillState(SerializableMixin):
    """
    Runtime state of a skill for a specific hero.
    Persisted to status/hero_state.json
    """
    skill_id: str
    level: int = 1
    xp_total: float = 0.0
    xp_marks: dict[int, float] = field(default_factory=dict)

    # Rolling accuracy - stored as list for JSON serialization
    _recent_results: list[bool] = field(default_factory=list)
    window_size: int = 100

    # Trial state
    eligible_for_trial: bool = False
    last_trial_step: Optional[int] = None
    consecutive_trial_failures: int = 0

    @property
    def recent_results(self) -> list[bool]:
        """Get recent results list."""
        return self._recent_results

    @property
    def accuracy(self) -> float:
        """Calculate current rolling accuracy."""
        if not self._recent_results:
            return 0.0
        return sum(self._recent_results) / len(self._recent_results)

    @property
    def xp_since_last_level(self) -> float:
        """XP earned since last level-up."""
        if self.level not in self.xp_marks:
            return self.xp_total
        return self.xp_total - self.xp_marks.get(self.level, 0)

    def record_result(self, success: bool):
        """Record a quest result for accuracy tracking."""
        self._recent_results.append(success)
        if len(self._recent_results) > self.window_size:
            self._recent_results = self._recent_results[-self.window_size:]

    def record_level_up(self):
        """Record a level-up event."""
        self.level += 1
        self.xp_marks[self.level] = self.xp_total
        self.eligible_for_trial = False
        self.consecutive_trial_failures = 0

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "skill_id": self.skill_id,
            "level": self.level,
            "xp_total": self.xp_total,
            "xp_marks": {str(k): v for k, v in self.xp_marks.items()},
            "recent_results": self._recent_results,
            "window_size": self.window_size,
            "eligible_for_trial": self.eligible_for_trial,
            "last_trial_step": self.last_trial_step,
            "consecutive_trial_failures": self.consecutive_trial_failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillState":
        """Deserialize from dict."""
        state = cls(skill_id=data["skill_id"])
        state.level = data.get("level", 1)
        state.xp_total = data.get("xp_total", 0.0)
        state.xp_marks = {int(k): v for k, v in data.get("xp_marks", {}).items()}
        state._recent_results = data.get("recent_results", [])
        state.window_size = data.get("window_size", 100)
        state.eligible_for_trial = data.get("eligible_for_trial", False)
        state.last_trial_step = data.get("last_trial_step")
        state.consecutive_trial_failures = data.get("consecutive_trial_failures", 0)
        return state
