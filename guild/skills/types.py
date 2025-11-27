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
class SkillDisplay:
    """Display configuration for Tavern UI."""
    icon: str = "🎯"           # Emoji icon
    color: str = "#8B5CF6"     # Hex color for UI theming
    short_name: str = "SKL"    # 2-4 char abbreviation


@dataclass
class SkillAPI:
    """API server configuration for training data generation."""
    url: str                            # e.g., http://localhost:8080
    source_dir: Optional[str] = None    # Source code location
    start_command: Optional[str] = None # How to start the API
    endpoints: dict[str, str] = field(default_factory=lambda: {
        "health": "GET /health",
        "info": "GET /info",
        "levels": "GET /levels",
        "generate": "POST /generate",
    })


@dataclass
class SkillEval:
    """Evaluation configuration."""
    samples_per_level: int = 5          # Fixed eval samples per level
    endpoint: str = "/eval"             # GET /eval?level=N
    local_cache: str = ""               # Where to cache eval sets
    combinatorial_space: str = "infinite"
    overlap_probability: str = "~0"


@dataclass
class SkillConfig:
    """
    Configuration for a trainable skill/discipline.
    Loaded from configs/skills/{id}.yaml

    This is the SINGLE SOURCE OF TRUTH for skill configuration.
    All skill metadata, API config, display settings, and eval config
    are loaded from YAML and stored here.
    """
    id: str
    name: str
    description: str
    category: SkillCategory

    # Version - must match API's /info version
    version: str = "1.0.0"

    # Level system
    max_level: int = 10

    # Display (for Tavern UI)
    display: SkillDisplay = field(default_factory=SkillDisplay)

    # API server config
    api: Optional[SkillAPI] = None

    # Evaluation config
    eval: SkillEval = field(default_factory=SkillEval)

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

    @property
    def api_url(self) -> Optional[str]:
        """Get API URL (convenience property)."""
        return self.api.url if self.api else None

    @property
    def icon(self) -> str:
        """Get display icon (convenience property)."""
        return self.display.icon

    @property
    def color(self) -> str:
        """Get display color (convenience property)."""
        return self.display.color

    @property
    def short_name(self) -> str:
        """Get short name (convenience property)."""
        return self.display.short_name


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
