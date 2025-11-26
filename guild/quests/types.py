"""Quest type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from guild.types import generate_id, datetime_to_iso, iso_to_datetime


class QuestDifficulty(Enum):
    """Difficulty tiers."""
    BRONZE = 1
    SILVER = 2
    GOLD = 3
    PLATINUM = 4
    DRAGON = 5

    @classmethod
    def from_level(cls, level: int) -> "QuestDifficulty":
        """Map level (1-10) to difficulty tier."""
        if level <= 2:
            return cls.BRONZE
        elif level <= 4:
            return cls.SILVER
        elif level <= 6:
            return cls.GOLD
        elif level <= 8:
            return cls.PLATINUM
        else:
            return cls.DRAGON


class CombatResult(Enum):
    """Quest attempt outcomes."""
    CRITICAL_HIT = "crit"
    HIT = "hit"
    GLANCING = "glancing"
    MISS = "miss"
    CRITICAL_MISS = "crit_miss"


@dataclass
class QuestTemplate:
    """
    Blueprint for generating quest instances.
    Loaded from configs/quests/{category}/{id}.yaml
    """
    # Required fields
    id: str
    name: str
    description: str
    skills: list[str]
    regions: list[str]
    difficulty: QuestDifficulty
    difficulty_level: int  # 1-10
    generator_id: str
    evaluator_id: str

    # Optional fields with defaults
    generator_params: dict = field(default_factory=dict)
    evaluator_params: dict = field(default_factory=dict)
    base_xp: dict[str, int] = field(default_factory=dict)  # skill_id -> base XP
    tags: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class QuestInstance:
    """
    A concrete quest ready to be attempted.
    Created by QuestForge from a QuestTemplate.
    """
    id: str
    template_id: str

    skills: list[str]
    difficulty: QuestDifficulty
    difficulty_level: int

    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    expected: Optional[dict] = None

    metadata: dict = field(default_factory=dict)
    source: str = ""

    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, template: QuestTemplate, prompt: str,
               expected: Optional[dict] = None,
               context: Optional[dict] = None,
               metadata: Optional[dict] = None) -> "QuestInstance":
        """Factory method to create instance from template."""
        return cls(
            id=generate_id("quest"),
            template_id=template.id,
            skills=template.skills.copy(),
            difficulty=template.difficulty,
            difficulty_level=template.difficulty_level,
            prompt=prompt,
            expected=expected,
            context=context or {},
            metadata=metadata or {},
            source=f"forge:{template.generator_id}",
        )

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "template_id": self.template_id,
            "skills": self.skills,
            "difficulty": self.difficulty.value,
            "difficulty_level": self.difficulty_level,
            "prompt": self.prompt,
            "context": self.context,
            "expected": self.expected,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": datetime_to_iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuestInstance":
        """Deserialize from dict."""
        difficulty = data.get("difficulty")
        if isinstance(difficulty, int):
            difficulty = QuestDifficulty(difficulty)
        elif isinstance(difficulty, str):
            # Handle both "1" and "bronze" formats
            try:
                difficulty = QuestDifficulty(int(difficulty))
            except ValueError:
                difficulty = QuestDifficulty[difficulty.upper()]

        return cls(
            id=data["id"],
            template_id=data["template_id"],
            skills=data["skills"],
            difficulty=difficulty,
            difficulty_level=data["difficulty_level"],
            prompt=data["prompt"],
            context=data.get("context", {}),
            expected=data.get("expected"),
            metadata=data.get("metadata", {}),
            source=data.get("source", ""),
            created_at=iso_to_datetime(data.get("created_at")) or datetime.now(),
        )


@dataclass
class QuestResult:
    """
    Outcome of attempting a quest.
    Created by evaluator after hero attempts quest.
    """
    # Required fields
    quest_id: str
    hero_id: str
    response: str
    combat_result: CombatResult

    # Optional fields with defaults
    response_metadata: dict = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    xp_awarded: dict[str, int] = field(default_factory=dict)
    effects_triggered: list[str] = field(default_factory=list)
    attempted_at: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
    evaluator_notes: str = ""

    @property
    def success(self) -> bool:
        """Whether this counts as a success for accuracy."""
        return self.combat_result in [CombatResult.CRITICAL_HIT, CombatResult.HIT]

    @property
    def total_xp(self) -> int:
        """Total XP awarded across all skills."""
        return sum(self.xp_awarded.values())

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "quest_id": self.quest_id,
            "hero_id": self.hero_id,
            "response": self.response,
            "response_metadata": self.response_metadata,
            "combat_result": self.combat_result.value,
            "metrics": self.metrics,
            "xp_awarded": self.xp_awarded,
            "effects_triggered": self.effects_triggered,
            "attempted_at": datetime_to_iso(self.attempted_at),
            "duration_ms": self.duration_ms,
            "evaluator_notes": self.evaluator_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuestResult":
        """Deserialize from dict."""
        return cls(
            quest_id=data["quest_id"],
            hero_id=data["hero_id"],
            response=data["response"],
            response_metadata=data.get("response_metadata", {}),
            combat_result=CombatResult(data["combat_result"]),
            metrics=data.get("metrics", {}),
            xp_awarded=data.get("xp_awarded", {}),
            effects_triggered=data.get("effects_triggered", []),
            attempted_at=iso_to_datetime(data.get("attempted_at")) or datetime.now(),
            duration_ms=data.get("duration_ms", 0),
            evaluator_notes=data.get("evaluator_notes", ""),
        )
