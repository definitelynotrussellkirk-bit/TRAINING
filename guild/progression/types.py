"""Progression and status effect type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from guild.types import Severity, datetime_to_iso, iso_to_datetime
from guild.skills.types import SkillState


class EffectType(Enum):
    """Types of status effects."""
    DEBUFF = "debuff"
    BUFF = "buff"
    NEUTRAL = "neutral"


@dataclass
class StatusEffect:
    """A status effect (buff or debuff) affecting the hero."""
    id: str
    name: str
    description: str
    type: EffectType
    severity: Severity

    applied_at_step: int
    applied_at_time: datetime = field(default_factory=datetime.now)
    cause: dict = field(default_factory=dict)

    duration_steps: Optional[int] = None
    cure_condition: Optional[str] = None

    effects: dict[str, Any] = field(default_factory=dict)

    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def is_expired(self, current_step: int) -> bool:
        """Check if effect has expired by step count."""
        if self.duration_steps is None:
            return False
        return (current_step - self.applied_at_step) >= self.duration_steps

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "severity": self.severity.value,
            "applied_at_step": self.applied_at_step,
            "applied_at_time": datetime_to_iso(self.applied_at_time),
            "cause": self.cause,
            "duration_steps": self.duration_steps,
            "cure_condition": self.cure_condition,
            "effects": self.effects,
            "rpg_name": self.rpg_name,
            "rpg_description": self.rpg_description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StatusEffect":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            type=EffectType(data["type"]),
            severity=Severity(data["severity"]),
            applied_at_step=data["applied_at_step"],
            applied_at_time=iso_to_datetime(data.get("applied_at_time")) or datetime.now(),
            cause=data.get("cause", {}),
            duration_steps=data.get("duration_steps"),
            cure_condition=data.get("cure_condition"),
            effects=data.get("effects", {}),
            rpg_name=data.get("rpg_name"),
            rpg_description=data.get("rpg_description"),
        )


@dataclass
class EffectDefinition:
    """Definition of a status effect (loaded from config)."""
    id: str
    name: str
    description: str
    type: EffectType
    severity: Severity

    default_duration_steps: Optional[int] = None
    cure_condition: Optional[str] = None
    effects: dict[str, Any] = field(default_factory=dict)

    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def create_instance(self, step: int, cause: dict = None) -> StatusEffect:
        """Create an instance of this effect."""
        return StatusEffect(
            id=self.id,
            name=self.name,
            description=self.description,
            type=self.type,
            severity=self.severity,
            applied_at_step=step,
            cause=cause or {},
            duration_steps=self.default_duration_steps,
            cure_condition=self.cure_condition,
            effects=self.effects.copy(),
            rpg_name=self.rpg_name,
            rpg_description=self.rpg_description,
        )


@dataclass
class EffectRuleConfig:
    """
    Configuration for a rule that triggers effects.
    Loaded from configs/progression/effects.yaml
    Immutable - runtime state is separate.
    """
    id: str
    effect_id: str

    trigger_type: str  # "metric_threshold", "consecutive_failures", "event"
    trigger_config: dict = field(default_factory=dict)

    cooldown_steps: int = 100
    skill_id: Optional[str] = None


@dataclass
class EffectRuleState:
    """Runtime state for an effect rule (separated from config)."""
    rule_id: str
    last_triggered_step: int = 0
    trigger_count: int = 0


@dataclass
class HeroIdentity:
    """Identity information about a hero (model)."""
    id: str
    name: str

    architecture: str  # "qwen", "llama"
    generation: str    # "3", "2.5"
    size: str          # "0.6B", "7B"
    variant: str       # "base", "instruct"

    checkpoint_path: Optional[str] = None
    checkpoint_step: int = 0

    race: Optional[str] = None
    stature: Optional[str] = None
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "name": self.name,
            "architecture": self.architecture,
            "generation": self.generation,
            "size": self.size,
            "variant": self.variant,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "race": self.race,
            "stature": self.stature,
            "class_name": self.class_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HeroIdentity":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HeroState:
    """Complete state of the hero. Persisted to status/hero_state.json"""
    hero_id: str
    identity: HeroIdentity

    skills: dict[str, SkillState] = field(default_factory=dict)
    active_effects: list[StatusEffect] = field(default_factory=list)

    current_region: str = ""
    current_step: int = 0
    current_run_id: Optional[str] = None

    total_quests: int = 0
    total_xp: float = 0.0
    total_crits: int = 0
    total_misses: int = 0

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def health(self) -> str:
        """Compute health status from effects."""
        if not self.active_effects:
            return "healthy"

        severe = sum(1 for e in self.active_effects
                    if e.severity in [Severity.HIGH, Severity.CRITICAL])
        moderate = sum(1 for e in self.active_effects
                      if e.severity == Severity.MEDIUM)

        if severe >= 2:
            return "struggling"
        elif severe == 1:
            return "wounded"
        elif moderate >= 2:
            return "fatigued"
        else:
            return "minor_issues"

    def get_skill(self, skill_id: str) -> SkillState:
        """Get or create skill state."""
        if skill_id not in self.skills:
            self.skills[skill_id] = SkillState(skill_id=skill_id)
        return self.skills[skill_id]

    def add_effect(self, effect: StatusEffect):
        """Add a status effect (replaces existing with same ID)."""
        self.active_effects = [e for e in self.active_effects if e.id != effect.id]
        self.active_effects.append(effect)

    def remove_effect(self, effect_id: str):
        """Remove a status effect by ID."""
        self.active_effects = [e for e in self.active_effects if e.id != effect_id]

    def clear_expired_effects(self, current_step: int):
        """Remove effects that have expired."""
        self.active_effects = [e for e in self.active_effects
                               if not e.is_expired(current_step)]

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "hero_id": self.hero_id,
            "identity": self.identity.to_dict(),
            "skills": {k: v.to_dict() for k, v in self.skills.items()},
            "active_effects": [e.to_dict() for e in self.active_effects],
            "current_region": self.current_region,
            "current_step": self.current_step,
            "current_run_id": self.current_run_id,
            "total_quests": self.total_quests,
            "total_xp": self.total_xp,
            "total_crits": self.total_crits,
            "total_misses": self.total_misses,
            "created_at": datetime_to_iso(self.created_at),
            "updated_at": datetime_to_iso(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HeroState":
        """Deserialize from dict."""
        identity = HeroIdentity.from_dict(data["identity"])
        skills = {k: SkillState.from_dict(v) for k, v in data.get("skills", {}).items()}
        effects = [StatusEffect.from_dict(e) for e in data.get("active_effects", [])]

        return cls(
            hero_id=data["hero_id"],
            identity=identity,
            skills=skills,
            active_effects=effects,
            current_region=data.get("current_region", ""),
            current_step=data.get("current_step", 0),
            current_run_id=data.get("current_run_id"),
            total_quests=data.get("total_quests", 0),
            total_xp=data.get("total_xp", 0.0),
            total_crits=data.get("total_crits", 0),
            total_misses=data.get("total_misses", 0),
            created_at=iso_to_datetime(data.get("created_at")) or datetime.now(),
            updated_at=iso_to_datetime(data.get("updated_at")) or datetime.now(),
        )
