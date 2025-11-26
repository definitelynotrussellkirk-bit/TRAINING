# Guild Implementation Plan

## Complete Step-by-Step Implementation Guide

**Created:** 2025-11-26
**Purpose:** Detailed implementation plan to transform TRAINING into the Guild framework
**Principle:** Fix bugs as we go, don't reproduce them. Each step must be testable.

---

# Table of Contents

1. [Pre-Implementation Setup](#phase-0-pre-implementation-setup)
2. [Phase 1: Foundation Types](#phase-1-foundation-types)
3. [Phase 2: Configuration System](#phase-2-configuration-system)
4. [Phase 3: Facilities & Paths](#phase-3-facilities--paths)
5. [Phase 4: Skills Registry](#phase-4-skills-registry)
6. [Phase 5: Quest System](#phase-5-quest-system)
7. [Phase 6: Progression Engine](#phase-6-progression-engine)
8. [Phase 7: Combat Calculator](#phase-7-combat-calculator)
9. [Phase 8: Incidents System](#phase-8-incidents-system)
10. [Phase 9: Runs System](#phase-9-runs-system)
11. [Phase 10: View Layer](#phase-10-view-layer)
12. [Phase 11: Integration & Migration](#phase-11-integration--migration)
13. [Phase 12: Open Source Prep](#phase-12-open-source-prep)

---

# Phase 0: Pre-Implementation Setup

## 0.1 Create Branch

```bash
git checkout -b feature/guild-refactor
```

## 0.2 Create Directory Structure

```bash
mkdir -p guild/{skills,quests,progression,facilities,runs,incidents,hero,combat}
mkdir -p configs/{skills,quests/syllo,facilities,progression,regions,heroes,runs,incidents}
mkdir -p views/{technical,tavern}
mkdir -p tests/guild
```

## 0.3 Update .gitignore

```bash
cat >> .gitignore << 'EOF'

# Guild local configs (user-specific)
configs/facilities/local.yaml
configs/**/*.local.yaml
*.local.yaml

# Environment
.env
.env.local
EOF
```

## 0.4 Add to pyproject.toml

```toml
[project.optional-dependencies]
guild = [
    "pyyaml>=6.0",
    "jsonschema>=4.0",
]
```

## Checkpoint 0
- [ ] Branch created
- [ ] Directories exist
- [ ] .gitignore updated
- [ ] No existing tests broken

---

# Phase 1: Foundation Types

## Goal
Create all dataclasses with no behavior. This is pure type definitions.

## 1.1 Create guild/__init__.py

```python
# guild/__init__.py
"""
Guild Trainer - A generic framework for LLM training with RPG-style progression.

The Guild framework provides:
- Skills: Trainable capabilities with metrics and progression
- Quests: Task instances with templates and evaluation
- Progression: XP, levels, and status effects
- Facilities: Hardware abstraction for multi-machine setups
- Runs: Unified training/eval/audit execution
- Incidents: Structured error tracking
- Combat: Result calculation (CRIT/HIT/MISS)
"""

__version__ = "0.1.0"
__all__ = []
```

## 1.2 Create guild/types.py

```python
# guild/types.py
"""Common types used across the guild module."""

from enum import Enum
from typing import TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime

# Generic type for registries
T = TypeVar('T')


class Severity(Enum):
    """Universal severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(Enum):
    """Universal status values."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Timestamped:
    """Mixin for objects with timestamps."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Identifiable:
    """Mixin for objects with IDs and names."""
    id: str
    name: str
    description: str = ""


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID."""
    from uuid import uuid4
    uid = str(uuid4())[:8]
    return f"{prefix}_{uid}" if prefix else uid
```

## 1.3 Create guild/skills/types.py

```python
# guild/skills/types.py
"""Skill/Discipline type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import deque


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
    format_string: str = "{:.2%}"  # How to display


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

    # Classification
    tags: list[str] = field(default_factory=list)

    # Metrics
    metrics: list[str] = field(default_factory=list)  # metric IDs
    primary_metric: str = "accuracy"

    # Progression thresholds: level -> required accuracy
    accuracy_thresholds: dict[int, float] = field(default_factory=dict)

    # XP multipliers for this skill
    xp_multiplier: float = 1.0

    # RPG flavor (optional)
    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def get_threshold(self, level: int) -> float:
        """Get accuracy threshold for a level."""
        if level in self.accuracy_thresholds:
            return self.accuracy_thresholds[level]
        # Default: interpolate or use max
        if not self.accuracy_thresholds:
            return 0.6 + (level - 1) * 0.03  # Default curve
        max_defined = max(self.accuracy_thresholds.keys())
        if level > max_defined:
            return self.accuracy_thresholds[max_defined]
        return 0.6


@dataclass
class SkillState:
    """
    Current state of a skill for a specific hero.

    Persisted in status/hero_state.json
    """
    skill_id: str
    level: int = 1
    xp_total: float = 0.0

    # XP at each level-up (for cost calculation)
    xp_marks: dict[int, float] = field(default_factory=dict)

    # Rolling accuracy window
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))

    # Trial state
    eligible_for_trial: bool = False
    last_trial_step: Optional[int] = None
    consecutive_trial_failures: int = 0

    @property
    def accuracy(self) -> float:
        """Current rolling accuracy."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def xp_since_last_level(self) -> float:
        """XP earned since last level-up."""
        if self.level not in self.xp_marks:
            return self.xp_total
        return self.xp_total - self.xp_marks.get(self.level, 0)

    def record_result(self, success: bool):
        """Record a quest result."""
        self.recent_results.append(success)

    def record_level_up(self):
        """Record a level-up event."""
        self.level += 1
        self.xp_marks[self.level] = self.xp_total
        self.eligible_for_trial = False
        self.consecutive_trial_failures = 0
```

## 1.4 Create guild/quests/types.py

```python
# guild/quests/types.py
"""Quest type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from guild.types import generate_id


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


@dataclass
class QuestTemplate:
    """
    Blueprint for generating quest instances.

    Loaded from configs/quests/{category}/{id}.yaml
    """
    id: str
    name: str
    description: str

    # Classification
    skills: list[str]  # skill IDs this quest trains
    regions: list[str]  # region IDs where this appears
    difficulty: QuestDifficulty
    difficulty_level: int  # 1-10 granular

    # Generation
    generator_id: str  # registered generator function
    generator_params: dict = field(default_factory=dict)

    # Evaluation
    evaluator_id: str  # registered evaluator function
    evaluator_params: dict = field(default_factory=dict)

    # Rewards (skill_id -> base XP)
    base_xp: dict[str, int] = field(default_factory=dict)

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    # Active flag
    enabled: bool = True


@dataclass
class QuestInstance:
    """
    A concrete quest ready to be attempted.

    Created by QuestForge from a QuestTemplate.
    """
    id: str
    template_id: str

    # Inherited from template
    skills: list[str]
    difficulty: QuestDifficulty
    difficulty_level: int

    # Content
    prompt: str  # The actual prompt text
    context: dict[str, Any] = field(default_factory=dict)  # Additional context
    expected: Optional[dict] = None  # Golden answer if known

    # Metadata
    metadata: dict = field(default_factory=dict)
    source: str = ""  # Lineage info

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, template: QuestTemplate, prompt: str,
               expected: Optional[dict] = None, **kwargs) -> "QuestInstance":
        """Factory method to create instance from template."""
        return cls(
            id=generate_id("quest"),
            template_id=template.id,
            skills=template.skills.copy(),
            difficulty=template.difficulty,
            difficulty_level=template.difficulty_level,
            prompt=prompt,
            expected=expected,
            source=f"forge:{template.generator_id}",
            **kwargs
        )


class CombatResult(Enum):
    """Quest attempt outcomes."""
    CRITICAL_HIT = "crit"      # Perfect
    HIT = "hit"                # Correct
    GLANCING = "glancing"      # Partial
    MISS = "miss"              # Wrong
    CRITICAL_MISS = "crit_miss"  # Invalid/broken


@dataclass
class QuestResult:
    """
    Outcome of attempting a quest.

    Created by evaluator after hero attempts quest.
    """
    quest_id: str
    hero_id: str

    # Response
    response: str
    response_metadata: dict = field(default_factory=dict)

    # Evaluation
    combat_result: CombatResult
    metrics: dict[str, float] = field(default_factory=dict)

    # XP awarded (skill_id -> XP)
    xp_awarded: dict[str, int] = field(default_factory=dict)

    # Effects triggered
    effects_triggered: list[str] = field(default_factory=list)

    # Timing
    attempted_at: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0

    # Debug info
    evaluator_notes: str = ""

    @property
    def success(self) -> bool:
        """Whether this counts as a success for accuracy."""
        return self.combat_result in [CombatResult.CRITICAL_HIT, CombatResult.HIT]

    @property
    def total_xp(self) -> int:
        """Total XP awarded across all skills."""
        return sum(self.xp_awarded.values())
```

## 1.5 Create guild/facilities/types.py

```python
# guild/facilities/types.py
"""Facility (hardware) type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class FacilityType(Enum):
    """Types of facilities."""
    HUB = "hub"              # Central management (Inn)
    BATTLEFIELD = "battlefield"  # Training GPU (Arena)
    ARCHIVE = "archive"      # Long-term storage (Vault)
    OUTPOST = "outpost"      # Satellite workers (Scouts)
    LABORATORY = "laboratory"  # Experiments (Wizard's Study)


@dataclass
class FacilityResource:
    """A specific resource within a facility."""
    id: str
    type: str  # "gpu", "storage", "network"
    properties: dict[str, Any] = field(default_factory=dict)

    # GPU example: {"vram_gb": 24, "model": "RTX 4090"}
    # Storage example: {"capacity_tb": 4, "type": "nvme"}


@dataclass
class Facility:
    """
    A hardware location in the system.

    Loaded from configs/facilities/{local,example}.yaml
    """
    id: str
    name: str
    type: FacilityType
    description: str = ""

    # Connection
    host: str = "localhost"
    port: Optional[int] = None

    # Paths
    base_path: str = ""
    paths: dict[str, str] = field(default_factory=dict)
    # {"checkpoints": "current_model/", "logs": "logs/"}

    # Resources
    resources: list[FacilityResource] = field(default_factory=list)

    # Status
    is_local: bool = True
    is_available: bool = True

    # RPG names
    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def get_path(self, key: str, subpath: str = "") -> str:
        """Get a resolved path within this facility."""
        import os
        base = os.path.expandvars(os.path.expanduser(self.base_path))
        if key in self.paths:
            path = os.path.join(base, self.paths[key])
        else:
            path = os.path.join(base, key)
        if subpath:
            path = os.path.join(path, subpath)
        return path
```

## 1.6 Create guild/progression/types.py

```python
# guild/progression/types.py
"""Progression and status effect type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from guild.types import Severity
from guild.skills.types import SkillState


class EffectType(Enum):
    """Types of status effects."""
    DEBUFF = "debuff"
    BUFF = "buff"
    NEUTRAL = "neutral"


@dataclass
class StatusEffect:
    """
    A status effect (buff or debuff) affecting the hero.
    """
    id: str
    name: str
    description: str
    type: EffectType
    severity: Severity

    # When applied
    applied_at_step: int
    applied_at_time: datetime = field(default_factory=datetime.now)
    cause: dict = field(default_factory=dict)

    # Duration
    duration_steps: Optional[int] = None  # None = until cured
    cure_condition: Optional[str] = None

    # Mechanical effects
    effects: dict[str, Any] = field(default_factory=dict)
    # {"accuracy_penalty": -0.1, "xp_multiplier": 0.8}

    # RPG flavor
    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def is_expired(self, current_step: int) -> bool:
        """Check if effect has expired by step count."""
        if self.duration_steps is None:
            return False
        return (current_step - self.applied_at_step) >= self.duration_steps


@dataclass
class EffectDefinition:
    """
    Definition of a status effect (loaded from config).
    """
    id: str
    name: str
    description: str
    type: EffectType
    severity: Severity

    # Default duration
    default_duration_steps: Optional[int] = None
    cure_condition: Optional[str] = None

    # Effects to apply
    effects: dict[str, Any] = field(default_factory=dict)

    # RPG
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
            rpg_description=self.rpg_description
        )


@dataclass
class EffectRule:
    """
    Rule for automatically triggering a status effect.

    Loaded from configs/progression/effects.yaml
    """
    id: str
    effect_id: str  # Which effect to apply

    # Trigger conditions
    trigger_type: str  # "metric_threshold", "consecutive_failures", "event"
    trigger_config: dict = field(default_factory=dict)
    # metric_threshold: {"metric": "accuracy", "op": "lt", "value": 0.5, "window": 10}
    # consecutive_failures: {"count": 3, "result": "crit_miss"}
    # event: {"event_type": "oom"}

    # Cooldown (steps between applications)
    cooldown_steps: int = 100

    # Scope
    skill_id: Optional[str] = None  # Specific skill, or None for global

    # State tracking
    last_triggered_step: int = 0


@dataclass
class HeroIdentity:
    """
    Identity information about a hero (model).
    """
    id: str
    name: str

    # Model info
    architecture: str  # "qwen", "llama", etc.
    generation: str    # "3", "2.5", etc.
    size: str          # "0.6B", "7B", etc.
    variant: str       # "base", "instruct", "chat"

    # Current form
    checkpoint_path: Optional[str] = None
    checkpoint_step: int = 0

    # RPG
    race: Optional[str] = None        # "Qwen'dal"
    stature: Optional[str] = None     # "Sprite"
    class_name: Optional[str] = None  # "Guild Veteran"


@dataclass
class HeroState:
    """
    Complete state of the hero.

    Persisted to status/hero_state.json
    """
    hero_id: str
    identity: HeroIdentity

    # Skill states
    skills: dict[str, SkillState] = field(default_factory=dict)

    # Active effects
    active_effects: list[StatusEffect] = field(default_factory=list)

    # Current context
    current_region: str = ""
    current_step: int = 0
    current_run_id: Optional[str] = None

    # Aggregate stats
    total_quests: int = 0
    total_xp: float = 0.0
    total_crits: int = 0
    total_misses: int = 0

    # Timestamps
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
        """Add a status effect."""
        # Remove any existing effect with same ID
        self.active_effects = [e for e in self.active_effects if e.id != effect.id]
        self.active_effects.append(effect)

    def remove_effect(self, effect_id: str):
        """Remove a status effect by ID."""
        self.active_effects = [e for e in self.active_effects if e.id != effect_id]

    def clear_expired_effects(self, current_step: int):
        """Remove effects that have expired."""
        self.active_effects = [e for e in self.active_effects
                               if not e.is_expired(current_step)]
```

## 1.7 Create guild/runs/types.py

```python
# guild/runs/types.py
"""Run type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from guild.types import Status


class RunType(Enum):
    """Types of runs."""
    TRAINING = "training"       # Campaign
    EVALUATION = "evaluation"   # Trial / Dungeon
    AUDIT = "audit"             # Investigation
    EXPERIMENT = "experiment"   # Lab work
    GENERATION = "generation"   # Quest forge


@dataclass
class RunConfig:
    """
    Configuration for a run.

    Can be loaded from configs/runs/{type}.yaml or created programmatically.
    """
    id: str
    type: RunType
    name: str = ""
    description: str = ""

    # Where
    facility_id: str = ""

    # What
    hero_id: str = ""
    quest_filters: dict = field(default_factory=dict)
    # {"skills": ["logic_weaving"], "regions": ["novice_valley"], "difficulty_max": 3}

    # How much
    max_steps: Optional[int] = None
    max_quests: Optional[int] = None
    max_duration_seconds: Optional[int] = None

    # Training hyperparameters
    hyperparams: dict = field(default_factory=dict)

    # Logging
    log_level: str = "INFO"
    log_facility_id: str = ""

    # Checkpointing
    checkpoint_every_steps: int = 1000
    checkpoint_facility_id: str = ""

    # Tags
    tags: list[str] = field(default_factory=list)


@dataclass
class RunState:
    """
    Current state of a run.

    Persisted to status/runs/{run_id}.json
    """
    run_id: str
    config: RunConfig
    status: Status = Status.PENDING

    # Progress
    current_step: int = 0
    quests_completed: int = 0
    quests_succeeded: int = 0

    # Timing
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics (aggregated)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Checkpoints
    last_checkpoint_step: int = 0
    checkpoint_paths: list[str] = field(default_factory=list)

    # Incidents during this run
    incident_ids: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total run duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        """Quest success rate."""
        if self.quests_completed == 0:
            return 0.0
        return self.quests_succeeded / self.quests_completed
```

## 1.8 Create guild/incidents/types.py

```python
# guild/incidents/types.py
"""Incident type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from guild.types import Severity


class IncidentCategory(Enum):
    """Categories of incidents."""
    DATA = "data"           # Cursed scrolls
    TRAINING = "training"   # NaN dragon
    INFRA = "infra"         # OOM, disk
    LOGIC = "logic"         # Code bugs


class IncidentStatus(Enum):
    """Incident lifecycle."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    WONTFIX = "wontfix"


@dataclass
class Incident:
    """
    A detected problem/bug.

    Persisted to status/incidents/{id}.json
    """
    id: str
    category: IncidentCategory
    severity: Severity

    # Description
    title: str
    description: str

    # Context
    detected_at_step: int
    detected_at_time: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    quest_id: Optional[str] = None
    facility_id: Optional[str] = None

    # Technical details
    context: dict = field(default_factory=dict)
    # stack_trace, metrics_snapshot, etc.

    # Status
    status: IncidentStatus = IncidentStatus.OPEN
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # RPG flavor
    rpg_name: Optional[str] = None
    rpg_location: Optional[str] = None


@dataclass
class IncidentRule:
    """
    Rule for detecting incidents.

    Loaded from configs/incidents/rules.yaml
    """
    id: str
    name: str
    category: IncidentCategory
    severity: Severity

    # Detection config
    detector_type: str  # "metric", "exception", "pattern"
    detector_config: dict = field(default_factory=dict)

    # Template for incident
    title_template: str
    description_template: str

    # RPG
    rpg_name_template: Optional[str] = None
```

## 1.9 Create guild/combat/types.py

```python
# guild/combat/types.py
"""Combat system type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from guild.quests.types import CombatResult


class CombatStance(Enum):
    """Combat stances (protocol modes)."""
    THOUGHTFUL = "thoughtful"   # Emoji thinking mode
    QUICK_DRAW = "quick_draw"   # Direct mode
    ALTERNATING = "alternating" # 50/50


@dataclass
class CombatConfig:
    """
    Combat system configuration.
    """
    # XP awards per result
    xp_crit: int = 15
    xp_hit: int = 10
    xp_glancing: int = 5
    xp_miss: int = 2
    xp_crit_miss: int = 0

    # Difficulty multipliers
    difficulty_multipliers: dict[int, float] = field(default_factory=lambda: {
        1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.5,
        6: 1.7, 7: 2.0, 8: 2.3, 9: 2.6, 10: 3.0
    })

    # Stance settings
    default_stance: CombatStance = CombatStance.ALTERNATING

    # Debuff triggers
    crit_miss_debuff_threshold: int = 3  # consecutive
    miss_debuff_threshold: int = 5       # consecutive

    def get_base_xp(self, result: CombatResult) -> int:
        """Get base XP for a combat result."""
        return {
            CombatResult.CRITICAL_HIT: self.xp_crit,
            CombatResult.HIT: self.xp_hit,
            CombatResult.GLANCING: self.xp_glancing,
            CombatResult.MISS: self.xp_miss,
            CombatResult.CRITICAL_MISS: self.xp_crit_miss
        }.get(result, 0)

    def get_difficulty_multiplier(self, level: int) -> float:
        """Get XP multiplier for difficulty level."""
        return self.difficulty_multipliers.get(level, 1.0)


@dataclass
class StanceConfig:
    """Configuration for combat stances."""
    # Thinking tokens
    thinking_emojis: list[str] = field(default_factory=lambda: [
        "🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"
    ])
    stop_emojis: list[str] = field(default_factory=lambda: [
        "🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"
    ])

    # Counts
    min_thinking_count: int = 1
    max_thinking_count: int = 10
    min_stop_count: int = 2
    max_stop_count: int = 4
```

## 1.10 Create guild/hero/types.py

```python
# guild/hero/types.py
"""Hero type definitions (re-export from progression for convenience)."""

from guild.progression.types import HeroIdentity, HeroState

__all__ = ["HeroIdentity", "HeroState"]
```

## 1.11 Create tests/guild/test_types.py

```python
# tests/guild/test_types.py
"""Tests for guild type definitions."""

import pytest
from datetime import datetime

from guild.types import Severity, Status, generate_id
from guild.skills.types import SkillConfig, SkillState, SkillCategory
from guild.quests.types import (
    QuestTemplate, QuestInstance, QuestResult,
    QuestDifficulty, CombatResult
)
from guild.facilities.types import Facility, FacilityType
from guild.progression.types import (
    StatusEffect, EffectType, HeroState, HeroIdentity
)
from guild.runs.types import RunConfig, RunState, RunType
from guild.incidents.types import Incident, IncidentCategory


class TestBasicTypes:
    def test_severity_enum(self):
        assert Severity.LOW.value == "low"
        assert Severity.CRITICAL.value == "critical"

    def test_generate_id(self):
        id1 = generate_id("test")
        id2 = generate_id("test")
        assert id1.startswith("test_")
        assert id1 != id2  # Should be unique


class TestSkillTypes:
    def test_skill_config_creation(self):
        skill = SkillConfig(
            id="logic_weaving",
            name="Logic Weaving",
            description="Deductive reasoning",
            category=SkillCategory.REASONING,
            accuracy_thresholds={1: 0.6, 2: 0.65, 3: 0.7}
        )
        assert skill.id == "logic_weaving"
        assert skill.get_threshold(1) == 0.6
        assert skill.get_threshold(3) == 0.7

    def test_skill_state_accuracy(self):
        state = SkillState(skill_id="test")
        assert state.accuracy == 0.0

        state.record_result(True)
        state.record_result(True)
        state.record_result(False)
        assert state.accuracy == pytest.approx(2/3)

    def test_skill_state_level_up(self):
        state = SkillState(skill_id="test", xp_total=1000)
        state.record_level_up()
        assert state.level == 2
        assert state.xp_marks[2] == 1000


class TestQuestTypes:
    def test_quest_difficulty_from_level(self):
        assert QuestDifficulty.from_level(1) == QuestDifficulty.BRONZE
        assert QuestDifficulty.from_level(5) == QuestDifficulty.GOLD
        assert QuestDifficulty.from_level(10) == QuestDifficulty.DRAGON

    def test_quest_instance_creation(self):
        template = QuestTemplate(
            id="syllo_basic",
            name="Basic SYLLO",
            description="Easy puzzle",
            skills=["logic_weaving"],
            regions=["novice_valley"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="syllo_gen",
            evaluator_id="syllo_eval"
        )

        instance = QuestInstance.create(
            template=template,
            prompt="Solve this puzzle..."
        )

        assert instance.template_id == "syllo_basic"
        assert instance.skills == ["logic_weaving"]
        assert instance.difficulty == QuestDifficulty.BRONZE

    def test_quest_result_success(self):
        result = QuestResult(
            quest_id="q1",
            hero_id="h1",
            response="answer",
            combat_result=CombatResult.HIT,
            xp_awarded={"logic_weaving": 10}
        )
        assert result.success is True
        assert result.total_xp == 10


class TestFacilityTypes:
    def test_facility_get_path(self):
        facility = Facility(
            id="arena",
            name="Arena",
            type=FacilityType.BATTLEFIELD,
            base_path="/home/user/training",
            paths={"checkpoints": "current_model/"}
        )

        path = facility.get_path("checkpoints")
        assert path == "/home/user/training/current_model/"

        path = facility.get_path("checkpoints", "checkpoint-1000")
        assert path == "/home/user/training/current_model/checkpoint-1000"


class TestProgressionTypes:
    def test_status_effect_expiry(self):
        effect = StatusEffect(
            id="confusion",
            name="Confusion",
            description="Confused",
            type=EffectType.DEBUFF,
            severity=Severity.MEDIUM,
            applied_at_step=100,
            duration_steps=50
        )

        assert effect.is_expired(100) is False
        assert effect.is_expired(149) is False
        assert effect.is_expired(150) is True

    def test_hero_state_health(self):
        identity = HeroIdentity(
            id="hero1",
            name="Test Hero",
            architecture="qwen",
            generation="3",
            size="0.6B",
            variant="base"
        )

        state = HeroState(hero_id="hero1", identity=identity)
        assert state.health == "healthy"

        # Add severe effect
        effect = StatusEffect(
            id="nan_dragon",
            name="NaN Dragon",
            description="Training collapsed",
            type=EffectType.DEBUFF,
            severity=Severity.CRITICAL,
            applied_at_step=0
        )
        state.add_effect(effect)
        assert state.health == "wounded"


class TestRunTypes:
    def test_run_config_creation(self):
        config = RunConfig(
            id="run1",
            type=RunType.TRAINING,
            name="Test Run",
            facility_id="arena_4090",
            hero_id="qwendal",
            max_steps=1000
        )
        assert config.type == RunType.TRAINING

    def test_run_state_success_rate(self):
        config = RunConfig(id="r1", type=RunType.TRAINING)
        state = RunState(run_id="r1", config=config)

        state.quests_completed = 100
        state.quests_succeeded = 75
        assert state.success_rate == 0.75


class TestIncidentTypes:
    def test_incident_creation(self):
        incident = Incident(
            id="inc1",
            category=IncidentCategory.TRAINING,
            severity=Severity.CRITICAL,
            title="NaN Loss Detected",
            description="Loss became NaN at step 1000",
            detected_at_step=1000
        )
        assert incident.category == IncidentCategory.TRAINING
        assert incident.severity == Severity.CRITICAL
```

## Checkpoint 1

Run tests:
```bash
cd /path/to/training
pytest tests/guild/test_types.py -v
```

- [ ] All imports work
- [ ] All dataclasses instantiate
- [ ] All tests pass
- [ ] No circular imports

---

# Phase 2: Configuration System

## Goal
Create YAML config loading infrastructure with validation.

## 2.1 Create guild/config/__init__.py

```python
# guild/config/__init__.py
"""Configuration loading and validation."""

from guild.config.loader import (
    load_config,
    load_all_configs,
    get_config_path,
    ConfigLoader
)

__all__ = ["load_config", "load_all_configs", "get_config_path", "ConfigLoader"]
```

## 2.2 Create guild/config/loader.py

```python
# guild/config/loader.py
"""YAML configuration loader with environment variable support."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Optional, TypeVar, Type
from dataclasses import fields, is_dataclass

T = TypeVar('T')

# Default config directory
_config_dir: Optional[Path] = None


def set_config_dir(path: str | Path):
    """Set the global config directory."""
    global _config_dir
    _config_dir = Path(path)


def get_config_dir() -> Path:
    """Get the config directory."""
    global _config_dir
    if _config_dir is None:
        # Check environment variable
        env_path = os.environ.get("GUILD_CONFIG_DIR")
        if env_path:
            _config_dir = Path(env_path)
        else:
            # Default to configs/ relative to this file's grandparent
            _config_dir = Path(__file__).parent.parent.parent / "configs"
    return _config_dir


def get_config_path(category: str, name: str, ext: str = ".yaml") -> Path:
    """Get path to a specific config file."""
    return get_config_dir() / category / f"{name}{ext}"


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in strings.

    Supports:
    - ${VAR} - required variable
    - ${VAR:-default} - variable with default
    """
    if isinstance(value, str):
        # Pattern for ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(f"Environment variable {var_name} not set and no default provided")

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [expand_env_vars(v) for v in value]

    return value


def load_yaml(path: Path | str) -> dict:
    """Load a YAML file with environment variable expansion."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return expand_env_vars(data)


def dict_to_dataclass(data: dict, cls: Type[T]) -> T:
    """Convert a dict to a dataclass instance."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    # Get field names and types
    field_names = {f.name for f in fields(cls)}
    field_types = {f.name: f.type for f in fields(cls)}

    # Filter to known fields
    filtered = {k: v for k, v in data.items() if k in field_names}

    # Handle nested dataclasses and enums
    for field_name, value in filtered.items():
        field_type = field_types.get(field_name)
        if field_type and hasattr(field_type, '__origin__'):
            # Handle Optional, List, Dict etc.
            continue
        elif field_type and is_dataclass(field_type):
            if isinstance(value, dict):
                filtered[field_name] = dict_to_dataclass(value, field_type)
        elif field_type and hasattr(field_type, '__members__'):
            # Enum
            if isinstance(value, str):
                filtered[field_name] = field_type(value)

    return cls(**filtered)


def load_config(category: str, name: str, cls: Optional[Type[T]] = None) -> T | dict:
    """
    Load a config file and optionally convert to dataclass.

    Args:
        category: Config category (e.g., "skills", "facilities")
        name: Config name without extension (e.g., "logic_weaving")
        cls: Optional dataclass to convert to

    Returns:
        Loaded config as dict or dataclass instance
    """
    path = get_config_path(category, name)
    data = load_yaml(path)

    if cls is not None:
        return dict_to_dataclass(data, cls)
    return data


def load_all_configs(category: str, cls: Optional[Type[T]] = None,
                     pattern: str = "*.yaml") -> dict[str, T | dict]:
    """
    Load all config files in a category.

    Args:
        category: Config category
        cls: Optional dataclass to convert to
        pattern: Glob pattern for files

    Returns:
        Dict mapping name -> config
    """
    category_dir = get_config_dir() / category
    if not category_dir.exists():
        return {}

    configs = {}
    for path in category_dir.glob(pattern):
        if path.name.startswith("_"):  # Skip schema files
            continue
        name = path.stem
        try:
            configs[name] = load_config(category, name, cls)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    return configs


class ConfigLoader:
    """
    Manages loading and caching of configurations.
    """

    def __init__(self, config_dir: Optional[Path | str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = get_config_dir()

        self._cache: dict[str, Any] = {}

    def load(self, category: str, name: str, cls: Optional[Type[T]] = None,
             use_cache: bool = True) -> T | dict:
        """Load a config with optional caching."""
        cache_key = f"{category}/{name}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.config_dir / category / f"{name}.yaml"
        data = load_yaml(path)

        if cls is not None:
            result = dict_to_dataclass(data, cls)
        else:
            result = data

        if use_cache:
            self._cache[cache_key] = result

        return result

    def load_all(self, category: str, cls: Optional[Type[T]] = None) -> dict[str, T | dict]:
        """Load all configs in a category."""
        category_dir = self.config_dir / category
        if not category_dir.exists():
            return {}

        configs = {}
        for path in category_dir.glob("*.yaml"):
            if path.name.startswith("_"):
                continue
            name = path.stem
            try:
                configs[name] = self.load(category, name, cls)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        return configs

    def clear_cache(self):
        """Clear the config cache."""
        self._cache.clear()
```

## 2.3 Create Example Configs

```yaml
# configs/skills/logic_weaving.yaml
id: logic_weaving
name: Logic Weaving
description: >
  The discipline of chaining deductions, solving puzzles, and
  constructing valid arguments. Primary skill for SYLLO tasks.
category: reasoning

tags:
  - reasoning
  - deduction
  - puzzles
  - syllo

metrics:
  - accuracy
  - word_accuracy
  - json_validity
primary_metric: accuracy

accuracy_thresholds:
  1: 0.60
  2: 0.65
  3: 0.70
  4: 0.72
  5: 0.75
  6: 0.78
  7: 0.80
  8: 0.82
  9: 0.85
  10: 0.88

xp_multiplier: 1.0

rpg_name: Logic Weaving
rpg_description: >
  The art of weaving threads of logic into coherent tapestries of proof.
  Masters of this discipline can untangle the most complex syllogisms.
```

```yaml
# configs/skills/oath_binding.yaml
id: oath_binding
name: Oath Binding
description: >
  The discipline of following instructions precisely and honoring
  constraints. Essential for reliable model behavior.
category: instruction

tags:
  - instruction
  - constraints
  - format
  - compliance

metrics:
  - instruction_accuracy
  - format_compliance
primary_metric: instruction_accuracy

accuracy_thresholds:
  1: 0.55
  2: 0.60
  3: 0.65
  4: 0.70
  5: 0.75
  6: 0.80
  7: 0.85
  8: 0.88
  9: 0.90
  10: 0.92

xp_multiplier: 1.2

rpg_name: Oath Binding
rpg_description: >
  The sacred art of binding oneself to constraints and honoring instructions.
  Those who master it are trusted with the most precise tasks.
```

```yaml
# configs/facilities/example.yaml
# Example facility configuration
# Copy to local.yaml and customize for your setup

facilities:
  main_hub:
    id: main_hub
    name: Main Hub
    type: hub
    description: Central management and monitoring
    host: localhost
    port: 8765
    base_path: "${GUILD_BASE_DIR:-.}"
    paths:
      status: status/
      logs: logs/
      queue: queue/
      ui: monitoring/ui/
    is_local: true
    rpg_name: The Inn
    rpg_description: Central hub where heroes rest and quests are managed

  training_gpu:
    id: training_gpu
    name: Training GPU
    type: battlefield
    description: Primary training facility
    host: localhost
    base_path: "${GUILD_BASE_DIR:-.}"
    paths:
      checkpoints: current_model/
      snapshots: snapshots/
      models: models/
    resources:
      - id: gpu_0
        type: gpu
        properties:
          vram_gb: 24
          model: GPU
    is_local: true
    rpg_name: The Arena
    rpg_description: Battlefield where heroes train and grow stronger

  archive:
    id: archive
    name: Archive Storage
    type: archive
    description: Long-term storage
    base_path: "${GUILD_ARCHIVE_DIR:-./archive}"
    paths:
      backups: backups/
      datasets: datasets/
      logs: logs/
    is_local: true
    rpg_name: The Deep Vault
    rpg_description: Grand archive where soul anchors and ancient tomes are kept

default_facility: training_gpu
```

```yaml
# configs/progression/effects.yaml
# Status effect definitions

effects:
  tunnel_vision:
    id: tunnel_vision
    name: Tunnel Vision
    description: Overfitting - high training accuracy, low validation accuracy
    type: debuff
    severity: medium
    default_duration_steps: null  # Until cured
    cure_condition: val_gap < 0.25 for 100 steps
    effects:
      accuracy_display_warning: true
    rpg_name: Tunnel Vision
    rpg_description: >
      The hero has become too focused on familiar patterns,
      losing sight of the broader picture.

  confusion:
    id: confusion
    name: Confusion
    description: Mode collapse or repeated failures
    type: debuff
    severity: medium
    default_duration_steps: 500
    cure_condition: 10 consecutive successes
    effects:
      xp_multiplier: 0.8
    rpg_name: Confusion
    rpg_description: >
      The hero's mind is clouded, leading to erratic behavior
      and repeated mistakes.

  curse_of_repetition:
    id: curse_of_repetition
    name: Curse of Repetition
    description: Degenerate outputs with loops or garbage
    type: debuff
    severity: high
    default_duration_steps: null
    cure_condition: manual intervention
    effects:
      requires_attention: true
    rpg_name: Curse of Repetition
    rpg_description: >
      A dark curse has befallen the hero, causing them to repeat
      the same words endlessly like a broken spell.

  exhaustion:
    id: exhaustion
    name: Exhaustion
    description: OOM or resource exhaustion
    type: debuff
    severity: high
    default_duration_steps: null
    cure_condition: restart with reduced load
    effects:
      training_blocked: true
    rpg_name: Exhaustion
    rpg_description: >
      The hero has collapsed from carrying too heavy a burden.
      They cannot continue until the load is lightened.

  reality_tear:
    id: reality_tear
    name: Reality Tear
    description: NaN loss or training collapse
    type: debuff
    severity: critical
    default_duration_steps: null
    cure_condition: manual intervention required
    effects:
      training_blocked: true
      requires_immediate_attention: true
    rpg_name: Reality Tear
    rpg_description: >
      A catastrophic rift in training reality. The very fabric
      of the hero's learning has been torn asunder.

rules:
  - id: detect_tunnel_vision
    effect_id: tunnel_vision
    trigger_type: metric_threshold
    trigger_config:
      metric: val_train_gap
      op: gt
      value: 0.3
      window: 10
    cooldown_steps: 500

  - id: detect_confusion
    effect_id: confusion
    trigger_type: consecutive_failures
    trigger_config:
      count: 5
      result: miss
    cooldown_steps: 200

  - id: detect_curse
    effect_id: curse_of_repetition
    trigger_type: pattern_match
    trigger_config:
      patterns:
        - "user.*user.*user"
        - "(\\w+\\s+){3,}\\1"
    cooldown_steps: 100

  - id: detect_nan
    effect_id: reality_tear
    trigger_type: metric_threshold
    trigger_config:
      metric: loss
      op: is_nan
    cooldown_steps: 0  # Always trigger
```

```yaml
# configs/regions/novice_valley.yaml
id: novice_valley
name: Novice Valley
description: Starting region with easy quests for beginners

level_range:
  min: 1
  max: 3

difficulty_range:
  min: bronze
  max: silver

quest_templates:
  - syllo_basic_4word
  - syllo_basic_5word

skills_trained:
  - logic_weaving

unlock_requirements: null  # Starting region

rpg_name: Novice Valley
rpg_description: >
  A peaceful valley where new heroes begin their journey.
  The quests here are gentle, designed to build confidence
  and establish fundamental skills.
```

## 2.4 Create tests/guild/test_config.py

```python
# tests/guild/test_config.py
"""Tests for configuration loading."""

import pytest
import tempfile
import os
from pathlib import Path

from guild.config.loader import (
    load_yaml, expand_env_vars, dict_to_dataclass,
    load_config, ConfigLoader, set_config_dir
)
from guild.skills.types import SkillConfig, SkillCategory


class TestEnvVarExpansion:
    def test_simple_var(self):
        os.environ["TEST_VAR"] = "hello"
        result = expand_env_vars("${TEST_VAR}")
        assert result == "hello"

    def test_var_with_default(self):
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]
        result = expand_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_nested_dict(self):
        os.environ["TEST_PATH"] = "/custom/path"
        data = {
            "base": "${TEST_PATH}",
            "nested": {
                "path": "${TEST_PATH}/subdir"
            }
        }
        result = expand_env_vars(data)
        assert result["base"] == "/custom/path"
        assert result["nested"]["path"] == "/custom/path/subdir"


class TestDictToDataclass:
    def test_simple_conversion(self):
        data = {
            "id": "test_skill",
            "name": "Test Skill",
            "description": "A test",
            "category": "reasoning"
        }
        result = dict_to_dataclass(data, SkillConfig)
        assert result.id == "test_skill"
        assert result.category == SkillCategory.REASONING


class TestConfigLoader:
    @pytest.fixture
    def temp_config_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create skills directory
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create a test skill config
            skill_yaml = """
id: test_skill
name: Test Skill
description: A test skill
category: reasoning
tags:
  - test
metrics:
  - accuracy
primary_metric: accuracy
accuracy_thresholds:
  1: 0.6
  2: 0.7
"""
            (skills_dir / "test_skill.yaml").write_text(skill_yaml)

            yield tmpdir

    def test_load_skill_config(self, temp_config_dir):
        set_config_dir(temp_config_dir)

        config = load_config("skills", "test_skill", SkillConfig)
        assert config.id == "test_skill"
        assert config.name == "Test Skill"
        assert config.category == SkillCategory.REASONING
        assert config.get_threshold(1) == 0.6

    def test_config_loader_caching(self, temp_config_dir):
        loader = ConfigLoader(temp_config_dir)

        # Load twice
        config1 = loader.load("skills", "test_skill", SkillConfig)
        config2 = loader.load("skills", "test_skill", SkillConfig)

        # Should be same object due to caching
        assert config1 is config2

        # Clear cache and reload
        loader.clear_cache()
        config3 = loader.load("skills", "test_skill", SkillConfig)
        assert config3 is not config1
```

## Checkpoint 2

```bash
pytest tests/guild/test_config.py -v
```

- [ ] YAML loading works
- [ ] Environment variable expansion works
- [ ] Dataclass conversion works
- [ ] Caching works
- [ ] Example configs are valid

---

# Phase 3: Facilities & Paths

## Goal
Replace all hardcoded paths with facility-based resolution.

## 3.1 Create guild/facilities/resolver.py

```python
# guild/facilities/resolver.py
"""Path resolution using facility configurations."""

import os
from pathlib import Path
from typing import Optional

from guild.facilities.types import Facility, FacilityType
from guild.config.loader import load_yaml, dict_to_dataclass


class PathResolver:
    """
    Resolves logical paths to physical paths using facility configs.

    Path formats:
    - "facility:arena_4090:checkpoints" -> /path/to/arena/checkpoints
    - "facility:inn_3090:logs/training" -> /path/to/inn/logs/training
    - "@checkpoints" -> current facility's checkpoints
    - "@checkpoints/step-1000" -> current facility's checkpoints/step-1000
    - "./relative" -> relative to cwd
    - "/absolute" -> unchanged
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        self._facilities: dict[str, Facility] = {}
        self._current_facility: Optional[str] = None
        self._default_facility: Optional[str] = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str | Path):
        """Load facility configuration from YAML."""
        data = load_yaml(config_path)

        self._default_facility = data.get("default_facility")

        for fac_id, fac_data in data.get("facilities", {}).items():
            # Handle nested resources
            resources = fac_data.pop("resources", [])

            # Convert type string to enum
            if "type" in fac_data and isinstance(fac_data["type"], str):
                fac_data["type"] = FacilityType(fac_data["type"])

            facility = Facility(**fac_data)
            self._facilities[fac_id] = facility

    def add_facility(self, facility: Facility):
        """Add a facility directly."""
        self._facilities[facility.id] = facility

    def resolve(self, path_spec: str) -> Path:
        """Resolve a path specification to a physical path."""
        if path_spec.startswith("facility:"):
            # Explicit facility path
            parts = path_spec.split(":", 2)
            if len(parts) < 3:
                raise ValueError(f"Invalid facility path: {path_spec}")
            _, facility_id, subpath = parts
            return self._resolve_facility_path(facility_id, subpath)

        elif path_spec.startswith("@"):
            # Shorthand for current/default facility
            key = path_spec[1:]
            facility_id = self._current_facility or self._default_facility
            if not facility_id:
                raise ValueError("No current or default facility set")

            # Check if it's a path alias or direct path
            if "/" in key:
                alias, subpath = key.split("/", 1)
            else:
                alias, subpath = key, ""

            return self._resolve_facility_path(facility_id, alias, subpath)

        elif path_spec.startswith("~"):
            return Path(path_spec).expanduser()

        else:
            # Regular path - expand env vars
            expanded = os.path.expandvars(path_spec)
            return Path(expanded).expanduser()

    def _resolve_facility_path(self, facility_id: str, key: str,
                                subpath: str = "") -> Path:
        """Resolve a path within a facility."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")

        facility = self._facilities[facility_id]
        base = Path(os.path.expandvars(facility.base_path)).expanduser()

        # Check if key is a path alias
        if key in facility.paths:
            path = base / facility.paths[key]
        else:
            path = base / key

        if subpath:
            path = path / subpath

        return path

    def set_current_facility(self, facility_id: str):
        """Set the current facility for @ shortcuts."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")
        self._current_facility = facility_id

    def get_facility(self, facility_id: str) -> Facility:
        """Get a facility by ID."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")
        return self._facilities[facility_id]

    def list_facilities(self, type_filter: Optional[FacilityType] = None) -> list[str]:
        """List facility IDs, optionally filtered by type."""
        if type_filter:
            return [fid for fid, f in self._facilities.items()
                    if f.type == type_filter]
        return list(self._facilities.keys())

    @property
    def current_facility_id(self) -> Optional[str]:
        return self._current_facility or self._default_facility


# Global resolver instance
_resolver: Optional[PathResolver] = None


def init_resolver(config_path: Optional[str | Path] = None) -> PathResolver:
    """
    Initialize the global path resolver.

    If no config_path is provided, looks for:
    1. GUILD_FACILITIES_CONFIG environment variable
    2. configs/facilities/local.yaml
    3. configs/facilities/example.yaml
    """
    global _resolver

    if config_path is None:
        # Check environment variable
        config_path = os.environ.get("GUILD_FACILITIES_CONFIG")

        if not config_path:
            # Try local.yaml first, fall back to example.yaml
            from guild.config.loader import get_config_dir
            local_path = get_config_dir() / "facilities" / "local.yaml"
            example_path = get_config_dir() / "facilities" / "example.yaml"

            if local_path.exists():
                config_path = local_path
            elif example_path.exists():
                config_path = example_path
            else:
                raise FileNotFoundError(
                    "No facility config found. Create configs/facilities/local.yaml "
                    "or set GUILD_FACILITIES_CONFIG environment variable."
                )

    _resolver = PathResolver(config_path)
    return _resolver


def get_resolver() -> PathResolver:
    """Get the global resolver, initializing if needed."""
    global _resolver
    if _resolver is None:
        init_resolver()
    return _resolver


def resolve(path_spec: str) -> Path:
    """Resolve a path using the global resolver."""
    return get_resolver().resolve(path_spec)


def get_facility(facility_id: str) -> Facility:
    """Get a facility by ID using the global resolver."""
    return get_resolver().get_facility(facility_id)


def set_current_facility(facility_id: str):
    """Set the current facility."""
    get_resolver().set_current_facility(facility_id)
```

## 3.2 Create guild/facilities/registry.py

```python
# guild/facilities/registry.py
"""Facility registry for managing hardware resources."""

from typing import Optional
from guild.facilities.types import Facility, FacilityType
from guild.facilities.resolver import get_resolver


class FacilityRegistry:
    """
    Registry for accessing facility information.

    Wraps the resolver for convenience methods.
    """

    def __init__(self):
        self._resolver = get_resolver()

    def get(self, facility_id: str) -> Facility:
        """Get a facility by ID."""
        return self._resolver.get_facility(facility_id)

    def list(self, type_filter: Optional[FacilityType] = None) -> list[Facility]:
        """List all facilities, optionally filtered by type."""
        ids = self._resolver.list_facilities(type_filter)
        return [self._resolver.get_facility(fid) for fid in ids]

    def get_hub(self) -> Optional[Facility]:
        """Get the hub facility (Inn)."""
        hubs = self.list(FacilityType.HUB)
        return hubs[0] if hubs else None

    def get_battlefield(self) -> Optional[Facility]:
        """Get the primary battlefield (Arena)."""
        battlefields = self.list(FacilityType.BATTLEFIELD)
        return battlefields[0] if battlefields else None

    def get_archive(self) -> Optional[Facility]:
        """Get the archive facility (Vault)."""
        archives = self.list(FacilityType.ARCHIVE)
        return archives[0] if archives else None


# Convenience function
def get_registry() -> FacilityRegistry:
    """Get the facility registry."""
    return FacilityRegistry()
```

## 3.3 Update guild/__init__.py

```python
# guild/__init__.py (updated)
"""
Guild Trainer - A generic framework for LLM training with RPG-style progression.
"""

__version__ = "0.1.0"

from guild.facilities.resolver import (
    init_resolver,
    resolve,
    get_facility,
    set_current_facility,
    get_resolver
)

__all__ = [
    "init_resolver",
    "resolve",
    "get_facility",
    "set_current_facility",
    "get_resolver"
]
```

## 3.4 Create Backward Compatibility Wrapper

```python
# core/paths.py (replacement)
"""
Path utilities with backward compatibility.

This module now uses the Guild facility resolver under the hood,
but maintains the same API for backward compatibility.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Lazy initialization flag
_initialized = False
_base_dir: Optional[Path] = None


def _ensure_initialized():
    """Ensure Guild resolver is initialized."""
    global _initialized, _base_dir

    if _initialized:
        return

    try:
        from guild.facilities.resolver import init_resolver, get_resolver

        # Try to initialize resolver
        try:
            init_resolver()
            resolver = get_resolver()
            _base_dir = resolver.resolve("@")
            logger.info(f"Guild resolver initialized, base_dir: {_base_dir}")
        except FileNotFoundError:
            # No facility config - fall back to legacy detection
            _base_dir = _detect_base_dir_legacy()
            logger.info(f"Using legacy path detection, base_dir: {_base_dir}")

        _initialized = True

    except ImportError:
        # Guild not installed - use legacy
        _base_dir = _detect_base_dir_legacy()
        _initialized = True


def _detect_base_dir_legacy() -> Path:
    """Legacy base directory detection."""
    # Check environment variable
    env_path = os.environ.get("GUILD_BASE_DIR") or os.environ.get("TRAINING_BASE_DIR")
    if env_path:
        return Path(env_path)

    # Try to detect from current file location
    current = Path(__file__).resolve()

    # Walk up looking for marker files
    markers = ["config.json", "CLAUDE.md", "pyproject.toml"]

    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    # Fall back to current directory
    return Path.cwd()


def get_base_dir() -> Path:
    """
    Get the base directory for the training system.

    Returns:
        Path to the base directory
    """
    _ensure_initialized()
    return _base_dir


def resolve_path(path_spec: str) -> Path:
    """
    Resolve a path specification.

    Supports:
    - facility:id:path - Guild facility paths
    - @path - Current facility shorthand
    - Relative paths - Relative to base_dir
    - Absolute paths - Unchanged

    Args:
        path_spec: Path specification

    Returns:
        Resolved Path object
    """
    _ensure_initialized()

    # Try Guild resolver first
    if path_spec.startswith("facility:") or path_spec.startswith("@"):
        try:
            from guild.facilities.resolver import resolve
            return resolve(path_spec)
        except ImportError:
            pass

    # Handle regular paths
    if path_spec.startswith("/") or path_spec.startswith("~"):
        return Path(path_spec).expanduser()

    # Relative to base_dir
    return _base_dir / path_spec


# Convenience functions for common paths
def get_status_dir() -> Path:
    """Get the status directory."""
    return resolve_path("status")


def get_logs_dir() -> Path:
    """Get the logs directory."""
    return resolve_path("logs")


def get_queue_dir() -> Path:
    """Get the queue directory."""
    return resolve_path("queue")


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory."""
    return resolve_path("current_model")


def get_models_dir() -> Path:
    """Get the models directory."""
    return resolve_path("models")


def get_config_path() -> Path:
    """Get the main config.json path."""
    return resolve_path("config.json")
```

## 3.5 Create configs/facilities/local.yaml for Current Setup

```yaml
# configs/facilities/local.yaml
# Local facility configuration for this machine
# GITIGNORED - do not commit

facilities:
  inn_3090:
    id: inn_3090
    name: The 3090 Inn
    type: hub
    description: RTX 3090 server - central hub and inference
    host: 192.168.x.x
    port: 8765
    base_path: /path/to/training
    paths:
      status: status/
      logs: logs/
      queue: queue/
      ui: monitoring/ui/
      inbox: inbox/
    is_local: false
    rpg_name: The 3090 Inn
    rpg_description: Central hub where heroes rest and quests are managed

  arena_4090:
    id: arena_4090
    name: The 4090 Arena
    type: battlefield
    description: RTX 4090 - primary training GPU
    host: localhost
    base_path: /path/to/training
    paths:
      checkpoints: current_model/
      snapshots: snapshots/
      models: models/
      data: data/
      training: current_model/
    resources:
      - id: gpu_0
        type: gpu
        properties:
          vram_gb: 24
          model: RTX 4090
    is_local: true
    rpg_name: The 4090 Arena
    rpg_description: Battlefield where the hero trains and grows stronger

  vault_synology:
    id: vault_synology
    name: The Deep Vault
    type: archive
    description: Synology NAS - long-term storage
    host: 192.168.x.x
    base_path: /volume1/training_archive
    paths:
      backups: backups/
      datasets: datasets/
      campaign_logs: logs/
      soul_anchors: checkpoints/
    is_local: false
    rpg_name: The Deep Vault
    rpg_description: Grand archive beneath the Inn where soul anchors and ancient tomes are kept

  wizard_study:
    id: wizard_study
    name: Wizard's Study
    type: laboratory
    description: LM Studio on workstation - experiments
    host: localhost
    base_path: /path/to/training
    paths:
      experiments: experiments/
      scratch: scratch/
    is_local: true
    rpg_name: Wizard's Study
    rpg_description: Personal lab for testing new spellbooks

default_facility: arena_4090
```

## 3.6 Create tests/guild/test_facilities.py

```python
# tests/guild/test_facilities.py
"""Tests for facility resolution."""

import pytest
import tempfile
import os
from pathlib import Path

from guild.facilities.resolver import PathResolver, init_resolver, resolve
from guild.facilities.types import Facility, FacilityType


class TestPathResolver:
    @pytest.fixture
    def temp_config(self):
        config = """
facilities:
  test_arena:
    id: test_arena
    name: Test Arena
    type: battlefield
    base_path: /tmp/test_training
    paths:
      checkpoints: checkpoints/
      logs: logs/

  test_hub:
    id: test_hub
    name: Test Hub
    type: hub
    base_path: /tmp/test_hub
    paths:
      status: status/

default_facility: test_arena
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            yield f.name
        os.unlink(f.name)

    def test_resolve_facility_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("facility:test_arena:checkpoints")
        assert path == Path("/tmp/test_training/checkpoints/")

    def test_resolve_facility_with_subpath(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("facility:test_arena:checkpoints")
        # Can also do manual join
        full_path = path / "checkpoint-1000"
        assert str(full_path) == "/tmp/test_training/checkpoints/checkpoint-1000"

    def test_resolve_shorthand(self, temp_config):
        resolver = PathResolver(temp_config)

        # Uses default facility
        path = resolver.resolve("@checkpoints")
        assert path == Path("/tmp/test_training/checkpoints/")

    def test_resolve_with_current_facility(self, temp_config):
        resolver = PathResolver(temp_config)
        resolver.set_current_facility("test_hub")

        path = resolver.resolve("@status")
        assert path == Path("/tmp/test_hub/status/")

    def test_resolve_regular_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("/absolute/path")
        assert path == Path("/absolute/path")

        path = resolver.resolve("./relative")
        assert path == Path("./relative")

    def test_resolve_env_var(self, temp_config):
        os.environ["TEST_BASE"] = "/custom/path"

        config = """
facilities:
  env_test:
    id: env_test
    name: Env Test
    type: battlefield
    base_path: ${TEST_BASE}
    paths:
      data: data/
default_facility: env_test
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config)
            config_path = f.name

        try:
            resolver = PathResolver(config_path)
            path = resolver.resolve("@data")
            assert path == Path("/custom/path/data/")
        finally:
            os.unlink(config_path)

    def test_list_facilities(self, temp_config):
        resolver = PathResolver(temp_config)

        all_facilities = resolver.list_facilities()
        assert "test_arena" in all_facilities
        assert "test_hub" in all_facilities

        battlefields = resolver.list_facilities(FacilityType.BATTLEFIELD)
        assert battlefields == ["test_arena"]
```

## Checkpoint 3

```bash
pytest tests/guild/test_facilities.py -v
```

- [ ] Path resolution works
- [ ] Environment variables expand
- [ ] @ shorthand works
- [ ] Facility listing works
- [ ] Backward compatible paths.py works

---

# Phase 4: Skills Registry

## Goal
Create central skill registry loaded from YAML configs.

## 4.1 Create guild/skills/registry.py

```python
# guild/skills/registry.py
"""Skill registry - central management of skill definitions."""

from typing import Optional
from guild.skills.types import SkillConfig, SkillCategory, MetricDefinition
from guild.config.loader import ConfigLoader, get_config_dir


class SkillRegistry:
    """
    Central registry for skill definitions.

    Loads skills from configs/skills/*.yaml
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self._skills: dict[str, SkillConfig] = {}
        self._metrics: dict[str, MetricDefinition] = {}
        self._loader = config_loader or ConfigLoader()

        # Register built-in metrics
        self._register_builtin_metrics()

    def _register_builtin_metrics(self):
        """Register common metrics."""
        builtins = [
            MetricDefinition("accuracy", "Accuracy", "Fraction of correct responses"),
            MetricDefinition("word_accuracy", "Word Accuracy", "Fraction of correct words"),
            MetricDefinition("json_validity", "JSON Validity", "Whether output is valid JSON"),
            MetricDefinition("rouge_l", "ROUGE-L", "Longest common subsequence score"),
            MetricDefinition("loss", "Loss", "Training loss", higher_is_better=False),
        ]
        for metric in builtins:
            self._metrics[metric.id] = metric

    def load_all(self):
        """Load all skill configs from the config directory."""
        configs = self._loader.load_all("skills", SkillConfig)
        for skill_id, config in configs.items():
            self._skills[skill_id] = config

    def register(self, skill: SkillConfig):
        """Register a skill directly."""
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Optional[SkillConfig]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def get_or_raise(self, skill_id: str) -> SkillConfig:
        """Get a skill by ID, raising if not found."""
        skill = self.get(skill_id)
        if skill is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        return skill

    def list(self, category: Optional[SkillCategory] = None,
             tag: Optional[str] = None) -> list[SkillConfig]:
        """List skills, optionally filtered."""
        skills = list(self._skills.values())

        if category:
            skills = [s for s in skills if s.category == category]

        if tag:
            skills = [s for s in skills if tag in s.tags]

        return skills

    def list_ids(self) -> list[str]:
        """List all skill IDs."""
        return list(self._skills.keys())

    def get_metric(self, metric_id: str) -> Optional[MetricDefinition]:
        """Get a metric definition."""
        return self._metrics.get(metric_id)

    def register_metric(self, metric: MetricDefinition):
        """Register a metric definition."""
        self._metrics[metric.id] = metric

    @property
    def count(self) -> int:
        """Number of registered skills."""
        return len(self._skills)


# Global registry instance
_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry, loading if needed."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _registry.load_all()
    return _registry


def get_skill(skill_id: str) -> Optional[SkillConfig]:
    """Get a skill from the global registry."""
    return get_skill_registry().get(skill_id)


def list_skills(category: Optional[SkillCategory] = None,
                tag: Optional[str] = None) -> list[SkillConfig]:
    """List skills from the global registry."""
    return get_skill_registry().list(category, tag)
```

## 4.2 Create tests/guild/test_skills.py

```python
# tests/guild/test_skills.py
"""Tests for skill registry."""

import pytest
import tempfile
from pathlib import Path

from guild.skills.registry import SkillRegistry, get_skill_registry
from guild.skills.types import SkillConfig, SkillCategory
from guild.config.loader import ConfigLoader, set_config_dir


class TestSkillRegistry:
    @pytest.fixture
    def temp_skills_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create test skills
            logic_yaml = """
id: logic_weaving
name: Logic Weaving
description: Deductive reasoning
category: reasoning
tags:
  - reasoning
  - syllo
metrics:
  - accuracy
  - word_accuracy
primary_metric: accuracy
accuracy_thresholds:
  1: 0.6
  2: 0.7
  3: 0.8
"""
            (skills_dir / "logic_weaving.yaml").write_text(logic_yaml)

            oath_yaml = """
id: oath_binding
name: Oath Binding
description: Following instructions
category: instruction
tags:
  - instruction
metrics:
  - accuracy
primary_metric: accuracy
accuracy_thresholds:
  1: 0.55
  2: 0.65
"""
            (skills_dir / "oath_binding.yaml").write_text(oath_yaml)

            set_config_dir(tmpdir)
            yield tmpdir

    def test_load_skills(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        assert registry.count == 2
        assert "logic_weaving" in registry.list_ids()
        assert "oath_binding" in registry.list_ids()

    def test_get_skill(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        skill = registry.get("logic_weaving")
        assert skill is not None
        assert skill.name == "Logic Weaving"
        assert skill.category == SkillCategory.REASONING

    def test_filter_by_category(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        reasoning = registry.list(category=SkillCategory.REASONING)
        assert len(reasoning) == 1
        assert reasoning[0].id == "logic_weaving"

    def test_filter_by_tag(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        syllo = registry.list(tag="syllo")
        assert len(syllo) == 1
        assert syllo[0].id == "logic_weaving"

    def test_threshold_lookup(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        skill = registry.get("logic_weaving")
        assert skill.get_threshold(1) == 0.6
        assert skill.get_threshold(3) == 0.8

    def test_builtin_metrics(self, temp_skills_dir):
        registry = SkillRegistry()

        accuracy = registry.get_metric("accuracy")
        assert accuracy is not None
        assert accuracy.higher_is_better is True

        loss = registry.get_metric("loss")
        assert loss is not None
        assert loss.higher_is_better is False
```

## Checkpoint 4

```bash
pytest tests/guild/test_skills.py -v
```

- [ ] Skills load from YAML
- [ ] Filtering by category works
- [ ] Filtering by tag works
- [ ] Threshold lookup works
- [ ] Built-in metrics exist

---

# Continuing...

This implementation plan continues for phases 5-12. Due to length, I'll summarize the remaining phases:

## Phase 5: Quest System
- QuestRegistry loading templates from YAML
- QuestForge for generating instances
- QuestBoard wrapping queue/ directory
- Integration with training loop

## Phase 6: Progression Engine
- ProgressionEngine class
- XP calculation from quest results
- Level eligibility checking
- Effect tracker for debuffs
- State persistence to status/hero_state.json

## Phase 7: Combat Calculator
- ResultCalculator with SYLLO-specific logic
- Generic fallback for other task types
- Stance management
- XP award calculation

## Phase 8: Incidents System
- IncidentDetector with rule evaluation
- IncidentTracker for logging
- Integration with training loop
- RPG name generation

## Phase 9: Runs System
- RunManager for unified execution
- RunConfig loading
- State persistence
- Integration with existing daemon

## Phase 10: View Layer
- TavernView mappings from LORE.md
- AdventureLogFormatter
- Technical view (passthrough)
- UI components

## Phase 11: Integration & Migration
- Wire all systems together
- Migrate existing code
- Deprecation warnings
- Full integration tests

## Phase 12: Open Source Prep
- Remove personal data
- CI/CD setup
- Documentation
- License

---

# Phase 13: World Consistency Checker

## Goal
Use the RPG world model as a validation layer. If something can't be explained in world terms, it's either a system bug or a lore gap that needs fixing.

## 13.1 The Principle

```
Every system event MUST map to a world concept.
If it doesn't map, we have one of:
  1. A SYSTEM BUG - the system is doing something it shouldn't
  2. A LORE GAP - the world model is missing a concept
  3. A DESIGN FLAW - the system design violates our mental model
```

This makes the RPG layer more than aesthetic - it's a **semantic validation layer**.

## 13.2 Create guild/consistency/types.py

```python
# guild/consistency/types.py
"""World consistency types."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class InconsistencyType(Enum):
    """Types of world inconsistencies."""
    UNMAPPED_EVENT = "unmapped_event"       # Event has no world equivalent
    RULE_VIOLATION = "rule_violation"       # Violates stated world rules
    STATE_PARADOX = "state_paradox"         # State doesn't make sense in world
    MISSING_CONCEPT = "missing_concept"     # Needs new world concept
    TERMINOLOGY_CONFLICT = "term_conflict"  # Two things using same term


class Resolution(Enum):
    """How to resolve an inconsistency."""
    FIX_SYSTEM = "fix_system"       # The system is wrong
    UPDATE_LORE = "update_lore"     # The lore needs expansion
    ADD_EXCEPTION = "add_exception" # Document as known exception
    IGNORE = "ignore"               # Not worth fixing


@dataclass
class WorldRule:
    """A rule about how the world works."""
    id: str
    category: str          # "progression", "combat", "facilities", etc.
    description: str
    validation_fn: Optional[str] = None  # Function name to check rule

    # Examples
    valid_examples: list[str] = field(default_factory=list)
    invalid_examples: list[str] = field(default_factory=list)


@dataclass
class Inconsistency:
    """A detected world inconsistency."""
    id: str
    type: InconsistencyType
    description: str

    # What triggered it
    trigger_event: str
    trigger_data: dict = field(default_factory=dict)

    # Context
    detected_at: datetime = field(default_factory=datetime.now)
    rule_id: Optional[str] = None  # Which rule was violated

    # Resolution
    resolution: Optional[Resolution] = None
    resolution_notes: str = ""
    resolved_at: Optional[datetime] = None

    # Suggested fixes
    suggested_system_fix: str = ""
    suggested_lore_addition: str = ""


@dataclass
class WorldMapping:
    """Mapping from technical concept to world concept."""
    technical_term: str
    world_term: str
    category: str
    description: str

    # Validation
    bidirectional: bool = True  # Can map both ways?
    examples: list[str] = field(default_factory=list)
```

## 13.3 Create guild/consistency/checker.py

```python
# guild/consistency/checker.py
"""World consistency checker."""

import logging
from typing import Any, Optional, Callable
from datetime import datetime

from guild.consistency.types import (
    WorldRule, Inconsistency, InconsistencyType, WorldMapping
)
from guild.types import generate_id

logger = logging.getLogger(__name__)


class WorldConsistencyChecker:
    """
    Validates system events against the world model.

    Every significant system event should pass through here.
    If it can't be mapped to world terms, it's flagged.
    """

    def __init__(self):
        self._rules: dict[str, WorldRule] = {}
        self._mappings: dict[str, WorldMapping] = {}
        self._inconsistencies: list[Inconsistency] = []
        self._validators: dict[str, Callable] = {}

    def register_rule(self, rule: WorldRule):
        """Register a world rule."""
        self._rules[rule.id] = rule

    def register_mapping(self, mapping: WorldMapping):
        """Register a term mapping."""
        self._mappings[mapping.technical_term] = mapping

    def register_validator(self, rule_id: str, fn: Callable[[Any], bool]):
        """Register a validation function for a rule."""
        self._validators[rule_id] = fn

    def map_to_world(self, technical_term: str) -> Optional[str]:
        """Map a technical term to its world equivalent."""
        mapping = self._mappings.get(technical_term)
        return mapping.world_term if mapping else None

    def can_map(self, technical_term: str) -> bool:
        """Check if a technical term has a world mapping."""
        return technical_term in self._mappings

    def validate_event(self, event_type: str, event_data: dict) -> Optional[Inconsistency]:
        """
        Validate a system event against world rules.

        Returns an Inconsistency if something is wrong, None if OK.
        """
        # Check if event type is mapped
        if not self.can_map(event_type):
            inc = Inconsistency(
                id=generate_id("inc"),
                type=InconsistencyType.UNMAPPED_EVENT,
                description=f"Event type '{event_type}' has no world mapping",
                trigger_event=event_type,
                trigger_data=event_data,
                suggested_lore_addition=f"Add mapping for '{event_type}' to LORE.md"
            )
            self._inconsistencies.append(inc)
            logger.warning(f"World inconsistency: {inc.description}")
            return inc

        # Check relevant rules
        for rule_id, validator in self._validators.items():
            rule = self._rules.get(rule_id)
            if rule and rule.category in event_data.get("categories", [event_type]):
                try:
                    if not validator(event_data):
                        inc = Inconsistency(
                            id=generate_id("inc"),
                            type=InconsistencyType.RULE_VIOLATION,
                            description=f"Event violates rule: {rule.description}",
                            trigger_event=event_type,
                            trigger_data=event_data,
                            rule_id=rule_id
                        )
                        self._inconsistencies.append(inc)
                        logger.warning(f"Rule violation: {inc.description}")
                        return inc
                except Exception as e:
                    logger.error(f"Validator {rule_id} failed: {e}")

        return None

    def validate_state(self, state_type: str, state: dict) -> list[Inconsistency]:
        """
        Validate a system state for internal consistency.

        Returns list of inconsistencies found.
        """
        inconsistencies = []

        # Example: Hero can't be in two regions
        if state_type == "hero_state":
            regions = state.get("active_regions", [])
            if len(regions) > 1:
                inc = Inconsistency(
                    id=generate_id("inc"),
                    type=InconsistencyType.STATE_PARADOX,
                    description="Hero in multiple regions simultaneously",
                    trigger_event="state_check",
                    trigger_data=state,
                    suggested_system_fix="Hero should only be in one region"
                )
                inconsistencies.append(inc)

            # Hero can't have negative XP
            for skill_id, skill_state in state.get("skills", {}).items():
                if skill_state.get("xp_total", 0) < 0:
                    inc = Inconsistency(
                        id=generate_id("inc"),
                        type=InconsistencyType.STATE_PARADOX,
                        description=f"Skill {skill_id} has negative XP",
                        trigger_event="state_check",
                        trigger_data=state
                    )
                    inconsistencies.append(inc)

        self._inconsistencies.extend(inconsistencies)
        return inconsistencies

    def get_unmapped_terms(self) -> list[str]:
        """Get list of technical terms that need world mappings."""
        # This would be populated by scanning codebase
        # For now, return terms from logged inconsistencies
        unmapped = set()
        for inc in self._inconsistencies:
            if inc.type == InconsistencyType.UNMAPPED_EVENT:
                unmapped.add(inc.trigger_event)
        return list(unmapped)

    def get_pending_inconsistencies(self) -> list[Inconsistency]:
        """Get unresolved inconsistencies."""
        return [i for i in self._inconsistencies if i.resolution is None]

    def resolve(self, inconsistency_id: str, resolution: "Resolution",
                notes: str = ""):
        """Mark an inconsistency as resolved."""
        for inc in self._inconsistencies:
            if inc.id == inconsistency_id:
                inc.resolution = resolution
                inc.resolution_notes = notes
                inc.resolved_at = datetime.now()
                break

    def export_for_lore_update(self) -> dict:
        """
        Export inconsistencies that suggest lore updates.

        Returns dict suitable for updating LORE.md
        """
        suggestions = {
            "new_mappings": [],
            "new_rules": [],
            "clarifications": []
        }

        for inc in self.get_pending_inconsistencies():
            if inc.type == InconsistencyType.UNMAPPED_EVENT:
                suggestions["new_mappings"].append({
                    "technical": inc.trigger_event,
                    "suggested_world": inc.suggested_lore_addition
                })
            elif inc.type == InconsistencyType.MISSING_CONCEPT:
                suggestions["new_rules"].append({
                    "description": inc.description,
                    "context": inc.trigger_data
                })

        return suggestions


# Default world rules based on LORE.md
DEFAULT_RULES = [
    WorldRule(
        id="hero_single_region",
        category="progression",
        description="Hero can only be in one region at a time"
    ),
    WorldRule(
        id="xp_non_negative",
        category="progression",
        description="XP cannot be negative"
    ),
    WorldRule(
        id="level_progression",
        category="progression",
        description="Levels must increase sequentially (1->2->3, not 1->3)"
    ),
    WorldRule(
        id="quest_requires_hero",
        category="quests",
        description="A quest cannot be attempted without an active hero"
    ),
    WorldRule(
        id="training_in_arena",
        category="facilities",
        description="Training (heavy computation) happens in Arena, not Inn"
    ),
    WorldRule(
        id="checkpoints_are_soul_anchors",
        category="hero",
        description="Every checkpoint is a Soul Anchor - it preserves hero state"
    ),
    WorldRule(
        id="debuffs_have_causes",
        category="effects",
        description="Every debuff must have a documented cause"
    ),
]


# Default mappings from LORE.md
DEFAULT_MAPPINGS = [
    WorldMapping("training_step", "Quest Attempt", "training",
                 "Each training step is the hero attempting a quest"),
    WorldMapping("checkpoint", "Soul Anchor", "hero",
                 "Checkpoints preserve the hero's state"),
    WorldMapping("loss", "Distance from Mastery", "combat",
                 "Lower loss = closer to mastery"),
    WorldMapping("accuracy", "Quest Success Rate", "progression",
                 "Accuracy is how often the hero succeeds"),
    WorldMapping("overfitting", "Tunnel Vision", "effects",
                 "Overfitting is the Tunnel Vision debuff"),
    WorldMapping("oom", "Exhaustion", "effects",
                 "Out of memory is the Exhaustion debuff"),
    WorldMapping("nan_loss", "Reality Tear", "incidents",
                 "NaN loss is a Reality Tear incident"),
    WorldMapping("batch_size", "Party Size", "training",
                 "Batch size is how many quests attempted together"),
    WorldMapping("learning_rate", "Training Intensity", "training",
                 "Learning rate is how intensely the hero trains"),
    WorldMapping("epoch", "Campaign Cycle", "training",
                 "One epoch is one full cycle through all quests"),
    WorldMapping("validation", "Trial", "evaluation",
                 "Validation is a Trial to test readiness"),
    WorldMapping("inference", "Real Combat", "evaluation",
                 "Inference is the hero in actual combat"),
]


def create_default_checker() -> WorldConsistencyChecker:
    """Create a checker with default rules and mappings."""
    checker = WorldConsistencyChecker()

    for rule in DEFAULT_RULES:
        checker.register_rule(rule)

    for mapping in DEFAULT_MAPPINGS:
        checker.register_mapping(mapping)

    return checker


# Global checker
_checker: Optional[WorldConsistencyChecker] = None


def get_checker() -> WorldConsistencyChecker:
    """Get the global world consistency checker."""
    global _checker
    if _checker is None:
        _checker = create_default_checker()
    return _checker


def check_event(event_type: str, event_data: dict) -> Optional[Inconsistency]:
    """Check an event against world rules."""
    return get_checker().validate_event(event_type, event_data)


def check_state(state_type: str, state: dict) -> list[Inconsistency]:
    """Check state for consistency."""
    return get_checker().validate_state(state_type, state)


def map_term(technical_term: str) -> Optional[str]:
    """Map a technical term to world term."""
    return get_checker().map_to_world(technical_term)
```

## 13.4 Create guild/consistency/reporter.py

```python
# guild/consistency/reporter.py
"""Report world inconsistencies for fixing."""

from pathlib import Path
from datetime import datetime
from typing import Optional

from guild.consistency.checker import get_checker
from guild.consistency.types import Inconsistency, InconsistencyType, Resolution


def generate_inconsistency_report() -> str:
    """Generate a markdown report of pending inconsistencies."""
    checker = get_checker()
    pending = checker.get_pending_inconsistencies()

    if not pending:
        return "# World Consistency Report\n\nNo inconsistencies detected."

    lines = [
        "# World Consistency Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\nPending Inconsistencies: {len(pending)}",
        "\n---\n"
    ]

    # Group by type
    by_type: dict[InconsistencyType, list[Inconsistency]] = {}
    for inc in pending:
        by_type.setdefault(inc.type, []).append(inc)

    for inc_type, incs in by_type.items():
        lines.append(f"## {inc_type.value.replace('_', ' ').title()}\n")

        for inc in incs:
            lines.append(f"### {inc.id}")
            lines.append(f"**Description:** {inc.description}")
            lines.append(f"**Trigger:** {inc.trigger_event}")

            if inc.suggested_system_fix:
                lines.append(f"**System Fix:** {inc.suggested_system_fix}")
            if inc.suggested_lore_addition:
                lines.append(f"**Lore Addition:** {inc.suggested_lore_addition}")

            lines.append("")

    return "\n".join(lines)


def generate_lore_suggestions() -> str:
    """Generate suggested additions to LORE.md."""
    checker = get_checker()
    suggestions = checker.export_for_lore_update()

    if not any(suggestions.values()):
        return ""

    lines = [
        "# Suggested LORE.md Updates",
        f"\nGenerated: {datetime.now().isoformat()}",
        "\n---\n"
    ]

    if suggestions["new_mappings"]:
        lines.append("## New Term Mappings Needed\n")
        lines.append("Add these to the Quick Reference section:\n")
        lines.append("```")
        for m in suggestions["new_mappings"]:
            lines.append(f"{m['technical']:30} → ???")
        lines.append("```\n")

    if suggestions["new_rules"]:
        lines.append("## New Rules/Concepts Needed\n")
        for r in suggestions["new_rules"]:
            lines.append(f"- {r['description']}")
        lines.append("")

    return "\n".join(lines)


def save_report(output_dir: Path | str = "status"):
    """Save consistency report to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    report = generate_inconsistency_report()
    (output_dir / "world_consistency_report.md").write_text(report)

    suggestions = generate_lore_suggestions()
    if suggestions:
        (output_dir / "lore_suggestions.md").write_text(suggestions)
```

## 13.5 Integration Points

Add consistency checks at key system points:

```python
# In training loop (core/train.py or equivalent)
from guild.consistency.checker import check_event, check_state

# Before each training step
check_event("training_step", {
    "step": current_step,
    "batch_size": batch_size,
    "quest_id": current_quest.id
})

# After state changes
check_state("hero_state", hero_state.to_dict())

# On anomalies
if loss != loss:  # NaN check
    check_event("nan_loss", {"step": current_step, "last_loss": prev_loss})
```

## 13.6 World Rules from LORE.md

Extract and codify all rules from LORE.md:

```yaml
# configs/consistency/world_rules.yaml
rules:
  # Progression Rules
  - id: hero_single_location
    category: progression
    description: Hero can only be in one region at a time
    validation: hero.active_regions.length <= 1

  - id: xp_monotonic
    category: progression
    description: XP only increases (never decreases)
    validation: new_xp >= old_xp

  - id: level_requires_threshold
    category: progression
    description: Level-up requires meeting accuracy threshold
    validation: accuracy >= skill.get_threshold(level)

  - id: level_requires_trial
    category: progression
    description: Level-up requires passing a Promotion Trial
    validation: trial_passed == true

  # Quest Rules
  - id: quest_from_template
    category: quests
    description: Every quest instance comes from a template
    validation: quest.template_id in templates

  - id: quest_has_skills
    category: quests
    description: Every quest trains at least one skill
    validation: quest.skills.length >= 1

  # Combat Rules
  - id: crit_is_perfect
    category: combat
    description: CRITICAL HIT means perfect match + perfect format
    validation: result == CRIT implies exact_match and valid_format

  - id: miss_gives_xp
    category: combat
    description: Even MISS gives some XP (learning from failure)
    validation: result == MISS implies xp > 0

  # Facility Rules
  - id: training_in_arena
    category: facilities
    description: Training runs happen on battlefield facilities
    validation: run.facility.type == BATTLEFIELD

  - id: storage_in_vault
    category: facilities
    description: Long-term storage goes to archive facilities
    validation: backup.facility.type == ARCHIVE

  # Effect Rules
  - id: debuff_has_cause
    category: effects
    description: Every debuff must have a documented cause
    validation: effect.cause != null

  - id: debuff_clears_eventually
    category: effects
    description: Debuffs should have cure conditions
    validation: effect.cure_condition != null or effect.duration != null
```

## 13.7 Consistency Dashboard Card

Add to monitoring UI:

```html
<!-- In master_dashboard.html -->
<div class="card" id="world-consistency-card">
  <h3>World Consistency</h3>
  <div class="consistency-status">
    <span class="status-icon" id="consistency-icon">✓</span>
    <span class="status-text" id="consistency-text">All systems nominal</span>
  </div>
  <div class="inconsistency-count" id="inconsistency-count" style="display:none">
    <span class="count">0</span> unmapped events
  </div>
  <div class="latest-inconsistency" id="latest-inconsistency" style="display:none">
    <!-- Shows most recent inconsistency -->
  </div>
</div>
```

## Checkpoint 13

```bash
pytest tests/guild/test_consistency.py -v
```

- [ ] Rules load from config
- [ ] Mappings work bidirectionally
- [ ] Events are validated
- [ ] States are validated
- [ ] Inconsistencies are logged
- [ ] Report generation works
- [ ] Dashboard shows status

---

# Execution Checklist

## Before Starting
- [ ] Read entire plan
- [ ] Understand dependencies between phases
- [ ] Set up test environment

## Per Phase
- [ ] Create files in order listed
- [ ] Run tests after each file
- [ ] Commit after each checkpoint passes
- [ ] Update CHANGELOG.md

## After Completing
- [ ] Run full test suite
- [ ] Verify existing functionality unchanged
- [ ] Update CLAUDE.md with new architecture
- [ ] Create PR with summary

---

**Ready to implement? Start with Phase 0: Pre-Implementation Setup.**
