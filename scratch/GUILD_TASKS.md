# Guild Refactor - Task Breakdown

**Created:** 2025-11-26
**Purpose:** Self-contained tasks that, when completed, deliver the full Guild refactor

---

## How to Use This Document

1. Tasks are numbered: `P0.1`, `P1.3`, etc. (Phase.Task)
2. Each task has: Description, Files, Dependencies, Acceptance Criteria
3. Work through tasks in order within each phase
4. Run acceptance criteria before marking complete
5. Commit after each task (atomic commits)

---

## Task Status Key

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked

---

# Phase 0: Setup & Foundation

**Goal:** Create branch, directories, and base infrastructure

---

### P0.1 - Create Feature Branch

**Description:** Create git branch for the refactor work

**Commands:**
```bash
git checkout -b feature/guild-refactor
git push -u origin feature/guild-refactor
```

**Dependencies:** None

**Acceptance Criteria:**
- [ ] On branch `feature/guild-refactor`
- [ ] Branch pushed to remote

**Effort:** S (5 min)

---

### P0.2 - Create Directory Structure

**Description:** Create all guild/ and configs/ directories

**Commands:**
```bash
cd /path/to/training

# Core guild module
mkdir -p guild/{skills,quests,progression,facilities,runs,incidents,hero,combat,config,consistency}

# Configuration directories
mkdir -p configs/{skills,quests/syllo,quests/discrimination,facilities,progression,regions,heroes,runs,incidents,consistency}

# Views
mkdir -p views/{technical,tavern}

# Tests
mkdir -p tests/guild
```

**Dependencies:** P0.1

**Acceptance Criteria:**
- [ ] `ls guild/` shows 10 subdirectories
- [ ] `ls configs/` shows 9 subdirectories
- [ ] `ls views/` shows 2 subdirectories
- [ ] `ls tests/guild/` exists

**Effort:** S (5 min)

---

### P0.3 - Update .gitignore

**Description:** Add guild-specific ignores for local configs

**File:** `.gitignore` (append)

**Content to Add:**
```
# Guild local configs (user-specific, not committed)
configs/facilities/local.yaml
configs/**/*.local.yaml
*.local.yaml

# Guild runtime state
status/hero_state.json
status/run_*.json
status/incidents/

# Consistency reports (generated)
status/world_consistency_report.md
status/lore_suggestions.md
```

**Dependencies:** P0.1

**Acceptance Criteria:**
- [ ] `.gitignore` contains guild entries
- [ ] `git status` doesn't show configs/facilities/local.yaml after creating it

**Effort:** S (5 min)

---

### P0.4 - Update pyproject.toml

**Description:** Add guild dependencies to optional extras

**File:** `pyproject.toml`

**Changes:**
```toml
[project.optional-dependencies]
# ... existing ...
guild = [
    "pyyaml>=6.0",
    "jsonschema>=4.0",
]
```

**Dependencies:** P0.1

**Acceptance Criteria:**
- [ ] `pip install -e ".[guild]"` succeeds
- [ ] `python -c "import yaml; import jsonschema"` works

**Effort:** S (10 min)

---

### P0.5 - Create guild/__init__.py (Stub)

**Description:** Create initial package marker with version

**File:** `guild/__init__.py`

```python
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
- Consistency: World model validation
"""

__version__ = "0.1.0"

# Populated as modules are created
__all__ = []
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] `python -c "import guild; print(guild.__version__)"` prints "0.1.0"

**Effort:** S (5 min)

---

### P0.6 - Create Subpackage __init__.py Files

**Description:** Create empty __init__.py for all subpackages

**Files to Create:**
```
guild/skills/__init__.py
guild/quests/__init__.py
guild/progression/__init__.py
guild/facilities/__init__.py
guild/runs/__init__.py
guild/incidents/__init__.py
guild/hero/__init__.py
guild/combat/__init__.py
guild/config/__init__.py
guild/consistency/__init__.py
views/__init__.py
views/technical/__init__.py
views/tavern/__init__.py
```

**Content:** Empty or simple docstring for each

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] All subpackages importable: `python -c "from guild import skills, quests, progression"`

**Effort:** S (10 min)

---

### P0.7 - Commit Phase 0

**Description:** Commit all Phase 0 work

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 0 - Setup directory structure and package scaffolding"
git tag guild-p0-complete
```

**Dependencies:** P0.1-P0.6

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p0-complete` exists

**Effort:** S (5 min)

---

# Phase 1: Foundation Types

**Goal:** Create all dataclasses with serialization support

---

### P1.1 - Create guild/types.py (Base Types)

**Description:** Common types, enums, and mixins used across guild

**File:** `guild/types.py`

```python
"""Common types used across the guild module."""

from enum import Enum
from typing import TypeVar, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from uuid import uuid4

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


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid4())[:8]
    return f"{prefix}_{uid}" if prefix else uid


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO string."""
    return dt.isoformat() if dt else None


def iso_to_datetime(s: str) -> datetime:
    """Convert ISO string to datetime."""
    return datetime.fromisoformat(s) if s else None


class SerializableMixin:
    """Mixin providing to_dict/from_dict for dataclasses."""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, datetime):
                result[k] = datetime_to_iso(v)
            elif isinstance(v, Enum):
                result[k] = v.value
            elif hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in v
                ]
            elif isinstance(v, dict):
                result[k] = {
                    dk: dv.to_dict() if hasattr(dv, 'to_dict') else dv
                    for dk, dv in v.items()
                }
            else:
                result[k] = v
        return result
```

**Dependencies:** P0.5

**Acceptance Criteria:**
- [ ] `from guild.types import Severity, Status, generate_id` works
- [ ] `generate_id("test")` returns string like "test_a1b2c3d4"
- [ ] `Severity.HIGH.value == "high"`

**Effort:** S (15 min)

---

### P1.2 - Create guild/skills/types.py

**Description:** Skill configuration and state types

**File:** `guild/skills/types.py`

```python
"""Skill/Discipline type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from collections import deque

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

    # Rolling accuracy - stored as list for serialization
    _recent_results: list[bool] = field(default_factory=list)
    window_size: int = 100

    # Trial state
    eligible_for_trial: bool = False
    last_trial_step: Optional[int] = None
    consecutive_trial_failures: int = 0

    @property
    def recent_results(self) -> list[bool]:
        return self._recent_results

    @property
    def accuracy(self) -> float:
        if not self._recent_results:
            return 0.0
        return sum(self._recent_results) / len(self._recent_results)

    @property
    def xp_since_last_level(self) -> float:
        if self.level not in self.xp_marks:
            return self.xp_total
        return self.xp_total - self.xp_marks.get(self.level, 0)

    def record_result(self, success: bool):
        """Record a quest result."""
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
        return {
            "skill_id": self.skill_id,
            "level": self.level,
            "xp_total": self.xp_total,
            "xp_marks": self.xp_marks,
            "recent_results": self._recent_results,
            "window_size": self.window_size,
            "eligible_for_trial": self.eligible_for_trial,
            "last_trial_step": self.last_trial_step,
            "consecutive_trial_failures": self.consecutive_trial_failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillState":
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
```

**Dependencies:** P1.1

**Acceptance Criteria:**
- [ ] `from guild.skills.types import SkillConfig, SkillState, SkillCategory` works
- [ ] SkillState can serialize: `json.dumps(state.to_dict())` succeeds
- [ ] SkillState can deserialize: `SkillState.from_dict(data)` works
- [ ] `state.accuracy` computes correctly

**Effort:** M (30 min)

---

### P1.3 - Create guild/quests/types.py

**Description:** Quest template, instance, and result types

**File:** `guild/quests/types.py`

```python
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
    id: str
    name: str
    description: str

    skills: list[str]
    regions: list[str]
    difficulty: QuestDifficulty
    difficulty_level: int  # 1-10

    generator_id: str
    generator_params: dict = field(default_factory=dict)

    evaluator_id: str
    evaluator_params: dict = field(default_factory=dict)

    # skill_id -> base XP
    base_xp: dict[str, int] = field(default_factory=dict)

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
        return cls(
            id=data["id"],
            template_id=data["template_id"],
            skills=data["skills"],
            difficulty=QuestDifficulty(data["difficulty"]),
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
    quest_id: str
    hero_id: str

    response: str
    response_metadata: dict = field(default_factory=dict)

    combat_result: CombatResult
    metrics: dict[str, float] = field(default_factory=dict)
    xp_awarded: dict[str, int] = field(default_factory=dict)

    effects_triggered: list[str] = field(default_factory=list)

    attempted_at: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0

    evaluator_notes: str = ""

    @property
    def success(self) -> bool:
        return self.combat_result in [CombatResult.CRITICAL_HIT, CombatResult.HIT]

    @property
    def total_xp(self) -> int:
        return sum(self.xp_awarded.values())

    def to_dict(self) -> dict:
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
```

**Dependencies:** P1.1

**Acceptance Criteria:**
- [ ] All quest types importable
- [ ] `QuestDifficulty.from_level(5)` returns `GOLD`
- [ ] `QuestInstance.create()` works with template
- [ ] `QuestResult.success` property works
- [ ] Serialization round-trips correctly

**Effort:** M (30 min)

---

### P1.4 - Create guild/facilities/types.py

**Description:** Facility and resource types

**File:** `guild/facilities/types.py`

```python
"""Facility (hardware) type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import os


class FacilityType(Enum):
    """Types of facilities."""
    HUB = "hub"
    BATTLEFIELD = "battlefield"
    ARCHIVE = "archive"
    OUTPOST = "outpost"
    LABORATORY = "laboratory"


@dataclass
class FacilityResource:
    """A specific resource within a facility."""
    id: str
    type: str  # "gpu", "storage", "network"
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FacilityResource":
        return cls(
            id=data["id"],
            type=data["type"],
            properties=data.get("properties", {}),
        )


@dataclass
class Facility:
    """
    A hardware location in the system.
    Loaded from configs/facilities/*.yaml
    """
    id: str
    name: str
    type: FacilityType
    description: str = ""

    host: str = "localhost"
    port: Optional[int] = None

    base_path: str = ""
    paths: dict[str, str] = field(default_factory=dict)

    resources: list[FacilityResource] = field(default_factory=list)

    is_local: bool = True
    is_available: bool = True

    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

    def get_path(self, key: str, subpath: str = "") -> str:
        """Get a resolved path within this facility."""
        base = os.path.expandvars(os.path.expanduser(self.base_path))
        if key in self.paths:
            path = os.path.join(base, self.paths[key])
        else:
            path = os.path.join(base, key)
        if subpath:
            path = os.path.join(path, subpath)
        return path

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "host": self.host,
            "port": self.port,
            "base_path": self.base_path,
            "paths": self.paths,
            "resources": [r.to_dict() for r in self.resources],
            "is_local": self.is_local,
            "is_available": self.is_available,
            "rpg_name": self.rpg_name,
            "rpg_description": self.rpg_description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Facility":
        resources = [FacilityResource.from_dict(r) for r in data.get("resources", [])]
        ftype = data.get("type")
        if isinstance(ftype, str):
            ftype = FacilityType(ftype)
        return cls(
            id=data["id"],
            name=data["name"],
            type=ftype,
            description=data.get("description", ""),
            host=data.get("host", "localhost"),
            port=data.get("port"),
            base_path=data.get("base_path", ""),
            paths=data.get("paths", {}),
            resources=resources,
            is_local=data.get("is_local", True),
            is_available=data.get("is_available", True),
            rpg_name=data.get("rpg_name"),
            rpg_description=data.get("rpg_description"),
        )
```

**Dependencies:** P1.1

**Acceptance Criteria:**
- [ ] `from guild.facilities.types import Facility, FacilityType` works
- [ ] `facility.get_path("checkpoints")` resolves correctly
- [ ] Environment variables in base_path expand
- [ ] Round-trip serialization works

**Effort:** M (25 min)

---

### P1.5 - Create guild/progression/types.py

**Description:** Status effects, hero identity, and hero state types

**File:** `guild/progression/types.py`

```python
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
        if self.duration_steps is None:
            return False
        return (current_step - self.applied_at_step) >= self.duration_steps

    def to_dict(self) -> dict:
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
    """Runtime state for an effect rule."""
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
        return cls(**data)


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
        if skill_id not in self.skills:
            self.skills[skill_id] = SkillState(skill_id=skill_id)
        return self.skills[skill_id]

    def add_effect(self, effect: StatusEffect):
        self.active_effects = [e for e in self.active_effects if e.id != effect.id]
        self.active_effects.append(effect)

    def remove_effect(self, effect_id: str):
        self.active_effects = [e for e in self.active_effects if e.id != effect_id]

    def clear_expired_effects(self, current_step: int):
        self.active_effects = [e for e in self.active_effects
                               if not e.is_expired(current_step)]

    def to_dict(self) -> dict:
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
```

**Dependencies:** P1.1, P1.2

**Acceptance Criteria:**
- [ ] All progression types importable
- [ ] `StatusEffect.is_expired()` works correctly
- [ ] `HeroState.health` property computes correctly
- [ ] Full round-trip serialization works

**Effort:** L (45 min)

---

### P1.6 - Create guild/runs/types.py

**Description:** Run configuration and state types

**File:** `guild/runs/types.py`

```python
"""Run type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from guild.types import Status, datetime_to_iso, iso_to_datetime


class RunType(Enum):
    """Types of runs."""
    TRAINING = "training"
    EVALUATION = "evaluation"
    AUDIT = "audit"
    EXPERIMENT = "experiment"
    GENERATION = "generation"


@dataclass
class RunConfig:
    """Configuration for a run."""
    id: str
    type: RunType
    name: str = ""
    description: str = ""

    facility_id: str = ""
    hero_id: str = ""

    quest_filters: dict = field(default_factory=dict)

    max_steps: Optional[int] = None
    max_quests: Optional[int] = None
    max_duration_seconds: Optional[int] = None

    hyperparams: dict = field(default_factory=dict)

    log_level: str = "INFO"
    log_facility_id: str = ""

    checkpoint_every_steps: int = 1000
    checkpoint_facility_id: str = ""

    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "facility_id": self.facility_id,
            "hero_id": self.hero_id,
            "quest_filters": self.quest_filters,
            "max_steps": self.max_steps,
            "max_quests": self.max_quests,
            "max_duration_seconds": self.max_duration_seconds,
            "hyperparams": self.hyperparams,
            "log_level": self.log_level,
            "log_facility_id": self.log_facility_id,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "checkpoint_facility_id": self.checkpoint_facility_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(
            id=data["id"],
            type=RunType(data["type"]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            facility_id=data.get("facility_id", ""),
            hero_id=data.get("hero_id", ""),
            quest_filters=data.get("quest_filters", {}),
            max_steps=data.get("max_steps"),
            max_quests=data.get("max_quests"),
            max_duration_seconds=data.get("max_duration_seconds"),
            hyperparams=data.get("hyperparams", {}),
            log_level=data.get("log_level", "INFO"),
            log_facility_id=data.get("log_facility_id", ""),
            checkpoint_every_steps=data.get("checkpoint_every_steps", 1000),
            checkpoint_facility_id=data.get("checkpoint_facility_id", ""),
            tags=data.get("tags", []),
        )


@dataclass
class RunState:
    """Current state of a run."""
    run_id: str
    config: RunConfig
    status: Status = Status.PENDING

    current_step: int = 0
    quests_completed: int = 0
    quests_succeeded: int = 0

    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    metrics: dict[str, Any] = field(default_factory=dict)

    last_checkpoint_step: int = 0
    checkpoint_paths: list[str] = field(default_factory=list)

    incident_ids: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        if self.quests_completed == 0:
            return 0.0
        return self.quests_succeeded / self.quests_completed

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "current_step": self.current_step,
            "quests_completed": self.quests_completed,
            "quests_succeeded": self.quests_succeeded,
            "started_at": datetime_to_iso(self.started_at),
            "paused_at": datetime_to_iso(self.paused_at),
            "completed_at": datetime_to_iso(self.completed_at),
            "metrics": self.metrics,
            "last_checkpoint_step": self.last_checkpoint_step,
            "checkpoint_paths": self.checkpoint_paths,
            "incident_ids": self.incident_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunState":
        return cls(
            run_id=data["run_id"],
            config=RunConfig.from_dict(data["config"]),
            status=Status(data["status"]),
            current_step=data.get("current_step", 0),
            quests_completed=data.get("quests_completed", 0),
            quests_succeeded=data.get("quests_succeeded", 0),
            started_at=iso_to_datetime(data.get("started_at")),
            paused_at=iso_to_datetime(data.get("paused_at")),
            completed_at=iso_to_datetime(data.get("completed_at")),
            metrics=data.get("metrics", {}),
            last_checkpoint_step=data.get("last_checkpoint_step", 0),
            checkpoint_paths=data.get("checkpoint_paths", []),
            incident_ids=data.get("incident_ids", []),
        )
```

**Dependencies:** P1.1

**Acceptance Criteria:**
- [ ] All run types importable
- [ ] `RunState.success_rate` computes correctly
- [ ] Round-trip serialization works

**Effort:** M (25 min)

---

### P1.7 - Create guild/incidents/types.py

**Description:** Incident tracking types

**File:** `guild/incidents/types.py`

```python
"""Incident type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from guild.types import Severity, datetime_to_iso, iso_to_datetime


class IncidentCategory(Enum):
    """Categories of incidents."""
    DATA = "data"
    TRAINING = "training"
    INFRA = "infra"
    LOGIC = "logic"


class IncidentStatus(Enum):
    """Incident lifecycle."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    WONTFIX = "wontfix"


@dataclass
class Incident:
    """A detected problem/bug."""
    id: str
    category: IncidentCategory
    severity: Severity

    title: str
    description: str

    detected_at_step: int
    detected_at_time: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    quest_id: Optional[str] = None
    facility_id: Optional[str] = None

    context: dict = field(default_factory=dict)

    status: IncidentStatus = IncidentStatus.OPEN
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    rpg_name: Optional[str] = None
    rpg_location: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "detected_at_step": self.detected_at_step,
            "detected_at_time": datetime_to_iso(self.detected_at_time),
            "run_id": self.run_id,
            "quest_id": self.quest_id,
            "facility_id": self.facility_id,
            "context": self.context,
            "status": self.status.value,
            "resolution": self.resolution,
            "resolved_at": datetime_to_iso(self.resolved_at),
            "rpg_name": self.rpg_name,
            "rpg_location": self.rpg_location,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Incident":
        return cls(
            id=data["id"],
            category=IncidentCategory(data["category"]),
            severity=Severity(data["severity"]),
            title=data["title"],
            description=data["description"],
            detected_at_step=data["detected_at_step"],
            detected_at_time=iso_to_datetime(data.get("detected_at_time")) or datetime.now(),
            run_id=data.get("run_id"),
            quest_id=data.get("quest_id"),
            facility_id=data.get("facility_id"),
            context=data.get("context", {}),
            status=IncidentStatus(data.get("status", "open")),
            resolution=data.get("resolution"),
            resolved_at=iso_to_datetime(data.get("resolved_at")),
            rpg_name=data.get("rpg_name"),
            rpg_location=data.get("rpg_location"),
        )


@dataclass
class IncidentRule:
    """Rule for detecting incidents."""
    id: str
    name: str
    category: IncidentCategory
    severity: Severity

    detector_type: str
    detector_config: dict = field(default_factory=dict)

    title_template: str = ""
    description_template: str = ""

    rpg_name_template: Optional[str] = None
```

**Dependencies:** P1.1

**Acceptance Criteria:**
- [ ] All incident types importable
- [ ] Round-trip serialization works

**Effort:** S (20 min)

---

### P1.8 - Create guild/combat/types.py

**Description:** Combat result and stance types

**File:** `guild/combat/types.py`

```python
"""Combat system type definitions."""

from dataclasses import dataclass, field
from enum import Enum

from guild.quests.types import CombatResult


class CombatStance(Enum):
    """Combat stances (protocol modes)."""
    THOUGHTFUL = "thoughtful"   # Emoji thinking
    QUICK_DRAW = "quick_draw"   # Direct mode
    ALTERNATING = "alternating" # 50/50


@dataclass
class CombatConfig:
    """Combat system configuration."""
    xp_crit: int = 15
    xp_hit: int = 10
    xp_glancing: int = 5
    xp_miss: int = 2
    xp_crit_miss: int = 0

    difficulty_multipliers: dict[int, float] = field(default_factory=lambda: {
        1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.5,
        6: 1.7, 7: 2.0, 8: 2.3, 9: 2.6, 10: 3.0
    })

    default_stance: CombatStance = CombatStance.ALTERNATING

    crit_miss_debuff_threshold: int = 3
    miss_debuff_threshold: int = 5

    def get_base_xp(self, result: CombatResult) -> int:
        return {
            CombatResult.CRITICAL_HIT: self.xp_crit,
            CombatResult.HIT: self.xp_hit,
            CombatResult.GLANCING: self.xp_glancing,
            CombatResult.MISS: self.xp_miss,
            CombatResult.CRITICAL_MISS: self.xp_crit_miss,
        }.get(result, 0)

    def get_difficulty_multiplier(self, level: int) -> float:
        return self.difficulty_multipliers.get(level, 1.0)


@dataclass
class StanceConfig:
    """Configuration for combat stances."""
    thinking_emojis: list[str] = field(default_factory=lambda: [
        "🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"
    ])
    stop_emojis: list[str] = field(default_factory=lambda: [
        "🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"
    ])

    min_thinking_count: int = 1
    max_thinking_count: int = 10
    min_stop_count: int = 2
    max_stop_count: int = 4
```

**Dependencies:** P1.3

**Acceptance Criteria:**
- [ ] All combat types importable
- [ ] `CombatConfig.get_base_xp()` works
- [ ] `CombatConfig.get_difficulty_multiplier()` works

**Effort:** S (15 min)

---

### P1.9 - Create guild/hero/types.py (Re-export)

**Description:** Convenience re-export of hero types

**File:** `guild/hero/types.py`

```python
"""Hero type definitions (re-export from progression)."""

from guild.progression.types import HeroIdentity, HeroState

__all__ = ["HeroIdentity", "HeroState"]
```

**Dependencies:** P1.5

**Acceptance Criteria:**
- [ ] `from guild.hero.types import HeroIdentity, HeroState` works

**Effort:** S (5 min)

---

### P1.10 - Create tests/guild/test_types.py

**Description:** Unit tests for all type definitions

**File:** `tests/guild/test_types.py`

```python
"""Tests for guild type definitions."""

import pytest
import json
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
        assert id1 != id2
        assert len(id1) == 13  # "test_" + 8 chars


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

    def test_skill_state_serialization(self):
        state = SkillState(skill_id="test", level=3, xp_total=1500.0)
        state.record_result(True)
        state.record_result(False)

        # Serialize
        data = state.to_dict()
        json_str = json.dumps(data)  # Should not raise

        # Deserialize
        loaded = SkillState.from_dict(json.loads(json_str))
        assert loaded.skill_id == "test"
        assert loaded.level == 3
        assert loaded.xp_total == 1500.0
        assert len(loaded.recent_results) == 2


class TestQuestTypes:
    def test_quest_difficulty_from_level(self):
        assert QuestDifficulty.from_level(1) == QuestDifficulty.BRONZE
        assert QuestDifficulty.from_level(5) == QuestDifficulty.GOLD
        assert QuestDifficulty.from_level(10) == QuestDifficulty.DRAGON

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

    def test_quest_instance_serialization(self):
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

        instance = QuestInstance.create(template, prompt="Solve this...")
        data = instance.to_dict()
        loaded = QuestInstance.from_dict(data)

        assert loaded.id == instance.id
        assert loaded.template_id == "syllo_basic"
        assert loaded.difficulty == QuestDifficulty.BRONZE


class TestFacilityTypes:
    def test_facility_get_path(self):
        facility = Facility(
            id="arena",
            name="Arena",
            type=FacilityType.BATTLEFIELD,
            base_path="/tmp/test",
            paths={"checkpoints": "models/"}
        )

        assert facility.get_path("checkpoints") == "/tmp/test/models/"
        assert facility.get_path("checkpoints", "step-1000") == "/tmp/test/models/step-1000"
        assert facility.get_path("other") == "/tmp/test/other"


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
            id="hero1", name="Test Hero",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )

        state = HeroState(hero_id="hero1", identity=identity)
        assert state.health == "healthy"

        effect = StatusEffect(
            id="nan_dragon", name="NaN Dragon",
            description="Training collapsed",
            type=EffectType.DEBUFF,
            severity=Severity.CRITICAL,
            applied_at_step=0
        )
        state.add_effect(effect)
        assert state.health == "wounded"

    def test_hero_state_serialization(self):
        identity = HeroIdentity(
            id="hero1", name="Test Hero",
            architecture="qwen", generation="3",
            size="0.6B", variant="base"
        )

        state = HeroState(hero_id="hero1", identity=identity)
        state.get_skill("logic").record_result(True)
        state.total_quests = 100

        data = state.to_dict()
        json_str = json.dumps(data)

        loaded = HeroState.from_dict(json.loads(json_str))
        assert loaded.hero_id == "hero1"
        assert loaded.total_quests == 100
        assert "logic" in loaded.skills


class TestRunTypes:
    def test_run_state_success_rate(self):
        config = RunConfig(id="r1", type=RunType.TRAINING)
        state = RunState(run_id="r1", config=config)

        state.quests_completed = 100
        state.quests_succeeded = 75
        assert state.success_rate == 0.75


class TestIncidentTypes:
    def test_incident_serialization(self):
        incident = Incident(
            id="inc1",
            category=IncidentCategory.TRAINING,
            severity=Severity.CRITICAL,
            title="NaN Loss",
            description="Loss became NaN",
            detected_at_step=1000
        )

        data = incident.to_dict()
        loaded = Incident.from_dict(data)

        assert loaded.id == "inc1"
        assert loaded.category == IncidentCategory.TRAINING
        assert loaded.severity == Severity.CRITICAL
```

**Dependencies:** P1.1-P1.9

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_types.py -v` passes all tests
- [ ] No import errors

**Effort:** M (30 min)

---

### P1.11 - Commit Phase 1

**Description:** Commit all Phase 1 type definitions

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 1 - Foundation types with serialization

- guild/types.py: Base enums, SerializableMixin, generate_id
- guild/skills/types.py: SkillConfig, SkillState
- guild/quests/types.py: QuestTemplate, QuestInstance, QuestResult
- guild/facilities/types.py: Facility, FacilityResource
- guild/progression/types.py: StatusEffect, HeroIdentity, HeroState
- guild/runs/types.py: RunConfig, RunState
- guild/incidents/types.py: Incident, IncidentRule
- guild/combat/types.py: CombatConfig, StanceConfig
- tests/guild/test_types.py: Comprehensive unit tests

All types support JSON serialization via to_dict/from_dict"
git tag guild-p1-complete
```

**Dependencies:** P1.1-P1.10

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p1-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 2: Configuration System

**Goal:** YAML loading with environment variable support and validation

---

### P2.1 - Create guild/config/loader.py

**Description:** YAML configuration loader with env var expansion

**File:** `guild/config/loader.py`

```python
"""YAML configuration loader with environment variable support."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Optional, TypeVar, Type
from dataclasses import fields, is_dataclass
from enum import Enum

T = TypeVar('T')

_config_dir: Optional[Path] = None


def set_config_dir(path: str | Path):
    """Set the global config directory."""
    global _config_dir
    _config_dir = Path(path)


def get_config_dir() -> Path:
    """Get the config directory."""
    global _config_dir
    if _config_dir is None:
        env_path = os.environ.get("GUILD_CONFIG_DIR")
        if env_path:
            _config_dir = Path(env_path)
        else:
            # Default: configs/ relative to project root
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
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # Return original if no value and no default
            return match.group(0)

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

    field_info = {f.name: f for f in fields(cls)}
    filtered = {}

    for key, value in data.items():
        if key not in field_info:
            continue

        field_type = field_info[key].type

        # Handle Enum conversion
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            if isinstance(value, str):
                value = field_type(value)

        # Handle nested dataclass
        elif is_dataclass(field_type) and isinstance(value, dict):
            value = dict_to_dataclass(value, field_type)

        filtered[key] = value

    return cls(**filtered)


def load_config(category: str, name: str, cls: Optional[Type[T]] = None) -> T | dict:
    """Load a config file and optionally convert to dataclass."""
    path = get_config_path(category, name)
    data = load_yaml(path)

    if cls is not None:
        return dict_to_dataclass(data, cls)
    return data


def load_all_configs(category: str, cls: Optional[Type[T]] = None,
                     pattern: str = "*.yaml") -> dict[str, T | dict]:
    """Load all config files in a category."""
    category_dir = get_config_dir() / category
    if not category_dir.exists():
        return {}

    configs = {}
    for path in category_dir.glob(pattern):
        if path.name.startswith("_"):
            continue
        name = path.stem
        try:
            configs[name] = load_config(category, name, cls)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")

    return configs


class ConfigLoader:
    """Manages loading and caching of configurations."""

    def __init__(self, config_dir: Optional[Path | str] = None):
        self.config_dir = Path(config_dir) if config_dir else get_config_dir()
        self._cache: dict[str, Any] = {}

    def load(self, category: str, name: str, cls: Optional[Type[T]] = None,
             use_cache: bool = True) -> T | dict:
        """Load a config with optional caching."""
        cache_key = f"{category}/{name}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.config_dir / category / f"{name}.yaml"
        data = load_yaml(path)

        result = dict_to_dataclass(data, cls) if cls else data

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

**Dependencies:** P0.4, P1.11

**Acceptance Criteria:**
- [ ] `from guild.config.loader import load_yaml, load_config, ConfigLoader` works
- [ ] Environment variable expansion works: `${VAR:-default}`
- [ ] YAML files load correctly

**Effort:** M (30 min)

---

### P2.2 - Update guild/config/__init__.py

**Description:** Export config loader functions

**File:** `guild/config/__init__.py`

```python
"""Configuration loading and validation."""

from guild.config.loader import (
    load_config,
    load_all_configs,
    load_yaml,
    get_config_path,
    get_config_dir,
    set_config_dir,
    ConfigLoader,
    dict_to_dataclass,
    expand_env_vars,
)

__all__ = [
    "load_config",
    "load_all_configs",
    "load_yaml",
    "get_config_path",
    "get_config_dir",
    "set_config_dir",
    "ConfigLoader",
    "dict_to_dataclass",
    "expand_env_vars",
]
```

**Dependencies:** P2.1

**Acceptance Criteria:**
- [ ] `from guild.config import load_config, ConfigLoader` works

**Effort:** S (5 min)

---

### P2.3 - Create configs/skills/logic_weaving.yaml

**Description:** First skill configuration file

**File:** `configs/skills/logic_weaving.yaml`

```yaml
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

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File exists and is valid YAML
- [ ] `yaml.safe_load(open('configs/skills/logic_weaving.yaml'))` works

**Effort:** S (10 min)

---

### P2.4 - Create configs/skills/oath_binding.yaml

**Description:** Second skill configuration file

**File:** `configs/skills/oath_binding.yaml`

```yaml
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

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File exists and is valid YAML

**Effort:** S (10 min)

---

### P2.5 - Create configs/facilities/example.yaml

**Description:** Example facility configuration for open source users

**File:** `configs/facilities/example.yaml`

```yaml
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

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File exists and is valid YAML
- [ ] Contains at least 3 facilities

**Effort:** S (15 min)

---

### P2.6 - Create configs/facilities/local.yaml (Your Setup)

**Description:** Your actual facility configuration (gitignored)

**File:** `configs/facilities/local.yaml`

```yaml
# Local facility configuration - GITIGNORED
# Your actual hardware setup

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
    rpg_description: Grand archive beneath the Inn

  wizard_study:
    id: wizard_study
    name: Wizard's Study
    type: laboratory
    description: LM Studio experiments
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

**Dependencies:** P0.2, P0.3

**Acceptance Criteria:**
- [ ] File exists and is valid YAML
- [ ] File is gitignored: `git status` doesn't show it after creation

**Effort:** S (15 min)

---

### P2.7 - Create configs/progression/effects.yaml

**Description:** Status effect definitions

**File:** `configs/progression/effects.yaml`

```yaml
# Status effect definitions

effects:
  tunnel_vision:
    id: tunnel_vision
    name: Tunnel Vision
    description: Overfitting - high training accuracy, low validation accuracy
    type: debuff
    severity: medium
    default_duration_steps: null
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
      The hero's mind is clouded, leading to erratic behavior.

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
      A dark curse causing the hero to repeat words endlessly.

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
      A catastrophic rift in training reality.

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

  - id: detect_nan
    effect_id: reality_tear
    trigger_type: metric_threshold
    trigger_config:
      metric: loss
      op: is_nan
    cooldown_steps: 0
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File exists and is valid YAML
- [ ] Contains at least 5 effects

**Effort:** M (20 min)

---

### P2.8 - Create tests/guild/test_config.py

**Description:** Tests for configuration loading

**File:** `tests/guild/test_config.py`

```python
"""Tests for configuration loading."""

import pytest
import tempfile
import os
from pathlib import Path

from guild.config.loader import (
    load_yaml, expand_env_vars, dict_to_dataclass,
    load_config, ConfigLoader, set_config_dir, get_config_dir
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
            "nested": {"path": "${TEST_PATH}/subdir"}
        }
        result = expand_env_vars(data)
        assert result["base"] == "/custom/path"
        assert result["nested"]["path"] == "/custom/path/subdir"

    def test_list_expansion(self):
        os.environ["TEST_ITEM"] = "expanded"
        data = ["${TEST_ITEM}", "static"]
        result = expand_env_vars(data)
        assert result == ["expanded", "static"]


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

    def test_extra_fields_ignored(self):
        data = {
            "id": "test",
            "name": "Test",
            "description": "Test",
            "category": "reasoning",
            "unknown_field": "ignored"
        }
        result = dict_to_dataclass(data, SkillConfig)
        assert result.id == "test"


class TestConfigLoader:
    @pytest.fixture
    def temp_config_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

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

            old_dir = get_config_dir()
            set_config_dir(tmpdir)
            yield tmpdir
            set_config_dir(old_dir)

    def test_load_skill_config(self, temp_config_dir):
        config = load_config("skills", "test_skill", SkillConfig)
        assert config.id == "test_skill"
        assert config.name == "Test Skill"
        assert config.category == SkillCategory.REASONING
        assert config.get_threshold(1) == 0.6

    def test_config_loader_caching(self, temp_config_dir):
        loader = ConfigLoader(temp_config_dir)

        config1 = loader.load("skills", "test_skill", SkillConfig)
        config2 = loader.load("skills", "test_skill", SkillConfig)

        assert config1 is config2  # Same object due to caching

        loader.clear_cache()
        config3 = loader.load("skills", "test_skill", SkillConfig)
        assert config3 is not config1

    def test_load_raw_dict(self, temp_config_dir):
        data = load_config("skills", "test_skill")
        assert isinstance(data, dict)
        assert data["id"] == "test_skill"
```

**Dependencies:** P2.1, P2.2

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_config.py -v` passes

**Effort:** M (25 min)

---

### P2.9 - Commit Phase 2

**Description:** Commit configuration system

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 2 - Configuration system with YAML loading

- guild/config/loader.py: YAML loader with env var expansion
- configs/skills/: logic_weaving.yaml, oath_binding.yaml
- configs/facilities/: example.yaml, local.yaml
- configs/progression/: effects.yaml
- tests/guild/test_config.py: Config loading tests

Supports \${VAR:-default} syntax for environment variables"
git tag guild-p2-complete
```

**Dependencies:** P2.1-P2.8

**Acceptance Criteria:**
- [ ] Clean git status (local.yaml should not appear)
- [ ] Tag `guild-p2-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 3: Facilities & Path Resolution

**Goal:** Replace hardcoded paths with facility-based resolution

---

### P3.1 - Create guild/facilities/resolver.py

**Description:** Path resolver using facility configurations

**File:** `guild/facilities/resolver.py`

```python
"""Path resolution using facility configurations."""

import os
from pathlib import Path
from typing import Optional

from guild.facilities.types import Facility, FacilityType
from guild.config.loader import load_yaml


class PathResolver:
    """
    Resolves logical paths to physical paths using facility configs.

    Path formats:
    - "facility:arena_4090:checkpoints" -> resolved path
    - "facility:arena_4090:checkpoints/step-1000" -> with subpath
    - "@checkpoints" -> current facility's checkpoints
    - "@checkpoints/step-1000" -> with subpath
    - "~/path" -> home expansion
    - "/absolute" -> unchanged
    - "relative" -> relative to cwd
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
            facility = Facility.from_dict({"id": fac_id, **fac_data})
            self._facilities[fac_id] = facility

    def add_facility(self, facility: Facility):
        """Add a facility directly."""
        self._facilities[facility.id] = facility

    def resolve(self, path_spec: str) -> Path:
        """Resolve a path specification to a physical path."""
        if path_spec.startswith("facility:"):
            parts = path_spec.split(":", 2)
            if len(parts) < 3:
                raise ValueError(f"Invalid facility path: {path_spec}")
            _, facility_id, subpath = parts
            return self._resolve_facility_path(facility_id, subpath)

        elif path_spec.startswith("@"):
            key = path_spec[1:]
            facility_id = self._current_facility or self._default_facility
            if not facility_id:
                raise ValueError("No current or default facility set")
            return self._resolve_facility_path(facility_id, key)

        elif path_spec.startswith("~"):
            return Path(path_spec).expanduser()

        else:
            expanded = os.path.expandvars(path_spec)
            return Path(expanded)

    def _resolve_facility_path(self, facility_id: str, path_key: str) -> Path:
        """Resolve a path within a facility."""
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")

        facility = self._facilities[facility_id]
        base = Path(os.path.expandvars(facility.base_path)).expanduser()

        # Split path_key into alias and subpath
        if "/" in path_key:
            parts = path_key.split("/", 1)
            alias, subpath = parts[0], parts[1]
        else:
            alias, subpath = path_key, ""

        # Resolve alias
        if alias in facility.paths:
            resolved = base / facility.paths[alias]
        else:
            resolved = base / alias

        # Add subpath
        if subpath:
            resolved = resolved / subpath

        return resolved

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


# Global resolver
_resolver: Optional[PathResolver] = None


def init_resolver(config_path: Optional[str | Path] = None) -> PathResolver:
    """Initialize the global path resolver."""
    global _resolver

    if config_path is None:
        config_path = os.environ.get("GUILD_FACILITIES_CONFIG")

        if not config_path:
            from guild.config.loader import get_config_dir
            local_path = get_config_dir() / "facilities" / "local.yaml"
            example_path = get_config_dir() / "facilities" / "example.yaml"

            if local_path.exists():
                config_path = local_path
            elif example_path.exists():
                config_path = example_path
            else:
                raise FileNotFoundError(
                    "No facility config found. Create configs/facilities/local.yaml"
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
    """Get a facility by ID."""
    return get_resolver().get_facility(facility_id)


def set_current_facility(facility_id: str):
    """Set the current facility."""
    get_resolver().set_current_facility(facility_id)
```

**Dependencies:** P1.4, P2.1

**Acceptance Criteria:**
- [ ] `from guild.facilities.resolver import resolve, init_resolver` works
- [ ] Path resolution works for all formats
- [ ] Global resolver auto-initializes

**Effort:** M (35 min)

---

### P3.2 - Update guild/facilities/__init__.py

**Description:** Export facility functions

**File:** `guild/facilities/__init__.py`

```python
"""Facility management and path resolution."""

from guild.facilities.types import Facility, FacilityType, FacilityResource
from guild.facilities.resolver import (
    PathResolver,
    init_resolver,
    get_resolver,
    resolve,
    get_facility,
    set_current_facility,
)

__all__ = [
    "Facility",
    "FacilityType",
    "FacilityResource",
    "PathResolver",
    "init_resolver",
    "get_resolver",
    "resolve",
    "get_facility",
    "set_current_facility",
]
```

**Dependencies:** P3.1

**Acceptance Criteria:**
- [ ] `from guild.facilities import resolve, Facility` works

**Effort:** S (5 min)

---

### P3.3 - Update guild/__init__.py (Add Facilities)

**Description:** Export facility functions from main package

**File:** `guild/__init__.py` (update)

```python
"""
Guild Trainer - A generic framework for LLM training with RPG-style progression.
"""

__version__ = "0.1.0"

from guild.facilities.resolver import (
    init_resolver,
    resolve,
    get_facility,
    set_current_facility,
    get_resolver,
)

__all__ = [
    "init_resolver",
    "resolve",
    "get_facility",
    "set_current_facility",
    "get_resolver",
]
```

**Dependencies:** P3.1, P3.2

**Acceptance Criteria:**
- [ ] `from guild import resolve, init_resolver` works

**Effort:** S (5 min)

---

### P3.4 - Create tests/guild/test_facilities.py

**Description:** Tests for facility resolution

**File:** `tests/guild/test_facilities.py`

```python
"""Tests for facility resolution."""

import pytest
import tempfile
import os
from pathlib import Path

from guild.facilities.resolver import PathResolver, init_resolver
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

        path = resolver.resolve("facility:test_arena:checkpoints/step-1000")
        assert path == Path("/tmp/test_training/checkpoints/step-1000")

    def test_resolve_shorthand(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("@checkpoints")
        assert path == Path("/tmp/test_training/checkpoints/")

    def test_resolve_shorthand_with_subpath(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("@checkpoints/step-1000")
        assert path == Path("/tmp/test_training/checkpoints/step-1000")

    def test_resolve_with_current_facility(self, temp_config):
        resolver = PathResolver(temp_config)
        resolver.set_current_facility("test_hub")

        path = resolver.resolve("@status")
        assert path == Path("/tmp/test_hub/status/")

    def test_resolve_regular_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("/absolute/path")
        assert path == Path("/absolute/path")

    def test_resolve_home_path(self, temp_config):
        resolver = PathResolver(temp_config)

        path = resolver.resolve("~/some/path")
        assert str(path).startswith(os.path.expanduser("~"))

    def test_resolve_env_var(self):
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

    def test_get_facility(self, temp_config):
        resolver = PathResolver(temp_config)

        facility = resolver.get_facility("test_arena")
        assert facility.name == "Test Arena"
        assert facility.type == FacilityType.BATTLEFIELD

    def test_unknown_facility_raises(self, temp_config):
        resolver = PathResolver(temp_config)

        with pytest.raises(ValueError, match="Unknown facility"):
            resolver.resolve("facility:nonexistent:path")
```

**Dependencies:** P3.1

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_facilities.py -v` passes

**Effort:** M (25 min)

---

### P3.5 - Create Backward-Compatible core/paths.py Wrapper

**Description:** Update existing paths.py to use guild resolver

**File:** `core/paths.py` (replace existing)

```python
"""
Path utilities with backward compatibility.

This module uses the Guild facility resolver under the hood,
but maintains the same API for backward compatibility.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_initialized = False
_base_dir: Optional[Path] = None


def _ensure_initialized():
    """Ensure path resolution is ready."""
    global _initialized, _base_dir

    if _initialized:
        return

    try:
        from guild.facilities.resolver import init_resolver, get_resolver

        try:
            init_resolver()
            resolver = get_resolver()
            facility = resolver.get_facility(resolver.current_facility_id)
            _base_dir = Path(facility.base_path).expanduser()
            logger.info(f"Guild resolver initialized, base_dir: {_base_dir}")
        except FileNotFoundError:
            _base_dir = _detect_base_dir_legacy()
            logger.info(f"Using legacy path detection, base_dir: {_base_dir}")

        _initialized = True

    except ImportError:
        _base_dir = _detect_base_dir_legacy()
        _initialized = True
        logger.info(f"Guild not available, using legacy detection: {_base_dir}")


def _detect_base_dir_legacy() -> Path:
    """Legacy base directory detection."""
    env_path = os.environ.get("GUILD_BASE_DIR") or os.environ.get("TRAINING_BASE_DIR")
    if env_path:
        return Path(env_path)

    current = Path(__file__).resolve()
    markers = ["config.json", "CLAUDE.md", "pyproject.toml"]

    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    return Path.cwd()


def get_base_dir() -> Path:
    """Get the base directory for the training system."""
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
    """
    _ensure_initialized()

    if path_spec.startswith("facility:") or path_spec.startswith("@"):
        try:
            from guild.facilities.resolver import resolve
            return resolve(path_spec)
        except ImportError:
            pass

    if path_spec.startswith("/") or path_spec.startswith("~"):
        return Path(path_spec).expanduser()

    return _base_dir / path_spec


# Convenience functions
def get_status_dir() -> Path:
    return resolve_path("status")


def get_logs_dir() -> Path:
    return resolve_path("logs")


def get_queue_dir() -> Path:
    return resolve_path("queue")


def get_checkpoints_dir() -> Path:
    return resolve_path("current_model")


def get_models_dir() -> Path:
    return resolve_path("models")


def get_config_path() -> Path:
    return resolve_path("config.json")


def get_inbox_dir() -> Path:
    return resolve_path("inbox")


def get_data_dir() -> Path:
    return resolve_path("data")
```

**Dependencies:** P3.1, existing core/paths.py

**Acceptance Criteria:**
- [ ] `from core.paths import get_base_dir, resolve_path` works
- [ ] Backward compatible with existing code
- [ ] Falls back to legacy if guild not configured
- [ ] Existing tests still pass

**Effort:** M (30 min)

---

### P3.6 - Test Backward Compatibility

**Description:** Verify existing code still works with new paths.py

**Commands:**
```bash
# Run existing tests
pytest tests/ -v -k "not guild"

# Test specific imports that use paths
python -c "from core.paths import get_base_dir; print(get_base_dir())"
python -c "from core.training_daemon import *; print('daemon imports ok')"
```

**Dependencies:** P3.5

**Acceptance Criteria:**
- [ ] Existing tests pass
- [ ] `get_base_dir()` returns correct path
- [ ] No import errors in existing code

**Effort:** M (20 min)

---

### P3.7 - Commit Phase 3

**Description:** Commit facility resolution system

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 3 - Facility-based path resolution

- guild/facilities/resolver.py: PathResolver with @shorthand support
- core/paths.py: Backward-compatible wrapper using guild resolver
- tests/guild/test_facilities.py: Path resolution tests

Path formats: facility:id:path, @alias, @alias/subpath
Falls back to legacy detection if guild not configured"
git tag guild-p3-complete
```

**Dependencies:** P3.1-P3.6

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p3-complete` exists
- [ ] All tests pass (both guild and existing)

**Effort:** S (5 min)

---

# Checkpoint: Validate Phase 0-3

**Before proceeding to Phase 4, validate the foundation:**

```bash
# All guild tests pass
pytest tests/guild/ -v

# All existing tests still pass
pytest tests/ -v

# Imports work
python -c "
from guild import resolve, init_resolver
from guild.skills.types import SkillConfig, SkillState
from guild.quests.types import QuestInstance, QuestResult
from guild.facilities import Facility
from guild.config import load_config, ConfigLoader
from core.paths import get_base_dir
print('All imports successful!')
print(f'Base dir: {get_base_dir()}')
"

# Path resolution works
python -c "
from guild import init_resolver, resolve
init_resolver()
print(f'Checkpoints: {resolve(\"@checkpoints\")}')
print(f'Status: {resolve(\"@status\")}')
"
```

**Decision Point:**
- If all checks pass: Continue to Phase 4
- If issues found: Fix before proceeding
- If path resolution provides value: Consider using it for 1-2 weeks before continuing

---

# Phase 4-13: Summary

The remaining phases follow the same pattern. Here's what each delivers:

| Phase | Deliverable | Key Files |
|-------|-------------|-----------|
| **4** | Skills Registry | `guild/skills/registry.py`, skill loading |
| **5** | Quest System | `guild/quests/forge.py`, `guild/quests/board.py` |
| **6** | Progression Engine | `guild/progression/engine.py`, XP/level logic |
| **7** | Combat Calculator | `guild/combat/calculator.py`, CRIT/HIT/MISS |
| **8** | Incidents System | `guild/incidents/tracker.py`, bug detection |
| **9** | Runs System | `guild/runs/runner.py`, campaign management |
| **10** | View Layer | `views/tavern/`, RPG dashboard |
| **11** | Integration | Wire guild to existing daemon/train.py |
| **12** | Open Source | CI, docs, license, examples |
| **13** | Consistency | `guild/consistency/checker.py`, world validation |

---

**Total Tasks in Phases 0-3:** 25 tasks
**Estimated Time for Phases 0-3:** 1-2 weeks

Shall I continue with the detailed task breakdown for Phases 4-13?
