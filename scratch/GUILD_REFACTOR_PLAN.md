# Guild Refactor Plan

## Master Plan: Generic Abstraction Layer + Open Source Preparation

**Created:** 2025-11-26
**Goal:** Transform scattered scripts into a clean, configurable framework suitable for open-sourcing

---

## Part 1: Vision

### What We Have Now

```
Current State:
├── Hardcoded paths (/path/to/training everywhere)
├── Implicit skills (scattered in generator names, directories, eval configs)
├── Ad-hoc task handling (each generator has its own format)
├── Tightly coupled components (daemon knows about specific file layouts)
├── Special-case logic ("if task == 'syllo' then...")
└── Mixed concerns (business logic + paths + UI + config all intertwined)
```

### What We Want

```
Target State:
├── guild/              # Core engine (generic, reusable)
│   ├── skills/         # Skill registry + metrics
│   ├── quests/         # Quest templates + instances
│   ├── progression/    # XP, levels, status effects
│   ├── facilities/     # Hardware abstraction
│   ├── runs/           # Training/eval/audit runs
│   ├── incidents/      # Error/bug tracking
│   └── hero/           # Model identity + checkpoints
│
├── configs/            # ALL configuration (YAML/JSON)
│   └── (no hardcoded values in code)
│
├── views/              # Presentation layers
│   ├── technical/      # Raw metrics view
│   └── tavern/         # RPG skin (uses LORE.md mappings)
│
└── implementations/    # Concrete implementations
    ├── generators/     # SYLLO, discrimination, etc.
    ├── evaluators/     # Task-specific eval logic
    └── adapters/       # Connect guild/ to existing code
```

### Core Principles

1. **Everything is configurable** - No magic strings, no hardcoded paths
2. **Separation of concerns** - Engine doesn't know about presentation
3. **Plugin architecture** - New skills/quests/facilities = config changes
4. **Backward compatible** - Existing code keeps working during migration
5. **Open-source ready** - Anyone can clone and configure for their setup

---

## Part 2: New Module Structure

### 2.1 `guild/` - The Core Engine

This is the heart of the refactor. Pure abstractions, no hardcoded values.

```
guild/
├── __init__.py
├── types.py                    # Common types (enums, base classes)
│
├── skills/
│   ├── __init__.py
│   ├── types.py                # SkillConfig, SkillState
│   ├── registry.py             # SKILLS dict, loader
│   └── metrics.py              # MetricDefinition, compute helpers
│
├── quests/
│   ├── __init__.py
│   ├── types.py                # QuestTemplate, QuestInstance, QuestResult
│   ├── registry.py             # QUEST_TEMPLATES dict, loader
│   ├── forge.py                # QuestForge - dispatches to generators
│   └── board.py                # QuestBoard - queue abstraction
│
├── progression/
│   ├── __init__.py
│   ├── types.py                # HeroSkillState, ProgressionConfig
│   ├── engine.py               # ProgressionEngine - XP, levels
│   ├── effects.py              # StatusEffect, DebuffRule, EffectTracker
│   └── trials.py               # PromotionTrial logic
│
├── facilities/
│   ├── __init__.py
│   ├── types.py                # Facility, FacilityType
│   ├── registry.py             # FACILITIES dict, loader
│   └── resolver.py             # PathResolver - translates facility:path
│
├── runs/
│   ├── __init__.py
│   ├── types.py                # RunConfig, RunState, RunType
│   ├── runner.py               # RunManager - unified execution
│   └── campaigns.py            # Campaign - long-running sequences
│
├── incidents/
│   ├── __init__.py
│   ├── types.py                # Incident, IncidentCategory, Severity
│   ├── detector.py             # IncidentDetector - rule-based
│   └── tracker.py              # IncidentTracker - log + history
│
├── hero/
│   ├── __init__.py
│   ├── types.py                # HeroIdentity, HeroForm
│   ├── registry.py             # HEROES dict (known models)
│   └── forms.py                # FormManager - checkpoint handling
│
└── combat/
    ├── __init__.py
    ├── types.py                # CombatResult, CombatStance
    ├── calculator.py           # ResultCalculator - CRIT/HIT/MISS logic
    └── stances.py              # StanceManager - protocol modes
```

### 2.2 `configs/` - All Configuration

```
configs/
├── guild.yaml                  # Master config (which sub-configs to load)
│
├── skills/
│   ├── _schema.yaml            # Schema documentation
│   ├── logic_weaving.yaml      # SYLLO skill
│   ├── oath_binding.yaml       # Instruction following
│   ├── arcane_compression.yaml # Summarization
│   └── ...
│
├── quests/
│   ├── _schema.yaml
│   ├── syllo/
│   │   ├── basic.yaml          # L1-L3 templates
│   │   ├── intermediate.yaml   # L4-L6
│   │   └── advanced.yaml       # L7-L10
│   ├── discrimination/
│   │   └── templates.yaml
│   └── ...
│
├── facilities/
│   ├── _schema.yaml
│   ├── default.yaml            # Generic facility definitions
│   ├── example.yaml            # Example user config
│   └── local.yaml              # GITIGNORED - user's actual paths
│
├── progression/
│   ├── thresholds.yaml         # Level requirements
│   ├── xp_curves.yaml          # XP per result type
│   └── effects.yaml            # Debuff definitions + rules
│
├── regions/
│   ├── novice_valley.yaml
│   ├── logic_foothills.yaml
│   ├── reasoning_mountains.yaml
│   └── summit.yaml
│
├── heroes/
│   ├── _schema.yaml
│   └── qwendal_sprite.yaml     # Current hero definition
│
├── runs/
│   ├── training.yaml           # Default training run config
│   ├── evaluation.yaml         # Default eval config
│   └── audit.yaml              # Audit run config
│
└── incidents/
    └── rules.yaml              # Incident detection rules
```

### 2.3 `views/` - Presentation Layers

```
views/
├── __init__.py
├── base.py                     # ViewAdapter base class
│
├── technical/                  # "Guild Master View"
│   ├── __init__.py
│   ├── formatters.py           # Raw metric formatting
│   └── templates/
│       └── ...
│
└── tavern/                     # "Tavern View" (RPG skin)
    ├── __init__.py
    ├── mappings.py             # Technical → RPG name mapping
    ├── formatters.py           # Adventure log, combat results
    ├── narrator.py             # Event narration generator
    └── templates/
        └── ...
```

---

## Part 3: Core Type Definitions

### 3.1 Skills

```python
# guild/skills/types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class SkillCategory(Enum):
    REASONING = "reasoning"
    COMPRESSION = "compression"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    TOOL_USE = "tool_use"

@dataclass
class MetricDefinition:
    id: str                         # "accuracy", "rouge_l", etc.
    name: str                       # Display name
    higher_is_better: bool = True
    range: tuple[float, float] = (0.0, 1.0)

@dataclass
class SkillConfig:
    """Definition of a trainable skill/discipline."""
    id: str                         # "logic_weaving"
    name: str                       # "Logic Weaving"
    description: str
    category: SkillCategory
    tags: list[str] = field(default_factory=list)

    # Metrics
    metrics: list[str] = field(default_factory=list)  # metric IDs
    primary_metric: str = "accuracy"

    # Progression
    accuracy_thresholds: dict[int, float] = field(default_factory=dict)
    # {1: 0.60, 2: 0.65, 3: 0.70, ...}

    # RPG flavor (optional, loaded from LORE if not specified)
    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None

@dataclass
class SkillState:
    """Current state of a skill for a hero."""
    skill_id: str
    level: int = 1
    xp_total: float = 0.0
    xp_marks: dict[int, float] = field(default_factory=dict)
    accuracy_window: list[bool] = field(default_factory=list)
    accuracy_window_size: int = 100

    @property
    def accuracy(self) -> float:
        if not self.accuracy_window:
            return 0.0
        return sum(self.accuracy_window) / len(self.accuracy_window)
```

### 3.2 Quests

```python
# guild/quests/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

class QuestDifficulty(Enum):
    BRONZE = 1
    SILVER = 2
    GOLD = 3
    PLATINUM = 4
    DRAGON = 5

@dataclass
class QuestTemplate:
    """Blueprint for generating quest instances."""
    id: str                         # "syllo_basic_4word"
    name: str                       # "Basic Syllable Puzzle (4 words)"
    description: str

    # Classification
    skills: list[str]               # skill IDs this quest trains
    regions: list[str]              # region IDs where this appears
    difficulty: QuestDifficulty
    difficulty_level: int           # 1-10 granular

    # Generation
    generator_id: str               # "syllo_generator"
    generator_params: dict = field(default_factory=dict)

    # Evaluation
    evaluator_id: str               # "syllo_evaluator"
    evaluator_params: dict = field(default_factory=dict)

    # Rewards
    base_xp: dict[str, int] = field(default_factory=dict)
    # {"logic_weaving": 10, "oath_binding": 5}

@dataclass
class QuestInstance:
    """A concrete quest ready to be attempted."""
    id: str                         # UUID
    template_id: str

    # Inherited from template
    skills: list[str]
    difficulty: QuestDifficulty
    difficulty_level: int

    # Content
    payload: dict[str, Any]         # prompt, context, etc.
    expected: Optional[dict] = None # golden answer, if known
    metadata: dict = field(default_factory=dict)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""                # generator lineage

@dataclass
class QuestResult:
    """Outcome of attempting a quest."""
    quest_id: str
    hero_id: str

    # Response
    response: str
    response_metadata: dict = field(default_factory=dict)

    # Evaluation
    combat_result: str              # "CRIT", "HIT", "GLANCING", "MISS", "CRIT_MISS"
    metrics: dict[str, float] = field(default_factory=dict)
    xp_awarded: dict[str, int] = field(default_factory=dict)

    # Effects
    effects_triggered: list[str] = field(default_factory=list)

    # Timing
    attempted_at: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
```

### 3.3 Facilities

```python
# guild/facilities/types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class FacilityType(Enum):
    HUB = "hub"                     # Central management (Inn)
    BATTLEFIELD = "battlefield"     # Training GPU (Arena)
    ARCHIVE = "archive"             # Long-term storage (Vault)
    OUTPOST = "outpost"             # Satellite workers (Scouts)
    LABORATORY = "laboratory"       # Experiments (Wizard's Study)

@dataclass
class FacilityResource:
    """A specific resource within a facility."""
    id: str                         # "gpu_0", "main_storage"
    type: str                       # "gpu", "storage", "network"
    properties: dict = field(default_factory=dict)
    # GPU: {"vram_gb": 24, "model": "RTX 4090"}
    # Storage: {"capacity_tb": 4, "type": "nvme"}

@dataclass
class Facility:
    """A hardware location in the system."""
    id: str                         # "arena_4090"
    name: str                       # "The 4090 Arena"
    type: FacilityType
    description: str = ""

    # Connection
    host: str = "localhost"         # hostname or IP
    port: Optional[int] = None      # if networked

    # Paths
    base_path: str = ""             # root directory on this facility
    paths: dict[str, str] = field(default_factory=dict)
    # {"checkpoints": "current_model/", "logs": "logs/", ...}

    # Resources
    resources: list[FacilityResource] = field(default_factory=list)

    # Status
    is_local: bool = True
    is_available: bool = True

    # RPG
    rpg_name: Optional[str] = None
    rpg_description: Optional[str] = None
```

### 3.4 Progression

```python
# guild/progression/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class EffectSeverity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class StatusEffect:
    """A buff or debuff affecting the hero."""
    id: str                         # "tunnel_vision"
    name: str                       # "Tunnel Vision"
    description: str
    severity: EffectSeverity

    # When applied
    applied_at_step: int
    applied_at_time: datetime = field(default_factory=datetime.now)
    cause: dict = field(default_factory=dict)

    # Duration
    duration_steps: Optional[int] = None  # None = until cured
    cure_condition: Optional[str] = None  # "accuracy > 0.7 for 50 steps"

    # Effects
    effects: dict = field(default_factory=dict)
    # {"accuracy_penalty": -0.1, "xp_multiplier": 0.8}

@dataclass
class DebuffRule:
    """Rule for automatically applying a status effect."""
    id: str
    effect_id: str                  # which StatusEffect to apply

    # Trigger conditions
    metric: str                     # "accuracy", "loss", "val_gap"
    skill_id: Optional[str] = None  # specific skill, or None for global
    threshold: float = 0.0
    comparison: str = "lt"          # "lt", "gt", "eq"
    window_size: int = 10           # how many observations

    # Cooldown
    cooldown_steps: int = 100

@dataclass
class HeroSkillState:
    """Complete progression state for one skill."""
    skill_id: str
    level: int = 1
    xp_total: float = 0.0
    xp_marks: dict[int, float] = field(default_factory=dict)

    # Recent performance
    recent_outcomes: list[bool] = field(default_factory=list)
    window_size: int = 100

    # Trial eligibility
    eligible_for_trial: bool = False
    last_trial_step: Optional[int] = None
    trial_cooldown: int = 1000

    @property
    def accuracy(self) -> float:
        if not self.recent_outcomes:
            return 0.0
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    @property
    def xp_to_next_level(self) -> float:
        # Computed from progression config
        pass

@dataclass
class HeroState:
    """Complete state of the hero."""
    hero_id: str
    name: str

    # Skills
    skills: dict[str, HeroSkillState] = field(default_factory=dict)

    # Status effects
    active_effects: list[StatusEffect] = field(default_factory=list)

    # Current context
    current_region: str = ""
    current_step: int = 0
    current_run_id: Optional[str] = None

    # Stats
    total_quests: int = 0
    total_xp: float = 0.0

    @property
    def health(self) -> str:
        """Compute health status from effects."""
        severe_count = sum(1 for e in self.active_effects
                         if e.severity in [EffectSeverity.SEVERE, EffectSeverity.CRITICAL])
        if severe_count >= 2:
            return "struggling"
        elif severe_count == 1:
            return "fatigued"
        elif self.active_effects:
            return "minor_issues"
        return "healthy"
```

### 3.5 Runs

```python
# guild/runs/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any

class RunType(Enum):
    TRAINING = "training"           # Campaign
    EVALUATION = "evaluation"       # Trial / Dungeon Run
    AUDIT = "audit"                 # Forensic investigation
    EXPERIMENT = "experiment"       # Wizard's Study work
    GENERATION = "generation"       # Quest Forge work

class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RunConfig:
    """Configuration for a run."""
    id: str                         # UUID
    type: RunType
    name: str = ""
    description: str = ""

    # Where
    facility_id: str = ""           # which facility runs it

    # What
    hero_id: str = ""               # which model/checkpoint
    quest_filters: dict = field(default_factory=dict)
    # {"skills": ["logic_weaving"], "regions": ["novice_valley"], "difficulty_max": 3}

    # How much
    max_steps: Optional[int] = None
    max_quests: Optional[int] = None
    max_duration_seconds: Optional[int] = None

    # Hyperparameters (for training)
    hyperparams: dict = field(default_factory=dict)

    # Logging
    log_level: str = "INFO"
    log_facility: str = ""          # where to write logs

    # Checkpointing
    checkpoint_every_steps: int = 1000
    checkpoint_facility: str = ""

@dataclass
class RunState:
    """Current state of a run."""
    run_id: str
    config: RunConfig
    status: RunStatus = RunStatus.PENDING

    # Progress
    current_step: int = 0
    quests_completed: int = 0

    # Timing
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Checkpoints
    last_checkpoint_step: int = 0
    checkpoint_paths: list[str] = field(default_factory=list)

    # Incidents
    incident_ids: list[str] = field(default_factory=list)
```

### 3.6 Incidents

```python
# guild/incidents/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class IncidentCategory(Enum):
    DATA = "data"                   # Cursed scrolls, format issues
    TRAINING = "training"           # NaN, gradient issues
    INFRASTRUCTURE = "infra"        # OOM, disk, network
    LOGIC = "logic"                 # Code bugs

class IncidentSeverity(Enum):
    LOW = "low"                     # Gremlin
    MEDIUM = "medium"               # Ogre
    HIGH = "high"                   # Dragon
    CRITICAL = "critical"           # Demon Lord

@dataclass
class Incident:
    """A detected problem/bug."""
    id: str                         # UUID
    category: IncidentCategory
    severity: IncidentSeverity

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

    # Resolution
    status: str = "open"            # "open", "investigating", "resolved", "wontfix"
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # RPG (computed from LORE mappings)
    rpg_name: Optional[str] = None  # "The NaN Dragon"
    rpg_location: Optional[str] = None  # "Gradient Caverns"

@dataclass
class IncidentRule:
    """Rule for detecting incidents."""
    id: str
    name: str
    category: IncidentCategory
    severity: IncidentSeverity

    # Detection
    detector_type: str              # "metric_threshold", "pattern_match", "exception"
    detector_config: dict = field(default_factory=dict)

    # Incident template
    title_template: str             # "Loss became {value}"
    description_template: str
```

---

## Part 4: Path Resolution System

### 4.1 The Problem

Currently paths are hardcoded everywhere:
```python
# Bad - scattered throughout codebase
BASE_DIR = "/path/to/training"
checkpoint_dir = os.path.join(BASE_DIR, "current_model")
log_dir = os.path.join(BASE_DIR, "logs")
```

### 4.2 The Solution

```python
# guild/facilities/resolver.py

from pathlib import Path
from typing import Optional
import os

class PathResolver:
    """Resolves logical paths to physical paths using facility configs."""

    def __init__(self, facilities_config: dict):
        self._facilities = {}
        self._load_facilities(facilities_config)
        self._current_facility: Optional[str] = None

    def _load_facilities(self, config: dict):
        for fac_id, fac_data in config.get("facilities", {}).items():
            self._facilities[fac_id] = Facility(**fac_data)

    def resolve(self, path_spec: str) -> Path:
        """
        Resolve a path specification to an actual path.

        Formats:
        - "facility:arena_4090:checkpoints" -> /path/to/arena/checkpoints
        - "facility:inn_3090:logs/training" -> /path/to/inn/logs/training
        - "@checkpoints" -> current facility's checkpoints
        - "./relative/path" -> relative to current working dir
        - "/absolute/path" -> unchanged
        """
        if path_spec.startswith("facility:"):
            parts = path_spec.split(":", 2)
            if len(parts) < 3:
                raise ValueError(f"Invalid facility path: {path_spec}")
            _, facility_id, subpath = parts
            return self._resolve_facility_path(facility_id, subpath)

        elif path_spec.startswith("@"):
            # Shorthand for current facility
            key = path_spec[1:]
            if not self._current_facility:
                raise ValueError("No current facility set")
            return self._resolve_facility_path(self._current_facility, key)

        else:
            # Regular path
            return Path(path_spec).expanduser()

    def _resolve_facility_path(self, facility_id: str, subpath: str) -> Path:
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")

        facility = self._facilities[facility_id]
        base = Path(facility.base_path)

        # Check if subpath is a known alias
        if subpath in facility.paths:
            subpath = facility.paths[subpath]

        return base / subpath

    def set_current_facility(self, facility_id: str):
        if facility_id not in self._facilities:
            raise ValueError(f"Unknown facility: {facility_id}")
        self._current_facility = facility_id

    def get_facility(self, facility_id: str) -> Facility:
        return self._facilities[facility_id]

    def list_facilities(self, type_filter: Optional[FacilityType] = None) -> list[str]:
        if type_filter:
            return [f for f, data in self._facilities.items()
                    if data.type == type_filter]
        return list(self._facilities.keys())


# Global resolver instance
_resolver: Optional[PathResolver] = None

def init_resolver(config_path: str = "configs/facilities/local.yaml"):
    """Initialize the global path resolver."""
    global _resolver
    config = load_yaml(config_path)
    _resolver = PathResolver(config)

def resolve(path_spec: str) -> Path:
    """Resolve a path using the global resolver."""
    if _resolver is None:
        raise RuntimeError("PathResolver not initialized. Call init_resolver() first.")
    return _resolver.resolve(path_spec)

def get_facility(facility_id: str) -> Facility:
    """Get a facility by ID."""
    if _resolver is None:
        raise RuntimeError("PathResolver not initialized.")
    return _resolver.get_facility(facility_id)
```

### 4.3 Example Facility Config

```yaml
# configs/facilities/local.yaml
# GITIGNORED - each user creates their own

facilities:
  inn_3090:
    id: "inn_3090"
    name: "The 3090 Inn"
    type: "hub"
    description: "Central management and inference server"
    host: "192.168.x.x"
    port: 8765
    base_path: "/path/to/training"
    paths:
      status: "status/"
      logs: "logs/"
      queue: "queue/"
      ui: "monitoring/ui/"
    resources:
      - id: "gpu_0"
        type: "gpu"
        properties:
          vram_gb: 24
          model: "RTX 3090"
    rpg_name: "The 3090 Inn"
    rpg_description: "Central hub where heroes rest and quests are managed"

  arena_4090:
    id: "arena_4090"
    name: "The 4090 Arena"
    type: "battlefield"
    description: "Primary training GPU"
    host: "localhost"
    base_path: "/path/to/training"
    paths:
      checkpoints: "current_model/"
      snapshots: "snapshots/"
      models: "models/"
    resources:
      - id: "gpu_0"
        type: "gpu"
        properties:
          vram_gb: 24
          model: "RTX 4090"
    rpg_name: "The 4090 Arena"
    rpg_description: "Battlefield where the hero trains"

  vault_synology:
    id: "vault_synology"
    name: "The Deep Vault"
    type: "archive"
    description: "Long-term storage on Synology NAS"
    host: "192.168.x.x"
    base_path: "/volume1/training_archive"
    is_local: false
    paths:
      backups: "backups/"
      datasets: "datasets/"
      campaign_logs: "logs/"
    rpg_name: "The Deep Vault"
    rpg_description: "Grand archive beneath the Inn"

# Default facility for unqualified paths
default_facility: "arena_4090"
```

### 4.4 Example Default Config (for open source)

```yaml
# configs/facilities/example.yaml
# Example configuration - copy to local.yaml and customize

facilities:
  main_hub:
    id: "main_hub"
    name: "Main Hub"
    type: "hub"
    host: "localhost"
    port: 8765
    base_path: "${GUILD_BASE_DIR:-./}"  # Uses env var or current dir
    paths:
      status: "status/"
      logs: "logs/"
      queue: "queue/"
      ui: "monitoring/ui/"

  training_gpu:
    id: "training_gpu"
    name: "Training GPU"
    type: "battlefield"
    host: "localhost"
    base_path: "${GUILD_BASE_DIR:-./}"
    paths:
      checkpoints: "checkpoints/"
      models: "models/"

  archive:
    id: "archive"
    name: "Archive"
    type: "archive"
    base_path: "${GUILD_ARCHIVE_DIR:-./archive}"
    paths:
      backups: "backups/"
      datasets: "datasets/"

default_facility: "training_gpu"
```

---

## Part 5: Migration Strategy

### Phase 1: Foundation (Week 1)

**Goal:** Create guild/ module with all types, no behavior changes yet.

**Tasks:**
1. Create `guild/` directory structure
2. Implement all dataclasses from Part 3
3. Create `configs/` directory structure
4. Write schema documentation for each config type
5. Create example configs

**Files to create:**
```
guild/__init__.py
guild/types.py
guild/skills/types.py
guild/skills/registry.py
guild/quests/types.py
guild/quests/registry.py
guild/facilities/types.py
guild/facilities/resolver.py
guild/progression/types.py
guild/runs/types.py
guild/incidents/types.py
guild/hero/types.py

configs/guild.yaml
configs/skills/_schema.yaml
configs/facilities/example.yaml
configs/facilities/local.yaml.example
```

**Tests:**
- Type instantiation
- Config loading
- Path resolution

### Phase 2: Facilities & Paths (Week 2)

**Goal:** Replace all hardcoded paths with facility-based resolution.

**Tasks:**
1. Implement PathResolver fully
2. Create local.yaml for current setup
3. Find and replace all hardcoded paths
4. Update core/paths.py to use facility resolver
5. Test all path-dependent code

**Files to modify:**
```
core/paths.py -> wrap with facility resolver
core/training_daemon.py -> use @checkpoints, @logs, etc.
monitoring/api/server.py -> use facility:inn_3090:ui
management/backup_manager.py -> use facility:vault_synology:backups
```

**Backward compatibility:**
```python
# core/paths.py
def get_base_dir() -> Path:
    """Backward compatible - returns arena base path."""
    return resolve("@")  # Current facility base
```

### Phase 3: Skills Registry (Week 3)

**Goal:** Central skill definitions, tag existing components.

**Tasks:**
1. Define all current skills in YAML
2. Implement SkillRegistry loader
3. Tag generators with skill IDs
4. Tag evaluators with skill IDs
5. Update UI to show skill names

**Skill configs to create:**
```yaml
# configs/skills/logic_weaving.yaml
id: "logic_weaving"
name: "Logic Weaving"
description: "Chain deductions, solve puzzles, syllogistic reasoning"
category: "reasoning"
tags: ["reasoning", "deduction", "syllo"]
metrics: ["accuracy", "word_accuracy", "json_validity"]
primary_metric: "accuracy"
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
rpg_name: "Logic Weaving"
rpg_description: "The discipline of chaining deductions and weaving proofs"
```

### Phase 4: Quest System (Week 4)

**Goal:** Unified quest flow through the system.

**Tasks:**
1. Define QuestTemplates for all current task types
2. Implement QuestForge dispatcher
3. Implement QuestBoard (wraps queue/)
4. Modify training loop to consume QuestInstances
5. Implement QuestResult recording

**New components:**
```python
# guild/quests/forge.py
class QuestForge:
    """Generates quest instances from templates."""

    def __init__(self, generators: dict[str, Callable]):
        self._generators = generators

    def forge(self, template: QuestTemplate, count: int = 1) -> list[QuestInstance]:
        generator = self._generators.get(template.generator_id)
        if not generator:
            raise ValueError(f"Unknown generator: {template.generator_id}")

        instances = []
        for _ in range(count):
            payload, expected = generator(**template.generator_params)
            instance = QuestInstance(
                id=str(uuid4()),
                template_id=template.id,
                skills=template.skills,
                difficulty=template.difficulty,
                difficulty_level=template.difficulty_level,
                payload=payload,
                expected=expected,
                source=f"forge:{template.generator_id}"
            )
            instances.append(instance)
        return instances
```

### Phase 5: Progression Engine (Week 5)

**Goal:** XP, levels, and status effects working.

**Tasks:**
1. Implement ProgressionEngine
2. Implement EffectTracker
3. Connect quest results to progression
4. Implement trial eligibility checks
5. Add progression state to status files

**New status file structure:**
```json
{
  "training_status": { ... },
  "hero_state": {
    "hero_id": "qwendal_sprite_iii",
    "skills": {
      "logic_weaving": {
        "level": 3,
        "xp_total": 45000,
        "accuracy": 0.72,
        "eligible_for_trial": true
      }
    },
    "active_effects": [],
    "health": "healthy"
  }
}
```

### Phase 6: View Layer (Week 6)

**Goal:** Tavern View working alongside technical view.

**Tasks:**
1. Create views/tavern/mappings.py from LORE.md
2. Implement adventure log formatter
3. Create tavern_view.html
4. Add view toggle to UI
5. Connect to real data

### Phase 7: Open Source Prep (Week 7)

**Goal:** Repository ready for public release.

**Tasks:**
1. Remove all personal paths (audit configs)
2. Create comprehensive .gitignore
3. Write setup documentation
4. Create GitHub Actions CI
5. Write CONTRIBUTING.md
6. License selection

---

## Part 6: File Changes Inventory

### New Files to Create

```
# Core engine
guild/__init__.py
guild/types.py
guild/skills/__init__.py
guild/skills/types.py
guild/skills/registry.py
guild/skills/metrics.py
guild/quests/__init__.py
guild/quests/types.py
guild/quests/registry.py
guild/quests/forge.py
guild/quests/board.py
guild/progression/__init__.py
guild/progression/types.py
guild/progression/engine.py
guild/progression/effects.py
guild/progression/trials.py
guild/facilities/__init__.py
guild/facilities/types.py
guild/facilities/registry.py
guild/facilities/resolver.py
guild/runs/__init__.py
guild/runs/types.py
guild/runs/runner.py
guild/runs/campaigns.py
guild/incidents/__init__.py
guild/incidents/types.py
guild/incidents/detector.py
guild/incidents/tracker.py
guild/hero/__init__.py
guild/hero/types.py
guild/hero/registry.py
guild/hero/forms.py
guild/combat/__init__.py
guild/combat/types.py
guild/combat/calculator.py
guild/combat/stances.py

# Configuration
configs/guild.yaml
configs/skills/_schema.yaml
configs/skills/logic_weaving.yaml
configs/skills/oath_binding.yaml
configs/quests/_schema.yaml
configs/quests/syllo/basic.yaml
configs/quests/syllo/intermediate.yaml
configs/quests/syllo/advanced.yaml
configs/facilities/_schema.yaml
configs/facilities/example.yaml
configs/facilities/local.yaml.example
configs/progression/thresholds.yaml
configs/progression/xp_curves.yaml
configs/progression/effects.yaml
configs/regions/novice_valley.yaml
configs/regions/logic_foothills.yaml
configs/regions/reasoning_mountains.yaml
configs/heroes/_schema.yaml
configs/heroes/qwendal_sprite.yaml
configs/runs/training.yaml
configs/runs/evaluation.yaml
configs/incidents/rules.yaml

# Views
views/__init__.py
views/base.py
views/technical/__init__.py
views/technical/formatters.py
views/tavern/__init__.py
views/tavern/mappings.py
views/tavern/formatters.py
views/tavern/narrator.py

# UI
monitoring/ui/tavern_view.html
monitoring/js/tavern_view.js
monitoring/js/combat_calculator.js
monitoring/js/adventure_log.js
monitoring/css/tavern_view.css
monitoring/css/effects.css
```

### Files to Modify

```
# Path resolution
core/paths.py                    -> Use facility resolver
core/training_daemon.py          -> Use @paths, QuestBoard
core/training_status.py          -> Add hero_state
core/train.py                    -> Consume QuestInstances

# Generators
data_manager/generators/*.py     -> Register with QuestForge
monitoring/discrimination_generator.py -> Register

# Evaluators
data_manager/evaluators/*.py     -> Return QuestResult

# Monitoring
monitoring/api/server.py         -> Add /api/hero, /api/quests
monitoring/ui/master_dashboard.html -> Add view toggle
```

### Files to Gitignore

```
# Add to .gitignore
configs/facilities/local.yaml
configs/heroes/*.local.yaml
*.local.yaml
.env
.secrets/
```

---

## Part 7: Open Source Considerations

### 7.1 Repository Structure

```
guild-trainer/                   # Proposed repo name
├── README.md
├── LICENSE                      # MIT or Apache 2.0
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── lint.yml
├── guild/                       # Core engine (the reusable part)
├── configs/
│   ├── examples/               # Example configs
│   └── schemas/                # JSON schemas for validation
├── views/
├── docs/
│   ├── getting-started.md
│   ├── configuration.md
│   ├── skills.md
│   ├── quests.md
│   └── lore.md                 # Optional RPG documentation
├── examples/
│   ├── simple_training/
│   └── multi_gpu/
└── tests/
```

### 7.2 What's Generic vs Specific

**Generic (goes in guild/):**
- Skill/Quest/Progression abstractions
- Facility resolver
- Run management
- Incident tracking
- Combat calculator

**Specific (stays in implementations/ or examples/):**
- SYLLO generator
- Your specific facility configs
- Custom evaluators
- Qwen-specific settings

### 7.3 Environment Variables

```bash
# Required
GUILD_BASE_DIR=/path/to/training    # Base directory
GUILD_CONFIG_DIR=/path/to/configs   # Config directory

# Optional
GUILD_FACILITY=arena_4090           # Default facility
GUILD_LOG_LEVEL=INFO
GUILD_HERO_ID=qwendal_sprite
```

### 7.4 Installation

```bash
# From PyPI (future)
pip install guild-trainer

# From source
git clone https://github.com/username/guild-trainer
cd guild-trainer
pip install -e ".[dev]"

# Configure
cp configs/facilities/example.yaml configs/facilities/local.yaml
# Edit local.yaml with your paths
```

---

## Part 8: Parallel Implementation Order

Since we want to do abstraction AND open-source prep simultaneously:

### Track A: Abstraction Layer (guild/)

```
A1: guild/types.py + guild/skills/types.py
A2: guild/facilities/types.py + resolver.py
A3: guild/quests/types.py + registry.py
A4: guild/progression/types.py + engine.py
A5: guild/runs/types.py + runner.py
A6: guild/incidents/types.py + tracker.py
A7: guild/combat/types.py + calculator.py
A8: Integration with existing code
```

### Track B: Open Source Prep

```
B1: pyproject.toml cleanup (remove personal deps)
B2: Create configs/examples/ with sanitized configs
B3: .gitignore audit (ensure no personal data)
B4: Environment variable support
B5: Documentation (README, getting-started)
B6: GitHub Actions CI
B7: License + CONTRIBUTING.md
B8: Example implementations
```

### Interleaved Schedule

| Day | Track A | Track B |
|-----|---------|---------|
| 1 | A1: Core types | B1: pyproject.toml |
| 2 | A2: Facilities | B2: Example configs |
| 3 | A2: Path resolver integration | B3: .gitignore audit |
| 4 | A3: Quest types | B4: Env var support |
| 5 | A3: Quest registry | B5: README draft |
| 6 | A4: Progression types | B5: Getting started doc |
| 7 | A4: Progression engine | B6: GitHub Actions |
| 8 | A5: Run types | B7: License |
| 9 | A6: Incidents | B8: Simple example |
| 10 | A7: Combat calculator | B8: Multi-GPU example |
| 11 | A8: Integration | Final review |
| 12 | Testing | Testing |

---

## Part 9: First Implementation Batch

To start immediately, here's the first batch of files:

### Batch 1: Foundation Types

```python
# guild/__init__.py
"""
Guild Trainer - A generic framework for LLM training with RPG-style progression.
"""

__version__ = "0.1.0"

from guild.facilities.resolver import init_resolver, resolve, get_facility

__all__ = ["init_resolver", "resolve", "get_facility"]
```

```python
# guild/types.py
"""Common types used across the guild module."""

from enum import Enum

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
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## Summary

This plan transforms your training system from a collection of scripts into a proper framework by:

1. **Abstracting everything** - Skills, quests, facilities, runs all become configurable objects
2. **Separating concerns** - Engine logic vs presentation vs configuration
3. **Enabling open source** - No hardcoded paths, example configs, proper packaging
4. **Preserving the RPG layer** - views/tavern/ provides the game-like experience
5. **Incremental migration** - Each phase adds value without breaking existing functionality

The end result is a system where:
- Adding a new skill = adding a YAML file
- Adding a new quest type = adding a YAML + registering a generator
- Deploying on new hardware = editing facility config
- Switching presentation = toggling between technical/tavern views

Ready to start implementing?
