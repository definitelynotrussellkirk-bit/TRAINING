# Guild Refactor - Tasks Phase 10-13

**Phases:** View Layer, Integration, Open Source Prep, World Consistency
**Prerequisites:** Phases 0-9 complete (guild-p9-complete tag)

---

# Phase 10: View Layer

**Goal:** RPG-flavored views and technical passthrough views

---

### P10.1 - Create views/base.py

**Description:** Base view classes and interfaces

**File:** `views/base.py`

```python
"""Base view classes for rendering guild data."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

from guild.progression.types import HeroState
from guild.quests.types import QuestResult
from guild.runs.types import RunState
from guild.incidents.types import Incident


@dataclass
class ViewContext:
    """Context passed to views for rendering."""
    hero: Optional[HeroState] = None
    run: Optional[RunState] = None
    results: list[QuestResult] = None
    incidents: list[Incident] = None
    metrics: dict[str, Any] = None

    def __post_init__(self):
        self.results = self.results or []
        self.incidents = self.incidents or []
        self.metrics = self.metrics or {}


class BaseView(ABC):
    """Base class for views."""

    @property
    @abstractmethod
    def name(self) -> str:
        """View name."""
        pass

    @abstractmethod
    def render_hero(self, hero: HeroState) -> dict:
        """Render hero state."""
        pass

    @abstractmethod
    def render_result(self, result: QuestResult) -> dict:
        """Render a quest result."""
        pass

    @abstractmethod
    def render_run(self, run: RunState) -> dict:
        """Render run state."""
        pass

    @abstractmethod
    def render_incident(self, incident: Incident) -> dict:
        """Render an incident."""
        pass

    def render_context(self, context: ViewContext) -> dict:
        """Render full context."""
        return {
            "hero": self.render_hero(context.hero) if context.hero else None,
            "run": self.render_run(context.run) if context.run else None,
            "results": [self.render_result(r) for r in context.results],
            "incidents": [self.render_incident(i) for i in context.incidents],
            "metrics": context.metrics,
        }


class ViewRegistry:
    """Registry of available views."""

    def __init__(self):
        self._views: dict[str, BaseView] = {}
        self._default: Optional[str] = None

    def register(self, view: BaseView, is_default: bool = False):
        """Register a view."""
        self._views[view.name] = view
        if is_default:
            self._default = view.name

    def get(self, name: str) -> Optional[BaseView]:
        """Get a view by name."""
        return self._views.get(name)

    def get_default(self) -> Optional[BaseView]:
        """Get the default view."""
        if self._default:
            return self._views.get(self._default)
        return None

    def list(self) -> list[str]:
        """List available view names."""
        return list(self._views.keys())


# Global registry
_registry = ViewRegistry()


def get_view_registry() -> ViewRegistry:
    """Get the global view registry."""
    return _registry


def register_view(view: BaseView, is_default: bool = False):
    """Register a view globally."""
    _registry.register(view, is_default)


def get_view(name: str) -> Optional[BaseView]:
    """Get a view by name."""
    return _registry.get(name)
```

**Dependencies:** P1.5, P1.3, P1.6, P1.7

**Acceptance Criteria:**
- [ ] `from views.base import BaseView, ViewContext, get_view_registry` works
- [ ] Registry pattern works

**Effort:** M (25 min)

---

### P10.2 - Create views/technical/__init__.py

**Description:** Technical passthrough view

**File:** `views/technical/__init__.py`

```python
"""Technical view - minimal transformation, developer-focused."""

from typing import Optional

from views.base import BaseView
from guild.progression.types import HeroState
from guild.quests.types import QuestResult
from guild.runs.types import RunState
from guild.incidents.types import Incident


class TechnicalView(BaseView):
    """
    Technical view - returns data with minimal transformation.
    Useful for debugging and developer tools.
    """

    @property
    def name(self) -> str:
        return "technical"

    def render_hero(self, hero: HeroState) -> dict:
        """Render hero as technical data."""
        return {
            "id": hero.hero_id,
            "identity": {
                "architecture": hero.identity.architecture,
                "generation": hero.identity.generation,
                "size": hero.identity.size,
                "checkpoint": hero.identity.checkpoint_path,
                "step": hero.identity.checkpoint_step,
            },
            "skills": {
                skill_id: {
                    "level": state.level,
                    "xp": state.xp_total,
                    "accuracy": round(state.accuracy, 4),
                    "recent_count": len(state.recent_results),
                    "eligible_for_trial": state.eligible_for_trial,
                }
                for skill_id, state in hero.skills.items()
            },
            "effects": [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "severity": e.severity.value,
                    "applied_step": e.applied_at_step,
                    "duration": e.duration_steps,
                }
                for e in hero.active_effects
            ],
            "stats": {
                "total_quests": hero.total_quests,
                "total_xp": hero.total_xp,
                "crits": hero.total_crits,
                "misses": hero.total_misses,
                "health": hero.health,
            },
            "current": {
                "step": hero.current_step,
                "region": hero.current_region,
                "run_id": hero.current_run_id,
            },
        }

    def render_result(self, result: QuestResult) -> dict:
        """Render result as technical data."""
        return {
            "quest_id": result.quest_id,
            "hero_id": result.hero_id,
            "result": result.combat_result.value,
            "success": result.success,
            "xp": result.xp_awarded,
            "metrics": result.metrics,
            "duration_ms": result.duration_ms,
            "time": result.attempted_at.isoformat(),
        }

    def render_run(self, run: RunState) -> dict:
        """Render run as technical data."""
        return {
            "id": run.run_id,
            "type": run.config.type.value,
            "status": run.status.value,
            "progress": {
                "step": run.current_step,
                "quests_completed": run.quests_completed,
                "quests_succeeded": run.quests_succeeded,
                "success_rate": round(run.success_rate, 4),
            },
            "timing": {
                "started": run.started_at.isoformat() if run.started_at else None,
                "duration_sec": round(run.duration_seconds, 1),
            },
            "checkpoints": {
                "last_step": run.last_checkpoint_step,
                "paths": run.checkpoint_paths[-3:],  # Last 3
            },
            "incidents": run.incident_ids,
        }

    def render_incident(self, incident: Incident) -> dict:
        """Render incident as technical data."""
        return {
            "id": incident.id,
            "category": incident.category.value,
            "severity": incident.severity.value,
            "title": incident.title,
            "description": incident.description,
            "status": incident.status.value,
            "step": incident.detected_at_step,
            "time": incident.detected_at_time.isoformat(),
            "resolution": incident.resolution,
        }


# Create and export instance
technical_view = TechnicalView()
```

**Dependencies:** P10.1

**Acceptance Criteria:**
- [ ] `from views.technical import technical_view` works
- [ ] Renders hero/result/run/incident correctly

**Effort:** M (25 min)

---

### P10.3 - Create views/tavern/__init__.py

**Description:** RPG-flavored tavern view

**File:** `views/tavern/__init__.py`

```python
"""Tavern view - RPG-flavored rendering for the dashboard."""

from typing import Optional

from views.base import BaseView
from guild.progression.types import HeroState, StatusEffect
from guild.quests.types import QuestResult, CombatResult
from guild.runs.types import RunState
from guild.incidents.types import Incident
from guild.types import Severity


class TavernView(BaseView):
    """
    Tavern view - renders data with RPG flavor.
    Used for the dashboard and user-facing displays.
    """

    # Combat result translations
    COMBAT_NAMES = {
        CombatResult.CRITICAL_HIT: "Critical Hit!",
        CombatResult.HIT: "Hit",
        CombatResult.GLANCING: "Glancing Blow",
        CombatResult.MISS: "Miss",
        CombatResult.CRITICAL_MISS: "Critical Miss!",
    }

    # Health descriptions
    HEALTH_DESCRIPTIONS = {
        "healthy": "In perfect fighting condition",
        "minor_issues": "A few scratches, nothing serious",
        "fatigued": "Showing signs of wear",
        "wounded": "Bearing significant injuries",
        "struggling": "Barely standing",
    }

    # Severity colors/icons
    SEVERITY_ICONS = {
        Severity.LOW: "🟡",
        Severity.MEDIUM: "🟠",
        Severity.HIGH: "🔴",
        Severity.CRITICAL: "💀",
    }

    @property
    def name(self) -> str:
        return "tavern"

    def render_hero(self, hero: HeroState) -> dict:
        """Render hero in RPG style."""
        identity = hero.identity

        # Build title
        title_parts = []
        if identity.class_name:
            title_parts.append(identity.class_name)
        if identity.stature:
            title_parts.append(identity.stature)
        title = " ".join(title_parts) if title_parts else "Adventurer"

        return {
            "name": identity.name,
            "title": title,
            "race": identity.race or self._infer_race(identity.architecture),
            "description": self._build_hero_description(hero),
            "health": {
                "status": hero.health,
                "description": self.HEALTH_DESCRIPTIONS.get(hero.health, "Unknown condition"),
                "icon": self._health_icon(hero.health),
            },
            "disciplines": [
                self._render_discipline(skill_id, state, hero)
                for skill_id, state in hero.skills.items()
            ],
            "afflictions": [
                self._render_affliction(effect)
                for effect in hero.active_effects
            ],
            "achievements": {
                "quests_completed": hero.total_quests,
                "total_xp": int(hero.total_xp),
                "critical_hits": hero.total_crits,
                "perfect_strikes": f"{hero.total_crits} critical hits",
            },
            "current_campaign": hero.current_run_id,
            "current_region": hero.current_region or "Wandering",
        }

    def _infer_race(self, architecture: str) -> str:
        """Infer race from model architecture."""
        races = {
            "qwen": "Qwen'dal",
            "llama": "Llamantine",
            "mistral": "Mistralian",
            "phi": "Phi'ling",
            "gemma": "Gemmite",
        }
        return races.get(architecture.lower(), "Unknown Origin")

    def _build_hero_description(self, hero: HeroState) -> str:
        """Build narrative description of hero."""
        parts = []

        # Identity
        identity = hero.identity
        race = identity.race or self._infer_race(identity.architecture)
        parts.append(f"A {race} of the {identity.size} order")

        # Experience level
        if hero.total_quests > 1000:
            parts.append("veteran of countless battles")
        elif hero.total_quests > 100:
            parts.append("seasoned adventurer")
        elif hero.total_quests > 10:
            parts.append("promising initiate")
        else:
            parts.append("newcomer to the Guild")

        # Current state
        if hero.health == "healthy":
            parts.append("standing ready for action")
        elif hero.health in ["wounded", "struggling"]:
            parts.append("recovering from recent trials")

        return ", ".join(parts) + "."

    def _render_discipline(self, skill_id: str, state, hero: HeroState) -> dict:
        """Render a skill as a discipline."""
        # Try to get RPG name from registry
        from guild.skills.registry import get_skill
        skill_config = get_skill(skill_id)
        rpg_name = skill_config.rpg_name if skill_config else skill_id.replace("_", " ").title()

        # Progress bar calculation
        next_threshold = skill_config.get_threshold(state.level + 1) if skill_config else 0.7
        progress = min(1.0, state.accuracy / next_threshold) if next_threshold > 0 else 0

        return {
            "name": rpg_name,
            "technical_id": skill_id,
            "rank": self._level_to_rank(state.level),
            "level": state.level,
            "mastery": f"{state.accuracy:.1%}",
            "xp": int(state.xp_total),
            "progress_to_next": f"{progress:.0%}",
            "trial_ready": state.eligible_for_trial,
            "trial_message": "Ready for promotion trial!" if state.eligible_for_trial else None,
        }

    def _level_to_rank(self, level: int) -> str:
        """Convert numeric level to rank title."""
        ranks = {
            1: "Novice",
            2: "Apprentice",
            3: "Journeyman",
            4: "Adept",
            5: "Expert",
            6: "Master",
            7: "Grandmaster",
            8: "Sage",
            9: "Archmage",
            10: "Legendary",
        }
        return ranks.get(level, f"Level {level}")

    def _render_affliction(self, effect: StatusEffect) -> dict:
        """Render status effect as affliction."""
        return {
            "name": effect.rpg_name or effect.name,
            "description": effect.rpg_description or effect.description,
            "severity": effect.severity.value,
            "icon": self.SEVERITY_ICONS.get(effect.severity, "⚠️"),
            "since_step": effect.applied_at_step,
            "cure": effect.cure_condition or "Unknown",
        }

    def _health_icon(self, health: str) -> str:
        """Get icon for health status."""
        icons = {
            "healthy": "💚",
            "minor_issues": "💛",
            "fatigued": "🧡",
            "wounded": "❤️",
            "struggling": "🖤",
        }
        return icons.get(health, "❓")

    def render_result(self, result: QuestResult) -> dict:
        """Render quest result in RPG style."""
        return {
            "outcome": self.COMBAT_NAMES.get(result.combat_result, "Unknown"),
            "success": result.success,
            "icon": self._result_icon(result.combat_result),
            "xp_gained": sum(result.xp_awarded.values()),
            "xp_breakdown": {
                skill_id: f"+{xp} XP"
                for skill_id, xp in result.xp_awarded.items()
            },
            "effects_triggered": result.effects_triggered,
            "notes": result.evaluator_notes,
            "time": result.attempted_at.strftime("%H:%M:%S"),
        }

    def _result_icon(self, result: CombatResult) -> str:
        """Get icon for combat result."""
        icons = {
            CombatResult.CRITICAL_HIT: "⚔️✨",
            CombatResult.HIT: "⚔️",
            CombatResult.GLANCING: "🗡️",
            CombatResult.MISS: "💨",
            CombatResult.CRITICAL_MISS: "💥",
        }
        return icons.get(result, "❓")

    def render_run(self, run: RunState) -> dict:
        """Render run as campaign."""
        run_type_names = {
            "training": "Training Campaign",
            "evaluation": "Trial Run",
            "audit": "Investigation",
            "experiment": "Expedition",
            "generation": "Quest Forge Session",
        }

        return {
            "name": run.config.name or f"Campaign {run.run_id}",
            "type": run_type_names.get(run.config.type.value, "Adventure"),
            "status": self._run_status_description(run),
            "progress": {
                "quests_completed": run.quests_completed,
                "victories": run.quests_succeeded,
                "success_rate": f"{run.success_rate:.1%}",
            },
            "duration": self._format_duration(run.duration_seconds),
            "incidents_count": len(run.incident_ids),
        }

    def _run_status_description(self, run: RunState) -> str:
        """Get description for run status."""
        descriptions = {
            "pending": "Preparing for departure",
            "active": "Currently underway",
            "paused": "Resting at camp",
            "completed": "Successfully concluded",
            "failed": "Ended in defeat",
            "cancelled": "Abandoned",
        }
        return descriptions.get(run.status.value, "Unknown")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"

    def render_incident(self, incident: Incident) -> dict:
        """Render incident in RPG style."""
        return {
            "name": incident.rpg_name or incident.title,
            "location": incident.rpg_location or f"Step {incident.detected_at_step}",
            "severity": incident.severity.value,
            "icon": self.SEVERITY_ICONS.get(incident.severity, "⚠️"),
            "description": incident.description,
            "status": "Resolved" if incident.status.value == "resolved" else "Active",
            "resolution": incident.resolution,
        }


# Create and export instance
tavern_view = TavernView()
```

**Dependencies:** P10.1

**Acceptance Criteria:**
- [ ] `from views.tavern import tavern_view` works
- [ ] RPG-flavored rendering works
- [ ] Health/rank/duration formatting works

**Effort:** L (45 min)

---

### P10.4 - Create views/__init__.py

**Description:** Export views and register defaults

**File:** `views/__init__.py`

```python
"""View layer - rendering guild data for display."""

from views.base import (
    BaseView,
    ViewContext,
    ViewRegistry,
    get_view_registry,
    register_view,
    get_view,
)
from views.technical import technical_view
from views.tavern import tavern_view

# Register views
register_view(technical_view)
register_view(tavern_view, is_default=True)

__all__ = [
    "BaseView",
    "ViewContext",
    "ViewRegistry",
    "get_view_registry",
    "register_view",
    "get_view",
    "technical_view",
    "tavern_view",
]
```

**Dependencies:** P10.1, P10.2, P10.3

**Acceptance Criteria:**
- [ ] `from views import tavern_view, technical_view, get_view` works
- [ ] Default view is tavern

**Effort:** S (5 min)

---

### P10.5 - Create tests/guild/test_views.py

**Description:** Tests for view layer

**File:** `tests/guild/test_views.py`

```python
"""Tests for view layer."""

import pytest
from datetime import datetime

from views import get_view, technical_view, tavern_view, ViewContext
from guild.progression.types import HeroState, HeroIdentity, StatusEffect, EffectType
from guild.quests.types import QuestResult, CombatResult
from guild.runs.types import RunState, RunConfig, RunType
from guild.incidents.types import Incident, IncidentCategory, IncidentStatus
from guild.types import Severity, Status


class TestTechnicalView:
    @pytest.fixture
    def sample_hero(self):
        identity = HeroIdentity(
            id="hero1",
            name="Test Hero",
            architecture="qwen",
            generation="3",
            size="0.6B",
            variant="base",
        )
        hero = HeroState(hero_id="hero1", identity=identity)
        hero.get_skill("logic_weaving").level = 3
        hero.get_skill("logic_weaving").xp_total = 5000
        hero.total_quests = 100
        return hero

    def test_render_hero(self, sample_hero):
        rendered = technical_view.render_hero(sample_hero)

        assert rendered["id"] == "hero1"
        assert rendered["identity"]["architecture"] == "qwen"
        assert "logic_weaving" in rendered["skills"]
        assert rendered["skills"]["logic_weaving"]["level"] == 3
        assert rendered["stats"]["total_quests"] == 100

    def test_render_result(self):
        result = QuestResult(
            quest_id="q1",
            hero_id="hero1",
            response="answer",
            combat_result=CombatResult.HIT,
            xp_awarded={"logic_weaving": 10},
            metrics={"accuracy": 1.0},
        )

        rendered = technical_view.render_result(result)

        assert rendered["quest_id"] == "q1"
        assert rendered["result"] == "hit"
        assert rendered["success"] is True
        assert rendered["xp"]["logic_weaving"] == 10

    def test_render_run(self):
        config = RunConfig(id="run1", type=RunType.TRAINING, name="Test Run")
        state = RunState(run_id="run1", config=config, status=Status.ACTIVE)
        state.quests_completed = 50
        state.quests_succeeded = 40
        state.started_at = datetime.now()

        rendered = technical_view.render_run(state)

        assert rendered["id"] == "run1"
        assert rendered["type"] == "training"
        assert rendered["progress"]["success_rate"] == 0.8


class TestTavernView:
    @pytest.fixture
    def sample_hero(self):
        identity = HeroIdentity(
            id="hero1",
            name="Qwendal",
            architecture="qwen",
            generation="3",
            size="0.6B",
            variant="base",
            race="Qwen'dal",
            class_name="Guild Veteran",
        )
        hero = HeroState(hero_id="hero1", identity=identity)
        hero.get_skill("logic_weaving").level = 5
        hero.get_skill("logic_weaving").xp_total = 50000
        hero.total_quests = 500
        hero.total_crits = 50
        return hero

    def test_render_hero_rpg_style(self, sample_hero):
        rendered = tavern_view.render_hero(sample_hero)

        assert rendered["name"] == "Qwendal"
        assert rendered["title"] == "Guild Veteran"
        assert rendered["race"] == "Qwen'dal"
        assert "disciplines" in rendered
        assert "afflictions" in rendered
        assert rendered["health"]["status"] == "healthy"

    def test_render_discipline(self, sample_hero):
        rendered = tavern_view.render_hero(sample_hero)
        discipline = rendered["disciplines"][0]

        assert discipline["level"] == 5
        assert discipline["rank"] == "Expert"
        assert "mastery" in discipline
        assert "xp" in discipline

    def test_render_affliction(self, sample_hero):
        effect = StatusEffect(
            id="confusion",
            name="Confusion",
            description="Mind clouded",
            type=EffectType.DEBUFF,
            severity=Severity.MEDIUM,
            applied_at_step=100,
            rpg_name="Mind Fog",
            rpg_description="A strange fog clouds the hero's thoughts",
        )
        sample_hero.add_effect(effect)

        rendered = tavern_view.render_hero(sample_hero)

        assert len(rendered["afflictions"]) == 1
        assert rendered["afflictions"][0]["name"] == "Mind Fog"
        assert rendered["health"]["status"] != "healthy"

    def test_render_result_rpg_style(self):
        result = QuestResult(
            quest_id="q1",
            hero_id="hero1",
            response="answer",
            combat_result=CombatResult.CRITICAL_HIT,
            xp_awarded={"logic_weaving": 15},
        )

        rendered = tavern_view.render_result(result)

        assert rendered["outcome"] == "Critical Hit!"
        assert rendered["icon"] == "⚔️✨"
        assert rendered["xp_gained"] == 15

    def test_level_to_rank(self):
        assert tavern_view._level_to_rank(1) == "Novice"
        assert tavern_view._level_to_rank(5) == "Expert"
        assert tavern_view._level_to_rank(10) == "Legendary"

    def test_format_duration(self):
        assert "seconds" in tavern_view._format_duration(45)
        assert "minutes" in tavern_view._format_duration(300)
        assert "h" in tavern_view._format_duration(7200)


class TestViewRegistry:
    def test_get_default_view(self):
        view = get_view("tavern")
        assert view is not None
        assert view.name == "tavern"

    def test_list_views(self):
        from views import get_view_registry
        registry = get_view_registry()
        views = registry.list()

        assert "technical" in views
        assert "tavern" in views
```

**Dependencies:** P10.1, P10.2, P10.3

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_views.py -v` passes all tests

**Effort:** M (30 min)

---

### P10.6 - Commit Phase 10

**Description:** Commit view layer

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 10 - View Layer

- views/base.py: BaseView, ViewContext, ViewRegistry
- views/technical/: TechnicalView (developer-focused)
- views/tavern/: TavernView (RPG-flavored)
- tests/guild/test_views.py: View tests

Two rendering modes: technical (raw data) and tavern (RPG style)"
git tag guild-p10-complete
```

**Dependencies:** P10.1-P10.5

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p10-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 11: Integration & Migration

**Goal:** Wire guild to existing systems, migration path

---

### P11.1 - Create guild/integration/daemon_adapter.py

**Description:** Adapter to integrate guild with training daemon

**File:** `guild/integration/daemon_adapter.py`

```python
"""Adapter for integrating guild with training daemon."""

from typing import Optional, Callable
from pathlib import Path
import json

from guild.runs.runner import RunManager
from guild.runs.types import RunConfig, RunType
from guild.quests.types import QuestInstance, QuestResult, QuestDifficulty
from guild.quests.board import QuestBoard
from guild.progression.engine import ProgressionEngine, create_progression_engine
from guild.progression.types import HeroIdentity
from guild.combat.calculator import CombatCalculator
from guild.combat.evaluators import register_domain_evaluators
from guild.incidents.tracker import IncidentTracker
from guild.facilities.resolver import resolve, init_resolver


class DaemonAdapter:
    """
    Adapts guild components for use with training daemon.

    Provides hooks to integrate with existing training infrastructure.
    """

    def __init__(self, base_dir: Optional[Path | str] = None):
        # Initialize facilities
        try:
            init_resolver()
        except FileNotFoundError:
            pass  # Use legacy paths

        # Set up directories
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            from core.paths import get_base_dir
            self.base_dir = get_base_dir()

        self.status_dir = self.base_dir / "status"
        self.queue_dir = self.base_dir / "queue"

        # Initialize components
        self.board = QuestBoard(persist_dir=self.status_dir / "quests")
        self.incidents = IncidentTracker(persist_dir=self.status_dir / "incidents")
        self.combat = CombatCalculator()
        register_domain_evaluators(self.combat)

        # Hero/progression loaded lazily
        self._progression: Optional[ProgressionEngine] = None
        self._run_manager: Optional[RunManager] = None

    def load_hero(self, hero_path: Optional[Path] = None) -> ProgressionEngine:
        """Load or create hero state."""
        hero_path = hero_path or (self.status_dir / "hero_state.json")

        if hero_path.exists():
            data = json.loads(hero_path.read_text())
            from guild.progression.engine import load_progression_engine
            self._progression = load_progression_engine(data)
        else:
            # Create default hero from config
            identity = self._create_default_identity()
            self._progression = create_progression_engine("qwendal", identity)

        return self._progression

    def _create_default_identity(self) -> HeroIdentity:
        """Create default hero identity from config.json."""
        config_path = self.base_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            model_name = config.get("model_name", "unknown")
        else:
            model_name = "unknown"

        # Parse model name
        parts = model_name.lower().split("_")
        arch = parts[0] if parts else "unknown"
        size = parts[1] if len(parts) > 1 else "unknown"

        return HeroIdentity(
            id="qwendal",
            name="Qwen'dal",
            architecture=arch,
            generation="3",
            size=size,
            variant="base",
            race="Qwen'dal",
            class_name="Guild Trainee",
        )

    def save_hero(self, hero_path: Optional[Path] = None):
        """Save hero state."""
        if self._progression is None:
            return

        hero_path = hero_path or (self.status_dir / "hero_state.json")
        hero_path.parent.mkdir(parents=True, exist_ok=True)
        hero_path.write_text(json.dumps(
            self._progression.hero.to_dict(),
            indent=2
        ))

    def get_run_manager(self) -> RunManager:
        """Get or create run manager."""
        if self._run_manager is None:
            if self._progression is None:
                self.load_hero()

            self._run_manager = RunManager(
                quest_board=self.board,
                progression=self._progression,
                combat=self.combat,
                incidents=self.incidents,
                persist_dir=self.status_dir / "runs",
            )

        return self._run_manager

    # Training integration hooks

    def on_training_file_loaded(self, file_path: Path, examples: list[dict]):
        """Called when a training file is loaded."""
        # Convert examples to quests and post to board
        for i, example in enumerate(examples):
            quest = self._example_to_quest(example, file_path, i)
            if quest:
                self.board.post(quest)

    def _example_to_quest(self, example: dict, source_path: Path, index: int
                          ) -> Optional[QuestInstance]:
        """Convert a training example to a quest instance."""
        # Extract from messages format
        messages = example.get("messages", [])
        if not messages:
            return None

        # Find user and assistant messages
        prompt = ""
        expected = None
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
            elif msg.get("role") == "assistant":
                expected = {"answer": msg.get("content", "")}

        if not prompt:
            return None

        # Determine skill from filename or metadata
        skill = self._infer_skill(source_path.name, example)

        return QuestInstance(
            id=f"train_{source_path.stem}_{index}",
            template_id="training_example",
            skills=[skill] if skill else [],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt=prompt,
            expected=expected,
            metadata={
                "source": str(source_path),
                "index": index,
            },
            source=f"training:{source_path.name}",
        )

    def _infer_skill(self, filename: str, example: dict) -> Optional[str]:
        """Infer skill from filename or example metadata."""
        filename_lower = filename.lower()

        if "syllo" in filename_lower:
            return "logic_weaving"
        elif "discrimination" in filename_lower:
            return "oath_binding"
        elif "math" in filename_lower:
            return "numerical_sorcery"
        elif "code" in filename_lower:
            return "artificer_arts"

        return example.get("skill")

    def on_training_step(self, step: int, loss: float, metrics: dict):
        """Called after each training step."""
        if self._progression:
            self._progression.update_step(step)

            # Check for incidents
            check_context = {"loss": loss, "step": step, **metrics}
            self.incidents.check(check_context, step)

    def on_eval_result(self, prompt: str, response: str, expected: str,
                       correct: bool, step: int):
        """Called when an eval result is available."""
        # Create quest for the eval
        quest = QuestInstance(
            id=f"eval_{step}",
            template_id="eval",
            skills=["logic_weaving"],  # Default
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt=prompt,
            expected={"answer": expected},
        )

        # Evaluate
        manager = self.get_run_manager()
        result = manager.execute_quest(quest, response)

        return result

    def get_status(self) -> dict:
        """Get current guild status for monitoring."""
        status = {
            "board": self.board.stats(),
            "incidents": self.incidents.stats(),
        }

        if self._progression:
            status["hero"] = {
                "id": self._progression.hero.hero_id,
                "health": self._progression.hero.health,
                "step": self._progression.hero.current_step,
                "total_xp": self._progression.hero.total_xp,
                "skills": {
                    sid: {
                        "level": s.level,
                        "accuracy": round(s.accuracy, 3),
                    }
                    for sid, s in self._progression.hero.skills.items()
                },
                "active_effects": len(self._progression.hero.active_effects),
            }

        return status


# Global adapter
_adapter: Optional[DaemonAdapter] = None


def get_daemon_adapter(base_dir: Optional[Path | str] = None) -> DaemonAdapter:
    """Get the global daemon adapter."""
    global _adapter
    if _adapter is None:
        _adapter = DaemonAdapter(base_dir)
    return _adapter


def reset_daemon_adapter():
    """Reset the global adapter."""
    global _adapter
    _adapter = None
```

**Dependencies:** P5.3, P6.1, P7.1, P8.1, P9.1

**Acceptance Criteria:**
- [ ] `from guild.integration.daemon_adapter import get_daemon_adapter` works
- [ ] Training file loading creates quests
- [ ] Status reporting works

**Effort:** L (50 min)

---

### P11.2 - Create guild/integration/__init__.py

**Description:** Export integration components

**File:** `guild/integration/__init__.py`

```python
"""Integration adapters for existing systems."""

from guild.integration.daemon_adapter import (
    DaemonAdapter,
    get_daemon_adapter,
    reset_daemon_adapter,
)

__all__ = [
    "DaemonAdapter",
    "get_daemon_adapter",
    "reset_daemon_adapter",
]
```

**Dependencies:** P11.1

**Acceptance Criteria:**
- [ ] `from guild.integration import DaemonAdapter` works

**Effort:** S (5 min)

---

### P11.3 - Create guild/api/endpoints.py

**Description:** API endpoints for guild data

**File:** `guild/api/endpoints.py`

```python
"""API endpoints for guild data."""

from typing import Optional
from pathlib import Path

from views import get_view, ViewContext
from guild.integration.daemon_adapter import get_daemon_adapter


def get_hero_status(view_name: str = "tavern") -> dict:
    """Get hero status in requested view format."""
    adapter = get_daemon_adapter()

    if adapter._progression is None:
        adapter.load_hero()

    view = get_view(view_name)
    if view is None:
        view = get_view("technical")

    return view.render_hero(adapter._progression.hero)


def get_quest_board_status() -> dict:
    """Get quest board status."""
    adapter = get_daemon_adapter()
    return adapter.board.to_dict()


def get_incidents(limit: int = 20, status: Optional[str] = None) -> list[dict]:
    """Get recent incidents."""
    adapter = get_daemon_adapter()

    from guild.incidents.types import IncidentStatus
    status_filter = IncidentStatus(status) if status else None

    incidents = adapter.incidents.list(status=status_filter)[:limit]

    view = get_view("tavern")
    return [view.render_incident(i) for i in incidents]


def get_run_status(run_id: Optional[str] = None) -> Optional[dict]:
    """Get run status."""
    adapter = get_daemon_adapter()
    manager = adapter.get_run_manager()

    if run_id:
        run = manager.get_run(run_id)
        if run:
            view = get_view("tavern")
            return view.render_run(run)
        return None

    # Return all active runs
    active = manager.active_runs()
    view = get_view("tavern")
    return [view.render_run(r) for r in active]


def get_full_status(view_name: str = "tavern") -> dict:
    """Get complete guild status."""
    adapter = get_daemon_adapter()

    if adapter._progression is None:
        adapter.load_hero()

    view = get_view(view_name)

    context = ViewContext(
        hero=adapter._progression.hero if adapter._progression else None,
        incidents=adapter.incidents.open_incidents()[:5],
    )

    return {
        **view.render_context(context),
        "board": adapter.board.stats(),
        "incident_stats": adapter.incidents.stats(),
    }


# Flask/FastAPI integration helpers

def register_flask_routes(app, prefix: str = "/api/guild"):
    """Register guild API routes with Flask app."""
    from flask import jsonify, request

    @app.route(f"{prefix}/hero")
    def api_hero():
        view = request.args.get("view", "tavern")
        return jsonify(get_hero_status(view))

    @app.route(f"{prefix}/board")
    def api_board():
        return jsonify(get_quest_board_status())

    @app.route(f"{prefix}/incidents")
    def api_incidents():
        limit = int(request.args.get("limit", 20))
        status = request.args.get("status")
        return jsonify(get_incidents(limit, status))

    @app.route(f"{prefix}/status")
    def api_status():
        view = request.args.get("view", "tavern")
        return jsonify(get_full_status(view))


def register_fastapi_routes(app, prefix: str = "/api/guild"):
    """Register guild API routes with FastAPI app."""
    from fastapi import Query

    @app.get(f"{prefix}/hero")
    def api_hero(view: str = Query("tavern")):
        return get_hero_status(view)

    @app.get(f"{prefix}/board")
    def api_board():
        return get_quest_board_status()

    @app.get(f"{prefix}/incidents")
    def api_incidents(limit: int = Query(20), status: str = Query(None)):
        return get_incidents(limit, status)

    @app.get(f"{prefix}/status")
    def api_status(view: str = Query("tavern")):
        return get_full_status(view)
```

**Dependencies:** P10.4, P11.1

**Acceptance Criteria:**
- [ ] `from guild.api.endpoints import get_hero_status` works
- [ ] Flask/FastAPI registration helpers work

**Effort:** M (35 min)

---

### P11.4 - Create guild/api/__init__.py

**Description:** Export API components

**File:** `guild/api/__init__.py`

```python
"""Guild API endpoints."""

from guild.api.endpoints import (
    get_hero_status,
    get_quest_board_status,
    get_incidents,
    get_run_status,
    get_full_status,
    register_flask_routes,
    register_fastapi_routes,
)

__all__ = [
    "get_hero_status",
    "get_quest_board_status",
    "get_incidents",
    "get_run_status",
    "get_full_status",
    "register_flask_routes",
    "register_fastapi_routes",
]
```

**Dependencies:** P11.3

**Acceptance Criteria:**
- [ ] `from guild.api import get_hero_status` works

**Effort:** S (5 min)

---

### P11.5 - Create Migration Guide

**Description:** Document migration path from old system

**File:** `guild/MIGRATION.md`

```markdown
# Guild Migration Guide

## Overview

This guide explains how to migrate from the existing training system to the Guild framework.

## Migration Phases

### Phase 1: Parallel Running (Recommended First Step)

Run guild alongside existing system without replacing anything:

```python
# In training_daemon.py or similar
from guild.integration import get_daemon_adapter

adapter = get_daemon_adapter()
adapter.load_hero()

# Existing training loop continues unchanged
# Guild just observes and tracks
```

### Phase 2: Enable Guild Tracking

Add guild hooks to training callbacks:

```python
# After each training step
adapter.on_training_step(step=current_step, loss=loss, metrics={
    "learning_rate": lr,
    "accuracy": accuracy,
})

# After each eval
adapter.on_eval_result(
    prompt=eval_prompt,
    response=model_response,
    expected=expected_answer,
    correct=is_correct,
    step=current_step,
)

# Periodically save hero state
adapter.save_hero()
```

### Phase 3: Enable Guild API

Add guild endpoints to your API server:

```python
# In monitoring/api/server.py
from guild.api import register_flask_routes

# After creating Flask app
register_flask_routes(app, prefix="/api/guild")
```

New endpoints available:
- `GET /api/guild/hero` - Hero status (RPG view)
- `GET /api/guild/hero?view=technical` - Hero status (technical view)
- `GET /api/guild/board` - Quest board status
- `GET /api/guild/incidents` - Recent incidents
- `GET /api/guild/status` - Full status

### Phase 4: Replace Queue with QuestBoard (Optional)

Replace file-based queue with QuestBoard:

```python
# Old way
files = list(queue_dir.glob("*.jsonl"))
file = files[0]
# process file...

# New way
adapter = get_daemon_adapter()
quest = adapter.board.draw()
if quest:
    # Execute quest
    result = adapter.get_run_manager().execute_quest(quest, response)
```

## Configuration Files

### Create configs/facilities/local.yaml

```yaml
facilities:
  arena:
    id: arena
    name: Training GPU
    type: battlefield
    base_path: /path/to/training
    paths:
      checkpoints: current_model/
      status: status/
      queue: queue/

default_facility: arena
```

### Create configs/skills/ Files

Copy from configs/skills/logic_weaving.yaml template and customize.

## Backward Compatibility

Guild is designed for gradual adoption:

1. **Paths**: `core/paths.py` auto-detects guild or falls back to legacy
2. **Config**: Existing `config.json` still works
3. **Queue**: File-based queue still works alongside QuestBoard
4. **Status**: Existing status files preserved

## Rollback

To disable guild:

1. Remove guild import hooks from training code
2. Guild components become no-ops
3. Existing system continues unchanged

No data migration needed - guild creates separate state files.
```

**Dependencies:** None

**Acceptance Criteria:**
- [ ] Migration guide is complete and accurate
- [ ] Examples are correct

**Effort:** M (25 min)

---

### P11.6 - Commit Phase 11

**Description:** Commit integration layer

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 11 - Integration & Migration

- guild/integration/daemon_adapter.py: Adapter for training daemon
- guild/api/endpoints.py: REST API endpoints
- guild/MIGRATION.md: Migration guide

Hooks for training_step, eval_result, file_loaded.
Flask/FastAPI route registration helpers."
git tag guild-p11-complete
```

**Dependencies:** P11.1-P11.5

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p11-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 12: Open Source Prep

**Goal:** Prepare for public release

---

### P12.1 - Create LICENSE

**Description:** Add open source license

**File:** `LICENSE`

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Dependencies:** None

**Acceptance Criteria:**
- [ ] LICENSE file exists
- [ ] Copyright year and name filled in

**Effort:** S (5 min)

---

### P12.2 - Update pyproject.toml for Publishing

**Description:** Complete pyproject.toml for PyPI

**File:** `pyproject.toml` (update)

Add/update these sections:

```toml
[project]
name = "guild-trainer"
version = "0.1.0"
description = "RPG-style framework for LLM training progression"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["llm", "training", "machine-learning", "ai", "rpg"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.10"

[project.urls]
Homepage = "https://github.com/yourusername/guild-trainer"
Documentation = "https://github.com/yourusername/guild-trainer#readme"
Repository = "https://github.com/yourusername/guild-trainer"
Issues = "https://github.com/yourusername/guild-trainer/issues"

[project.optional-dependencies]
guild = [
    "pyyaml>=6.0",
    "jsonschema>=4.0",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
]
```

**Dependencies:** P0.4

**Acceptance Criteria:**
- [ ] pyproject.toml has all required fields
- [ ] `pip install -e ".[guild,dev]"` works

**Effort:** S (15 min)

---

### P12.3 - Create .github/workflows/tests.yml

**Description:** GitHub Actions CI workflow

**File:** `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, master, feature/*]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[guild,dev]"

      - name: Run tests
        run: |
          pytest tests/guild/ -v --cov=guild --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: false

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install linters
        run: |
          pip install ruff black

      - name: Check formatting
        run: black --check guild/ views/ tests/guild/

      - name: Lint
        run: ruff check guild/ views/
```

**Dependencies:** None

**Acceptance Criteria:**
- [ ] Workflow file is valid YAML
- [ ] CI would pass on current code

**Effort:** M (20 min)

---

### P12.4 - Create Example Project

**Description:** Minimal example showing guild usage

**File:** `examples/quickstart.py`

```python
#!/usr/bin/env python3
"""
Guild Quickstart Example

This example shows basic guild usage:
1. Create a hero
2. Create some quests
3. Execute quests and track progression
4. View results
"""

from guild.progression import create_progression_engine, HeroIdentity
from guild.quests import QuestInstance, QuestDifficulty, get_quest_board
from guild.combat import get_combat_calculator, evaluate_quest
from views import tavern_view


def main():
    # 1. Create a hero
    print("Creating hero...")
    identity = HeroIdentity(
        id="example_hero",
        name="Example",
        architecture="example",
        generation="1",
        size="small",
        variant="base",
        race="Questling",
        class_name="Initiate",
    )
    engine = create_progression_engine("example_hero", identity)
    print(f"Hero created: {engine.hero.identity.name}")

    # 2. Create some quests
    print("\nCreating quests...")
    quests = []
    for i in range(5):
        quest = QuestInstance(
            id=f"quest_{i}",
            template_id="example",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt=f"What is {i} + {i}?",
            expected={"answer": str(i * 2)},
        )
        quests.append(quest)
        print(f"  Created: {quest.id}")

    # 3. Execute quests
    print("\nExecuting quests...")
    calc = get_combat_calculator()

    for quest in quests:
        # Simulate response (in real use, this comes from model)
        response = str(int(quest.expected["answer"]))

        # Evaluate
        result = calc.evaluate(quest, response, hero_id=engine.hero.hero_id)

        # Award XP
        awarded = engine.award_xp(result)

        print(f"  {quest.id}: {result.combat_result.value} -> +{sum(awarded.values())} XP")

    # 4. View results
    print("\nHero Status:")
    rendered = tavern_view.render_hero(engine.hero)

    print(f"  Name: {rendered['name']}")
    print(f"  Title: {rendered['title']}")
    print(f"  Health: {rendered['health']['status']}")

    for disc in rendered['disciplines']:
        print(f"  {disc['name']}: Level {disc['level']} ({disc['rank']}) - {disc['mastery']} mastery")

    print(f"\n  Total XP: {rendered['achievements']['total_xp']}")
    print(f"  Quests Completed: {rendered['achievements']['quests_completed']}")


if __name__ == "__main__":
    main()
```

**Dependencies:** All guild modules

**Acceptance Criteria:**
- [ ] Example runs without errors: `python examples/quickstart.py`
- [ ] Output shows hero progression

**Effort:** M (25 min)

---

### P12.5 - Create README.md for Guild

**Description:** Public-facing README for guild module

**File:** `guild/README.md`

```markdown
# Guild Trainer

An RPG-style framework for LLM training progression.

## Overview

Guild provides a structured way to track and visualize LLM training as an RPG adventure:

- **Heroes** - Your models, with stats and progression
- **Skills** - Trainable capabilities (reasoning, math, code, etc.)
- **Quests** - Training tasks with difficulty levels
- **Combat** - Evaluation with CRIT/HIT/MISS outcomes
- **Progression** - XP, levels, and promotion trials
- **Effects** - Buffs and debuffs (overfitting, OOM, etc.)

## Quick Start

```python
from guild.progression import create_progression_engine, HeroIdentity
from guild.quests import QuestInstance, QuestDifficulty
from guild.combat import evaluate_quest

# Create a hero
identity = HeroIdentity(
    id="my_model",
    name="My Model",
    architecture="llama",
    generation="3",
    size="7B",
    variant="instruct",
)
engine = create_progression_engine("my_model", identity)

# Create a quest
quest = QuestInstance(
    id="quest_1",
    template_id="math",
    skills=["numerical_sorcery"],
    difficulty=QuestDifficulty.BRONZE,
    difficulty_level=1,
    prompt="What is 2 + 2?",
    expected={"answer": "4"},
)

# Evaluate response
result = evaluate_quest(quest, response="4", hero_id="my_model")
print(f"Result: {result.combat_result.value}")  # "hit"

# Award XP
awarded = engine.award_xp(result)
print(f"XP awarded: {awarded}")
```

## Installation

```bash
pip install guild-trainer
# Or with all dependencies:
pip install guild-trainer[guild]
```

## Documentation

- [Migration Guide](MIGRATION.md) - Integrating with existing systems
- [Configuration](../configs/skills/_schema.yaml) - Skill configuration
- [Examples](../examples/) - Usage examples

## License

MIT
```

**Dependencies:** None

**Acceptance Criteria:**
- [ ] README is clear and accurate
- [ ] Quick start example works

**Effort:** M (20 min)

---

### P12.6 - Commit Phase 12

**Description:** Commit open source prep

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 12 - Open Source Prep

- LICENSE: MIT license
- pyproject.toml: Publishing metadata
- .github/workflows/tests.yml: CI workflow
- examples/quickstart.py: Usage example
- guild/README.md: Public documentation

Ready for PyPI publishing."
git tag guild-p12-complete
```

**Dependencies:** P12.1-P12.5

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p12-complete` exists
- [ ] CI would pass

**Effort:** S (5 min)

---

# Phase 13: World Consistency Checker

**Goal:** Validate system against world model (optional, advanced)

---

### P13.1 - Create guild/consistency/types.py

**Description:** World consistency types

**File:** `guild/consistency/types.py`

```python
"""World consistency types."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any


class InconsistencyType(Enum):
    """Types of world inconsistencies."""
    UNMAPPED_EVENT = "unmapped_event"
    RULE_VIOLATION = "rule_violation"
    STATE_PARADOX = "state_paradox"
    MISSING_CONCEPT = "missing_concept"
    TERMINOLOGY_CONFLICT = "term_conflict"


class Resolution(Enum):
    """How to resolve an inconsistency."""
    FIX_SYSTEM = "fix_system"
    UPDATE_LORE = "update_lore"
    ADD_EXCEPTION = "add_exception"
    IGNORE = "ignore"


@dataclass
class WorldRule:
    """A rule about how the world works."""
    id: str
    category: str
    description: str
    validation_fn: Optional[str] = None
    valid_examples: list[str] = field(default_factory=list)
    invalid_examples: list[str] = field(default_factory=list)


@dataclass
class WorldMapping:
    """Mapping from technical concept to world concept."""
    technical_term: str
    world_term: str
    category: str
    description: str
    bidirectional: bool = True
    examples: list[str] = field(default_factory=list)


@dataclass
class Inconsistency:
    """A detected world inconsistency."""
    id: str
    type: InconsistencyType
    description: str

    trigger_event: str
    trigger_data: dict = field(default_factory=dict)

    detected_at: datetime = field(default_factory=datetime.now)
    rule_id: Optional[str] = None

    resolution: Optional[Resolution] = None
    resolution_notes: str = ""
    resolved_at: Optional[datetime] = None

    suggested_system_fix: str = ""
    suggested_lore_addition: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "trigger_event": self.trigger_event,
            "trigger_data": self.trigger_data,
            "detected_at": self.detected_at.isoformat(),
            "rule_id": self.rule_id,
            "resolution": self.resolution.value if self.resolution else None,
            "resolution_notes": self.resolution_notes,
            "suggested_system_fix": self.suggested_system_fix,
            "suggested_lore_addition": self.suggested_lore_addition,
        }
```

**Dependencies:** None

**Acceptance Criteria:**
- [ ] `from guild.consistency.types import Inconsistency, WorldRule` works

**Effort:** S (20 min)

---

### P13.2 - Create guild/consistency/checker.py

**Description:** World consistency checker

**File:** `guild/consistency/checker.py`

```python
"""World consistency checker."""

import logging
from typing import Optional, Callable, Any
from datetime import datetime

from guild.consistency.types import (
    WorldRule, Inconsistency, InconsistencyType, WorldMapping, Resolution
)
from guild.types import generate_id

logger = logging.getLogger(__name__)


class WorldConsistencyChecker:
    """
    Validates system events against the world model.
    If something can't be mapped to world terms, it's flagged.
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
        Returns an Inconsistency if something is wrong.
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

        # Check validators
        for rule_id, validator in self._validators.items():
            rule = self._rules.get(rule_id)
            if rule:
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
                        return inc
                except Exception as e:
                    logger.error(f"Validator {rule_id} failed: {e}")

        return None

    def validate_state(self, state_type: str, state: dict) -> list[Inconsistency]:
        """Validate system state for internal consistency."""
        inconsistencies = []

        if state_type == "hero_state":
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

    def get_pending(self) -> list[Inconsistency]:
        """Get unresolved inconsistencies."""
        return [i for i in self._inconsistencies if i.resolution is None]

    def resolve(self, inconsistency_id: str, resolution: Resolution, notes: str = ""):
        """Mark an inconsistency as resolved."""
        for inc in self._inconsistencies:
            if inc.id == inconsistency_id:
                inc.resolution = resolution
                inc.resolution_notes = notes
                inc.resolved_at = datetime.now()
                break

    def export_lore_suggestions(self) -> dict:
        """Export suggestions for LORE.md updates."""
        suggestions = {"new_mappings": [], "new_rules": []}

        for inc in self.get_pending():
            if inc.type == InconsistencyType.UNMAPPED_EVENT:
                suggestions["new_mappings"].append({
                    "technical": inc.trigger_event,
                    "suggestion": inc.suggested_lore_addition
                })

        return suggestions


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
    WorldMapping("chat_template", "Hero's Voice", "hero",
                 "Chat template defines how the hero speaks"),
    WorldMapping("mixed_precision", "Efficient Form", "training",
                 "Trading precision for speed"),
]


def create_default_checker() -> WorldConsistencyChecker:
    """Create a checker with default mappings."""
    checker = WorldConsistencyChecker()
    for mapping in DEFAULT_MAPPINGS:
        checker.register_mapping(mapping)
    return checker


# Global checker
_checker: Optional[WorldConsistencyChecker] = None


def get_consistency_checker() -> WorldConsistencyChecker:
    """Get the global checker."""
    global _checker
    if _checker is None:
        _checker = create_default_checker()
    return _checker


def check_event(event_type: str, event_data: dict) -> Optional[Inconsistency]:
    """Check an event against world rules."""
    return get_consistency_checker().validate_event(event_type, event_data)
```

**Dependencies:** P13.1

**Acceptance Criteria:**
- [ ] `from guild.consistency.checker import check_event` works
- [ ] Unmapped events create inconsistencies
- [ ] Default mappings are registered

**Effort:** L (40 min)

---

### P13.3 - Update guild/consistency/__init__.py

**Description:** Export consistency components

**File:** `guild/consistency/__init__.py`

```python
"""World consistency checking."""

from guild.consistency.types import (
    WorldRule,
    WorldMapping,
    Inconsistency,
    InconsistencyType,
    Resolution,
)
from guild.consistency.checker import (
    WorldConsistencyChecker,
    get_consistency_checker,
    check_event,
    create_default_checker,
    DEFAULT_MAPPINGS,
)

__all__ = [
    # Types
    "WorldRule",
    "WorldMapping",
    "Inconsistency",
    "InconsistencyType",
    "Resolution",
    # Checker
    "WorldConsistencyChecker",
    "get_consistency_checker",
    "check_event",
    "create_default_checker",
    "DEFAULT_MAPPINGS",
]
```

**Dependencies:** P13.1, P13.2

**Acceptance Criteria:**
- [ ] `from guild.consistency import check_event, WorldMapping` works

**Effort:** S (5 min)

---

### P13.4 - Commit Phase 13

**Description:** Commit world consistency checker

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 13 - World Consistency Checker

- guild/consistency/types.py: WorldRule, WorldMapping, Inconsistency
- guild/consistency/checker.py: WorldConsistencyChecker

Validates system events against LORE.md world model.
Flags unmapped events as potential bugs or lore gaps."
git tag guild-p13-complete
git tag guild-complete
```

**Dependencies:** P13.1-P13.3

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tags `guild-p13-complete` and `guild-complete` exist
- [ ] All tests pass

**Effort:** S (5 min)

---

# Final Checkpoint: Complete Validation

```bash
# All tests pass
pytest tests/guild/ -v

# Example runs
python examples/quickstart.py

# Full import check
python -c "
from guild.skills import get_skill_registry
from guild.quests import get_quest_forge, QuestBoard
from guild.progression import ProgressionEngine
from guild.combat import evaluate_quest
from guild.incidents import report_incident
from guild.runs import RunManager
from guild.consistency import check_event
from guild.integration import DaemonAdapter
from guild.api import get_hero_status
from views import tavern_view, technical_view

print('All imports successful!')
print(f'Skills: {get_skill_registry().count}')
"

# API endpoint test
python -c "
from guild.api import get_full_status
status = get_full_status('technical')
print(f'Status keys: {list(status.keys())}')
"
```

---

**Total Tasks in Phases 10-13:** 18 tasks
**Estimated Time:** 1-2 weeks

---

# Summary: All 77 Tasks

| Phase | Tasks | Description |
|-------|-------|-------------|
| 0 | 7 | Setup & Foundation |
| 1 | 11 | Foundation Types |
| 2 | 9 | Configuration System |
| 3 | 7 | Facilities & Paths |
| 4 | 6 | Skills Registry |
| 5 | 7 | Quest System |
| 6 | 7 | Progression Engine |
| 7 | 5 | Combat Calculator |
| 8 | 5 | Incidents System |
| 9 | 4 | Runs System |
| 10 | 6 | View Layer |
| 11 | 6 | Integration & Migration |
| 12 | 6 | Open Source Prep |
| 13 | 4 | World Consistency |
| **Total** | **77** | |

**Estimated Total Time:** 6-8 weeks

**Recommended Approach:**
1. Complete Phases 0-3 (Foundation) - 1-2 weeks
2. Validate and use path resolution for 1 week
3. Complete Phases 4-6 (Core Systems) - 1-2 weeks
4. Complete Phases 7-9 (Evaluation & Runs) - 1-2 weeks
5. Complete Phase 11 (Integration) - 1 week
6. Optional: Phases 10, 12, 13 as time permits
