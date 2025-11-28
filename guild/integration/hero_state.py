"""
Hero-aware Skill State Management.

Provides multi-hero/multi-campaign support for skill states.
Each hero can have independent progression on each skill.

Key pattern:
    get_hero_skill_state(hero_id, skill_id) -> HeroSkillState
    set_hero_skill_state(hero_id, skill_id, state)

Storage: status/hero_skill_states/{hero_id}.json
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class HeroSkillState:
    """
    Skill state for a specific hero.

    This is the hero-aware version of SkillState.
    Each hero has independent progression per skill.
    """
    hero_id: str
    skill_id: str
    level: int = 1
    xp_total: float = 0.0
    xp_marks: Dict[int, float] = field(default_factory=dict)

    # Accuracy tracking
    recent_results: List[bool] = field(default_factory=list)
    window_size: int = 100

    # Trial state
    eligible_for_trial: bool = False
    last_trial_step: Optional[int] = None
    consecutive_trial_failures: int = 0

    # Per-primitive tracking (from Skill Engine)
    primitive_accuracy: Dict[str, float] = field(default_factory=dict)
    primitive_history: Dict[str, List[bool]] = field(default_factory=dict)

    # Eval tracking
    total_evals: int = 0
    total_samples_seen: int = 0
    last_eval_accuracy: Optional[float] = None
    last_eval_timestamp: Optional[str] = None

    @property
    def accuracy(self) -> float:
        """Calculate current rolling accuracy."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def record_result(self, success: bool):
        """Record a quest result."""
        self.recent_results.append(success)
        if len(self.recent_results) > self.window_size:
            self.recent_results = self.recent_results[-self.window_size:]

    def record_level_up(self):
        """Record level advancement."""
        self.level += 1
        self.xp_marks[self.level] = self.xp_total
        self.eligible_for_trial = False
        self.consecutive_trial_failures = 0

    def record_eval(
        self,
        accuracy: float,
        per_primitive: Optional[Dict[str, float]] = None,
        samples: int = 0,
    ):
        """Record an eval result."""
        self.total_evals += 1
        self.total_samples_seen += samples
        self.last_eval_accuracy = accuracy
        self.last_eval_timestamp = datetime.now().isoformat()

        if per_primitive:
            # Update primitive accuracy (exponential moving average)
            for prim, acc in per_primitive.items():
                if prim in self.primitive_accuracy:
                    # EMA with alpha=0.3
                    old = self.primitive_accuracy[prim]
                    self.primitive_accuracy[prim] = 0.3 * acc + 0.7 * old
                else:
                    self.primitive_accuracy[prim] = acc

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "hero_id": self.hero_id,
            "skill_id": self.skill_id,
            "level": self.level,
            "xp_total": self.xp_total,
            "xp_marks": {str(k): v for k, v in self.xp_marks.items()},
            "recent_results": self.recent_results,
            "window_size": self.window_size,
            "eligible_for_trial": self.eligible_for_trial,
            "last_trial_step": self.last_trial_step,
            "consecutive_trial_failures": self.consecutive_trial_failures,
            "primitive_accuracy": self.primitive_accuracy,
            "primitive_history": self.primitive_history,
            "total_evals": self.total_evals,
            "total_samples_seen": self.total_samples_seen,
            "last_eval_accuracy": self.last_eval_accuracy,
            "last_eval_timestamp": self.last_eval_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HeroSkillState":
        """Deserialize from dict."""
        state = cls(
            hero_id=data.get("hero_id", "default"),
            skill_id=data.get("skill_id", "unknown"),
        )
        state.level = data.get("level", 1)
        state.xp_total = data.get("xp_total", 0.0)
        state.xp_marks = {int(k): v for k, v in data.get("xp_marks", {}).items()}
        state.recent_results = data.get("recent_results", [])
        state.window_size = data.get("window_size", 100)
        state.eligible_for_trial = data.get("eligible_for_trial", False)
        state.last_trial_step = data.get("last_trial_step")
        state.consecutive_trial_failures = data.get("consecutive_trial_failures", 0)
        state.primitive_accuracy = data.get("primitive_accuracy", {})
        state.primitive_history = data.get("primitive_history", {})
        state.total_evals = data.get("total_evals", 0)
        state.total_samples_seen = data.get("total_samples_seen", 0)
        state.last_eval_accuracy = data.get("last_eval_accuracy")
        state.last_eval_timestamp = data.get("last_eval_timestamp")
        return state


class HeroStateManager:
    """
    Manages skill states for all heroes.

    Storage pattern: status/hero_skill_states/{hero_id}.json
    Each file contains all skill states for that hero.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        from core.paths import get_base_dir
        self.base_dir = Path(base_dir) if base_dir else get_base_dir()
        self.state_dir = self.base_dir / "status" / "hero_skill_states"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Cache: hero_id -> {skill_id -> HeroSkillState}
        self._cache: Dict[str, Dict[str, HeroSkillState]] = {}

    def _hero_file(self, hero_id: str) -> Path:
        """Get state file path for a hero."""
        return self.state_dir / f"{hero_id}.json"

    def _load_hero(self, hero_id: str) -> Dict[str, HeroSkillState]:
        """Load all skill states for a hero."""
        if hero_id in self._cache:
            return self._cache[hero_id]

        hero_file = self._hero_file(hero_id)
        states: Dict[str, HeroSkillState] = {}

        if hero_file.exists():
            try:
                with open(hero_file) as f:
                    data = json.load(f)

                for skill_id, state_data in data.get("skills", {}).items():
                    state_data["hero_id"] = hero_id
                    state_data["skill_id"] = skill_id
                    states[skill_id] = HeroSkillState.from_dict(state_data)

            except Exception as e:
                logger.error(f"Failed to load hero state for {hero_id}: {e}")

        self._cache[hero_id] = states
        return states

    def _save_hero(self, hero_id: str):
        """Save all skill states for a hero."""
        if hero_id not in self._cache:
            return

        states = self._cache[hero_id]
        hero_file = self._hero_file(hero_id)

        data = {
            "hero_id": hero_id,
            "updated_at": datetime.now().isoformat(),
            "skills": {
                skill_id: state.to_dict()
                for skill_id, state in states.items()
            }
        }

        try:
            with open(hero_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved hero state for {hero_id}")
        except Exception as e:
            logger.error(f"Failed to save hero state for {hero_id}: {e}")

    def get(self, hero_id: str, skill_id: str) -> HeroSkillState:
        """
        Get skill state for a hero.

        Creates default state if not exists.
        """
        states = self._load_hero(hero_id)

        if skill_id not in states:
            states[skill_id] = HeroSkillState(
                hero_id=hero_id,
                skill_id=skill_id,
            )

        return states[skill_id]

    def set(self, hero_id: str, skill_id: str, state: HeroSkillState):
        """Set skill state for a hero."""
        states = self._load_hero(hero_id)
        states[skill_id] = state
        self._save_hero(hero_id)

    def update(
        self,
        hero_id: str,
        skill_id: str,
        **updates,
    ) -> HeroSkillState:
        """Update specific fields of a hero's skill state."""
        state = self.get(hero_id, skill_id)

        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        self.set(hero_id, skill_id, state)
        return state

    def list_heroes(self) -> List[str]:
        """List all heroes with state files."""
        heroes = []
        for f in self.state_dir.glob("*.json"):
            heroes.append(f.stem)
        return heroes

    def list_skills(self, hero_id: str) -> List[str]:
        """List all skills for a hero."""
        states = self._load_hero(hero_id)
        return list(states.keys())

    def get_all(self, hero_id: str) -> Dict[str, HeroSkillState]:
        """Get all skill states for a hero."""
        return self._load_hero(hero_id).copy()

    def clear_cache(self):
        """Clear the state cache."""
        self._cache.clear()


# Singleton instance
_manager: Optional[HeroStateManager] = None


def get_hero_state_manager() -> HeroStateManager:
    """Get singleton HeroStateManager."""
    global _manager
    if _manager is None:
        _manager = HeroStateManager()
    return _manager


def get_hero_skill_state(hero_id: str, skill_id: str) -> HeroSkillState:
    """Convenience function to get a hero's skill state."""
    return get_hero_state_manager().get(hero_id, skill_id)


def set_hero_skill_state(hero_id: str, skill_id: str, state: HeroSkillState):
    """Convenience function to set a hero's skill state."""
    get_hero_state_manager().set(hero_id, skill_id, state)


def list_hero_skills(hero_id: str) -> List[str]:
    """Convenience function to list a hero's skills."""
    return get_hero_state_manager().list_skills(hero_id)
