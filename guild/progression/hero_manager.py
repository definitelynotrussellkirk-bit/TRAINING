"""Hero state management and persistence."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from guild.progression.types import HeroState, HeroIdentity, StatusEffect
from guild.progression.xp import XPCalculator, get_calculator, LevelConfig
from guild.progression.effects import EffectManager, get_effect_manager
from guild.skills.types import SkillState
from guild.quests.types import QuestResult, CombatResult


logger = logging.getLogger(__name__)


class HeroManager:
    """
    Manages hero state persistence and updates.

    Responsibilities:
    - Load/save hero state from JSON
    - Process quest results (XP, effects, stats)
    - Track win streaks
    - Coordinate with effect manager
    """

    def __init__(
        self,
        state_dir: Path,
        state_file: str = "hero_state.json",
        xp_calculator: Optional[XPCalculator] = None,
        effect_manager: Optional[EffectManager] = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / state_file
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.xp_calculator = xp_calculator or get_calculator()
        self.effect_manager = effect_manager or get_effect_manager()

        self._hero: Optional[HeroState] = None
        self._win_streak: int = 0
        self._recent_results: list[CombatResult] = []
        self._max_recent_results: int = 20

    def _load_state(self) -> Optional[HeroState]:
        """Load hero state from disk."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return HeroState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load hero state: {e}")
            return None

    def _save_state(self):
        """Save hero state to disk."""
        if self._hero is None:
            return

        self._hero.updated_at = datetime.now()

        # Atomic write
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            json.dump(self._hero.to_dict(), f, indent=2)
        temp_file.rename(self.state_file)

        logger.debug("Hero state saved")

    def get_hero(self) -> Optional[HeroState]:
        """Get the current hero, loading if needed."""
        if self._hero is None:
            self._hero = self._load_state()
        return self._hero

    def create_hero(
        self,
        hero_id: str,
        identity: HeroIdentity,
        initial_skills: Optional[list[str]] = None,
    ) -> HeroState:
        """
        Create a new hero.

        Args:
            hero_id: Unique hero identifier
            identity: Hero identity info
            initial_skills: Optional list of skill IDs to initialize

        Returns:
            New HeroState
        """
        self._hero = HeroState(
            hero_id=hero_id,
            identity=identity,
        )

        if initial_skills:
            for skill_id in initial_skills:
                self._hero.skills[skill_id] = SkillState(skill_id=skill_id)

        self._save_state()
        logger.info(f"Created new hero: {hero_id}")
        return self._hero

    def get_or_create_hero(
        self,
        hero_id: str,
        identity: HeroIdentity,
    ) -> HeroState:
        """Get existing hero or create new one."""
        hero = self.get_hero()
        if hero is None or hero.hero_id != hero_id:
            hero = self.create_hero(hero_id, identity)
        return hero

    def update_step(self, step: int):
        """Update current training step."""
        hero = self.get_hero()
        if hero:
            hero.current_step = step
            self._save_state()

    def record_result(
        self,
        result: QuestResult,
        skill_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Record a quest result and update hero state.

        Args:
            result: The quest result
            skill_multiplier: Skill-specific XP multiplier

        Returns:
            Summary of updates (xp_gained, levels_gained, effects_triggered)
        """
        hero = self.get_hero()
        if hero is None:
            raise RuntimeError("No hero loaded")

        summary = {
            "xp_gained": {},
            "levels_gained": {},
            "effects_triggered": [],
            "streak": 0,
        }

        # Update recent results
        self._recent_results.append(result.combat_result)
        if len(self._recent_results) > self._max_recent_results:
            self._recent_results = self._recent_results[-self._max_recent_results:]

        # Update win streak
        if result.success:
            self._win_streak += 1
        else:
            self._win_streak = 0
        summary["streak"] = self._win_streak

        # Get effect multiplier
        effect_mult = self.effect_manager.get_xp_multiplier(hero)

        # Calculate final XP
        xp_gained = self.xp_calculator.calculate_xp(
            result,
            skill_multiplier=skill_multiplier,
            effect_multipliers=[effect_mult],
            streak=self._win_streak,
        )
        summary["xp_gained"] = xp_gained

        # Apply XP to skills
        for skill_id, xp in xp_gained.items():
            skill_state = hero.get_skill(skill_id)
            old_xp = skill_state.xp_total
            skill_state.xp_total += xp

            # Check level up
            old_level = self.xp_calculator.level_config.level_from_xp(old_xp)
            new_level, levels = self.xp_calculator.check_level_up(
                old_level, old_xp, skill_state.xp_total
            )

            if levels:
                skill_state.level = new_level
                summary["levels_gained"][skill_id] = levels
                for lvl in levels:
                    logger.info(f"[{skill_id}] LEVEL UP! Now level {lvl}")

            # Record result for accuracy
            skill_state.record_result(result.success)

        # Update stats
        hero.total_quests += 1
        hero.total_xp += sum(xp_gained.values())
        if result.combat_result == CombatResult.CRITICAL_HIT:
            hero.total_crits += 1
        if result.combat_result in [CombatResult.MISS, CombatResult.CRITICAL_MISS]:
            hero.total_misses += 1

        # Evaluate effects
        new_effects = self.effect_manager.update_effects(
            hero,
            recent_results=self._recent_results,
        )
        summary["effects_triggered"] = [e.id for e in new_effects]

        self._save_state()
        return summary

    def add_xp(
        self,
        skill_id: str,
        amount: int,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """
        Add XP to a skill directly.

        Args:
            skill_id: Skill to add XP to
            amount: XP amount
            source: Source of XP (for logging)

        Returns:
            Summary with level info
        """
        hero = self.get_hero()
        if hero is None:
            raise RuntimeError("No hero loaded")

        skill_state = hero.get_skill(skill_id)
        old_xp = skill_state.xp_total
        skill_state.xp_total += amount
        hero.total_xp += amount

        old_level = self.xp_calculator.level_config.level_from_xp(old_xp)
        new_level, levels = self.xp_calculator.check_level_up(
            old_level, old_xp, skill_state.xp_total
        )

        if levels:
            skill_state.level = new_level
            for lvl in levels:
                logger.info(f"[{skill_id}] LEVEL UP to {lvl} (source: {source})")

        self._save_state()

        return {
            "skill_id": skill_id,
            "xp_added": amount,
            "new_total": skill_state.xp_total,
            "new_level": skill_state.level,
            "levels_gained": levels,
        }

    def apply_effect(
        self,
        effect_id: str,
        cause: Optional[dict] = None,
    ) -> Optional[StatusEffect]:
        """Apply a status effect to the hero."""
        hero = self.get_hero()
        if hero is None:
            raise RuntimeError("No hero loaded")

        effect = self.effect_manager.apply_effect(hero, effect_id, cause)
        if effect:
            self._save_state()
        return effect

    def remove_effect(self, effect_id: str):
        """Remove a status effect from the hero."""
        hero = self.get_hero()
        if hero is None:
            raise RuntimeError("No hero loaded")

        self.effect_manager.remove_effect(hero, effect_id)
        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get hero status summary."""
        hero = self.get_hero()
        if hero is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "hero_id": hero.hero_id,
            "name": hero.identity.name,
            "current_step": hero.current_step,
            "health": hero.health,
            "total_xp": hero.total_xp,
            "total_quests": hero.total_quests,
            "total_crits": hero.total_crits,
            "total_misses": hero.total_misses,
            "active_effects": [e.id for e in hero.active_effects],
            "skills": {
                sid: {
                    "level": s.level,
                    "xp": s.xp_total,
                    "accuracy": s.accuracy,
                }
                for sid, s in hero.skills.items()
            },
            "win_streak": self._win_streak,
        }

    def get_skill_progress(self, skill_id: str) -> Dict[str, Any]:
        """Get detailed progress for a skill."""
        hero = self.get_hero()
        if hero is None:
            return {}

        skill = hero.get_skill(skill_id)
        progress = self.xp_calculator.calculate_level_progress(
            skill.level, skill.xp_total
        )
        xp_to_next = self.xp_calculator.xp_to_next_level(skill.level, skill.xp_total)

        return {
            "skill_id": skill_id,
            "level": skill.level,
            "xp_total": skill.xp_total,
            "progress_to_next": progress,
            "xp_to_next_level": xp_to_next,
            "accuracy": skill.accuracy,
            "recent_results": len(skill.recent_results),
        }


# Global manager
_hero_manager: Optional[HeroManager] = None


def init_hero_manager(
    state_dir: Path,
    state_file: str = "hero_state.json",
) -> HeroManager:
    """Initialize the global hero manager."""
    global _hero_manager
    _hero_manager = HeroManager(state_dir, state_file)
    return _hero_manager


def get_hero_manager() -> HeroManager:
    """Get the global hero manager."""
    global _hero_manager
    if _hero_manager is None:
        raise RuntimeError(
            "Hero manager not initialized. Call init_hero_manager() first."
        )
    return _hero_manager


def reset_hero_manager():
    """Reset the global manager (for testing)."""
    global _hero_manager
    _hero_manager = None


# Convenience functions

def get_hero() -> Optional[HeroState]:
    """Get the current hero."""
    return get_hero_manager().get_hero()


def record_result(
    result: QuestResult,
    skill_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """Record a quest result."""
    return get_hero_manager().record_result(result, skill_multiplier)


def get_hero_status() -> Dict[str, Any]:
    """Get hero status summary."""
    return get_hero_manager().get_status()
