"""XP calculation, level thresholds, and progression logic."""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from guild.quests.types import QuestResult, CombatResult
from guild.combat.types import CombatConfig


logger = logging.getLogger(__name__)


@dataclass
class LevelConfig:
    """Configuration for the leveling system."""
    base_xp: int = 100
    growth_rate: float = 1.5
    max_level: int = 100

    # Bonus thresholds
    crit_bonus_mult: float = 1.5
    streak_bonus_per: float = 0.05  # 5% per streak level
    max_streak_bonus: float = 0.5  # Cap at 50%

    def xp_for_level(self, level: int) -> int:
        """Calculate XP required to reach a level."""
        if level <= 1:
            return 0
        return int(self.base_xp * (self.growth_rate ** (level - 2)))

    def total_xp_for_level(self, level: int) -> int:
        """Calculate total cumulative XP for a level."""
        return sum(self.xp_for_level(i) for i in range(1, level + 1))

    def level_from_xp(self, total_xp: float) -> int:
        """Calculate level from total XP."""
        level = 1
        accumulated = 0
        while level < self.max_level:
            next_req = self.xp_for_level(level + 1)
            if accumulated + next_req > total_xp:
                break
            accumulated += next_req
            level += 1
        return level


@dataclass
class XPModifiers:
    """Modifiers affecting XP calculation."""
    skill_multiplier: float = 1.0
    difficulty_multiplier: float = 1.0
    effect_multiplier: float = 1.0
    streak_multiplier: float = 1.0

    @property
    def total_multiplier(self) -> float:
        """Combined multiplier."""
        return (
            self.skill_multiplier
            * self.difficulty_multiplier
            * self.effect_multiplier
            * self.streak_multiplier
        )


class XPCalculator:
    """
    Calculates XP rewards from quest results.

    Integrates:
    - Combat config for base XP
    - Level config for thresholds
    - Skill multipliers
    - Status effect modifiers
    - Win streak bonuses
    """

    def __init__(
        self,
        combat_config: Optional[CombatConfig] = None,
        level_config: Optional[LevelConfig] = None,
    ):
        self.combat_config = combat_config or CombatConfig()
        self.level_config = level_config or LevelConfig()

    def calculate_base_xp(self, result: CombatResult) -> int:
        """Get base XP for a combat result."""
        return self.combat_config.get_base_xp(result)

    def calculate_difficulty_mult(self, difficulty_level: int) -> float:
        """Get difficulty multiplier."""
        return self.combat_config.get_difficulty_multiplier(difficulty_level)

    def calculate_streak_mult(self, streak: int) -> float:
        """Calculate streak bonus multiplier."""
        if streak <= 0:
            return 1.0

        bonus = streak * self.level_config.streak_bonus_per
        bonus = min(bonus, self.level_config.max_streak_bonus)
        return 1.0 + bonus

    def calculate_effect_mult(self, effect_multipliers: list[float]) -> float:
        """Combine effect multipliers."""
        if not effect_multipliers:
            return 1.0
        result = 1.0
        for mult in effect_multipliers:
            result *= mult
        return result

    def calculate_xp(
        self,
        result: QuestResult,
        skill_multiplier: float = 1.0,
        effect_multipliers: Optional[list[float]] = None,
        streak: int = 0,
    ) -> dict[str, int]:
        """
        Calculate XP rewards for a quest result.

        Args:
            result: The quest result
            skill_multiplier: Multiplier from skill config
            effect_multipliers: Multipliers from active effects
            streak: Current win streak count

        Returns:
            Dict mapping skill_id to XP awarded
        """
        if not result.xp_awarded:
            return {}

        # Calculate modifiers
        mods = XPModifiers(
            skill_multiplier=skill_multiplier,
            difficulty_multiplier=1.0,  # Already applied in result
            effect_multiplier=self.calculate_effect_mult(effect_multipliers or []),
            streak_multiplier=self.calculate_streak_mult(streak),
        )

        # Apply modifiers to each skill's XP
        final_xp = {}
        for skill_id, base_xp in result.xp_awarded.items():
            final_xp[skill_id] = int(base_xp * mods.total_multiplier)

        return final_xp

    def xp_to_next_level(self, current_level: int, current_xp: float) -> int:
        """Calculate XP needed to reach next level."""
        total_for_next = self.level_config.total_xp_for_level(current_level + 1)
        remaining = total_for_next - current_xp
        return max(0, int(remaining))

    def check_level_up(
        self,
        current_level: int,
        current_xp: float,
        new_xp: float,
    ) -> tuple[int, list[int]]:
        """
        Check if XP gain triggers level up(s).

        Args:
            current_level: Current level
            current_xp: XP before gain
            new_xp: XP after gain

        Returns:
            (new_level, list of levels gained)
        """
        levels_gained = []
        level = current_level

        while level < self.level_config.max_level:
            next_level_xp = self.level_config.total_xp_for_level(level + 1)
            if new_xp >= next_level_xp:
                level += 1
                levels_gained.append(level)
            else:
                break

        return level, levels_gained

    def calculate_level_progress(
        self,
        level: int,
        total_xp: float,
    ) -> float:
        """
        Calculate progress to next level (0.0 to 1.0).

        Args:
            level: Current level
            total_xp: Total accumulated XP

        Returns:
            Progress percentage (0.0 to 1.0)
        """
        if level >= self.level_config.max_level:
            return 1.0

        current_level_xp = self.level_config.total_xp_for_level(level)
        next_level_xp = self.level_config.total_xp_for_level(level + 1)

        xp_in_level = total_xp - current_level_xp
        xp_for_level = next_level_xp - current_level_xp

        if xp_for_level <= 0:
            return 1.0

        return min(1.0, max(0.0, xp_in_level / xp_for_level))


# Global calculator
_calculator: Optional[XPCalculator] = None


def get_calculator() -> XPCalculator:
    """Get the global XP calculator."""
    global _calculator
    if _calculator is None:
        _calculator = XPCalculator()
    return _calculator


def reset_calculator():
    """Reset the global calculator (for testing)."""
    global _calculator
    _calculator = None


def init_calculator(
    combat_config: Optional[CombatConfig] = None,
    level_config: Optional[LevelConfig] = None,
) -> XPCalculator:
    """Initialize the global calculator."""
    global _calculator
    _calculator = XPCalculator(combat_config, level_config)
    return _calculator


# Convenience functions

def calculate_xp(
    result: QuestResult,
    skill_multiplier: float = 1.0,
    effect_multipliers: Optional[list[float]] = None,
    streak: int = 0,
) -> dict[str, int]:
    """Calculate XP using the global calculator."""
    return get_calculator().calculate_xp(
        result, skill_multiplier, effect_multipliers, streak
    )


def xp_for_level(level: int) -> int:
    """Get XP required for a level."""
    return get_calculator().level_config.xp_for_level(level)


def level_from_xp(total_xp: float) -> int:
    """Get level from total XP."""
    return get_calculator().level_config.level_from_xp(total_xp)
