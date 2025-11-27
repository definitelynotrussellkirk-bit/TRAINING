"""Combat XP calculator - calculates XP from combat results."""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from guild.quests.types import CombatResult, QuestInstance, QuestResult, QuestDifficulty
from guild.combat.types import CombatConfig
from guild.combat.evaluator import EvaluationResult, get_skill_id


logger = logging.getLogger(__name__)


@dataclass
class XPBreakdown:
    """Breakdown of XP calculation."""

    base_xp: int = 0
    difficulty_bonus: float = 0.0
    skill_bonus: float = 0.0
    streak_bonus: float = 0.0
    effect_bonus: float = 0.0

    total_xp: int = 0

    multipliers: Dict[str, float] = field(default_factory=dict)
    bonuses: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "base_xp": self.base_xp,
            "difficulty_bonus": self.difficulty_bonus,
            "skill_bonus": self.skill_bonus,
            "streak_bonus": self.streak_bonus,
            "effect_bonus": self.effect_bonus,
            "total_xp": self.total_xp,
            "multipliers": self.multipliers,
            "bonuses": self.bonuses,
        }


class CombatCalculator:
    """
    Calculates XP rewards from combat results.

    Factors:
    - Base XP from combat result (CRIT > HIT > GLANCING > MISS > CRIT_MISS)
    - Difficulty multiplier (higher difficulty = more XP)
    - Skill multiplier (some skills worth more)
    - Streak bonus (consecutive wins)
    - Effect multipliers (buffs/debuffs)
    """

    def __init__(self, config: Optional[CombatConfig] = None):
        self.config = config or CombatConfig()

    def calculate_xp(
        self,
        combat_result: CombatResult,
        difficulty: int = 1,
        skill_multiplier: float = 1.0,
        streak: int = 0,
        effect_multipliers: Optional[List[float]] = None,
        flat_bonuses: Optional[List[int]] = None,
    ) -> XPBreakdown:
        """
        Calculate XP from combat result.

        Args:
            combat_result: The combat outcome
            difficulty: Quest difficulty level (1-10)
            skill_multiplier: Skill-specific multiplier
            streak: Consecutive win count
            effect_multipliers: List of effect multipliers (status effects)
            flat_bonuses: List of flat XP bonuses

        Returns:
            XPBreakdown with all components
        """
        breakdown = XPBreakdown()

        # Base XP from combat result
        breakdown.base_xp = self.config.get_base_xp(combat_result)
        current_xp = float(breakdown.base_xp)

        # Difficulty multiplier
        diff_mult = self.config.get_difficulty_multiplier(difficulty)
        breakdown.difficulty_bonus = current_xp * (diff_mult - 1.0)
        breakdown.multipliers["difficulty"] = diff_mult
        current_xp *= diff_mult

        # Skill multiplier
        if skill_multiplier != 1.0:
            breakdown.skill_bonus = current_xp * (skill_multiplier - 1.0)
            breakdown.multipliers["skill"] = skill_multiplier
            current_xp *= skill_multiplier

        # Streak bonus (only for successes)
        if streak > 0 and combat_result in [
            CombatResult.CRITICAL_HIT,
            CombatResult.HIT,
            CombatResult.GLANCING,
        ]:
            # 5% per win, capped at 50%
            streak_mult = min(1.0 + (streak * 0.05), 1.5)
            breakdown.streak_bonus = current_xp * (streak_mult - 1.0)
            breakdown.multipliers["streak"] = streak_mult
            current_xp *= streak_mult

        # Effect multipliers
        if effect_multipliers:
            total_effect_mult = 1.0
            for i, mult in enumerate(effect_multipliers):
                total_effect_mult *= mult
                breakdown.multipliers[f"effect_{i}"] = mult

            breakdown.effect_bonus = current_xp * (total_effect_mult - 1.0)
            current_xp *= total_effect_mult

        # Flat bonuses
        if flat_bonuses:
            for i, bonus in enumerate(flat_bonuses):
                current_xp += bonus
                breakdown.bonuses[f"flat_{i}"] = bonus

        # Round to integer
        breakdown.total_xp = max(0, int(round(current_xp)))

        return breakdown

    def calculate_from_quest(
        self,
        quest: QuestInstance,
        combat_result: CombatResult,
        skill_multiplier: float = 1.0,
        streak: int = 0,
        effect_multipliers: Optional[List[float]] = None,
    ) -> XPBreakdown:
        """Calculate XP from a quest attempt."""
        difficulty = quest.difficulty.value if quest.difficulty else 1
        return self.calculate_xp(
            combat_result=combat_result,
            difficulty=difficulty,
            skill_multiplier=skill_multiplier,
            streak=streak,
            effect_multipliers=effect_multipliers,
        )

    def calculate_from_result(
        self,
        result: QuestResult,
        skill_multiplier: float = 1.0,
        streak: int = 0,
        effect_multipliers: Optional[List[float]] = None,
    ) -> XPBreakdown:
        """Calculate XP from a QuestResult."""
        difficulty = result.difficulty.value if result.difficulty else 1
        return self.calculate_xp(
            combat_result=result.combat_result,
            difficulty=difficulty,
            skill_multiplier=skill_multiplier,
            streak=streak,
            effect_multipliers=effect_multipliers,
        )


class CombatReporter:
    """
    Generates reports and summaries from combat results.
    """

    def __init__(self, calculator: Optional[CombatCalculator] = None):
        self.calculator = calculator or CombatCalculator()
        self._history: List[Dict[str, Any]] = []

    def record(
        self,
        quest: QuestInstance,
        eval_result: EvaluationResult,
        xp_breakdown: XPBreakdown,
    ):
        """Record a combat result."""
        self._history.append({
            "quest_id": quest.id,
            "skill_id": get_skill_id(quest),
            "difficulty": quest.difficulty.value if quest.difficulty else 1,
            "combat_result": eval_result.combat_result.value,
            "match_quality": eval_result.match_quality.value,
            "xp_earned": xp_breakdown.total_xp,
            "success": eval_result.success,
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._history:
            return {
                "total_combats": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_xp": 0,
                "by_result": {},
            }

        total = len(self._history)
        wins = sum(1 for h in self._history if h["success"])
        total_xp = sum(h["xp_earned"] for h in self._history)

        by_result = {}
        for h in self._history:
            result = h["combat_result"]
            by_result[result] = by_result.get(result, 0) + 1

        return {
            "total_combats": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_xp": total_xp,
            "avg_xp": total_xp / total if total > 0 else 0.0,
            "by_result": by_result,
        }

    def get_skill_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get breakdown by skill."""
        by_skill: Dict[str, List[Dict]] = {}

        for h in self._history:
            skill = h["skill_id"]
            if skill not in by_skill:
                by_skill[skill] = []
            by_skill[skill].append(h)

        result = {}
        for skill, records in by_skill.items():
            total = len(records)
            wins = sum(1 for r in records if r["success"])
            total_xp = sum(r["xp_earned"] for r in records)

            result[skill] = {
                "attempts": total,
                "wins": wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "total_xp": total_xp,
            }

        return result

    def clear(self):
        """Clear history."""
        self._history.clear()


# Global calculator
_calculator: Optional[CombatCalculator] = None


def init_combat_calculator(
    config: Optional[CombatConfig] = None,
) -> CombatCalculator:
    """Initialize the global combat calculator."""
    global _calculator
    _calculator = CombatCalculator(config)
    return _calculator


def get_combat_calculator() -> CombatCalculator:
    """Get the global combat calculator."""
    global _calculator
    if _calculator is None:
        _calculator = CombatCalculator()
    return _calculator


def reset_combat_calculator():
    """Reset the global combat calculator (for testing)."""
    global _calculator
    _calculator = None


def calculate_combat_xp(
    combat_result: CombatResult,
    difficulty: int = 1,
    skill_multiplier: float = 1.0,
    streak: int = 0,
    effect_multipliers: Optional[List[float]] = None,
) -> XPBreakdown:
    """Calculate XP using the global calculator."""
    return get_combat_calculator().calculate_xp(
        combat_result=combat_result,
        difficulty=difficulty,
        skill_multiplier=skill_multiplier,
        streak=streak,
        effect_multipliers=effect_multipliers,
    )
