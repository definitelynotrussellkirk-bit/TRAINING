"""Combat system type definitions."""

from dataclasses import dataclass, field
from enum import Enum

from guild.quests.types import CombatResult


class CombatStance(Enum):
    """Combat stances (protocol modes)."""
    THOUGHTFUL = "thoughtful"   # Emoji thinking mode
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
        """Get base XP for a combat result."""
        return {
            CombatResult.CRITICAL_HIT: self.xp_crit,
            CombatResult.HIT: self.xp_hit,
            CombatResult.GLANCING: self.xp_glancing,
            CombatResult.MISS: self.xp_miss,
            CombatResult.CRITICAL_MISS: self.xp_crit_miss,
        }.get(result, 0)

    def get_difficulty_multiplier(self, level: int) -> float:
        """Get XP multiplier for difficulty level."""
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
