"""Stance management - protocol mode selection and output formatting."""

import logging
import random
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

from guild.combat.types import CombatStance, StanceConfig


logger = logging.getLogger(__name__)


@dataclass
class StanceSelection:
    """Result of stance selection."""

    stance: CombatStance
    use_thinking: bool
    thinking_emoji: str = ""
    stop_emojis: str = ""
    reason: str = ""


class StanceManager:
    """
    Manages combat stances (protocol modes).

    Stances:
    - THOUGHTFUL: Always use emoji thinking (💭...🔚)
    - QUICK_DRAW: Never use emoji thinking (direct answers)
    - ALTERNATING: 50/50 based on index (deterministic)

    Example:
        manager = StanceManager()
        selection = manager.select_stance(index=0)

        if selection.use_thinking:
            # Apply thinking template
            prompt = f"{selection.thinking_emoji} Think step by step..."
    """

    def __init__(self, config: Optional[StanceConfig] = None):
        self.config = config or StanceConfig()
        self._default_stance = CombatStance.ALTERNATING

    def set_default_stance(self, stance: CombatStance):
        """Set the default stance."""
        self._default_stance = stance

    def select_stance(
        self,
        index: int = 0,
        stance: Optional[CombatStance] = None,
        seed: Optional[int] = None,
    ) -> StanceSelection:
        """
        Select a stance for a combat.

        Args:
            index: Example index (for deterministic alternation)
            stance: Override stance (None uses default)
            seed: Random seed for emoji selection

        Returns:
            StanceSelection with stance details
        """
        active_stance = stance or self._default_stance

        if active_stance == CombatStance.THOUGHTFUL:
            use_thinking = True
            reason = "THOUGHTFUL stance: always thinking"
        elif active_stance == CombatStance.QUICK_DRAW:
            use_thinking = False
            reason = "QUICK_DRAW stance: direct answers"
        else:  # ALTERNATING
            use_thinking = (index % 2) == 0
            reason = f"ALTERNATING stance: index {index} -> {'thinking' if use_thinking else 'direct'}"

        selection = StanceSelection(
            stance=active_stance,
            use_thinking=use_thinking,
            reason=reason,
        )

        if use_thinking:
            selection.thinking_emoji = self.select_thinking_emoji(seed or index)
            selection.stop_emojis = self.select_stop_emojis(seed or index)

        return selection

    def select_thinking_emoji(self, seed: int = 0) -> str:
        """Select a thinking emoji."""
        emojis = self.config.thinking_emojis
        if not emojis:
            return "💭"

        # Deterministic selection based on seed
        idx = seed % len(emojis)
        return emojis[idx]

    def select_stop_emojis(self, seed: int = 0) -> str:
        """Select stop emoji sequence."""
        emojis = self.config.stop_emojis
        if not emojis:
            return "🔚🔚"

        # Select count
        count = self.config.min_stop_count + (seed % (
            self.config.max_stop_count - self.config.min_stop_count + 1
        ))

        # Select emoji
        idx = seed % len(emojis)
        return emojis[idx] * count

    def get_all_thinking_emojis(self) -> List[str]:
        """Get all configured thinking emojis."""
        return self.config.thinking_emojis.copy()

    def get_all_stop_emojis(self) -> List[str]:
        """Get all configured stop emojis."""
        return self.config.stop_emojis.copy()


class ResponseFormatter:
    """
    Formats model responses based on stance.

    Handles:
    - Adding thinking markers to prompts
    - Validating response format
    - Extracting content from thinking blocks
    """

    def __init__(self, config: Optional[StanceConfig] = None):
        self.config = config or StanceConfig()

    def format_system_prompt(
        self,
        base_prompt: str,
        selection: StanceSelection,
    ) -> str:
        """
        Format system prompt with stance instructions.

        Args:
            base_prompt: Base system prompt
            selection: Stance selection

        Returns:
            Formatted prompt with stance instructions
        """
        if selection.use_thinking:
            thinking_instruction = (
                f"\n\nThink step by step, starting with {selection.thinking_emoji}. "
                f"End your thinking with {selection.stop_emojis}, then give your final answer."
            )
            return base_prompt + thinking_instruction
        else:
            return base_prompt + "\n\nProvide your answer directly and concisely."

    def validate_response(
        self,
        response: str,
        selection: StanceSelection,
    ) -> Tuple[bool, str]:
        """
        Validate that response matches expected stance format.

        Args:
            response: Model response
            selection: Expected stance

        Returns:
            (is_valid, reason)
        """
        if not response:
            return False, "Empty response"

        if selection.use_thinking:
            # Should have thinking emoji at start
            has_thinking = any(
                response.strip().startswith(emoji)
                for emoji in self.config.thinking_emojis
            )
            if not has_thinking:
                return False, "Missing thinking emoji at start"

            # Should have stop emoji
            has_stop = any(
                emoji in response
                for emoji in self.config.stop_emojis
            )
            if not has_stop:
                return False, "Missing stop emoji"

            return True, "Valid thinking format"

        else:
            # Should NOT have thinking emoji at start
            has_thinking = any(
                response.strip().startswith(emoji)
                for emoji in self.config.thinking_emojis
            )
            if has_thinking:
                return False, "Unexpected thinking emoji in direct mode"

            return True, "Valid direct format"

    def extract_thinking(
        self,
        response: str,
    ) -> Tuple[str, str]:
        """
        Extract thinking block and final answer from response.

        Args:
            response: Full model response

        Returns:
            (thinking_block, final_answer)
        """
        if not response:
            return "", ""

        # Build stop pattern
        stop_pattern = "|".join(
            re.escape(emoji)
            for emoji in self.config.stop_emojis
        )
        stop_regex = f"({stop_pattern})+\\s*"

        # Find stop marker
        match = re.search(stop_regex, response)

        if match:
            thinking = response[:match.start()].strip()
            answer = response[match.end():].strip()
            return thinking, answer

        # No stop marker found - entire response is answer
        return "", response.strip()

    def extract_answer(
        self,
        response: str,
        selection: StanceSelection,
    ) -> str:
        """
        Extract the final answer based on stance.

        Args:
            response: Full model response
            selection: Stance used

        Returns:
            Final answer string
        """
        if selection.use_thinking:
            _, answer = self.extract_thinking(response)
            return answer
        else:
            return response.strip()


# Global stance manager
_stance_manager: Optional[StanceManager] = None


def init_stance_manager(
    config: Optional[StanceConfig] = None,
) -> StanceManager:
    """Initialize the global stance manager."""
    global _stance_manager
    _stance_manager = StanceManager(config)
    return _stance_manager


def get_stance_manager() -> StanceManager:
    """Get the global stance manager."""
    global _stance_manager
    if _stance_manager is None:
        _stance_manager = StanceManager()
    return _stance_manager


def reset_stance_manager():
    """Reset the global stance manager (for testing)."""
    global _stance_manager
    _stance_manager = None


def select_stance(
    index: int = 0,
    stance: Optional[CombatStance] = None,
) -> StanceSelection:
    """Select a stance using the global manager."""
    return get_stance_manager().select_stance(index, stance)
