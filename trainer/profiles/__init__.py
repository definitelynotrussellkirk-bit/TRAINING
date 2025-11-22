#!/usr/bin/env python3
"""
Data Profile System

Profiles define how training data is transformed and what logit processors
are applied during training/generation.

Each profile encapsulates:
- Data transformation logic (system prompts, special tokens, formatting)
- Logit processor configuration (penalties, rewards, constraints)
- Profile-specific constants and helpers
"""

from trainer.profiles.base import DataProfile
from trainer.profiles.emoji_think import EmojiThinkProfile

# Profile registry
PROFILE_REGISTRY = {
    "emoji_think": EmojiThinkProfile,
    # Future: "regime3": Regime3Profile,
    # Future: "plain_sft": PlainSFTProfile,
}


def get_profile(name: str) -> DataProfile:
    """
    Get profile instance by name.

    Args:
        name: Profile name (e.g., "emoji_think", "regime3")

    Returns:
        DataProfile instance

    Raises:
        ValueError: If profile not found
    """
    if name not in PROFILE_REGISTRY:
        available = ", ".join(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")

    profile_class = PROFILE_REGISTRY[name]
    return profile_class()


__all__ = [
    "DataProfile",
    "EmojiThinkProfile",
    "get_profile",
    "PROFILE_REGISTRY",
]
