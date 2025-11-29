#!/usr/bin/env python3
"""
Locked Configuration Builder

Single source of truth for locked architecture fields that cannot change
during training. These fields define fundamental model architecture and
must be consistent across runs.

Used by:
- ConfigBuilder.to_trainer_config_dict() (campaign path)
- ConfigLoader._merge_config() (config.json path)

Having one source prevents drift when the two paths evolve independently.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from guild.heroes.types import HeroProfile


def build_locked_config(
    hero: Optional['HeroProfile'] = None,
    model_path: str = "",
    max_length: int = 4096,
    model_version: str = "v1",
) -> dict:
    """
    Build locked config from hero profile or fallback values.

    The locked config contains fields that cannot be changed during training:
    - base_model: The model being trained
    - model_architecture: Transformers architecture class
    - max_context_length: Training sequence length
    - vocab_size: Vocabulary size (must match tokenizer)
    - model_version: Version string for tracking

    Args:
        hero: HeroProfile with model specs (preferred - uses YAML as source of truth)
        model_path: Fallback model path if no hero
        max_length: Training context length
        model_version: Version string for tracking

    Returns:
        Dict suitable for LockedConfig:
        {
            'base_model': str,
            'model_architecture': str,
            'max_context_length': int,
            'vocab_size': int,
            'model_version': str,
        }

    Examples:
        # With hero (preferred - architecture from YAML)
        hero = get_hero("titan-qwen3-4b")
        locked = build_locked_config(hero=hero, max_length=512)
        # {'model_architecture': 'Qwen3ForCausalLM', 'vocab_size': 151936, ...}

        # Without hero (fallback for config.json-only usage)
        locked = build_locked_config(model_path="/path/to/model", max_length=2048)
        # {'model_architecture': 'AutoModelForCausalLM', 'vocab_size': 151936, ...}
    """
    if hero is not None:
        # Hero profile is the source of truth for architecture
        return {
            'base_model': hero.model.hf_name,
            'model_architecture': hero.model.architecture,
            'max_context_length': max_length,
            'vocab_size': hero.model.vocab_size,
            'model_version': model_version,
        }
    else:
        # Fallback for non-campaign usage (config.json direct)
        # Use AutoModelForCausalLM which auto-detects
        return {
            'base_model': model_path,
            'model_architecture': 'AutoModelForCausalLM',
            'max_context_length': max_length,
            'vocab_size': 151936,  # Qwen3 default - most common
            'model_version': model_version,
        }


def validate_locked_compatibility(current: dict, previous: dict) -> list:
    """
    Check if two locked configs are compatible for checkpoint resumption.

    Args:
        current: Current locked config
        previous: Previous locked config (from lock file)

    Returns:
        List of incompatibility messages (empty if compatible)
    """
    incompatibilities = []

    # These fields must match exactly
    critical_fields = ['base_model', 'model_architecture', 'vocab_size']

    for field in critical_fields:
        if field in previous and previous[field] != current.get(field):
            incompatibilities.append(
                f"{field}: was {previous[field]!r}, now {current.get(field)!r}"
            )

    # max_context_length can decrease but not increase beyond original
    if 'max_context_length' in previous:
        prev_len = previous['max_context_length']
        curr_len = current.get('max_context_length', prev_len)
        if curr_len > prev_len:
            incompatibilities.append(
                f"max_context_length: cannot increase from {prev_len} to {curr_len}"
            )

    return incompatibilities
