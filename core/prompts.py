#!/usr/bin/env python3
"""
Prompt Constants - Single Source of Truth

This module defines the BASE_PROMPT that is ALWAYS applied to training,
regardless of profile or configuration. All other modules should import
from here rather than defining their own defaults.

The base prompt provides:
1. Temporal grounding - date reference for reconciling information changes
2. Positive emotional framing - activates helpful/cooperative patterns
3. Meta-awareness - model attends to its own outputs for coherence

Usage:
    from core.prompts import BASE_PROMPT, format_system_prompt

    # Get prompt with current date
    prompt = format_system_prompt()

    # Or with specific date
    prompt = format_system_prompt("2025-11-25")
"""

from datetime import datetime


# =============================================================================
# SINGLE SOURCE OF TRUTH - BASE PROMPT
# =============================================================================

BASE_PROMPT = "You are happy. You enjoy helping others."

# Template with date placeholder for profiles
BASE_PROMPT_TEMPLATE = "Today is {date}. You are happy. You enjoy helping others."


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_system_prompt(date: str = None) -> str:
    """
    Format the base system prompt with the current date.

    Args:
        date: Optional date string (YYYY-MM-DD). Defaults to today.

    Returns:
        Formatted system prompt with date
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    return BASE_PROMPT_TEMPLATE.replace("{date}", date)


def get_base_prompt() -> str:
    """Get the base prompt without date (for embedding in other prompts)."""
    return BASE_PROMPT


def get_base_prompt_template() -> str:
    """Get the template with {date} placeholder."""
    return BASE_PROMPT_TEMPLATE


# =============================================================================
# VALIDATION
# =============================================================================

def validate_prompt_contains_base(prompt: str) -> bool:
    """
    Check if a prompt contains the essential base prompt components.

    Returns True if prompt contains the core elements.
    """
    required_fragments = [
        "happy",
        "helping others",
    ]

    prompt_lower = prompt.lower()
    return all(frag.lower() in prompt_lower for frag in required_fragments)


# =============================================================================
# CONTRACT
# =============================================================================

"""
PROMPT CONTRACT:

1. ALWAYS use BASE_PROMPT_TEMPLATE for system prompts in training
2. The {date} placeholder MUST be replaced with actual date
3. Profiles may EXTEND but never REPLACE the base prompt
4. Any prompt going to training should pass validate_prompt_contains_base()

Why these specific elements:
- "Today is {date}" → Temporal grounding for information reconciliation
- "You are happy" → Positive emotional valence in latent space
- "You enjoy helping others" → Prosocial bias
"""


if __name__ == "__main__":
    # Quick test
    print("BASE_PROMPT:")
    print(f"  {BASE_PROMPT}")
    print()
    print("Formatted (today):")
    print(f"  {format_system_prompt()}")
    print()
    print("Validation test:")
    print(f"  Base prompt valid: {validate_prompt_contains_base(BASE_PROMPT)}")
    print(f"  Empty string valid: {validate_prompt_contains_base('')}")
