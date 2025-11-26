#!/usr/bin/env python3
"""
Chat Template Overrides
=======================

Qwen3 models have a chat template that auto-injects <think></think> blocks
around all assistant content. This conflicts with custom thinking paradigms
like emoji_think.

This module provides clean chat templates that DON'T auto-inject thinking tags,
allowing profiles to implement their own thinking patterns.

Usage:
    from core.chat_templates import apply_chat_template_override

    # After loading tokenizer:
    apply_chat_template_override(tokenizer, profile_name="emoji_think")
"""

from typing import Optional


# =============================================================================
# CHAT TEMPLATE CONSTANTS
# =============================================================================

# Standard Qwen/ChatML template WITHOUT <think> injection
# This is the clean baseline - just wraps messages in im_start/im_end tags
CHATTML_SIMPLE = """{%- for message in messages %}
{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}
{%- endfor %}
{%- if add_generation_prompt %}
{{ '<|im_start|>assistant\\n' }}
{%- endif %}"""


# ChatML with explicit system prompt default (if no system message provided)
CHATML_WITH_DEFAULT_SYSTEM = """{%- if messages[0]['role'] != 'system' %}
{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}
{%- endfor %}
{%- if add_generation_prompt %}
{{ '<|im_start|>assistant\\n' }}
{%- endif %}"""


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

# Map profile names to preferred chat templates
PROFILE_TEMPLATES = {
    # emoji_think needs clean template (no <think> injection)
    "emoji_think": CHATTML_SIMPLE,

    # regime3 also needs clean template for symbolic reasoning
    "regime3": CHATTML_SIMPLE,

    # Default: use simple template (safe fallback)
    "default": CHATTML_SIMPLE,
}


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_chat_template(profile_name: Optional[str] = None) -> str:
    """
    Get the appropriate chat template for a profile.

    Args:
        profile_name: Name of the active profile (emoji_think, regime3, etc.)

    Returns:
        Jinja2 chat template string
    """
    if profile_name and profile_name in PROFILE_TEMPLATES:
        return PROFILE_TEMPLATES[profile_name]
    return PROFILE_TEMPLATES["default"]


def should_override_template(tokenizer, profile_name: Optional[str] = None) -> bool:
    """
    Check if the tokenizer's template needs to be overridden.

    Qwen3 templates contain <think> injection logic that conflicts with
    custom thinking paradigms. This detects that pattern.

    Args:
        tokenizer: The loaded tokenizer
        profile_name: Active profile name

    Returns:
        True if template should be overridden
    """
    if not tokenizer.chat_template:
        return False

    # Detect Qwen3's think injection patterns
    think_patterns = [
        "<think>",
        "</think>",
        "reasoning_content",
        "enable_thinking",
    ]

    template = tokenizer.chat_template
    has_think_injection = any(pattern in template for pattern in think_patterns)

    # Override if template has think injection AND we're using a custom thinking profile
    if has_think_injection and profile_name in ["emoji_think", "regime3"]:
        return True

    return False


def apply_chat_template_override(
    tokenizer,
    profile_name: Optional[str] = None,
    force: bool = False,
    verbose: bool = True
) -> bool:
    """
    Override tokenizer's chat template if needed.

    This is the main entry point. Call after loading the tokenizer
    to ensure clean template without <think> injection.

    Args:
        tokenizer: The loaded tokenizer
        profile_name: Active profile name
        force: Force override even if detection says no
        verbose: Print status messages

    Returns:
        True if template was overridden, False otherwise
    """
    if not force and not should_override_template(tokenizer, profile_name):
        if verbose:
            print(f"   Chat template: Using tokenizer default (no override needed)")
        return False

    # Get the appropriate template for this profile
    new_template = get_chat_template(profile_name)
    old_template = tokenizer.chat_template

    # Apply override
    tokenizer.chat_template = new_template

    if verbose:
        reason = "forced" if force else "detected <think> injection conflict"
        print(f"   Chat template: Overridden ({reason})")
        print(f"   Chat template: Using CHATML_SIMPLE (no auto-<think>)")

    return True


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnose_chat_template(tokenizer, profile_name: Optional[str] = None):
    """
    Print diagnostic info about chat template configuration.

    Useful for debugging template issues.
    """
    print("\n=== Chat Template Diagnostics ===")
    print(f"Profile: {profile_name or 'None'}")
    print(f"Has chat_template: {tokenizer.chat_template is not None}")

    if tokenizer.chat_template:
        template = tokenizer.chat_template
        print(f"Template length: {len(template)} chars")
        print(f"Contains '<think>': {'<think>' in template}")
        print(f"Contains 'reasoning_content': {'reasoning_content' in template}")
        print(f"Should override: {should_override_template(tokenizer, profile_name)}")

        # Test with sample messages
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        try:
            result = tokenizer.apply_chat_template(messages, tokenize=False)
            print(f"\nSample output:")
            print(result)
        except Exception as e:
            print(f"Error applying template: {e}")

    print("=================================\n")


# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    "CHATTML_SIMPLE",
    "CHATML_WITH_DEFAULT_SYSTEM",
    "PROFILE_TEMPLATES",
    "get_chat_template",
    "should_override_template",
    "apply_chat_template_override",
    "diagnose_chat_template",
]
