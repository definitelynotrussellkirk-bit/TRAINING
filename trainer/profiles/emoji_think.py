#!/usr/bin/env python3
"""
Emoji Think Profile

Implements the emoji-based thinking and stop signal contract.

Training Contract:
- Each assistant response starts with N thinking emojis (random 2-8, random emoji)
- Each assistant response ends with M stop emojis (random 2-4, random emoji)
- User messages include instructions for both patterns
- Logit processors enforce penalties to prevent unwanted tokens

This creates a structured reasoning format that models can learn.
"""

import json
import random
from typing import Dict, List, Any
from transformers import LogitsProcessorList

from trainer.profiles.base import DataProfile

# Import base prompt (single source of truth)
from core.prompts import BASE_PROMPT_TEMPLATE

# Import logit penalty builders from core
from core.logit_penalty import (
    build_think_penalty_processor,
    build_post_stop_penalty_processor,
    DEFAULT_PENALTY_SCHEDULE,
)


# ============================================================================
# CONSTANTS
# ============================================================================

# Thinking emoji pool - each example gets a RANDOM emoji and count
THINKING_EMOJIS = [
    "🤔",  # Classic thinking
    "💭",  # Thought bubble
    "🧠",  # Brain
    "💡",  # Lightbulb (idea)
    "🎯",  # Target (focus)
    "🔍",  # Magnifying glass (analyze)
    "🤨",  # Raised eyebrow (skeptical)
    "🧐",  # Monocle (scrutinize)
    "⚡",  # Lightning (quick thought)
    "✨",  # Sparkles (insight)
]

# Stop emoji pool - randomly select from these
STOP_EMOJI_POOL = ["🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"]

# Stop emoji count range
STOP_COUNT_MIN = 2
STOP_COUNT_MAX = 4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_random_stop_emoji() -> str:
    """Select a random stop emoji from the pool."""
    return random.choice(STOP_EMOJI_POOL)


def get_random_stop_count() -> int:
    """Select a random stop count (2-4)."""
    return random.randint(STOP_COUNT_MIN, STOP_COUNT_MAX)


def get_stop_instruction(emoji: str, count: int) -> str:
    """Generate stop instruction for a specific emoji and count."""
    count_words = {2: "twice", 3: "three times", 4: "four times"}
    count_text = count_words.get(count, f"{count} times")
    return f"When finished, emit {emoji} /{count_text}/ to signal completion."


def get_stop_suffix(emoji: str, count: int) -> str:
    """Generate stop suffix for a specific emoji and count."""
    return "\n" + emoji * count


def get_thinking_pattern(example_index: int):
    """
    Get RANDOM thinking emoji and count for this example.

    Each example gets:
    - Random emoji from THINKING_EMOJIS pool
    - Random count between 2-8

    Uses example_index as seed for reproducibility.

    Args:
        example_index: Position in dataset

    Returns:
        Tuple of (emoji, count, count_word, prefix, instruction)
    """
    random.seed(example_index)  # Reproducible randomness

    emoji = random.choice(THINKING_EMOJIS)
    count = random.randint(2, 8)

    count_words = ["two", "three", "four", "five", "six", "seven", "eight"]
    count_word = count_words[count - 2]  # count 2 -> index 0

    prefix = emoji * count + "\n"
    instruction = f"For this task, think with {emoji} /{count_word}/ times."

    return emoji, count, count_word, prefix, instruction


# ============================================================================
# EMOJI THINK PROFILE
# ============================================================================

class EmojiThinkProfile(DataProfile):
    """
    Emoji-based thinking and stop signal profile.

    Transforms examples to include:
    - Thinking emoji prefix (random emoji, random 2-8 count)
    - Stop emoji suffix (random emoji, random 2-4 count)
    - Instructions in user messages
    - Sanitization of disallowed tags

    Logit processors:
    - Penalty for <think> tags
    - Escalating penalty for tokens after stop signal
    - Reward for EOS after stop signal
    """

    name = "emoji_think"
    description = "Emoji-based thinking and stop signal contract"
    version = "1.0"

    def sanitize_example(self, example: dict) -> dict:
        """
        Strip disallowed tags like <think> from conversation content.

        Args:
            example: Example with messages

        Returns:
            Cleaned example
        """
        cleaned_messages = []
        for msg in example.get('messages', []):
            content = msg.get('content')
            if msg.get('role') == 'assistant' and isinstance(content, str):
                content = content.replace("<think>", "").replace("</think>", "")
            cleaned_messages.append({**msg, "content": content})

        new_ex = dict(example)
        new_ex['messages'] = cleaned_messages
        return new_ex

    def enforce_thinking_requirement(
        self,
        messages: List[Dict[str, Any]],
        example_index: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Apply thinking pattern to messages with RANDOM emoji and count.

        Args:
            messages: List of message dicts
            example_index: Position in dataset (determines random pattern)

        Returns:
            Messages with thinking patterns applied

        Each example gets a random thinking emoji and repetition count (2-8).
        """
        # Get RANDOM pattern for this specific example
        emoji, count, count_word, prefix, instruction = get_thinking_pattern(example_index)

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            if role == "user":
                # Check if ANY thinking instruction is already present (any emoji variant)
                has_instruction = (
                    any(f"think with {e}" in content.lower() for e in THINKING_EMOJIS)
                    or "think with" in content.lower()
                )
                if not has_instruction:
                    content = content.rstrip() + "\n\n" + instruction

            elif role == "assistant":
                # Check if starts with ANY thinking emoji
                has_prefix = any(content.startswith(e) for e in THINKING_EMOJIS)
                if not has_prefix:
                    content = prefix + content.lstrip()

            msg["content"] = content

        return messages

    def enforce_stop_requirement(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enforce stop emoji pattern in conversations.

        Adds:
        - Stop instruction to user messages (after think instruction)
        - Stop suffix to assistant responses (at end, before EOT)

        Uses random stop emoji and count (2-4) for each conversation.

        Args:
            messages: List of message dicts

        Returns:
            Messages with stop patterns applied
        """
        # Pick ONE random stop emoji and count for this entire conversation
        stop_emoji = get_random_stop_emoji()
        stop_count = get_random_stop_count()
        stop_instruction = get_stop_instruction(stop_emoji, stop_count)
        stop_suffix = get_stop_suffix(stop_emoji, stop_count)

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            if role == "user":
                # Add stop instruction to USER messages
                # Check if ANY stop instruction already present
                has_stop_instruction = any(
                    emoji in content and "When finished" in content
                    for emoji in STOP_EMOJI_POOL
                )
                if not has_stop_instruction:
                    content = content.rstrip() + "\n\n" + stop_instruction

            elif role == "assistant":
                # Append stop suffix to ASSISTANT responses (at END)
                # Check if ANY stop emoji sequence already present at end
                has_stop_suffix = any(
                    content.rstrip().endswith(emoji * count)
                    for emoji in STOP_EMOJI_POOL
                    for count in range(STOP_COUNT_MIN, STOP_COUNT_MAX + 1)
                )
                if not has_stop_suffix:
                    content = content.rstrip() + stop_suffix

            msg["content"] = content

        return messages

    def transform_example(
        self,
        example: dict,
        index: int,
        system_prompt: str
    ) -> dict:
        """
        Transform example with emoji thinking contract.

        Applies in order:
        1. Sanitize (remove <think> tags)
        2. Inject system prompt (if not present)
        3. Enforce thinking requirement (random emoji, random count)
        4. Enforce stop requirement (random emoji, random count)

        Args:
            example: Raw JSONL example
            index: Example index (for random seed)
            system_prompt: System prompt to inject

        Returns:
            Transformed example
        """
        # 1. Sanitize
        example = self.sanitize_example(example)

        # 2. Get messages
        messages = example.get('messages', [])

        # 3. Inject system prompt if not present
        if len(messages) > 0 and messages[0].get('role') != 'system':
            messages.insert(0, {'role': 'system', 'content': system_prompt})
        elif len(messages) > 0 and messages[0].get('role') == 'system':
            # Update existing system prompt
            messages[0]['content'] = system_prompt

        # 4. Enforce thinking requirement (random pattern per example)
        messages = self.enforce_thinking_requirement(messages, index)

        # 5. Enforce stop requirement (random pattern per conversation)
        messages = self.enforce_stop_requirement(messages)

        # Update example
        example['messages'] = messages
        return example

    def build_logits_processors(
        self,
        tokenizer
    ) -> LogitsProcessorList:
        """
        Build logits processors for emoji think profile.

        Processors:
        1. Think tag penalty - penalizes <think> tags in generation
        2. Post-stop penalty - escalating penalty after stop emoji, reward for EOS

        Args:
            tokenizer: Model tokenizer

        Returns:
            LogitsProcessorList with penalties configured
        """
        combined_processors = LogitsProcessorList()

        # 1. Penalize <think> tags during generation
        think_processors = build_think_penalty_processor(
            tokenizer,
            penalty=80.0,
            schedule=DEFAULT_PENALTY_SCHEDULE,
        )
        if len(think_processors) > 0:
            combined_processors.extend(think_processors)

        # 2. Penalize tokens after stop emoji sequences with escalating penalties
        # Supports variable emojis and counts (2-4 repetitions)
        # Also reward EOT tokens to encourage proper termination
        post_stop_processors = build_post_stop_penalty_processor(
            tokenizer,
            stop_emoji_pool=STOP_EMOJI_POOL,  # Pool of 10 stop emojis
            stop_count_min=STOP_COUNT_MIN,    # Minimum 2 repetitions
            stop_count_max=STOP_COUNT_MAX,    # Maximum 4 repetitions
            base_penalty=100.0,  # MASSIVE penalty - nuclear option
            escalation_rate=10.0,  # Extreme escalation
            eot_reward=50.0,  # HUGE reward for EOT
            eot_sequence=None,  # None = use tokenizer.eos_token_id (default)
        )
        if len(post_stop_processors) > 0:
            combined_processors.extend(post_stop_processors)

        return combined_processors if len(combined_processors) > 0 else LogitsProcessorList()

    def get_system_prompt_template(self) -> str:
        """
        Get system prompt template for emoji think profile.

        Returns:
            System prompt with {date} placeholder (from core.prompts)
        """
        return BASE_PROMPT_TEMPLATE


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EmojiThinkProfile",
    "THINKING_EMOJIS",
    "STOP_EMOJI_POOL",
    "STOP_COUNT_MIN",
    "STOP_COUNT_MAX",
]
