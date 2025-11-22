#!/usr/bin/env python3
"""
Regime-3 Profile

Symbolic reasoning profile with canonical form enforcement.

Training Contract:
- Uses canonical symbolic form: (op arg1 arg2)
- Wraps answers with <<ANS_START>> ... <<ANS_END>> markers
- Simpler logit penalties (no emoji patterns)
- Designed for structured logical reasoning

Example:
    User: "What is 2 + 3?"
    Assistant: "<<ANS_START>> (add 2 3) = 5 <<ANS_END>>"
"""

import json
from typing import Dict, List, Any
from transformers import LogitsProcessorList

from trainer.profiles.base import DataProfile


# ============================================================================
# CONSTANTS
# ============================================================================

# Answer markers for regime-3
ANS_START_MARKER = "<<ANS_START>>"
ANS_END_MARKER = "<<ANS_END>>"


# ============================================================================
# REGIME-3 PROFILE
# ============================================================================

class Regime3Profile(DataProfile):
    """
    Regime-3 symbolic reasoning profile.

    Transforms examples to use:
    - Canonical symbolic form: (op arg1 arg2)
    - Answer markers: <<ANS_START>> ... <<ANS_END>>
    - Clean system prompts for reasoning tasks

    No emoji patterns - designed for pure symbolic reasoning.
    """

    name = "regime3"
    description = "Symbolic reasoning with canonical form and answer markers"
    version = "1.0"

    def sanitize_example(self, example: dict) -> dict:
        """
        Sanitize example (remove unwanted tags).

        For regime-3, we assume examples are already clean.
        Override this if you need custom sanitization.

        Args:
            example: Example to sanitize

        Returns:
            Sanitized example
        """
        # Regime-3 examples should already be in canonical form
        # Just return as-is (could add validation here)
        return example

    def enforce_answer_markers(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enforce answer markers in assistant responses.

        Adds <<ANS_START>> and <<ANS_END>> to assistant messages
        if not already present.

        Args:
            messages: List of message dicts

        Returns:
            Messages with answer markers enforced
        """
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            if role == "assistant":
                # Check if markers already present
                has_start = ANS_START_MARKER in content
                has_end = ANS_END_MARKER in content

                if not has_start and not has_end:
                    # Wrap entire response with markers
                    content = f"{ANS_START_MARKER} {content.strip()} {ANS_END_MARKER}"
                elif has_start and not has_end:
                    # Add missing end marker
                    content = content.rstrip() + f" {ANS_END_MARKER}"
                elif not has_start and has_end:
                    # Add missing start marker
                    content = f"{ANS_START_MARKER} " + content.lstrip()

            msg["content"] = content

        return messages

    def transform_example(
        self,
        example: dict,
        index: int,
        system_prompt: str
    ) -> dict:
        """
        Transform example with regime-3 contract.

        Applies:
        1. Sanitize (if needed)
        2. Inject system prompt
        3. Enforce answer markers

        Args:
            example: Raw JSONL example
            index: Example index (unused for regime-3)
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

        # 4. Enforce answer markers
        messages = self.enforce_answer_markers(messages)

        # Update example
        example['messages'] = messages
        return example

    def build_logits_processors(
        self,
        tokenizer
    ) -> LogitsProcessorList:
        """
        Build logits processors for regime-3 profile.

        Regime-3 uses simpler penalties:
        - No emoji patterns
        - Could add penalties for malformed symbolic expressions
        - Could add rewards for answer markers

        For now, return empty list (no special penalties).

        Args:
            tokenizer: Model tokenizer

        Returns:
            LogitsProcessorList (currently empty)
        """
        # Regime-3 doesn't need complex emoji penalties
        # Future: Could add penalties for:
        # - Malformed symbolic expressions
        # - Missing answer markers
        # - Non-canonical forms

        return LogitsProcessorList()

    def get_system_prompt_template(self) -> str:
        """
        Get system prompt template for regime-3 profile.

        Returns:
            System prompt emphasizing symbolic reasoning and canonical form
        """
        return """Current date: {date}.

You are a symbolic reasoning assistant. Follow these rules:

1. Use canonical symbolic form for operations: (op arg1 arg2)
2. Wrap your final answer with <<ANS_START>> ... <<ANS_END>>
3. Show your reasoning step-by-step
4. Be precise and concise

Example:
User: "What is 2 + 3?"
Assistant: "<<ANS_START>> (add 2 3) = 5 <<ANS_END>>"
"""


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Regime3Profile",
    "ANS_START_MARKER",
    "ANS_END_MARKER",
]
