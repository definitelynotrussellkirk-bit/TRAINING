#!/usr/bin/env python3
"""
None Profile - Passthrough profile with minimal transformation.

Used when you want standard SFT training without any special
formatting, tokens, or logit processors.
"""

from typing import Dict, Any
from transformers import LogitsProcessorList

from trainer.profiles.base import DataProfile


class NoneProfile(DataProfile):
    """
    Passthrough profile - minimal transformation.

    - No special tokens
    - No logit processors
    - Simple system prompt
    - Standard chat format
    """

    name = "none"
    description = "Passthrough profile with no special processing"
    version = "1.0"

    def transform_example(
        self,
        example: dict,
        index: int,
        system_prompt: str
    ) -> dict:
        """
        Minimal transformation - just inject system prompt if needed.
        """
        messages = example.get("messages", [])

        # Check if system prompt already exists
        has_system = any(m.get("role") == "system" for m in messages)

        if not has_system and system_prompt:
            # Inject system prompt at start
            messages = [{"role": "system", "content": system_prompt}] + messages

        return {"messages": messages}

    def build_logits_processors(self, tokenizer) -> LogitsProcessorList:
        """
        No logit processors - vanilla generation.
        """
        return LogitsProcessorList([])

    def get_system_prompt_template(self) -> str:
        """
        Minimal system prompt.
        """
        return "You are a helpful assistant."
