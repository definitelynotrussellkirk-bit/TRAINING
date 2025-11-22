#!/usr/bin/env python3
"""
Base Profile Interface

Defines the contract that all data profiles must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from transformers import LogitsProcessorList


class DataProfile(ABC):
    """
    Base interface for data transformation profiles.

    Each profile defines:
    1. How to transform raw examples into training format
    2. What logit processors to apply during generation
    3. System prompt templates
    """

    # Profile metadata
    name: str = "base"
    description: str = "Base profile interface"
    version: str = "1.0"

    @abstractmethod
    def transform_example(
        self,
        example: dict,
        index: int,
        system_prompt: str
    ) -> dict:
        """
        Transform raw example into training format.

        Args:
            example: Raw JSONL example {"messages": [...]}
            index: Example index in dataset (for reproducible randomness)
            system_prompt: Base system prompt to inject

        Returns:
            Transformed example with:
            - System prompt injected
            - Profile-specific formatting applied
            - Special tokens/patterns added
            - Sanitization performed

        Example:
            >>> profile = EmojiThinkProfile()
            >>> raw = {"messages": [{"role": "user", "content": "Hello"}]}
            >>> transformed = profile.transform_example(raw, 0, "You are helpful.")
            >>> # transformed now has thinking emoji patterns, stop sequences, etc.
        """
        pass

    @abstractmethod
    def build_logits_processors(
        self,
        tokenizer
    ) -> LogitsProcessorList:
        """
        Build logits processors for this profile.

        Args:
            tokenizer: Model tokenizer (for token ID lookups)

        Returns:
            LogitsProcessorList with:
            - Penalties for unwanted tokens/patterns
            - Rewards for desired tokens
            - Constraints specific to this profile

        Example:
            >>> profile = EmojiThinkProfile()
            >>> processors = profile.build_logits_processors(tokenizer)
            >>> # Use during generation:
            >>> model.generate(..., logits_processor=processors)
        """
        pass

    @abstractmethod
    def get_system_prompt_template(self) -> str:
        """
        Get system prompt template for this profile.

        Returns:
            System prompt string with optional {date} placeholder.

        Example:
            >>> profile = EmojiThinkProfile()
            >>> template = profile.get_system_prompt_template()
            >>> # "Current date: {date}. Respond naturally..."
        """
        pass

    def validate_example(self, example: dict) -> bool:
        """
        Optional: Validate example format.

        Args:
            example: Example to validate

        Returns:
            True if valid, False otherwise

        Default implementation checks for basic structure.
        Profiles can override for custom validation.
        """
        if not isinstance(example, dict):
            return False

        if "messages" not in example:
            return False

        messages = example["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get profile metadata.

        Returns:
            Dict with name, description, version
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }
