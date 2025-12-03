"""
Base LLM interface for Arcana planner.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Complete a prompt and return the response.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message with world state and goal

        Returns:
            The LLM's response (should be S-expressions)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"
