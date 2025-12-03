"""
Claude API interface for Arcana planner.

Requires ANTHROPIC_API_KEY environment variable.
"""

import os
from typing import Optional

from .base import BaseLLM


class ClaudeLLM(BaseLLM):
    """
    Anthropic Claude API interface.

    Usage:
        llm = ClaudeLLM()  # Uses ANTHROPIC_API_KEY
        llm = ClaudeLLM(api_key="sk-...")
        llm = ClaudeLLM(model="claude-3-haiku-20240307")  # Cheaper/faster

        response = llm.complete(system_prompt, user_prompt)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY or pass api_key="
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Lazy import
        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude API and return response."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract text from response
        return message.content[0].text

    def __repr__(self):
        return f"ClaudeLLM(model={self.model!r})"
