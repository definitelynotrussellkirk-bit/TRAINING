"""
Mock LLM for testing Arcana planner without actual API calls.
"""

from typing import Callable, Optional, List

from .base import BaseLLM


class MockLLM(BaseLLM):
    """
    Mock LLM for testing.

    Usage:
        # Fixed response
        llm = MockLLM(response='(train :quest q1 :steps 100)')

        # Response based on goal keywords
        llm = MockLLM.with_rules({
            'accuracy': '(train :quest q-bin-l2 :steps 200)',
            'loss': '(when (> (metric :name train_loss) 0.1) (train :quest q1 :steps 100))',
        })

        # Custom function
        llm = MockLLM(fn=lambda sys, user: my_logic(user))
    """

    def __init__(
        self,
        response: Optional[str] = None,
        fn: Optional[Callable[[str, str], str]] = None,
        rules: Optional[dict] = None,
    ):
        self.response = response
        self.fn = fn
        self.rules = rules or {}
        self.call_history: List[tuple] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return mock response."""
        self.call_history.append((system_prompt, user_prompt))

        # Custom function
        if self.fn:
            return self.fn(system_prompt, user_prompt)

        # Rule-based
        if self.rules:
            for keyword, response in self.rules.items():
                if keyword.lower() in user_prompt.lower():
                    return response

        # Fixed response
        if self.response:
            return self.response

        # Default
        return '(log "No action taken")'

    @classmethod
    def with_rules(cls, rules: dict) -> 'MockLLM':
        """Create a MockLLM with keyword-based rules."""
        return cls(rules=rules)

    @classmethod
    def smart(cls) -> 'MockLLM':
        """Create a MockLLM with reasonable default behaviors."""
        return cls(rules={
            'accuracy': '''
(when (< (metric :name accuracy) 0.8)
  (train :quest q-bin-l2 :steps 300))
''',
            'loss': '''
(when (> (metric :name train_loss) 0.05)
  (train :quest q-bin-l2 :steps 200))
''',
            'maintain': '''
(do
  (log "Checking queue status")
  (queue-status))
''',
            'explore': '''
(train :quest q-sy-l1 :steps 200)
''',
        })

    def __repr__(self):
        if self.fn:
            return "MockLLM(fn=...)"
        if self.rules:
            return f"MockLLM(rules={list(self.rules.keys())})"
        return f"MockLLM(response={self.response!r})"
