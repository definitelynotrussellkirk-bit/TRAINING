"""
Logic Passive - Boolean reasoning and deduction.

Tests: AND, OR, XOR, NOT, simple deduction.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class LogicPassive(PassiveModule):
    """Boolean logic and reasoning tests."""

    id = "logic"
    name = "Logic Gates"
    category = "logic"
    description = "Boolean AND, OR, XOR, NOT operations"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._and_gate,
            self._or_gate,
            self._xor_gate,
            self._not_gate,
            self._compound,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _bool_to_str(self, b: bool) -> str:
        return "True" if b else "False"

    def _and_gate(self) -> Dict[str, Any]:
        a = random.choice([True, False])
        b = random.choice([True, False])
        result = a and b
        return {
            "prompt": f"What is {self._bool_to_str(a)} AND {self._bool_to_str(b)}?",
            "expected": self._bool_to_str(result),
            "type": "and",
        }

    def _or_gate(self) -> Dict[str, Any]:
        a = random.choice([True, False])
        b = random.choice([True, False])
        result = a or b
        return {
            "prompt": f"What is {self._bool_to_str(a)} OR {self._bool_to_str(b)}?",
            "expected": self._bool_to_str(result),
            "type": "or",
        }

    def _xor_gate(self) -> Dict[str, Any]:
        a = random.choice([True, False])
        b = random.choice([True, False])
        result = a != b  # XOR
        return {
            "prompt": f"What is {self._bool_to_str(a)} XOR {self._bool_to_str(b)}?",
            "expected": self._bool_to_str(result),
            "type": "xor",
        }

    def _not_gate(self) -> Dict[str, Any]:
        a = random.choice([True, False])
        result = not a
        return {
            "prompt": f"What is NOT {self._bool_to_str(a)}?",
            "expected": self._bool_to_str(result),
            "type": "not",
        }

    def _compound(self) -> Dict[str, Any]:
        a = random.choice([True, False])
        b = random.choice([True, False])
        c = random.choice([True, False])
        # (A AND B) OR C
        result = (a and b) or c
        return {
            "prompt": f"What is ({self._bool_to_str(a)} AND {self._bool_to_str(b)}) OR {self._bool_to_str(c)}?",
            "expected": self._bool_to_str(result),
            "type": "compound",
        }

    def check_answer(self, expected: str, got: str) -> bool:
        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Check for True/False
        if expected_norm == "true":
            return "true" in got_norm or "yes" in got_norm or got_norm == "1"
        elif expected_norm == "false":
            return "false" in got_norm or "no" in got_norm or got_norm == "0"

        return expected_norm in got_norm
