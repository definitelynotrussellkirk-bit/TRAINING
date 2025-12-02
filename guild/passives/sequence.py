"""
Sequence Passive - Pattern recognition eval module.

Tests: number sequences, letter patterns, alternating patterns.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class SequencePassive(PassiveModule):
    """Pattern recognition and sequence tests."""

    id = "sequence"
    name = "Sequence"
    category = "reasoning"
    description = "Pattern recognition, number sequences, letter patterns"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """Generate sequence problems. Level parameter accepted but not yet used."""
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._arithmetic_sequence,
            self._geometric_sequence,
            self._fibonacci_like,
            self._alternating_sequence,
            self._square_sequence,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _arithmetic_sequence(self) -> Dict[str, Any]:
        """Arithmetic sequence (constant difference)."""
        start = random.randint(1, 20)
        diff = random.randint(1, 10)
        length = random.randint(4, 6)

        sequence = [start + i * diff for i in range(length)]
        answer = start + length * diff

        seq_str = ", ".join(map(str, sequence))

        return {
            "prompt": f"What is the next number in this sequence: {seq_str}, ?",
            "expected": str(answer),
            "primitive_id": "arithmetic_sequence",
            "type": "sequence",
            "sequence": sequence,
            "pattern": f"+{diff}",
        }

    def _geometric_sequence(self) -> Dict[str, Any]:
        """Geometric sequence (constant ratio)."""
        start = random.randint(1, 5)
        ratio = random.randint(2, 3)
        length = random.randint(4, 5)

        sequence = [start * (ratio ** i) for i in range(length)]
        answer = start * (ratio ** length)

        seq_str = ", ".join(map(str, sequence))

        return {
            "prompt": f"What is the next number in this sequence: {seq_str}, ?",
            "expected": str(answer),
            "primitive_id": "geometric_sequence",
            "type": "sequence",
            "sequence": sequence,
            "pattern": f"*{ratio}",
        }

    def _fibonacci_like(self) -> Dict[str, Any]:
        """Fibonacci-like sequence (sum of previous two)."""
        a = random.randint(1, 5)
        b = random.randint(1, 5)

        sequence = [a, b]
        for _ in range(4):
            sequence.append(sequence[-1] + sequence[-2])

        answer = sequence[-1] + sequence[-2]
        seq_str = ", ".join(map(str, sequence))

        return {
            "prompt": f"What is the next number in this sequence: {seq_str}, ?",
            "expected": str(answer),
            "primitive_id": "fibonacci_sequence",
            "type": "sequence",
            "sequence": sequence,
            "pattern": "sum of previous two",
        }

    def _alternating_sequence(self) -> Dict[str, Any]:
        """Alternating increment sequence."""
        start = random.randint(1, 10)
        inc1 = random.randint(1, 5)
        inc2 = random.randint(1, 5)

        sequence = [start]
        for i in range(5):
            if i % 2 == 0:
                sequence.append(sequence[-1] + inc1)
            else:
                sequence.append(sequence[-1] + inc2)

        # Next increment depends on position
        if len(sequence) % 2 == 1:
            answer = sequence[-1] + inc1
        else:
            answer = sequence[-1] + inc2

        seq_str = ", ".join(map(str, sequence))

        return {
            "prompt": f"What is the next number in this sequence: {seq_str}, ?",
            "expected": str(answer),
            "primitive_id": "alternating_sequence",
            "type": "sequence",
            "sequence": sequence,
            "pattern": f"+{inc1}, +{inc2} alternating",
        }

    def _square_sequence(self) -> Dict[str, Any]:
        """Sequence of perfect squares."""
        start = random.randint(1, 5)
        length = random.randint(4, 6)

        sequence = [(start + i) ** 2 for i in range(length)]
        answer = (start + length) ** 2

        seq_str = ", ".join(map(str, sequence))

        return {
            "prompt": f"What is the next number in this sequence: {seq_str}, ?",
            "expected": str(answer),
            "primitive_id": "square_sequence",
            "type": "sequence",
            "sequence": sequence,
            "pattern": "perfect squares",
        }

    def check_answer(self, expected: str, got: str) -> bool:
        """Check if model's answer matches expected."""
        expected_norm = expected.strip()
        got_norm = got.strip()

        # Direct match
        if expected_norm == got_norm:
            return True

        # Check if expected number appears in response
        if expected_norm in got_norm.split():
            return True

        # Check if expected appears anywhere
        if expected_norm in got_norm:
            return True

        return False
