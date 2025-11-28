"""
Arithmetic Passive - Basic number sense.

Tests: digit sum, even/odd, comparison, modulo, basic operations.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class ArithmeticPassive(PassiveModule):
    """Basic arithmetic and number sense tests."""

    id = "arithmetic"
    name = "Arithmetic"
    category = "arithmetic"
    description = "Basic number sense: digit sum, even/odd, comparison, modulo"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._digit_sum,
            self._even_odd,
            self._comparison,
            self._modulo,
            self._basic_ops,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _digit_sum(self) -> Dict[str, Any]:
        num = random.randint(10, 9999)
        answer = sum(int(d) for d in str(num))
        return {
            "prompt": f"What is the sum of the digits in {num}?",
            "expected": str(answer),
            "primitive_id": "digit_sum",
            "type": "digit_sum",
        }

    def _even_odd(self) -> Dict[str, Any]:
        num = random.randint(1, 1000)
        answer = "even" if num % 2 == 0 else "odd"
        return {
            "prompt": f"Is {num} even or odd?",
            "expected": answer,
            "primitive_id": "parity",
            "type": "even_odd",
        }

    def _comparison(self) -> Dict[str, Any]:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        while b == a:
            b = random.randint(1, 100)
        answer = str(max(a, b))
        return {
            "prompt": f"Which is larger: {a} or {b}?",
            "expected": answer,
            "primitive_id": "compare_integers",
            "type": "comparison",
        }

    def _modulo(self) -> Dict[str, Any]:
        a = random.randint(10, 100)
        b = random.randint(2, 10)
        answer = a % b
        return {
            "prompt": f"What is {a} mod {b}?",
            "expected": str(answer),
            "primitive_id": "modulo",
            "type": "modulo",
        }

    def _basic_ops(self) -> Dict[str, Any]:
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(['+', '-', '*'])

        # Map operations to primitive IDs
        primitive_map = {
            '+': 'add_two_digit',
            '-': 'sub_two_digit',
            '*': 'mul_single_digit',
        }

        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b
        return {
            "prompt": f"What is {a} {op} {b}?",
            "expected": str(answer),
            "primitive_id": primitive_map[op],
            "type": "basic_ops",
        }

    def check_answer(self, expected: str, got: str) -> bool:
        # Normalize and check if expected appears in response
        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Direct match
        if expected_norm == got_norm:
            return True

        # Check if the answer number appears in response
        if expected_norm in got_norm:
            return True

        # For even/odd, check keywords
        if expected_norm in ['even', 'odd']:
            return expected_norm in got_norm

        return False
