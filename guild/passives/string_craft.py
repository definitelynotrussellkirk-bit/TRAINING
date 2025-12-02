"""
String Craft Passive - Text manipulation tasks.

Tests: reverse, palindrome check, first/last N chars, uppercase/lowercase.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class StringCraftPassive(PassiveModule):
    """Text manipulation tests."""

    id = "string_craft"
    name = "String Craft"
    category = "string_craft"
    description = "Reverse, palindrome, first/last N chars, case conversion"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    WORDS = [
        "hello", "world", "python", "coding", "machine",
        "learning", "neural", "network", "training", "model",
        "radar", "level", "civic", "kayak", "refer",  # Some palindromes
    ]

    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """Generate string craft problems. Level parameter accepted but not yet used."""
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._reverse,
            self._palindrome,
            self._first_n,
            self._last_n,
            self._uppercase,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _reverse(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        reversed_word = word[::-1]
        return {
            "prompt": f"What is '{word}' reversed?",
            "expected": reversed_word,
            "primitive_id": "reverse_string",
            "type": "reverse",
        }

    def _palindrome(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        is_palindrome = word == word[::-1]
        answer = "yes" if is_palindrome else "no"
        return {
            "prompt": f"Is '{word}' a palindrome? Answer yes or no.",
            "expected": answer,
            "primitive_id": "palindrome_check",
            "type": "palindrome",
        }

    def _first_n(self) -> Dict[str, Any]:
        word = random.choice([w for w in self.WORDS if len(w) >= 4])
        n = random.randint(2, min(4, len(word)))
        result = word[:n]
        return {
            "prompt": f"What are the first {n} characters of '{word}'?",
            "expected": result,
            "primitive_id": "substring_check",
            "type": "first_n",
        }

    def _last_n(self) -> Dict[str, Any]:
        word = random.choice([w for w in self.WORDS if len(w) >= 4])
        n = random.randint(2, min(4, len(word)))
        result = word[-n:]
        return {
            "prompt": f"What are the last {n} characters of '{word}'?",
            "expected": result,
            "primitive_id": "substring_check",
            "type": "last_n",
        }

    def _uppercase(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        result = word.upper()
        return {
            "prompt": f"What is '{word}' in uppercase?",
            "expected": result,
            "primitive_id": "sort_letters",  # Using sort_letters as closest primitive
            "type": "uppercase",
        }

    def check_answer(self, expected: str, got: str) -> bool:
        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Direct match (case insensitive for most)
        if expected_norm == got_norm:
            return True

        # Check if expected appears in response
        if expected_norm in got_norm:
            return True

        # For yes/no questions
        if expected_norm in ['yes', 'no']:
            if expected_norm == 'yes':
                return 'yes' in got_norm or 'true' in got_norm or 'is a palindrome' in got_norm
            else:
                return 'no' in got_norm or 'false' in got_norm or 'not a palindrome' in got_norm

        return False
