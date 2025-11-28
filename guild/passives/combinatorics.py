"""
Combinatorics Passive - Counting and arrangement eval module.

Tests: permutations, combinations, factorials, counting principles.
"""

import random
import math
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class CombinatoricsPassive(PassiveModule):
    """Combinatorics and counting tests."""

    id = "combinatorics"
    name = "Combinatorics"
    category = "math"
    description = "Permutations, combinations, factorials, counting"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._factorial,
            self._permutation_count,
            self._combination_count,
            self._arrangement_word,
            self._selection_problem,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _factorial(self) -> Dict[str, Any]:
        """Calculate factorial."""
        n = random.randint(1, 7)
        answer = math.factorial(n)

        return {
            "prompt": f"What is {n}! (factorial of {n})?",
            "expected": str(answer),
            "primitive_id": "factorial",
            "type": "factorial",
            "n": n,
        }

    def _permutation_count(self) -> Dict[str, Any]:
        """Count permutations P(n,r)."""
        n = random.randint(4, 8)
        r = random.randint(2, min(4, n))
        # P(n,r) = n! / (n-r)!
        answer = math.factorial(n) // math.factorial(n - r)

        return {
            "prompt": f"How many ways can you arrange {r} items from a set of {n} distinct items? (order matters)",
            "expected": str(answer),
            "primitive_id": "permutation_count",
            "type": "permutation",
            "n": n,
            "r": r,
        }

    def _combination_count(self) -> Dict[str, Any]:
        """Count combinations C(n,r)."""
        n = random.randint(4, 10)
        r = random.randint(2, min(4, n))
        # C(n,r) = n! / (r! * (n-r)!)
        answer = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

        return {
            "prompt": f"How many ways can you choose {r} items from a set of {n} distinct items? (order doesn't matter)",
            "expected": str(answer),
            "primitive_id": "combination_count",
            "type": "combination",
            "n": n,
            "r": r,
        }

    def _arrangement_word(self) -> Dict[str, Any]:
        """Count arrangements of letters in a word."""
        words = [
            ("CAT", 6),      # 3! = 6
            ("DOG", 6),      # 3! = 6
            ("MATH", 24),    # 4! = 24
            ("BOOK", 12),    # 4!/2! = 12 (two O's)
            ("MOON", 12),    # 4!/2! = 12 (two O's)
            ("BALL", 12),    # 4!/2! = 12 (two L's)
            ("TREE", 12),    # 4!/2! = 12 (two E's)
            ("CODE", 24),    # 4! = 24
        ]

        word, answer = random.choice(words)

        return {
            "prompt": f"How many different ways can you arrange the letters in '{word}'?",
            "expected": str(answer),
            "primitive_id": "arrangement_word",
            "type": "arrangement",
            "word": word,
        }

    def _selection_problem(self) -> Dict[str, Any]:
        """Simple selection/counting problem."""
        problems = [
            # (question, answer)
            ("You have 3 shirts and 4 pants. How many different outfits can you make?", "12"),
            ("You have 2 hats and 5 scarves. How many hat-scarf combinations are possible?", "10"),
            ("A menu has 4 appetizers and 3 main courses. How many different meals (1 appetizer + 1 main) can you order?", "12"),
            ("You roll a die and flip a coin. How many different outcomes are possible?", "12"),
            ("You have 3 routes to work and 2 routes back. How many different round trips are possible?", "6"),
            ("A password has 2 digits. Each digit can be 0-9. How many passwords are possible?", "100"),
            ("You pick 1 card from each of 2 decks. How many combinations if each deck has 4 cards?", "16"),
            ("A lock has 3 switches, each ON or OFF. How many combinations are possible?", "8"),
        ]

        question, answer = random.choice(problems)

        return {
            "prompt": question,
            "expected": answer,
            "primitive_id": "multiplication_principle",
            "type": "counting",
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

        # Check for number anywhere in response
        if expected_norm in got_norm:
            return True

        # Try to extract numbers from response
        import re
        numbers = re.findall(r'\b\d+\b', got_norm)
        if expected_norm in numbers:
            return True

        return False
