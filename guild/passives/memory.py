"""
Memory Passive - Fact retention and recall eval module.

Tests: simple facts, lists, associations, order memory.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


# Fact categories with questions and answers
SIMPLE_FACTS = [
    ("How many days are in a week?", "7"),
    ("How many months are in a year?", "12"),
    ("How many hours are in a day?", "24"),
    ("How many minutes are in an hour?", "60"),
    ("How many seconds are in a minute?", "60"),
    ("How many cents are in a dollar?", "100"),
    ("How many legs does a spider have?", "8"),
    ("How many sides does a triangle have?", "3"),
    ("How many sides does a square have?", "4"),
    ("How many planets are in our solar system?", "8"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many continents are there?", "7"),
    ("How many colors are in a rainbow?", "7"),
    ("How many letters are in the English alphabet?", "26"),
]

ORDINAL_FACTS = [
    ("What is the first month of the year?", "January"),
    ("What is the last month of the year?", "December"),
    ("What is the seventh day of the week?", "Sunday"),
    ("What is the first day of the week?", "Monday"),
    ("What is the fifth planet from the Sun?", "Jupiter"),
    ("What is the third planet from the Sun?", "Earth"),
    ("What is the smallest prime number?", "2"),
    ("What is the first even number?", "2"),
]

ASSOCIATION_FACTS = [
    ("What color is the sky on a clear day?", "blue"),
    ("What color is grass?", "green"),
    ("What color is a banana?", "yellow"),
    ("What season comes after winter?", "spring"),
    ("What season comes after summer?", "fall"),
    ("What is the opposite of hot?", "cold"),
    ("What is the opposite of up?", "down"),
    ("What animal says 'moo'?", "cow"),
    ("What animal says 'woof'?", "dog"),
]


class MemoryPassive(PassiveModule):
    """Fact retention and recall tests."""

    id = "memory"
    name = "Memory"
    category = "memory"
    description = "Simple facts, ordinal knowledge, associations"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """Generate memory problems. Level parameter accepted but not yet used."""
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._simple_fact,
            self._ordinal_fact,
            self._association_fact,
            self._list_recall,
            self._order_recall,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _simple_fact(self) -> Dict[str, Any]:
        """Simple numeric fact recall."""
        question, answer = random.choice(SIMPLE_FACTS)
        return {
            "prompt": question,
            "expected": answer,
            "primitive_id": "simple_fact",
            "type": "fact",
        }

    def _ordinal_fact(self) -> Dict[str, Any]:
        """Ordinal position fact recall."""
        question, answer = random.choice(ORDINAL_FACTS)
        return {
            "prompt": question,
            "expected": answer,
            "primitive_id": "ordinal_fact",
            "type": "fact",
        }

    def _association_fact(self) -> Dict[str, Any]:
        """Association/relationship fact recall."""
        question, answer = random.choice(ASSOCIATION_FACTS)
        return {
            "prompt": question,
            "expected": answer,
            "primitive_id": "association_fact",
            "type": "association",
        }

    def _list_recall(self) -> Dict[str, Any]:
        """Given a short list, recall items."""
        # Generate a short list of random items
        categories = {
            "colors": ["red", "blue", "green", "yellow", "orange", "purple", "pink"],
            "animals": ["cat", "dog", "bird", "fish", "horse", "cow", "sheep"],
            "fruits": ["apple", "banana", "orange", "grape", "mango", "pear"],
            "numbers": ["one", "two", "three", "four", "five", "six", "seven"],
        }

        category = random.choice(list(categories.keys()))
        items = random.sample(categories[category], random.randint(3, 5))
        target_pos = random.randint(0, len(items) - 1)
        ordinal = ["first", "second", "third", "fourth", "fifth"][target_pos]

        items_str = ", ".join(items)

        return {
            "prompt": f"Given this list: {items_str}. What is the {ordinal} item?",
            "expected": items[target_pos],
            "primitive_id": "list_recall",
            "type": "list",
            "items": items,
            "position": target_pos,
        }

    def _order_recall(self) -> Dict[str, Any]:
        """Recall ordering/sequence from common knowledge."""
        sequences = [
            ("days of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            ("months", ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]),
            ("single digits", ["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        ]

        name, seq = random.choice(sequences)
        idx = random.randint(1, len(seq) - 1)
        prev = seq[idx - 1]
        answer = seq[idx]

        return {
            "prompt": f"In the sequence of {name}, what comes after {prev}?",
            "expected": answer,
            "primitive_id": "order_recall",
            "type": "order",
            "sequence_name": name,
        }

    def check_answer(self, expected: str, got: str) -> bool:
        """Check if model's answer matches expected."""
        expected_norm = expected.strip().lower()
        got_norm = got.strip().lower()

        # Direct match
        if expected_norm == got_norm:
            return True

        # Check if expected appears in response
        if expected_norm in got_norm:
            return True

        # Check for numeric equivalents
        number_words = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
            "10": "ten", "11": "eleven", "12": "twelve", "24": "twenty-four",
            "26": "twenty-six", "60": "sixty", "100": "hundred",
        }

        if expected_norm in number_words:
            word = number_words[expected_norm]
            if word in got_norm:
                return True

        # Reverse lookup
        for num, word in number_words.items():
            if expected_norm == word and num in got_norm.split():
                return True

        return False
