"""
Counting Passive - Enumeration and frequency tasks.

Tests: letter count, vowel count, word count, digit count.
"""

import random
import string
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


class CountingPassive(PassiveModule):
    """Counting and enumeration tests."""

    id = "counting"
    name = "Counting"
    category = "counting"
    description = "Letter count, vowel count, word count, digit count"
    lite_count = 5
    full_count = 30

    VOWELS = set('aeiouAEIOU')
    WORDS = [
        "apple", "banana", "cherry", "dragon", "elephant",
        "flower", "guitar", "harmony", "island", "jungle",
        "kitchen", "library", "mountain", "notebook", "orange",
        "penguin", "quantum", "rainbow", "sunshine", "thunder",
    ]

    def generate_problems(self, count: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._letter_count,
            self._vowel_count,
            self._word_count,
            self._specific_letter,
            self._consonant_count,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _letter_count(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        count = len(word)
        return {
            "prompt": f"How many letters are in the word '{word}'?",
            "expected": str(count),
            "type": "letter_count",
        }

    def _vowel_count(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        count = sum(1 for c in word if c in self.VOWELS)
        return {
            "prompt": f"How many vowels are in the word '{word}'?",
            "expected": str(count),
            "type": "vowel_count",
        }

    def _word_count(self) -> Dict[str, Any]:
        num_words = random.randint(3, 7)
        words = random.sample(self.WORDS, num_words)
        sentence = " ".join(words)
        return {
            "prompt": f"How many words are in this sentence: '{sentence}'?",
            "expected": str(num_words),
            "type": "word_count",
        }

    def _specific_letter(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        # Pick a letter that exists in the word
        letter = random.choice(word)
        count = word.lower().count(letter.lower())
        return {
            "prompt": f"How many times does the letter '{letter}' appear in '{word}'?",
            "expected": str(count),
            "type": "specific_letter",
        }

    def _consonant_count(self) -> Dict[str, Any]:
        word = random.choice(self.WORDS)
        count = sum(1 for c in word.lower() if c.isalpha() and c not in self.VOWELS)
        return {
            "prompt": f"How many consonants are in the word '{word}'?",
            "expected": str(count),
            "type": "consonant_count",
        }

    def check_answer(self, expected: str, got: str) -> bool:
        expected_norm = expected.strip()
        got_norm = got.strip()

        # Direct match
        if expected_norm == got_norm:
            return True

        # Check if number appears in response
        if expected_norm in got_norm:
            return True

        # Try to find number in response
        import re
        numbers = re.findall(r'\d+', got_norm)
        return expected_norm in numbers
