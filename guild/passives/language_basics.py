"""
Language Basics Passive - Basic language understanding eval module.

Tests: synonyms, antonyms, plurals, sentence structure, pronoun resolution.
These are foundational language understanding skills.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule


# Word pairs for synonyms/antonyms
SYNONYMS = [
    ("big", "large"), ("small", "tiny"), ("fast", "quick"), ("slow", "sluggish"),
    ("happy", "joyful"), ("sad", "unhappy"), ("angry", "furious"), ("calm", "peaceful"),
    ("hot", "warm"), ("cold", "chilly"), ("bright", "luminous"), ("dark", "dim"),
    ("old", "ancient"), ("new", "fresh"), ("strong", "powerful"), ("weak", "feeble"),
    ("smart", "intelligent"), ("dumb", "foolish"), ("rich", "wealthy"), ("poor", "needy"),
    ("beautiful", "gorgeous"), ("ugly", "hideous"), ("clean", "spotless"), ("dirty", "filthy"),
    ("loud", "noisy"), ("quiet", "silent"), ("hard", "difficult"), ("easy", "simple"),
]

ANTONYMS = [
    ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"),
    ("young", "old"), ("new", "old"), ("light", "dark"), ("loud", "quiet"),
    ("hard", "soft"), ("wet", "dry"), ("clean", "dirty"), ("rich", "poor"),
    ("tall", "short"), ("wide", "narrow"), ("thick", "thin"), ("strong", "weak"),
    ("love", "hate"), ("push", "pull"), ("buy", "sell"), ("win", "lose"),
    ("open", "close"), ("start", "stop"), ("come", "go"), ("up", "down"),
]

# Irregular plurals
PLURALS = [
    ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("house", "houses"),
    ("child", "children"), ("man", "men"), ("woman", "women"), ("person", "people"),
    ("foot", "feet"), ("tooth", "teeth"), ("mouse", "mice"), ("goose", "geese"),
    ("fish", "fish"), ("sheep", "sheep"), ("deer", "deer"), ("series", "series"),
    ("leaf", "leaves"), ("knife", "knives"), ("wife", "wives"), ("life", "lives"),
    ("box", "boxes"), ("bus", "buses"), ("class", "classes"), ("potato", "potatoes"),
]

# Sentence subjects with answers
SUBJECT_SENTENCES = [
    ("The dog runs fast.", "dog"),
    ("My sister loves chocolate.", "sister"),
    ("The old man walked slowly.", "man"),
    ("Birds fly south in winter.", "Birds"),
    ("The red car stopped suddenly.", "car"),
    ("Our teacher explained the lesson.", "teacher"),
    ("The children played in the park.", "children"),
    ("Heavy rain fell all night.", "rain"),
]

# Pronoun resolution examples
PRONOUN_EXAMPLES = [
    ("John gave Mary a book. She was happy.", "Mary", "She"),
    ("The cat saw the mouse. It ran away.", "mouse", "It"),
    ("Sarah called her mother. She answered.", "mother", "She"),
    ("Tom and Jerry are friends. They play together.", "Tom and Jerry", "They"),
    ("The teacher praised the student. He smiled.", "student", "He"),
]


class LanguageBasicsPassive(PassiveModule):
    """Basic language understanding tests."""

    id = "language_basics"
    name = "Language Basics"
    category = "reasoning"
    description = "Synonyms, antonyms, plurals, sentence structure"
    version = "1.0.0"
    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """Generate language basics problems. Level parameter accepted but not yet used."""
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._synonym,
            self._antonym,
            self._pluralize,
            self._identify_subject,
            self._sentence_type,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _synonym(self) -> Dict[str, Any]:
        """Find a synonym for a word."""
        word1, word2 = random.choice(SYNONYMS)
        # Randomly choose which word to ask about
        if random.choice([True, False]):
            word1, word2 = word2, word1

        return {
            "prompt": f"What is a synonym for '{word1}'?",
            "expected": word2,
            "primitive_id": "synonym_easy",
            "type": "synonym",
            "word": word1,
        }

    def _antonym(self) -> Dict[str, Any]:
        """Find an antonym for a word."""
        word1, word2 = random.choice(ANTONYMS)
        # Randomly choose which word to ask about
        if random.choice([True, False]):
            word1, word2 = word2, word1

        return {
            "prompt": f"What is an antonym (opposite) of '{word1}'?",
            "expected": word2,
            "primitive_id": "antonym_easy",
            "type": "antonym",
            "word": word1,
        }

    def _pluralize(self) -> Dict[str, Any]:
        """Find the plural of a word."""
        singular, plural = random.choice(PLURALS)

        return {
            "prompt": f"What is the plural of '{singular}'?",
            "expected": plural,
            "primitive_id": "pluralize_regular",
            "type": "plural",
            "singular": singular,
        }

    def _identify_subject(self) -> Dict[str, Any]:
        """Identify the subject of a sentence."""
        sentence, subject = random.choice(SUBJECT_SENTENCES)

        return {
            "prompt": f"What is the subject of this sentence: '{sentence}'",
            "expected": subject,
            "primitive_id": "identify_subject",
            "type": "grammar",
            "sentence": sentence,
        }

    def _sentence_type(self) -> Dict[str, Any]:
        """Classify sentence type."""
        sentences = [
            ("The sky is blue.", "statement"),
            ("Is it raining outside?", "question"),
            ("Please close the door.", "command"),
            ("What a beautiful day!", "exclamation"),
            ("How are you?", "question"),
            ("She works at the hospital.", "statement"),
            ("Don't touch that!", "command"),
            ("Wow, that's amazing!", "exclamation"),
            ("Where did you go?", "question"),
            ("I love ice cream.", "statement"),
        ]

        sentence, sent_type = random.choice(sentences)

        return {
            "prompt": f"Is this sentence a statement, question, command, or exclamation: '{sentence}'",
            "expected": sent_type,
            "primitive_id": "sentence_type",
            "type": "classification",
            "sentence": sentence,
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

        # For synonym/antonym, accept if the expected word is present
        words = got_norm.split()
        if expected_norm in words:
            return True

        # Accept variants
        variants = {
            "statement": ["statement", "declarative"],
            "question": ["question", "interrogative"],
            "command": ["command", "imperative"],
            "exclamation": ["exclamation", "exclamatory"],
        }

        if expected_norm in variants:
            for var in variants[expected_norm]:
                if var in got_norm:
                    return True

        return False
