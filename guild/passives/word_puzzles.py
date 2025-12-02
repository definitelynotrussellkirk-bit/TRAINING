"""
Word Puzzles Passive - Syllable and word assembly eval module.

Tests: syllable assembly, syllable counting, word from clues.
Designed to match the SY (Syllacrostic) skill's core competencies.
"""

import random
from typing import List, Dict, Any, Optional

from guild.passives.base import PassiveModule, PassiveTier


# Simple word database with syllable breakdowns
# Format: (word, syllables, definition)
WORD_DATABASE = [
    # 2-syllable words (easier)
    ("APPLE", ["AP", "PLE"], "A common red or green fruit"),
    ("TABLE", ["TA", "BLE"], "Furniture with a flat top"),
    ("WATER", ["WA", "TER"], "Clear liquid essential for life"),
    ("PAPER", ["PA", "PER"], "Material made from wood pulp"),
    ("MUSIC", ["MU", "SIC"], "Art form using organized sounds"),
    ("TIGER", ["TI", "GER"], "Large striped wild cat"),
    ("LEMON", ["LEM", "ON"], "Yellow citrus fruit"),
    ("HAPPY", ["HAP", "PY"], "Feeling joy or pleasure"),
    ("ROBOT", ["RO", "BOT"], "Programmable machine"),
    ("CANDY", ["CAN", "DY"], "Sweet confection"),
    ("BUTTER", ["BUT", "TER"], "Dairy product spread"),
    ("GARDEN", ["GAR", "DEN"], "Plot for growing plants"),
    ("WINDOW", ["WIN", "DOW"], "Opening in a wall with glass"),
    ("SILVER", ["SIL", "VER"], "Shiny precious metal"),
    ("MONKEY", ["MON", "KEY"], "Primate with a tail"),
    # 3-syllable words (medium)
    ("ELEPHANT", ["EL", "E", "PHANT"], "Large gray mammal with trunk"),
    ("UMBRELLA", ["UM", "BREL", "LA"], "Device to keep rain off"),
    ("TOMATO", ["TO", "MA", "TO"], "Red fruit often used in salads"),
    ("COMPUTER", ["COM", "PU", "TER"], "Electronic device for processing data"),
    ("BANANA", ["BA", "NA", "NA"], "Yellow curved fruit"),
    ("ANIMAL", ["AN", "I", "MAL"], "Living creature that moves"),
    ("HOSPITAL", ["HOS", "PI", "TAL"], "Building where sick people are treated"),
    ("MAGAZINE", ["MAG", "A", "ZINE"], "Periodic publication with articles"),
    ("DANGEROUS", ["DAN", "GER", "OUS"], "Likely to cause harm"),
    ("BEAUTIFUL", ["BEAU", "TI", "FUL"], "Pleasing to look at"),
    # 4-syllable words (harder)
    ("DICTIONARY", ["DIC", "TION", "AR", "Y"], "Book of word definitions"),
    ("ALLIGATOR", ["AL", "LI", "GA", "TOR"], "Large reptile with long snout"),
    ("HELICOPTER", ["HEL", "I", "COP", "TER"], "Aircraft with rotating blades"),
    ("INFORMATION", ["IN", "FOR", "MA", "TION"], "Facts or knowledge"),
    ("CELEBRATION", ["CEL", "E", "BRA", "TION"], "Event to honor something"),
]


class WordPuzzlesPassive(PassiveModule):
    """Word puzzle and syllable assembly tests - matches SY skill."""

    id = "word_puzzles"
    name = "Word Puzzles"
    category = "reasoning"
    description = "Syllable assembly, word clues, counting"
    version = "1.0.0"

    # Core passive - matches SY skill, catches skill-specific regression
    tier = PassiveTier.CORE
    priority = 25

    lite_count = 5
    full_count = 30

    def generate_problems(self, count: int, seed: Optional[int] = None, level: int = 1) -> List[Dict[str, Any]]:
        """Generate word puzzles. Level parameter accepted but not yet used."""
        if seed is not None:
            random.seed(seed)

        problems = []
        problem_types = [
            self._syllable_assembly,
            self._syllable_assembly_shuffled,
            self._syllable_count,
            self._definition_to_word,
            self._word_from_bank,
        ]

        for i in range(count):
            generator = problem_types[i % len(problem_types)]
            problems.append(generator())

        return problems

    def _syllable_assembly(self) -> Dict[str, Any]:
        """Given syllables in order, assemble the word."""
        word, syllables, _ = random.choice(WORD_DATABASE)
        syllable_str = "-".join(syllables)

        return {
            "prompt": f"Assemble the word from these syllables (in order): {syllable_str}",
            "expected": word,
            "primitive_id": "syllable_assembly_ordered",
            "type": "assembly",
            "syllables": syllables,
        }

    def _syllable_assembly_shuffled(self) -> Dict[str, Any]:
        """Given shuffled syllables, assemble the word."""
        word, syllables, definition = random.choice(WORD_DATABASE)
        shuffled = syllables.copy()
        random.shuffle(shuffled)

        # Make sure it's actually shuffled for words with 3+ syllables
        while len(syllables) > 2 and shuffled == syllables:
            random.shuffle(shuffled)

        syllable_str = "-".join(shuffled)

        return {
            "prompt": f"Arrange these syllables to form a word meaning '{definition}': {syllable_str}",
            "expected": word,
            "primitive_id": "syllable_assembly_shuffled",
            "type": "assembly",
            "syllables": shuffled,
            "definition": definition,
        }

    def _syllable_count(self) -> Dict[str, Any]:
        """Count syllables in a word."""
        word, syllables, _ = random.choice(WORD_DATABASE)
        count = len(syllables)

        return {
            "prompt": f"How many syllables are in the word '{word}'?",
            "expected": str(count),
            "primitive_id": "syllable_count",
            "type": "counting",
            "word": word,
            "syllables": syllables,
        }

    def _definition_to_word(self) -> Dict[str, Any]:
        """Given definition, name the word."""
        word, syllables, definition = random.choice(WORD_DATABASE)

        return {
            "prompt": f"What word matches this definition: '{definition}'?",
            "expected": word,
            "primitive_id": "definition_to_word",
            "type": "vocabulary",
            "definition": definition,
        }

    def _word_from_bank(self) -> Dict[str, Any]:
        """
        Given a syllable bank with extra syllables, form the target word.
        This mimics the core SY skill mechanic.
        """
        word, syllables, definition = random.choice(WORD_DATABASE)

        # Add some distractor syllables
        distractors = ["ING", "TION", "LY", "ER", "EST", "PRE", "RE", "UN", "FUL"]
        num_distractors = random.randint(1, 3)
        selected_distractors = random.sample(distractors, num_distractors)

        # Create bank with target syllables + distractors
        bank = syllables.copy() + selected_distractors
        random.shuffle(bank)
        bank_str = ", ".join(bank)

        return {
            "prompt": f"From this syllable bank, form a word meaning '{definition}': [{bank_str}]",
            "expected": word,
            "primitive_id": "word_from_bank",
            "type": "assembly",
            "syllables": syllables,
            "bank": bank,
            "distractors": selected_distractors,
            "definition": definition,
        }

    def check_answer(self, expected: str, got: str) -> bool:
        """
        Check if model's answer matches expected.

        Handles:
        - Case insensitivity
        - Extra whitespace
        - Answer embedded in longer text
        """
        expected_norm = expected.strip().upper()
        got_norm = got.strip().upper()

        # Direct match
        if expected_norm == got_norm:
            return True

        # Check if expected appears in response (for verbose answers)
        if expected_norm in got_norm:
            return True

        # For number answers (syllable count)
        if expected_norm.isdigit():
            # Look for the number in the response
            import re
            numbers = re.findall(r'\b(\d+)\b', got_norm)
            if expected_norm in numbers:
                return True

            # Also check word forms
            word_numbers = {
                "1": ["ONE", "1"],
                "2": ["TWO", "2"],
                "3": ["THREE", "3"],
                "4": ["FOUR", "4"],
                "5": ["FIVE", "5"],
            }
            if expected_norm in word_numbers:
                for form in word_numbers[expected_norm]:
                    if form in got_norm:
                        return True

        return False
