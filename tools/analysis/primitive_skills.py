#!/usr/bin/env python3
"""
Primitive Skills Generator and Evaluator

Generates validation data for testing fundamental cognitive skills.
Useful for measuring transfer learning effects and baseline capabilities.

Known Benchmark Databases:
- BIG-Bench (Google): https://github.com/google/BIG-bench
  Relevant tasks: simple_arithmetic, word_sorting, logical_deduction,
  elementary_math_qa, tracking_shuffled_objects

- bAbI (Facebook): https://research.fb.com/downloads/babi/
  20 QA tasks testing basic reasoning (counting, path finding, etc.)

- SVAMP: Simple arithmetic word problems
- AddSub: Addition/subtraction problems
- LAMA: Knowledge probing

Usage:
    # Generate validation data for all skills
    python3 primitive_skills.py generate --all --count 100

    # Generate for specific skill
    python3 primitive_skills.py generate --skill letter_count --count 200

    # List available skills
    python3 primitive_skills.py list
"""

import argparse
import json
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re


# =============================================================================
# SKILL REGISTRY
# =============================================================================

SKILL_CATEGORIES = {
    "counting": ["letter_count", "vowel_count", "word_count", "digit_count", "char_frequency"],
    "conversion": ["decimal_to_hex", "decimal_to_octal", "binary_to_decimal", "roman_numerals"],
    "string": ["reverse_word", "reverse_sentence", "first_n_chars", "palindrome_check"],
    "arithmetic": ["digit_sum", "even_odd", "modulo", "compare_numbers", "simple_addition"],
    "logic": ["boolean_and", "boolean_or", "boolean_xor"],
    "sequence": ["next_in_sequence", "alphabetical_order", "position_in_alphabet"],
    "set": ["membership", "unique_elements"],
}

# Flatten for easy lookup
ALL_SKILLS = [skill for skills in SKILL_CATEGORIES.values() for skill in skills]


# =============================================================================
# WORD LISTS
# =============================================================================

# Common English words by difficulty
EASY_WORDS = [
    "cat", "dog", "run", "sun", "hat", "pen", "cup", "box", "red", "big",
    "hot", "wet", "new", "old", "sad", "fun", "win", "top", "bed", "bus"
]

MEDIUM_WORDS = [
    "elephant", "computer", "beautiful", "mountain", "telephone", "adventure",
    "chocolate", "butterfly", "dangerous", "celebrate", "education", "fantastic",
    "hamburger", "important", "knowledge", "literature", "mysterious", "nightmare",
    "orchestra", "pineapple", "quarantine", "restaurant", "strawberry", "telephone"
]

HARD_WORDS = [
    "antidisestablishmentarianism", "supercalifragilisticexpialidocious",
    "pneumonoultramicroscopicsilicovolcanoconiosis", "hippopotomonstrosesquippedaliophobia",
    "pseudopseudohypoparathyroidism", "floccinaucinihilipilification",
    "honorificabilitudinity", "electroencephalographically", "incomprehensibilities",
    "counterrevolutionaries", "internationalization", "deinstitutionalization"
]

SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "How much wood would a woodchuck chuck",
    "Peter Piper picked a peck of pickled peppers",
    "I scream you scream we all scream for ice cream",
    "A stitch in time saves nine",
    "Actions speak louder than words",
    "Better late than never",
    "Every cloud has a silver lining",
    "Fortune favors the bold"
]


# =============================================================================
# SKILL DEFINITIONS
# =============================================================================

@dataclass
class SkillExample:
    """A single example for a skill."""
    prompt: str
    answer: str
    difficulty: str
    skill: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        return {
            "skill": self.skill,
            "difficulty": self.difficulty,
            "user_prompt": self.prompt,
            "expected_answer": self.answer,
            "metadata": self.metadata or {}
        }


class PrimitiveSkill(ABC):
    """Base class for primitive skills."""

    name: str = "base"
    category: str = "misc"
    description: str = "Base skill"

    @abstractmethod
    def generate(self, difficulty: str) -> SkillExample:
        """Generate a single example."""
        pass

    @abstractmethod
    def compare(self, model_output: str, expected: str) -> bool:
        """Compare model output to expected answer."""
        pass

    def extract_answer(self, text: str) -> str:
        """Extract answer from model output (strip think blocks, etc.)."""
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove emoji patterns
        text = re.sub(r'[🧠🎯🚦❌✓✗]+', '', text)
        return text.strip()


# =============================================================================
# COUNTING SKILLS
# =============================================================================

class LetterCount(PrimitiveSkill):
    name = "letter_count"
    category = "counting"
    description = "Count the number of letters in a word"

    def generate(self, difficulty: str) -> SkillExample:
        words = {"easy": EASY_WORDS, "medium": MEDIUM_WORDS, "hard": HARD_WORDS}
        word = random.choice(words.get(difficulty, EASY_WORDS))
        count = len(word)

        return SkillExample(
            prompt=f"How many letters are in the word '{word}'? Answer with just the number.",
            answer=str(count),
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        # Find first number in output
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class VowelCount(PrimitiveSkill):
    name = "vowel_count"
    category = "counting"
    description = "Count vowels in a word"

    def generate(self, difficulty: str) -> SkillExample:
        words = {"easy": EASY_WORDS, "medium": MEDIUM_WORDS, "hard": HARD_WORDS}
        word = random.choice(words.get(difficulty, EASY_WORDS))
        count = sum(1 for c in word.lower() if c in 'aeiou')

        return SkillExample(
            prompt=f"How many vowels (a, e, i, o, u) are in the word '{word}'? Answer with just the number.",
            answer=str(count),
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word, "vowels": [c for c in word.lower() if c in 'aeiou']}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class WordCount(PrimitiveSkill):
    name = "word_count"
    category = "counting"
    description = "Count words in a sentence"

    def generate(self, difficulty: str) -> SkillExample:
        if difficulty == "easy":
            words = random.sample(EASY_WORDS, random.randint(3, 5))
        elif difficulty == "medium":
            words = random.sample(EASY_WORDS + MEDIUM_WORDS[:10], random.randint(6, 10))
        else:
            sentence = random.choice(SENTENCES)
            words = sentence.split()

        sentence = " ".join(words)
        count = len(words)

        return SkillExample(
            prompt=f"How many words are in this sentence: \"{sentence}\"? Answer with just the number.",
            answer=str(count),
            difficulty=difficulty,
            skill=self.name,
            metadata={"sentence": sentence}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class DigitCount(PrimitiveSkill):
    name = "digit_count"
    category = "counting"
    description = "Count digits in a number"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (10, 999), "medium": (1000, 999999), "hard": (1000000, 999999999)}
        lo, hi = ranges.get(difficulty, (10, 999))
        number = random.randint(lo, hi)
        count = len(str(number))

        return SkillExample(
            prompt=f"How many digits are in the number {number}? Answer with just the number.",
            answer=str(count),
            difficulty=difficulty,
            skill=self.name,
            metadata={"number": number}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class CharFrequency(PrimitiveSkill):
    name = "char_frequency"
    category = "counting"
    description = "Count occurrences of a specific character"

    def generate(self, difficulty: str) -> SkillExample:
        # Pick words with repeated characters
        targets = {
            "easy": [("mississippi", "s"), ("banana", "a"), ("letter", "t")],
            "medium": [("abracadabra", "a"), ("bookkeeper", "e"), ("committee", "m")],
            "hard": [("supercalifragilisticexpialidocious", "i"), ("pneumonoultramicroscopicsilicovolcanoconiosis", "o")]
        }
        word, char = random.choice(targets.get(difficulty, targets["easy"]))
        count = word.lower().count(char.lower())

        return SkillExample(
            prompt=f"How many times does the letter '{char}' appear in '{word}'? Answer with just the number.",
            answer=str(count),
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word, "char": char}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


# =============================================================================
# CONVERSION SKILLS
# =============================================================================

class DecimalToHex(PrimitiveSkill):
    name = "decimal_to_hex"
    category = "conversion"
    description = "Convert decimal to hexadecimal"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (1, 15), "medium": (16, 255), "hard": (256, 4095)}
        lo, hi = ranges.get(difficulty, (1, 15))
        number = random.randint(lo, hi)
        hex_val = hex(number)[2:].upper()

        return SkillExample(
            prompt=f"Convert the decimal number {number} to hexadecimal. Answer with just the hex value (no 0x prefix).",
            answer=hex_val,
            difficulty=difficulty,
            skill=self.name,
            metadata={"decimal": number}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).upper()
        # Remove common prefixes
        output = output.replace("0X", "").replace("0x", "").strip()
        # Find hex pattern
        match = re.search(r'[0-9A-F]+', output, re.IGNORECASE)
        if match:
            return match.group().upper() == expected.upper()
        return False


class DecimalToOctal(PrimitiveSkill):
    name = "decimal_to_octal"
    category = "conversion"
    description = "Convert decimal to octal"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (1, 7), "medium": (8, 63), "hard": (64, 511)}
        lo, hi = ranges.get(difficulty, (1, 7))
        number = random.randint(lo, hi)
        oct_val = oct(number)[2:]

        return SkillExample(
            prompt=f"Convert the decimal number {number} to octal. Answer with just the octal value (no 0o prefix).",
            answer=oct_val,
            difficulty=difficulty,
            skill=self.name,
            metadata={"decimal": number}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        output = output.replace("0o", "").replace("0O", "").strip()
        match = re.search(r'[0-7]+', output)
        if match:
            return match.group() == expected
        return False


class BinaryToDecimal(PrimitiveSkill):
    name = "binary_to_decimal"
    category = "conversion"
    description = "Convert binary to decimal"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (1, 15), "medium": (16, 255), "hard": (256, 1023)}
        lo, hi = ranges.get(difficulty, (1, 15))
        number = random.randint(lo, hi)
        binary = bin(number)[2:]

        return SkillExample(
            prompt=f"Convert the binary number {binary} to decimal. Answer with just the number.",
            answer=str(number),
            difficulty=difficulty,
            skill=self.name,
            metadata={"binary": binary}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class RomanNumerals(PrimitiveSkill):
    name = "roman_numerals"
    category = "conversion"
    description = "Convert decimal to Roman numerals"

    ROMAN_MAP = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    def _to_roman(self, num: int) -> str:
        result = []
        for value, numeral in self.ROMAN_MAP:
            while num >= value:
                result.append(numeral)
                num -= value
        return ''.join(result)

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (1, 10), "medium": (11, 100), "hard": (101, 999)}
        lo, hi = ranges.get(difficulty, (1, 10))
        number = random.randint(lo, hi)
        roman = self._to_roman(number)

        return SkillExample(
            prompt=f"Convert {number} to Roman numerals. Answer with just the Roman numeral.",
            answer=roman,
            difficulty=difficulty,
            skill=self.name,
            metadata={"decimal": number}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).upper()
        # Find Roman numeral pattern
        match = re.search(r'[IVXLCDM]+', output)
        if match:
            return match.group() == expected
        return False


# =============================================================================
# STRING SKILLS
# =============================================================================

class ReverseWord(PrimitiveSkill):
    name = "reverse_word"
    category = "string"
    description = "Reverse a word"

    def generate(self, difficulty: str) -> SkillExample:
        words = {"easy": EASY_WORDS, "medium": MEDIUM_WORDS, "hard": HARD_WORDS[:6]}
        word = random.choice(words.get(difficulty, EASY_WORDS))
        reversed_word = word[::-1]

        return SkillExample(
            prompt=f"Reverse the word '{word}'. Answer with just the reversed word.",
            answer=reversed_word,
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower().strip()
        # Get first word-like token
        match = re.search(r'[a-zA-Z]+', output)
        if match:
            return match.group().lower() == expected.lower()
        return output == expected.lower()


class ReverseSentence(PrimitiveSkill):
    name = "reverse_sentence"
    category = "string"
    description = "Reverse word order in a sentence"

    def generate(self, difficulty: str) -> SkillExample:
        if difficulty == "easy":
            words = random.sample(EASY_WORDS, 3)
        elif difficulty == "medium":
            words = random.sample(EASY_WORDS, 5)
        else:
            words = random.choice(SENTENCES).split()[:7]

        sentence = " ".join(words)
        reversed_sentence = " ".join(reversed(words))

        return SkillExample(
            prompt=f"Reverse the word order in: \"{sentence}\". Answer with just the reversed sentence.",
            answer=reversed_sentence.lower(),
            difficulty=difficulty,
            skill=self.name,
            metadata={"sentence": sentence}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower().strip()
        # Remove punctuation for comparison
        output = re.sub(r'[^\w\s]', '', output)
        expected = re.sub(r'[^\w\s]', '', expected)
        return ' '.join(output.split()) == ' '.join(expected.split())


class FirstNChars(PrimitiveSkill):
    name = "first_n_chars"
    category = "string"
    description = "Extract first N characters from a word"

    def generate(self, difficulty: str) -> SkillExample:
        words = {"easy": EASY_WORDS, "medium": MEDIUM_WORDS, "hard": HARD_WORDS[:6]}
        word = random.choice(words.get(difficulty, EASY_WORDS))
        n = min(random.randint(2, 4), len(word))
        result = word[:n]

        return SkillExample(
            prompt=f"What are the first {n} letters of '{word}'? Answer with just the letters.",
            answer=result,
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word, "n": n}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower().strip()
        match = re.search(r'[a-zA-Z]+', output)
        if match:
            return match.group().lower() == expected.lower()
        return output == expected.lower()


class PalindromeCheck(PrimitiveSkill):
    name = "palindrome_check"
    category = "string"
    description = "Check if a word is a palindrome"

    PALINDROMES = ["racecar", "level", "radar", "rotor", "civic", "kayak", "madam", "refer", "noon", "mom", "dad", "pop"]
    NON_PALINDROMES = ["hello", "world", "python", "program", "computer", "testing", "example"]

    def generate(self, difficulty: str) -> SkillExample:
        is_palindrome = random.choice([True, False])
        if is_palindrome:
            word = random.choice(self.PALINDROMES)
            answer = "yes"
        else:
            word = random.choice(self.NON_PALINDROMES)
            answer = "no"

        return SkillExample(
            prompt=f"Is '{word}' a palindrome? Answer with just 'yes' or 'no'.",
            answer=answer,
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word, "is_palindrome": is_palindrome}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        if "yes" in output or "true" in output:
            return expected == "yes"
        if "no" in output or "false" in output:
            return expected == "no"
        return False


# =============================================================================
# ARITHMETIC SKILLS
# =============================================================================

class DigitSum(PrimitiveSkill):
    name = "digit_sum"
    category = "arithmetic"
    description = "Sum all digits of a number"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (10, 99), "medium": (100, 9999), "hard": (10000, 999999)}
        lo, hi = ranges.get(difficulty, (10, 99))
        number = random.randint(lo, hi)
        digit_sum = sum(int(d) for d in str(number))

        return SkillExample(
            prompt=f"What is the sum of all digits in {number}? Answer with just the number.",
            answer=str(digit_sum),
            difficulty=difficulty,
            skill=self.name,
            metadata={"number": number, "digits": list(str(number))}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class EvenOdd(PrimitiveSkill):
    name = "even_odd"
    category = "arithmetic"
    description = "Determine if a number is even or odd"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (1, 20), "medium": (21, 1000), "hard": (1001, 100000)}
        lo, hi = ranges.get(difficulty, (1, 20))
        number = random.randint(lo, hi)
        answer = "even" if number % 2 == 0 else "odd"

        return SkillExample(
            prompt=f"Is {number} even or odd? Answer with just 'even' or 'odd'.",
            answer=answer,
            difficulty=difficulty,
            skill=self.name,
            metadata={"number": number}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        if "even" in output:
            return expected == "even"
        if "odd" in output:
            return expected == "odd"
        return False


class Modulo(PrimitiveSkill):
    name = "modulo"
    category = "arithmetic"
    description = "Calculate remainder (modulo operation)"

    def generate(self, difficulty: str) -> SkillExample:
        if difficulty == "easy":
            a = random.randint(5, 20)
            b = random.randint(2, 5)
        elif difficulty == "medium":
            a = random.randint(20, 100)
            b = random.randint(3, 10)
        else:
            a = random.randint(100, 1000)
            b = random.randint(7, 20)

        result = a % b

        return SkillExample(
            prompt=f"What is {a} mod {b} (the remainder when {a} is divided by {b})? Answer with just the number.",
            answer=str(result),
            difficulty=difficulty,
            skill=self.name,
            metadata={"a": a, "b": b}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class CompareNumbers(PrimitiveSkill):
    name = "compare_numbers"
    category = "arithmetic"
    description = "Compare two numbers"

    def generate(self, difficulty: str) -> SkillExample:
        ranges = {"easy": (1, 100), "medium": (100, 10000), "hard": (10000, 1000000)}
        lo, hi = ranges.get(difficulty, (1, 100))
        a = random.randint(lo, hi)
        b = random.randint(lo, hi)
        while a == b:
            b = random.randint(lo, hi)

        larger = max(a, b)

        return SkillExample(
            prompt=f"Which is larger: {a} or {b}? Answer with just the number.",
            answer=str(larger),
            difficulty=difficulty,
            skill=self.name,
            metadata={"a": a, "b": b}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class SimpleAddition(PrimitiveSkill):
    name = "simple_addition"
    category = "arithmetic"
    description = "Add two numbers"

    def generate(self, difficulty: str) -> SkillExample:
        if difficulty == "easy":
            a = random.randint(1, 10)
            b = random.randint(1, 10)
        elif difficulty == "medium":
            a = random.randint(10, 100)
            b = random.randint(10, 100)
        else:
            a = random.randint(100, 1000)
            b = random.randint(100, 1000)

        result = a + b

        return SkillExample(
            prompt=f"What is {a} + {b}? Answer with just the number.",
            answer=str(result),
            difficulty=difficulty,
            skill=self.name,
            metadata={"a": a, "b": b}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


# =============================================================================
# LOGIC SKILLS
# =============================================================================

class BooleanAnd(PrimitiveSkill):
    name = "boolean_and"
    category = "logic"
    description = "Evaluate boolean AND operation"

    def generate(self, difficulty: str) -> SkillExample:
        a = random.choice([True, False])
        b = random.choice([True, False])
        result = a and b

        a_str = "True" if a else "False"
        b_str = "True" if b else "False"
        answer = "true" if result else "false"

        return SkillExample(
            prompt=f"What is {a_str} AND {b_str}? Answer with just 'true' or 'false'.",
            answer=answer,
            difficulty=difficulty,
            skill=self.name,
            metadata={"a": a, "b": b}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        if "true" in output:
            return expected == "true"
        if "false" in output:
            return expected == "false"
        return False


class BooleanOr(PrimitiveSkill):
    name = "boolean_or"
    category = "logic"
    description = "Evaluate boolean OR operation"

    def generate(self, difficulty: str) -> SkillExample:
        a = random.choice([True, False])
        b = random.choice([True, False])
        result = a or b

        a_str = "True" if a else "False"
        b_str = "True" if b else "False"
        answer = "true" if result else "false"

        return SkillExample(
            prompt=f"What is {a_str} OR {b_str}? Answer with just 'true' or 'false'.",
            answer=answer,
            difficulty=difficulty,
            skill=self.name,
            metadata={"a": a, "b": b}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        if "true" in output:
            return expected == "true"
        if "false" in output:
            return expected == "false"
        return False


class BooleanXor(PrimitiveSkill):
    name = "boolean_xor"
    category = "logic"
    description = "Evaluate boolean XOR operation"

    def generate(self, difficulty: str) -> SkillExample:
        a = random.choice([True, False])
        b = random.choice([True, False])
        result = a != b  # XOR

        a_str = "True" if a else "False"
        b_str = "True" if b else "False"
        answer = "true" if result else "false"

        return SkillExample(
            prompt=f"What is {a_str} XOR {b_str}? Answer with just 'true' or 'false'.",
            answer=answer,
            difficulty=difficulty,
            skill=self.name,
            metadata={"a": a, "b": b}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        if "true" in output:
            return expected == "true"
        if "false" in output:
            return expected == "false"
        return False


# =============================================================================
# SEQUENCE SKILLS
# =============================================================================

class NextInSequence(PrimitiveSkill):
    name = "next_in_sequence"
    category = "sequence"
    description = "Find the next number in a sequence"

    SEQUENCES = {
        "easy": [
            ([2, 4, 6, 8], 10, "+2"),
            ([1, 2, 3, 4], 5, "+1"),
            ([5, 10, 15, 20], 25, "+5"),
            ([10, 20, 30, 40], 50, "+10"),
        ],
        "medium": [
            ([2, 4, 8, 16], 32, "*2"),
            ([1, 1, 2, 3, 5], 8, "fibonacci"),
            ([1, 4, 9, 16], 25, "squares"),
            ([3, 6, 12, 24], 48, "*2"),
        ],
        "hard": [
            ([1, 8, 27, 64], 125, "cubes"),
            ([2, 6, 12, 20], 30, "n*(n+1)"),
            ([1, 3, 6, 10, 15], 21, "triangular"),
        ]
    }

    def generate(self, difficulty: str) -> SkillExample:
        seq_data = random.choice(self.SEQUENCES.get(difficulty, self.SEQUENCES["easy"]))
        sequence, answer, pattern = seq_data

        seq_str = ", ".join(map(str, sequence))

        return SkillExample(
            prompt=f"What comes next in this sequence: {seq_str}, ? Answer with just the number.",
            answer=str(answer),
            difficulty=difficulty,
            skill=self.name,
            metadata={"sequence": sequence, "pattern": pattern}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


class AlphabeticalOrder(PrimitiveSkill):
    name = "alphabetical_order"
    category = "sequence"
    description = "Sort letters alphabetically"

    def generate(self, difficulty: str) -> SkillExample:
        if difficulty == "easy":
            n = 3
        elif difficulty == "medium":
            n = 5
        else:
            n = 7

        letters = random.sample(string.ascii_lowercase, n)
        sorted_letters = sorted(letters)

        return SkillExample(
            prompt=f"Sort these letters alphabetically: {', '.join(letters)}. Answer with just the sorted letters separated by commas.",
            answer=", ".join(sorted_letters),
            difficulty=difficulty,
            skill=self.name,
            metadata={"letters": letters}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        # Extract all letters
        output_letters = re.findall(r'[a-z]', output)
        expected_letters = re.findall(r'[a-z]', expected)
        return output_letters == expected_letters


class PositionInAlphabet(PrimitiveSkill):
    name = "position_in_alphabet"
    category = "sequence"
    description = "Find position of letter in alphabet"

    def generate(self, difficulty: str) -> SkillExample:
        letter = random.choice(string.ascii_lowercase)
        position = ord(letter) - ord('a') + 1

        return SkillExample(
            prompt=f"What position is the letter '{letter}' in the alphabet? Answer with just the number.",
            answer=str(position),
            difficulty=difficulty,
            skill=self.name,
            metadata={"letter": letter}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


# =============================================================================
# SET SKILLS
# =============================================================================

class Membership(PrimitiveSkill):
    name = "membership"
    category = "set"
    description = "Check if element is in set"

    def generate(self, difficulty: str) -> SkillExample:
        word = random.choice(MEDIUM_WORDS)
        char = random.choice(string.ascii_lowercase)
        is_member = char in word.lower()
        answer = "yes" if is_member else "no"

        return SkillExample(
            prompt=f"Is the letter '{char}' in the word '{word}'? Answer with just 'yes' or 'no'.",
            answer=answer,
            difficulty=difficulty,
            skill=self.name,
            metadata={"word": word, "char": char, "is_member": is_member}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output).lower()
        if "yes" in output:
            return expected == "yes"
        if "no" in output:
            return expected == "no"
        return False


class UniqueElements(PrimitiveSkill):
    name = "unique_elements"
    category = "set"
    description = "Count unique elements in a list"

    def generate(self, difficulty: str) -> SkillExample:
        if difficulty == "easy":
            elements = [random.randint(1, 5) for _ in range(5)]
        elif difficulty == "medium":
            elements = [random.randint(1, 10) for _ in range(8)]
        else:
            elements = [random.randint(1, 10) for _ in range(12)]

        unique_count = len(set(elements))

        return SkillExample(
            prompt=f"How many unique numbers are in this list: {elements}? Answer with just the number.",
            answer=str(unique_count),
            difficulty=difficulty,
            skill=self.name,
            metadata={"elements": elements, "unique": list(set(elements))}
        )

    def compare(self, model_output: str, expected: str) -> bool:
        output = self.extract_answer(model_output)
        match = re.search(r'\d+', output)
        if match:
            return match.group() == expected
        return False


# =============================================================================
# SKILL REGISTRY
# =============================================================================

SKILL_CLASSES = {
    # Counting
    "letter_count": LetterCount,
    "vowel_count": VowelCount,
    "word_count": WordCount,
    "digit_count": DigitCount,
    "char_frequency": CharFrequency,
    # Conversion
    "decimal_to_hex": DecimalToHex,
    "decimal_to_octal": DecimalToOctal,
    "binary_to_decimal": BinaryToDecimal,
    "roman_numerals": RomanNumerals,
    # String
    "reverse_word": ReverseWord,
    "reverse_sentence": ReverseSentence,
    "first_n_chars": FirstNChars,
    "palindrome_check": PalindromeCheck,
    # Arithmetic
    "digit_sum": DigitSum,
    "even_odd": EvenOdd,
    "modulo": Modulo,
    "compare_numbers": CompareNumbers,
    "simple_addition": SimpleAddition,
    # Logic
    "boolean_and": BooleanAnd,
    "boolean_or": BooleanOr,
    "boolean_xor": BooleanXor,
    # Sequence
    "next_in_sequence": NextInSequence,
    "alphabetical_order": AlphabeticalOrder,
    "position_in_alphabet": PositionInAlphabet,
    # Set
    "membership": Membership,
    "unique_elements": UniqueElements,
}


def get_skill(name: str) -> PrimitiveSkill:
    """Get a skill instance by name."""
    if name not in SKILL_CLASSES:
        raise ValueError(f"Unknown skill: {name}. Available: {list(SKILL_CLASSES.keys())}")
    return SKILL_CLASSES[name]()


def generate_validation_set(
    skill_name: str,
    count_per_difficulty: int = 50,
    difficulties: List[str] = None
) -> List[Dict]:
    """Generate validation examples for a skill."""
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    skill = get_skill(skill_name)
    examples = []

    for difficulty in difficulties:
        for _ in range(count_per_difficulty):
            example = skill.generate(difficulty)
            examples.append(example.to_dict())

    return examples


def save_validation_set(
    skill_name: str,
    base_dir: Path,
    count_per_difficulty: int = 50
):
    """Generate and save validation set for a skill."""
    examples = generate_validation_set(skill_name, count_per_difficulty)

    # Group by difficulty
    by_difficulty = {"easy": [], "medium": [], "hard": []}
    for ex in examples:
        by_difficulty[ex["difficulty"]].append(ex)

    # Save each difficulty
    skill_dir = base_dir / "data" / "validation" / "primitives" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    for difficulty, diff_examples in by_difficulty.items():
        filepath = skill_dir / f"val_{skill_name}_{difficulty}_{len(diff_examples)}.jsonl"
        with open(filepath, 'w') as f:
            for ex in diff_examples:
                f.write(json.dumps(ex) + '\n')
        print(f"  Saved {len(diff_examples)} {difficulty} examples to {filepath}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Primitive Skills Generator")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List available skills")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate validation data")
    gen_parser.add_argument("--skill", help="Skill to generate (or 'all')")
    gen_parser.add_argument("--all", action="store_true", help="Generate all skills")
    gen_parser.add_argument("--count", type=int, default=50, help="Examples per difficulty")
    gen_parser.add_argument("--base-dir", default=str(Path(__file__).parent.parent.parent))
    gen_parser.add_argument("--category", help="Generate all skills in category")

    # Test command (single example)
    test_parser = subparsers.add_parser("test", help="Generate single test example")
    test_parser.add_argument("--skill", required=True, help="Skill to test")
    test_parser.add_argument("--difficulty", default="medium")

    args = parser.parse_args()

    if args.command == "list":
        print("\n=== PRIMITIVE SKILLS ===\n")
        for category, skills in SKILL_CATEGORIES.items():
            print(f"{category.upper()}:")
            for skill in skills:
                skill_obj = get_skill(skill)
                print(f"  - {skill}: {skill_obj.description}")
            print()
        print(f"Total: {len(ALL_SKILLS)} skills")

    elif args.command == "generate":
        base_dir = Path(args.base_dir)

        if args.all:
            skills_to_gen = ALL_SKILLS
        elif args.category:
            skills_to_gen = SKILL_CATEGORIES.get(args.category, [])
        elif args.skill:
            skills_to_gen = [args.skill]
        else:
            print("Specify --skill, --category, or --all")
            return

        print(f"\nGenerating {len(skills_to_gen)} skills with {args.count} examples per difficulty...\n")

        for skill in skills_to_gen:
            print(f"Generating: {skill}")
            save_validation_set(skill, base_dir, args.count)

        print(f"\nDone! Validation data saved to {base_dir}/data/validation/primitives/")

    elif args.command == "test":
        skill = get_skill(args.skill)
        example = skill.generate(args.difficulty)
        print(f"\n=== {args.skill.upper()} ({args.difficulty}) ===")
        print(f"Prompt: {example.prompt}")
        print(f"Answer: {example.answer}")
        print(f"Metadata: {example.metadata}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
