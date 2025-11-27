#!/usr/bin/env python3
"""
Skill Evaluators - Pluggable evaluation adapters for different skills.

Each skill needs:
1. generate_problems(level, count) → problems list
2. get_prompt(problem) → prompt string
3. get_expected(problem) → expected answer
4. extract_answer(response) → model's answer
5. check_correct(expected, actual) → bool

This allows the eval loop to work with any skill without modification.
"""

import re
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SkillEvaluator(ABC):
    """Base class for skill evaluators."""

    skill_name: str = "base"

    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout

    @abstractmethod
    def generate_problems(self, level: int, count: int) -> List[Dict]:
        """Generate test problems from the skill API."""
        pass

    @abstractmethod
    def get_prompt(self, problem: Dict) -> str:
        """Extract the prompt to send to the model."""
        pass

    @abstractmethod
    def get_expected(self, problem: Dict) -> Any:
        """Extract the expected answer."""
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> Any:
        """Extract the answer from model's response."""
        pass

    @abstractmethod
    def check_correct(self, expected: Any, actual: Any) -> Tuple[bool, float]:
        """
        Check if actual matches expected.

        Returns:
            (is_correct, partial_score)
        """
        pass

    def health_check(self) -> bool:
        """Check if skill API is available."""
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


class BinaryEvaluator(SkillEvaluator):
    """Evaluator for Binary arithmetic skill."""

    skill_name = "binary"

    def generate_problems(self, level: int, count: int) -> List[Dict]:
        """Generate binary problems from Bin API."""
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={"level": level, "count": count},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("samples", [])
        except Exception as e:
            logger.error(f"Failed to generate binary problems: {e}")
            return []

    def get_prompt(self, problem: Dict) -> str:
        """Get prompt from binary sample."""
        return problem.get("user_prompt", "")

    def get_expected(self, problem: Dict) -> str:
        """Get expected answer from binary sample."""
        # The full response is the expected answer
        return problem.get("assistant_response", "")

    def extract_answer(self, response: str) -> str:
        """
        Extract the answer from model's response.

        Binary answers are typically in format:
        - "add(①, ①) = ①⓪"
        - "①⓪" (just the result)
        - With verification: "result\nVerification: ..."
        """
        if not response:
            return ""

        # Clean up response - remove <think> tags if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()

        # Take first line if multiline
        first_line = response.split('\n')[0].strip()

        return first_line

    def check_correct(self, expected: str, actual: str) -> Tuple[bool, float]:
        """
        Check if binary answer is correct.

        Compares the results, handling:
        - Full equation match: "add(①, ①) = ①⓪"
        - Result only match: "①⓪"
        - Verification suffix ignored
        """
        if not expected or not actual:
            return False, 0.0

        # Extract the result (part after = if present)
        def extract_result(s: str) -> str:
            # Remove verification line
            s = s.split('\n')[0].strip()
            # Get part after =
            if '=' in s:
                s = s.split('=')[-1].strip()
            return s

        expected_result = extract_result(expected)
        actual_result = extract_result(actual)

        # Normalize: remove spaces, compare
        expected_norm = expected_result.replace(' ', '')
        actual_norm = actual_result.replace(' ', '')

        is_correct = expected_norm == actual_norm

        # Partial score: character overlap
        if not is_correct and expected_norm and actual_norm:
            matches = sum(1 for e, a in zip(expected_norm, actual_norm) if e == a)
            partial = matches / max(len(expected_norm), len(actual_norm))
        else:
            partial = 1.0 if is_correct else 0.0

        return is_correct, partial


class SylloEvaluator(SkillEvaluator):
    """Evaluator for SYLLO (syllable) skill."""

    skill_name = "syllo"

    def generate_problems(self, level: int, count: int) -> List[Dict]:
        """Generate SYLLO puzzles."""
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={"level": level, "count": count},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("puzzles", [])
        except Exception as e:
            logger.error(f"Failed to generate syllo problems: {e}")
            return []

    def get_prompt(self, problem: Dict) -> str:
        """Get prompt from SYLLO puzzle."""
        return problem.get("prompt", "")

    def get_expected(self, problem: Dict) -> List[str]:
        """Get expected words from SYLLO puzzle."""
        words = problem.get("words", [])
        return [w.get("label", "").lower() for w in words]

    def extract_answer(self, response: str) -> List[str]:
        """Extract words from model's JSON response."""
        if not response:
            return []

        # Clean up response
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()

        # Remove emoji preamble if present
        response = re.sub(r'^[🤔💭🧠💡🎯🔍🤨🧐⚡✨]+', '', response).strip()

        words = []

        # Try to parse as JSON
        try:
            data = json.loads(response)

            # Try various formats
            for key in ['solutions', 'solutions_map', 'sequence', 'answers']:
                if key in data:
                    items = data[key]
                    if isinstance(items, dict):
                        items = list(items.values())
                    for item in items:
                        if isinstance(item, dict):
                            word = item.get('word', item.get('answer', ''))
                        else:
                            word = str(item)
                        if word:
                            words.append(word.lower())
                    if words:
                        return words
        except json.JSONDecodeError:
            pass

        # Fallback: regex for capitalized words
        caps = re.findall(r'\b([A-Z]{4,})\b', response)
        if caps:
            return [w.lower() for w in caps]

        return words

    def check_correct(self, expected: List[str], actual: List[str]) -> Tuple[bool, float]:
        """Check if word lists match (order doesn't matter)."""
        if not expected:
            return False, 0.0

        expected_set = set(expected)
        actual_set = set(actual)

        correct_count = len(expected_set & actual_set)
        total = len(expected_set)

        partial = correct_count / total if total > 0 else 0.0
        is_correct = expected_set == actual_set

        return is_correct, partial


# =============================================================================
# EVALUATOR REGISTRY
# =============================================================================

SKILL_EVALUATORS = {
    "binary": {
        "class": BinaryEvaluator,
        "default_url": "http://localhost:8090",
        "description": "Binary arithmetic with circled notation",
    },
    "syllo": {
        "class": SylloEvaluator,
        "default_url": "http://localhost:8080",
        "description": "Syllable puzzles",
    },
}


def get_evaluator(skill: str, api_url: str = None) -> SkillEvaluator:
    """
    Get an evaluator for a skill.

    Args:
        skill: Skill name ("binary", "syllo")
        api_url: Optional custom API URL

    Returns:
        SkillEvaluator instance
    """
    if skill not in SKILL_EVALUATORS:
        raise ValueError(f"Unknown skill: {skill}. Available: {list(SKILL_EVALUATORS.keys())}")

    config = SKILL_EVALUATORS[skill]
    url = api_url or config["default_url"]

    return config["class"](api_url=url)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Skill Evaluators Test")
    print("=" * 50)

    # Test Binary evaluator
    print("\n--- Binary Evaluator ---")
    bin_eval = get_evaluator("binary")

    if bin_eval.health_check():
        print("✅ Bin API is up")

        problems = bin_eval.generate_problems(level=1, count=2)
        print(f"Generated {len(problems)} problems")

        for p in problems[:1]:
            prompt = bin_eval.get_prompt(p)
            expected = bin_eval.get_expected(p)
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Expected: {expected[:50]}...")

            # Test answer checking
            correct, partial = bin_eval.check_correct(expected, expected)
            print(f"  Self-check: correct={correct}, partial={partial}")
    else:
        print("❌ Bin API is down")

    # Test Syllo evaluator
    print("\n--- Syllo Evaluator ---")
    syllo_eval = get_evaluator("syllo")

    if syllo_eval.health_check():
        print("✅ Sy API is up")
    else:
        print("❌ Sy API is down (expected - we disabled it)")
