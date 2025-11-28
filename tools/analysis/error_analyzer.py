#!/usr/bin/env python3
"""
Error Analyzer Module
=====================

Deep analysis of model outputs to categorize errors and identify partial matches.
Used by adversarial mining and evaluation systems for detailed error reporting.

Error Categories:
- SCHEMA_ERROR: Invalid JSON or missing required keys
- PARTIAL_CORRECT: Some solutions correct, others wrong
- THINKING_LEAK: Model leaked <think> tags in output
- HALLUCINATION: Made up words not in expected solutions
- SYLLABLE_MISMATCH: Wrong syllables assigned to a word
- MISSING_WORDS: Fewer solutions than expected
- EXTRA_WORDS: More solutions than expected
- CASE_ERROR: Correct answer but wrong case
- CORRECT: Fully correct answer

Usage:
    from tools.analysis.error_analyzer import ErrorAnalyzer, analyze_syllable_response

    analyzer = ErrorAnalyzer()
    result = analyzer.analyze_syllable(model_output, expected_output)
    print(result.category)  # 'PARTIAL_CORRECT'
    print(result.partial_score)  # 3/5 words correct
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum


class ErrorCategory(Enum):
    """Error categories for classification"""
    CORRECT = "CORRECT"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    PARTIAL_CORRECT = "PARTIAL_CORRECT"
    THINKING_LEAK = "THINKING_LEAK"
    HALLUCINATION = "HALLUCINATION"
    SYLLABLE_MISMATCH = "SYLLABLE_MISMATCH"
    MISSING_WORDS = "MISSING_WORDS"
    EXTRA_WORDS = "EXTRA_WORDS"
    CASE_ERROR = "CASE_ERROR"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    UNKNOWN = "UNKNOWN"


@dataclass
class AnalysisResult:
    """Result of error analysis"""
    category: ErrorCategory
    is_correct: bool
    partial_score: float = 0.0  # 0.0 to 1.0

    # Detailed breakdown
    correct_words: List[str] = field(default_factory=list)
    wrong_words: List[str] = field(default_factory=list)
    missing_words: List[str] = field(default_factory=list)
    extra_words: List[str] = field(default_factory=list)

    # Schema issues
    schema_valid: bool = True
    schema_errors: List[str] = field(default_factory=list)

    # Thinking tag issues
    has_thinking_tags: bool = False
    thinking_content: str = ""

    # Detailed analysis
    analysis_notes: List[str] = field(default_factory=list)

    # Raw data for debugging
    model_solutions: List[Dict] = field(default_factory=list)
    expected_solutions: List[Dict] = field(default_factory=list)


class ErrorAnalyzer:
    """Analyzes model outputs for detailed error categorization"""

    def __init__(self):
        self.thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        self.json_pattern = re.compile(r'\{[^{}]*"solutions"[^{}]*\}|\{.*\}', re.DOTALL)

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text"""
        if not text or not text.strip():
            return None

        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        match = self.json_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def check_thinking_tags(self, text: str) -> tuple[bool, str]:
        """Check for thinking tag leakage"""
        matches = self.thinking_pattern.findall(text)
        if matches:
            return True, " ".join(matches)

        # Also check for unclosed tags
        if '<think>' in text.lower() or '</think>' in text.lower():
            return True, text.split('<think>')[-1].split('</think>')[0] if '<think>' in text.lower() else ""

        return False, ""

    def validate_schema(self, data: Dict) -> tuple[bool, List[str]]:
        """Validate SYLLABLE schema"""
        errors = []

        if not isinstance(data, dict):
            errors.append("Response is not a JSON object")
            return False, errors

        if "solutions" not in data:
            errors.append("Missing 'solutions' key")
            return False, errors

        solutions = data.get("solutions", [])
        if not isinstance(solutions, list):
            errors.append("'solutions' is not a list")
            return False, errors

        for i, sol in enumerate(solutions):
            if not isinstance(sol, dict):
                errors.append(f"Solution {i} is not a dict")
                continue
            if "answer" not in sol:
                errors.append(f"Solution {i} missing 'answer' key")
            if "syllables" not in sol and "ans_num" not in sol:
                errors.append(f"Solution {i} missing identifying keys")

        return len(errors) == 0, errors

    def compare_solutions(
        self,
        model_solutions: List[Dict],
        expected_solutions: List[Dict]
    ) -> Dict[str, Any]:
        """Compare model solutions against expected"""
        result = {
            "correct_words": [],
            "wrong_words": [],
            "missing_words": [],
            "extra_words": [],
            "case_errors": [],
            "syllable_mismatches": []
        }

        # Build expected answer map
        expected_answers = {}
        for i, sol in enumerate(expected_solutions):
            answer = sol.get("answer", "").upper()
            expected_answers[i] = {
                "answer": answer,
                "syllables": sol.get("syllables", [])
            }

        # Build model answer map
        model_answers = {}
        for i, sol in enumerate(model_solutions):
            answer = sol.get("answer", "")
            model_answers[i] = {
                "answer": answer,
                "answer_upper": answer.upper(),
                "syllables": sol.get("syllables", [])
            }

        # Compare by position
        for i in range(max(len(expected_answers), len(model_answers))):
            if i >= len(expected_answers):
                # Extra word from model
                model_ans = model_answers.get(i, {}).get("answer", "")
                result["extra_words"].append(model_ans)
                continue

            if i >= len(model_answers):
                # Missing word
                expected_ans = expected_answers.get(i, {}).get("answer", "")
                result["missing_words"].append(expected_ans)
                continue

            expected = expected_answers[i]
            model = model_answers[i]

            if model["answer_upper"] == expected["answer"]:
                result["correct_words"].append(expected["answer"])
            elif model["answer"].upper() != model["answer"] and model["answer_upper"] == expected["answer"]:
                # Case issue
                result["case_errors"].append(expected["answer"])
                result["correct_words"].append(expected["answer"])  # Still count as correct
            else:
                result["wrong_words"].append({
                    "expected": expected["answer"],
                    "got": model["answer"],
                    "position": i + 1
                })

                # Check syllable match
                if model["syllables"] != expected["syllables"]:
                    result["syllable_mismatches"].append({
                        "expected": expected["syllables"],
                        "got": model["syllables"],
                        "position": i + 1
                    })

        return result

    def analyze_syllable(
        self,
        model_output: str,
        expected_output: str
    ) -> AnalysisResult:
        """
        Analyze SYLLABLE skill model output against expected.

        Args:
            model_output: Raw model output string
            expected_output: Expected output string (JSON)

        Returns:
            AnalysisResult with detailed breakdown
        """
        result = AnalysisResult(
            category=ErrorCategory.UNKNOWN,
            is_correct=False
        )

        # Check for empty response
        if not model_output or not model_output.strip():
            result.category = ErrorCategory.EMPTY_RESPONSE
            result.analysis_notes.append("Empty model response")
            return result

        # Check for thinking tag leakage
        has_think, think_content = self.check_thinking_tags(model_output)
        if has_think:
            result.has_thinking_tags = True
            result.thinking_content = think_content[:200]  # Truncate
            result.analysis_notes.append("Thinking tags leaked in output")

        # Extract JSON from model output
        model_json = self.extract_json(model_output)
        if model_json is None:
            result.category = ErrorCategory.SCHEMA_ERROR
            result.schema_valid = False
            result.schema_errors.append("Could not parse JSON from output")

            # Still check if correct answer tokens exist in raw output
            expected_json = self.extract_json(expected_output)
            if expected_json:
                for sol in expected_json.get("solutions", []):
                    answer = sol.get("answer", "").upper()
                    if answer and answer in model_output.upper():
                        result.analysis_notes.append(f"Correct answer '{answer}' found in raw output")

            return result

        # Validate schema
        schema_valid, schema_errors = self.validate_schema(model_json)
        result.schema_valid = schema_valid
        result.schema_errors = schema_errors

        if not schema_valid:
            result.category = ErrorCategory.SCHEMA_ERROR
            return result

        # Parse expected
        expected_json = self.extract_json(expected_output)
        if expected_json is None:
            result.analysis_notes.append("Could not parse expected output")
            return result

        # Get solutions
        model_solutions = model_json.get("solutions", [])
        expected_solutions = expected_json.get("solutions", [])

        result.model_solutions = model_solutions
        result.expected_solutions = expected_solutions

        # Compare
        comparison = self.compare_solutions(model_solutions, expected_solutions)

        result.correct_words = comparison["correct_words"]
        result.wrong_words = [w.get("expected", str(w)) for w in comparison["wrong_words"]]
        result.missing_words = comparison["missing_words"]
        result.extra_words = comparison["extra_words"]

        # Calculate partial score
        total_expected = len(expected_solutions)
        total_correct = len(result.correct_words)

        if total_expected > 0:
            result.partial_score = total_correct / total_expected

        # Determine category
        if result.partial_score == 1.0:
            result.is_correct = True
            result.category = ErrorCategory.CORRECT
        elif result.has_thinking_tags:
            result.category = ErrorCategory.THINKING_LEAK
        elif result.extra_words:
            result.category = ErrorCategory.EXTRA_WORDS
        elif result.missing_words:
            result.category = ErrorCategory.MISSING_WORDS
        elif result.partial_score > 0:
            result.category = ErrorCategory.PARTIAL_CORRECT
        elif comparison["syllable_mismatches"]:
            result.category = ErrorCategory.SYLLABLE_MISMATCH
        else:
            result.category = ErrorCategory.HALLUCINATION

        # Add notes
        if comparison["case_errors"]:
            result.analysis_notes.append(f"Case errors: {comparison['case_errors']}")
        if comparison["syllable_mismatches"]:
            result.analysis_notes.append(f"Syllable mismatches: {len(comparison['syllable_mismatches'])}")

        return result

    def analyze_binary(
        self,
        model_output: str,
        expected_output: str
    ) -> AnalysisResult:
        """
        Analyze BINARY skill model output against expected.

        Args:
            model_output: Raw model output string
            expected_output: Expected output string

        Returns:
            AnalysisResult with detailed breakdown
        """
        result = AnalysisResult(
            category=ErrorCategory.UNKNOWN,
            is_correct=False
        )

        if not model_output or not model_output.strip():
            result.category = ErrorCategory.EMPTY_RESPONSE
            return result

        # Check for thinking tags
        has_think, think_content = self.check_thinking_tags(model_output)
        if has_think:
            result.has_thinking_tags = True
            result.thinking_content = think_content[:200]
            result.analysis_notes.append("Thinking tags leaked")

        # Normalize outputs
        model_norm = model_output.strip().lower()
        expected_norm = expected_output.strip().lower()

        # Get first line (main answer)
        model_first = model_norm.split('\n')[0].strip()
        expected_first = expected_norm.split('\n')[0].strip()

        # Check exact match
        if model_first == expected_first:
            result.is_correct = True
            result.category = ErrorCategory.CORRECT
            result.partial_score = 1.0
            return result

        # Check result part (after =)
        try:
            model_result = model_first.split('=')[-1].strip()
            expected_result = expected_first.split('=')[-1].strip()

            if model_result == expected_result:
                result.is_correct = True
                result.category = ErrorCategory.CORRECT
                result.partial_score = 1.0
                result.analysis_notes.append("Result matched (different format)")
                return result

            # Check if result is close (off by 1)
            try:
                model_val = int(model_result, 2) if '0' in model_result or '1' in model_result else int(model_result)
                expected_val = int(expected_result, 2) if '0' in expected_result or '1' in expected_result else int(expected_result)

                if abs(model_val - expected_val) == 1:
                    result.partial_score = 0.5
                    result.category = ErrorCategory.PARTIAL_CORRECT
                    result.analysis_notes.append(f"Off by 1: got {model_val}, expected {expected_val}")
                    return result
            except ValueError:
                pass

        except Exception:
            pass

        # No match
        if result.has_thinking_tags:
            result.category = ErrorCategory.THINKING_LEAK
        else:
            result.category = ErrorCategory.HALLUCINATION

        return result


def analyze_syllable_response(model_output: str, expected_output: str) -> Dict[str, Any]:
    """
    Convenience function to analyze SYLLABLE response.

    Returns dict with:
    - category: Error category string
    - is_correct: bool
    - partial_score: float 0-1
    - correct_count: int
    - total_expected: int
    - errors: list of error notes
    """
    analyzer = ErrorAnalyzer()
    result = analyzer.analyze_syllable(model_output, expected_output)

    return {
        "category": result.category.value,
        "is_correct": result.is_correct,
        "partial_score": result.partial_score,
        "correct_count": len(result.correct_words),
        "total_expected": len(result.expected_solutions),
        "correct_words": result.correct_words,
        "wrong_words": result.wrong_words,
        "missing_words": result.missing_words,
        "schema_valid": result.schema_valid,
        "has_thinking_tags": result.has_thinking_tags,
        "errors": result.schema_errors + result.analysis_notes
    }


def analyze_binary_response(model_output: str, expected_output: str) -> Dict[str, Any]:
    """
    Convenience function to analyze BINARY response.
    """
    analyzer = ErrorAnalyzer()
    result = analyzer.analyze_binary(model_output, expected_output)

    return {
        "category": result.category.value,
        "is_correct": result.is_correct,
        "partial_score": result.partial_score,
        "has_thinking_tags": result.has_thinking_tags,
        "errors": result.analysis_notes
    }


if __name__ == "__main__":
    # Test the analyzer
    print("Testing ErrorAnalyzer...")

    # Test SYLLABLE
    expected = '{"solutions": [{"ans_num": 1, "syllables": ["SYL", "LA", "BLE"], "answer": "SYLLABLE"}]}'

    # Test correct
    model = '{"solutions": [{"ans_num": 1, "syllables": ["SYL", "LA", "BLE"], "answer": "SYLLABLE"}]}'
    result = analyze_syllable_response(model, expected)
    print(f"Correct: {result}")

    # Test wrong
    model = '{"solutions": [{"ans_num": 1, "syllables": ["SYL", "LA", "BLE"], "answer": "WRONG"}]}'
    result = analyze_syllable_response(model, expected)
    print(f"Wrong: {result}")

    # Test schema error
    model = 'This is not JSON'
    result = analyze_syllable_response(model, expected)
    print(f"Schema error: {result}")

    # Test thinking leak
    model = '<think>Let me think...</think>{"solutions": [{"ans_num": 1, "syllables": ["SYL"], "answer": "WRONG"}]}'
    result = analyze_syllable_response(model, expected)
    print(f"Thinking leak: {result}")

    print("\nDone!")
