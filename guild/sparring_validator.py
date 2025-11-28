#!/usr/bin/env python3
"""
Sparring Data Validator - Quality checks for sparring-generated training data

Validates the 3 types of sparring examples:
1. sparring_identify_incorrect - Must answer "It is incorrect."
2. sparring_correction - Must have valid golden answer
3. sparring_confirm_correct - Must answer "It is correct."

Usage:
    python3 guild/sparring_validator.py path/to/sparring_data.jsonl
    python3 guild/sparring_validator.py --check-all  # Validate all sparring files
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validator identification for data lineage
VALIDATOR_NAME = "sparring_validator"
VALIDATOR_VERSION = "1.0.0"


@dataclass
class ValidationResult:
    """Result of validating a single example"""
    valid: bool
    example_type: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FileValidationReport:
    """Report for validating an entire file"""
    filepath: str
    total_examples: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    by_type: Dict[str, Dict] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)
    passed: bool = False


class SparringValidator:
    """
    Validates sparring-generated training data.

    Checks:
    - Correct message structure (user/assistant pairs)
    - Type-specific response validation
    - No truncation or corruption
    - Proper metadata
    """

    # Expected responses (normalized)
    INCORRECT_RESPONSES = {
        "it is incorrect.",
        "it is incorrect",
        "incorrect.",
        "incorrect",
    }

    CORRECT_RESPONSES = {
        "it is correct.",
        "it is correct",
        "correct.",
        "correct",
    }

    # Minimum lengths
    MIN_PROBLEM_LENGTH = 20  # Characters
    MIN_ANSWER_LENGTH = 5    # Characters for golden answers

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, warnings become errors
        """
        self.strict = strict

    def validate_example(self, example: Dict, line_num: int = 0) -> ValidationResult:
        """
        Validate a single sparring example.

        Returns:
            ValidationResult with valid flag and any errors/warnings
        """
        errors = []
        warnings = []

        # Check required fields
        if "messages" not in example:
            errors.append("Missing 'messages' field")
            return ValidationResult(False, "unknown", errors)

        messages = example["messages"]
        example_type = example.get("type", "unknown")

        # Check message structure
        if len(messages) < 2:
            errors.append(f"Need at least 2 messages, got {len(messages)}")

        # Validate message roles
        for i, msg in enumerate(messages):
            if "role" not in msg:
                errors.append(f"Message {i} missing 'role'")
            if "content" not in msg:
                errors.append(f"Message {i} missing 'content'")

        if errors:
            return ValidationResult(False, example_type, errors)

        # Get user prompt and assistant response
        user_msg = messages[0] if messages[0]["role"] == "user" else None
        asst_msg = messages[1] if len(messages) > 1 and messages[1]["role"] == "assistant" else None

        if not user_msg:
            errors.append("First message must be from 'user'")
        if not asst_msg:
            errors.append("Second message must be from 'assistant'")

        if errors:
            return ValidationResult(False, example_type, errors)

        user_content = user_msg["content"]
        asst_content = asst_msg["content"]

        # Check user prompt has content
        if len(user_content) < self.MIN_PROBLEM_LENGTH:
            errors.append(f"User prompt too short ({len(user_content)} chars)")

        # Type-specific validation
        if example_type == "sparring_identify_incorrect":
            errors.extend(self._validate_identify_incorrect(user_content, asst_content))

        elif example_type == "sparring_correction":
            errors.extend(self._validate_correction(user_content, asst_content))

        elif example_type == "sparring_confirm_correct":
            errors.extend(self._validate_confirm_correct(user_content, asst_content))

        else:
            warnings.append(f"Unknown example type: {example_type}")

        # Check metadata
        if "skill" not in example:
            warnings.append("Missing 'skill' metadata")
        if "level" not in example:
            warnings.append("Missing 'level' metadata")
        if "generator_id" not in example:
            warnings.append("Missing 'generator_id' metadata")

        # In strict mode, warnings become errors
        if self.strict:
            errors.extend(warnings)
            warnings = []

        valid = len(errors) == 0
        return ValidationResult(valid, example_type, errors, warnings)

    def _validate_identify_incorrect(self, user_content: str, asst_content: str) -> List[str]:
        """Validate 'identify incorrect' example"""
        errors = []

        # User prompt should ask "Is this correct?"
        if "correct" not in user_content.lower():
            errors.append("User prompt should ask about correctness")

        # User prompt should contain "Proposed answer:"
        if "proposed answer" not in user_content.lower():
            errors.append("User prompt should contain proposed answer section")

        # Assistant should say "It is incorrect"
        normalized = asst_content.strip().lower()
        if normalized not in self.INCORRECT_RESPONSES:
            errors.append(f"Assistant should say 'It is incorrect.', got: '{asst_content[:50]}'")

        return errors

    def _validate_correction(self, user_content: str, asst_content: str) -> List[str]:
        """Validate 'correction' example"""
        errors = []

        # User prompt should mention incorrect
        if "incorrect" not in user_content.lower():
            errors.append("User prompt should mention the answer is incorrect")

        # User prompt should ask for correct solution
        if "correct solution" not in user_content.lower() and "find the correct" not in user_content.lower():
            errors.append("User prompt should ask for correct solution")

        # Assistant response should be substantial (the actual answer)
        if len(asst_content) < self.MIN_ANSWER_LENGTH:
            errors.append(f"Assistant answer too short ({len(asst_content)} chars)")

        # Should NOT be just "It is correct/incorrect"
        normalized = asst_content.strip().lower()
        if normalized in self.INCORRECT_RESPONSES or normalized in self.CORRECT_RESPONSES:
            errors.append("Assistant should provide actual solution, not just correctness")

        return errors

    def _validate_confirm_correct(self, user_content: str, asst_content: str) -> List[str]:
        """Validate 'confirm correct' example"""
        errors = []

        # User prompt should ask "Is this correct?"
        if "correct" not in user_content.lower():
            errors.append("User prompt should ask about correctness")

        # User prompt should contain "Proposed answer:"
        if "proposed answer" not in user_content.lower():
            errors.append("User prompt should contain proposed answer section")

        # Assistant should say "It is correct"
        normalized = asst_content.strip().lower()
        if normalized not in self.CORRECT_RESPONSES:
            errors.append(f"Assistant should say 'It is correct.', got: '{asst_content[:50]}'")

        return errors

    def validate_file(self, filepath: Path) -> FileValidationReport:
        """
        Validate an entire JSONL file of sparring examples.

        Returns:
            FileValidationReport with aggregate stats
        """
        report = FileValidationReport(filepath=str(filepath))

        report.by_type = {
            "sparring_identify_incorrect": {"total": 0, "valid": 0, "invalid": 0},
            "sparring_correction": {"total": 0, "valid": 0, "invalid": 0},
            "sparring_confirm_correct": {"total": 0, "valid": 0, "invalid": 0},
            "unknown": {"total": 0, "valid": 0, "invalid": 0},
        }

        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    report.total_examples += 1

                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError as e:
                        report.invalid_examples += 1
                        report.errors.append({
                            "line": line_num,
                            "error": f"Invalid JSON: {e}"
                        })
                        continue

                    result = self.validate_example(example, line_num)

                    # Track by type
                    type_key = result.example_type if result.example_type in report.by_type else "unknown"
                    report.by_type[type_key]["total"] += 1

                    if result.valid:
                        report.valid_examples += 1
                        report.by_type[type_key]["valid"] += 1
                    else:
                        report.invalid_examples += 1
                        report.by_type[type_key]["invalid"] += 1
                        report.errors.append({
                            "line": line_num,
                            "type": result.example_type,
                            "errors": result.errors,
                        })

        except Exception as e:
            report.errors.append({"error": f"File read error: {e}"})

        # Pass if >95% valid
        if report.total_examples > 0:
            pass_rate = report.valid_examples / report.total_examples
            report.passed = pass_rate >= 0.95

        return report

    def print_report(self, report: FileValidationReport):
        """Print formatted validation report"""
        print("\n" + "="*60)
        print(f"SPARRING VALIDATION REPORT")
        print("="*60)
        print(f"File: {report.filepath}")
        print(f"Total examples: {report.total_examples}")
        print(f"Valid:   {report.valid_examples} ({report.valid_examples/report.total_examples*100:.1f}%)" if report.total_examples else "N/A")
        print(f"Invalid: {report.invalid_examples}")

        print("\nBy Type:")
        for type_name, stats in report.by_type.items():
            if stats["total"] > 0:
                pct = stats["valid"] / stats["total"] * 100
                print(f"  {type_name}: {stats['valid']}/{stats['total']} ({pct:.1f}%)")

        if report.errors and len(report.errors) <= 10:
            print("\nErrors:")
            for err in report.errors[:10]:
                print(f"  Line {err.get('line', '?')}: {err.get('errors', err.get('error', 'Unknown'))}")

        if len(report.errors) > 10:
            print(f"\n  ... and {len(report.errors) - 10} more errors")

        status = "✅ PASSED" if report.passed else "❌ FAILED"
        print(f"\nStatus: {status}")
        print("="*60 + "\n")


def validate_for_training(filepath: Path, strict: bool = False) -> Tuple[bool, Dict]:
    """
    Validation function for integration with training queue.

    Returns:
        (passed, report_dict)
    """
    validator = SparringValidator(strict=strict)
    report = validator.validate_file(filepath)

    report_dict = {
        "validator": VALIDATOR_NAME,
        "validator_version": VALIDATOR_VERSION,
        "filepath": str(filepath),
        "total": report.total_examples,
        "valid": report.valid_examples,
        "invalid": report.invalid_examples,
        "pass_rate": report.valid_examples / report.total_examples if report.total_examples else 0,
        "passed": report.passed,
        "by_type": report.by_type,
        "error_count": len(report.errors),
    }

    return report.passed, report_dict


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate sparring training data")
    parser.add_argument("file", nargs="?", help="JSONL file to validate")
    parser.add_argument("--check-all", action="store_true", help="Validate all sparring files")
    parser.add_argument("--strict", action="store_true", help="Strict mode (warnings = errors)")
    parser.add_argument("--base-dir", default=None, help="Base dir (default: from core.paths)")

    args = parser.parse_args()
    from core.paths import get_base_dir
    base_dir = Path(args.base_dir) if args.base_dir else get_base_dir()

    validator = SparringValidator(strict=args.strict)

    if args.check_all:
        # Find all sparring files
        sparring_dir = base_dir / "guild" / "sparring_data"
        inbox = base_dir / "inbox"
        queue_dirs = [
            base_dir / "queue" / "high",
            base_dir / "queue" / "normal",
            base_dir / "queue" / "low",
        ]

        files = []
        for d in [sparring_dir, inbox] + queue_dirs:
            if d.exists():
                files.extend(d.glob("sparring_*.jsonl"))

        if not files:
            print("No sparring files found")
            return

        print(f"Found {len(files)} sparring files")

        all_passed = True
        for f in sorted(files):
            report = validator.validate_file(f)
            status = "✓" if report.passed else "✗"
            pct = report.valid_examples / report.total_examples * 100 if report.total_examples else 0
            print(f"  {status} {f.name}: {report.valid_examples}/{report.total_examples} ({pct:.0f}%)")
            if not report.passed:
                all_passed = False

        print(f"\nOverall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")

    elif args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)

        report = validator.validate_file(filepath)
        validator.print_report(report)

        sys.exit(0 if report.passed else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
