#!/usr/bin/env python3
"""
Data Validator - Check JSONL training data before dropping in inbox.

Validates:
1. JSON syntax (each line must be valid JSON)
2. Message format (must have "messages" array)
3. Role structure (user/assistant pairs)
4. Content presence (non-empty strings)

Usage:
    python3 -m training validate my_data.jsonl
    python3 -m training validate inbox/*.jsonl
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validating a single file."""
    path: Path
    valid: bool
    total_lines: int
    valid_lines: int
    errors: List[Tuple[int, str]]  # (line_number, error_message)
    warnings: List[Tuple[int, str]]

    def summary(self) -> str:
        """Return a one-line summary."""
        if self.valid:
            return f"OK: {self.valid_lines} examples"
        else:
            return f"FAIL: {len(self.errors)} errors, {self.valid_lines}/{self.total_lines} valid"


def validate_message(msg: Dict[str, Any], idx: int) -> List[str]:
    """Validate a single message in the messages array."""
    errors = []

    if not isinstance(msg, dict):
        errors.append(f"message[{idx}] is not an object")
        return errors

    if "role" not in msg:
        errors.append(f"message[{idx}] missing 'role'")
    elif msg["role"] not in ("user", "assistant", "system"):
        errors.append(f"message[{idx}] has invalid role '{msg['role']}' (expected: user/assistant/system)")

    if "content" not in msg:
        errors.append(f"message[{idx}] missing 'content'")
    elif not isinstance(msg["content"], str):
        errors.append(f"message[{idx}] content is not a string")
    elif len(msg["content"].strip()) == 0:
        errors.append(f"message[{idx}] has empty content")

    return errors


def validate_line(line: str, line_num: int) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a single line of JSONL.

    Returns:
        (valid, errors, warnings)
    """
    errors = []
    warnings = []

    # Skip empty lines
    if not line.strip():
        return True, [], []

    # Parse JSON
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"], []

    # Check for messages array
    if "messages" not in data:
        return False, ["Missing 'messages' key"], []

    messages = data["messages"]
    if not isinstance(messages, list):
        return False, ["'messages' is not an array"], []

    if len(messages) == 0:
        return False, ["'messages' array is empty"], []

    # Validate each message
    for i, msg in enumerate(messages):
        msg_errors = validate_message(msg, i)
        errors.extend(msg_errors)

    # Check for user/assistant pattern
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "user" not in roles:
        warnings.append("No 'user' message found")
    if "assistant" not in roles:
        warnings.append("No 'assistant' message found")

    # Check content lengths
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            content_len = len(msg["content"])
            if content_len > 32000:
                warnings.append(f"message[{i}] content is very long ({content_len} chars)")

    return len(errors) == 0, errors, warnings


def validate_file(path: Path, verbose: bool = False, max_errors: int = 10) -> ValidationResult:
    """Validate an entire JSONL file."""
    errors = []
    warnings = []
    valid_lines = 0
    total_lines = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1

                valid, line_errors, line_warnings = validate_line(line, line_num)

                if valid:
                    valid_lines += 1
                else:
                    for err in line_errors:
                        if len(errors) < max_errors:
                            errors.append((line_num, err))

                for warn in line_warnings:
                    if len(warnings) < max_errors:
                        warnings.append((line_num, warn))

                if verbose and line_num % 1000 == 0:
                    print(f"  Processed {line_num} lines...", end="\r")

    except FileNotFoundError:
        return ValidationResult(
            path=path,
            valid=False,
            total_lines=0,
            valid_lines=0,
            errors=[(0, f"File not found: {path}")],
            warnings=[]
        )
    except UnicodeDecodeError as e:
        return ValidationResult(
            path=path,
            valid=False,
            total_lines=0,
            valid_lines=0,
            errors=[(0, f"Encoding error: {e}")],
            warnings=[]
        )

    return ValidationResult(
        path=path,
        valid=len(errors) == 0,
        total_lines=total_lines,
        valid_lines=valid_lines,
        errors=errors,
        warnings=warnings
    )


def run_validate(paths: List[str], verbose: bool = False, quiet: bool = False) -> int:
    """Run validation on multiple files."""
    from glob import glob

    # Expand globs
    all_paths = []
    for p in paths:
        expanded = glob(p)
        if expanded:
            all_paths.extend(expanded)
        else:
            all_paths.append(p)

    if not all_paths:
        print("No files to validate.")
        return 1

    results = []
    for path_str in all_paths:
        path = Path(path_str)

        if not quiet:
            print(f"\nValidating: {path}")

        result = validate_file(path, verbose=verbose)
        results.append(result)

        if not quiet:
            # Show summary
            icon = "\u2705" if result.valid else "\u274c"
            print(f"  {icon} {result.summary()}")

            # Show errors
            if result.errors and verbose:
                print("  Errors:")
                for line_num, err in result.errors[:10]:
                    print(f"    Line {line_num}: {err}")
                if len(result.errors) > 10:
                    print(f"    ... and {len(result.errors) - 10} more errors")

            # Show warnings
            if result.warnings and verbose:
                print("  Warnings:")
                for line_num, warn in result.warnings[:5]:
                    print(f"    Line {line_num}: {warn}")

    # Summary
    if len(results) > 1 and not quiet:
        print("\n" + "=" * 40)
        valid_count = sum(1 for r in results if r.valid)
        total_examples = sum(r.valid_lines for r in results)
        print(f"Summary: {valid_count}/{len(results)} files valid, {total_examples} total examples")

    # Return 0 if all valid, 1 otherwise
    return 0 if all(r.valid for r in results) else 1


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate JSONL training data files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSONL files to validate (supports globs)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed error messages"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only output if there are errors"
    )

    args = parser.parse_args()

    return run_validate(args.files, verbose=args.verbose, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
