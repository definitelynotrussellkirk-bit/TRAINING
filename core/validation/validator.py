#!/usr/bin/env python3
"""
Unified Data Validator - Single source of truth for dataset validation.

Supports multiple validation levels for different use cases:
- QUICK: Schema only (fast, for daemon)
- STANDARD: Schema + token lengths
- DEEP: Schema + lengths + content quality (for trainer)

Usage:
    from validation.validator import DataValidator, ValidationLevel

    validator = DataValidator(tokenizer, max_length=4096)
    result = validator.validate(Path("data.jsonl"), ValidationLevel.STANDARD)

    if result.should_proceed():
        # Dataset is valid
        pass
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import random

logger = logging.getLogger(__name__)

# =============================================================================
# DATA LINEAGE - Validator identification for tracking
# =============================================================================
# Bump DATA_VALIDATOR_VERSION when validation logic changes significantly
VALIDATOR_NAME = "data_validator"
VALIDATOR_VERSION = "1.0.0"

# =============================================================================
# PROTOCOL CONSTANTS (Single Source of Truth)
# =============================================================================

# Thinking emoji pool - assistant responses may start with these
THINKING_EMOJIS = [
    "🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"
]

# Stop emoji pool - assistant responses may end with these (2-4 repetitions)
STOP_EMOJIS = ["🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"]

STOP_COUNT_MIN = 2
STOP_COUNT_MAX = 4


class ValidationLevel(Enum):
    """Validation thoroughness levels."""
    QUICK = "quick"        # Schema only, fast
    STANDARD = "standard"  # Schema + lengths
    DEEP = "deep"          # Schema + lengths + content quality


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    # Lineage tracking fields
    validator_name: str = VALIDATOR_NAME
    validator_version: str = VALIDATOR_VERSION
    generator_id: Optional[str] = None
    generator_version: Optional[str] = None

    def should_proceed(self) -> bool:
        """Check if validation passed and can proceed."""
        return self.valid and len(self.errors) == 0

    def to_lineage_report(self) -> Dict[str, Any]:
        """Generate a report suitable for lineage tracking."""
        return {
            "valid": self.valid,
            "validator_name": self.validator_name,
            "validator_version": self.validator_version,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "stats": self.stats,
        }


class DataValidator:
    """
    Unified data validator for trainer and daemon.

    Validates JSONL datasets with configurable depth:
    - Schema checks (required fields, structure)
    - Token length analysis
    - Content quality checks (duplicates, leakage)

    Example:
        validator = DataValidator(tokenizer, max_length=4096)

        # Quick check (daemon)
        result = validator.validate(path, ValidationLevel.QUICK)

        # Deep check (trainer)
        result = validator.validate(path, ValidationLevel.DEEP)
    """

    REQUIRED_ROLES = {"user", "assistant"}
    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(self, tokenizer=None, max_length: int = 4096):
        """
        Initialize validator.

        Args:
            tokenizer: Optional tokenizer for length checks
            max_length: Maximum token length for training
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def validate(
        self,
        file_path: Path,
        level: ValidationLevel = ValidationLevel.STANDARD,
        sample_size: int = 100,
        generator_id: Optional[str] = None,
        generator_version: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a dataset file.

        Args:
            file_path: Path to JSONL file
            level: How thorough to validate
            sample_size: Max samples for length analysis
            generator_id: Optional generator ID for lineage tracking
            generator_version: Optional generator version for lineage tracking

        Returns:
            ValidationResult with errors, warnings, stats, and lineage info
        """
        file_path = Path(file_path)
        errors = []
        warnings = []
        stats = {"file": str(file_path), "level": level.value}

        # Try to read lineage from sidecar if not provided
        if generator_id is None:
            try:
                from core.lineage import read_file_lineage
                lineage = read_file_lineage(file_path)
                if lineage:
                    generator_id = lineage.generator_id
                    generator_version = lineage.generator_version
            except ImportError:
                pass

        # Check file exists
        if not file_path.exists():
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {file_path}"],
                generator_id=generator_id,
                generator_version=generator_version,
            )

        # Always check schema
        schema_result = self._check_schema(file_path)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)
        stats.update(schema_result.stats)

        if level in [ValidationLevel.STANDARD, ValidationLevel.DEEP]:
            # Check token lengths
            if self.tokenizer:
                length_result = self._check_lengths(file_path, sample_size)
                warnings.extend(length_result.warnings)
                stats.update(length_result.stats)
            else:
                warnings.append("No tokenizer provided, skipping length checks")

        if level == ValidationLevel.DEEP:
            # Check content quality
            content_result = self._check_content(file_path, sample_size)
            warnings.extend(content_result.warnings)
            stats.update(content_result.stats)

        valid = len(errors) == 0
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
            generator_id=generator_id,
            generator_version=generator_version,
        )

    def _check_schema(self, file_path: Path) -> ValidationResult:
        """Check JSONL structure and required fields."""
        errors = []
        warnings = []
        stats = {"total_examples": 0, "valid_examples": 0}

        try:
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    stats["total_examples"] += 1

                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
                        continue

                    # Check for messages field
                    if "messages" not in example:
                        errors.append(f"Line {line_num}: Missing 'messages' field")
                        continue

                    messages = example["messages"]
                    if not isinstance(messages, list) or len(messages) == 0:
                        errors.append(f"Line {line_num}: 'messages' must be non-empty list")
                        continue

                    # Check roles
                    roles_found = set()
                    for i, msg in enumerate(messages):
                        role = msg.get("role")
                        if role not in self.VALID_ROLES:
                            errors.append(f"Line {line_num}, msg {i}: Invalid role '{role}'")
                        else:
                            roles_found.add(role)

                        if "content" not in msg:
                            errors.append(f"Line {line_num}, msg {i}: Missing 'content'")

                    # Check required roles present
                    missing_roles = self.REQUIRED_ROLES - roles_found
                    if missing_roles:
                        warnings.append(f"Line {line_num}: Missing roles {missing_roles}")

                    stats["valid_examples"] += 1

                    # Stop early if too many errors
                    if len(errors) > 100:
                        errors.append("Too many errors, stopping validation")
                        break

        except Exception as e:
            errors.append(f"Failed to read file: {e}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)

    def _check_lengths(self, file_path: Path, sample_size: int) -> ValidationResult:
        """Analyze token lengths."""
        warnings = []
        stats = {}

        try:
            # Sample lines for analysis
            with open(file_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            if len(lines) > sample_size:
                random.seed(42)
                lines = random.sample(lines, sample_size)

            lengths = []
            for line in lines:
                try:
                    example = json.loads(line)
                    text = self._format_example(example)
                    tokens = self.tokenizer.encode(text)
                    lengths.append(len(tokens))
                except Exception:
                    continue

            if lengths:
                lengths.sort()
                stats["sampled"] = len(lengths)
                stats["min_length"] = lengths[0]
                stats["max_length"] = lengths[-1]
                stats["median_length"] = lengths[len(lengths) // 2]
                stats["p95_length"] = lengths[int(len(lengths) * 0.95)]
                stats["p99_length"] = lengths[int(len(lengths) * 0.99)]

                # Check if lengths exceed max_length
                over_limit = sum(1 for l in lengths if l > self.max_length)
                if over_limit > 0:
                    pct = over_limit / len(lengths) * 100
                    warnings.append(
                        f"{over_limit}/{len(lengths)} ({pct:.1f}%) examples exceed "
                        f"max_length={self.max_length}"
                    )

                if stats["p95_length"] > self.max_length:
                    warnings.append(
                        f"p95 length ({stats['p95_length']}) exceeds max_length ({self.max_length})"
                    )

        except Exception as e:
            warnings.append(f"Length analysis failed: {e}")

        return ValidationResult(valid=True, warnings=warnings, stats=stats)

    def _check_content(self, file_path: Path, sample_size: int) -> ValidationResult:
        """Check for duplicates, leakage, quality issues."""
        warnings = []
        stats = {}

        try:
            seen_hashes: Set[int] = set()
            duplicates = 0
            examples = []

            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    h = hash(line)
                    if h in seen_hashes:
                        duplicates += 1
                    else:
                        seen_hashes.add(h)

                    # Collect examples for leakage check
                    if len(examples) < sample_size:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

            stats["unique_examples"] = len(seen_hashes)
            stats["duplicates"] = duplicates

            if duplicates > 0:
                warnings.append(f"Found {duplicates} duplicate examples")

            # Run answer leakage detection
            leakage_result = self._check_answer_leakage(examples)
            warnings.extend(leakage_result.warnings)
            stats.update(leakage_result.stats)

            # Run protocol conformance check (50% emoji mode)
            protocol_result = self._check_protocol(examples)
            warnings.extend(protocol_result.warnings)
            stats["protocol_stats"] = protocol_result.stats

        except Exception as e:
            warnings.append(f"Content check failed: {e}")

        return ValidationResult(valid=True, warnings=warnings, stats=stats)

    def _check_answer_leakage(self, examples: List[Dict]) -> ValidationResult:
        """
        Detect if answers appear in inputs (critical data quality issue).

        Checks for:
        - Full answer appearing in prompt
        - Answer preview (first line) in prompt
        - Composition patterns like "(1 6)" appearing in both

        This catches training data issues where the answer is already
        visible in the input, leading to memorization rather than learning.
        """
        import re
        warnings = []
        stats = {"leakage_samples_checked": 0, "leakage_found": 0}

        leakage_examples = []

        for example in examples:
            messages = example.get("messages", [])
            if len(messages) < 2:
                continue

            # Find user and assistant content
            user_content = ""
            assistant_content = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    user_content = content
                elif role == "assistant":
                    assistant_content = content

            if not user_content or not assistant_content:
                continue

            stats["leakage_samples_checked"] += 1
            leakage_found = []

            # Check 1: Full answer in input
            if assistant_content.strip() in user_content:
                leakage_found.append("Full answer appears in input")

            # Check 2: Answer preview (first 50 chars or first line) in input
            answer_preview = assistant_content.strip().split('\n')[0][:50]
            if len(answer_preview) > 10 and answer_preview in user_content:
                leakage_found.append(f"Answer preview in input: '{answer_preview[:30]}...'")

            # Check 3: Composition patterns like "(1 6)" in both
            comp_pattern = r'\([0-9\s]+\)'
            user_comps = set(re.findall(comp_pattern, user_content))
            assistant_comps = set(re.findall(comp_pattern, assistant_content))
            common_comps = user_comps & assistant_comps
            if common_comps:
                leakage_found.append(f"Common patterns in both: {common_comps}")

            if leakage_found:
                stats["leakage_found"] += 1
                leakage_examples.append({
                    "issues": leakage_found,
                    "user_preview": user_content[:100],
                    "assistant_preview": assistant_content[:100]
                })

        # Generate warnings
        if stats["leakage_found"] > 0:
            pct = stats["leakage_found"] / max(stats["leakage_samples_checked"], 1) * 100
            warnings.append(
                f"ANSWER LEAKAGE DETECTED: {stats['leakage_found']}/{stats['leakage_samples_checked']} "
                f"({pct:.1f}%) examples have answer visible in input"
            )
            # Add first few examples as warnings
            for i, ex in enumerate(leakage_examples[:3]):
                warnings.append(f"  Leakage example {i+1}: {ex['issues']}")

        return ValidationResult(valid=True, warnings=warnings, stats=stats)

    def _check_protocol(self, examples: List[Dict]) -> ValidationResult:
        """
        Check emoji protocol conformance (50% mode).

        The protocol allows two valid modes:
        1. EMOJI MODE: Thinking emoji at start AND stop emoji at end
        2. DIRECT MODE: NO thinking emoji at start (stop emoji optional)

        INVALID: Thinking emoji at start WITHOUT stop emoji at end (malformed)

        This supports 50/50 training where half examples use emoji thinking
        and half give direct answers without the thinking protocol.

        Args:
            examples: List of parsed example dicts

        Returns:
            ValidationResult with protocol stats
        """
        warnings = []
        stats = {
            "protocol_checked": 0,
            "emoji_mode_valid": 0,
            "direct_mode_valid": 0,
            "malformed": 0,
            "malformed_details": [],
            "emoji_mode_pct": 0.0,
            "direct_mode_pct": 0.0,
        }

        for idx, example in enumerate(examples):
            messages = example.get("messages", [])

            # Find assistant content
            assistant_content = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    break

            if not assistant_content:
                continue

            stats["protocol_checked"] += 1

            # Check for thinking emoji at start
            has_thinking = any(
                assistant_content.strip().startswith(emoji)
                for emoji in THINKING_EMOJIS
            )

            # Check for stop emoji sequence at end
            has_stop = False
            stripped_end = assistant_content.rstrip()
            for emoji in STOP_EMOJIS:
                for count in range(STOP_COUNT_MIN, STOP_COUNT_MAX + 1):
                    if stripped_end.endswith(emoji * count):
                        has_stop = True
                        break
                if has_stop:
                    break

            # Classify
            if has_thinking and has_stop:
                # Valid emoji mode
                stats["emoji_mode_valid"] += 1
            elif not has_thinking:
                # Valid direct mode (no thinking = direct answer)
                stats["direct_mode_valid"] += 1
            else:
                # Malformed: has thinking but no stop
                stats["malformed"] += 1
                if len(stats["malformed_details"]) < 5:
                    stats["malformed_details"].append({
                        "idx": idx,
                        "preview": assistant_content[:100],
                        "issue": "thinking_without_stop"
                    })

        # Calculate percentages
        total = stats["protocol_checked"]
        if total > 0:
            stats["emoji_mode_pct"] = round(stats["emoji_mode_valid"] / total * 100, 1)
            stats["direct_mode_pct"] = round(stats["direct_mode_valid"] / total * 100, 1)

        # Generate warnings
        if stats["malformed"] > 0:
            pct = stats["malformed"] / max(total, 1) * 100
            warnings.append(
                f"PROTOCOL MALFORMED: {stats['malformed']}/{total} ({pct:.1f}%) "
                f"examples have thinking emoji without stop emoji"
            )

        # Warn if mix is heavily skewed (should be ~50/50)
        emoji_pct = stats["emoji_mode_pct"]
        if total >= 20:  # Only warn with enough samples
            if emoji_pct < 30:
                warnings.append(
                    f"Protocol mix skewed: only {emoji_pct}% emoji mode "
                    f"(expected ~50%). Consider balancing."
                )
            elif emoji_pct > 70:
                warnings.append(
                    f"Protocol mix skewed: {emoji_pct}% emoji mode "
                    f"(expected ~50%). Consider adding direct examples."
                )

        return ValidationResult(valid=True, warnings=warnings, stats=stats)

    def _format_example(self, example: Dict) -> str:
        """Format example for tokenization."""
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    example.get("messages", []),
                    tokenize=False
                )
            except Exception:
                pass

        # Fallback: concatenate contents
        parts = []
        for msg in example.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)


if __name__ == "__main__":
    # Quick test
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Create test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(10):
            example = {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"}
                ]
            }
            f.write(json.dumps(example) + '\n')
        # Add duplicate
        f.write(json.dumps(example) + '\n')
        # Add invalid line
        f.write("not valid json\n")
        test_file = f.name

    print(f"Testing with {test_file}")

    # Test QUICK validation
    validator = DataValidator(max_length=4096)
    result = validator.validate(Path(test_file), ValidationLevel.QUICK)
    print(f"\nQUICK validation:")
    print(f"  Valid: {result.valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Stats: {result.stats}")

    # Test DEEP validation (without tokenizer)
    result = validator.validate(Path(test_file), ValidationLevel.DEEP)
    print(f"\nDEEP validation:")
    print(f"  Valid: {result.valid}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Stats: {result.stats}")

    # Cleanup
    import os
    os.unlink(test_file)

    print("\nDataValidator ready for use!")
