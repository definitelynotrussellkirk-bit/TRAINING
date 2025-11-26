#!/usr/bin/env python3
"""
Tests for core/validation/validator.py - DataValidator

Tests the new validators added 2025-11-26:
- Repetition detection
- Degenerate content detection
- JSON validity
- Encoding/unicode issues
- Conversation flow
- Content length ratios
"""

import json
import pytest
import tempfile
from pathlib import Path

from core.validation.validator import DataValidator, ValidationLevel


@pytest.fixture
def validator():
    """Create a DataValidator instance."""
    return DataValidator(max_length=4096)


@pytest.fixture
def temp_jsonl():
    """Create a temporary JSONL file with test data."""
    def _create(examples):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
            return Path(f.name)
    return _create


class TestRepetitionDetection:
    """Tests for _check_repetition validator."""

    def test_detects_word_repetition(self, validator, temp_jsonl):
        """Should detect same word repeated 5+ times."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Say yes'},
                {'role': 'assistant', 'content': 'yes yes yes yes yes yes'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert 'repetition_stats' in result.stats
        assert result.stats['repetition_stats']['repetition_found'] >= 1
        file_path.unlink()

    def test_no_false_positive_normal_text(self, validator, temp_jsonl):
        """Should not flag normal text as repetitive."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Write a sentence'},
                {'role': 'assistant', 'content': 'The quick brown fox jumps over the lazy dog.'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['repetition_stats']['repetition_found'] == 0
        file_path.unlink()


class TestDegenerateContent:
    """Tests for _check_degenerate_content validator."""

    def test_detects_empty_response(self, validator, temp_jsonl):
        """Should detect empty assistant responses."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': ''}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['degenerate_stats']['empty_responses'] == 1
        file_path.unlink()

    def test_detects_whitespace_only(self, validator, temp_jsonl):
        """Should detect whitespace-only responses."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': '   \n\t  '}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['degenerate_stats']['whitespace_only'] == 1
        file_path.unlink()

    def test_detects_too_short(self, validator, temp_jsonl):
        """Should detect responses shorter than 5 chars."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'What is life?'},
                {'role': 'assistant', 'content': 'ok'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['degenerate_stats']['too_short'] == 1
        file_path.unlink()


class TestJSONValidity:
    """Tests for _check_json_validity validator."""

    def test_detects_invalid_json(self, validator, temp_jsonl):
        """Should detect invalid JSON in responses."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Give me JSON'},
                {'role': 'assistant', 'content': '{"name": "test",}'}  # trailing comma
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['json_stats']['json_invalid'] == 1
        file_path.unlink()

    def test_accepts_valid_json(self, validator, temp_jsonl):
        """Should accept valid JSON responses."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Give me JSON'},
                {'role': 'assistant', 'content': '{"solutions": [{"answer": "TEST"}]}'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['json_stats']['json_valid'] == 1
        assert result.stats['json_stats']['json_invalid'] == 0
        file_path.unlink()

    def test_skips_non_json_responses(self, validator, temp_jsonl):
        """Should not check responses that don't look like JSON."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['json_stats']['json_checked'] == 0
        file_path.unlink()


class TestEncodingIssues:
    """Tests for _check_encoding validator."""

    def test_detects_mojibake(self, validator, temp_jsonl):
        """Should detect mojibake (encoding corruption)."""
        # Ã© is what é looks like when UTF-8 is misinterpreted as Latin-1
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'The caf\u00c3\u00a9 is nice'}  # Ã© mojibake
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['encoding_stats']['mojibake_found'] >= 1
        file_path.unlink()

    def test_accepts_clean_unicode(self, validator, temp_jsonl):
        """Should accept properly encoded unicode."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'The café is nice with émojis 🎉'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['encoding_stats']['mojibake_found'] == 0
        file_path.unlink()


class TestConversationFlow:
    """Tests for _check_conversation_flow validator."""

    def test_detects_orphaned_assistant(self, validator, temp_jsonl):
        """Should detect assistant response without user message."""
        examples = [
            {'messages': [
                {'role': 'assistant', 'content': 'I am responding first'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['flow_stats']['orphaned_assistant'] == 1
        file_path.unlink()

    def test_detects_doubled_roles(self, validator, temp_jsonl):
        """Should detect consecutive messages with same role."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'First'},
                {'role': 'user', 'content': 'Second'},
                {'role': 'assistant', 'content': 'Response'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['flow_stats']['doubled_roles'] == 1
        file_path.unlink()

    def test_detects_misplaced_system(self, validator, temp_jsonl):
        """Should detect system message not at start."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'system', 'content': 'You are helpful'},
                {'role': 'assistant', 'content': 'Hi'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['flow_stats']['system_misplaced'] == 1
        file_path.unlink()

    def test_accepts_valid_flow(self, validator, temp_jsonl):
        """Should accept proper conversation flow."""
        examples = [
            {'messages': [
                {'role': 'system', 'content': 'You are helpful'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['flow_stats']['flow_valid'] == 1
        file_path.unlink()


class TestContentRatio:
    """Tests for _check_content_ratio validator."""

    def test_detects_prompt_heavy(self, validator, temp_jsonl):
        """Should detect very long prompts with short responses."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'A' * 5000},
                {'role': 'assistant', 'content': 'Yes, correct.'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['ratio_stats']['prompt_heavy'] == 1
        file_path.unlink()

    def test_accepts_balanced_ratio(self, validator, temp_jsonl):
        """Should accept reasonably balanced prompt/response."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'What is the capital of France?'},
                {'role': 'assistant', 'content': 'The capital of France is Paris.'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        assert result.stats['ratio_stats']['balanced'] == 1
        file_path.unlink()


class TestValidationLevels:
    """Tests for validation levels."""

    def test_quick_validation_skips_content_checks(self, validator, temp_jsonl):
        """QUICK validation should only check schema, not content."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': ''}  # Empty - would fail content check
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.QUICK)

        # Should pass schema validation
        assert result.valid
        # Should not have content stats
        assert 'degenerate_stats' not in result.stats
        file_path.unlink()

    def test_deep_validation_runs_all_checks(self, validator, temp_jsonl):
        """DEEP validation should run all content checks."""
        examples = [
            {'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'}
            ]}
        ]
        file_path = temp_jsonl(examples)
        result = validator.validate(file_path, ValidationLevel.DEEP)

        # Should have all stats
        assert 'repetition_stats' in result.stats
        assert 'degenerate_stats' in result.stats
        assert 'json_stats' in result.stats
        assert 'encoding_stats' in result.stats
        assert 'flow_stats' in result.stats
        assert 'ratio_stats' in result.stats
        file_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
