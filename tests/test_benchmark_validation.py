#!/usr/bin/env python3
"""
Schema smoke tests for newly added transfer benchmarks.

Checks a small sample from each skill file to ensure required keys
and non-empty values are present. This is a fast offline validation.
"""

import json
from pathlib import Path

import pytest

NEW_SKILLS = [
    "math_word",
    "unit_conversion",
    "date_arithmetic",
    "sorting_setops",
    "string_transform",
    "analogies",
    "error_correction",
    "logic_tables",
    "short_classification",
    "multihop_facts",
    "format_obedience",
    "temporal_ordering",
    "counterfactual_bool",
    "aime_style",
]

REQUIRED_KEYS = {"skill", "difficulty", "user_prompt", "expected_answer", "metadata"}


def iter_skill_files(skill: str):
    skill_dir = Path("data/validation/primitives") / skill
    return sorted(skill_dir.glob("val_*_*.jsonl"))


@pytest.mark.parametrize("skill", NEW_SKILLS)
def test_benchmark_files_present(skill: str):
    files = list(iter_skill_files(skill))
    assert files, f"No benchmark files found for {skill}"


@pytest.mark.parametrize("skill", NEW_SKILLS)
def test_benchmark_sample_rows_have_required_keys(skill: str):
    files = list(iter_skill_files(skill))
    assert files, f"No benchmark files found for {skill}"

    # Sample up to 5 rows per file
    for file_path in files:
        with file_path.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = REQUIRED_KEYS - set(row.keys())
                assert not missing, f"{file_path} missing keys: {missing}"
                assert row["skill"] == skill, f"{file_path} has mismatched skill"
                assert isinstance(row["user_prompt"], str) and row["user_prompt"].strip()
                assert isinstance(row["expected_answer"], str) and row["expected_answer"].strip()
                assert isinstance(row["metadata"], dict)
                if i >= 4:
                    break
