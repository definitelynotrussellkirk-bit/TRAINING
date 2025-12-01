#!/usr/bin/env python3
"""
Generate training-format validation files for bin and sy skills.

Uses the skill API /generate endpoints to create deterministic eval sets
that match the training data format.
"""

import json
import requests
from pathlib import Path
import sys

def generate_bin_validation(level: int, count: int = 5, seed: int = None) -> dict:
    """Generate binary validation file for a level."""
    url = "http://localhost:8090/generate"

    # Use deterministic seed based on level
    if seed is None:
        seed = 1000 + level

    payload = {
        "level": level,
        "count": count,
        "seed": seed,
        "difficulty": {"easy": 1.0}  # Keep it simple for eval
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    # Convert to validation format
    problems = []
    for sample in data["samples"]:
        problems.append({
            "prompt": sample["user_prompt"],
            "expected": sample["assistant_response"],
            "metadata": {
                "scenario": sample["scenario"],
                "tags": sample["tags"]
            }
        })

    # Get level info
    level_info = data["level_info"]

    return {
        "level": level,
        "level_name": level_info["description"],
        "bits": level_info["bits"],
        "eval_version": "2.0-training-format",
        "pass_threshold": 0.8,
        "scoring_rule": "Response must contain the expected result pattern",
        "problems": problems
    }


def generate_sy_validation(level: int, count: int = 5, seed: int = None) -> dict:
    """Generate syllo validation file for a level."""
    url = "http://localhost:8080/generate"

    # Use deterministic seed based on level
    if seed is None:
        seed = 2000 + level

    payload = {
        "level": level,
        "count": count,
        "seed": seed
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    # Convert to validation format
    problems = []
    for puzzle in data["puzzles"]:
        problems.append({
            "prompt": puzzle["user_prompt"],
            "expected": puzzle["assistant_response"],
            "metadata": {
                "num_words": puzzle.get("metadata", {}).get("num_words", 0),
                "solution_style": puzzle.get("metadata", {}).get("solution_style", "standard"),
                "puzzle_id": puzzle.get("puzzle_id", "")
            }
        })

    # Get level info
    level_tier = data.get("level_tier", f"Level {level}")

    return {
        "level": level,
        "level_name": level_tier,
        "eval_version": "2.0-training-format",
        "pass_threshold": 0.8,
        "scoring_rule": "Response must match expected format (JSON or structured output)",
        "problems": problems
    }


def main():
    base_dir = Path(__file__).parent.parent
    bin_dir = base_dir / "data" / "validation" / "bin"
    sy_dir = base_dir / "data" / "validation" / "sy"

    # Ensure directories exist
    bin_dir.mkdir(parents=True, exist_ok=True)
    sy_dir.mkdir(parents=True, exist_ok=True)

    print("Generating validation files...")
    print()

    # Skip bin files - already generated
    # Generate bin files (L2-30, L1 already exists)
    # print("Binary (bin) levels 2-30:")
    # for level in range(2, 31):
    #     try:
    #         validation_data = generate_bin_validation(level)
    #         output_path = bin_dir / f"level_{level:02d}.json"
    #
    #         with open(output_path, 'w') as f:
    #             json.dump(validation_data, f, indent=2, ensure_ascii=False)
    #
    #         print(f"  ✓ Level {level:02d} - {validation_data['level_name']} ({validation_data['bits']}-bit)")
    #     except Exception as e:
    #         print(f"  ✗ Level {level:02d} - Error: {e}")
    #         continue
    #
    # print()

    # Generate sy files (L1-50)
    print("Syllacrostic (sy) levels 1-50:")
    for level in range(1, 51):
        try:
            validation_data = generate_sy_validation(level)
            output_path = sy_dir / f"level_{level:02d}.json"

            with open(output_path, 'w') as f:
                json.dump(validation_data, f, indent=2, ensure_ascii=False)

            print(f"  ✓ Level {level:02d} - {validation_data['level_name']}")
        except Exception as e:
            print(f"  ✗ Level {level:02d} - Error: {e}")
            continue

    print()
    print("Done!")


if __name__ == "__main__":
    main()
