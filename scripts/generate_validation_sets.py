#!/usr/bin/env python3
"""
Generate static validation sets for all skills.

Creates 5 fixed problems per level for each skill.
These are deterministic (seeded) so they're the same every time.
"""

import json
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "validation"


def generate_binary_validation():
    """Generate validation set for Binary skill (30 levels, 5 problems each)."""
    print("Generating Binary validation set...")

    api_url = "http://localhost:8090"
    output_file = OUTPUT_DIR / "bin_validation.json"

    validation = {}

    for level in range(1, 31):  # Levels 1-30
        print(f"  Level {level}...", end=" ")
        try:
            resp = requests.post(
                f"{api_url}/generate",
                json={"level": level, "count": 5, "seed": 42 + level},  # Deterministic seed
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            problems = []
            for sample in data.get("samples", []):
                # Chat format: messages array
                problems.append({
                    "messages": [
                        {"role": "user", "content": sample.get("user_prompt", "")},
                        {"role": "assistant", "content": sample.get("assistant_response", "")},
                    ],
                    "metadata": {
                        "skill": "bin",
                        "level": level,
                    }
                })

            validation[str(level)] = problems
            print(f"{len(problems)} problems")

        except Exception as e:
            print(f"ERROR: {e}")
            validation[str(level)] = []

    # Save
    with open(output_file, "w") as f:
        json.dump(validation, f, indent=2)

    total = sum(len(p) for p in validation.values())
    print(f"\nSaved {total} problems to {output_file}")


def generate_syllo_validation():
    """Generate validation set for Syllo skill (50 levels, 5 problems each)."""
    print("Generating Syllo validation set...")

    api_url = "http://localhost:8080"
    output_file = OUTPUT_DIR / "sy_validation.json"

    validation = {}

    for level in range(1, 51):  # Levels 1-50
        print(f"  Level {level}...", end=" ")
        try:
            resp = requests.post(
                f"{api_url}/generate",
                json={"level": level, "count": 5, "seed": 42 + level},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            problems = []
            for puzzle in data.get("puzzles", []):
                # API returns user_prompt and assistant_response
                user_prompt = puzzle.get("user_prompt", "")
                assistant_response = puzzle.get("assistant_response", "")
                # Chat format: messages array
                problems.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response},
                    ],
                    "metadata": {
                        "skill": "sy",
                        "level": level,
                        "puzzle_id": puzzle.get("puzzle_id", ""),
                    }
                })

            validation[str(level)] = problems
            print(f"{len(problems)} problems")

        except Exception as e:
            print(f"ERROR: {e}")
            validation[str(level)] = []

    # Save
    with open(output_file, "w") as f:
        json.dump(validation, f, indent=2)

    total = sum(len(p) for p in validation.values())
    print(f"\nSaved {total} problems to {output_file}")


if __name__ == "__main__":
    import sys

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        skill = sys.argv[1].lower()
        if skill in ("bin", "binary"):
            generate_binary_validation()
        elif skill in ("sy", "syllo"):
            generate_syllo_validation()
        else:
            print(f"Unknown skill: {skill}")
    else:
        # Generate both
        generate_binary_validation()
        print()
        generate_syllo_validation()
