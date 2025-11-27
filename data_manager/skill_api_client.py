#!/usr/bin/env python3
"""
Unified Skill API Client

Connects to singleSKILL API servers to generate training data.

Skills supported:
- syllo: Syllable puzzles (5 levels, word count 4-8)
- binary: Binary emoji arithmetic (7 levels, magnitude ranges)

Usage:
    # Generate SYLLO data
    python3 data_manager/skill_api_client.py syllo --level 3 --count 100

    # Generate Binary data
    python3 data_manager/skill_api_client.py binary --level 1 --count 50

    # Generate both skills, all levels
    python3 data_manager/skill_api_client.py all --count 50
"""

import argparse
import json
import requests
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Skill API configuration
SKILL_APIS = {
    # DISABLED: Re-enable when Sy trainer is ready
    # "syllo": {
    #     "name": "SYLLO Puzzles",
    #     "base_url": "http://127.0.0.1:8080",
    #     "levels": 5,
    #     "server_script": "/path/to/skills/skill_syllo_variant/api_server.py",
    # },
    "binary": {
        "name": "Binary Arithmetic",
        "base_url": "http://127.0.0.1:8090",
        "levels": 30,  # 30 levels: 2-bit to 32-bit
        "server_script": "/path/to/skills/skill_binary/api_server.py",
    }
}

# Output directory
BASE_DIR = Path(__file__).resolve().parents[1]
QUEUE_DIR = BASE_DIR / "queue" / "normal"


class SkillAPIClient:
    """Client for skill API servers."""

    def __init__(self, skill: str, base_url: Optional[str] = None):
        if skill not in SKILL_APIS:
            raise ValueError(f"Unknown skill: {skill}. Available: {list(SKILL_APIS.keys())}")

        self.skill = skill
        self.config = SKILL_APIS[skill]
        self.base_url = base_url or self.config["base_url"]

    def health_check(self) -> bool:
        """Check if API server is running."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get server info and metadata."""
        r = requests.get(f"{self.base_url}/info", timeout=10)
        r.raise_for_status()
        return r.json()

    def get_levels(self) -> List[Dict[str, Any]]:
        """Get available difficulty levels."""
        r = requests.get(f"{self.base_url}/levels", timeout=10)
        r.raise_for_status()
        return r.json().get("levels", [])

    def generate(self, **params) -> Dict[str, Any]:
        """Generate training data."""
        r = requests.post(
            f"{self.base_url}/generate",
            json=params,
            timeout=120
        )
        r.raise_for_status()
        return r.json()


def syllo_to_training_format(puzzle: Dict[str, Any], puzzle_index: int = 1) -> Dict[str, Any]:
    """
    Convert SYLLO puzzle to training messages format.

    The SYLLO API now returns `prompt` and `solution` directly in the
    correct training format (matching the 1M+ training examples).

    Just use them directly - no local formatting needed.
    """
    # The API now returns these fields directly
    user_content = puzzle.get("prompt")
    assistant_content = puzzle.get("solution")

    # Fallback if API doesn't have new fields (shouldn't happen)
    if not user_content or not assistant_content:
        raise ValueError(
            f"SYLLO API response missing prompt/solution fields. "
            f"Make sure the SYLLO API (singleSKILL) is updated."
        )

    # Extract metadata
    puzzle_id = puzzle.get("puzzle_id", f"syllo_api_{puzzle_index:05d}")
    rules = puzzle.get("rules", {})
    word_count = rules.get("word_count", len(puzzle.get("words", [])))

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            "skill": "syllo",
            "puzzle_id": puzzle_id,
            "word_count": word_count,
            "output_variant": puzzle.get("output_variant", "unknown"),
        }
    }


GENERATOR_ID = "bin_api"
GENERATOR_VERSION = "1.0.0"


def binary_to_training_format(sample: Dict[str, Any], level_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert binary sample to training messages format.

    Bin API returns samples with user_prompt/assistant_response fields.
    Convert to standard messages[] format for training.
    """
    return {
        "messages": [
            {"role": "user", "content": sample["user_prompt"]},
            {"role": "assistant", "content": sample["assistant_response"]}
        ],
        "metadata": {
            "source": "bin_api",
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "skill": "binary",
            "sample_id": sample.get("id"),
            "scenario": sample.get("scenario"),
            "tags": [t for t in sample.get("tags", []) if t not in ["easy", "medium", "hard", "expert"]],  # Level is the only scale
            "rubric": sample.get("rubric"),
            "level": level_info.get("level") if level_info else None,
            "bits": level_info.get("bits") if level_info else None,
        }
    }


def generate_skill_data(
    skill: str,
    level: int,
    count: int,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    **extra_params
) -> Path:
    """
    Generate training data for a skill at a specific level.

    Returns path to the generated JSONL file.
    """
    output_dir = output_dir or QUEUE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    client = SkillAPIClient(skill)

    if not client.health_check():
        raise ConnectionError(
            f"{skill} API not running. Start with:\n"
            f"  cd /path/to/skills && python3 {client.config['server_script']} --port {client.base_url.split(':')[-1]}"
        )

    # Build request params
    params = {"count": count, "level": level}
    if seed is not None:
        params["seed"] = seed
    params.update(extra_params)

    print(f"Generating {count} {skill} examples at level {level}...")
    response = client.generate(**params)

    # Convert to training format
    training_examples = []

    if skill == "syllo":
        for i, puzzle in enumerate(response.get("puzzles", []), 1):
            training_examples.append(syllo_to_training_format(puzzle, puzzle_index=i))
    elif skill == "binary":
        level_info = response.get("level_info", {})
        for sample in response.get("samples", []):
            training_examples.append(binary_to_training_format(sample, level_info))

    # Write to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"train_{skill}_level{level}_{count}_{timestamp}.jsonl"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"  Wrote {len(training_examples)} examples to {output_path.name}")
    return output_path


def generate_all_levels(
    skill: str,
    count_per_level: int,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """Generate training data for all levels of a skill."""
    config = SKILL_APIS[skill]
    paths = []

    for level in range(1, config["levels"] + 1):
        path = generate_skill_data(
            skill=skill,
            level=level,
            count=count_per_level,
            seed=seed + level if seed else None,
            output_dir=output_dir
        )
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from singleSKILL APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 SYLLO puzzles at level 3
  python3 skill_api_client.py syllo --level 3 --count 100

  # Generate 50 binary conversations at level 1
  python3 skill_api_client.py binary --level 1 --count 50

  # Generate all levels for both skills
  python3 skill_api_client.py all --count 30

  # Check API status
  python3 skill_api_client.py status
"""
    )

    parser.add_argument(
        "skill",
        choices=["syllo", "binary", "all", "status"],
        help="Skill to generate (or 'all' for both, 'status' to check APIs)"
    )
    parser.add_argument("--level", type=int, help="Difficulty level (skill-specific)")
    parser.add_argument("--count", type=int, default=50, help="Number of examples per level")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: queue/normal)")

    args = parser.parse_args()

    if args.skill == "status":
        print("Skill API Status:")
        for skill, config in SKILL_APIS.items():
            client = SkillAPIClient(skill)
            status = "OK" if client.health_check() else "NOT RUNNING"
            print(f"  {skill:8} ({config['base_url']}): {status}")
            if status == "NOT RUNNING":
                print(f"           Start: cd /path/to/skills && python3 {config['server_script']} --port {config['base_url'].split(':')[-1]}")
        return

    if args.skill == "all":
        # Generate all levels for both skills
        for skill in ["syllo", "binary"]:
            print(f"\n=== Generating {SKILL_APIS[skill]['name']} ===")
            try:
                generate_all_levels(
                    skill=skill,
                    count_per_level=args.count,
                    seed=args.seed,
                    output_dir=args.output_dir
                )
            except ConnectionError as e:
                print(f"ERROR: {e}")
        return

    # Single skill
    if args.level:
        # Single level
        generate_skill_data(
            skill=args.skill,
            level=args.level,
            count=args.count,
            seed=args.seed,
            output_dir=args.output_dir
        )
    else:
        # All levels for this skill
        generate_all_levels(
            skill=args.skill,
            count_per_level=args.count,
            seed=args.seed,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
