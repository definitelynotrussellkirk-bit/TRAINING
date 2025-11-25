#!/usr/bin/env python3
"""
Probe Set Manager - Fixed set of prompts for consistent tracking.

Maintains a canonical set of ~500 prompts used across all checkpoints
to track how the model's representations evolve.

Usage:
    # Initialize probe set
    python3 probe_set.py --init

    # Show probe set stats
    python3 probe_set.py --stats

    # Export for external analysis
    python3 probe_set.py --export probes.json

Output:
    config/probe_set.json - Fixed probe prompts + metadata
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Probe:
    """A single probe prompt for tracking."""
    id: str
    prompt: str
    category: str
    difficulty: str
    expected_answer: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# SYLLO probe templates by difficulty
SYLLO_TEMPLATES = {
    "easy": [
        ("All {A} are {B}. {x} is a {A}. Is {x} a {B}?", "Yes"),
        ("No {A} are {B}. {x} is a {A}. Is {x} a {B}?", "No"),
        ("All {A} are {B}. All {B} are {C}. Is every {A} a {C}?", "Yes"),
        ("Some {A} are {B}. {x} is a {A}. Is {x} a {B}?", "Cannot determine"),
    ],
    "medium": [
        ("All {A} are {B}. No {B} are {C}. Are any {A} also {C}?", "No"),
        ("Some {A} are {B}. Some {B} are {C}. Are some {A} also {C}?", "Cannot determine"),
        ("No {A} are {B}. All {C} are {A}. Are any {C} also {B}?", "No"),
        ("All {A} are {B}. Some {C} are not {B}. Are all {C} also {A}?", "No"),
    ],
    "hard": [
        ("It is not true that all {A} are {B}. Is it true that some {A} are not {B}?", "Yes"),
        ("No {A} are {B}. No {B} are {C}. Are any {A} also {C}?", "Cannot determine"),
        ("If all {A} are {B}, and all {B} are {C}, and {x} is not a {C}, is {x} a {A}?", "No"),
        ("All {A} that are {B} are also {C}. {x} is a {A} but not a {C}. Is {x} a {B}?", "No"),
    ],
}

# Word banks for generating SYLLO problems
CATEGORIES = [
    ("cats", "animals", "pets", "Felix"),
    ("dogs", "mammals", "pets", "Rex"),
    ("birds", "animals", "creatures", "Tweety"),
    ("fish", "animals", "swimmers", "Nemo"),
    ("trees", "plants", "living things", "Oak"),
    ("roses", "flowers", "plants", "Rosa"),
    ("cars", "vehicles", "machines", "Tesla"),
    ("books", "objects", "items", "Novel"),
    ("students", "people", "learners", "Alice"),
    ("teachers", "professionals", "educators", "Bob"),
]

# Edge case categories
EDGE_CASES = {
    "negation": [
        ("No birds are fish. All sparrows are birds. Are any sparrows fish?", "No"),
        ("Not all cats are black. Are some cats not black?", "Yes"),
        ("No mammals can fly. Bats are mammals. Can bats fly?", "No"),  # Trick - real world wrong
    ],
    "double_negation": [
        ("It is not the case that no dogs bark. Do some dogs bark?", "Yes"),
        ("It is false that all swans are not white. Are some swans white?", "Yes"),
    ],
    "quantifier_scope": [
        ("Everyone loves someone. Does someone love everyone?", "Cannot determine"),
        ("All students passed some exam. Did some exam fail all students?", "Cannot determine"),
    ],
    "existential": [
        ("Some birds can fly. Are all birds able to fly?", "Cannot determine"),
        ("Some cats are black. Is every cat black?", "No"),
    ],
    "vacuous": [
        ("All unicorns are purple. Are there purple unicorns?", "Cannot determine"),
        ("Every square circle is red. Are there red square circles?", "Cannot determine"),
    ],
}


def generate_syllo_probe(template: str, answer: str, difficulty: str, idx: int) -> Probe:
    """Generate a SYLLO probe from template."""
    # Pick random word bank
    words = random.choice(CATEGORIES)
    A, B, C, x = words

    prompt = template.format(A=A, B=B, C=C, x=x)

    return Probe(
        id=f"syllo_{difficulty}_{idx}",
        prompt=prompt,
        category="syllo",
        difficulty=difficulty,
        expected_answer=answer,
        metadata={"template": template, "words": words}
    )


def generate_edge_case_probe(category: str, prompt: str, answer: str, idx: int) -> Probe:
    """Generate an edge case probe."""
    return Probe(
        id=f"edge_{category}_{idx}",
        prompt=prompt,
        category=f"edge_{category}",
        difficulty="hard",
        expected_answer=answer,
        metadata={"edge_type": category}
    )


class ProbeSetManager:
    """Manage the fixed probe set for representation tracking."""

    def __init__(self, base_dir: str = "/path/to/training"):
        self.base_dir = Path(base_dir)
        self.probe_file = self.base_dir / "config" / "probe_set.json"
        self.probe_file.parent.mkdir(parents=True, exist_ok=True)

        self.probes: List[Probe] = []
        if self.probe_file.exists():
            self._load()

    def _load(self) -> None:
        """Load probes from file."""
        with open(self.probe_file) as f:
            data = json.load(f)
            self.probes = [Probe(**p) for p in data.get("probes", [])]
        logger.info(f"Loaded {len(self.probes)} probes")

    def _save(self) -> None:
        """Save probes to file."""
        data = {
            "created": datetime.now().isoformat(),
            "total": len(self.probes),
            "probes": [asdict(p) for p in self.probes]
        }
        with open(self.probe_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.probes)} probes to {self.probe_file}")

    def initialize(self, seed: int = 42) -> None:
        """Initialize the probe set with canonical probes."""
        random.seed(seed)
        self.probes = []

        # Generate SYLLO probes by difficulty
        # 100 easy, 100 medium, 100 hard = 300 SYLLO
        for difficulty, templates in SYLLO_TEMPLATES.items():
            count = 100
            for i in range(count):
                template, answer = random.choice(templates)
                probe = generate_syllo_probe(template, answer, difficulty, i)
                self.probes.append(probe)

        # Generate edge case probes
        # ~20 per category = ~100 edge cases
        for category, examples in EDGE_CASES.items():
            for i, (prompt, answer) in enumerate(examples):
                probe = generate_edge_case_probe(category, prompt, answer, i)
                self.probes.append(probe)

            # Add variations
            for i in range(17):  # Fill to ~20 per category
                prompt, answer = random.choice(examples)
                probe = generate_edge_case_probe(category, prompt, answer, len(examples) + i)
                self.probes.append(probe)

        # Add some random/diverse probes for calibration
        random_probes = [
            ("What is 2 + 2?", "4", "math", "easy"),
            ("What is the capital of France?", "Paris", "knowledge", "easy"),
            ("If it rains, the ground gets wet. It rained. Is the ground wet?", "Yes", "logic", "easy"),
            ("Complete: The quick brown fox jumps over the lazy ___", "dog", "completion", "easy"),
            ("Is the following statement true: 'This statement is false'?", "Cannot determine", "paradox", "hard"),
        ]

        for i, (prompt, answer, cat, diff) in enumerate(random_probes):
            probe = Probe(
                id=f"misc_{cat}_{i}",
                prompt=prompt,
                category=cat,
                difficulty=diff,
                expected_answer=answer
            )
            self.probes.append(probe)

        # Shuffle to mix categories
        random.shuffle(self.probes)

        # Re-index
        for i, probe in enumerate(self.probes):
            probe.id = f"probe_{i:04d}"

        self._save()
        logger.info(f"Initialized probe set with {len(self.probes)} probes")

    def get_probes(self, category: Optional[str] = None, difficulty: Optional[str] = None) -> List[Probe]:
        """Get probes, optionally filtered."""
        probes = self.probes

        if category:
            probes = [p for p in probes if p.category == category or category in p.category]

        if difficulty:
            probes = [p for p in probes if p.difficulty == difficulty]

        return probes

    def get_prompts(self) -> List[str]:
        """Get just the prompt strings."""
        return [p.prompt for p in self.probes]

    def stats(self) -> Dict:
        """Get statistics about the probe set."""
        stats = {
            "total": len(self.probes),
            "by_category": {},
            "by_difficulty": {},
        }

        for probe in self.probes:
            # Count by category
            cat = probe.category
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Count by difficulty
            diff = probe.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        return stats


def main():
    parser = argparse.ArgumentParser(description="Probe Set Manager")
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory')
    parser.add_argument('--init', action='store_true',
                       help='Initialize probe set')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for initialization')
    parser.add_argument('--stats', action='store_true',
                       help='Show probe set statistics')
    parser.add_argument('--export', type=str,
                       help='Export probes to file')
    parser.add_argument('--list', action='store_true',
                       help='List all probes')
    parser.add_argument('--category', type=str,
                       help='Filter by category')

    args = parser.parse_args()

    manager = ProbeSetManager(args.base_dir)

    if args.init:
        manager.initialize(args.seed)
        print(f"Initialized {len(manager.probes)} probes")

    elif args.stats:
        stats = manager.stats()
        print(f"\nProbe Set Statistics:")
        print(f"  Total: {stats['total']}")
        print(f"\n  By Category:")
        for cat, count in sorted(stats['by_category'].items()):
            print(f"    {cat}: {count}")
        print(f"\n  By Difficulty:")
        for diff, count in sorted(stats['by_difficulty'].items()):
            print(f"    {diff}: {count}")

    elif args.export:
        output_path = Path(args.export)
        with open(output_path, 'w') as f:
            json.dump([asdict(p) for p in manager.probes], f, indent=2)
        print(f"Exported {len(manager.probes)} probes to {output_path}")

    elif args.list:
        probes = manager.get_probes(category=args.category)
        for p in probes[:20]:  # Show first 20
            print(f"[{p.id}] ({p.category}/{p.difficulty}) {p.prompt[:60]}...")
        if len(probes) > 20:
            print(f"... and {len(probes) - 20} more")

    else:
        # Default: show summary
        if manager.probes:
            stats = manager.stats()
            print(f"Probe Set: {stats['total']} probes")
            print(f"Use --stats for details, --init to reinitialize")
        else:
            print("No probe set found. Use --init to create one.")


if __name__ == "__main__":
    main()
