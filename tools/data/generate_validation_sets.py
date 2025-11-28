#!/usr/bin/env python3
"""
Generate Validation Sets with Various Sizes

Creates validation sets of different sizes (50, 100, 200) for each difficulty level.
Uses existing validation data and ensures no overlap.

Usage:
    python3 generate_validation_sets.py --source data/validation/syllo_validation_1000.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

def load_and_categorize(source_file: Path):
    """Load validation data and categorize by difficulty"""
    examples_by_difficulty = defaultdict(list)

    with open(source_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            difficulty = example.get('metadata', {}).get('difficulty', 'unknown')
            examples_by_difficulty[difficulty].append(example)

    return examples_by_difficulty


def generate_validation_sets(
    examples_by_difficulty: dict,
    output_dir: Path,
    sizes: list = [50, 100, 200],
    seed: int = 42
):
    """Generate validation sets of various sizes"""
    random.seed(seed)

    for difficulty, examples in examples_by_difficulty.items():
        if difficulty == 'unknown':
            print(f"Skipping {len(examples)} examples with unknown difficulty")
            continue

        print(f"\n{difficulty.upper()}: {len(examples)} total examples")

        # Shuffle examples
        shuffled = examples.copy()
        random.shuffle(shuffled)

        for size in sizes:
            if size > len(shuffled):
                print(f"  WARNING: Requested {size} but only {len(shuffled)} available, using all")
                selected = shuffled
            else:
                selected = shuffled[:size]

            output_file = output_dir / f"{difficulty}_{size}.jsonl"

            with open(output_file, 'w') as f:
                for example in selected:
                    f.write(json.dumps(example) + '\n')

            print(f"  Created {output_file.name}: {len(selected)} examples")


def main():
    parser = argparse.ArgumentParser(description="Generate validation sets")
    parser.add_argument(
        '--source',
        type=Path,
        default=Path('data/validation/syllo_validation_1000.jsonl'),
        help='Source validation file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/validation'),
        help='Output directory'
    )
    parser.add_argument(
        '--sizes',
        type=str,
        default='50,100,200',
        help='Comma-separated sizes to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',')]

    print(f"Generating validation sets")
    print(f"  Source: {args.source}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Sizes: {sizes}")
    print(f"  Seed: {args.seed}")

    # Load and categorize
    examples_by_difficulty = load_and_categorize(args.source)

    # Generate sets
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_validation_sets(examples_by_difficulty, args.output_dir, sizes, args.seed)

    print(f"\nDone! Validation sets created in {args.output_dir}")


if __name__ == '__main__':
    main()
