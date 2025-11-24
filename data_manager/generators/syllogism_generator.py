#!/usr/bin/env python3
"""
Syllogism Generator - Local data generation

Wraps the singleSKILL/skill_syllo_variant export script to generate syllogism training data
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SyllogismGenerator:
    """
    Generate syllogism training data using the local export script

    Wraps: /path/to/skills/skill_syllo_variant/scripts/export_training_data.py
    """

    DEFAULT_SCRIPT = Path("/path/to/skills/skill_syllo_variant/scripts/export_training_data.py")
    DEFAULT_WORD_DB = Path("/path/to/skills/skill_syllo_variant/HELPERS/resources/word_syllable_db.jsonl")

    def __init__(
        self,
        script_path: Optional[Path] = None,
        word_db: Optional[Path] = None
    ):
        self.script_path = script_path or self.DEFAULT_SCRIPT
        self.word_db = word_db or self.DEFAULT_WORD_DB

        # Validate paths
        if not self.script_path.exists():
            raise FileNotFoundError(f"Export script not found: {self.script_path}")
        if not self.word_db.exists():
            logger.warning(f"Word database not found: {self.word_db} (will be auto-built)")

    def generate(
        self,
        count: int,
        output_path: Optional[Path] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        super_hard: bool = False,
        antonym_clues: bool = False,
        output_variants: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        Generate syllogism training data

        Args:
            count: Number of examples to generate
            output_path: Where to save output (temp file if None)
            difficulty: Difficulty distribution (e.g., "easy:0.3,medium:0.5,hard:0.2")
            seed: Random seed for reproducibility
            super_hard: Enable super-hard mode
            antonym_clues: Use antonym clues
            output_variants: Output format distribution
            **kwargs: Additional arguments passed to export script

        Returns:
            Path to generated JSONL file
        """
        # Create temp output if not specified
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(tempfile.gettempdir()) / f"syllo_gen_{timestamp}.jsonl"

        # Build command
        cmd = [
            "python3",
            str(self.script_path),
            "--count", str(count),
            "--output", str(output_path),
            "--word-db", str(self.word_db)
        ]

        if difficulty:
            cmd.extend(["--difficulty", difficulty])

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if super_hard:
            cmd.append("--super-hard")

        if antonym_clues:
            cmd.append("--antonym-clues")

        if output_variants:
            cmd.extend(["--output-variants", output_variants])

        # Add any extra kwargs as flags
        for key, value in kwargs.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        logger.info(f"Running syllogism generation: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=True
            )

            logger.info(f"Generation successful: {output_path}")
            if result.stdout:
                logger.debug(f"Output: {result.stdout}")

            # Verify file was created
            if not output_path.exists():
                raise RuntimeError(f"Output file not created: {output_path}")

            return output_path

        except subprocess.TimeoutExpired as e:
            logger.error(f"Generation timed out after 600s")
            raise RuntimeError("Generation timed out") from e

        except subprocess.CalledProcessError as e:
            logger.error(f"Generation failed with code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Generation failed: {e.stderr}") from e

    def generate_to_queue(
        self,
        count: int,
        queue_dir: Path,
        priority: str = "normal",
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Path:
        """
        Generate data and save directly to training queue

        Args:
            count: Number of examples
            queue_dir: Base queue directory (e.g., /path/to/TRAINING/queue)
            priority: Queue priority (high/normal/low)
            difficulty: Difficulty distribution
            seed: Random seed
            **kwargs: Additional generation parameters

        Returns:
            Path to generated file in queue
        """
        # Determine output path in queue
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        difficulty_str = difficulty.replace(":", "").replace(",", "_") if difficulty else "mixed"
        filename = f"syllo_{difficulty_str}_{timestamp}_count{count}.jsonl"

        queue_subdir = queue_dir / priority
        queue_subdir.mkdir(parents=True, exist_ok=True)

        output_path = queue_subdir / filename

        # Generate
        return self.generate(
            count=count,
            output_path=output_path,
            difficulty=difficulty,
            seed=seed,
            **kwargs
        )

    def load_generated(self, path: Path) -> List[Dict[str, Any]]:
        """
        Load generated JSONL data

        Args:
            path: Path to JSONL file

        Returns:
            List of training examples
        """
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        logger.info(f"Loaded {len(examples)} examples from {path}")
        return examples

    def get_stats(self, path: Path) -> Dict[str, Any]:
        """
        Get statistics about generated data

        Args:
            path: Path to generated JSONL file

        Returns:
            Statistics dictionary
        """
        examples = self.load_generated(path)

        if not examples:
            return {"count": 0, "error": "No examples found"}

        # Collect stats
        difficulties = []
        word_counts = []
        red_herring_counts = []

        for ex in examples:
            meta = ex.get("metadata", {})
            difficulties.append(meta.get("difficulty", "unknown"))
            word_counts.append(meta.get("word_count", 0))
            red_herring_counts.append(meta.get("red_herring_count", 0))

        # Count difficulty distribution
        from collections import Counter
        diff_dist = Counter(difficulties)

        return {
            "count": len(examples),
            "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "difficulty_distribution": dict(diff_dist),
            "avg_word_count": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
            "avg_red_herrings": round(sum(red_herring_counts) / len(red_herring_counts), 2) if red_herring_counts else 0,
            "min_word_count": min(word_counts) if word_counts else 0,
            "max_word_count": max(word_counts) if word_counts else 0
        }


def main():
    """Demo usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate syllogism training data")
    parser.add_argument("--count", type=int, default=100, help="Number of examples")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--difficulty", type=str, help="Difficulty distribution (e.g., easy:0.3,medium:0.5,hard:0.2)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--stats", action="store_true", help="Show stats after generation")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Generate
    generator = SyllogismGenerator()

    output_file = generator.generate(
        count=args.count,
        output_path=args.output,
        difficulty=args.difficulty,
        seed=args.seed
    )

    print(f"\n✓ Generated {args.count} examples")
    print(f"  File: {output_file}")

    # Show stats if requested
    if args.stats:
        stats = generator.get_stats(output_file)
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
