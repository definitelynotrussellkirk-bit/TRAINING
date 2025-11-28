#!/usr/bin/env python3
"""
Training Data Generation CLI

Unified tool for generating syllogism training data with curriculum integration
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.generators.syllogism_generator import SyllogismGenerator
from data_manager.curriculum_manager import CurriculumManager

logger = logging.getLogger(__name__)


class TrainingDataCLI:
    """CLI for training data generation"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.queue_dir = base_dir / "queue"
        self.config_path = base_dir / "config.json"
        self.curriculum_state = base_dir / "data_manager" / "curriculum_state.json"

        # Load config
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Initialize generator
        self.generator = SyllogismGenerator()

        # Initialize curriculum manager if state exists
        self.curriculum = None
        if self.curriculum_state.exists():
            try:
                self.curriculum = CurriculumManager(str(self.curriculum_state))
                logger.info(f"Loaded curriculum (level {self.curriculum.level})")
            except Exception as e:
                logger.warning(f"Could not load curriculum: {e}")

    def generate(
        self,
        count: int,
        priority: str = "normal",
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        use_curriculum: bool = False,
        **kwargs
    ) -> Path:
        """
        Generate training data

        Args:
            count: Number of examples
            priority: Queue priority (high/normal/low)
            difficulty: Difficulty distribution (overrides curriculum)
            seed: Random seed
            use_curriculum: Use curriculum manager for adaptive difficulty
            **kwargs: Additional generation parameters

        Returns:
            Path to generated file
        """
        # Use curriculum if requested and available
        if use_curriculum and self.curriculum:
            logger.info(f"Using curriculum (level {self.curriculum.level})")
            difficulty = self.curriculum.get_difficulty_config()
            logger.info(f"Curriculum difficulty: {difficulty}")

        # Use default from config if not specified
        if not difficulty:
            difficulty = self.config.get("default_difficulty", "easy:0.3,medium:0.5,hard:0.2")

        logger.info(f"Generating {count} examples")
        logger.info(f"  Priority: {priority}")
        logger.info(f"  Difficulty: {difficulty}")
        if seed:
            logger.info(f"  Seed: {seed}")

        # Generate directly to queue
        output_file = self.generator.generate_to_queue(
            count=count,
            queue_dir=self.queue_dir,
            priority=priority,
            difficulty=difficulty,
            seed=seed,
            **kwargs
        )

        # Update curriculum if used
        if use_curriculum and self.curriculum:
            # For now, just mark as generated
            # In the future, this could be updated based on training performance
            pass

        return output_file

    def stats(self, file_path: Optional[Path] = None) -> Dict:
        """Get statistics about generated data or queue"""
        if file_path:
            # Stats for specific file
            return self.generator.get_stats(file_path)
        else:
            # Stats for entire queue
            stats = {
                "queue": {},
                "total_files": 0,
                "total_size_mb": 0
            }

            for priority in ["high", "normal", "low", "processing"]:
                priority_dir = self.queue_dir / priority
                if not priority_dir.exists():
                    continue

                files = list(priority_dir.glob("*.jsonl"))
                total_size = sum(f.stat().st_size for f in files)

                stats["queue"][priority] = {
                    "files": len(files),
                    "size_mb": round(total_size / 1024 / 1024, 2)
                }
                stats["total_files"] += len(files)
                stats["total_size_mb"] += total_size / 1024 / 1024

            stats["total_size_mb"] = round(stats["total_size_mb"], 2)

            return stats

    def curriculum_status(self) -> Dict:
        """Get curriculum status"""
        if not self.curriculum:
            return {"error": "Curriculum not loaded"}

        return {
            "level": self.curriculum.level,
            "difficulty": self.curriculum.get_difficulty_config(),
            "total_generated": self.curriculum.stats.get("total_generated", 0),
            "last_update": self.curriculum.stats.get("last_update")
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate syllogism training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100k examples with mixed difficulty
  %(prog)s --count 100000 --difficulty "easy:0.33,medium:0.33,hard:0.34"

  # Use curriculum manager (adaptive difficulty)
  %(prog)s --count 100000 --curriculum

  # Generate to high-priority queue
  %(prog)s --count 50000 --priority high

  # Super-hard mode
  %(prog)s --count 10000 --super-hard

  # Show queue statistics
  %(prog)s --stats

  # Show curriculum status
  %(prog)s --curriculum-status
        """
    )

    # Operation mode
    parser.add_argument("--stats", action="store_true", help="Show queue statistics")
    parser.add_argument("--curriculum-status", action="store_true", help="Show curriculum status")

    # Generation parameters
    parser.add_argument("--count", type=int, default=100000, help="Number of examples (default: 100000)")
    parser.add_argument("--priority", choices=["high", "normal", "low"], default="normal", help="Queue priority")
    parser.add_argument("--difficulty", type=str, help="Difficulty distribution (e.g., easy:0.3,medium:0.5,hard:0.2)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum manager for adaptive difficulty")

    # Special modes
    parser.add_argument("--super-hard", action="store_true", help="Super-hard mode (8 words, high overlap)")
    parser.add_argument("--antonym-clues", action="store_true", help="Use antonym clues")

    # Output
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--base-dir", type=Path, default=None, help="Training base directory (auto-detect if not set)")

    args = parser.parse_args()

    # Auto-detect base_dir if not provided
    if args.base_dir is None:
        from core.paths import get_base_dir
        args.base_dir = get_base_dir()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s"
    )

    # Create CLI
    cli = TrainingDataCLI(args.base_dir)

    # Handle different modes
    if args.stats:
        # Show queue stats
        stats = cli.stats()
        print("\n=== Queue Statistics ===\n")
        for priority, info in stats["queue"].items():
            print(f"{priority.capitalize():12} {info['files']:3} files  {info['size_mb']:8.2f} MB")
        print(f"\n{'Total':12} {stats['total_files']:3} files  {stats['total_size_mb']:8.2f} MB\n")
        return 0

    if args.curriculum_status:
        # Show curriculum status
        status = cli.curriculum_status()
        if "error" in status:
            print(f"Error: {status['error']}")
            return 1

        print("\n=== Curriculum Status ===\n")
        print(f"Level:          {status['level']}")
        print(f"Difficulty:     {status['difficulty']}")
        print(f"Total Generated: {status['total_generated']}")
        print(f"Last Update:    {status.get('last_update', 'Never')}\n")
        return 0

    # Generate data
    try:
        print(f"\n🎲 Generating {args.count:,} syllogism examples...\n")

        output_file = cli.generate(
            count=args.count,
            priority=args.priority,
            difficulty=args.difficulty,
            seed=args.seed,
            use_curriculum=args.curriculum,
            super_hard=args.super_hard,
            antonym_clues=args.antonym_clues
        )

        # Get stats
        stats = cli.generator.get_stats(output_file)

        print(f"\n✅ Generation Complete!\n")
        print(f"  File: {output_file.name}")
        print(f"  Location: {output_file.parent}")
        print(f"  Size: {stats['file_size_mb']} MB")
        print(f"  Examples: {stats['count']:,}")
        print(f"\n  Difficulty Distribution:")
        for diff, count in stats['difficulty_distribution'].items():
            pct = (count / stats['count']) * 100
            print(f"    {diff:12} {count:6,} ({pct:5.1f}%)")

        print(f"\n  Avg Words/Puzzle: {stats['avg_word_count']}")
        print(f"  Avg Red Herrings: {stats['avg_red_herrings']}")
        print(f"\n  The training daemon will automatically pick up this file.")
        print(f"  Monitor progress at: http://localhost:8080/live_monitor_ui.html\n")

        return 0

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
