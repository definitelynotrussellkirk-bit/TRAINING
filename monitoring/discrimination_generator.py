#!/usr/bin/env python3
"""
DEPRECATED - Use guild/sparring.py instead

This module is superseded by the new Sparring system (2025-11-27).
See: guild/sparring.py - "Sparring with the Trainers"

The new system:
- Tests against actual checkpoint (not random wrong answers)
- Generates 3 training examples per REAL mistake
- Always queues with HIGH priority (data becomes stale)
- Has dedicated validator: guild/sparring_validator.py

Usage of new system:
    python3 guild/sparring.py --skill binary --count 100

---
ORIGINAL DOCSTRING (for reference):

Discrimination + Correction Training Generator

Creates training data that teaches the model to:
1. Recognize INCORRECT outputs
2. Produce the CORRECT output after identifying errors

Format:
- Turn 1: "Here's a solution. Is it CORRECT or INCORRECT?"
- Turn 2: "It was INCORRECT. Now produce the correct solution."
- Turn 3: [Actual correct solution]

Runs as cron job to continuously generate fresh training data.

Ratio Mode (--ratio):
- Scans queue directories for SYLLO training data
- Calculates discrimination needed to reach TARGET_RATIO (20%)
- Generates deficit in batches (respects --max-per-run cap)
"""

import warnings
warnings.warn(
    "discrimination_generator.py is DEPRECATED. Use guild/sparring.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import requests
import random
import argparse
import logging
import os
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.paths import get_base_dir
except ImportError:
    def get_base_dir():
        from core.paths import get_base_dir; return get_base_dir()

# =============================================================================
# DATA LINEAGE - Generator identification for tracking
# =============================================================================
# Bump GENERATOR_VERSION when generation logic changes significantly
GENERATOR_ID = "discrimination"
GENERATOR_VERSION = "1.0.0"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SYLLO_API = "http://localhost:8080"
INFERENCE_API = "http://inference.local:8765"
API_KEY = os.environ.get("INFERENCE_ADMIN_KEY", "admin123")

# Ratio configuration
TARGET_RATIO = 0.20  # 20% discrimination to SYLLO ratio
MAX_PER_RUN = 1500   # Max examples per cron run (~1 hour @ 27/min)
MIN_PER_RUN = 50     # Minimum to generate even if caught up

# 10-level system (signal degradation model)
# Level 1-10: progressively harder via word count, vocab rarity, overlap, red herrings
ALL_LEVELS = list(range(1, 11))

# File patterns
SYLLO_PATTERNS = ['auto_gen_*.jsonl', 'syllo_*.jsonl', 'train_SYLLO_*.jsonl']
DISCRIM_PATTERNS = ['discrimination_*.jsonl']


class DiscriminationGenerator:
    """Generate discrimination + correction training data"""

    def __init__(self, base_dir: Path = None, level: str = "auto"):
        self.base_dir = base_dir or get_base_dir()
        self.level_arg = level  # "auto" or numeric like "5" or "1-3" or "all"
        self.curriculum_level = self._get_curriculum_level()
        self.levels = self._get_levels()
        self.generated_puzzles = set()  # Track to avoid duplicates

    @property
    def difficulty(self) -> str:
        """Get difficulty string from levels for filenames/logging."""
        if len(self.levels) == 1:
            return f"L{self.levels[0]}"
        elif len(self.levels) == 10:
            return "all"
        else:
            return f"L{min(self.levels)}-{max(self.levels)}"

    def _get_curriculum_level(self) -> int:
        """Read current SYLLO level from curriculum state"""
        state_file = self.base_dir / "data_manager" / "curriculum_state.json"
        try:
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    # State structure: {"skills": {"syllo": {"current_level": N}}}
                    level = state.get('skills', {}).get('syllo', {}).get('current_level', 1)
                    logger.info(f"Read curriculum level from state: {level}")
                    return level
        except Exception as e:
            logger.warning(f"Could not read curriculum state: {e}")
        return 1  # Default to level 1

    def _get_levels(self):
        """Get SYLLO levels based on level argument or curriculum"""
        if self.level_arg == "auto":
            # Use curriculum level directly
            logger.info(f"Using curriculum level: {self.curriculum_level}")
            return [self.curriculum_level]
        elif self.level_arg == "all":
            logger.info("Using all levels 1-10")
            return ALL_LEVELS
        elif "-" in str(self.level_arg):
            # Range like "1-3" or "4-7"
            try:
                start, end = map(int, self.level_arg.split("-"))
                levels = list(range(start, end + 1))
                logger.info(f"Using level range: {levels}")
                return levels
            except ValueError:
                logger.warning(f"Invalid level range '{self.level_arg}', using curriculum level")
                return [self.curriculum_level]
        else:
            # Single level like "5"
            try:
                level = int(self.level_arg)
                if 1 <= level <= 10:
                    logger.info(f"Using level: {level}")
                    return [level]
            except ValueError:
                pass
            logger.warning(f"Invalid level '{self.level_arg}', using curriculum level")
            return [self.curriculum_level]

    def _count_lines_in_files(self, patterns: list, dirs: list) -> int:
        """Count total lines in files matching patterns across directories"""
        total = 0
        seen_files = set()

        for dir_path in dirs:
            if not dir_path.exists():
                continue
            for pattern in patterns:
                for file_path in dir_path.glob(pattern):
                    # Avoid counting same file twice
                    if file_path.name in seen_files:
                        continue
                    seen_files.add(file_path.name)
                    try:
                        with open(file_path) as f:
                            lines = sum(1 for _ in f)
                            total += lines
                    except Exception as e:
                        logger.warning(f"Could not count {file_path}: {e}")
        return total

    def calculate_ratio_deficit(self) -> tuple:
        """
        Calculate how many discrimination examples needed to reach TARGET_RATIO.

        Returns:
            (deficit, syllo_count, discrim_count, current_ratio)
        """
        queue_dirs = [
            self.base_dir / "queue" / "high",
            self.base_dir / "queue" / "normal",
            self.base_dir / "queue" / "low",
            self.base_dir / "queue" / "processing",
            self.base_dir / "queue" / "recently_completed",
        ]

        # Count SYLLO examples
        syllo_count = self._count_lines_in_files(SYLLO_PATTERNS, queue_dirs)

        # Count discrimination examples
        discrim_count = self._count_lines_in_files(DISCRIM_PATTERNS, queue_dirs)

        # Calculate target and deficit
        target_discrim = int(syllo_count * TARGET_RATIO)
        deficit = max(0, target_discrim - discrim_count)

        # Current ratio
        if syllo_count > 0:
            current_ratio = discrim_count / syllo_count
        else:
            current_ratio = 0.0

        logger.info(f"SYLLO examples: {syllo_count:,}")
        logger.info(f"Discrimination examples: {discrim_count:,}")
        logger.info(f"Current ratio: {current_ratio:.1%} (target: {TARGET_RATIO:.0%})")
        logger.info(f"Target discrimination: {target_discrim:,}")
        logger.info(f"Deficit: {deficit:,}")

        return deficit, syllo_count, discrim_count, current_ratio

    def run_ratio_mode(self, max_per_run: int = None, priority: str = "high") -> Path:
        """
        Generate discrimination examples to reach TARGET_RATIO.

        Args:
            max_per_run: Max examples per run (default: MAX_PER_RUN)
            priority: Queue priority for output

        Returns:
            Path to generated file, or None if nothing needed
        """
        max_per_run = max_per_run or MAX_PER_RUN

        logger.info("=" * 60)
        logger.info("RATIO MODE - Calculating deficit")
        logger.info("=" * 60)

        deficit, syllo_count, discrim_count, current_ratio = self.calculate_ratio_deficit()

        if deficit == 0:
            logger.info(f"Already at target ratio ({current_ratio:.1%} >= {TARGET_RATIO:.0%})")
            # Still generate MIN_PER_RUN to keep learning fresh patterns
            count = MIN_PER_RUN
            logger.info(f"Generating minimum batch: {count}")
        else:
            # Generate up to max_per_run to close the gap
            count = min(deficit, max_per_run)
            logger.info(f"Generating {count} examples (deficit: {deficit}, cap: {max_per_run})")

        examples = self.generate_batch(count)

        if examples:
            path = self.save_batch(examples, priority)
            self.write_status(len(examples), path)
            logger.info(f"Generated {len(examples)} examples → {path}")
            return path
        else:
            self.write_status(0, None)
            logger.warning("No examples generated")
            return None

    def generate_fresh_puzzle(self) -> dict:
        """Generate a single fresh SYLLO puzzle"""
        level = random.choice(self.levels)

        try:
            resp = requests.post(
                f"{SYLLO_API}/generate",
                json={"count": 1, "level": level},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get('puzzles'):
                    puzzle = data['puzzles'][0]
                    return {
                        'puzzle': puzzle,
                        'level': level,
                        'puzzle_id': puzzle.get('puzzle_id', f'gen_{datetime.now().timestamp()}')
                    }
        except Exception as e:
            logger.error(f"Failed to generate puzzle: {e}")
        return None

    def format_puzzle_prompt(self, puzzle: dict, puzzle_index: int = 1) -> str:
        """
        Get the prompt for a puzzle.

        The SYLLO API now returns `prompt` directly in the correct training
        format. Just use it.
        """
        p = puzzle['puzzle']

        # The API now returns the prompt directly
        prompt = p.get("prompt")
        if prompt:
            return prompt

        # Fallback: API doesn't have new format yet - raise error
        raise ValueError(
            f"SYLLO API response missing 'prompt' field. "
            f"Make sure the SYLLO API (singleSKILL) is updated."
        )

    def _get_current_model(self) -> str:
        """Get currently loaded model from inference server"""
        try:
            resp = requests.get(f"{INFERENCE_API}/models/info", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                model = data.get("model_name", data.get("loaded_model", "current"))
                logger.debug(f"Using model from inference server: {model}")
                return model
        except Exception as e:
            logger.debug(f"Could not query model info: {e}")
        # Fallback to environment or default
        return os.environ.get("INFERENCE_MODEL", "current")

    def get_model_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Get model's response using currently loaded checkpoint"""
        try:
            model = self._get_current_model()
            resp = requests.post(
                f"{INFERENCE_API}/v1/chat/completions",
                headers={"X-API-Key": API_KEY},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Inference failed: {e}")
        return None

    def create_discrimination_correction_example(self, puzzle_data: dict) -> dict:
        """
        Create a multi-turn training example (task-agnostic format):
        1. Show output → Model says "INCORRECT" or "CORRECT"
        2. If wrong, ask for correct → Model produces raw golden answer
        """
        puzzle_prompt = self.format_puzzle_prompt(puzzle_data)
        correct_answer = puzzle_data['puzzle']['solution']

        # Get model's (likely wrong) response
        model_response = self.get_model_response(puzzle_prompt)
        if not model_response:
            return None

        # Build multi-turn conversation (GENERIC - works for ANY task)
        messages = [
            # Turn 1: User shows problem + proposed solution, asks if correct
            {
                "role": "user",
                "content": f"""{puzzle_prompt}

---
Proposed answer:
{model_response}
---

Did the model answer correctly?"""
            },
            # Turn 2: Assistant says incorrect (no explanation)
            {
                "role": "assistant",
                "content": "INCORRECT"
            },
            # Turn 3: User asks for correct answer
            {
                "role": "user",
                "content": "What should the answer have been?"
            },
            # Turn 4: Assistant provides JUST the golden answer (no prefix)
            {
                "role": "assistant",
                "content": correct_answer
            }
        ]

        return {"messages": messages, "type": "discrimination_correction"}

    def create_correct_example(self, puzzle_data: dict) -> dict:
        """
        Create example where correct answer is shown and model confirms CORRECT
        (task-agnostic format - just "CORRECT", no explanation)
        """
        puzzle_prompt = self.format_puzzle_prompt(puzzle_data)
        correct_answer = puzzle_data['puzzle']['solution']

        messages = [
            {
                "role": "user",
                "content": f"""{puzzle_prompt}

---
Proposed answer:
{correct_answer}
---

Did the model answer correctly?"""
            },
            {
                "role": "assistant",
                "content": "CORRECT"
            }
        ]

        return {"messages": messages, "type": "correct_verification"}

    def create_direct_solve_example(self, puzzle_data: dict) -> dict:
        """
        Create direct solving example (no discrimination, just solve)
        """
        puzzle_prompt = self.format_puzzle_prompt(puzzle_data)
        correct_answer = puzzle_data['puzzle']['solution']

        messages = [
            {"role": "user", "content": puzzle_prompt},
            {"role": "assistant", "content": correct_answer}
        ]

        return {"messages": messages, "type": "direct_solve"}

    def generate_batch(self, count: int = 20) -> list:
        """
        Generate a batch of training examples

        Mix:
        - 60% discrimination + correction (model wrong → identify → fix)
        - 40% correct verification (correct answer → confirm correct)

        No direct solve - regular training batches handle that.
        """
        examples = []

        disc_count = int(count * 0.6)
        correct_count = count - disc_count

        logger.info(f"Generating {disc_count} discrimination+correction, {correct_count} correct verification")

        # Generate discrimination + correction examples
        for i in range(disc_count):
            puzzle = self.generate_fresh_puzzle()
            if puzzle:
                ex = self.create_discrimination_correction_example(puzzle)
                if ex:
                    examples.append(ex)
                    logger.info(f"  Discrimination+correction {i+1}/{disc_count}")

        # Generate correct verification examples (fresh puzzles)
        for i in range(correct_count):
            puzzle = self.generate_fresh_puzzle()
            if puzzle:
                ex = self.create_correct_example(puzzle)
                if ex:
                    examples.append(ex)
                    logger.info(f"  Correct verification {i+1}/{correct_count}")

        # Shuffle
        random.shuffle(examples)

        return examples

    def save_batch(self, examples: list, priority: str = "high") -> Path:
        """Save batch to queue with lineage metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"discrimination_{self.difficulty}_{timestamp}.jsonl"

        queue_dir = self.base_dir / f"queue/{priority}"
        queue_dir.mkdir(parents=True, exist_ok=True)

        output_path = queue_dir / filename

        with open(output_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        logger.info(f"Saved {len(examples)} examples to {output_path}")

        # Write lineage metadata sidecar
        self._write_lineage_metadata(output_path, len(examples))

        return output_path

    def _write_lineage_metadata(self, output_path: Path, example_count: int):
        """Write .meta.json sidecar for data lineage tracking."""
        try:
            from core.lineage import FileLineage, write_file_lineage

            lineage = FileLineage(
                generator_id=GENERATOR_ID,
                generator_version=GENERATOR_VERSION,
                example_count=example_count,
                params={
                    "levels": self.levels,
                    "curriculum_level": self.curriculum_level,
                    "difficulty": self.difficulty,
                },
                source="monitoring/discrimination_generator.py",
            )
            write_file_lineage(output_path, lineage)
            logger.info(f"Wrote lineage metadata: {output_path}.meta.json")
        except ImportError:
            logger.debug("core.lineage not available, skipping metadata write")
        except Exception as e:
            logger.warning(f"Failed to write lineage metadata: {e}")

    def write_status(self, generated_count: int, output_path: Path = None):
        """Write status to status/discrimination_generator.json for dashboard"""
        status_dir = self.base_dir / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        status_file = status_dir / "discrimination_generator.json"

        # Get current ratio info
        deficit, syllo_count, discrim_count, current_ratio = self.calculate_ratio_deficit()

        status = {
            "last_run": datetime.now().isoformat(),
            "curriculum_level": self.curriculum_level,
            "difficulty": self.difficulty,
            "generated_this_run": generated_count,
            "output_file": str(output_path) if output_path else None,
            "ratio": {
                "current": round(current_ratio, 4),
                "target": TARGET_RATIO,
                "syllo_examples": syllo_count,
                "discrimination_examples": discrim_count,
                "deficit": deficit
            }
        }

        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        logger.info(f"Wrote status to {status_file}")

    def run(self, count: int = 20, priority: str = "high"):
        """Generate and save a batch"""
        logger.info("=" * 60)
        logger.info(f"DISCRIMINATION GENERATOR - {self.difficulty.upper()} level")
        logger.info("=" * 60)

        examples = self.generate_batch(count)

        if examples:
            path = self.save_batch(examples, priority)
            self.write_status(len(examples), path)
            logger.info(f"Generated {len(examples)} examples")
            return path
        else:
            self.write_status(0, None)
            logger.warning("No examples generated")
            return None


def main():
    parser = argparse.ArgumentParser(description="Generate discrimination training data")
    parser.add_argument('--count', type=int, default=20, help='Number of examples to generate (fixed mode)')
    parser.add_argument('--level', default='auto',
                       help='Level: auto (curriculum), 1-10, range (e.g. 1-3), or all')
    parser.add_argument('--priority', choices=['high', 'normal', 'low'], default='high',
                       help='Queue priority')
    parser.add_argument('--base-dir', default=None,
                       help='Base directory')
    parser.add_argument('--ratio', action='store_true',
                       help='Ratio mode: generate enough to reach 20%% of SYLLO data')
    parser.add_argument('--max-per-run', type=int, default=MAX_PER_RUN,
                       help=f'Max examples per run in ratio mode (default: {MAX_PER_RUN})')
    parser.add_argument('--check-only', action='store_true',
                       help='Just report current ratio, do not generate')

    args = parser.parse_args()

    generator = DiscriminationGenerator(
        base_dir=Path(args.base_dir),
        level=args.level
    )

    if args.check_only:
        # Just report the ratio
        deficit, syllo, discrim, ratio = generator.calculate_ratio_deficit()
        print(f"\nSYLLO examples:         {syllo:,}")
        print(f"Discrimination examples: {discrim:,}")
        print(f"Current ratio:           {ratio:.1%}")
        print(f"Target ratio:            {TARGET_RATIO:.0%}")
        print(f"Deficit:                 {deficit:,}")
    elif args.ratio:
        # Ratio mode - generate based on deficit
        generator.run_ratio_mode(max_per_run=args.max_per_run, priority=args.priority)
    else:
        # Fixed count mode
        generator.run(count=args.count, priority=args.priority)


if __name__ == "__main__":
    main()
