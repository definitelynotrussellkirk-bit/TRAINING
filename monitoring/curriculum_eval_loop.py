#!/usr/bin/env python3
"""
Curriculum Evaluation Loop

Periodically evaluates the trained model on current curriculum level,
records accuracy, and advances level when threshold is met.

Flow:
1. Get current SYLLO level from CurriculumManager
2. Generate test problems at that level via SYLLO API
3. Get model answers via inference server
4. Score answers and record accuracy
5. Check for level advancement (80% over 3 evals)
6. Repeat every N minutes

Usage:
    python3 monitoring/curriculum_eval_loop.py --interval 600
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.curriculum_manager import CurriculumManager, SKILL_LEVELS

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurriculumEvalLoop:
    """
    Evaluates model on curriculum problems and manages progression.
    """

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        syllo_host: str = "localhost",
        syllo_port: int = 8080,
        inference_host: str = "192.168.x.x",
        inference_port: int = 8765,
        problems_per_eval: int = 20,
        interval: int = 600,  # 10 minutes
    ):
        self.base_dir = Path(base_dir)
        self.syllo_url = f"http://{syllo_host}:{syllo_port}"
        self.inference_url = f"http://{inference_host}:{inference_port}"
        self.problems_per_eval = problems_per_eval
        self.interval = interval

        # Load config
        config_path = self.base_dir / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize curriculum manager
        self.curriculum = CurriculumManager(self.base_dir, self.config)

        # Status file
        self.status_file = self.base_dir / "status" / "curriculum_eval.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def check_services(self) -> Tuple[bool, str]:
        """Check if required services are running."""
        # Check SYLLO API
        try:
            r = requests.get(f"{self.syllo_url}/health", timeout=5)
            if r.status_code != 200:
                return False, f"SYLLO API unhealthy: {r.status_code}"
        except Exception as e:
            return False, f"SYLLO API unreachable: {e}"

        # Check inference server
        try:
            r = requests.get(f"{self.inference_url}/health", timeout=5)
            if r.status_code != 200:
                return False, f"Inference server unhealthy: {r.status_code}"
        except Exception as e:
            return False, f"Inference server unreachable: {e}"

        return True, "All services OK"

    def get_current_step(self) -> int:
        """Get current training step from status file."""
        try:
            status_path = self.base_dir / "status" / "training_status.json"
            if status_path.exists():
                with open(status_path) as f:
                    status = json.load(f)
                return status.get("current_step", 0)
        except Exception:
            pass
        return 0

    def generate_test_problems(self, skill: str, level_config: Dict) -> List[Dict]:
        """Generate test problems for the current level."""
        word_count = level_config.get("word_count", 4)

        try:
            response = requests.post(
                f"{self.syllo_url}/generate",
                json={
                    "count": self.problems_per_eval,
                    "word_count": word_count,
                    "min_words": word_count,
                    "max_words": word_count,
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("puzzles", [])
        except Exception as e:
            logger.error(f"Failed to generate problems: {e}")
            return []

    def format_puzzle_prompt(self, puzzle: Dict) -> str:
        """Format a puzzle into a prompt matching the ACTUAL training data format (JSON output)."""
        rules = puzzle.get("rules", {})
        words = puzzle.get("words", [])
        syllable_bank = puzzle.get("syllable_bank", [])
        word_count = rules.get("word_count", len(words))
        puzzle_id = puzzle.get("puzzle_id", "eval_001")

        # Build prompt to match actual training data format
        prompt_lines = [
            f"SYLLO Puzzle {puzzle_id}",
            "You must recover every hidden word by assigning syllable tiles to definitions.",
            f"Difficulty: Easy (level 1 evaluation).",
            "Rules:",
            f"- {word_count} target words.",
            "- Each word lists its syllable count via blank slots.",
            "- Return your answers as JSON with keys `solutions` and `inventory_check`.",
            "",
            "Word slots:"
        ]

        # Add word clues with blanks
        for i, word in enumerate(words, 1):
            hint = word.get("definition", word.get("hint", ""))
            syllables = word.get("syllables", [])
            syllable_count = len(syllables) if syllables else 3
            blanks = " ".join(["___"] * syllable_count)
            prompt_lines.append(f"{i}. {blanks} — {hint}")

        # Add syllable bank
        prompt_lines.extend([
            "",
            f"Syllable bank: {syllable_bank}",
            "",
            "Return valid JSON:"
        ])

        return "\n".join(prompt_lines)

    def get_model_answer(self, prompt: str) -> Optional[str]:
        """Get model's answer via inference server."""
        try:
            response = requests.post(
                f"{self.inference_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.1,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Extract answer
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return None
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None

    def extract_words_from_answer(self, answer: str) -> List[str]:
        """Extract word list from model's JSON answer format."""
        if not answer:
            return []

        words = []

        # Try to parse JSON from the response
        try:
            # Find JSON in the response (might have text before/after)
            json_match = re.search(r'\{[^{}]*"(?:letters|solutions)"[^{}]*\[.*?\][^{}]*\}', answer, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Handle both "letters" and "solutions" keys
                items = data.get("letters", data.get("solutions", []))
                for item in items:
                    if isinstance(item, dict) and "word" in item:
                        words.append(item["word"].lower())
                if words:
                    return words
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: try to find words in WORD format or quoted strings
        # Look for "word": "WORDNAME" patterns
        word_matches = re.findall(r'"word":\s*"([A-Za-z]+)"', answer, re.IGNORECASE)
        if word_matches:
            return [w.lower() for w in word_matches]

        # Last resort: find capitalized words that look like answers
        cap_words = re.findall(r'\b([A-Z]{4,})\b', answer)
        if cap_words:
            return [w.lower() for w in cap_words]

        return words

    def check_answer(self, puzzle: Dict, model_answer: str) -> Tuple[bool, float]:
        """
        Check if model's answer is correct.

        Returns (is_correct, partial_score)
        """
        expected_words = [w.get("label", "").lower() for w in puzzle.get("words", [])]
        model_words = self.extract_words_from_answer(model_answer)

        if not expected_words:
            return False, 0.0

        # Check exact match (order doesn't matter)
        expected_set = set(expected_words)
        model_set = set(model_words)

        correct_count = len(expected_set & model_set)
        total = len(expected_set)

        partial_score = correct_count / total if total > 0 else 0.0
        is_correct = expected_set == model_set

        return is_correct, partial_score

    def run_evaluation(self, skill: str = "syllo") -> Dict:
        """
        Run a full evaluation cycle.

        Returns evaluation results.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {skill.upper()} evaluation")
        logger.info(f"{'='*60}")

        # Get current level
        level_config = self.curriculum.get_current_level(skill)
        level_num = level_config.get("level", 1)
        level_name = level_config.get("name", "Unknown")
        word_count = level_config.get("word_count", 4)

        logger.info(f"Current level: {level_num} ({level_name}) - {word_count} words")

        # Generate test problems
        logger.info(f"Generating {self.problems_per_eval} test problems...")
        puzzles = self.generate_test_problems(skill, level_config)

        if not puzzles:
            logger.error("Failed to generate test problems")
            return {"success": False, "error": "No problems generated"}

        logger.info(f"Generated {len(puzzles)} puzzles")

        # Evaluate each puzzle
        results = []
        correct = 0
        total = len(puzzles)

        for i, puzzle in enumerate(puzzles, 1):
            prompt = self.format_puzzle_prompt(puzzle)
            answer = self.get_model_answer(prompt)

            is_correct, partial = self.check_answer(puzzle, answer)

            if is_correct:
                correct += 1
                logger.info(f"  [{i}/{total}] ✓ Correct")
            else:
                expected = [w.get("label") for w in puzzle.get("words", [])]
                got = self.extract_words_from_answer(answer)
                logger.info(f"  [{i}/{total}] ✗ Wrong - expected {expected}, got {got}")

            results.append({
                "puzzle_id": puzzle.get("puzzle_id"),
                "correct": is_correct,
                "partial_score": partial,
                "expected": [w.get("label") for w in puzzle.get("words", [])],
                "model_answer": self.extract_words_from_answer(answer)
            })

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"\nResults: {correct}/{total} correct ({accuracy:.1%})")

        # Record in curriculum
        step = self.get_current_step()
        self.curriculum.record_accuracy(
            skill=skill,
            accuracy=accuracy,
            step=step,
            metadata={
                "level": level_num,
                "problems": total,
                "correct": correct,
            }
        )
        logger.info(f"Recorded accuracy in curriculum (step {step})")

        # Check for progression
        should_advance, reason = self.curriculum.should_progress(skill)
        logger.info(f"Should advance: {should_advance} - {reason}")

        advanced = False
        new_level = level_num

        if should_advance:
            result = self.curriculum.progress_to_next_level(skill)
            if result.get("success"):
                advanced = True
                new_level = result.get("new_level", level_num + 1)
                logger.info(f"🎉 ADVANCED to level {new_level}!")
            else:
                logger.warning(f"Advancement failed: {result.get('message')}")

        # Save curriculum state
        self.curriculum._save_state()

        return {
            "success": True,
            "skill": skill,
            "level": level_num,
            "level_name": level_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "step": step,
            "advanced": advanced,
            "new_level": new_level,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }

    def write_status(self, eval_result: Dict):
        """Write evaluation status to file."""
        status = {
            "last_eval": eval_result,
            "curriculum_state": self.curriculum.get_status(),
            "updated_at": datetime.now().isoformat()
        }

        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def run_loop(self):
        """Main evaluation loop."""
        logger.info("="*60)
        logger.info("Curriculum Evaluation Loop Starting")
        logger.info("="*60)
        logger.info(f"SYLLO API: {self.syllo_url}")
        logger.info(f"Inference: {self.inference_url}")
        logger.info(f"Problems per eval: {self.problems_per_eval}")
        logger.info(f"Interval: {self.interval}s ({self.interval/60:.1f} min)")
        logger.info("="*60)

        # Check services
        ok, msg = self.check_services()
        if not ok:
            logger.error(f"Service check failed: {msg}")
            return
        logger.info(f"Services: {msg}")

        # Show current state
        status = self.curriculum.get_status()
        logger.info(f"Active skill: {status.get('active_skill', 'syllo')}")
        for skill, data in status.get("skills", {}).items():
            logger.info(f"  {skill}: level {data.get('level')}, {len(data.get('accuracy_history', []))} evals")

        while True:
            try:
                # Run evaluation on active skill
                skill = self.curriculum.state.get("active_skill", "syllo")
                result = self.run_evaluation(skill)

                if result.get("success"):
                    self.write_status(result)

                    # Log summary
                    logger.info(f"\n📊 Evaluation Summary:")
                    logger.info(f"   Skill: {result['skill']} Level {result['level']}")
                    logger.info(f"   Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
                    if result.get("advanced"):
                        logger.info(f"   🎉 LEVEL UP → {result['new_level']}!")

                logger.info(f"\n💤 Next evaluation in {self.interval}s ({self.interval/60:.1f} min)...\n")
                time.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("\nStopping evaluation loop...")
                break
            except Exception as e:
                logger.error(f"Evaluation error: {e}", exc_info=True)
                logger.info(f"Retrying in {self.interval}s...")
                time.sleep(self.interval)


def run_with_scheduler(scheduler_url: str, interval: int, problems: int):
    """Run as a scheduler client - submit tasks instead of executing directly."""
    from monitoring.task_client import TaskClient

    client = TaskClient(scheduler_url)

    logger.info("="*60)
    logger.info("Curriculum Eval - Scheduler Mode")
    logger.info("="*60)
    logger.info(f"Scheduler: {scheduler_url}")
    logger.info(f"Interval: {interval}s")
    logger.info(f"Problems per eval: {problems}")

    if not client.is_healthy():
        logger.error("Scheduler not available!")
        return

    while True:
        try:
            logger.info("\nSubmitting curriculum_eval task to scheduler...")

            result = client.submit_and_wait(
                task_type="curriculum_eval",
                params={
                    "skill": "syllo",
                    "num_problems": problems
                },
                priority=1,  # HIGH
                timeout=300.0
            )

            if result:
                logger.info(f"Task completed: {result.get('status')}")
                if result.get('result'):
                    r = result['result']
                    logger.info(f"  Accuracy: {r.get('accuracy', 0)*100:.1f}%")
            else:
                logger.warning("Task failed or timed out")

            logger.info(f"\n💤 Next submission in {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\nStopping...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Curriculum Evaluation Loop")
    parser.add_argument("--base-dir", default="/path/to/training",
                        help="Base directory")
    parser.add_argument("--syllo-host", default="localhost",
                        help="SYLLO API host")
    parser.add_argument("--syllo-port", type=int, default=8080,
                        help="SYLLO API port")
    parser.add_argument("--inference-host", default="192.168.x.x",
                        help="Inference server host")
    parser.add_argument("--inference-port", type=int, default=8765,
                        help="Inference server port")
    parser.add_argument("--problems", type=int, default=20,
                        help="Problems per evaluation")
    parser.add_argument("--interval", type=int, default=600,
                        help="Evaluation interval in seconds (default: 600 = 10 min)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no loop)")
    parser.add_argument("--use-scheduler", action="store_true",
                        help="Submit tasks to GPU Task Scheduler instead of running directly")
    parser.add_argument("--scheduler-url", default="http://192.168.x.x:8766",
                        help="GPU Task Scheduler URL")

    args = parser.parse_args()

    # Scheduler mode
    if args.use_scheduler:
        run_with_scheduler(args.scheduler_url, args.interval, args.problems)
        return

    # Direct execution mode
    loop = CurriculumEvalLoop(
        base_dir=args.base_dir,
        syllo_host=args.syllo_host,
        syllo_port=args.syllo_port,
        inference_host=args.inference_host,
        inference_port=args.inference_port,
        problems_per_eval=args.problems,
        interval=args.interval,
    )

    if args.once:
        # Single evaluation
        result = loop.run_evaluation("syllo")
        print(json.dumps(result, indent=2))
    else:
        # Continuous loop
        loop.run_loop()


if __name__ == "__main__":
    main()
