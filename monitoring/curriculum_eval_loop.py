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
import os
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
        model: str = None,  # Model ID to use (None = auto-detect)
    ):
        self.base_dir = Path(base_dir)
        self.syllo_url = f"http://{syllo_host}:{syllo_port}"
        self.inference_url = f"http://{inference_host}:{inference_port}"
        self.problems_per_eval = problems_per_eval
        self.interval = interval

        # API key for inference server (from environment or secrets file)
        self.api_key = os.environ.get("INFERENCE_API_KEY", "admin123")

        # Model name - use provided model or auto-detect from inference server
        # DYNAMIC: Model IDs like "checkpoint-177000", "Qwen3-0.6B-base" come from 3090 /models/info
        self.model_name = model if model else self._get_current_model()
        self._model_was_specified = model is not None  # Track if user specified model

        # Load config
        config_path = self.base_dir / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Initialize curriculum manager
        self.curriculum = CurriculumManager(self.base_dir, self.config)

        # Status file
        self.status_file = self.base_dir / "status" / "curriculum_eval.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_current_model(self) -> str:
        """Get currently loaded model from inference server"""
        try:
            resp = requests.get(f"{self.inference_url}/models/info", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                model = data.get("model_name", data.get("loaded_model", "current"))
                logger.info(f"Detected model from inference server: {model}")
                return model
        except Exception as e:
            logger.debug(f"Could not query model info: {e}")
        # Fallback to environment or default
        return os.environ.get("INFERENCE_MODEL", "current")

    def check_services(self) -> Tuple[bool, str]:
        """Check if required services are running."""
        from monitoring.skill_evaluators import get_evaluator, SKILL_EVALUATORS

        # Get active skill from curriculum
        active_skill = self.curriculum.state.get("active_skill", "syllo")

        # Check skill API using the evaluator
        if active_skill not in SKILL_EVALUATORS:
            return False, f"No evaluator for active skill: {active_skill}"

        evaluator = get_evaluator(active_skill)
        if not evaluator.health_check():
            return False, f"{active_skill.upper()} API unreachable (port {SKILL_EVALUATORS[active_skill]['default_url']})"

        # Check inference server
        try:
            r = requests.get(f"{self.inference_url}/health", timeout=5)
            if r.status_code != 200:
                return False, f"Inference server unhealthy: {r.status_code}"
        except Exception as e:
            return False, f"Inference server unreachable: {e}"

        return True, f"All services OK (active skill: {active_skill})"

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
        # Pass level directly to API - it handles word_count mapping
        level = level_config.get("level", 1)

        try:
            response = requests.post(
                f"{self.syllo_url}/generate",
                json={
                    "count": self.problems_per_eval,
                    "level": level,  # API maps level to word_count internally
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("puzzles", [])
        except Exception as e:
            logger.error(f"Failed to generate problems: {e}")
            return []

    def format_puzzle_prompt(self, puzzle: Dict, puzzle_index: int = 1) -> str:
        """
        Get the prompt for a puzzle.

        The SYLLO API now returns `prompt` directly in the correct training
        format. Just use it.
        """
        # The API now returns the prompt directly
        prompt = puzzle.get("prompt")
        if prompt:
            return prompt

        # Fallback: API doesn't have new format yet - raise error
        raise ValueError(
            f"SYLLO API response missing 'prompt' field. "
            f"Make sure the SYLLO API (singleSKILL) is updated."
        )

    def get_model_answer(self, prompt: str) -> Optional[str]:
        """Get model's answer via inference server."""
        try:
            response = requests.post(
                f"{self.inference_url}/v1/chat/completions",
                headers={"X-API-Key": self.api_key},
                json={
                    "model": self.model_name,
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
        """Extract word list from model's JSON answer format.

        Handles all 12+ output variants from the SYLLO API:
        - solutions_list, solutions_sequence, solutions_map, solutions_only
        - solutions_text, solutions_table, solutions_markdown
        - inventory_vector, array_bundle, report_wrapper
        - analysis_enriched, solutions_by_letter
        """
        if not answer:
            return []

        words = []

        # Try to parse full JSON response
        try:
            data = json.loads(answer)

            # solutions_list: ["1. pub + lic → PUBLIC", ...]
            if "solutions_list" in data:
                for s in data["solutions_list"]:
                    match = re.search(r'[→=]\s*([A-Z]+)\s*$', s)
                    if match:
                        words.append(match.group(1).lower())
                if words:
                    return words

            # sequence: [{"word": "PUBLIC", ...}]
            if "sequence" in data:
                for item in data["sequence"]:
                    if isinstance(item, dict) and "word" in item:
                        words.append(item["word"].lower())
                if words:
                    return words

            # solutions_map: {"1": {"word": "PUBLIC"}, ...}
            if "solutions_map" in data:
                for k, v in data["solutions_map"].items():
                    if isinstance(v, dict) and "word" in v:
                        words.append(v["word"].lower())
                if words:
                    return words

            # solutions: [{"answer": "PUBLIC"}] or [{"word": "PUBLIC"}]
            if "solutions" in data:
                for item in data["solutions"]:
                    if isinstance(item, dict):
                        word = item.get("answer", item.get("word", ""))
                        if word:
                            words.append(word.lower())
                if words:
                    return words

            # solutions_text: "1) do + ing -> DOING; 2) ..."
            if "solutions_text" in data:
                text = data["solutions_text"]
                # Extract words after -> or = or →
                matches = re.findall(r'[→=\->]\s*([A-Z]+)', text)
                if matches:
                    return [w.lower() for w in matches]

            # solutions_table: {"headers": [...], "rows": [[syllables, word], ...]}
            if "solutions_table" in data:
                table = data["solutions_table"]
                if isinstance(table, dict) and "rows" in table:
                    for row in table["rows"]:
                        if isinstance(row, list) and len(row) >= 2:
                            word = row[-1]  # Word is last column
                            if isinstance(word, str):
                                words.append(word.lower())
                if words:
                    return words

            # inventory_vector: {"answers": {"1": "PUBLIC", "2": "MAKING"}}
            if "answers" in data:
                answers = data["answers"]
                if isinstance(answers, dict):
                    for k, v in answers.items():
                        if isinstance(v, str):
                            words.append(v.lower())
                elif isinstance(answers, list):
                    for item in answers:
                        if isinstance(item, dict) and "word" in item:
                            words.append(item["word"].lower())
                if words:
                    return words

            # report_wrapper: {"report": {"solutions": [...]}}
            if "report" in data and isinstance(data.get("report"), dict):
                report = data["report"]
                if "solutions" in report:
                    for item in report["solutions"]:
                        if isinstance(item, dict):
                            word = item.get("answer", item.get("word", ""))
                            if word:
                                words.append(word.lower())
                if words:
                    return words

            # analysis_enriched: {"analysis": {"solutions": [...]}}
            if "analysis" in data and isinstance(data.get("analysis"), dict):
                analysis = data["analysis"]
                if "solutions" in analysis:
                    for item in analysis["solutions"]:
                        if isinstance(item, dict):
                            word = item.get("answer", item.get("word", ""))
                            if word:
                                words.append(word.lower())
                if words:
                    return words

            # solutions_by_letter: {"A": {"word": "APPLE"}, ...} or {"A": "APPLE", ...}
            if "solutions_by_letter" in data:
                for letter, val in data["solutions_by_letter"].items():
                    if isinstance(val, str):
                        words.append(val.lower())
                    elif isinstance(val, dict) and "word" in val:
                        words.append(val["word"].lower())
                if words:
                    return words

        except json.JSONDecodeError:
            pass

        # Fallback: try to find words in "word": "WORDNAME" pattern
        word_matches = re.findall(r'"(?:answer|word)":\s*"([A-Za-z]+)"', answer, re.IGNORECASE)
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
        Run a full evaluation cycle using pluggable skill evaluators.

        Returns evaluation results.
        """
        from monitoring.skill_evaluators import get_evaluator, SKILL_EVALUATORS

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {skill.upper()} evaluation")
        logger.info(f"{'='*60}")

        # Get evaluator for this skill
        if skill not in SKILL_EVALUATORS:
            logger.error(f"No evaluator for skill: {skill}")
            return {"success": False, "error": f"Unknown skill: {skill}"}

        evaluator = get_evaluator(skill)

        # Check skill API is up
        if not evaluator.health_check():
            logger.error(f"{skill} API is not available")
            return {"success": False, "error": f"{skill} API unavailable"}

        # Get current level
        level_config = self.curriculum.get_current_level(skill)
        level_num = level_config.get("level", 1)
        level_name = level_config.get("name", "Unknown")

        logger.info(f"Current level: {level_num} ({level_name})")

        # Generate test problems using skill evaluator
        logger.info(f"Generating {self.problems_per_eval} test problems...")
        problems = evaluator.generate_problems(level=level_num, count=self.problems_per_eval)

        if not problems:
            logger.error("Failed to generate test problems")
            return {"success": False, "error": "No problems generated"}

        logger.info(f"Generated {len(problems)} problems")

        # Evaluate each problem
        results = []
        correct = 0
        total = len(problems)

        for i, problem in enumerate(problems, 1):
            # Get prompt and expected answer
            prompt = evaluator.get_prompt(problem)
            expected = evaluator.get_expected(problem)

            # Get model's answer
            model_response = self.get_model_answer(prompt)
            actual = evaluator.extract_answer(model_response)

            # Check correctness
            is_correct, partial = evaluator.check_correct(expected, actual)

            if is_correct:
                correct += 1
                logger.info(f"  [{i}/{total}] ✓ Correct")
            else:
                # Truncate for logging
                exp_str = str(expected)[:50]
                act_str = str(actual)[:50]
                logger.info(f"  [{i}/{total}] ✗ Wrong - expected {exp_str}..., got {act_str}...")

            results.append({
                "problem_id": problem.get("id", f"{skill}_{i}"),
                "correct": is_correct,
                "partial_score": partial,
                "expected": str(expected)[:200],
                "model_answer": str(actual)[:200]
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
            # DYNAMIC: model_id comes from --model arg or auto-detected from 3090
            "model": self.model_name,
            "model_specified": self._model_was_specified,  # True if user explicitly chose model
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
        from monitoring.skill_evaluators import SKILL_EVALUATORS

        # Get active skill for startup display
        active_skill = self.curriculum.state.get("active_skill", "syllo")
        skill_url = SKILL_EVALUATORS.get(active_skill, {}).get("default_url", "unknown")

        logger.info("="*60)
        logger.info("Curriculum Evaluation Loop Starting (Arena)")
        logger.info("="*60)
        logger.info(f"Active Trainer: {active_skill.upper()} ({skill_url})")
        logger.info(f"Inference: {self.inference_url}")
        # DYNAMIC: Model ID comes from --model arg or auto-detected
        logger.info(f"Model: {self.model_name} {'(specified)' if self._model_was_specified else '(auto-detected)'}")
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


def run_with_scheduler(scheduler_url: str, interval: int, problems: int, base_dir: str = "/path/to/training"):
    """Run as a scheduler client - submit tasks instead of executing directly."""
    import sys
    sys.path.insert(0, base_dir)
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
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID to use (e.g., 'checkpoint-177000', 'Qwen3-0.6B-base'). "
                             "If not specified, auto-detects from inference server.")
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
        model=args.model,  # DYNAMIC: Model ID like "checkpoint-177000" or "Qwen3-0.6B-base"
    )

    if args.once:
        # Single evaluation on active skill
        active_skill = loop.curriculum.state.get("active_skill", "syllo")
        result = loop.run_evaluation(active_skill)
        print(json.dumps(result, indent=2))
    else:
        # Continuous loop
        loop.run_loop()


if __name__ == "__main__":
    main()
