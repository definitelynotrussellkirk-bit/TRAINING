#!/usr/bin/env python3
"""
10-Level SYLLO Baseline Evaluation

Runs the same validation set against multiple models to establish
baseline performance metrics across all 10 difficulty levels.

Usage:
    # Test both models (auto-detected from server)
    python3 monitoring/baseline_eval_10level.py

    # Test specific models
    python3 monitoring/baseline_eval_10level.py --models checkpoint-177000 Qwen3-0.6B-base

    # Test single model
    python3 monitoring/baseline_eval_10level.py --models checkpoint-177000

    # Limit samples per level
    python3 monitoring/baseline_eval_10level.py --samples-per-level 5
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DYNAMIC VALUES - Auto-detected from config
# ============================================================================

# Defaults if auto-detection fails
DEFAULT_API_KEY = "admin123"
VALIDATION_DATA_PATH = "data/validation/syllo_10level/syllo_eval_10level_20251126_051511.jsonl"

# ============================================================================

def get_default_inference_url():
    """Get default inference URL from host registry"""
    try:
        from core.hosts import get_service_url
        return get_service_url("inference")
    except (ImportError, Exception):
        return "http://192.168.x.x:8765"

def get_default_base_dir():
    """Get default base directory"""
    try:
        from core.paths import get_base_dir
        return str(get_base_dir())
    except (ImportError, Exception):
        from core.paths import get_base_dir; return str(get_base_dir())


class BaselineEvaluator:
    """Evaluates models across all 10 SYLLO difficulty levels."""

    def __init__(
        self,
        base_dir: str = None,
        inference_url: str = None,
        api_key: str = None,
        samples_per_level: int = None,  # None = use all
    ):
        if base_dir is None:
            base_dir = get_default_base_dir()
        if inference_url is None:
            inference_url = get_default_inference_url()
        self.base_dir = Path(base_dir)
        self.inference_url = inference_url.rstrip('/')
        # API key from env var, parameter, or default
        self.api_key = api_key or os.environ.get("INFERENCE_ADMIN_KEY", DEFAULT_API_KEY)
        self.samples_per_level = samples_per_level

        # Output paths
        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def get_available_models(self) -> List[str]:
        """Get list of models loaded on inference server."""
        try:
            resp = requests.get(
                f"{self.inference_url}/models/info",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                # DYNAMIC: Model IDs like "checkpoint-177000", "Qwen3-0.6B-base"
                return [m.get("model_id") for m in models if m.get("model_id")]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
        return []

    def load_validation_data(self) -> Dict[int, List[Dict]]:
        """Load validation data grouped by level."""
        # DYNAMIC: Path to validation data file
        data_path = self.base_dir / VALIDATION_DATA_PATH

        if not data_path.exists():
            logger.error(f"Validation data not found: {data_path}")
            return {}

        by_level: Dict[int, List[Dict]] = {i: [] for i in range(1, 11)}

        with open(data_path, 'r') as f:
            for line in f:
                try:
                    puzzle = json.loads(line.strip())
                    level = puzzle.get("level", 1)
                    if 1 <= level <= 10:
                        by_level[level].append(puzzle)
                except json.JSONDecodeError:
                    continue

        # Apply sample limit per level
        if self.samples_per_level:
            for level in by_level:
                by_level[level] = by_level[level][:self.samples_per_level]

        return by_level

    def get_model_response(self, model_id: str, prompt: str) -> Optional[str]:
        """Get model's response to a prompt."""
        try:
            # DYNAMIC: API endpoint and authentication
            resp = requests.post(
                f"{self.inference_url}/v1/chat/completions",
                headers={"X-API-Key": self.api_key},
                json={
                    "model": model_id,  # DYNAMIC: Model ID
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.1,
                },
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.debug(f"Request failed: {e}")
        return None

    def extract_answers(self, response: str) -> List[str]:
        """Extract word answers from model response."""
        if not response:
            return []

        words = []

        # Try JSON parsing first
        try:
            json_match = re.search(
                r'\{[^{}]*"(?:letters|solutions)"[^{}]*\[.*?\][^{}]*\}',
                response, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group())
                items = data.get("letters", data.get("solutions", []))
                for item in items:
                    if isinstance(item, dict):
                        word = item.get("answer", item.get("word", ""))
                        if word:
                            words.append(word.lower())
                if words:
                    return words
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for "answer": "WORD" patterns
        word_matches = re.findall(
            r'"(?:answer|word)":\s*"([A-Za-z]+)"',
            response, re.IGNORECASE
        )
        if word_matches:
            return [w.lower() for w in word_matches]

        # Last resort: capitalized words
        cap_words = re.findall(r'\b([A-Z]{4,})\b', response)
        if cap_words:
            return [w.lower() for w in cap_words]

        return words

    def score_response(
        self, puzzle: Dict, model_response: str
    ) -> Tuple[bool, float, List[str], List[str]]:
        """
        Score a model's response against expected answer.

        Returns: (is_correct, partial_score, expected_words, model_words)
        """
        expected = [w.get("label", "").lower() for w in puzzle.get("words", [])]
        model_words = self.extract_answers(model_response)

        if not expected:
            return False, 0.0, expected, model_words

        expected_set = set(expected)
        model_set = set(model_words)

        correct_count = len(expected_set & model_set)
        total = len(expected_set)

        partial_score = correct_count / total if total > 0 else 0.0
        is_correct = expected_set == model_set

        return is_correct, partial_score, expected, model_words

    def evaluate_model(
        self, model_id: str, data_by_level: Dict[int, List[Dict]]
    ) -> Dict:
        """Evaluate a single model across all levels."""
        results = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "levels": {},
            "overall": {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "partial_avg": 0.0
            }
        }

        total_correct = 0
        total_count = 0
        total_partial = 0.0

        for level in range(1, 11):
            puzzles = data_by_level.get(level, [])
            if not puzzles:
                continue

            level_correct = 0
            level_partial = 0.0
            level_results = []

            logger.info(f"  Level {level}: {len(puzzles)} puzzles...")

            for i, puzzle in enumerate(puzzles, 1):
                prompt = puzzle.get("prompt", "")
                if not prompt:
                    continue

                response = self.get_model_response(model_id, prompt)
                is_correct, partial, expected, got = self.score_response(
                    puzzle, response
                )

                if is_correct:
                    level_correct += 1
                level_partial += partial

                level_results.append({
                    "puzzle_id": puzzle.get("puzzle_id"),
                    "correct": is_correct,
                    "partial_score": partial,
                    "expected": expected,
                    "got": got
                })

                # Progress indicator
                if i % 5 == 0:
                    logger.info(f"    [{i}/{len(puzzles)}] {level_correct} correct so far")

            level_count = len(puzzles)
            level_accuracy = level_correct / level_count if level_count > 0 else 0.0
            level_partial_avg = level_partial / level_count if level_count > 0 else 0.0

            results["levels"][level] = {
                "total": level_count,
                "correct": level_correct,
                "accuracy": level_accuracy,
                "partial_avg": level_partial_avg,
                "results": level_results
            }

            logger.info(
                f"    Level {level} complete: {level_correct}/{level_count} "
                f"({level_accuracy:.1%})"
            )

            total_correct += level_correct
            total_count += level_count
            total_partial += level_partial

        # Overall stats
        results["overall"]["total"] = total_count
        results["overall"]["correct"] = total_correct
        results["overall"]["accuracy"] = (
            total_correct / total_count if total_count > 0 else 0.0
        )
        results["overall"]["partial_avg"] = (
            total_partial / total_count if total_count > 0 else 0.0
        )

        return results

    def run_evaluation(self, model_ids: List[str] = None) -> Dict:
        """Run full evaluation across specified models."""
        logger.info("=" * 60)
        logger.info("10-Level SYLLO Baseline Evaluation")
        logger.info("=" * 60)

        # Get models to evaluate
        if not model_ids:
            model_ids = self.get_available_models()
            if not model_ids:
                logger.error("No models available on inference server")
                return {"error": "No models available"}

        # DYNAMIC: Model IDs being evaluated
        logger.info(f"Models to evaluate: {model_ids}")

        # Load validation data
        data_by_level = self.load_validation_data()
        total_puzzles = sum(len(p) for p in data_by_level.values())
        logger.info(f"Loaded {total_puzzles} puzzles across 10 levels")

        if self.samples_per_level:
            logger.info(f"Using {self.samples_per_level} samples per level")

        # Evaluate each model
        all_results = {
            "run_timestamp": datetime.now().isoformat(),
            "inference_server": self.inference_url,  # DYNAMIC
            "samples_per_level": self.samples_per_level,
            "total_puzzles": total_puzzles,
            "models": {}
        }

        for model_id in model_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_id}")
            logger.info(f"{'='*60}")

            model_results = self.evaluate_model(model_id, data_by_level)
            all_results["models"][model_id] = model_results

            # Summary
            overall = model_results["overall"]
            logger.info(f"\n{model_id} Summary:")
            logger.info(f"  Overall: {overall['correct']}/{overall['total']} ({overall['accuracy']:.1%})")
            logger.info(f"  Partial score avg: {overall['partial_avg']:.2%}")

            # Per-level summary
            logger.info("  By level:")
            for level in range(1, 11):
                level_data = model_results["levels"].get(level, {})
                if level_data:
                    logger.info(
                        f"    L{level:2d}: {level_data.get('correct', 0):2d}/"
                        f"{level_data.get('total', 0):2d} "
                        f"({level_data.get('accuracy', 0):.1%})"
                    )

        # Save results
        output_file = self.status_dir / "baseline_eval_10level.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to: {output_file}")

        # Print comparison table
        self._print_comparison_table(all_results)

        return all_results

    def _print_comparison_table(self, results: Dict):
        """Print a comparison table of model performance."""
        models = list(results.get("models", {}).keys())
        if len(models) < 2:
            return

        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)

        # Header
        header = "Level  | " + " | ".join(f"{m[:15]:>15}" for m in models)
        logger.info(header)
        logger.info("-" * len(header))

        # Per-level comparison
        for level in range(1, 11):
            row = f"L{level:2d}    | "
            for model_id in models:
                level_data = results["models"][model_id]["levels"].get(level, {})
                acc = level_data.get("accuracy", 0) * 100
                row += f"{acc:>14.1f}% | "
            logger.info(row.rstrip(" | "))

        # Overall
        logger.info("-" * len(header))
        row = "TOTAL  | "
        for model_id in models:
            overall = results["models"][model_id]["overall"]
            acc = overall.get("accuracy", 0) * 100
            row += f"{acc:>14.1f}% | "
        logger.info(row.rstrip(" | "))


def main():
    parser = argparse.ArgumentParser(
        description="10-Level SYLLO Baseline Evaluation"
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory (default: auto-detected)"
    )
    parser.add_argument(
        "--inference-url",
        default=None,
        help="Inference server URL (default: auto-detected from host registry)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model IDs to evaluate (default: auto-detect from server)"
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=None,
        help="Limit samples per level (default: use all)"
    )

    args = parser.parse_args()

    evaluator = BaselineEvaluator(
        base_dir=args.base_dir,
        inference_url=args.inference_url,
        samples_per_level=args.samples_per_level,
    )

    results = evaluator.run_evaluation(args.models)

    # Exit with error code if no results
    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
