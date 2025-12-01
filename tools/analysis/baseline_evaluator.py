#!/usr/bin/env python3
"""
Baseline Model Evaluator
========================

Tests models against validation sets and saves results for comparison.
Supports per-skill evaluation and stores historical baselines.

Usage:
    # Test base model
    python3 baseline_evaluator.py --model-path /path/to/Qwen3-0.6B --tag base_model

    # Test trained checkpoint
    python3 baseline_evaluator.py --model-path /path/to/checkpoint-156000 --tag checkpoint_156k

    # Compare results
    python3 baseline_evaluator.py --compare base_model checkpoint_156k
"""

import argparse
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluates models against validation sets and tracks results."""

    def __init__(
        self,
        base_dir: str = None,
        api_url: str = None,
        results_dir: str = None
    ):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        if api_url is None:
            from core.paths import get_remote_api_url
            api_url = get_remote_api_url()
        self.base_dir = Path(base_dir)
        self.api_url = api_url
        self.results_dir = Path(results_dir or self.base_dir / "status" / "baselines")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.validation_dir = self.base_dir / "data" / "validation"

    def load_validation_data(self, difficulty: str = None) -> Dict[str, List[Dict]]:
        """Load validation data, optionally filtered by difficulty."""
        data_by_difficulty = defaultdict(list)

        # Load new validation files
        for pattern in ["val_easy_*.jsonl", "val_medium_*.jsonl", "val_hard_*.jsonl"]:
            for filepath in self.validation_dir.glob(pattern):
                difficulty_name = filepath.stem.split("_")[1]  # val_easy_200 -> easy
                with open(filepath) as f:
                    for line in f:
                        if line.strip():
                            example = json.loads(line)
                            data_by_difficulty[difficulty_name].append(example)

        if difficulty:
            return {difficulty: data_by_difficulty.get(difficulty, [])}
        return dict(data_by_difficulty)

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract JSON answer from model output."""
        # Try to find JSON in the response
        try:
            # Look for JSON object
            match = re.search(r'\{[^{}]*"solutions"[^{}]*\}', text, re.DOTALL)
            if match:
                return match.group(0)

            # Try parsing the whole thing as JSON
            json.loads(text)
            return text
        except:
            pass
        return text.strip()

    def evaluate_single(self, example: Dict, api_url: str) -> Dict:
        """Evaluate a single example."""
        import requests

        messages = example.get("messages", [])
        if not messages:
            return {"error": "No messages in example"}

        user_msg = messages[0].get("content", "") if messages else ""
        expected = messages[1].get("content", "") if len(messages) > 1 else ""

        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": user_msg}],
                    "max_tokens": 2048,
                    "temperature": 0.1
                },
                timeout=60
            )
            inference_time = time.time() - start_time

            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}"}

            result = response.json()
            model_output = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Compare answers
            model_answer = self.extract_answer(model_output)
            expected_answer = self.extract_answer(expected)

            # Normalize for comparison
            try:
                model_json = json.loads(model_answer) if model_answer else {}
                expected_json = json.loads(expected_answer) if expected_answer else {}

                # Check solutions match
                model_solutions = model_json.get("solutions", [])
                expected_solutions = expected_json.get("solutions", [])

                correct = len(model_solutions) == len(expected_solutions)
                if correct:
                    for ms, es in zip(model_solutions, expected_solutions):
                        if ms.get("answer", "").upper() != es.get("answer", "").upper():
                            correct = False
                            break
            except:
                # Fallback to string comparison
                correct = model_answer.strip().lower() == expected_answer.strip().lower()

            return {
                "correct": correct,
                "inference_time": inference_time,
                "model_output": model_output[:500],  # Truncate for storage
                "expected": expected[:500]
            }

        except Exception as e:
            return {"error": str(e)}

    def evaluate_model(
        self,
        tag: str,
        model_path: str = None,
        max_per_difficulty: int = 50,
        difficulties: List[str] = None
    ) -> Dict:
        """
        Evaluate model against validation sets.

        Args:
            tag: Identifier for this evaluation run
            model_path: Path to model (if using local loading)
            max_per_difficulty: Max examples per difficulty level
            difficulties: List of difficulties to test (default: all)
        """
        import requests

        difficulties = difficulties or ["easy", "medium", "hard"]

        logger.info(f"Starting evaluation: {tag}")
        logger.info(f"Testing difficulties: {difficulties}")
        logger.info(f"Max examples per difficulty: {max_per_difficulty}")

        # Load validation data
        all_data = self.load_validation_data()

        results = {
            "tag": tag,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_per_difficulty": max_per_difficulty,
                "api_url": self.api_url
            },
            "by_difficulty": {},
            "by_skill": {},
            "overall": {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "avg_inference_time": 0.0
            }
        }

        total_correct = 0
        total_tested = 0
        all_times = []

        for difficulty in difficulties:
            examples = all_data.get(difficulty, [])
            if not examples:
                logger.warning(f"No examples for difficulty: {difficulty}")
                continue

            # Sample if needed
            import random
            if len(examples) > max_per_difficulty:
                examples = random.sample(examples, max_per_difficulty)

            logger.info(f"Testing {len(examples)} {difficulty} examples...")

            diff_results = {
                "total": len(examples),
                "correct": 0,
                "errors": 0,
                "accuracy": 0.0,
                "avg_inference_time": 0.0,
                "examples": []
            }

            inference_times = []

            for i, example in enumerate(examples):
                result = self.evaluate_single(example, self.api_url)

                if "error" in result:
                    diff_results["errors"] += 1
                else:
                    if result["correct"]:
                        diff_results["correct"] += 1
                        total_correct += 1
                    inference_times.append(result["inference_time"])
                    all_times.append(result["inference_time"])

                total_tested += 1

                # Store sample results
                if len(diff_results["examples"]) < 5:
                    diff_results["examples"].append({
                        "correct": result.get("correct", False),
                        "inference_time": result.get("inference_time", 0)
                    })

                if (i + 1) % 10 == 0:
                    current_acc = diff_results["correct"] / (i + 1) * 100
                    logger.info(f"  Progress: {i+1}/{len(examples)} | Accuracy: {current_acc:.1f}%")

            diff_results["accuracy"] = diff_results["correct"] / diff_results["total"] if diff_results["total"] > 0 else 0.0
            diff_results["avg_inference_time"] = sum(inference_times) / len(inference_times) if inference_times else 0.0

            results["by_difficulty"][difficulty] = diff_results

            logger.info(f"  {difficulty.upper()}: {diff_results['correct']}/{diff_results['total']} ({diff_results['accuracy']*100:.1f}%)")

        # Calculate overall
        results["overall"]["total"] = total_tested
        results["overall"]["correct"] = total_correct
        results["overall"]["accuracy"] = total_correct / total_tested if total_tested > 0 else 0.0
        results["overall"]["avg_inference_time"] = sum(all_times) / len(all_times) if all_times else 0.0

        # Per-skill tracking (currently just SYLLO)
        results["by_skill"]["syllo"] = {
            "accuracy": results["overall"]["accuracy"],
            "by_difficulty": {d: r["accuracy"] for d, r in results["by_difficulty"].items()}
        }

        # Save results
        self.save_results(tag, results)

        return results

    def save_results(self, tag: str, results: Dict):
        """Save evaluation results."""
        filepath = self.results_dir / f"baseline_{tag}.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filepath}")

        # Also update summary file
        summary_file = self.results_dir / "baselines_summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)

        summary[tag] = {
            "timestamp": results["timestamp"],
            "overall_accuracy": results["overall"]["accuracy"],
            "by_difficulty": {d: r["accuracy"] for d, r in results["by_difficulty"].items()},
            "by_skill": results.get("by_skill", {})
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def compare(self, tags: List[str]) -> Dict:
        """Compare multiple evaluation runs."""
        results = {}

        for tag in tags:
            filepath = self.results_dir / f"baseline_{tag}.json"
            if filepath.exists():
                with open(filepath) as f:
                    results[tag] = json.load(f)
            else:
                logger.warning(f"No results found for tag: {tag}")

        if len(results) < 2:
            logger.error("Need at least 2 valid tags to compare")
            return {}

        # Print comparison table
        print("\n" + "="*70)
        print("BASELINE COMPARISON")
        print("="*70)

        print(f"\n{'Tag':<20} {'Overall':>10} {'Easy':>10} {'Medium':>10} {'Hard':>10}")
        print("-"*70)

        for tag, data in results.items():
            overall = data["overall"]["accuracy"] * 100
            easy = data["by_difficulty"].get("easy", {}).get("accuracy", 0) * 100
            medium = data["by_difficulty"].get("medium", {}).get("accuracy", 0) * 100
            hard = data["by_difficulty"].get("hard", {}).get("accuracy", 0) * 100
            print(f"{tag:<20} {overall:>9.1f}% {easy:>9.1f}% {medium:>9.1f}% {hard:>9.1f}%")

        # Calculate deltas if comparing exactly 2
        if len(results) == 2:
            tags_list = list(results.keys())
            r1, r2 = results[tags_list[0]], results[tags_list[1]]

            print("-"*70)
            delta_overall = (r2["overall"]["accuracy"] - r1["overall"]["accuracy"]) * 100
            delta_easy = (r2["by_difficulty"].get("easy", {}).get("accuracy", 0) -
                         r1["by_difficulty"].get("easy", {}).get("accuracy", 0)) * 100
            delta_medium = (r2["by_difficulty"].get("medium", {}).get("accuracy", 0) -
                          r1["by_difficulty"].get("medium", {}).get("accuracy", 0)) * 100
            delta_hard = (r2["by_difficulty"].get("hard", {}).get("accuracy", 0) -
                         r1["by_difficulty"].get("hard", {}).get("accuracy", 0)) * 100

            def fmt_delta(d):
                return f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"

            print(f"{'Delta':<20} {fmt_delta(delta_overall):>10} {fmt_delta(delta_easy):>10} {fmt_delta(delta_medium):>10} {fmt_delta(delta_hard):>10}")

        print("="*70 + "\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Evaluator")
    parser.add_argument("--base-dir", default=None, help="Base directory (default: auto-detect)")
    parser.add_argument("--api-url", default=None, help="Inference API URL (default: from hosts.json)")
    parser.add_argument("--tag", help="Tag for this evaluation run")
    parser.add_argument("--model-path", help="Path to model (for reference)")
    parser.add_argument("--max-per-difficulty", type=int, default=50)
    parser.add_argument("--difficulties", help="Comma-separated difficulties (easy,medium,hard)")
    parser.add_argument("--compare", nargs="+", help="Compare multiple tags")

    args = parser.parse_args()

    evaluator = BaselineEvaluator(
        base_dir=args.base_dir,
        api_url=args.api_url
    )

    if args.compare:
        evaluator.compare(args.compare)
    elif args.tag:
        difficulties = args.difficulties.split(",") if args.difficulties else None
        results = evaluator.evaluate_model(
            tag=args.tag,
            model_path=args.model_path,
            max_per_difficulty=args.max_per_difficulty,
            difficulties=difficulties
        )

        print(f"\n{'='*50}")
        print(f"EVALUATION COMPLETE: {args.tag}")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {results['overall']['accuracy']*100:.1f}%")
        print(f"Avg Inference Time: {results['overall']['avg_inference_time']*1000:.0f}ms")
        print(f"\nBy Difficulty:")
        for diff, data in results["by_difficulty"].items():
            print(f"  {diff.capitalize()}: {data['accuracy']*100:.1f}% ({data['correct']}/{data['total']})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
