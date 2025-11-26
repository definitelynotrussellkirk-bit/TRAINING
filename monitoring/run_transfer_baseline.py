#!/usr/bin/env python3
"""
Transfer Effects Baseline Runner
================================

Tests model performance on primitives and bAbI benchmarks to measure
transfer effects from syllable/binary training.

Usage:
    # Test current trained model on all primitives and bAbI
    python3 monitoring/run_transfer_baseline.py --tag trained_ckpt157000

    # Test base model
    python3 monitoring/run_transfer_baseline.py --tag base_qwen3_0.6b --model-path /path/to/base

    # Quick test (10 samples per skill)
    python3 monitoring/run_transfer_baseline.py --tag quick_test --samples 10
"""

import argparse
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransferBaselineRunner:
    """Runs transfer effect baselines on primitives and benchmarks."""

    def __init__(
        self,
        api_url: str = "http://192.168.x.x:8765",
        base_dir: str = "/path/to/training",
        samples_per_skill: int = 30
    ):
        self.api_url = api_url
        self.base_dir = Path(base_dir)
        self.samples_per_skill = samples_per_skill

        # Data directories
        self.primitives_dir = self.base_dir / "data" / "validation" / "primitives"
        self.benchmarks_dir = self.base_dir / "data" / "validation" / "benchmarks"
        self.output_dir = self.base_dir / "status" / "baselines"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def call_api(self, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """Call inference API"""
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": "Qwen3-0.6B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    return result['choices'][0]['message']['content']
            else:
                logger.warning(f"API error: {response.status_code}")
        except Exception as e:
            logger.error(f"API call failed: {e}")
        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if answer is None:
            return ""
        # Strip, lowercase, remove punctuation
        ans = answer.strip().lower()
        # Remove common prefixes
        for prefix in ['the answer is ', 'answer: ', 'result: ']:
            if ans.startswith(prefix):
                ans = ans[len(prefix):]
        return ans.strip().rstrip('.,!?')

    def check_answer(self, expected: str, actual: str) -> bool:
        """Check if answer is correct"""
        exp_norm = self.normalize_answer(expected)
        act_norm = self.normalize_answer(actual)

        # Exact match
        if exp_norm == act_norm:
            return True

        # Check if expected is contained in actual (for longer responses)
        if exp_norm in act_norm:
            return True

        return False

    def load_skill_data(self, skill_dir: Path, difficulty: str = "easy") -> List[Dict]:
        """Load validation data for a skill"""
        examples = []

        # Find validation file
        pattern = f"val_{skill_dir.name}_{difficulty}_*.jsonl"
        files = list(skill_dir.glob(pattern))

        if not files:
            # Try without difficulty
            files = list(skill_dir.glob("*.jsonl"))

        if not files:
            return []

        # Load from first matching file
        with open(files[0]) as f:
            for line in f:
                if line.strip():
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return examples[:self.samples_per_skill]

    def test_skill(self, skill_name: str, examples: List[Dict]) -> Dict:
        """Test a single skill and return results"""
        results = {
            "skill": skill_name,
            "total": 0,
            "correct": 0,
            "errors": [],
            "accuracy": 0.0,
            "avg_time": 0.0
        }

        if not examples:
            return results

        times = []

        for i, example in enumerate(examples):
            prompt = example.get("user_prompt", "")
            expected = example.get("expected_answer", "")

            if not prompt or not expected:
                continue

            start = time.time()
            actual = self.call_api(prompt)
            elapsed = time.time() - start
            times.append(elapsed)

            results["total"] += 1

            if actual and self.check_answer(expected, actual):
                results["correct"] += 1
            else:
                # Store first few errors for debugging
                if len(results["errors"]) < 3:
                    results["errors"].append({
                        "prompt": prompt[:100],
                        "expected": expected,
                        "actual": actual[:100] if actual else None
                    })

            if (i + 1) % 10 == 0:
                logger.info(f"    {skill_name}: {i+1}/{len(examples)} done")

        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
        if times:
            results["avg_time"] = sum(times) / len(times)

        return results

    def run_primitives(self) -> Dict[str, Dict]:
        """Run all primitive skill tests"""
        results = {}

        if not self.primitives_dir.exists():
            logger.warning(f"Primitives dir not found: {self.primitives_dir}")
            return results

        skill_dirs = sorted([d for d in self.primitives_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(skill_dirs)} primitive skills")

        for skill_dir in skill_dirs:
            skill_name = skill_dir.name
            logger.info(f"  Testing: {skill_name}")

            examples = self.load_skill_data(skill_dir, "easy")
            if not examples:
                logger.warning(f"    No data for {skill_name}")
                continue

            result = self.test_skill(skill_name, examples)
            results[skill_name] = result

            logger.info(f"    {skill_name}: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")

        return results

    def run_babi(self) -> Dict[str, Dict]:
        """Run all bAbI QA tests"""
        results = {}

        if not self.benchmarks_dir.exists():
            logger.warning(f"Benchmarks dir not found: {self.benchmarks_dir}")
            return results

        babi_dirs = sorted([d for d in self.benchmarks_dir.iterdir()
                           if d.is_dir() and d.name.startswith("babi_")])
        logger.info(f"Found {len(babi_dirs)} bAbI tasks")

        for task_dir in babi_dirs:
            task_name = task_dir.name
            logger.info(f"  Testing: {task_name}")

            examples = self.load_skill_data(task_dir, "easy")
            if not examples:
                logger.warning(f"    No data for {task_name}")
                continue

            result = self.test_skill(task_name, examples)
            results[task_name] = result

            logger.info(f"    {task_name}: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")

        return results

    def run_all(self, tag: str) -> Dict:
        """Run all transfer baseline tests"""
        logger.info("=" * 60)
        logger.info(f"TRANSFER BASELINE: {tag}")
        logger.info("=" * 60)

        results = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "samples_per_skill": self.samples_per_skill,
            "skills": {}
        }

        # Run primitives
        logger.info("\n[PRIMITIVES]")
        primitives = self.run_primitives()
        results["skills"].update(primitives)

        # Run bAbI
        logger.info("\n[bAbI]")
        babi = self.run_babi()
        results["skills"].update(babi)

        # Calculate summaries
        prim_acc = [r["accuracy"] for r in primitives.values() if r["total"] > 0]
        babi_acc = [r["accuracy"] for r in babi.values() if r["total"] > 0]

        results["summary"] = {
            "primitives": {
                "count": len(prim_acc),
                "avg_accuracy": sum(prim_acc) / len(prim_acc) if prim_acc else 0
            },
            "babi": {
                "count": len(babi_acc),
                "avg_accuracy": sum(babi_acc) / len(babi_acc) if babi_acc else 0
            }
        }

        # Save results
        output_file = self.output_dir / f"baseline_{tag}.json"

        # Load existing and merge
        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)
            existing["skills"].update(results["skills"])
            existing["timestamp"] = results["timestamp"]
            if "summary" not in existing:
                existing["summary"] = {}
            existing["summary"].update(results["summary"])
            results = existing

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_file}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Primitives: {results['summary']['primitives']['count']} skills, "
                   f"avg {results['summary']['primitives']['avg_accuracy']:.1%}")
        logger.info(f"bAbI: {results['summary']['babi']['count']} tasks, "
                   f"avg {results['summary']['babi']['avg_accuracy']:.1%}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Run transfer effect baselines")
    parser.add_argument("--tag", required=True, help="Tag for this baseline run")
    parser.add_argument("--api-url", default="http://192.168.x.x:8765",
                       help="Inference API URL")
    parser.add_argument("--base-dir", default="/path/to/training",
                       help="Base training directory")
    parser.add_argument("--samples", type=int, default=30,
                       help="Samples per skill (default: 30)")
    parser.add_argument("--primitives-only", action="store_true",
                       help="Only run primitives tests")
    parser.add_argument("--babi-only", action="store_true",
                       help="Only run bAbI tests")

    args = parser.parse_args()

    runner = TransferBaselineRunner(
        api_url=args.api_url,
        base_dir=args.base_dir,
        samples_per_skill=args.samples
    )

    if args.primitives_only:
        results = {"tag": args.tag, "timestamp": datetime.now().isoformat(), "skills": {}}
        results["skills"] = runner.run_primitives()
    elif args.babi_only:
        results = {"tag": args.tag, "timestamp": datetime.now().isoformat(), "skills": {}}
        results["skills"] = runner.run_babi()
    else:
        results = runner.run_all(args.tag)


if __name__ == "__main__":
    main()
