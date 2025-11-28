#!/usr/bin/env python3
"""
Transfer Effects Baseline Runner (Local Model)
==============================================

Tests model performance locally (loads model on this machine's GPU).
Use this for testing base model without disrupting 3090 inference server.

Usage:
    # Test base model
    python3 monitoring/run_transfer_baseline_local.py \
        --tag base_qwen3_0.6b \
        --model-path models/Qwen3-0.6B \
        --samples 10
"""

import argparse
import json
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalTransferBaselineRunner:
    """Runs transfer effect baselines using local model inference."""

    def __init__(
        self,
        model_path: str,
        base_dir: str = None,
        samples_per_skill: int = 30,
        device: str = "cuda",
        max_memory_gb: int = None
    ):
        if base_dir is None:
            from core.paths import require_base_dir
            base_dir = str(require_base_dir())
        self.model_path = Path(model_path)
        self.base_dir = Path(base_dir)
        self.samples_per_skill = samples_per_skill
        self.device = device
        self.max_memory_gb = max_memory_gb  # For concurrent runs

        # Data directories
        self.primitives_dir = self.base_dir / "data" / "validation" / "primitives"
        self.benchmarks_dir = self.base_dir / "data" / "validation" / "benchmarks"
        self.output_dir = self.base_dir / "status" / "baselines"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model (loaded lazily)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        if self.model is not None:
            return

        logger.info(f"Loading model from {self.model_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Build load kwargs
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True
        }

        # Add max_memory for concurrent runs
        if self.max_memory_gb:
            load_kwargs["max_memory"] = {0: f"{self.max_memory_gb}GB"}
            logger.info(f"Using max_memory: {self.max_memory_gb}GB")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        self.model.eval()
        logger.info("Model loaded!")

    def generate(self, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """Generate response using local model"""
        try:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = f"User: {prompt}\nAssistant:"

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            generated = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if answer is None:
            return ""
        ans = answer.strip().lower()
        for prefix in ['the answer is ', 'answer: ', 'result: ']:
            if ans.startswith(prefix):
                ans = ans[len(prefix):]
        return ans.strip().rstrip('.,!?')

    def check_answer(self, expected: str, actual: str) -> bool:
        """Check if answer is correct"""
        exp_norm = self.normalize_answer(expected)
        act_norm = self.normalize_answer(actual)
        if exp_norm == act_norm:
            return True
        if exp_norm in act_norm:
            return True
        return False

    def load_skill_data(self, skill_dir: Path, difficulty: str = "easy") -> List[Dict]:
        """Load validation data for a skill"""
        examples = []
        pattern = f"val_{skill_dir.name}_{difficulty}_*.jsonl"
        files = list(skill_dir.glob(pattern))
        if not files:
            files = list(skill_dir.glob("*.jsonl"))
        if not files:
            return []

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
            actual = self.generate(prompt)
            elapsed = time.time() - start
            times.append(elapsed)

            results["total"] += 1

            if actual and self.check_answer(expected, actual):
                results["correct"] += 1
            else:
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
        logger.info(f"TRANSFER BASELINE (LOCAL): {tag}")
        logger.info(f"Model: {self.model_path}")
        logger.info("=" * 60)

        # Load model
        self.load_model()

        results = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
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

        # Save results - MERGE with existing file
        output_file = self.output_dir / f"baseline_{tag}.json"

        if output_file.exists():
            with open(output_file) as f:
                existing = json.load(f)
            # Merge skills
            existing["skills"].update(results["skills"])
            existing["timestamp"] = results["timestamp"]
            existing["model_path"] = results.get("model_path")
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

        # Cleanup
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

        return results


def main():
    parser = argparse.ArgumentParser(description="Run transfer effect baselines (local model)")
    parser.add_argument("--tag", required=True, help="Tag for this baseline run")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--base-dir", default=None,
                       help="Base training directory (default: auto-detected)")
    parser.add_argument("--samples", type=int, default=30,
                       help="Samples per skill (default: 30)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-memory", type=int, default=None,
                       help="Max GPU memory in GB (for concurrent runs)")

    args = parser.parse_args()

    runner = LocalTransferBaselineRunner(
        model_path=args.model_path,
        base_dir=args.base_dir,
        samples_per_skill=args.samples,
        device=args.device,
        max_memory_gb=args.max_memory
    )

    runner.run_all(args.tag)


if __name__ == "__main__":
    main()
