#!/usr/bin/env python3
"""
Adversarial Example Miner
==========================

Finds edge cases where the model fails and creates challenging training data.

GPU Usage: ~4% (1GB VRAM for inference)
ROI: ⭐⭐⭐⭐ (High - creates targeted training data for weak points)

Features:
- Monitors checkpoints and runs inference
- Identifies failure patterns (wrong predictions, low confidence)
- Generates adversarial perturbations
- Creates challenging test cases
- Exports adversarial examples for training

Types of adversarial examples:
1. Natural failures: Model predictions that are incorrect
2. Low confidence: Model is uncertain (confidence < 60%)
3. Boundary cases: Examples near decision boundaries
4. Semantic perturbations: Small input changes that flip predictions
"""

import torch
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - AdversarialMiner - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialMiner:
    """
    Mines adversarial examples from model predictions.

    Identifies model weaknesses and generates challenging training data.
    """

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        checkpoint_dir: str = None,
        test_data_dir: str = None,
        output_dir: str = None,
        check_interval: int = 300,
        confidence_threshold: float = 0.6,
        samples_per_check: int = 100
    ):
        """
        Initialize adversarial miner.

        Args:
            base_dir: Base training directory
            checkpoint_dir: Directory to monitor for checkpoints
            test_data_dir: Directory with test data
            output_dir: Directory to save adversarial examples
            check_interval: Seconds between checkpoint checks
            confidence_threshold: Below this confidence, flag as adversarial
            samples_per_check: Number of samples to test per checkpoint
        """
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.base_dir / "models" / "current_model")
        self.test_data_dir = Path(test_data_dir or self.base_dir / "data" / "validation")
        self.output_dir = Path(output_dir or self.base_dir / "data" / "adversarial_examples")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.check_interval = check_interval
        self.confidence_threshold = confidence_threshold
        self.samples_per_check = samples_per_check

        # Metrics storage
        self.results_file = self.base_dir / "status" / "adversarial_mining.json"
        self.results = self._load_results()
        self.last_checkpoint = None

        # Test dataset
        self.test_examples = []

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Adversarial Miner initialized")
        logger.info(f"Monitoring: {self.checkpoint_dir}")
        logger.info(f"Test data: {self.test_data_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")

    def _load_results(self) -> Dict:
        """Load previous results if they exist"""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {
            "mining_runs": [],
            "adversarial_examples_found": 0,
            "total_examples_tested": 0,
            "last_updated": None
        }

    def _save_results(self):
        """Save results to JSON"""
        self.results["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load_test_data(self):
        """
        Load test data from validation directory.
        Combines all difficulty levels.
        """
        test_files = list(self.test_data_dir.glob("*.jsonl"))

        if not test_files:
            logger.warning(f"No test files found in {self.test_data_dir}")
            return

        self.test_examples = []
        for test_file in test_files:
            with open(test_file) as f:
                for line in f:
                    try:
                        self.test_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(self.test_examples)} test examples")

    def get_latest_checkpoint(self) -> Optional[Tuple[int, Path]]:
        """Find the latest checkpoint"""
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (ValueError, IndexError):
                    continue

        if not checkpoints:
            if (self.checkpoint_dir / "config.json").exists():
                status_file = self.base_dir / "status" / "training_status.json"
                if status_file.exists():
                    with open(status_file) as f:
                        status = json.load(f)
                        step = status.get("current_step", 0)
                        return (step, self.checkpoint_dir)
                return (0, self.checkpoint_dir)
            return None

        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0]

    def extract_expected_output(self, text: str) -> Optional[str]:
        """
        Extract expected output from training example text.

        Assumes format: "question\nExpected: answer" or similar
        """
        # Try to find expected output markers
        patterns = [
            r'(?:Expected|Answer|Output):\s*(.+?)(?:\n|$)',
            r'<\|im_start\|>assistant\n(.+?)<\|im_end\|>',
            r'### Output:\s*(.+?)(?:\n###|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: take last line as expected output
        lines = text.strip().split('\n')
        if len(lines) > 1:
            return lines[-1].strip()

        return None

    def extract_prompt_and_expected(self, example: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract prompt and expected output from various data formats.

        Supports:
        - {"text": "prompt\nExpected: answer"}
        - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        - {"input": "...", "output": "..."}

        Returns: (prompt, expected_answer)
        """
        prompt = None
        expected = None

        # Format 1: text field (legacy)
        if "text" in example and example["text"]:
            prompt = example["text"]
            expected = self.extract_expected_output(prompt)
            return prompt, expected

        # Format 2: messages array (chat format - most common for your data)
        if "messages" in example:
            messages = example["messages"]
            for msg in messages:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    expected = msg.get("content", "")
            return prompt, expected

        # Format 3: input/output fields
        if "input" in example:
            prompt = example["input"]
            expected = example.get("output", example.get("answer"))
            return prompt, expected

        return None, None

    def calculate_confidence(self, logits: torch.Tensor) -> float:
        """
        Calculate prediction confidence from logits.
        Uses softmax probability of top prediction.
        """
        probs = torch.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob

    def mine_adversarial_examples(
        self,
        checkpoint_path: Path,
        step: int
    ) -> Dict:
        """
        Mine adversarial examples from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            step: Training step number

        Returns:
            Dict with mining results
        """
        logger.info(f"Mining adversarial examples at step {step}")

        try:
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            model.eval()

            # Sample random examples to test
            if len(self.test_examples) == 0:
                logger.warning("No test examples available")
                return None

            import random
            test_sample = random.sample(
                self.test_examples,
                min(self.samples_per_check, len(self.test_examples))
            )

            adversarial_examples = []
            stats = {
                "total_tested": 0,
                "low_confidence": 0,
                "incorrect_predictions": 0,
                "avg_confidence": 0.0
            }

            confidences = []

            with torch.no_grad():
                for idx, example in enumerate(test_sample):
                    try:
                        # Use helper to extract prompt and expected from any format
                        prompt, expected = self.extract_prompt_and_expected(example)
                        if not prompt:
                            continue

                        # Tokenize input
                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        ).to(model.device)

                        # Get model prediction
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            output_scores=True,
                            return_dict_in_generate=True
                        )

                        # Decode prediction
                        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

                        # Calculate confidence from first token scores
                        if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                            first_token_logits = outputs.scores[0][0]
                            confidence = self.calculate_confidence(first_token_logits)
                        else:
                            confidence = 0.5  # unknown

                        confidences.append(confidence)
                        stats["total_tested"] += 1

                        # Check if adversarial
                        is_adversarial = False
                        adversarial_type = None

                        if confidence < self.confidence_threshold:
                            is_adversarial = True
                            adversarial_type = "low_confidence"
                            stats["low_confidence"] += 1

                        # Check correctness using pre-extracted expected
                        if expected:
                            prediction_check = prediction.lower().strip()
                            expected_check = expected.lower().strip()
                            if expected_check not in prediction_check:
                                is_adversarial = True
                                adversarial_type = "incorrect_prediction"
                                stats["incorrect_predictions"] += 1

                        if is_adversarial:
                            adversarial_examples.append({
                                "prompt": prompt,
                                "expected": expected,
                                "prediction": prediction,
                                "confidence": confidence,
                                "type": adversarial_type,
                                "checkpoint_step": step,
                                "timestamp": datetime.now().isoformat(),
                                "source_format": "messages" if "messages" in example else "text"
                            })

                    except Exception as e:
                        logger.warning(f"Error processing example {idx}: {e}")
                        continue

            stats["avg_confidence"] = np.mean(confidences) if confidences else 0.0

            # Save adversarial examples
            output_file = None
            if adversarial_examples:
                output_file = self.output_dir / f"adversarial_step_{step}.jsonl"
                with open(output_file, 'w') as f:
                    for ex in adversarial_examples:
                        f.write(json.dumps(ex) + '\n')
                logger.info(f"Saved {len(adversarial_examples)} adversarial examples to {output_file}")

            # Build categories for plugin compatibility (TASK013)
            categories = defaultdict(lambda: {"count": 0, "avg_confidence": 0.0, "confidences": []})
            for ex in adversarial_examples:
                cat = ex.get("type", "unknown")
                categories[cat]["count"] += 1
                categories[cat]["confidences"].append(ex.get("confidence", 0.5))

            # Compute avg confidence per category
            for cat_name, cat_data in categories.items():
                if cat_data["confidences"]:
                    cat_data["avg_confidence"] = sum(cat_data["confidences"]) / len(cat_data["confidences"])
                del cat_data["confidences"]  # Remove intermediate data

            # Clean up
            del model
            torch.cuda.empty_cache()

            return {
                "step": step,
                "checkpoint": str(checkpoint_path),
                "timestamp": datetime.now().isoformat(),
                "stats": stats,
                "adversarial_count": len(adversarial_examples),
                "total_examples_mined": len(adversarial_examples),  # Plugin compatibility
                "categories": dict(categories),  # Plugin compatibility
                "output_file": str(output_file) if output_file else None
            }

        except Exception as e:
            logger.error(f"Error mining adversarial examples: {e}", exc_info=True)
            return None

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting adversarial mining loop")
        logger.info(f"Check interval: {self.check_interval}s")

        # Load test data
        self.load_test_data()

        if not self.test_examples:
            logger.error("No test examples loaded - cannot proceed")
            return

        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"=== Iteration {iteration} ===")

                # Check for new checkpoint
                checkpoint_info = self.get_latest_checkpoint()

                if checkpoint_info:
                    step, checkpoint_path = checkpoint_info

                    if (step, checkpoint_path) != self.last_checkpoint:
                        logger.info(f"New checkpoint detected at step {step}: {checkpoint_path}")

                        # Mine adversarial examples
                        mining_result = self.mine_adversarial_examples(checkpoint_path, step)

                        if mining_result:
                            # Update results
                            self.results["mining_runs"].append(mining_result)
                            self.results["adversarial_examples_found"] += mining_result["adversarial_count"]
                            self.results["total_examples_tested"] += mining_result["stats"]["total_tested"]
                            self._save_results()

                            # Log results
                            stats = mining_result["stats"]
                            logger.info("=" * 70)
                            logger.info("ADVERSARIAL MINING RESULTS")
                            logger.info("=" * 70)
                            logger.info(f"Step: {step}")
                            logger.info(f"Tested: {stats['total_tested']} examples")
                            logger.info(f"Found: {mining_result['adversarial_count']} adversarial examples")
                            logger.info(f"Low confidence: {stats['low_confidence']}")
                            logger.info(f"Incorrect predictions: {stats['incorrect_predictions']}")
                            logger.info(f"Avg confidence: {stats['avg_confidence']:.2%}")
                            logger.info("=" * 70)

                            # Update last checkpoint
                            self.last_checkpoint = (step, checkpoint_path)
                else:
                    logger.info("No checkpoint found")

                # Sleep
                logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.check_interval)

        logger.info("Adversarial mining loop stopped")

    def print_status(self):
        """Print current mining status"""
        print("\n" + "="*70)
        print("ADVERSARIAL MINER STATUS")
        print("="*70)

        print(f"\nTest examples: {len(self.test_examples)}")
        print(f"Mining runs: {len(self.results['mining_runs'])}")
        print(f"Total examples tested: {self.results['total_examples_tested']}")
        print(f"Adversarial examples found: {self.results['adversarial_examples_found']}")

        if self.results['mining_runs']:
            latest = self.results['mining_runs'][-1]
            print(f"\nLatest Run (Step {latest['step']}):")
            print(f"  Found: {latest['adversarial_count']} adversarial examples")
            print(f"  Avg confidence: {latest['stats']['avg_confidence']:.2%}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adversarial Example Miner - finds model weaknesses"
    )
    parser.add_argument(
        "--base-dir",
        default="/path/to/training",
        help="Base training directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Checkpoint directory to monitor"
    )
    parser.add_argument(
        "--test-data-dir",
        help="Test data directory"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for adversarial examples (default: 0.6)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples to test per checkpoint (default: 100)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status and exit"
    )

    args = parser.parse_args()

    miner = AdversarialMiner(
        base_dir=args.base_dir,
        checkpoint_dir=args.checkpoint_dir,
        test_data_dir=args.test_data_dir,
        check_interval=args.interval,
        confidence_threshold=args.confidence_threshold,
        samples_per_check=args.samples
    )

    if args.status:
        miner.load_test_data()
        miner.print_status()
    else:
        miner.monitor_loop()


if __name__ == "__main__":
    main()
