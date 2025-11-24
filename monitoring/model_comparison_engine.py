#!/usr/bin/env python3
"""
Model Comparison Engine
=======================

Automatically A/B tests different checkpoints to find the best one.

GPU Usage: ~3% (750MB VRAM for dual model loading)
ROI: ⭐⭐⭐ (Ensures best checkpoint is selected)

Features:
- Tracks all checkpoints as they're created
- Periodically compares recent checkpoints
- Ranks by multiple metrics (loss, accuracy, speed)
- Identifies best checkpoint for deployment
- Generates comparison reports

Use cases:
- Find best checkpoint before model consolidation
- Compare training runs with different hyperparameters
- Validate checkpoint quality before deployment
"""

import torch
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ModelComparisonEngine - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparisonEngine:
    """
    Continuously compares checkpoints to rank and select the best one.
    """

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        checkpoint_dir: str = None,
        test_data_dir: str = None,
        comparison_interval: int = 600,  # 10 minutes
        test_samples: int = 100,
        min_checkpoints_for_comparison: int = 3
    ):
        """
        Initialize model comparison engine.

        Args:
            base_dir: Base training directory
            checkpoint_dir: Directory to monitor for checkpoints
            test_data_dir: Test data directory
            comparison_interval: Seconds between comparisons
            test_samples: Samples to test per model
            min_checkpoints_for_comparison: Min checkpoints needed
        """
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.base_dir / "models" / "current_model")
        self.test_data_dir = Path(test_data_dir or self.base_dir / "data" / "validation")
        self.comparison_interval = comparison_interval
        self.test_samples = test_samples
        self.min_checkpoints = min_checkpoints_for_comparison

        # Results storage
        self.results_file = self.base_dir / "status" / "model_comparisons.json"
        self.results = self._load_results()

        # Checkpoint tracking
        self.evaluated_checkpoints = {}
        self.test_examples = []

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Model Comparison Engine initialized")
        logger.info(f"Monitoring: {self.checkpoint_dir}")
        logger.info(f"Comparison interval: {self.comparison_interval}s")

    def _load_results(self) -> Dict:
        """Load previous results"""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {
            "comparisons": [],
            "rankings": [],
            "best_checkpoint": None,
            "last_updated": None
        }

    def _save_results(self):
        """Save results"""
        self.results["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load_test_data(self):
        """Load test examples"""
        test_files = list(self.test_data_dir.glob("*.jsonl"))

        if not test_files:
            logger.warning(f"No test files in {self.test_data_dir}")
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

    def discover_checkpoints(self) -> List[Tuple[int, Path]]:
        """Find all available checkpoints"""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except (ValueError, IndexError):
                    continue

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def evaluate_checkpoint(self, checkpoint_path: Path, step: int) -> Dict:
        """
        Comprehensive evaluation of a checkpoint.

        Returns metrics for ranking.
        """
        logger.info(f"Evaluating checkpoint at step {step}")

        try:
            start_time = time.time()

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

            load_time = time.time() - start_time

            # Sample test data
            import random
            test_sample = random.sample(
                self.test_examples,
                min(self.test_samples, len(self.test_examples))
            )

            # Run evaluation
            losses = []
            correct = 0
            total = 0
            inference_times = []

            with torch.no_grad():
                for example in test_sample:
                    try:
                        text = example.get("text", "")
                        if not text:
                            continue

                        inputs = tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        ).to(model.device)

                        # Time inference
                        inf_start = time.time()
                        outputs = model(**inputs, labels=inputs.input_ids)
                        inference_times.append(time.time() - inf_start)

                        losses.append(outputs.loss.item())

                        # Accuracy
                        logits = outputs.logits
                        predictions = torch.argmax(logits[:, :-1, :], dim=-1)
                        targets = inputs.input_ids[:, 1:]
                        correct += (predictions == targets).sum().item()
                        total += targets.numel()

                    except Exception as e:
                        logger.warning(f"Error processing example: {e}")
                        continue

            # Clean up
            del model
            torch.cuda.empty_cache()

            # Calculate metrics
            avg_loss = np.mean(losses) if losses else float('inf')
            avg_inference_time = np.mean(inference_times) if inference_times else 0

            metrics = {
                "step": step,
                "checkpoint": str(checkpoint_path),
                "avg_loss": avg_loss,
                "perplexity": np.exp(avg_loss) if avg_loss < 10 else float('inf'),
                "accuracy": correct / total if total > 0 else 0.0,
                "avg_inference_time_ms": avg_inference_time * 1000,
                "throughput_samples_per_sec": 1 / avg_inference_time if avg_inference_time > 0 else 0,
                "load_time_sec": load_time,
                "num_samples": len(test_sample),
                "evaluated_at": datetime.now().isoformat()
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating checkpoint: {e}", exc_info=True)
            return None

    def calculate_score(self, metrics: Dict) -> float:
        """
        Calculate composite score for ranking.

        Weighs multiple factors:
        - Loss (40%): Lower is better
        - Accuracy (40%): Higher is better
        - Speed (20%): Faster is better
        """
        # Normalize metrics (0-1 scale, higher is better)

        # Loss: inverse and normalize
        loss_score = 1 / (1 + metrics["avg_loss"]) if metrics["avg_loss"] < float('inf') else 0

        # Accuracy: already 0-1
        accuracy_score = metrics["accuracy"]

        # Speed: normalize by typical inference time (assume 0.1s baseline)
        speed_score = min(1.0, 0.1 / metrics["avg_inference_time_ms"] * 1000) if metrics["avg_inference_time_ms"] > 0 else 0

        # Weighted average
        composite_score = (
            0.4 * loss_score +
            0.4 * accuracy_score +
            0.2 * speed_score
        )

        return composite_score

    def compare_checkpoints(self) -> Optional[Dict]:
        """
        Compare all evaluated checkpoints and generate ranking.

        Returns:
            Comparison report with rankings
        """
        if len(self.evaluated_checkpoints) < self.min_checkpoints:
            logger.info(f"Not enough checkpoints for comparison ({len(self.evaluated_checkpoints)}/{self.min_checkpoints})")
            return None

        logger.info(f"Comparing {len(self.evaluated_checkpoints)} checkpoints")

        # Calculate scores
        scored_checkpoints = []
        for step, metrics in self.evaluated_checkpoints.items():
            score = self.calculate_score(metrics)
            scored_checkpoints.append({
                "step": step,
                "score": score,
                "metrics": metrics
            })

        # Sort by score (descending)
        scored_checkpoints.sort(key=lambda x: x["score"], reverse=True)

        # Generate ranking
        ranking = []
        for rank, item in enumerate(scored_checkpoints, 1):
            ranking.append({
                "rank": rank,
                "step": item["step"],
                "score": item["score"],
                "loss": item["metrics"]["avg_loss"],
                "accuracy": item["metrics"]["accuracy"],
                "inference_time_ms": item["metrics"]["avg_inference_time_ms"]
            })

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "num_checkpoints": len(scored_checkpoints),
            "ranking": ranking,
            "best_checkpoint": {
                "step": ranking[0]["step"],
                "score": ranking[0]["score"],
                "metrics": scored_checkpoints[0]["metrics"]
            }
        }

        return comparison

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting model comparison engine")
        logger.info(f"Comparison interval: {self.comparison_interval}s")

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

                # Discover checkpoints
                checkpoints = self.discover_checkpoints()
                logger.info(f"Found {len(checkpoints)} checkpoints")

                # Evaluate new checkpoints
                for step, checkpoint_path in checkpoints:
                    if step not in self.evaluated_checkpoints:
                        logger.info(f"New checkpoint detected: step {step}")
                        metrics = self.evaluate_checkpoint(checkpoint_path, step)
                        if metrics:
                            self.evaluated_checkpoints[step] = metrics
                            logger.info(f"Evaluated step {step}: loss={metrics['avg_loss']:.4f}, acc={metrics['accuracy']:.2%}")

                # Run comparison
                comparison = self.compare_checkpoints()

                if comparison:
                    self.results["comparisons"].append(comparison)
                    self.results["best_checkpoint"] = comparison["best_checkpoint"]
                    self._save_results()

                    # Log results
                    logger.info("=" * 70)
                    logger.info("CHECKPOINT RANKING")
                    logger.info("=" * 70)
                    for item in comparison["ranking"][:5]:  # Top 5
                        logger.info(
                            f"#{item['rank']}: Step {item['step']} | "
                            f"Score: {item['score']:.3f} | "
                            f"Loss: {item['loss']:.4f} | "
                            f"Acc: {item['accuracy']:.2%}"
                        )
                    logger.info("=" * 70)
                    logger.info(f"Best checkpoint: Step {comparison['best_checkpoint']['step']}")
                    logger.info("=" * 70)

                # Sleep
                logger.info(f"Sleeping for {self.comparison_interval}s...")
                time.sleep(self.comparison_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.comparison_interval)

        logger.info("Model comparison engine stopped")

    def print_status(self):
        """Print current status"""
        print("\n" + "="*70)
        print("MODEL COMPARISON ENGINE STATUS")
        print("="*70)

        print(f"\nEvaluated checkpoints: {len(self.evaluated_checkpoints)}")
        print(f"Comparisons run: {len(self.results['comparisons'])}")

        if self.results.get("best_checkpoint"):
            best = self.results["best_checkpoint"]
            print(f"\nBest Checkpoint:")
            print(f"  Step: {best['step']}")
            print(f"  Score: {best['score']:.3f}")
            print(f"  Loss: {best['metrics']['avg_loss']:.4f}")
            print(f"  Accuracy: {best['metrics']['accuracy']:.2%}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Model Comparison Engine - A/B tests checkpoints"
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
        default=600,
        help="Comparison interval in seconds (default: 600)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Test samples per checkpoint (default: 100)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status and exit"
    )

    args = parser.parse_args()

    engine = ModelComparisonEngine(
        base_dir=args.base_dir,
        checkpoint_dir=args.checkpoint_dir,
        test_data_dir=args.test_data_dir,
        comparison_interval=args.interval,
        test_samples=args.samples
    )

    if args.status:
        engine.load_test_data()
        engine.print_status()
    else:
        engine.monitor_loop()


if __name__ == "__main__":
    main()
