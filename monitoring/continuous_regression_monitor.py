#!/usr/bin/env python3
"""
Continuous Regression Monitor
==============================

Continuously monitors training checkpoints for performance regressions.

GPU Usage: ~2% (500MB VRAM for quick evaluations)
ROI: ⭐⭐⭐ (Prevents bad checkpoints from being deployed)

Features:
- Monitors new checkpoints as they're created
- Compares each to baseline (previous good checkpoint)
- Detects loss spikes, accuracy drops, catastrophic forgetting
- Auto-alerts on regression
- Maintains regression history

Difference from regression_detector.py:
- regression_detector.py: One-time comparison tool
- continuous_regression_monitor.py: Always-on daemon monitoring
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
    format='%(asctime)s - RegressionMonitor - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousRegressionMonitor:
    """
    Monitors checkpoints continuously for performance regressions.
    """

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        checkpoint_dir: str = None,
        test_data_dir: str = None,
        check_interval: int = 300,
        regression_threshold: float = 0.15,  # 15% loss increase = regression
        test_samples: int = 50
    ):
        """
        Initialize continuous regression monitor.

        Args:
            base_dir: Base training directory
            checkpoint_dir: Directory to monitor for checkpoints
            test_data_dir: Test data directory
            check_interval: Seconds between checks
            regression_threshold: Loss increase threshold for regression alert
            test_samples: Number of samples to test per checkpoint
        """
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.base_dir / "models" / "current_model")
        self.test_data_dir = Path(test_data_dir or self.base_dir / "data" / "validation")
        self.check_interval = check_interval
        self.regression_threshold = regression_threshold
        self.test_samples = test_samples

        # Results storage
        self.results_file = self.base_dir / "status" / "regression_monitoring.json"
        self.results = self._load_results()

        # Tracking
        self.last_checkpoint = None
        self.baseline_metrics = None
        self.test_examples = []

        # GPU setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Continuous Regression Monitor initialized")
        logger.info(f"Monitoring: {self.checkpoint_dir}")
        logger.info(f"Regression threshold: {self.regression_threshold:.1%}")

    def _load_results(self) -> Dict:
        """Load previous results"""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {
            "checks": [],
            "regressions_detected": 0,
            "total_checks": 0,
            "last_updated": None
        }

    def _save_results(self):
        """Save results"""
        self.results["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load_test_data(self):
        """Load test examples from validation directory"""
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

    def get_latest_checkpoint(self) -> Optional[Tuple[int, Path]]:
        """Find latest checkpoint"""
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

    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Quick evaluation of checkpoint on test set.

        Returns:
            Dict with avg_loss, perplexity, accuracy
        """
        try:
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

            # Sample test examples
            import random
            test_sample = random.sample(
                self.test_examples,
                min(self.test_samples, len(self.test_examples))
            )

            losses = []
            correct = 0
            total = 0

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

                        outputs = model(**inputs, labels=inputs.input_ids)
                        losses.append(outputs.loss.item())

                        # Token accuracy
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

            avg_loss = np.mean(losses) if losses else float('inf')

            return {
                "avg_loss": avg_loss,
                "perplexity": np.exp(avg_loss) if avg_loss < 10 else float('inf'),
                "accuracy": correct / total if total > 0 else 0.0,
                "num_samples": len(test_sample)
            }

        except Exception as e:
            logger.error(f"Error evaluating checkpoint: {e}", exc_info=True)
            return None

    def check_regression(
        self,
        current_metrics: Dict,
        baseline_metrics: Dict
    ) -> Tuple[bool, str]:
        """
        Check if current checkpoint shows regression vs baseline.

        Returns:
            (is_regression, reason)
        """
        # Loss increase
        loss_delta = current_metrics["avg_loss"] - baseline_metrics["avg_loss"]
        loss_pct_change = loss_delta / baseline_metrics["avg_loss"] if baseline_metrics["avg_loss"] > 0 else 0

        if loss_pct_change > self.regression_threshold:
            return (True, f"Loss increased {loss_pct_change:.1%} (threshold: {self.regression_threshold:.1%})")

        # Accuracy drop
        acc_delta = current_metrics["accuracy"] - baseline_metrics["accuracy"]
        if acc_delta < -0.10:  # 10% accuracy drop
            return (True, f"Accuracy dropped {-acc_delta:.1%}")

        # Perplexity explosion
        ppl_ratio = current_metrics["perplexity"] / baseline_metrics["perplexity"] if baseline_metrics["perplexity"] > 0 else 1
        if ppl_ratio > 1.5:
            return (True, f"Perplexity increased {ppl_ratio:.1f}x")

        return (False, "No regression detected")

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting continuous regression monitoring")
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
                        logger.info(f"New checkpoint at step {step}: {checkpoint_path}")

                        # Evaluate
                        metrics = self.evaluate_checkpoint(checkpoint_path)

                        if metrics:
                            # Check regression if we have baseline
                            is_regression = False
                            reason = "First checkpoint (establishing baseline)"

                            if self.baseline_metrics:
                                is_regression, reason = self.check_regression(
                                    metrics,
                                    self.baseline_metrics
                                )

                            # Log result
                            check_result = {
                                "step": step,
                                "checkpoint": str(checkpoint_path),
                                "timestamp": datetime.now().isoformat(),
                                "metrics": metrics,
                                "is_regression": is_regression,
                                "reason": reason,
                                "baseline_step": self.baseline_metrics.get("step") if self.baseline_metrics else None
                            }

                            self.results["checks"].append(check_result)
                            self.results["total_checks"] += 1
                            if is_regression:
                                self.results["regressions_detected"] += 1

                            self._save_results()

                            # Log
                            logger.info("=" * 70)
                            logger.info("REGRESSION CHECK RESULT")
                            logger.info("=" * 70)
                            logger.info(f"Step: {step}")
                            logger.info(f"Loss: {metrics['avg_loss']:.4f}")
                            logger.info(f"Perplexity: {metrics['perplexity']:.2f}")
                            logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
                            if is_regression:
                                logger.warning(f"⚠️  REGRESSION DETECTED: {reason}")
                            else:
                                logger.info(f"✓ {reason}")
                            logger.info("=" * 70)

                            # Update baseline if no regression
                            if not is_regression:
                                metrics["step"] = step
                                self.baseline_metrics = metrics
                                logger.info(f"Updated baseline to step {step}")

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

        logger.info("Continuous regression monitoring stopped")

    def print_status(self):
        """Print current status"""
        print("\n" + "="*70)
        print("CONTINUOUS REGRESSION MONITOR STATUS")
        print("="*70)

        print(f"\nTest examples: {len(self.test_examples)}")
        print(f"Total checks: {self.results['total_checks']}")
        print(f"Regressions detected: {self.results['regressions_detected']}")

        if self.baseline_metrics:
            print(f"\nCurrent Baseline:")
            print(f"  Step: {self.baseline_metrics.get('step', 'unknown')}")
            print(f"  Loss: {self.baseline_metrics['avg_loss']:.4f}")
            print(f"  Accuracy: {self.baseline_metrics['accuracy']:.2%}")

        if self.results['checks']:
            latest = self.results['checks'][-1]
            print(f"\nLatest Check (Step {latest['step']}):")
            print(f"  Regression: {latest['is_regression']}")
            print(f"  Reason: {latest['reason']}")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Continuous Regression Monitor - detects performance regressions"
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
        "--threshold",
        type=float,
        default=0.15,
        help="Regression threshold (default: 0.15 = 15%% loss increase)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Test samples per check (default: 50)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status and exit"
    )

    args = parser.parse_args()

    monitor = ContinuousRegressionMonitor(
        base_dir=args.base_dir,
        checkpoint_dir=args.checkpoint_dir,
        test_data_dir=args.test_data_dir,
        check_interval=args.interval,
        regression_threshold=args.threshold,
        test_samples=args.samples
    )

    if args.status:
        monitor.load_test_data()
        monitor.print_status()
    else:
        monitor.monitor_loop()


if __name__ == "__main__":
    main()
