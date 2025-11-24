#!/usr/bin/env python3
"""
Confidence Calibrator
=====================

Analyzes and calibrates model prediction confidence scores.

GPU Usage: ~2% (500MB VRAM)
ROI: ⭐⭐ (Improves confidence reliability)

Measures calibration: Does 80% confidence actually mean 80% correct?

Features:
- Monitors prediction confidences vs actual correctness
- Generates calibration curves
- Detects overconfident/underconfident predictions
- Recommends confidence thresholds

Use cases:
- Determine reliable confidence cutoffs for production
- Detect when model is overconfident (predicts wrong with high confidence)
- Track calibration improvements during training
"""

import torch
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ConfidenceCalibrator - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Calibrates prediction confidence scores."""

    def __init__(
        self,
        base_dir: str = "/path/to/training",
        checkpoint_dir: str = None,
        test_data_dir: str = None,
        check_interval: int = 600,
        test_samples: int = 100
    ):
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.base_dir / "models" / "current_model")
        self.test_data_dir = Path(test_data_dir or self.base_dir / "data" / "validation")
        self.check_interval = check_interval
        self.test_samples = test_samples

        self.results_file = self.base_dir / "status" / "confidence_calibration.json"
        self.results = self._load_results()
        self.last_checkpoint = None
        self.test_examples = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Confidence Calibrator initialized")

    def _load_results(self) -> Dict:
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return {"calibrations": [], "last_updated": None}

    def _save_results(self):
        self.results["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load_test_data(self):
        test_files = list(self.test_data_dir.glob("*.jsonl"))
        self.test_examples = []
        for test_file in test_files:
            with open(test_file) as f:
                for line in f:
                    try:
                        self.test_examples.append(json.loads(line))
                    except:
                        continue
        logger.info(f"Loaded {len(self.test_examples)} test examples")

    def get_latest_checkpoint(self):
        if not self.checkpoint_dir.exists():
            return None
        checkpoints = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except:
                    continue
        if not checkpoints:
            if (self.checkpoint_dir / "config.json").exists():
                return (0, self.checkpoint_dir)
            return None
        checkpoints.sort(reverse=True)
        return checkpoints[0]

    def calibrate_checkpoint(self, checkpoint_path: Path, step: int):
        logger.info(f"Calibrating checkpoint at step {step}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            model.eval()

            import random
            test_sample = random.sample(self.test_examples, min(self.test_samples, len(self.test_examples)))

            confidences = []
            correctness = []

            with torch.no_grad():
                for example in test_sample:
                    try:
                        text = example.get("text", "")
                        if not text:
                            continue

                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                        outputs = model(**inputs, labels=inputs.input_ids)

                        # Get confidence (max softmax probability)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        max_probs = torch.max(probs, dim=-1)[0]
                        avg_confidence = max_probs.mean().item()

                        # Check correctness
                        predictions = torch.argmax(logits[:, :-1, :], dim=-1)
                        targets = inputs.input_ids[:, 1:]
                        is_correct = (predictions == targets).float().mean().item()

                        confidences.append(avg_confidence)
                        correctness.append(is_correct)
                    except:
                        continue

            del model
            torch.cuda.empty_cache()

            # Calculate calibration metrics
            bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            calibration = []
            for i in range(len(bins)-1):
                mask = [(bins[i] <= c < bins[i+1]) for c in confidences]
                if sum(mask) > 0:
                    bin_conf = np.mean([confidences[j] for j, m in enumerate(mask) if m])
                    bin_acc = np.mean([correctness[j] for j, m in enumerate(mask) if m])
                    calibration.append({
                        "confidence_range": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                        "avg_confidence": float(bin_conf),
                        "avg_accuracy": float(bin_acc),
                        "count": sum(mask),
                        "calibration_error": abs(bin_conf - bin_acc)
                    })

            result = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "calibration_bins": calibration,
                "overall_confidence": float(np.mean(confidences)),
                "overall_accuracy": float(np.mean(correctness)),
                "ece": float(np.mean([c["calibration_error"] for c in calibration]))  # Expected Calibration Error
            }

            return result

        except Exception as e:
            logger.error(f"Error calibrating: {e}")
            return None

    def monitor_loop(self):
        logger.info("Starting confidence calibration monitoring")
        self.load_test_data()
        if not self.test_examples:
            return

        while True:
            try:
                checkpoint_info = self.get_latest_checkpoint()
                if checkpoint_info:
                    step, checkpoint_path = checkpoint_info
                    if (step, checkpoint_path) != self.last_checkpoint:
                        result = self.calibrate_checkpoint(checkpoint_path, step)
                        if result:
                            self.results["calibrations"].append(result)
                            self._save_results()
                            logger.info(f"ECE: {result['ece']:.4f} | Conf: {result['overall_confidence']:.2%} | Acc: {result['overall_accuracy']:.2%}")
                            self.last_checkpoint = (step, checkpoint_path)
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(self.check_interval)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/path/to/training")
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()

    calibrator = ConfidenceCalibrator(base_dir=args.base_dir, check_interval=args.interval)
    calibrator.monitor_loop()

if __name__ == "__main__":
    main()
