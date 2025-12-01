#!/usr/bin/env python3
"""
Test script for Fortune Teller surprise-weighted training.

This script demonstrates and compares:
1. Standard training (all tokens weighted equally)
2. Fortune Teller training (tokens weighted by surprise)

Usage:
    # Test on small synthetic dataset
    python3 scripts/test_fortune_teller.py --quick

    # Full comparison on real data
    python3 scripts/test_fortune_teller.py --dataset data/train.jsonl

    # Test specific surprise metric
    python3 scripts/test_fortune_teller.py --surprise-metric entropy

    # Visualize surprise metrics
    python3 scripts/test_fortune_teller.py --visualize results/fortune_teller_history.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from trainer.losses import FortuneTellerLoss, SurpriseMetric, FortuneTellerTracker


def create_synthetic_data(vocab_size: int = 1000, seq_len: int = 50, num_samples: int = 100) -> List[Dict]:
    """
    Create synthetic training data for testing.

    Mix of:
    - Easy patterns (repeated sequences)
    - Hard patterns (random)
    """
    import random

    data = []

    # Easy patterns (first half)
    for i in range(num_samples // 2):
        # Repeated pattern: [1, 2, 3, 1, 2, 3, ...]
        tokens = [1 + (j % 3) for j in range(seq_len)]
        data.append({"input_ids": tokens})

    # Hard patterns (second half)
    for i in range(num_samples // 2):
        # Random tokens
        tokens = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]
        data.append({"input_ids": tokens})

    return data


def test_surprise_metrics():
    """Test all surprise metrics on simple examples."""
    print("=" * 80)
    print("Testing Surprise Metrics")
    print("=" * 80)

    batch_size = 2
    seq_len = 10
    vocab_size = 100

    # Create sample data
    # Batch 0: Model is confident and correct
    # Batch 1: Model is uncertain
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Make batch 0 very confident in token 5
    logits[0, :, 5] = 10.0

    # Make batch 1 uniform (uncertain)
    logits[1, :, :] = 0.0

    labels = torch.full((batch_size, seq_len), 5, dtype=torch.long)

    # Test each metric
    for metric in [SurpriseMetric.ENTROPY, SurpriseMetric.CONFIDENCE, SurpriseMetric.PERPLEXITY, SurpriseMetric.MARGIN]:
        print(f"\nMetric: {metric}")
        print("-" * 40)

        loss_fn = FortuneTellerLoss(
            surprise_metric=metric,
            min_surprise=0.1,
            normalize_batch=False,
        )

        loss, details = loss_fn(logits, labels, return_details=True)

        print(f"Loss: {loss.item():.4f}")
        print(f"Avg surprise: {details['avg_surprise']:.4f}")
        print(f"Surprise std: {details['surprise_std']:.4f}")
        print(f"Max surprise: {details['max_surprise']:.4f}")
        print(f"Min surprise: {details['min_surprise']:.4f}")


def compare_training_modes(
    model,
    tokenizer,
    train_data,
    val_data,
    output_dir: str,
    surprise_metric: str = "entropy",
):
    """
    Compare standard training vs Fortune Teller training.

    Returns dict with metrics from both runs.
    """
    print("=" * 80)
    print("Comparing Training Modes")
    print("=" * 80)

    results = {}

    # 1. Standard training
    print("\n1. STANDARD TRAINING (baseline)")
    print("-" * 40)

    standard_output = Path(output_dir) / "standard"
    standard_output.mkdir(parents=True, exist_ok=True)

    # TODO: Run actual training here
    # For now, this is a template showing the structure

    print("Standard training would run here...")
    results["standard"] = {
        "final_loss": 0.0,  # placeholder
        "runtime_sec": 0.0,
    }

    # 2. Fortune Teller training
    print("\n2. FORTUNE TELLER TRAINING (surprise-weighted)")
    print("-" * 40)

    ft_output = Path(output_dir) / f"fortune_teller_{surprise_metric}"
    ft_output.mkdir(parents=True, exist_ok=True)

    print(f"Surprise metric: {surprise_metric}")
    print("Fortune Teller training would run here...")

    results["fortune_teller"] = {
        "final_loss": 0.0,  # placeholder
        "runtime_sec": 0.0,
        "avg_surprise": 0.0,
        "surprise_metric": surprise_metric,
    }

    return results


def visualize_surprise_history(history_path: str):
    """
    Visualize Fortune Teller metrics over training.

    Generates:
    - Surprise over time
    - Surprise distribution
    - Weight statistics
    """
    print("=" * 80)
    print(f"Visualizing Fortune Teller History: {history_path}")
    print("=" * 80)

    tracker = FortuneTellerTracker()
    tracker.load(history_path)

    if not tracker.history:
        print("No history found!")
        return

    # Print summary statistics
    stats = tracker.get_stats()
    print("\nSummary Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    # Show surprise evolution
    print("\nSurprise Evolution (every 100 steps):")
    print("-" * 40)
    print(f"{'Step':<10} {'Avg Surprise':<15} {'Surprise Std':<15} {'Avg Weight':<15}")
    print("-" * 60)

    window_size = 100
    for i, record in enumerate(tracker.history):
        if i % window_size == 0 or i == len(tracker.history) - 1:
            print(
                f"{record['step']:<10} "
                f"{record['avg_surprise']:<15.4f} "
                f"{record['surprise_std']:<15.4f} "
                f"{record['avg_weight']:<15.4f}"
            )

    # Optional: matplotlib visualization
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        steps = [r['step'] for r in tracker.history]
        surprises = [r['avg_surprise'] for r in tracker.history]
        weights = [r['avg_weight'] for r in tracker.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot surprise over time
        ax1.plot(steps, surprises, label='Avg Surprise', alpha=0.7)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Average Surprise')
        ax1.set_title('Surprise Over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot weights over time
        ax2.plot(steps, weights, label='Avg Weight', color='orange', alpha=0.7)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Average Weight')
        ax2.set_title('Surprise Weights Over Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = Path(history_path).parent / "fortune_teller_visualization.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nVisualization saved to: {output_path}")

        plt.show()

    except ImportError:
        print("\nNote: Install matplotlib for visualizations (pip install matplotlib)")


def main():
    parser = argparse.ArgumentParser(description="Test Fortune Teller surprise-weighted training")

    # Test modes
    parser.add_argument("--quick", action="store_true", help="Quick test with synthetic data")
    parser.add_argument("--test-metrics", action="store_true", help="Test all surprise metrics")
    parser.add_argument("--visualize", type=str, help="Visualize history from JSON file")

    # Training comparison
    parser.add_argument("--dataset", type=str, help="Dataset path for comparison")
    parser.add_argument("--model", type=str, default="models/current_model", help="Model path")
    parser.add_argument("--output-dir", type=str, default="results/fortune_teller_test", help="Output directory")

    # Fortune Teller config
    parser.add_argument(
        "--surprise-metric",
        type=str,
        choices=["entropy", "confidence", "perplexity", "margin"],
        default="entropy",
        help="Surprise metric to use"
    )
    parser.add_argument("--min-surprise", type=float, default=0.1, help="Minimum surprise weight")
    parser.add_argument("--max-surprise", type=float, default=10.0, help="Maximum surprise weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Surprise temperature")

    args = parser.parse_args()

    # Execute based on mode
    if args.test_metrics:
        test_surprise_metrics()

    elif args.visualize:
        visualize_surprise_history(args.visualize)

    elif args.quick:
        print("Quick test mode - testing surprise metrics on synthetic data")
        test_surprise_metrics()

    elif args.dataset:
        print("Full comparison mode")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print(f"Surprise metric: {args.surprise_metric}")
        print("\nNote: Full training comparison requires integration with your training pipeline")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
