#!/usr/bin/env python3
"""
Quick Accuracy Checker

Calculates accuracy from the training status JSON.
Shows how well the model is doing on evaluations.

Usage:
    python3 check_accuracy.py
    python3 check_accuracy.py --detailed
"""

import json
import argparse
from pathlib import Path

def calculate_accuracy(status_file: str = "status/training_status.json", detailed: bool = False):
    """Calculate accuracy from status file."""

    if not Path(status_file).exists():
        print(f"❌ Status file not found: {status_file}")
        return

    with open(status_file) as f:
        status = json.load(f)

    # Get evaluation samples
    eval_samples = status.get('eval_samples', [])
    recent_examples = status.get('recent_examples', [])

    if not eval_samples and not recent_examples:
        print("⚠️  No evaluation data available yet")
        print(f"   Current step: {status.get('current_step', 0)}")
        print(f"   Next eval at: {status.get('eval_steps', 10)} steps")
        return

    print("=" * 80)
    print("TRAINING ACCURACY REPORT")
    print("=" * 80)

    # Overall status
    print(f"\n📊 Training Status:")
    print(f"   Step: {status.get('current_step', 0):,} / {status.get('total_steps', 0):,}")
    print(f"   Loss: {status.get('loss', 0):.4f}")
    print(f"   File: {status.get('current_file', 'N/A')}")

    # Calculate overall accuracy
    if eval_samples:
        total = len(eval_samples)
        correct = sum(1 for ex in eval_samples if ex.get('answer_matches', False))
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\n✅ Overall Accuracy:")
        print(f"   Correct: {correct} / {total}")
        print(f"   Accuracy: {accuracy:.1f}%")

    # Recent examples
    if recent_examples:
        recent_total = len(recent_examples)
        recent_correct = sum(1 for ex in recent_examples if ex.get('matches', False))
        recent_accuracy = (recent_correct / recent_total * 100) if recent_total > 0 else 0

        print(f"\n📈 Recent Performance ({recent_total} examples):")
        print(f"   Correct: {recent_correct} / {recent_total}")
        print(f"   Accuracy: {recent_accuracy:.1f}%")

    # Loss statistics
    if eval_samples:
        losses = [ex.get('loss', 0) for ex in eval_samples if ex.get('loss') is not None]
        if losses:
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)

            print(f"\n📉 Loss Statistics:")
            print(f"   Average: {avg_loss:.4f}")
            print(f"   Best (min): {min_loss:.4f}")
            print(f"   Worst (max): {max_loss:.4f}")

    # Detailed view
    if detailed and recent_examples:
        print(f"\n🔍 Recent Examples (last {min(5, len(recent_examples))}):")
        print("-" * 80)

        for i, ex in enumerate(recent_examples[-5:], 1):
            step = ex.get('step', 'N/A')
            matches = ex.get('matches', False)
            loss = ex.get('loss', 0)
            status_icon = "✅" if matches else "❌"

            print(f"\n{i}. Step {step}: {status_icon}")
            print(f"   Loss: {loss:.4f}")

            if 'prompt' in ex:
                prompt = ex['prompt'][:100] + "..." if len(ex.get('prompt', '')) > 100 else ex.get('prompt', '')
                print(f"   Prompt: {prompt}")

            if 'model_output' in ex:
                output = ex['model_output'][:100] + "..." if len(ex.get('model_output', '')) > 100 else ex.get('model_output', '')
                print(f"   Output: {output}")

    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Check training accuracy")
    parser.add_argument("--status-file", default="status/training_status.json",
                       help="Path to training status JSON")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed recent examples")

    args = parser.parse_args()

    calculate_accuracy(args.status_file, args.detailed)

if __name__ == "__main__":
    main()
