#!/usr/bin/env python3
"""
CLI Wrapper for TrainerEngine

Demonstrates how to use the new API-driven training system.

Usage:
    python3 -m trainer.cli_main --dataset data/train.jsonl --model qwen3_0.6b --output outputs/run_001

This is a demonstration of the clean API architecture. For production training,
continue using core/train.py until full integration is complete.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.config.schema import create_default_config
from trainer.config.loader import ConfigLoader
from trainer.core.engine import TrainerEngine
from trainer.monitoring.status_writer import TrainingStatusWriter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="API-driven training with TrainerEngine (DEMO)"
    )

    # Required arguments
    parser.add_argument("--dataset", required=True, help="Path to training dataset (JSONL)")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")

    # Optional arguments
    parser.add_argument("--config", help="Base config JSON file")
    parser.add_argument("--profile", default="emoji_think", help="Data profile to use")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("🎯 TRAINER CLI - API-Driven Training (DEMONSTRATION)")
    print("=" * 80)
    print(f"Profile: {args.profile}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 80 + "\n")

    # Create configuration using new system
    print("📋 Creating configuration...")

    # For demonstration, create a minimal config
    # In production, this would use ConfigLoader.from_args_and_json()
    config = create_default_config(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        base_model="Qwen/Qwen3-0.6B",  # Would be looked up
        model_architecture="Qwen3ForCausalLM",
        max_context_length=4096,
        vocab_size=151936
    )

    # Override with CLI args if provided
    if args.batch_size:
        config.hyperparams.batch_size = args.batch_size
    if args.learning_rate:
        config.hyperparams.learning_rate = args.learning_rate
    if args.epochs:
        config.hyperparams.num_epochs = args.epochs

    config.profile.name = args.profile

    print(f"   ✓ Config created")
    print(f"   Batch size: {config.hyperparams.batch_size}")
    print(f"   Learning rate: {config.hyperparams.learning_rate}")
    print(f"   Profile: {config.profile.name}\n")

    # Create status writer
    print("📊 Creating status writer...")
    status_file = Path("status/training_status.json")
    status_writer = TrainingStatusWriter(
        status_file,
        max_output_tokens=config.hyperparams.max_new_tokens,
        context_window=config.hyperparams.max_length,
        model_name=config.model.model_path
    )
    print(f"   ✓ Status writer created: {status_file}\n")

    # Create engine
    print("🚀 Creating TrainerEngine...")
    engine = TrainerEngine(status_writer)
    print(f"   ✓ Engine created\n")

    # Run training
    print("🏃 Starting training job...")
    print("-" * 80)
    result = engine.run_job(config)
    print("-" * 80 + "\n")

    # Report results
    print("📊 Training Result:")
    print(f"   Success: {result.success}")
    print(f"   Global step: {result.global_step}")
    print(f"   Runtime: {result.runtime_sec:.1f}s")
    print(f"   Final loss: {result.final_loss:.4f}")
    if result.last_checkpoint_path:
        print(f"   Checkpoint: {result.last_checkpoint_path}")
    if result.error_message:
        print(f"   Error: {result.error_message}")
    print()

    # Print architecture demonstration message
    print("=" * 80)
    print("📐 ARCHITECTURE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("This CLI demonstrates the clean API architecture:")
    print()
    print("  1. ✅ Configuration System (trainer.config)")
    print("     - Single TrainerConfig object")
    print("     - Type-safe dataclasses")
    print("     - JSON + CLI merging")
    print()
    print("  2. ✅ Profile System (trainer.profiles)")
    print("     - Pluggable data transformations")
    print("     - emoji_think, regime3, plain_sft")
    print()
    print("  3. ✅ Monitoring System (trainer.monitoring)")
    print("     - TrainingStatusWriter")
    print("     - LiveMonitorCallback")
    print()
    print("  4. ✅ Engine API (trainer.core)")
    print("     - TrainerEngine.run_job(config)")
    print("     - Clean orchestration layer")
    print()
    print("For production training, continue using:")
    print("  python3 core/train.py --dataset ... --model ...")
    print()
    print("The new modules can be integrated incrementally.")
    print("=" * 80)
    print()

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
