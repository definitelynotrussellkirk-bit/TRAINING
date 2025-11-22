#!/usr/bin/env python3
"""
Test TrainerEngine with small dataset

This script tests the full TrainerEngine implementation.
Creates a tiny test dataset and runs a few training steps.
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core import TrainerEngine, TrainingResult
from trainer.config import create_default_config
from trainer.monitoring import TrainingStatusWriter


def create_tiny_test_dataset(output_path: Path, num_examples: int = 10):
    """Create a tiny test dataset for quick testing"""
    examples = []

    for i in range(num_examples):
        examples.append({
            "messages": [
                {"role": "user", "content": f"What is {i} + {i}?"},
                {"role": "assistant", "content": f"The answer is {i + i}."}
            ]
        })

    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Created test dataset: {output_path} ({num_examples} examples)")


def main():
    """Test TrainerEngine"""
    print("\n" + "=" * 80)
    print("TRAINER ENGINE TEST")
    print("=" * 80 + "\n")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_data = base_dir / "scratch" / "test_engine_data.jsonl"
    test_output = base_dir / "scratch" / "test_engine_output"
    model_path = base_dir / "models" / "Qwen3-0.6B"

    # Create test dataset
    print("Creating test dataset...")
    create_tiny_test_dataset(test_data, num_examples=10)
    print()

    # Create config
    print("Creating configuration...")
    config = create_default_config(
        model_path=str(model_path),
        dataset_path=str(test_data),
        output_dir=str(test_output),
        base_model="Qwen/Qwen3-0.6B",
        model_architecture="Qwen3ForCausalLM",
        max_context_length=4096,
        vocab_size=151936
    )

    # Override some settings for quick test
    config.hyperparams.num_epochs = 1
    config.hyperparams.max_steps = 3  # Just 3 steps for quick test
    config.hyperparams.save_steps = 2
    config.hyperparams.eval_steps = 2
    config.hyperparams.batch_size = 2
    config.profile.name = "emoji_think"  # Test with emoji profile

    print(f"  Model: {config.model.model_path}")
    print(f"  Dataset: {config.data.dataset_path}")
    print(f"  Output: {config.output.output_dir}")
    print(f"  Profile: {config.profile.name}")
    print(f"  Max steps: {config.hyperparams.max_steps}")
    print()

    # Create status writer
    status_writer = TrainingStatusWriter(
        "status/test_training_status.json",
        max_output_tokens=256,
        context_window=2048,
        model_name="test_engine"
    )

    # Create engine
    print("Creating TrainerEngine...")
    engine = TrainerEngine(status_writer)
    print()

    # Run training
    print("Starting training...")
    result = engine.run_job(config)
    print()

    # Check result
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    if result.success:
        print("✅ Training succeeded!")
        print(f"  Global step: {result.global_step}")
        print(f"  Runtime: {result.runtime_sec:.1f}s")
        print(f"  Final loss: {result.final_loss:.4f}")
        print(f"  Checkpoint: {result.last_checkpoint_path}")
        print()
        print("Summary:")
        for key, value in result.summary.items():
            if key not in ['config']:  # Skip config dump
                print(f"  {key}: {value}")
    else:
        print("❌ Training failed!")
        print(f"  Error: {result.error_message}")
    print("=" * 80)

    # Cleanup
    print("\nCleaning up test files...")
    if test_data.exists():
        test_data.unlink()
        print(f"  Deleted: {test_data}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
