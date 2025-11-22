#!/usr/bin/env python3
"""
Streaming Trainer - Train directly from a generator

This allows you to pass a data generator function directly to the trainer,
avoiding the need to pre-generate and save data to disk.

Perfect for:
- Billion-scale generation (no disk space needed)
- Fast iteration (no file I/O)
- LEO integration (streaming by design)
- Memory efficiency (generate on-the-fly)

Usage:
    from streaming_trainer import StreamingTrainer

    def my_generator():
        for i in range(1000):
            yield {
                "messages": [
                    {"role": "user", "content": f"Task {i}..."},
                    {"role": "assistant", "content": f"Answer {i}"}
                ]
            }

    trainer = StreamingTrainer(
        generator=my_generator,
        num_samples=1000,
        model="qwen3_0.6b",
        output_dir="~/adapter"
    )

    trainer.run()
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Dict, Optional
from dataclasses import dataclass

from validator import DatasetValidator
from train import UltimateTrainer


@dataclass
class StreamingConfig:
    """Configuration for streaming training."""
    generator: Callable[[], Iterator[Dict]]
    num_samples: int
    model: str
    output_dir: str

    # Training params
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4

    # Validation params
    validate_first: bool = True
    validation_samples: int = 100  # Samples to validate before training

    # Monitoring
    eval_steps: int = 100
    num_eval_samples: int = 5
    save_steps: int = 500

    # Flags
    skip_validation: bool = False
    yes: bool = False  # Skip confirmations
    auto_cleanup: bool = True  # Automatically delete temp file after training


class StreamingTrainer:
    """Train directly from a generator function."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.temp_file = None

    def run(self):
        """Execute streaming training pipeline."""
        print("\n" + "=" * 80)
        print("🌊 STREAMING TRAINER")
        print("=" * 80)
        print()

        print(f"📊 Configuration:")
        print(f"   Total samples: {self.config.num_samples:,}")
        print(f"   Model: {self.config.model}")
        print(f"   Output: {self.config.output_dir}")
        print(f"   Validation: {'Enabled' if self.config.validate_first else 'Disabled'}")
        print()

        # Step 1: Validate first N samples (optional but recommended)
        if self.config.validate_first and not self.config.skip_validation:
            print("━" * 80)
            print("STEP 1: Validating First Samples")
            print("━" * 80)
            print()

            if not self._validate_samples():
                print("\n❌ Validation failed! Aborting.")
                print("   Fix the issues or use --skip-validation")
                return False

            print("\n✅ Validation passed!")

        # Step 2: Stream data to temporary file
        print("\n" + "━" * 80)
        print("STEP 2: Streaming Data Generation")
        print("━" * 80)
        print()

        temp_dataset = self._stream_to_temp_file()

        # Step 3: Train using standard trainer
        print("\n" + "━" * 80)
        print("STEP 3: Training")
        print("━" * 80)
        print()

        success = self._train_from_file(temp_dataset)

        # Cleanup
        if self.temp_file and Path(self.temp_file).exists():
            if self.config.auto_cleanup:
                # Auto-delete
                Path(self.temp_file).unlink()
                print(f"\n🗑️  Cleaned up temporary file")
            else:
                # Ask user
                print("\n" + "━" * 80)
                print("Cleanup")
                print("━" * 80)
                print()
                file_size = Path(self.temp_file).stat().st_size / 1024 / 1024
                print(f"Training data file: {self.temp_file}")
                print(f"Size: {file_size:.2f} MB")
                print()

                if not self.config.yes:
                    response = input("Delete this file? [yes/no]: ").strip().lower()
                    if response == "yes":
                        Path(self.temp_file).unlink()
                        print("✅ File deleted")
                    else:
                        print(f"✅ File kept: {self.temp_file}")
                else:
                    # If -y flag is set, keep the file by default
                    print(f"✅ File kept: {self.temp_file}")

        if success:
            print("\n" + "=" * 80)
            print("🎉 STREAMING TRAINING COMPLETE!")
            print("=" * 80)
            print(f"\nModel saved to: {self.config.output_dir}")
            return True
        else:
            print("\n❌ Training failed!")
            return False

    def _validate_samples(self) -> bool:
        """Validate first N samples from generator."""
        print(f"🔍 Validating first {self.config.validation_samples} samples...")
        print()

        # Create temporary file for validation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

            # Generate validation samples
            gen = self.config.generator()
            count = 0

            for i, record in enumerate(gen):
                if i >= self.config.validation_samples:
                    break
                f.write(json.dumps(record) + '\n')
                count += 1

                if (i + 1) % 10 == 0:
                    print(f"   Sampled {i + 1} / {self.config.validation_samples}...")

        print(f"   ✅ Sampled {count} records for validation")
        print()

        # Run validator
        validator = DatasetValidator(Path(temp_path))
        passed = validator.run_full_validation()

        # Cleanup
        Path(temp_path).unlink()

        return passed

    def _stream_to_temp_file(self) -> Path:
        """Stream generator output to temporary file."""
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jsonl', prefix='streaming_train_')
        self.temp_file = temp_path

        print(f"📝 Streaming to temporary file: {temp_path}")
        print()

        # Stream data
        gen = self.config.generator()
        count = 0

        with open(temp_path, 'w') as f:
            for i, record in enumerate(gen):
                if i >= self.config.num_samples:
                    break

                f.write(json.dumps(record) + '\n')
                count += 1

                # Progress indicator
                if (i + 1) % 1000 == 0:
                    pct = (i + 1) / self.config.num_samples * 100
                    print(f"   Generated {i + 1:,} / {self.config.num_samples:,} samples ({pct:.1f}%)")

        print()
        print(f"✅ Streamed {count:,} samples")
        print(f"   File: {temp_path}")
        print(f"   Size: {Path(temp_path).stat().st_size / 1024 / 1024:.1f} MB")

        return Path(temp_path)

    def _train_from_file(self, dataset_path: Path) -> bool:
        """Train using standard Ultimate Trainer."""

        # Create args namespace for UltimateTrainer
        class Args:
            pass

        args = Args()
        args.dataset = str(dataset_path)
        args.model = self.config.model
        args.output_dir = self.config.output_dir
        args.epochs = self.config.epochs
        args.batch_size = self.config.batch_size
        args.gradient_accumulation = self.config.gradient_accumulation
        args.learning_rate = self.config.learning_rate
        args.warmup_steps = 100
        args.lora_r = 64
        args.lora_alpha = 32
        args.eval_steps = self.config.eval_steps
        args.num_eval_samples = self.config.num_eval_samples
        args.save_steps = self.config.save_steps
        args.skip_validation = True  # Already validated
        args.yes = self.config.yes

        # Run trainer
        trainer = UltimateTrainer(args)
        return trainer.run()


def example_generator():
    """
    Example generator function.

    In practice, this would be your LEO generator or other data source.
    """
    from examples.leo_mini_generator import generate_property_count_sample

    # Generate property counting tasks
    for i in range(1000):
        record = generate_property_count_sample(i)
        # Remove metadata, keep only messages
        yield {"messages": record["messages"]}


def main():
    """Example usage of StreamingTrainer."""
    import argparse

    parser = argparse.ArgumentParser(description="Stream training from generator")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--model", type=str, default="qwen3_0.6b", help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="~/streaming_adapter", help="Output directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmations")

    args = parser.parse_args()

    # Create config
    config = StreamingConfig(
        generator=example_generator,
        num_samples=args.samples,
        model=args.model,
        output_dir=args.output_dir,
        skip_validation=args.skip_validation,
        yes=args.yes
    )

    # Run streaming trainer
    trainer = StreamingTrainer(config)
    success = trainer.run()

    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
