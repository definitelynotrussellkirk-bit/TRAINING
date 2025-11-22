#!/usr/bin/env python3
"""
Training Time Estimator

Estimates how long training will take BEFORE starting.

Based on:
- Model size
- Dataset size
- Batch size
- Hardware (GPU type)
- Benchmarks from previous runs
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict


@dataclass
class TrainingEstimate:
    """Estimated training parameters."""
    total_steps: int
    steps_per_epoch: int
    num_epochs: int
    seconds_per_step: float
    total_seconds: float
    total_hours: float
    estimated_completion: str
    memory_gb: float
    checkpoints_gb: float


class TimeEstimator:
    """Estimate training time and resources."""

    # Benchmark data: model_size → seconds_per_step for different GPUs
    BENCHMARKS = {
        "RTX 4090": {
            1: 1.0,   # 1B params
            3: 1.8,   # 3B params
            7: 3.2,   # 7B params
            8: 3.5,   # legacy value (unused)
            13: 5.5,  # 13B params
        },
        "RTX 3090": {
            1: 1.5,
            3: 2.5,
            7: 5.0,
            8: 5.5,
            13: 8.0,
        },
        "A100": {
            1: 0.8,
            3: 1.2,
            7: 2.0,
            8: 2.2,
            13: 3.5,
        },
        "unknown": {
            1: 2.0,
            3: 3.5,
            7: 6.0,
            8: 6.5,
            13: 10.0,
        }
    }

    @staticmethod
    def detect_gpu() -> tuple[str, float]:
        """Detect GPU type and VRAM."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                name, mem = line.split(', ')
                mem_gb = float(mem.split()[0]) / 1024  # Convert MB to GB

                # Simplify GPU name
                if "4090" in name:
                    gpu_type = "RTX 4090"
                elif "3090" in name:
                    gpu_type = "RTX 3090"
                elif "A100" in name:
                    gpu_type = "A100"
                else:
                    gpu_type = "unknown"

                return gpu_type, mem_gb

        except Exception:
            pass

        return "unknown", 24.0  # Default guess

    @staticmethod
    def estimate_memory(
        model_size_b: float,
        batch_size: int,
        use_4bit: bool = True
    ) -> Dict[str, float]:
        """Estimate memory usage."""

        # Base model memory (4-bit quantized)
        if use_4bit:
            model_memory = model_size_b * 0.5  # ~0.5 GB per billion params (4-bit)
        else:
            model_memory = model_size_b * 2.0  # ~2 GB per billion params (fp16)

        # Optimizer states (AdamW)
        optimizer_memory = model_size_b * 0.6  # Rough estimate

        # Gradients
        gradient_memory = model_size_b * 0.25

        # Activations (depends on batch size and sequence length)
        activation_memory = batch_size * 0.8  # Rough estimate

        total = model_memory + optimizer_memory + gradient_memory + activation_memory

        return {
            "model": round(model_memory, 1),
            "optimizer": round(optimizer_memory, 1),
            "gradients": round(gradient_memory, 1),
            "activations": round(activation_memory, 1),
            "total": round(total, 1)
        }

    @classmethod
    def estimate_training(
        cls,
        num_examples: int,
        batch_size: int,
        num_epochs: int,
        model_size_b: float,
        gpu_type: Optional[str] = None
    ) -> TrainingEstimate:
        """Estimate total training time."""

        # Auto-detect GPU if not specified
        if gpu_type is None:
            gpu_type, _ = cls.detect_gpu()

        # Calculate steps
        steps_per_epoch = num_examples // batch_size
        total_steps = steps_per_epoch * num_epochs

        # Get benchmark for this model size
        benchmarks = cls.BENCHMARKS.get(gpu_type, cls.BENCHMARKS["unknown"])

        # Find closest model size in benchmarks
        available_sizes = sorted(benchmarks.keys())
        closest_size = min(available_sizes, key=lambda x: abs(x - model_size_b))
        seconds_per_step = benchmarks[closest_size]

        # Scale if model size is different
        size_ratio = model_size_b / closest_size
        seconds_per_step *= size_ratio ** 0.5  # Square root scaling

        # Calculate total time
        total_seconds = total_steps * seconds_per_step
        total_hours = total_seconds / 3600

        # Estimated completion time
        completion_time = datetime.now() + timedelta(seconds=total_seconds)

        # Memory estimate
        memory = cls.estimate_memory(model_size_b, batch_size)

        # Checkpoint size (assume saving every 500 steps)
        num_checkpoints = total_steps // 500
        checkpoint_size_gb = model_size_b * 0.7  # LoRA adapter size
        checkpoints_gb = num_checkpoints * checkpoint_size_gb

        return TrainingEstimate(
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
            seconds_per_step=round(seconds_per_step, 2),
            total_seconds=int(total_seconds),
            total_hours=round(total_hours, 2),
            estimated_completion=completion_time.strftime("%Y-%m-%d %I:%M %p"),
            memory_gb=memory["total"],
            checkpoints_gb=round(checkpoints_gb, 1)
        )

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration nicely."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    @classmethod
    def display_estimate(cls, estimate: TrainingEstimate):
        """Display estimate in a nice format."""
        print("\n" + "=" * 80)
        print("TRAINING TIME ESTIMATION")
        print("=" * 80)

        print(f"\n📊 Dataset:")
        print(f"   Total steps: {estimate.total_steps:,}")
        print(f"   Steps per epoch: {estimate.steps_per_epoch:,}")
        print(f"   Epochs: {estimate.num_epochs}")

        print(f"\n⏱️  Time Estimate:")
        print(f"   Speed: ~{estimate.seconds_per_step} sec/step")
        print(f"   Total time: {cls.format_duration(estimate.total_seconds)} ({estimate.total_hours:.1f} hours)")
        print(f"   Estimated completion: {estimate.estimated_completion}")

        print(f"\n💾 Resources:")
        print(f"   GPU Memory: ~{estimate.memory_gb:.1f} GB")
        print(f"   Disk (checkpoints): ~{estimate.checkpoints_gb:.1f} GB")

        # Timeline
        print(f"\n📅 Timeline:")
        now = datetime.now()
        milestone_steps = [
            estimate.total_steps // 4,
            estimate.total_steps // 2,
            estimate.total_steps * 3 // 4,
            estimate.total_steps
        ]

        for i, step in enumerate(milestone_steps):
            if estimate.total_steps == 0:
                break  # Skip milestones if no steps
            percent = (step / estimate.total_steps) * 100
            elapsed = step * estimate.seconds_per_step
            eta = now + timedelta(seconds=elapsed)

            if i == 0:
                label = "25% complete"
            elif i == 1:
                label = "Halfway    "
            elif i == 2:
                label = "75% complete"
            else:
                label = "DONE       "

            print(f"   {label}: {eta.strftime('%I:%M %p')} (step {step:,})")

        print("\n⚠️  Note: This is an estimate. Actual time may vary ±20%")
        print("=" * 80 + "\n")


def main():
    """Test estimator."""
    print("=" * 80)
    print("TIME ESTIMATOR TEST")
    print("=" * 80)

    # Detect GPU
    gpu_type, vram = TimeEstimator.detect_gpu()
    print(f"\n🖥️  Detected: {gpu_type} with {vram:.0f} GB VRAM")

    # Example: 50K examples, 2 epochs, ~1B model
    estimate = TimeEstimator.estimate_training(
        num_examples=50_000,
        batch_size=4,
        num_epochs=2,
        model_size_b=8.0,
        gpu_type=gpu_type
    )

    TimeEstimator.display_estimate(estimate)

    # Memory check
    _, vram = TimeEstimator.detect_gpu()
    if estimate.memory_gb > vram:
        print(f"⚠️  WARNING: Estimated memory ({estimate.memory_gb:.1f} GB) exceeds available VRAM ({vram:.0f} GB)")
        print("   Consider:")
        print("   • Reducing batch size")
        print("   • Using gradient checkpointing")
        print("   • Using 4-bit quantization")
        print()


if __name__ == "__main__":
    main()
