#!/usr/bin/env python3
"""
Training Time Estimator

Estimates training duration, resource usage, and completion time BEFORE starting training jobs.
Uses empirical GPU benchmarks and mathematical scaling formulas to predict training performance.

=== CORE RESPONSIBILITY ===
Prevent resource overruns by providing accurate pre-training estimates for:
- Training duration (hours)
- GPU memory usage (GB)
- Disk space for checkpoints (GB)
- Expected completion time

=== ESTIMATION METHODOLOGY ===

1. **Benchmark Lookup:**
   - Look up seconds/step for (GPU type, model size)
   - Benchmarks stored in TimeEstimator.BENCHMARKS
   - 3 GPU types: RTX 4090, RTX 3090, A100
   - 5 model sizes: 1B, 3B, 7B, 8B, 13B params

2. **Model Size Scaling:**
   - If model size not in benchmarks, use square root scaling:
   - seconds_per_step = benchmark[closest_size] * (actual_size / closest_size)^0.5
   - Example: 2B model → sqrt(2/1) = 1.41x slower than 1B benchmark

3. **Memory Estimation:**
   - Model: 0.5 GB/B (4-bit) or 2.0 GB/B (fp16)
   - Optimizer: 0.6 GB/B (AdamW states)
   - Gradients: 0.25 GB/B
   - Activations: batch_size * 0.8 GB
   - Total = model + optimizer + gradients + activations

4. **Time Calculation:**
   - steps_per_epoch = num_examples // batch_size
   - total_steps = steps_per_epoch * num_epochs
   - total_seconds = total_steps * seconds_per_step
   - completion_time = now + total_seconds

=== KEY FORMULAS ===

**Training Steps:**
```
steps_per_epoch = dataset_size / batch_size
total_steps = steps_per_epoch × num_epochs
```

**Time Estimate:**
```
total_time = total_steps × seconds_per_step
seconds_per_step = BENCHMARKS[gpu_type][model_size] × (actual_size / benchmark_size)^0.5
```

**Memory Estimate (4-bit quantization):**
```
model_memory = model_size_B × 0.5 GB
optimizer_memory = model_size_B × 0.6 GB
gradient_memory = model_size_B × 0.25 GB
activation_memory = batch_size × 0.8 GB × (max_length / 2048) × checkpoint_factor
total_memory = sum of above

where:
- checkpoint_factor = 0.35 if gradient_checkpointing else 1.0
- 0.8 GB/batch coefficient was empirically calibrated at max_length=2048
```

**Note:** VRAM depends on micro-batch size, NOT effective batch (batch × gradient_accumulation).
gradient_accumulation only affects training time, not memory.

=== USAGE EXAMPLE ===
```python
from core.time_estimator import TimeEstimator

# Estimate training for 50k examples, 2 epochs, 1B model
estimate = TimeEstimator.estimate_training(
    num_examples=50_000,
    batch_size=4,
    num_epochs=2,
    model_size_b=1.0,
    gpu_type="RTX 4090"  # Auto-detected if None
)

print(f"Training will take {estimate.total_hours:.1f} hours")
print(f"GPU memory needed: {estimate.memory_gb:.1f} GB")
print(f"Completion: {estimate.estimated_completion}")

# Check if estimate fits available VRAM
_, vram = TimeEstimator.detect_gpu()
if estimate.memory_gb > vram:
    print("WARNING: Not enough VRAM!")
```

=== INTEGRATION POINTS ===
- Used by: core/training_daemon.py (pre-flight checks)
- Used by: monitoring UI (display estimates)
- Inputs: config.json (batch_size), dataset (num_examples)
- Outputs: TrainingEstimate object (time/memory/disk predictions)

=== ACCURACY ===
- Estimates are ±20% accurate for known GPU/model combinations
- Square root scaling formula provides reasonable estimates for intermediate sizes
- Actual training speed depends on:
  - Gradient checkpointing (slows by ~10-15%)
  - Sequence length (longer = more memory)
  - System load (background processes)
"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict


@dataclass
class TrainingEstimate:
    """
    Complete training estimate with time, memory, and disk predictions.

    Returned by TimeEstimator.estimate_training(). Contains all metrics needed for
    pre-flight checks and UI display.

    Attributes:
        total_steps: Total training steps (steps_per_epoch × num_epochs)
        steps_per_epoch: Steps in one epoch (num_examples // batch_size)
        num_epochs: Number of epochs to train
        seconds_per_step: Estimated seconds per training step (from GPU benchmarks)
        total_seconds: Total training duration in seconds
        total_hours: Total training duration in hours (total_seconds / 3600)
        estimated_completion: Human-readable completion time ("2025-11-24 03:45 PM")
        memory_gb: Estimated GPU memory usage in GB (model + optimizer + gradients + activations)
        checkpoints_gb: Estimated disk space for checkpoints in GB

    Example:
        estimate = TimeEstimator.estimate_training(
            num_examples=50_000, batch_size=4, num_epochs=2, model_size_b=1.0
        )
        print(f"Training will take {estimate.total_hours:.1f} hours")
        print(f"Need {estimate.memory_gb:.1f} GB VRAM")
    """
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
    """
    Training time and resource estimation engine.

    Provides pre-flight estimates for training jobs using empirical GPU benchmarks
    and mathematical scaling formulas. Prevents resource overruns by predicting
    memory usage, training duration, and disk space requirements.

    === RESPONSIBILITIES ===
    1. GPU Detection - Identify GPU type and VRAM using nvidia-smi
    2. Memory Estimation - Calculate model + optimizer + gradient + activation memory
    3. Time Estimation - Predict training duration using GPU benchmarks
    4. Scaling - Interpolate for model sizes not in benchmark table
    5. Validation - Check if estimated memory exceeds available VRAM

    === DATA FLOW ===
    estimate_training() workflow:
    1. Auto-detect GPU type (if not provided)
    2. Calculate steps: steps_per_epoch = num_examples // batch_size
    3. Look up seconds_per_step from BENCHMARKS[gpu_type][model_size_b]
    4. Scale if model size not exact match: seconds_per_step *= (size_ratio)^0.5
    5. Calculate total time: total_seconds = total_steps * seconds_per_step
    6. Estimate memory: estimate_memory(model_size_b, batch_size)
    7. Return TrainingEstimate with all metrics

    === GPU BENCHMARKS (seconds per step) ===
    ```
                1B    3B    7B    8B    13B
    RTX 4090    1.0   1.8   3.2   3.5   5.5
    RTX 3090    1.5   2.5   5.0   5.5   8.0
    A100        0.8   1.2   2.0   2.2   3.5
    unknown     2.0   3.5   6.0   6.5   10.0  (conservative fallback)
    ```

    === SCALING FORMULA ===
    For model sizes between benchmarks:
    ```
    seconds_per_step = BENCHMARKS[closest_size] * sqrt(actual_size / closest_size)
    ```
    Example: 2B model on RTX 4090
    - Closest benchmark: 1B → 1.0 sec/step
    - Scale: sqrt(2/1) = 1.41
    - Result: 1.0 * 1.41 = 1.41 sec/step

    === MEMORY ESTIMATION ===
    Components (4-bit quantization):
    - Model: model_size_B × 0.5 GB
    - Optimizer (AdamW): model_size_B × 0.6 GB
    - Gradients: model_size_B × 0.25 GB
    - Activations: batch_size × 0.8 GB × (max_length / 2048) × checkpoint_factor

    checkpoint_factor = 0.35 if gradient_checkpointing else 1.0

    === USAGE ===
    ```python
    # Basic estimate
    estimate = TimeEstimator.estimate_training(
        num_examples=50_000,
        batch_size=4,
        num_epochs=2,
        model_size_b=1.0
    )

    # Check VRAM
    _, vram = TimeEstimator.detect_gpu()
    if estimate.memory_gb > vram:
        print("ERROR: Not enough VRAM!")

    # Display full report
    TimeEstimator.display_estimate(estimate)
    ```

    === ATTRIBUTES ===
    BENCHMARKS: Dict[str, Dict[int, float]]
        Empirical training speed for (GPU type, model size) combinations.
        Updated periodically based on real training runs.
    """

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
        """
        Detect GPU type and available VRAM using nvidia-smi.

        Runs nvidia-smi to query GPU name and total memory. Simplifies GPU names
        to match BENCHMARKS keys ("RTX 4090", "RTX 3090", "A100", "unknown").

        Algorithm:
            1. Run: nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            2. Parse first line (primary GPU)
            3. Extract GPU name and memory (MB)
            4. Convert memory to GB: mem_gb = MB / 1024
            5. Simplify name: "NVIDIA GeForce RTX 4090" → "RTX 4090"
            6. Return (gpu_type, vram_gb)

        Returns:
            tuple[str, float]: (GPU type, VRAM in GB)
            - GPU type: One of ["RTX 4090", "RTX 3090", "A100", "unknown"]
            - VRAM: Total GPU memory in gigabytes (e.g., 24.0)

        Fallback:
            If nvidia-smi fails or times out, returns ("unknown", 24.0) as safe default.

        Example:
            gpu_type, vram = TimeEstimator.detect_gpu()
            print(f"Detected: {gpu_type} with {vram:.0f} GB VRAM")
            # Output: "Detected: RTX 4090 with 24 GB VRAM"
        """
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
        use_4bit: bool = True,
        max_length: int = 2048,
        gradient_checkpointing: bool = True,
        num_layers: int = None,
        num_heads: int = None,
        hidden_dim: int = None
    ) -> Dict[str, float]:
        """
        Estimate GPU memory usage for training.

        Calculates total VRAM needed for model weights, optimizer states, gradients,
        and activations. Used for pre-flight checks to prevent OOM crashes.

        Args:
            model_size_b: Model size in billions of parameters (e.g., 0.6 for Qwen3-0.6B)
            batch_size: Training batch size (micro-batch, not effective batch)
            use_4bit: Whether using 4-bit quantization (QLoRA) - default True
            max_length: Maximum sequence length (default: 2048)
            gradient_checkpointing: Whether gradient checkpointing is enabled (default: True)
            num_layers: Number of transformer layers (auto-estimated if None)
            num_heads: Number of attention heads (auto-estimated if None)
            hidden_dim: Hidden dimension size (auto-estimated if None)

        Returns:
            Dict[str, float]: Memory breakdown in GB

        Memory Formulas:
            For bf16/fp16 full training (use_4bit=False):
                model = params × 2 bytes
                optimizer = params × 8 bytes (AdamW: momentum fp32 + variance fp32)
                gradients = params × 4 bytes (accumulated in fp32)
                activations = hidden_states + attention_matrices
                    - hidden: layers × batch × seq × hidden × 2 bytes
                    - attention: layers × batch × heads × seq² × 4 bytes (BIG!)

            For 4-bit QLoRA (use_4bit=True):
                model = params × 0.5 bytes (4-bit quantized)
                optimizer = trainable_params × 8 bytes (only LoRA weights)
                gradients = trainable_params × 4 bytes
                activations = similar but reduced due to frozen base

        Note: Attention matrices scale with seq_length² - this dominates VRAM
        for long sequences. Gradient checkpointing helps by recomputing activations.

        Example:
            # Qwen3-0.6B, batch_size=1, max_length=2048, bf16 full training
            mem = TimeEstimator.estimate_memory(0.6, 1, use_4bit=False)
            # Returns ~17 GB (attention matrices are 6+ GB alone!)
        """
        # Auto-estimate architecture params based on model size
        # These are approximate - real values vary by architecture
        if num_layers is None:
            if model_size_b <= 0.5:
                num_layers = 16
            elif model_size_b <= 1.0:
                num_layers = 24
            elif model_size_b <= 3.0:
                num_layers = 32
            elif model_size_b <= 8.0:
                num_layers = 40
            else:
                num_layers = 48

        if hidden_dim is None:
            if model_size_b <= 0.5:
                hidden_dim = 896
            elif model_size_b <= 1.0:
                hidden_dim = 1024
            elif model_size_b <= 3.0:
                hidden_dim = 2048
            elif model_size_b <= 8.0:
                hidden_dim = 4096
            else:
                hidden_dim = 5120

        if num_heads is None:
            if model_size_b <= 0.5:
                num_heads = 14
            elif model_size_b <= 1.0:
                num_heads = 16
            elif model_size_b <= 3.0:
                num_heads = 32
            elif model_size_b <= 8.0:
                num_heads = 32
            else:
                num_heads = 40

        params_billion = model_size_b

        if use_4bit:
            # 4-bit QLoRA training
            model_memory = params_billion * 0.5  # 4-bit = 0.5 bytes per param

            # Only LoRA params are trained (typically ~1-2% of model)
            trainable_fraction = 0.02
            trainable_params = params_billion * trainable_fraction

            optimizer_memory = trainable_params * 8  # AdamW states for LoRA only
            gradient_memory = trainable_params * 4   # Gradients for LoRA only

            # Activations still needed but base model frozen reduces overhead
            # Use simplified formula for QLoRA
            activation_memory = batch_size * 0.8 * (max_length / 2048.0)
            if gradient_checkpointing:
                activation_memory *= 0.35

        else:
            # Full bf16/fp16 training - ALL params trained
            model_memory = params_billion * 2  # bf16 = 2 bytes per param

            # AdamW optimizer: momentum (fp32) + variance (fp32) = 8 bytes per param
            optimizer_memory = params_billion * 8

            # Gradients accumulated in fp32 = 4 bytes per param
            gradient_memory = params_billion * 4

            # Activations: hidden states + attention matrices
            # Hidden states: layers × batch × seq × hidden × 2 bytes
            hidden_memory = (num_layers * batch_size * max_length * hidden_dim * 2) / 1e9

            # Attention matrices: layers × batch × heads × seq × seq × 4 bytes
            # This is the BIG one - scales with seq_length²!
            attention_memory = (num_layers * batch_size * num_heads * max_length * max_length * 4) / 1e9

            activation_memory = hidden_memory + attention_memory

            # Gradient checkpointing: recompute forward pass, only store layer boundaries
            # Reduces activation memory by ~70%
            if gradient_checkpointing:
                activation_memory *= 0.3

        # CUDA/PyTorch overhead (kernels, fragmentation, workspace)
        overhead = 1.5

        total = model_memory + optimizer_memory + gradient_memory + activation_memory + overhead

        return {
            "model": round(model_memory, 1),
            "optimizer": round(optimizer_memory, 1),
            "gradients": round(gradient_memory, 1),
            "activations": round(activation_memory, 1),
            "overhead": round(overhead, 1),
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
        """
        Estimate complete training job (time, memory, disk, completion time).

        Main entry point for training estimation. Uses GPU benchmarks + scaling formulas
        to predict training performance. Returns TrainingEstimate with all metrics.

        Args:
            num_examples: Total training examples in dataset
            batch_size: Training batch size
            num_epochs: Number of training epochs
            model_size_b: Model size in billions of parameters (e.g., 1.0, 3.0, 7.0)
            gpu_type: GPU type (auto-detected if None). One of:
                      ["RTX 4090", "RTX 3090", "A100", "unknown"]

        Returns:
            TrainingEstimate: Complete estimate with these fields:
                - total_steps: Total training steps
                - steps_per_epoch: Steps per epoch
                - num_epochs: Number of epochs
                - seconds_per_step: Estimated seconds per step
                - total_seconds: Total training time (seconds)
                - total_hours: Total training time (hours)
                - estimated_completion: Completion time (formatted string)
                - memory_gb: Estimated GPU memory (GB)
                - checkpoints_gb: Estimated disk space (GB)

        Algorithm:
            1. Auto-detect GPU if not provided
            2. Calculate training steps:
               steps_per_epoch = num_examples // batch_size
               total_steps = steps_per_epoch * num_epochs
            3. Look up benchmark speed from BENCHMARKS[gpu_type]
            4. Find closest model size in benchmarks
            5. Scale if needed: seconds_per_step *= (actual_size / closest_size)^0.5
            6. Calculate total time: total_seconds = total_steps * seconds_per_step
            7. Estimate memory: estimate_memory(model_size_b, batch_size)
            8. Estimate checkpoints: (total_steps / 10000) * 0.7 GB  (1 ckpt per 10k steps)
            9. Calculate completion time: now + total_seconds
            10. Return TrainingEstimate

        Scaling Formula:
            For model sizes between benchmarks, use square root scaling:
            seconds_per_step = BENCHMARKS[closest_size] * sqrt(actual_size / closest_size)

            Example: 2B model on RTX 4090
            - Closest benchmark: 1B → 1.0 sec/step
            - Scale factor: sqrt(2.0 / 1.0) = 1.41
            - Result: 1.0 * 1.41 = 1.41 sec/step

        Example Usage:
            # Estimate 50k examples, 2 epochs, 1B model
            estimate = TimeEstimator.estimate_training(
                num_examples=50_000,
                batch_size=4,
                num_epochs=2,
                model_size_b=1.0
            )
            print(f"Training: {estimate.total_hours:.1f} hours")
            print(f"Memory: {estimate.memory_gb:.1f} GB")
            print(f"Done: {estimate.estimated_completion}")

            # Check if fits in VRAM
            _, vram = TimeEstimator.detect_gpu()
            if estimate.memory_gb > vram:
                print("ERROR: Not enough VRAM!")

        Accuracy:
            - ±20% for known GPU/model combinations
            - Square root scaling provides reasonable estimates for intermediate sizes
            - Actual speed depends on: gradient checkpointing, sequence length, system load
        """

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

        # Checkpoint size (assume saving every 10k steps per config)
        num_checkpoints = total_steps // 10000
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
