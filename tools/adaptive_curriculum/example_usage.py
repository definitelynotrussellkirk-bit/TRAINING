#!/usr/bin/env python3
"""Example usage of adaptive curriculum system.

This demonstrates how to:
1. Set up generators
2. Run manual generation
3. Run evaluations
4. Track stats
"""
from pathlib import Path

from core.paths import get_base_dir
from tools.adaptive_curriculum.controller import DifficultyController, DifficultyConfig
from tools.adaptive_curriculum.orchestrator import (
    AdaptiveCurriculumOrchestrator,
    OrchestratorConfig
)
from tools.adaptive_curriculum.registry import create_syllo_generator
from tools.adaptive_curriculum.stats import StatsManager


def _get_base() -> Path:
    """Get base directory for examples."""
    return get_base_dir()


def example_basic_usage():
    """Basic usage: generate one batch."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)

    base_dir = _get_base()

    # Create config
    config = OrchestratorConfig(
        base_dir=base_dir,
        batch_size=100,  # Small batch for demo
        target_accuracy=0.8
    )

    # Create orchestrator
    orchestrator = AdaptiveCurriculumOrchestrator(config)

    # Register SYLLO generator
    syllo_config = create_syllo_generator()
    orchestrator.register_generator(syllo_config)

    print(f"Registered generators: {orchestrator.registry.list_generators()}")

    # Generate one batch
    print("\nGenerating batch...")
    output_path = orchestrator.generate_batch("syllo")

    if output_path:
        print(f"✓ Generated: {output_path}")
    else:
        print("✗ Generation failed")


def example_stats_tracking():
    """Track stats manually."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Stats Tracking")
    print("=" * 60)

    base_dir = _get_base()
    stats_file = base_dir / "status" / "curriculum_stats.json"

    # Create stats manager
    stats_manager = StatsManager(
        storage_path=stats_file,
        target_accuracy=0.8,
        window_size=200
    )

    # Simulate some evaluation results
    print("\nSimulating evaluations...")

    # Easy level: 90% accuracy (too easy)
    stats_manager.update(
        generator_id="syllo",
        difficulty=0,
        correct_count=90,
        total_count=100,
        toggles={"difficulty": "EASY"}
    )

    # Medium level: 75% accuracy (good)
    stats_manager.update(
        generator_id="syllo",
        difficulty=1,
        correct_count=75,
        total_count=100,
        toggles={"difficulty": "MEDIUM"}
    )

    # Get stats
    syllo_stats = stats_manager.get_stats("syllo")

    print(f"\nAccuracy at level 0 (easy): {syllo_stats.accuracy(0):.1%}")
    print(f"Accuracy at level 1 (medium): {syllo_stats.accuracy(1):.1%}")

    print(f"\nRecent evaluations:")
    for eval_result in syllo_stats.recent_evals(5):
        print(f"  Diff {eval_result.difficulty_level}: "
              f"{eval_result.accuracy:.1%} "
              f"({eval_result.correct_count}/{eval_result.num_examples})")


def example_difficulty_control():
    """Demonstrate difficulty controller."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Difficulty Controller")
    print("=" * 60)

    # Create controller
    config = DifficultyConfig(
        min_level=0,
        max_level=2,
        target_accuracy=0.8,
        accuracy_band=0.05,
        min_samples_required=20
    )
    controller = DifficultyController(config)

    # Create stats manager
    base_dir = _get_base()
    stats_file = base_dir / "status" / "curriculum_stats_example.json"
    stats_manager = StatsManager(storage_path=stats_file)

    # Simulate progression
    scenarios = [
        (0, 50, 100, "Bootstrap: no data yet"),
        (0, 90, 100, "Too easy at level 0"),
        (1, 78, 100, "Just right at level 1"),
        (1, 88, 100, "Getting too easy at level 1"),
        (2, 70, 100, "Too hard at level 2"),
    ]

    print("\nSimulating difficulty progression:")
    for difficulty, correct, total, description in scenarios:
        # Update stats
        stats_manager.update("syllo", difficulty, correct, total)

        # Get stats
        stats = stats_manager.get_stats("syllo")

        # Choose difficulty for next batch
        next_difficulty = controller.choose_difficulty("syllo", stats)

        accuracy = stats.accuracy(difficulty)
        acc_str = f"{accuracy:.1%}" if accuracy else "N/A"

        print(f"\n  {description}")
        print(f"    Current level: {difficulty}, Accuracy: {acc_str}")
        print(f"    Next level: {next_difficulty}")


def example_full_workflow():
    """Full workflow: generate -> evaluate -> adapt."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Full Workflow")
    print("=" * 60)

    base_dir = _get_base()

    config = OrchestratorConfig(
        base_dir=base_dir,
        batch_size=50,  # Small for demo
        eval_sample_size=10
    )

    orchestrator = AdaptiveCurriculumOrchestrator(config)

    # Register generator
    syllo_config = create_syllo_generator()
    orchestrator.register_generator(syllo_config)

    print("Starting workflow...\n")

    # 1. Generate
    print("1. Generating batch...")
    output_path = orchestrator.generate_batch("syllo")
    if output_path:
        print(f"   ✓ {output_path}")

    # 2. Evaluate (would normally call inference API)
    print("\n2. Evaluations (skipped - requires inference API)")
    print("   To run: orchestrator.run_evaluations()")

    # 3. Check status
    print("\n3. Status:")
    orchestrator.print_status()


def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_stats_tracking,
        example_difficulty_control,
        example_full_workflow
    ]

    for example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as exc:
            print(f"\n✗ Error in {example_func.__name__}: {exc}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
