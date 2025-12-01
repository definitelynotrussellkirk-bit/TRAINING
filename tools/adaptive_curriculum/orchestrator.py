#!/usr/bin/env python3
"""Main orchestrator for adaptive curriculum learning."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .controller import DifficultyController, DifficultyConfig
from .evaluator import ModelEvaluator, EvalSetBuilder, PeriodicEvaluator
from .registry import GeneratorRegistry, GeneratorConfig
from .stats import StatsManager


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator."""
    base_dir: Path

    # Directories
    inbox_dir: Path = None
    queue_dir: Path = None
    eval_sets_dir: Path = None
    eval_results_dir: Path = None
    stats_file: Path = None

    # Generation parameters
    batch_size: int = 1000  # Examples per generation
    queue_threshold: int = 2  # Generate when queue drops below this

    # Evaluation parameters
    eval_interval: int = 500  # Evaluate every N training steps
    eval_sample_size: int = 100  # Examples per eval set

    # Curriculum parameters
    target_accuracy: float = 0.8
    accuracy_band: float = 0.05
    window_size: int = 200
    min_samples: int = 20

    # Inference API
    inference_url: str = "http://192.168.x.x:8000  # TODO: Use core.hosts.get_service_url("inference")/generate"

    def __post_init__(self):
        """Set default paths if not provided."""
        if self.inbox_dir is None:
            self.inbox_dir = self.base_dir / "inbox"
        if self.queue_dir is None:
            self.queue_dir = self.base_dir / "queue"
        if self.eval_sets_dir is None:
            self.eval_sets_dir = self.base_dir / "eval_sets"
        if self.eval_results_dir is None:
            self.eval_results_dir = self.base_dir / "eval_results"
        if self.stats_file is None:
            self.stats_file = self.base_dir / "status" / "curriculum_stats.json"

        # Create directories
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.eval_sets_dir.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)


class AdaptiveCurriculumOrchestrator:
    """Main controller for adaptive curriculum learning.

    Workflow:
    1. Check queue depth
    2. If below threshold:
       - For each generator:
         - Get current difficulty from controller
         - Generate batch at that difficulty
         - Write to inbox (auto-queued by daemon)
    3. Periodically evaluate model
    4. Update stats and adjust difficulty
    """

    def __init__(self, config: OrchestratorConfig):
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config

        # Initialize components
        difficulty_config = DifficultyConfig(
            target_accuracy=config.target_accuracy,
            accuracy_band=config.accuracy_band,
            min_samples_required=config.min_samples
        )

        self.registry = GeneratorRegistry()
        self.stats_manager = StatsManager(
            storage_path=config.stats_file,
            target_accuracy=config.target_accuracy,
            window_size=config.window_size
        )
        self.controller = DifficultyController(config=difficulty_config)
        self.evaluator = ModelEvaluator(inference_url=config.inference_url)
        self.eval_builder = EvalSetBuilder(eval_sets_dir=config.eval_sets_dir)

        # State tracking
        self._last_eval_time = None
        self._generation_count = 0

    def register_generator(self, config: GeneratorConfig) -> None:
        """Register a new generator."""
        self.registry.register(config)

    def get_queue_depth(self) -> int:
        """Get current queue depth (files waiting for training)."""
        total = 0
        for priority in ("high", "normal", "low"):
            priority_dir = self.config.queue_dir / priority
            if priority_dir.exists():
                total += len(list(priority_dir.glob("*.jsonl")))
        return total

    def should_generate(self) -> bool:
        """Check if we should generate new training data."""
        return self.get_queue_depth() < self.config.queue_threshold

    def generate_batch(self, generator_id: str) -> Optional[Path]:
        """Generate a batch of training data.

        Args:
            generator_id: Generator to use

        Returns:
            Path to generated file, or None if generation failed
        """
        # Get stats for this generator
        stats = self.stats_manager.get_stats(generator_id)

        # Choose difficulty
        difficulty = self.controller.choose_difficulty(generator_id, stats)

        print(f"Generating {generator_id} at difficulty {difficulty}")
        print(f"  Current accuracy: {stats.accuracy(difficulty)}")
        print(f"  Sample count: {stats.sample_count(difficulty)}")

        # Generate examples
        try:
            examples = self.registry.generate(
                generator_id=generator_id,
                difficulty=difficulty,
                count=self.config.batch_size
            )
        except Exception as exc:
            print(f"Generation failed: {exc}")
            return None

        # Write to inbox
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{generator_id}_diff{difficulty}_{timestamp}.jsonl"
        output_path = self.config.inbox_dir / filename

        with output_path.open("w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        print(f"Generated {len(examples)} examples -> {output_path}")

        # Create eval set from this batch
        eval_path = self.eval_builder.create_eval_set(
            generator_id=generator_id,
            difficulty=difficulty,
            training_file=output_path,
            eval_size=self.config.eval_sample_size
        )
        print(f"Created eval set -> {eval_path}")

        self._generation_count += 1
        return output_path

    def run_evaluations(self) -> List[Dict]:
        """Run evaluations on all generators at all difficulty levels.

        Returns:
            List of evaluation results
        """
        print("Running evaluations...")

        periodic_eval = PeriodicEvaluator(
            evaluator=self.evaluator,
            eval_sets_dir=self.config.eval_sets_dir,
            results_dir=self.config.eval_results_dir
        )

        # Run all evals (uses default exact match judge)
        results = periodic_eval.run_all_evals()

        # Update stats with results
        for result in results:
            self.stats_manager.update(
                generator_id=result["generator_id"],
                difficulty=result["difficulty_level"],
                correct_count=result["correct_count"],
                total_count=result["total_count"],
                toggles=result.get("toggles", {})
            )

        print(f"Completed {len(results)} evaluations")

        # Print summary
        for result in results:
            print(f"  {result['generator_id']} diff{result['difficulty_level']}: "
                  f"{result['accuracy']:.1%} ({result['correct_count']}/{result['total_count']})")

        self._last_eval_time = datetime.now()
        return results

    def run_generation_cycle(self) -> None:
        """Run one generation cycle for all generators."""
        if not self.should_generate():
            print(f"Queue depth {self.get_queue_depth()} >= threshold {self.config.queue_threshold}")
            print("Skipping generation")
            return

        print(f"Queue depth {self.get_queue_depth()} < threshold {self.config.queue_threshold}")
        print("Starting generation cycle...")

        # Generate for each registered generator
        for generator_id in self.registry.list_generators():
            self.generate_batch(generator_id)

    def run_loop(self, check_interval: int = 300) -> None:
        """Run continuous generation loop.

        Args:
            check_interval: Seconds between queue checks
        """
        print("Starting adaptive curriculum orchestrator")
        print(f"Registered generators: {self.registry.list_generators()}")
        print(f"Queue threshold: {self.config.queue_threshold}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Check interval: {check_interval}s")
        print()

        while True:
            try:
                # Run generation if needed
                self.run_generation_cycle()

                # Periodic evaluation (every N generations or time-based)
                if self._generation_count % 5 == 0 and self._generation_count > 0:
                    self.run_evaluations()

                # Wait before next check
                print(f"\nWaiting {check_interval}s before next check...")
                time.sleep(check_interval)

            except KeyboardInterrupt:
                print("\nStopping orchestrator...")
                break
            except Exception as exc:
                print(f"Error in main loop: {exc}")
                print("Continuing...")
                time.sleep(check_interval)

    def status_report(self) -> Dict:
        """Generate status report."""
        stats_summary = self.stats_manager.summary()
        controller_summary = self.controller.summary()

        report = {
            "timestamp": datetime.now().isoformat(),
            "queue_depth": self.get_queue_depth(),
            "generation_count": self._generation_count,
            "last_eval_time": self._last_eval_time.isoformat() if self._last_eval_time else None,
            "generators": self.registry.list_generators(),
            "stats": stats_summary,
            "controller": controller_summary
        }

        return report

    def print_status(self) -> None:
        """Print human-readable status."""
        report = self.status_report()

        print("\n" + "=" * 60)
        print("ADAPTIVE CURRICULUM STATUS")
        print("=" * 60)
        print(f"Time: {report['timestamp']}")
        print(f"Queue depth: {report['queue_depth']}")
        print(f"Generations: {report['generation_count']}")
        print(f"Last eval: {report['last_eval_time'] or 'Never'}")
        print()

        print("Generators:")
        for gen_id in report["generators"]:
            current_level = report["controller"]["current_levels"].get(gen_id, 0)
            accuracies = report["stats"]["generators"].get(gen_id, {}).get("current_accuracies", {})

            print(f"  {gen_id}:")
            print(f"    Current level: {current_level}")
            print(f"    Accuracies by level:")
            for diff in sorted(accuracies.keys(), key=int):
                acc = accuracies[diff]
                if acc is not None:
                    print(f"      Level {diff}: {acc:.1%}")

        print("=" * 60)
