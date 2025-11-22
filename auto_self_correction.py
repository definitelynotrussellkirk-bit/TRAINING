#!/usr/bin/env python3
"""
Auto Self-Correction Generator

Automatically generates self-correction training data during eval steps.
Hooks into LiveInferenceMonitor to capture model predictions vs golden answers.

Usage:
    Integrates with train.py - runs automatically every N eval steps
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from self_correction_trainer import ErrorCodeGenerator, SelfCorrectionPipeline


class AutoSelfCorrectionGenerator:
    """Auto-generate self-correction training data during eval."""

    def __init__(
        self,
        output_dir: str = "data/self_correction",
        auto_queue: bool = True,
        queue_dir: str = "queue/normal",
        max_examples: int = 200,  # Auto-drop when we hit this many examples
        generation_interval: int = None  # Optional: also generate every N steps
    ):
        """
        Args:
            output_dir: Where to save generated training data
            auto_queue: If True, automatically add to training queue
            queue_dir: Queue directory for auto-queueing
            max_examples: Auto-drop to queue when we accumulate this many examples
            generation_interval: Optional: Also generate every N steps (None = example-driven only)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_queue = auto_queue
        self.queue_dir = Path(queue_dir)
        self.max_examples = max_examples
        self.generation_interval = generation_interval  # Optional

        self.pipeline = SelfCorrectionPipeline()
        self.pending_examples = []  # Accumulate examples
        self.last_generation_step = 0

        self.stats = {
            'total_evals': 0,
            'correct_first_try': 0,
            'needed_correction': 0,
            'files_generated': 0,
            'examples_generated': 0
        }

    def process_inference_results(
        self,
        results: List,  # List of InferenceResult from live_monitor.py
        step: int
    ):
        """
        Process live inference results and generate self-correction data.

        Args:
            results: List of InferenceResult objects from LiveInferenceMonitor
            step: Current training step
        """
        for result in results:
            self.stats['total_evals'] += 1

            # Extract data
            # For validation examples, we need the full prompt/response
            # The result has: input_text, expected, predicted, match
            prompt = result.input_text
            golden = result.expected
            initial_answer = result.predicted

            # Generate self-correction examples
            examples = self.pipeline.generate_from_qa_pair(
                prompt=prompt,
                golden_answer=golden,
                initial_answer=initial_answer
            )

            # Track stats
            if len(examples) == 1:  # Correct first try
                self.stats['correct_first_try'] += 1
            else:  # Needed correction
                self.stats['needed_correction'] += 1

            # Add to pending
            self.pending_examples.extend(examples)

        # Check if we should generate a file
        if self._should_generate_file(step):
            self._generate_training_file(step)

    def _should_generate_file(self, step: int) -> bool:
        """Determine if we should generate a training file."""
        # Primary trigger: hit max examples (auto-drop!)
        if len(self.pending_examples) >= self.max_examples:
            return True

        # Optional secondary trigger: step interval (if configured)
        if self.generation_interval is not None:
            if step - self.last_generation_step >= self.generation_interval:
                # Only generate if we have at least SOME examples
                if len(self.pending_examples) > 0:
                    return True

        return False

    def _generate_training_file(self, step: int):
        """Generate training file from pending examples."""
        if not self.pending_examples:
            return

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"self_correction_step{step}_{timestamp}.jsonl"
        filepath = self.output_dir / filename

        # Write examples
        with open(filepath, 'w') as f:
            for example in self.pending_examples:
                # Convert to standard training format
                training_example = {
                    "prompt": example['prompt'],
                    "response": example['response']
                }
                f.write(json.dumps(training_example) + '\n')

        num_examples = len(self.pending_examples)
        self.stats['examples_generated'] += num_examples
        self.stats['files_generated'] += 1

        print(f"\n🔄 Self-Correction: Auto-dropped {num_examples} examples → {filepath}")

        # Save metadata
        meta_filepath = filepath.with_suffix('.json')
        metadata = {
            'step': step,
            'timestamp': timestamp,
            'num_examples': num_examples,
            'breakdown': self._get_example_breakdown(),
            'stats': self.stats.copy()
        }
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Auto-queue if enabled
        if self.auto_queue:
            self._queue_for_training(filepath)

        # Reset
        self.pending_examples = []
        self.last_generation_step = step

    def _get_example_breakdown(self) -> Dict:
        """Get breakdown of example types."""
        breakdown = {
            'self_evaluations_correct': 0,
            'self_evaluations_incorrect': 0,
            'corrections': 0
        }

        for ex in self.pending_examples:
            ex_type = ex.get('type', 'self_evaluation')
            if ex_type == 'self_evaluation':
                if ex.get('correct', False):
                    breakdown['self_evaluations_correct'] += 1
                else:
                    breakdown['self_evaluations_incorrect'] += 1
            elif ex_type == 'correction':
                breakdown['corrections'] += 1

        return breakdown

    def _queue_for_training(self, filepath: Path):
        """Add generated file to training queue."""
        # Copy to queue
        queue_file = self.queue_dir / filepath.name

        # Copy file
        import shutil
        shutil.copy(filepath, queue_file)

        print(f"   → Queued for training: {queue_file}")

    def get_stats(self) -> Dict:
        """Get generation statistics."""
        return self.stats.copy()

    def force_generate(self, step: int):
        """Force generation even if interval/minimum not met."""
        if self.pending_examples:
            self._generate_training_file(step)


# Convenience function for integration
def create_self_correction_monitor(
    enable: bool = True,
    **kwargs
) -> AutoSelfCorrectionGenerator:
    """
    Create self-correction monitor for training integration.

    Args:
        enable: If False, returns None (disabled)
        **kwargs: Passed to AutoSelfCorrectionGenerator

    Returns:
        AutoSelfCorrectionGenerator or None
    """
    if not enable:
        return None

    return AutoSelfCorrectionGenerator(**kwargs)
