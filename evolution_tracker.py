#!/usr/bin/env python3
"""
Learning Evolution Tracker

Captures model predictions at regular intervals during training
to track learning progress on specific examples.

This enables viewing:
- Which examples learn fast vs slow
- When each example "gets it" (low loss)
- Learning curves for any example
- Regression detection
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class EvolutionSnapshot:
    """Single prediction snapshot at a specific training step"""
    snapshot_id: str
    training_step: int
    timestamp: str
    model_version: str
    examples: List[Dict[str, Any]]
    summary: Dict[str, Any]

class EvolutionTracker:
    """
    Tracks model evolution during training
    """

    def __init__(self, base_dir: Path, dataset_name: str):
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.evolution_dir = self.base_dir / "data" / "evolution_snapshots" / dataset_name

        # Create directories
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot schedule
        self.snapshot_schedule = self._create_snapshot_schedule()

        # Tracking state
        self.examples_data = {}  # Cache of example data
        self.last_snapshot_step = -1

        print(f"📊 Evolution Tracker initialized for: {dataset_name}")
        print(f"   Snapshots: {self.evolution_dir}")

    def _create_snapshot_schedule(self) -> List[int]:
        """
        Create schedule of when to take snapshots

        Dense early, sparse later:
        - 0 (baseline before training)
        - Early: 10, 25, 50, 100
        - Mid: 250, 500, 750, 1000
        - Late: 1500, 2000, 2500, 3000
        - Every 1000 after that
        """
        schedule = [
            0,  # Baseline
            # Early dense sampling
            10, 25, 50, 100, 150, 200, 250,
            # Mid sampling
            500, 750, 1000,
            # Late sampling
            1500, 2000, 2500, 3000, 4000, 5000,
            # Very late
            7500, 10000, 15000, 20000
        ]

        return sorted(schedule)

    def should_snapshot(self, current_step: int) -> bool:
        """Check if we should take a snapshot at this step"""
        in_schedule = current_step in self.snapshot_schedule
        not_last = current_step != self.last_snapshot_step
        print(f"🔍 should_snapshot({current_step}): in_schedule={in_schedule}, not_last={not_last}, last_snap={self.last_snapshot_step}")

        if current_step in self.snapshot_schedule:
            if current_step != self.last_snapshot_step:
                print(f"✅ Snapshot approved for step {current_step}")
                return True
        # Also snapshot every 1000 steps after 20k
        if current_step > 20000 and current_step % 1000 == 0:
            if current_step != self.last_snapshot_step:
                print(f"✅ Snapshot approved for step {current_step} (late schedule)")
                return True
        print(f"❌ Snapshot skipped for step {current_step}")
        return False

    def capture_snapshot(
        self,
        model,
        tokenizer,
        examples: List[Dict],
        current_step: int,
        model_version: str = "training",
        max_examples: int = 100
    ) -> Optional[str]:
        """
        Capture model predictions on examples at current training step

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            examples: List of training examples (dicts with 'messages')
            current_step: Current training step
            model_version: Version identifier
            max_examples: Max examples to snapshot (for performance)

        Returns:
            Snapshot ID if captured, None if skipped
        """
        if not self.should_snapshot(current_step):
            return None

        print(f"📸 Capturing evolution snapshot at step {current_step}")

        # Limit examples for performance
        sample_examples = examples[:max_examples]

        # Collect predictions
        snapshot_examples = []
        total_loss = 0
        correct_count = 0

        model.eval()  # Set to eval mode
        with torch.no_grad():
            for idx, example in enumerate(sample_examples):
                try:
                    result = self._evaluate_example(
                        model, tokenizer, example, current_step, idx
                    )
                    snapshot_examples.append(result)

                    total_loss += result.get('loss', 0)
                    if result.get('exact_match', False):
                        correct_count += 1

                except Exception as e:
                    print(f"⚠️  Error evaluating example {idx}: {e}")
                    continue

        model.train()  # Back to training mode

        # Create summary
        summary = {
            "avg_loss": total_loss / len(snapshot_examples) if snapshot_examples else 0,
            "accuracy": correct_count / len(snapshot_examples) if snapshot_examples else 0,
            "total_examples": len(snapshot_examples),
            "correct": correct_count,
            "incorrect": len(snapshot_examples) - correct_count
        }

        # Create snapshot
        snapshot_id = f"{self.dataset_name}_step_{current_step:06d}"
        timestamp = datetime.now().isoformat()

        snapshot = EvolutionSnapshot(
            snapshot_id=snapshot_id,
            training_step=current_step,
            timestamp=timestamp,
            model_version=model_version,
            examples=snapshot_examples,
            summary=summary
        )

        # Save snapshot
        snapshot_file = self.evolution_dir / f"step_{current_step:06d}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2)

        self.last_snapshot_step = current_step

        print(f"✓ Snapshot saved: {snapshot_file.name}")
        print(f"   Accuracy: {summary['accuracy']:.1%} ({correct_count}/{len(snapshot_examples)})")
        print(f"   Avg Loss: {summary['avg_loss']:.3f}")

        return snapshot_id

    def _evaluate_example(
        self,
        model,
        tokenizer,
        example: Dict,
        step: int,
        example_idx: int
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single example

        Returns dict with prediction, loss, etc.
        """
        # Extract input/output from messages format
        messages = example.get('messages', [])
        if len(messages) < 2:
            return {"error": "Invalid example format"}

        input_text = messages[0].get('content', '')
        expected_output = messages[1].get('content', '')

        # Create prompt
        prompt = f"{input_text}"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the new tokens (remove prompt)
        model_output = generated[len(prompt):].strip()

        # Calculate simple loss (negative log likelihood approximation)
        # For actual loss, we'd need to run forward pass
        # For now, use simple string similarity as proxy
        loss = self._calculate_simple_loss(model_output, expected_output)

        # Check if exact match
        exact_match = model_output.strip().lower() == expected_output.strip().lower()

        return {
            "example_id": f"ex_{example_idx:04d}",
            "step": step,
            "input": input_text[:200],  # Truncate for storage
            "expected_output": expected_output[:200],
            "model_output": model_output[:200],
            "loss": loss,
            "exact_match": exact_match,
            "output_length": len(model_output),
            "similarity": self._calculate_similarity(model_output, expected_output)
        }

    def _calculate_simple_loss(self, prediction: str, target: str) -> float:
        """
        Calculate simple loss based on string similarity
        (Proxy for actual cross-entropy loss)
        """
        if not target:
            return 5.0

        # Levenshtein distance as proxy
        similarity = self._calculate_similarity(prediction, target)

        # Convert similarity to loss (0=perfect, higher=worse)
        loss = (1 - similarity) * 5.0

        return float(loss)

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings (0-1)
        Uses normalized Levenshtein distance
        """
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # Simple character-level similarity
        str1_lower = str1.lower().strip()
        str2_lower = str2.lower().strip()

        if str1_lower == str2_lower:
            return 1.0

        # Calculate Levenshtein distance
        distance = self._levenshtein(str1_lower, str2_lower)
        max_len = max(len(str1_lower), len(str2_lower))

        if max_len == 0:
            return 1.0

        similarity = 1 - (distance / max_len)
        return max(0.0, min(1.0, similarity))

    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get summary of all snapshots for this dataset
        """
        snapshots = sorted(self.evolution_dir.glob("step_*.json"))

        if not snapshots:
            return {
                "dataset": self.dataset_name,
                "total_snapshots": 0,
                "steps": [],
                "status": "no_data"
            }

        steps = []
        for snapshot_file in snapshots:
            with open(snapshot_file) as f:
                data = json.load(f)
                steps.append({
                    "step": data['training_step'],
                    "accuracy": data['summary']['accuracy'],
                    "loss": data['summary']['avg_loss'],
                    "timestamp": data['timestamp']
                })

        return {
            "dataset": self.dataset_name,
            "total_snapshots": len(snapshots),
            "steps": steps,
            "first_step": steps[0]['step'] if steps else 0,
            "last_step": steps[-1]['step'] if steps else 0,
            "current_accuracy": steps[-1]['accuracy'] if steps else 0,
            "status": "tracking"
        }

    def get_example_evolution(self, example_id: str) -> List[Dict]:
        """
        Get evolution of a specific example across all snapshots
        """
        evolution = []

        snapshots = sorted(self.evolution_dir.glob("step_*.json"))
        for snapshot_file in snapshots:
            with open(snapshot_file) as f:
                data = json.load(f)
                # Find this example in the snapshot
                for ex in data['examples']:
                    if ex.get('example_id') == example_id:
                        evolution.append({
                            "step": data['training_step'],
                            "loss": ex.get('loss'),
                            "output": ex.get('model_output'),
                            "exact_match": ex.get('exact_match'),
                            "similarity": ex.get('similarity')
                        })
                        break

        return evolution

# Convenience function for integration
def create_tracker(base_dir: str, dataset_name: str) -> EvolutionTracker:
    """Create and return an evolution tracker"""
    return EvolutionTracker(Path(base_dir), dataset_name)
