#!/usr/bin/env python3
"""
Flagged Examples Tracker

Tracks training examples that need review:
- Mismatches (model output != golden answer)
- High loss examples (loss > threshold)
- Manual flags
- Error cases

Helps debug training issues and improve data quality.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class FlaggedExample:
    """A single flagged training example."""
    step: int
    timestamp: str
    prompt: str
    golden_answer: str
    model_output: str
    loss: float
    matches: bool
    flag_reason: str  # "mismatch", "high_loss", "error", "manual"
    notes: Optional[str] = None
    current_file: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class FlaggedExamplesTracker:
    """Track and manage flagged training examples."""

    def __init__(self, base_dir: Path, max_flagged: int = 1000):
        """
        Initialize flagged examples tracker.

        Args:
            base_dir: Base directory for training system
            max_flagged: Maximum number of flagged examples to keep
        """
        self.base_dir = Path(base_dir)
        self.flagged_dir = self.base_dir / "flagged_examples"
        self.flagged_dir.mkdir(parents=True, exist_ok=True)

        self.flagged_file = self.flagged_dir / "flagged_examples.json"
        self.max_flagged = max_flagged

        # Load existing flagged examples
        self.flagged_examples: List[FlaggedExample] = []
        self.load()

        # Thresholds
        self.high_loss_threshold = 3.0  # Flag examples with loss > 3.0

        print(f"📋 Flagged Examples Tracker initialized: {len(self.flagged_examples)} existing flags")

    def load(self):
        """Load flagged examples from disk."""
        if not self.flagged_file.exists():
            return

        try:
            with open(self.flagged_file, 'r') as f:
                data = json.load(f)
                self.flagged_examples = [
                    FlaggedExample(**item) for item in data
                ]
        except Exception as e:
            print(f"⚠️  Could not load flagged examples: {e}")
            self.flagged_examples = []

    def save(self):
        """Save flagged examples to disk."""
        try:
            with open(self.flagged_file, 'w') as f:
                json.dump(
                    [ex.to_dict() for ex in self.flagged_examples],
                    f,
                    indent=2
                )
        except Exception as e:
            print(f"⚠️  Could not save flagged examples: {e}")

    def flag_example(
        self,
        step: int,
        prompt: str,
        golden_answer: str,
        model_output: str,
        loss: float,
        matches: bool,
        reason: str,
        notes: Optional[str] = None,
        current_file: Optional[str] = None
    ):
        """
        Flag an example for review.

        Args:
            step: Training step number
            prompt: Input prompt
            golden_answer: Expected output
            model_output: Model's actual output
            loss: Loss value for this example
            matches: Whether output matches golden answer
            reason: Why this was flagged ("mismatch", "high_loss", "error", "manual")
            notes: Optional additional notes
            current_file: Current training file
        """
        flagged = FlaggedExample(
            step=step,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            golden_answer=golden_answer,
            model_output=model_output,
            loss=loss,
            matches=matches,
            flag_reason=reason,
            notes=notes,
            current_file=current_file
        )

        self.flagged_examples.append(flagged)

        # Keep only most recent max_flagged examples
        if len(self.flagged_examples) > self.max_flagged:
            self.flagged_examples = self.flagged_examples[-self.max_flagged:]

        self.save()

    def auto_flag_if_needed(
        self,
        step: int,
        prompt: str,
        golden_answer: str,
        model_output: str,
        loss: float,
        matches: bool,
        current_file: Optional[str] = None
    ):
        """
        Automatically flag example if it meets criteria.

        Flags if:
        - Model output doesn't match golden answer
        - Loss is above threshold

        Args:
            step: Training step
            prompt: Input prompt
            golden_answer: Expected output
            model_output: Model's output
            loss: Loss value
            matches: Whether output matches
            current_file: Current training file
        """
        # Flag mismatches
        if not matches:
            self.flag_example(
                step=step,
                prompt=prompt,
                golden_answer=golden_answer,
                model_output=model_output,
                loss=loss,
                matches=matches,
                reason="mismatch",
                notes="Model output does not match golden answer",
                current_file=current_file
            )
            return

        # Flag high loss (even if matches)
        if loss > self.high_loss_threshold:
            self.flag_example(
                step=step,
                prompt=prompt,
                golden_answer=golden_answer,
                model_output=model_output,
                loss=loss,
                matches=matches,
                reason="high_loss",
                notes=f"Loss ({loss:.3f}) exceeds threshold ({self.high_loss_threshold})",
                current_file=current_file
            )

    def get_all_flagged(self) -> List[Dict]:
        """Get all flagged examples."""
        return [ex.to_dict() for ex in self.flagged_examples]

    def get_flagged_by_reason(self, reason: str) -> List[Dict]:
        """Get flagged examples filtered by reason."""
        return [
            ex.to_dict()
            for ex in self.flagged_examples
            if ex.flag_reason == reason
        ]

    def get_statistics(self) -> Dict:
        """Get statistics about flagged examples."""
        if not self.flagged_examples:
            return {
                "total": 0,
                "by_reason": {},
                "avg_loss": 0.0,
                "match_rate": 0.0
            }

        by_reason = {}
        for ex in self.flagged_examples:
            by_reason[ex.flag_reason] = by_reason.get(ex.flag_reason, 0) + 1

        total_loss = sum(ex.loss for ex in self.flagged_examples)
        matches = sum(1 for ex in self.flagged_examples if ex.matches)

        return {
            "total": len(self.flagged_examples),
            "by_reason": by_reason,
            "avg_loss": total_loss / len(self.flagged_examples),
            "match_rate": matches / len(self.flagged_examples),
            "oldest_step": self.flagged_examples[0].step if self.flagged_examples else 0,
            "newest_step": self.flagged_examples[-1].step if self.flagged_examples else 0
        }

    def clear_all(self):
        """Clear all flagged examples."""
        self.flagged_examples = []
        self.save()

    def clear_by_reason(self, reason: str):
        """Clear flagged examples by reason."""
        self.flagged_examples = [
            ex for ex in self.flagged_examples
            if ex.flag_reason != reason
        ]
        self.save()


def create_flagged_tracker(base_dir: Path, max_flagged: int = 1000) -> FlaggedExamplesTracker:
    """
    Factory function to create a flagged examples tracker.

    Args:
        base_dir: Base directory for training system
        max_flagged: Maximum flagged examples to keep

    Returns:
        FlaggedExamplesTracker instance
    """
    return FlaggedExamplesTracker(base_dir=base_dir, max_flagged=max_flagged)


if __name__ == "__main__":
    print("Flagged Examples Tracker - Test Mode")

    # Quick test
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = create_flagged_tracker(Path(tmpdir))

        # Flag some examples
        tracker.flag_example(
            step=100,
            prompt="What is 2+2?",
            golden_answer="4",
            model_output="5",
            loss=2.5,
            matches=False,
            reason="mismatch"
        )

        tracker.flag_example(
            step=200,
            prompt="What is the capital of France?",
            golden_answer="Paris",
            model_output="Paris",
            loss=4.2,
            matches=True,
            reason="high_loss"
        )

        # Get statistics
        stats = tracker.get_statistics()
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(f"\nTotal flagged: {len(tracker.get_all_flagged())}")
