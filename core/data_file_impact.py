#!/usr/bin/env python3
"""
Data File Impact Tracker

Tracks the impact of individual training files on model performance.
Records before/after metrics to determine which files help vs hurt.

Usage:
    tracker = DataFileImpactTracker(status_dir)
    tracker.start_file("train_data.jsonl", step=1000, loss=1.5, accuracy=0.3)
    # ... training happens ...
    tracker.finish_file("train_data.jsonl", step=1500, loss=1.2, accuracy=0.4)

Output files:
    - status/data_file_impact.jsonl: Per-file impact records (append-only)
    - status/data_file_summary.json: Aggregated summary stats

Impact Score Calculation:
    - Positive: file improved metrics (loss decreased, accuracy increased)
    - Negative: file hurt metrics (loss increased, accuracy decreased)
    - Magnitude indicates strength of impact
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class FileTrainingStart:
    """Snapshot of metrics when training starts on a file."""
    filename: str
    start_step: int
    start_loss: Optional[float]
    start_accuracy: Optional[float]
    start_val_loss: Optional[float]
    start_timestamp: str
    examples_count: Optional[int] = None


@dataclass
class FileImpactRecord:
    """Complete record of a file's training impact."""
    filename: str
    start_step: int
    end_step: int
    steps_trained: int

    # Before metrics
    loss_before: Optional[float]
    accuracy_before: Optional[float]
    val_loss_before: Optional[float]

    # After metrics
    loss_after: Optional[float]
    accuracy_after: Optional[float]
    val_loss_after: Optional[float]

    # Computed impact
    loss_delta: Optional[float]  # Negative = improvement
    accuracy_delta: Optional[float]  # Positive = improvement
    val_loss_delta: Optional[float]  # Negative = improvement
    impact_score: float  # Composite score: positive = good, negative = bad

    # Metadata
    examples_count: Optional[int]
    start_timestamp: str
    end_timestamp: str

    # Classification
    classification: str  # 'positive', 'negative', 'neutral'


class DataFileImpactTracker:
    """
    Tracks the impact of training files on model performance.

    Call start_file() when training begins on a file.
    Call finish_file() when training completes on that file.

    Results are written to status/data_file_impact.jsonl.
    """

    def __init__(self, status_dir: Path):
        self.status_dir = Path(status_dir)
        self.status_dir.mkdir(parents=True, exist_ok=True)

        self.impact_file = self.status_dir / 'data_file_impact.jsonl'
        self.summary_file = self.status_dir / 'data_file_summary.json'

        # Track files currently being trained
        self._active_files: Dict[str, FileTrainingStart] = {}
        self._lock = Lock()

        # Summary stats
        self._total_files = 0
        self._positive_count = 0
        self._negative_count = 0
        self._neutral_count = 0

        # Load existing summary if present
        self._load_summary()

        logger.info(f"DataFileImpactTracker initialized, output: {self.impact_file}")

    def _load_summary(self):
        """Load existing summary stats."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file) as f:
                    data = json.load(f)
                    self._total_files = data.get('total_files_analyzed', 0)
                    self._positive_count = data.get('positive_impact_files', 0)
                    self._negative_count = data.get('negative_impact_files', 0)
                    self._neutral_count = data.get('neutral_impact_files', 0)
            except Exception as e:
                logger.warning(f"Could not load summary: {e}")

    def start_file(
        self,
        filename: str,
        step: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        val_loss: Optional[float] = None,
        examples_count: Optional[int] = None
    ):
        """
        Record the start of training on a file.

        Args:
            filename: Name of the training file
            step: Current training step
            loss: Current training loss
            accuracy: Current accuracy (if available)
            val_loss: Current validation loss (if available)
            examples_count: Number of examples in the file
        """
        with self._lock:
            start_record = FileTrainingStart(
                filename=filename,
                start_step=step,
                start_loss=loss,
                start_accuracy=accuracy,
                start_val_loss=val_loss,
                start_timestamp=datetime.now().isoformat(),
                examples_count=examples_count
            )
            self._active_files[filename] = start_record
            logger.debug(f"Started tracking: {filename} at step {step}, loss={loss}")

    def finish_file(
        self,
        filename: str,
        step: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        val_loss: Optional[float] = None
    ) -> Optional[FileImpactRecord]:
        """
        Record the end of training on a file and compute impact.

        Args:
            filename: Name of the training file
            step: Current training step
            loss: Current training loss
            accuracy: Current accuracy (if available)
            val_loss: Current validation loss (if available)

        Returns:
            FileImpactRecord with computed impact, or None if file wasn't tracked
        """
        with self._lock:
            if filename not in self._active_files:
                logger.warning(f"finish_file called for untracked file: {filename}")
                return None

            start = self._active_files.pop(filename)

            # Compute deltas (None if either value missing)
            loss_delta = None
            if loss is not None and start.start_loss is not None:
                loss_delta = loss - start.start_loss  # Negative = improved

            accuracy_delta = None
            if accuracy is not None and start.start_accuracy is not None:
                accuracy_delta = accuracy - start.start_accuracy  # Positive = improved

            val_loss_delta = None
            if val_loss is not None and start.start_val_loss is not None:
                val_loss_delta = val_loss - start.start_val_loss  # Negative = improved

            # Compute composite impact score
            # Score > 0 means file helped, < 0 means file hurt
            impact_score = self._compute_impact_score(
                loss_delta, accuracy_delta, val_loss_delta
            )

            # Classify the impact
            if impact_score > 0.01:
                classification = 'positive'
                self._positive_count += 1
            elif impact_score < -0.01:
                classification = 'negative'
                self._negative_count += 1
            else:
                classification = 'neutral'
                self._neutral_count += 1

            self._total_files += 1

            # Create the record
            record = FileImpactRecord(
                filename=filename,
                start_step=start.start_step,
                end_step=step,
                steps_trained=step - start.start_step,
                loss_before=start.start_loss,
                accuracy_before=start.start_accuracy,
                val_loss_before=start.start_val_loss,
                loss_after=loss,
                accuracy_after=accuracy,
                val_loss_after=val_loss,
                loss_delta=loss_delta,
                accuracy_delta=accuracy_delta,
                val_loss_delta=val_loss_delta,
                impact_score=round(impact_score, 4),
                examples_count=start.examples_count,
                start_timestamp=start.start_timestamp,
                end_timestamp=datetime.now().isoformat(),
                classification=classification
            )

            # Write to file
            self._write_record(record)
            self._update_summary()

            logger.info(
                f"File impact: {filename} score={impact_score:.4f} ({classification}), "
                f"loss: {start.start_loss} -> {loss}, "
                f"steps: {start.start_step} -> {step}"
            )

            return record

    def _compute_impact_score(
        self,
        loss_delta: Optional[float],
        accuracy_delta: Optional[float],
        val_loss_delta: Optional[float]
    ) -> float:
        """
        Compute a composite impact score.

        Positive score = file helped model
        Negative score = file hurt model

        Weights:
        - Loss decrease is good (weighted 0.5)
        - Accuracy increase is good (weighted 0.3)
        - Val loss decrease is good (weighted 0.2)
        """
        score = 0.0
        weights_used = 0.0

        if loss_delta is not None:
            # Negate because lower loss is better
            score -= loss_delta * 0.5
            weights_used += 0.5

        if accuracy_delta is not None:
            # Higher accuracy is better
            score += accuracy_delta * 0.3
            weights_used += 0.3

        if val_loss_delta is not None:
            # Negate because lower val_loss is better
            score -= val_loss_delta * 0.2
            weights_used += 0.2

        # Normalize by weights used
        if weights_used > 0:
            score /= weights_used

        return score

    def _write_record(self, record: FileImpactRecord):
        """Append a record to the JSONL file."""
        try:
            with open(self.impact_file, 'a') as f:
                f.write(json.dumps(asdict(record)) + '\n')
        except Exception as e:
            logger.error(f"Failed to write impact record: {e}")

    def _update_summary(self):
        """Update the summary JSON file."""
        summary = {
            'last_updated': datetime.now().isoformat(),
            'total_files_analyzed': self._total_files,
            'positive_impact_files': self._positive_count,
            'negative_impact_files': self._negative_count,
            'neutral_impact_files': self._neutral_count,
            'positive_ratio': self._positive_count / self._total_files if self._total_files > 0 else 0,
            'top_positive': self._get_top_files('positive', 5),
            'top_negative': self._get_top_files('negative', 5)
        }

        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write summary: {e}")

    def _get_top_files(self, classification: str, limit: int = 5) -> List[Dict]:
        """Get top files by impact score for a given classification."""
        if not self.impact_file.exists():
            return []

        try:
            records = []
            with open(self.impact_file) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get('classification') == classification:
                            records.append(rec)

            # Sort by absolute impact score
            records.sort(key=lambda x: abs(x.get('impact_score', 0)), reverse=True)

            return [
                {
                    'filename': r['filename'],
                    'impact_score': r['impact_score'],
                    'loss_delta': r.get('loss_delta'),
                    'steps_trained': r.get('steps_trained')
                }
                for r in records[:limit]
            ]
        except Exception as e:
            logger.error(f"Failed to read top files: {e}")
            return []

    def get_recent_impacts(self, limit: int = 20) -> List[Dict]:
        """Get the most recent impact records."""
        if not self.impact_file.exists():
            return []

        try:
            records = []
            with open(self.impact_file) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records[-limit:]
        except Exception as e:
            logger.error(f"Failed to read recent impacts: {e}")
            return []

    def cancel_file(self, filename: str):
        """Cancel tracking for a file (e.g., if training was skipped)."""
        with self._lock:
            if filename in self._active_files:
                del self._active_files[filename]
                logger.debug(f"Cancelled tracking for: {filename}")


# Singleton instance for global access
_tracker_instance: Optional[DataFileImpactTracker] = None


def get_tracker(status_dir: Optional[Path] = None) -> DataFileImpactTracker:
    """Get or create the global tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        if status_dir is None:
            from core.paths import get_status_dir
            status_dir = get_status_dir()
        _tracker_instance = DataFileImpactTracker(status_dir)
    return _tracker_instance


if __name__ == '__main__':
    # Test the tracker
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = DataFileImpactTracker(Path(tmpdir))

        # Simulate training on a file that helps
        tracker.start_file("good_data.jsonl", step=1000, loss=1.5, accuracy=0.3)
        result = tracker.finish_file("good_data.jsonl", step=1500, loss=1.2, accuracy=0.4)
        print(f"Good file: {result.classification}, score={result.impact_score}")

        # Simulate training on a file that hurts
        tracker.start_file("bad_data.jsonl", step=1500, loss=1.2, accuracy=0.4)
        result = tracker.finish_file("bad_data.jsonl", step=2000, loss=1.6, accuracy=0.35)
        print(f"Bad file: {result.classification}, score={result.impact_score}")

        # Check the files
        print(f"\nImpact file: {tracker.impact_file}")
        with open(tracker.impact_file) as f:
            for line in f:
                print(line.strip())

        print(f"\nSummary file: {tracker.summary_file}")
        with open(tracker.summary_file) as f:
            print(json.dumps(json.load(f), indent=2))
