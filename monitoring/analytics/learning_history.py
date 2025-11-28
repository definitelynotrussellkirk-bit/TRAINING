#!/usr/bin/env python3
"""
Learning History - Time series database for training metrics.

Stores historical data for all learning metrics in append-only format.
Enables visualization of learning dynamics over time.

Usage:
    # Append current metrics
    python3 learning_history.py --append

    # Query history
    python3 learning_history.py --query --last 100

    # Export for visualization
    python3 learning_history.py --export --format csv

Output:
    status/learning_history.jsonl - Append-only log of all snapshots
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LearningSnapshot:
    """Single point-in-time snapshot of learning state."""
    step: int
    timestamp: str

    # Core metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Skill-specific accuracy
    skill_scores: Dict[str, float] = field(default_factory=dict)

    # Weight space metrics
    total_weight_drift: Optional[float] = None
    layer_drifts: List[float] = field(default_factory=list)
    layer_norms: List[float] = field(default_factory=list)

    # Representation metrics
    geometry_drift: Optional[float] = None

    # Calibration metrics
    expected_calibration_error: Optional[float] = None
    mean_confidence: Optional[float] = None

    # Task performance
    hard_examples_correct: Optional[int] = None
    hard_examples_total: Optional[int] = None

    # Metadata
    checkpoint_path: Optional[str] = None
    data_files_trained: List[str] = field(default_factory=list)


class LearningHistory:
    """
    Append-only time series database for learning metrics.

    Stores data in JSONL format for durability and streaming reads.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.history_file = self.base_dir / "status" / "learning_history.jsonl"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, snapshot: LearningSnapshot) -> None:
        """Append a snapshot to the history."""
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(asdict(snapshot)) + '\n')
        logger.info(f"Appended snapshot for step {snapshot.step}")

    def read_all(self) -> List[LearningSnapshot]:
        """Read all snapshots from history."""
        snapshots = []
        if self.history_file.exists():
            with open(self.history_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        snapshots.append(LearningSnapshot(**data))
        return snapshots

    def read_last_n(self, n: int) -> List[LearningSnapshot]:
        """Read last N snapshots (memory efficient for large files)."""
        snapshots = []
        if self.history_file.exists():
            # Read file backwards efficiently
            with open(self.history_file, 'rb') as f:
                # Seek to end
                f.seek(0, 2)
                file_size = f.tell()

                # Read chunks from end
                chunk_size = 8192
                buffer = b''
                position = file_size
                lines_found = []

                while position > 0 and len(lines_found) < n:
                    read_size = min(chunk_size, position)
                    position -= read_size
                    f.seek(position)
                    chunk = f.read(read_size)
                    buffer = chunk + buffer

                    # Split into lines
                    while b'\n' in buffer and len(lines_found) < n:
                        line, buffer = buffer.rsplit(b'\n', 1)
                        if line.strip():
                            lines_found.append(line.decode('utf-8'))

                # Handle remaining buffer
                if buffer.strip() and len(lines_found) < n:
                    lines_found.append(buffer.decode('utf-8'))

                # Reverse to get chronological order
                lines_found.reverse()

                for line in lines_found[-n:]:
                    try:
                        data = json.loads(line)
                        snapshots.append(LearningSnapshot(**data))
                    except json.JSONDecodeError:
                        continue

        return snapshots

    def read_range(self, start_step: int, end_step: int) -> List[LearningSnapshot]:
        """Read snapshots within a step range."""
        snapshots = []
        if self.history_file.exists():
            with open(self.history_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        step = data.get('step', 0)
                        if start_step <= step <= end_step:
                            snapshots.append(LearningSnapshot(**data))
        return snapshots

    def get_latest(self) -> Optional[LearningSnapshot]:
        """Get the most recent snapshot."""
        last = self.read_last_n(1)
        return last[0] if last else None

    def export_csv(self, output_path: Path) -> None:
        """Export history to CSV for external tools."""
        import csv

        snapshots = self.read_all()
        if not snapshots:
            logger.warning("No snapshots to export")
            return

        # Flatten nested fields
        rows = []
        for s in snapshots:
            row = {
                'step': s.step,
                'timestamp': s.timestamp,
                'train_loss': s.train_loss,
                'val_loss': s.val_loss,
                'learning_rate': s.learning_rate,
                'total_weight_drift': s.total_weight_drift,
                'geometry_drift': s.geometry_drift,
                'ece': s.expected_calibration_error,
                'hard_examples_accuracy': (
                    s.hard_examples_correct / s.hard_examples_total
                    if s.hard_examples_total else None
                ),
            }
            # Add skill scores
            for skill, score in s.skill_scores.items():
                row[f'skill_{skill}'] = score
            rows.append(row)

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            # Ensure all rows have same fields
            all_fields = set()
            for row in rows:
                all_fields.update(row.keys())
            fieldnames = sorted(all_fields)

            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Exported {len(rows)} snapshots to {output_path}")

    def collect_current_metrics(self) -> LearningSnapshot:
        """Collect current metrics from all status files."""
        snapshot = LearningSnapshot(
            step=0,
            timestamp=datetime.now().isoformat()
        )

        # Training status
        training_status = self.base_dir / "status" / "training_status.json"
        if training_status.exists():
            with open(training_status) as f:
                data = json.load(f)
                snapshot.step = data.get('current_step', 0)
                snapshot.train_loss = data.get('loss')
                snapshot.val_loss = data.get('validation_loss')
                snapshot.learning_rate = data.get('learning_rate')
                snapshot.checkpoint_path = data.get('last_checkpoint')

        # Curriculum eval (skill scores)
        curriculum_eval = self.base_dir / "status" / "curriculum_eval.json"
        if curriculum_eval.exists():
            with open(curriculum_eval) as f:
                data = json.load(f)
                if 'results' in data:
                    for result in data['results']:
                        skill = result.get('skill', 'unknown')
                        level = result.get('level', 0)
                        acc = result.get('accuracy', 0)
                        snapshot.skill_scores[f"{skill}_level{level}"] = acc

        # Layer drift
        layer_drift = self.base_dir / "status" / "layer_drift.json"
        if layer_drift.exists():
            with open(layer_drift) as f:
                data = json.load(f)
                snapshot.total_weight_drift = data.get('total_relative_change')
                if 'layers' in data:
                    snapshot.layer_drifts = [
                        l.get('relative_change', 0) for l in data['layers']
                    ]

        # Parameter stability (norms)
        param_stability = self.base_dir / "status" / "parameter_stability.json"
        if param_stability.exists():
            with open(param_stability) as f:
                data = json.load(f)
                if 'layers' in data:
                    snapshot.layer_norms = [
                        l.get('weight_norm', 0) for l in data['layers']
                    ]

        # Hard example board
        hard_examples = self.base_dir / "status" / "hard_example_board.json"
        if hard_examples.exists():
            with open(hard_examples) as f:
                data = json.load(f)
                snapshot.hard_examples_correct = data.get('correct', 0)
                snapshot.hard_examples_total = data.get('total', 0)

        return snapshot


def main():
    parser = argparse.ArgumentParser(description="Learning History Manager")
    parser.add_argument('--base-dir', default=None,
                       help='Base directory')
    parser.add_argument('--append', action='store_true',
                       help='Append current metrics to history')
    parser.add_argument('--query', action='store_true',
                       help='Query history')
    parser.add_argument('--last', type=int, default=10,
                       help='Number of recent snapshots to show')
    parser.add_argument('--export', action='store_true',
                       help='Export to CSV')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                       help='Export format')

    args = parser.parse_args()

    history = LearningHistory(args.base_dir)

    if args.append:
        snapshot = history.collect_current_metrics()
        history.append(snapshot)
        print(f"Appended snapshot for step {snapshot.step}")
        print(f"  train_loss: {snapshot.train_loss}")
        print(f"  val_loss: {snapshot.val_loss}")
        print(f"  skills: {snapshot.skill_scores}")

    elif args.query:
        snapshots = history.read_last_n(args.last)
        print(f"Last {len(snapshots)} snapshots:")
        for s in snapshots:
            print(f"  Step {s.step}: loss={s.train_loss:.4f}" if s.train_loss else f"  Step {s.step}")

    elif args.export:
        output_path = Path(args.base_dir) / "status" / f"learning_history.{args.format}"
        if args.format == 'csv':
            history.export_csv(output_path)
        else:
            # JSON export
            snapshots = history.read_all()
            with open(output_path, 'w') as f:
                json.dump([asdict(s) for s in snapshots], f, indent=2)
            print(f"Exported to {output_path}")

    else:
        # Default: show summary
        snapshots = history.read_all()
        print(f"Learning History: {len(snapshots)} snapshots")
        if snapshots:
            first = snapshots[0]
            last = snapshots[-1]
            print(f"  Range: step {first.step} to {last.step}")
            print(f"  Time: {first.timestamp} to {last.timestamp}")


if __name__ == "__main__":
    main()
