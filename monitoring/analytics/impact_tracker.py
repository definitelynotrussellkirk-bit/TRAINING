#!/usr/bin/env python3
"""
Impact Tracker - Measure the effectiveness of correction training.

Tracks before/after performance to answer: "Did the corrections help?"

The workflow:
1. Before training corrections: Snapshot current hard example accuracy
2. After corrections trained: Compare against snapshot
3. Track improvement trends over time

This creates a feedback signal for the self-improving loop:
- If corrections helped → continue with similar patterns
- If corrections didn't help → adjust strategy

Usage:
    # Take baseline snapshot before training
    python3 impact_tracker.py --snapshot

    # Compare current performance to baseline
    python3 impact_tracker.py --compare

    # View improvement history
    python3 impact_tracker.py --history

    # Full analysis report
    python3 impact_tracker.py --report

Output:
    status/impact_tracker.json - Snapshot history and comparisons
    status/correction_impact.json - Per-correction-file effectiveness
"""

import argparse
import json
import logging
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
class PerformanceSnapshot:
    """A snapshot of model performance at a point in time."""
    timestamp: str
    step: int
    hard_example_accuracy: float
    hard_example_correct: int
    hard_example_total: int
    error_distribution: Dict[str, int]
    checkpoint: Optional[str] = None
    corrections_pending: int = 0  # Files in inbox waiting to train
    corrections_trained: int = 0  # Files trained since last snapshot
    notes: Optional[str] = None


@dataclass
class ImpactComparison:
    """Comparison between two snapshots."""
    baseline_timestamp: str
    current_timestamp: str
    baseline_accuracy: float
    current_accuracy: float
    accuracy_delta: float
    accuracy_improved: bool
    error_changes: Dict[str, int]  # Positive = more errors, negative = fewer
    steps_trained: int
    corrections_trained: int
    effectiveness_score: float  # -1 to 1, where 1 = perfect improvement


@dataclass
class CorrectionFileImpact:
    """Track impact of a specific correction file."""
    filename: str
    queued_at: str
    trained_at: Optional[str] = None
    baseline_accuracy: Optional[float] = None
    post_accuracy: Optional[float] = None
    impact_score: Optional[float] = None
    error_type: Optional[str] = None
    examples_count: int = 0


class ImpactTracker:
    """Track the effectiveness of correction training."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "status"
        self.inbox_dir = self.base_dir / "inbox"
        self.queue_dir = self.base_dir / "queue"

        # Status files
        self.tracker_file = self.status_dir / "impact_tracker.json"
        self.correction_file = self.status_dir / "correction_impact.json"

        self.status_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.snapshots: List[PerformanceSnapshot] = []
        self.corrections: List[CorrectionFileImpact] = []
        self._load()

    def _load(self):
        """Load existing tracker data."""
        if self.tracker_file.exists():
            with open(self.tracker_file) as f:
                data = json.load(f)
                self.snapshots = [
                    PerformanceSnapshot(**s) for s in data.get("snapshots", [])
                ]

        if self.correction_file.exists():
            with open(self.correction_file) as f:
                data = json.load(f)
                self.corrections = [
                    CorrectionFileImpact(**c) for c in data.get("corrections", [])
                ]

    def _save(self):
        """Save tracker data."""
        with open(self.tracker_file, 'w') as f:
            json.dump({
                "snapshots": [asdict(s) for s in self.snapshots],
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

        with open(self.correction_file, 'w') as f:
            json.dump({
                "corrections": [asdict(c) for c in self.corrections],
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current hard example metrics."""
        board_file = self.status_dir / "hard_example_board.json"

        if not board_file.exists():
            return {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "error_types": {},
                "step": 0
            }

        with open(board_file) as f:
            data = json.load(f)

        entries = data.get("entries", [])
        if not entries:
            return {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "error_types": {},
                "step": 0
            }

        latest = entries[-1]
        return {
            "accuracy": latest.get("accuracy", 0.0),
            "correct": latest.get("total_correct", 0),
            "total": latest.get("total", 0),
            "error_types": latest.get("error_types", {}),
            "step": latest.get("step", 0),
            "checkpoint": latest.get("checkpoint", "unknown")
        }

    def get_current_step(self) -> int:
        """Get current training step."""
        status_file = self.status_dir / "training_status.json"
        if status_file.exists():
            with open(status_file) as f:
                data = json.load(f)
                return data.get("current_step", 0)
        return 0

    def count_pending_corrections(self) -> int:
        """Count correction files pending training."""
        count = 0
        for jsonl_file in self.inbox_dir.glob("corrections_*.jsonl"):
            count += 1
        for queue_subdir in ['high', 'normal', 'low']:
            queue_path = self.queue_dir / queue_subdir
            if queue_path.exists():
                for jsonl_file in queue_path.glob("corrections_*.jsonl"):
                    count += 1
        return count

    def take_snapshot(self, notes: Optional[str] = None) -> PerformanceSnapshot:
        """Take a snapshot of current performance."""
        metrics = self.get_current_metrics()
        step = self.get_current_step()

        # Count corrections trained since last snapshot
        corrections_trained = 0
        if self.snapshots:
            last_snap = self.snapshots[-1]
            # Check recently completed queue
            completed_dir = self.queue_dir / "recently_completed"
            if completed_dir.exists():
                for f in completed_dir.glob("corrections_*.jsonl"):
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    last_time = datetime.fromisoformat(last_snap.timestamp)
                    if mtime > last_time:
                        corrections_trained += 1

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            step=step,
            hard_example_accuracy=metrics["accuracy"],
            hard_example_correct=metrics["correct"],
            hard_example_total=metrics["total"],
            error_distribution=metrics["error_types"],
            checkpoint=metrics.get("checkpoint"),
            corrections_pending=self.count_pending_corrections(),
            corrections_trained=corrections_trained,
            notes=notes
        )

        self.snapshots.append(snapshot)
        self._save()

        logger.info(f"Snapshot taken: accuracy={snapshot.hard_example_accuracy:.1%}, "
                   f"step={snapshot.step}")

        return snapshot

    def compare_to_baseline(
        self,
        baseline_idx: int = -2
    ) -> Optional[ImpactComparison]:
        """Compare current performance to a baseline snapshot."""
        if len(self.snapshots) < 2:
            logger.warning("Need at least 2 snapshots to compare")
            return None

        baseline = self.snapshots[baseline_idx]
        current = self.snapshots[-1]

        # Calculate accuracy delta
        delta = current.hard_example_accuracy - baseline.hard_example_accuracy
        improved = delta > 0

        # Calculate error changes
        error_changes = {}
        all_errors = set(baseline.error_distribution.keys()) | set(current.error_distribution.keys())
        for error_type in all_errors:
            before = baseline.error_distribution.get(error_type, 0)
            after = current.error_distribution.get(error_type, 0)
            error_changes[error_type] = after - before  # Positive = worse

        # Calculate effectiveness score (-1 to 1)
        # Based on accuracy improvement normalized by corrections trained
        corrections = current.corrections_trained or 1
        steps_diff = current.step - baseline.step

        if steps_diff > 0:
            # Normalize by training effort
            effectiveness = delta / (0.1 * corrections)  # Expect ~10% improvement per correction batch
            effectiveness = max(-1, min(1, effectiveness))
        else:
            effectiveness = 0.0

        comparison = ImpactComparison(
            baseline_timestamp=baseline.timestamp,
            current_timestamp=current.timestamp,
            baseline_accuracy=baseline.hard_example_accuracy,
            current_accuracy=current.hard_example_accuracy,
            accuracy_delta=delta,
            accuracy_improved=improved,
            error_changes=error_changes,
            steps_trained=steps_diff,
            corrections_trained=current.corrections_trained,
            effectiveness_score=effectiveness
        )

        return comparison

    def track_correction_file(self, filename: str, error_type: Optional[str] = None):
        """Start tracking a correction file."""
        # Count examples
        examples_count = 0
        filepath = self.inbox_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                examples_count = sum(1 for _ in f)

        # Get current baseline
        metrics = self.get_current_metrics()

        correction = CorrectionFileImpact(
            filename=filename,
            queued_at=datetime.now().isoformat(),
            baseline_accuracy=metrics["accuracy"],
            error_type=error_type,
            examples_count=examples_count
        )

        self.corrections.append(correction)
        self._save()

        logger.info(f"Tracking correction file: {filename} ({examples_count} examples)")

    def update_correction_impact(self, filename: str):
        """Update impact for a trained correction file."""
        metrics = self.get_current_metrics()

        for corr in self.corrections:
            if corr.filename == filename and corr.post_accuracy is None:
                corr.trained_at = datetime.now().isoformat()
                corr.post_accuracy = metrics["accuracy"]

                if corr.baseline_accuracy is not None:
                    corr.impact_score = corr.post_accuracy - corr.baseline_accuracy

                self._save()
                logger.info(f"Updated impact for {filename}: "
                           f"{corr.baseline_accuracy:.1%} -> {corr.post_accuracy:.1%}")
                return

    def get_effectiveness_report(self) -> Dict[str, Any]:
        """Generate a comprehensive effectiveness report."""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots for report"}

        # Overall trend
        first = self.snapshots[0]
        latest = self.snapshots[-1]
        overall_delta = latest.hard_example_accuracy - first.hard_example_accuracy

        # Per-correction effectiveness
        effective_corrections = []
        ineffective_corrections = []
        for corr in self.corrections:
            if corr.impact_score is not None:
                if corr.impact_score > 0:
                    effective_corrections.append(corr.filename)
                else:
                    ineffective_corrections.append(corr.filename)

        # Error type improvements
        all_error_types = set()
        for snap in self.snapshots:
            all_error_types.update(snap.error_distribution.keys())

        error_trends = {}
        for error_type in all_error_types:
            first_count = first.error_distribution.get(error_type, 0)
            latest_count = latest.error_distribution.get(error_type, 0)
            error_trends[error_type] = {
                "first": first_count,
                "latest": latest_count,
                "delta": latest_count - first_count,
                "improved": latest_count < first_count
            }

        return {
            "summary": {
                "snapshots_count": len(self.snapshots),
                "first_accuracy": first.hard_example_accuracy,
                "latest_accuracy": latest.hard_example_accuracy,
                "overall_delta": overall_delta,
                "overall_improved": overall_delta > 0,
                "total_steps_trained": latest.step - first.step,
            },
            "corrections": {
                "total_tracked": len(self.corrections),
                "effective": len(effective_corrections),
                "ineffective": len(ineffective_corrections),
                "pending_evaluation": len([c for c in self.corrections if c.impact_score is None])
            },
            "error_trends": error_trends,
            "recommendations": self._generate_recommendations(error_trends)
        }

    def _generate_recommendations(self, error_trends: Dict) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []

        # Find stubborn errors
        for error_type, trend in error_trends.items():
            if trend["delta"] >= 0 and trend["latest"] > 0:
                recommendations.append(
                    f"'{error_type}' not improving - consider more targeted training"
                )

        # Find successes to replicate
        for error_type, trend in error_trends.items():
            if trend["delta"] < 0:
                recommendations.append(
                    f"'{error_type}' improving - current approach working"
                )

        if not recommendations:
            recommendations.append("No clear trends yet - continue monitoring")

        return recommendations

    def show_history(self):
        """Display snapshot history."""
        print("\n" + "=" * 70)
        print("IMPACT TRACKER HISTORY")
        print("=" * 70)

        if not self.snapshots:
            print("No snapshots recorded yet.")
            return

        print(f"\n{'Timestamp':<22} {'Step':>8} {'Accuracy':>10} {'Correct':>8} {'Pending':>8}")
        print("-" * 70)

        for snap in self.snapshots[-10:]:  # Last 10
            ts = snap.timestamp[:19]  # Truncate
            print(f"{ts:<22} {snap.step:>8} {snap.hard_example_accuracy:>9.1%} "
                  f"{snap.hard_example_correct:>8} {snap.corrections_pending:>8}")

        # Show latest comparison
        if len(self.snapshots) >= 2:
            comparison = self.compare_to_baseline()
            if comparison:
                print("\n" + "-" * 70)
                print("Latest vs Previous:")
                print(f"  Accuracy: {comparison.baseline_accuracy:.1%} -> {comparison.current_accuracy:.1%} "
                      f"({comparison.accuracy_delta:+.1%})")
                print(f"  Improved: {'Yes' if comparison.accuracy_improved else 'No'}")
                print(f"  Effectiveness: {comparison.effectiveness_score:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Impact Tracker")
    parser.add_argument('--base-dir', default=None,
                       help='Base directory')
    parser.add_argument('--snapshot', action='store_true',
                       help='Take a performance snapshot')
    parser.add_argument('--compare', action='store_true',
                       help='Compare to previous snapshot')
    parser.add_argument('--history', action='store_true',
                       help='Show snapshot history')
    parser.add_argument('--report', action='store_true',
                       help='Generate full report')
    parser.add_argument('--track', type=str,
                       help='Track a correction file')
    parser.add_argument('--notes', type=str,
                       help='Notes for snapshot')

    args = parser.parse_args()

    tracker = ImpactTracker(args.base_dir)

    if args.snapshot:
        tracker.take_snapshot(args.notes)
        print("Snapshot taken.")

    elif args.compare:
        comparison = tracker.compare_to_baseline()
        if comparison:
            print(f"\nComparison Results:")
            print(f"  Baseline: {comparison.baseline_accuracy:.1%}")
            print(f"  Current:  {comparison.current_accuracy:.1%}")
            print(f"  Delta:    {comparison.accuracy_delta:+.1%}")
            print(f"  Improved: {'Yes' if comparison.accuracy_improved else 'No'}")
            print(f"  Effectiveness Score: {comparison.effectiveness_score:+.2f}")

    elif args.history:
        tracker.show_history()

    elif args.report:
        report = tracker.get_effectiveness_report()
        print(json.dumps(report, indent=2))

    elif args.track:
        tracker.track_correction_file(args.track)

    else:
        # Default: show status
        metrics = tracker.get_current_metrics()
        pending = tracker.count_pending_corrections()
        print(f"\nCurrent Status:")
        print(f"  Hard Example Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Corrections Pending:   {pending}")
        print(f"  Snapshots Recorded:    {len(tracker.snapshots)}")
        print(f"\nUse --snapshot to record, --history to view trends")


if __name__ == "__main__":
    main()
