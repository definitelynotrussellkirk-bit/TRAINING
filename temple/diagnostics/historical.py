"""
Historical Comparison - Learn from the Past
=============================================

Compare current training to past campaigns at the same step:
- "At step 5000, your best campaign had loss 0.8, current is 1.2"
- "You're ahead of 80% of past runs at this point"
- "Warning: At this loss, 60% of past runs eventually crashed"

This provides context that raw metrics can't give:
- Is this loss good or bad relative to history?
- Am I on track compared to successful runs?
- Should I be worried based on past patterns?

Usage:
    from temple.diagnostics import HistoricalComparison

    history = HistoricalComparison()

    # Compare current to past
    comparison = history.compare(
        step=5000,
        loss=1.2,
        campaign_id="current"
    )

    print(f"Percentile: {comparison.percentile}")  # 75th percentile
    print(f"Best at this step: {comparison.best_loss}")
    print(f"Warning: {comparison.warning}")  # "60% of runs with this loss crashed"
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HistoricalDataPoint:
    """Single data point from historical campaigns."""
    campaign_id: str
    hero: str
    step: int
    loss: float
    timestamp: str
    outcome: Optional[str] = None  # "completed", "crashed", "stopped", None


@dataclass
class ComparisonResult:
    """Result of comparing current training to history."""
    step: int
    current_loss: float

    # Comparison metrics
    percentile: float  # What percentile is current loss (lower = better, so 90th = top 10%)
    best_loss_at_step: Optional[float] = None
    worst_loss_at_step: Optional[float] = None
    median_loss_at_step: Optional[float] = None
    mean_loss_at_step: Optional[float] = None

    # Relative position
    better_than_count: int = 0
    worse_than_count: int = 0
    total_campaigns: int = 0

    # Predictions based on history
    similar_campaigns: List[str] = field(default_factory=list)  # Campaigns with similar trajectory
    crash_probability: Optional[float] = None  # Probability of crash based on similar runs
    expected_final_loss: Optional[float] = None  # Predicted final loss based on trajectory

    # Warnings and insights
    warnings: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "current_loss": self.current_loss,
            "percentile": self.percentile,
            "best_loss_at_step": self.best_loss_at_step,
            "worst_loss_at_step": self.worst_loss_at_step,
            "median_loss_at_step": self.median_loss_at_step,
            "better_than_count": self.better_than_count,
            "worse_than_count": self.worse_than_count,
            "total_campaigns": self.total_campaigns,
            "crash_probability": self.crash_probability,
            "expected_final_loss": self.expected_final_loss,
            "warnings": self.warnings,
            "insights": self.insights,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []

        if self.total_campaigns == 0:
            return "No historical data available for comparison."

        lines.append(f"At step {self.step:,}, your loss ({self.current_loss:.4f}):")

        # Percentile
        if self.percentile >= 90:
            lines.append(f"  🌟 Top {100 - self.percentile:.0f}% of all runs!")
        elif self.percentile >= 75:
            lines.append(f"  ✅ Better than {self.percentile:.0f}% of past runs")
        elif self.percentile >= 50:
            lines.append(f"  📊 Better than {self.percentile:.0f}% (above average)")
        elif self.percentile >= 25:
            lines.append(f"  ⚠️ Below median ({self.percentile:.0f}th percentile)")
        else:
            lines.append(f"  ⚡ Bottom {self.percentile:.0f}% - needs attention")

        # Best/median comparison
        if self.best_loss_at_step:
            diff = self.current_loss - self.best_loss_at_step
            pct = (diff / self.best_loss_at_step * 100) if self.best_loss_at_step > 0 else 0
            lines.append(f"  Best ever: {self.best_loss_at_step:.4f} ({pct:+.1f}% from best)")

        if self.median_loss_at_step:
            diff = self.current_loss - self.median_loss_at_step
            lines.append(f"  Median: {self.median_loss_at_step:.4f} (you're {diff:+.4f})")

        # Crash probability
        if self.crash_probability is not None and self.crash_probability > 0.3:
            lines.append(f"  ⚠️ {self.crash_probability:.0%} of similar runs eventually crashed")

        # Insights
        for insight in self.insights[:3]:
            lines.append(f"  💡 {insight}")

        # Warnings
        for warning in self.warnings[:2]:
            lines.append(f"  🚨 {warning}")

        return "\n".join(lines)


class HistoricalComparison:
    """
    Compare current training to historical campaigns.

    Data sources:
    1. Checkpoint ledger (primary)
    2. Campaign history files
    3. Battle log archives
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        ledger_path: Optional[str] = None,
        campaigns_dir: Optional[str] = None,
    ):
        if base_dir is None:
            base_dir = Path.cwd()
        self.base_dir = Path(base_dir)

        self.ledger_path = Path(ledger_path) if ledger_path else self.base_dir / "status" / "checkpoint_ledger.json"
        self.campaigns_dir = Path(campaigns_dir) if campaigns_dir else self.base_dir / "campaigns"

        # Cache
        self._history_cache: Optional[Dict[str, List[HistoricalDataPoint]]] = None
        self._cache_time: Optional[float] = None

    def compare(
        self,
        step: int,
        loss: float,
        campaign_id: Optional[str] = None,
        hero: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Compare current metrics to historical campaigns.

        Args:
            step: Current training step
            loss: Current loss value
            campaign_id: Current campaign (excluded from comparison)
            hero: Filter to same hero/model type

        Returns:
            ComparisonResult with percentiles, warnings, insights
        """
        history = self._load_history()

        # Filter out current campaign
        if campaign_id:
            history = {
                cid: points for cid, points in history.items()
                if cid != campaign_id
            }

        # Filter by hero if specified
        if hero:
            history = {
                cid: [p for p in points if p.hero == hero]
                for cid, points in history.items()
            }
            history = {k: v for k, v in history.items() if v}

        if not history:
            return ComparisonResult(
                step=step,
                current_loss=loss,
                percentile=50.0,  # No data = assume middle
                warnings=["No historical data for comparison"],
            )

        # Get losses at similar steps (within 10% window)
        step_window = max(100, int(step * 0.1))
        losses_at_step = []
        campaigns_at_step = []

        for cid, points in history.items():
            for p in points:
                if abs(p.step - step) <= step_window:
                    losses_at_step.append(p.loss)
                    campaigns_at_step.append(cid)

        if not losses_at_step:
            # Extrapolate from nearest points
            return ComparisonResult(
                step=step,
                current_loss=loss,
                percentile=50.0,
                total_campaigns=len(history),
                insights=["No data at this step yet, comparison unavailable"],
            )

        # Calculate percentile (lower loss = higher percentile)
        better_count = sum(1 for l in losses_at_step if l >= loss)
        percentile = (better_count / len(losses_at_step)) * 100

        # Statistics
        sorted_losses = sorted(losses_at_step)
        median_idx = len(sorted_losses) // 2

        result = ComparisonResult(
            step=step,
            current_loss=loss,
            percentile=round(percentile, 1),
            best_loss_at_step=min(losses_at_step),
            worst_loss_at_step=max(losses_at_step),
            median_loss_at_step=sorted_losses[median_idx],
            mean_loss_at_step=sum(losses_at_step) / len(losses_at_step),
            better_than_count=better_count,
            worse_than_count=len(losses_at_step) - better_count,
            total_campaigns=len(set(campaigns_at_step)),
        )

        # Find similar campaigns and analyze outcomes
        self._analyze_similar_campaigns(result, history, step, loss)

        # Generate insights
        self._generate_insights(result, history, step)

        return result

    def _analyze_similar_campaigns(
        self,
        result: ComparisonResult,
        history: Dict[str, List[HistoricalDataPoint]],
        step: int,
        loss: float,
    ) -> None:
        """Find campaigns with similar trajectory and analyze outcomes."""
        similar = []
        crashed_count = 0
        completed_count = 0
        final_losses = []

        # Define similarity: within 10% of current loss at similar step
        loss_threshold = loss * 0.1
        step_window = max(100, int(step * 0.1))

        for cid, points in history.items():
            # Find closest point to current step
            closest = None
            for p in points:
                if abs(p.step - step) <= step_window:
                    if closest is None or abs(p.step - step) < abs(closest.step - step):
                        closest = p

            if closest and abs(closest.loss - loss) <= loss_threshold:
                similar.append(cid)

                # Analyze outcome
                outcome = closest.outcome
                if outcome == "crashed":
                    crashed_count += 1
                elif outcome == "completed":
                    completed_count += 1

                # Get final loss
                campaign_points = sorted(points, key=lambda p: p.step)
                if campaign_points:
                    final_losses.append(campaign_points[-1].loss)

        result.similar_campaigns = similar[:10]  # Limit to 10

        if similar:
            result.crash_probability = crashed_count / len(similar)
            if final_losses:
                result.expected_final_loss = sum(final_losses) / len(final_losses)

        # Warnings based on crash probability
        if result.crash_probability and result.crash_probability > 0.5:
            result.warnings.append(
                f"{result.crash_probability:.0%} of similar runs crashed - monitor closely"
            )

    def _generate_insights(
        self,
        result: ComparisonResult,
        history: Dict[str, List[HistoricalDataPoint]],
        step: int,
    ) -> None:
        """Generate actionable insights from historical comparison."""

        # Insight: Rate of improvement
        if result.percentile >= 90:
            result.insights.append("Exceptional progress - this run is tracking ahead of history")
        elif result.percentile <= 25:
            result.insights.append("Behind typical progress - consider adjusting hyperparameters")

        # Insight: Comparison to best
        if result.best_loss_at_step and result.current_loss:
            gap = result.current_loss - result.best_loss_at_step
            if gap > 0 and result.best_loss_at_step > 0:
                gap_pct = (gap / result.best_loss_at_step) * 100
                if gap_pct > 50:
                    result.insights.append(f"50%+ gap to best-ever suggests room for optimization")
                elif gap_pct < 10:
                    result.insights.append("Within 10% of best-ever - excellent progress!")

        # Insight: Expected trajectory
        if result.expected_final_loss and result.current_loss:
            improvement = result.current_loss - result.expected_final_loss
            if improvement > 0:
                result.insights.append(
                    f"Similar runs improved by ~{improvement:.3f} loss by completion"
                )

    def _load_history(self) -> Dict[str, List[HistoricalDataPoint]]:
        """Load historical data from all sources."""
        if self._history_cache is not None:
            return self._history_cache

        history: Dict[str, List[HistoricalDataPoint]] = {}

        # 1. Load from checkpoint ledger
        self._load_from_ledger(history)

        # 2. Load from campaign directories
        self._load_from_campaigns(history)

        self._history_cache = history
        return history

    def _load_from_ledger(self, history: Dict[str, List[HistoricalDataPoint]]) -> None:
        """Load historical data from checkpoint ledger."""
        if not self.ledger_path.exists():
            return

        try:
            with open(self.ledger_path) as f:
                data = json.load(f)

            entries = data.get("entries", {})

            for name, entry in entries.items():
                step = entry.get("step", 0)
                stats = entry.get("stats", {})
                loss = stats.get("train_loss")

                if loss is None:
                    continue

                campaign_id = entry.get("campaign_id", "unknown")
                hero = entry.get("hero") or entry.get("model_name", "unknown")
                timestamp = entry.get("timestamp", "")

                point = HistoricalDataPoint(
                    campaign_id=campaign_id,
                    hero=hero,
                    step=step,
                    loss=loss,
                    timestamp=timestamp,
                )

                if campaign_id not in history:
                    history[campaign_id] = []
                history[campaign_id].append(point)

        except Exception as e:
            logger.warning(f"Error loading ledger: {e}")

    def _load_from_campaigns(self, history: Dict[str, List[HistoricalDataPoint]]) -> None:
        """Load historical data from campaign directories."""
        if not self.campaigns_dir.exists():
            return

        try:
            for hero_dir in self.campaigns_dir.iterdir():
                if not hero_dir.is_dir():
                    continue

                hero = hero_dir.name

                for campaign_dir in hero_dir.iterdir():
                    if not campaign_dir.is_dir():
                        continue

                    campaign_id = f"{hero}/{campaign_dir.name}"

                    # Load campaign.json
                    campaign_file = campaign_dir / "campaign.json"
                    if campaign_file.exists():
                        self._load_campaign_file(
                            campaign_file, campaign_id, hero, history
                        )

                    # Load battle_log.jsonl
                    battle_log = campaign_dir / "battle_log.jsonl"
                    if battle_log.exists():
                        self._load_battle_log(
                            battle_log, campaign_id, hero, history
                        )

        except Exception as e:
            logger.warning(f"Error loading campaigns: {e}")

    def _load_campaign_file(
        self,
        path: Path,
        campaign_id: str,
        hero: str,
        history: Dict[str, List[HistoricalDataPoint]],
    ) -> None:
        """Load data from a campaign.json file."""
        try:
            with open(path) as f:
                data = json.load(f)

            # Extract checkpoints
            checkpoints = data.get("checkpoints", [])
            outcome = data.get("outcome")  # "completed", "crashed", etc.

            if campaign_id not in history:
                history[campaign_id] = []

            for cp in checkpoints:
                step = cp.get("step", 0)
                loss = cp.get("loss") or cp.get("train_loss")

                if loss is not None:
                    point = HistoricalDataPoint(
                        campaign_id=campaign_id,
                        hero=hero,
                        step=step,
                        loss=loss,
                        timestamp=cp.get("timestamp", ""),
                        outcome=outcome,
                    )
                    history[campaign_id].append(point)

        except Exception as e:
            logger.debug(f"Error loading {path}: {e}")

    def _load_battle_log(
        self,
        path: Path,
        campaign_id: str,
        hero: str,
        history: Dict[str, List[HistoricalDataPoint]],
    ) -> None:
        """Load data from a battle_log.jsonl file."""
        try:
            if campaign_id not in history:
                history[campaign_id] = []

            with open(path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        step = entry.get("step", 0)
                        loss = entry.get("loss") or entry.get("train_loss")

                        if loss is not None:
                            point = HistoricalDataPoint(
                                campaign_id=campaign_id,
                                hero=hero,
                                step=step,
                                loss=loss,
                                timestamp=entry.get("timestamp", ""),
                            )
                            history[campaign_id].append(point)
                    except:
                        continue

        except Exception as e:
            logger.debug(f"Error loading {path}: {e}")

    def get_statistics(self, hero: Optional[str] = None) -> Dict[str, Any]:
        """Get overall statistics about historical campaigns."""
        history = self._load_history()

        if hero:
            history = {
                cid: [p for p in points if p.hero == hero]
                for cid, points in history.items()
            }
            history = {k: v for k, v in history.items() if v}

        total_campaigns = len(history)
        total_datapoints = sum(len(points) for points in history.values())

        all_losses = []
        max_steps = []

        for points in history.values():
            for p in points:
                all_losses.append(p.loss)
            if points:
                max_steps.append(max(p.step for p in points))

        return {
            "total_campaigns": total_campaigns,
            "total_datapoints": total_datapoints,
            "loss_range": (min(all_losses), max(all_losses)) if all_losses else (0, 0),
            "mean_loss": sum(all_losses) / len(all_losses) if all_losses else 0,
            "max_step_reached": max(max_steps) if max_steps else 0,
            "average_campaign_length": sum(max_steps) / len(max_steps) if max_steps else 0,
        }

    def get_best_campaign(self, hero: Optional[str] = None) -> Optional[str]:
        """Get the campaign with the lowest final loss."""
        history = self._load_history()

        best_campaign = None
        best_final_loss = float('inf')

        for cid, points in history.items():
            if hero:
                points = [p for p in points if p.hero == hero]
            if not points:
                continue

            # Get final loss
            sorted_points = sorted(points, key=lambda p: p.step)
            final_loss = sorted_points[-1].loss

            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_campaign = cid

        return best_campaign

    def clear_cache(self) -> None:
        """Clear the history cache."""
        self._history_cache = None
        self._cache_time = None


# ========== Convenience Functions ==========

def compare_to_history(
    step: int,
    loss: float,
    campaign_id: Optional[str] = None,
    hero: Optional[str] = None,
) -> ComparisonResult:
    """
    Quick comparison of current metrics to history.

    Usage:
        from temple.diagnostics.historical import compare_to_history

        result = compare_to_history(step=5000, loss=1.2)
        print(result.summary())
    """
    comp = HistoricalComparison()
    return comp.compare(step, loss, campaign_id, hero)


def get_historical_stats(hero: Optional[str] = None) -> Dict[str, Any]:
    """Get overall historical statistics."""
    comp = HistoricalComparison()
    return comp.get_statistics(hero)
