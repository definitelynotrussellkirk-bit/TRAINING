#!/usr/bin/env python3
"""
Monitoring Cost Tracker - Prevents expensive monitoring from causing training hangs

This module tracks the time and resource cost of all monitoring operations to ensure
they don't consume excessive training time. It provides warnings when monitoring
overhead exceeds acceptable thresholds.

LESSON LEARNED (2025-11-16):
- detail_collector ran inference every 50 steps → caused hangs
- evolution_tracker ran inference on 100 examples → caused hangs
- No visibility into costs → problems went unnoticed

This tracker ensures:
1. All monitoring costs are visible
2. Warnings when costs exceed budgets
3. Easy to identify expensive operations
4. Data for informed decisions about enabling/disabling features
"""

import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from contextlib import contextmanager


@dataclass
class OperationCost:
    """Cost metrics for a single monitoring operation"""
    name: str
    total_time: float = 0.0  # Total seconds spent
    call_count: int = 0       # Number of times called
    avg_time: float = 0.0     # Average time per call
    last_time: float = 0.0    # Time of last call
    percentage: float = 0.0   # Percentage of total training time

    def update_avg(self):
        """Recalculate average"""
        if self.call_count > 0:
            self.avg_time = self.total_time / self.call_count


class MonitoringCostTracker:
    """
    Tracks cost of all monitoring operations during training

    Usage:
        tracker = MonitoringCostTracker(output_dir="current_model/status")

        # Track an operation
        with tracker.track("detail_collector"):
            # Expensive monitoring operation here
            run_inference_on_validation_set()

        # Get report
        report = tracker.get_report()
        print(f"Monitoring overhead: {report['total_percentage']:.1f}%")

        # Check if over budget
        if tracker.is_over_budget():
            print("WARNING: Monitoring is consuming too much training time!")
    """

    # Budget thresholds (percentage of total training time)
    WARNING_THRESHOLD = 5.0   # Warn if monitoring >5% of training time
    CRITICAL_THRESHOLD = 10.0  # Critical if monitoring >10% of training time

    def __init__(self, output_dir: str = "status"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cost_file = self.output_dir / "monitoring_costs.json"

        # Cost tracking
        self.costs: Dict[str, OperationCost] = {}
        self.training_start_time = time.time()
        self.total_training_time = 0.0

        # Load existing costs if available
        self._load_costs()

    @contextmanager
    def track(self, operation_name: str):
        """
        Context manager to track cost of an operation

        Example:
            with tracker.track("evolution_snapshot"):
                capture_evolution_snapshot()
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._record_cost(operation_name, elapsed)

    def _record_cost(self, operation_name: str, elapsed_time: float):
        """Record cost of an operation"""
        if operation_name not in self.costs:
            self.costs[operation_name] = OperationCost(name=operation_name)

        cost = self.costs[operation_name]
        cost.total_time += elapsed_time
        cost.call_count += 1
        cost.last_time = elapsed_time
        cost.update_avg()

        # Update percentages
        self._update_percentages()

        # Save to file
        self._save_costs()

        # Check budget and warn if needed
        self._check_budget(operation_name, cost)

    def update_training_time(self):
        """Update total training time (call periodically)"""
        self.total_training_time = time.time() - self.training_start_time
        self._update_percentages()

    def _update_percentages(self):
        """Update percentage calculations for all operations"""
        if self.total_training_time > 0:
            for cost in self.costs.values():
                cost.percentage = (cost.total_time / self.total_training_time) * 100

    def _check_budget(self, operation_name: str, cost: OperationCost):
        """Check if operation is within budget and warn if not"""
        # Warn if single call took >30 seconds
        if cost.last_time > 30:
            print(f"\n⚠️  MONITORING COST WARNING:")
            print(f"   Operation: {operation_name}")
            print(f"   Last call took: {cost.last_time:.1f} seconds")
            print(f"   This is excessive for a monitoring operation!")

        # Check total percentage
        total_monitoring_pct = self.get_total_percentage()

        if total_monitoring_pct > self.CRITICAL_THRESHOLD:
            print(f"\n🚨 CRITICAL: Monitoring overhead is {total_monitoring_pct:.1f}%!")
            print(f"   This is significantly impacting training performance.")
            print(f"   Consider disabling some monitoring features.")
            self._print_cost_breakdown()

        elif total_monitoring_pct > self.WARNING_THRESHOLD:
            print(f"\n⚠️  WARNING: Monitoring overhead is {total_monitoring_pct:.1f}%")
            print(f"   Target is <{self.WARNING_THRESHOLD}% of training time.")
            self._print_cost_breakdown()

    def _print_cost_breakdown(self):
        """Print cost breakdown for all operations"""
        print(f"\n📊 Monitoring Cost Breakdown:")
        sorted_costs = sorted(self.costs.values(), key=lambda c: c.total_time, reverse=True)
        for cost in sorted_costs:
            print(f"   {cost.name:30s}: {cost.percentage:5.1f}% "
                  f"({cost.call_count} calls, {cost.avg_time:.1f}s avg)")

    def get_total_percentage(self) -> float:
        """Get total monitoring percentage of training time"""
        self.update_training_time()
        total_monitoring = sum(c.total_time for c in self.costs.values())
        if self.total_training_time > 0:
            return (total_monitoring / self.total_training_time) * 100
        return 0.0

    def is_over_budget(self) -> bool:
        """Check if monitoring is over budget"""
        return self.get_total_percentage() > self.WARNING_THRESHOLD

    def get_report(self) -> dict:
        """Get full cost report"""
        self.update_training_time()

        return {
            "timestamp": datetime.now().isoformat(),
            "total_training_time": self.total_training_time,
            "total_monitoring_time": sum(c.total_time for c in self.costs.values()),
            "total_percentage": self.get_total_percentage(),
            "is_over_budget": self.is_over_budget(),
            "operations": {
                name: asdict(cost) for name, cost in self.costs.items()
            },
            "budget_status": self._get_budget_status()
        }

    def _get_budget_status(self) -> str:
        """Get human-readable budget status"""
        pct = self.get_total_percentage()
        if pct > self.CRITICAL_THRESHOLD:
            return "CRITICAL"
        elif pct > self.WARNING_THRESHOLD:
            return "WARNING"
        else:
            return "OK"

    def _save_costs(self):
        """Save costs to JSON file"""
        try:
            report = self.get_report()
            with open(self.cost_file, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save monitoring costs: {e}")

    def _load_costs(self):
        """Load existing costs from file"""
        try:
            if self.cost_file.exists():
                with open(self.cost_file) as f:
                    data = json.load(f)
                    for name, cost_dict in data.get('operations', {}).items():
                        self.costs[name] = OperationCost(**cost_dict)
        except Exception as e:
            print(f"Warning: Failed to load monitoring costs: {e}")

    def print_summary(self):
        """Print summary of monitoring costs"""
        report = self.get_report()

        print("\n" + "="*80)
        print("📊 MONITORING COST SUMMARY")
        print("="*80)
        print(f"Budget Status: {report['budget_status']}")
        print(f"Total Monitoring Overhead: {report['total_percentage']:.1f}%")
        print(f"Total Training Time: {report['total_training_time']:.1f}s")
        print(f"Total Monitoring Time: {report['total_monitoring_time']:.1f}s")
        print()

        if report['operations']:
            print("Operation Costs:")
            sorted_ops = sorted(
                report['operations'].items(),
                key=lambda x: x[1]['total_time'],
                reverse=True
            )
            for name, cost in sorted_ops:
                print(f"  {name:30s}: {cost['percentage']:5.1f}% "
                      f"({cost['call_count']} calls, {cost['avg_time']:.1f}s avg)")
        else:
            print("No monitoring operations recorded yet.")

        print("="*80 + "\n")


# Global tracker instance
_global_tracker: Optional[MonitoringCostTracker] = None


def get_global_tracker() -> MonitoringCostTracker:
    """Get global monitoring cost tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MonitoringCostTracker()
    return _global_tracker


# Example cost estimates (measured on typical hardware)
KNOWN_OPERATION_COSTS = {
    "detail_collector_step": {
        "description": "Detailed monitoring per step",
        "estimated_time": "0.1-0.5s per call",
        "frequency": "Every 50 steps",
        "total_cost_per_1000_steps": "~2-10 seconds (0.2-1%)"
    },
    "evolution_snapshot": {
        "description": "Capture evolution snapshot (100 examples)",
        "estimated_time": "60-300s per call",
        "frequency": "Every 500 steps",
        "total_cost_per_1000_steps": "~120-600 seconds (12-60%)",
        "warning": "VERY EXPENSIVE - can cause significant hangs"
    },
    "validation_loss": {
        "description": "Compute validation loss (1000 examples)",
        "estimated_time": "30-120s per call",
        "frequency": "Every 200 steps",
        "total_cost_per_1000_steps": "~150-600 seconds (15-60%)",
        "warning": "Expensive - use smaller validation set if needed"
    },
    "live_inference": {
        "description": "Run inference on single example",
        "estimated_time": "1-5s per call",
        "frequency": "Every 200 steps",
        "total_cost_per_1000_steps": "~5-25 seconds (0.5-2.5%)"
    }
}


if __name__ == "__main__":
    # Example usage
    tracker = MonitoringCostTracker()

    # Simulate some operations
    print("Simulating monitoring operations...")

    with tracker.track("detail_collector"):
        time.sleep(0.5)

    with tracker.track("live_inference"):
        time.sleep(2.0)

    with tracker.track("evolution_snapshot"):
        time.sleep(60.0)  # Expensive!

    # Print summary
    tracker.print_summary()
