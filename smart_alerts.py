#!/usr/bin/env python3
"""
Smart Training Alerts

Actionable alerts that help you:
- Catch issues early (NaN, OOM, stalling)
- Make informed decisions (when to stop, which checkpoint to use)
- Avoid wasted compute (plateaus, overfitting)

Alert Priorities:
- CRITICAL: Stop training immediately (NaN, OOM)
- WARNING: Needs attention (plateau, gradient issues)
- INFO: For awareness (milestones, trends)
"""

import math
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import deque


class Alert:
    """Single training alert."""

    def __init__(
        self,
        severity: str,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recommendation: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Create an alert.

        Args:
            severity: "critical", "warning", or "info"
            category: Alert category (e.g., "nan_loss", "plateau")
            message: Human-readable message
            details: Additional context data
            recommendation: Suggested action
            timestamp: When alert was created
        """
        self.severity = severity
        self.category = category
        self.message = message
        self.details = details or {}
        self.recommendation = recommendation
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat()
        }


class SmartAlertsManager:
    """Manage training alerts with intelligent detection."""

    def __init__(self):
        """Initialize alerts manager."""
        self.active_alerts = {}  # category -> Alert
        self.alert_history = deque(maxlen=100)
        self.step_counter = 0

        # Alert thresholds
        self.plateau_threshold = 0.01  # 1% improvement over 50 steps
        self.plateau_window = 50

        self.gradient_explode_threshold = 100.0
        self.gradient_vanish_threshold = 1e-7

        self.calibration_error_threshold = 0.3  # ECE > 0.3 is poorly calibrated

        self.loss_spike_threshold = 2.0  # 2x increase is a spike

        print("🚨 Smart Alerts Manager initialized")

    def check_all(
        self,
        step: int,
        loss: float,
        streaming_ce: Optional[float] = None,
        gradient_norms: Optional[Dict[str, float]] = None,
        exact_match: Optional[float] = None,
        calibration_error: Optional[float] = None,
        learning_rate: Optional[float] = None,
        recent_losses: Optional[List[float]] = None
    ) -> List[Alert]:
        """
        Run all alert checks and return active alerts.

        Args:
            step: Current training step
            loss: Current loss value
            streaming_ce: Smoothed cross-entropy (if available)
            gradient_norms: Dict of layer -> gradient norm
            exact_match: Current EM score (0-1)
            calibration_error: ECE score (0-1)
            learning_rate: Current learning rate
            recent_losses: List of recent loss values

        Returns:
            List of active Alert objects
        """
        self.step_counter = step

        # Clear resolved alerts
        self._check_resolved()

        # Run all checks
        self._check_nan_loss(loss)
        self._check_loss_spike(loss, recent_losses)
        self._check_gradient_issues(gradient_norms)
        self._check_plateau(streaming_ce or loss, recent_losses)
        self._check_calibration(calibration_error, exact_match)
        self._check_learning_rate(learning_rate)

        # Return current active alerts
        return list(self.active_alerts.values())

    def _check_nan_loss(self, loss: float):
        """Check for NaN or Inf loss (CRITICAL)."""
        if math.isnan(loss) or math.isinf(loss):
            alert = Alert(
                severity="critical",
                category="nan_loss",
                message=f"Training collapsed: Loss is {'NaN' if math.isnan(loss) else 'Inf'}",
                details={"step": self.step_counter},
                recommendation="Stop training immediately. Check for: (1) Learning rate too high, "
                              "(2) Gradient clipping disabled, (3) Data issues (extreme values)"
            )
            self._add_alert(alert)
        else:
            self._clear_alert("nan_loss")

    def _check_loss_spike(self, loss: float, recent_losses: Optional[List[float]]):
        """Check for sudden loss spikes."""
        if recent_losses and len(recent_losses) >= 10:
            recent_avg = sum(recent_losses[-10:]) / 10
            if recent_avg > 0 and loss > recent_avg * self.loss_spike_threshold:
                alert = Alert(
                    severity="warning",
                    category="loss_spike",
                    message=f"Loss spike detected: {loss:.4f} vs recent avg {recent_avg:.4f}",
                    details={
                        "step": self.step_counter,
                        "current_loss": loss,
                        "recent_avg": recent_avg,
                        "spike_ratio": loss / recent_avg
                    },
                    recommendation="Monitor next few steps. If loss doesn't recover, consider: "
                                  "(1) Bad batch (may self-correct), (2) Learning rate too high, "
                                  "(3) Checkpoint corruption"
                )
                self._add_alert(alert)
            else:
                self._clear_alert("loss_spike")

    def _check_gradient_issues(self, gradient_norms: Optional[Dict[str, float]]):
        """Check for vanishing or exploding gradients."""
        if not gradient_norms:
            return

        max_grad = max(gradient_norms.values())
        min_grad = min(v for v in gradient_norms.values() if v > 0)

        # Check for exploding gradients
        if max_grad > self.gradient_explode_threshold:
            alert = Alert(
                severity="critical",
                category="exploding_gradients",
                message=f"Exploding gradients detected: max norm = {max_grad:.2e}",
                details={
                    "step": self.step_counter,
                    "max_gradient": max_grad,
                    "affected_layers": [k for k, v in gradient_norms.items() if v > self.gradient_explode_threshold]
                },
                recommendation="Reduce learning rate or enable gradient clipping"
            )
            self._add_alert(alert)
        else:
            self._clear_alert("exploding_gradients")

        # Check for vanishing gradients
        if min_grad < self.gradient_vanish_threshold:
            alert = Alert(
                severity="warning",
                category="vanishing_gradients",
                message=f"Vanishing gradients detected: min norm = {min_grad:.2e}",
                details={
                    "step": self.step_counter,
                    "min_gradient": min_grad,
                    "affected_layers": [k for k, v in gradient_norms.items() if v < self.gradient_vanish_threshold]
                },
                recommendation="Consider: (1) Increasing learning rate, (2) Checking activation functions, "
                              "(3) Using gradient scaling"
            )
            self._add_alert(alert)
        else:
            self._clear_alert("vanishing_gradients")

    def _check_plateau(self, current_loss: float, recent_losses: Optional[List[float]]):
        """Check if training has plateaued."""
        if not recent_losses or len(recent_losses) < self.plateau_window:
            return

        # Compare first half vs second half of window
        window = recent_losses[-self.plateau_window:]
        first_half = sum(window[:self.plateau_window//2]) / (self.plateau_window//2)
        second_half = sum(window[self.plateau_window//2:]) / (self.plateau_window//2)

        if first_half > 0:
            improvement = (first_half - second_half) / first_half

            if improvement < self.plateau_threshold:
                alert = Alert(
                    severity="warning",
                    category="plateau",
                    message=f"Training plateaued: Only {improvement*100:.2f}% improvement over {self.plateau_window} steps",
                    details={
                        "step": self.step_counter,
                        "improvement_pct": improvement * 100,
                        "window": self.plateau_window
                    },
                    recommendation="Consider: (1) Using current checkpoint (further training may not help), "
                                  "(2) Increasing learning rate, (3) Adding more diverse data"
                )
                self._add_alert(alert)
            else:
                self._clear_alert("plateau")

    def _check_calibration(self, calibration_error: Optional[float], exact_match: Optional[float]):
        """Check if model is poorly calibrated."""
        if calibration_error is None or calibration_error < 0:
            return

        if calibration_error > self.calibration_error_threshold:
            alert = Alert(
                severity="warning",
                category="poor_calibration",
                message=f"Model poorly calibrated: ECE = {calibration_error:.3f}",
                details={
                    "step": self.step_counter,
                    "calibration_error": calibration_error,
                    "exact_match": exact_match
                },
                recommendation="Model confidence scores don't match accuracy. "
                              "Consider: (1) Temperature scaling, (2) More training, (3) Confidence thresholding"
            )
            self._add_alert(alert)
        else:
            self._clear_alert("poor_calibration")

    def _check_learning_rate(self, learning_rate: Optional[float]):
        """Check if learning rate is too high/low."""
        if learning_rate is None:
            return

        # Very high LR
        if learning_rate > 1e-2:
            alert = Alert(
                severity="warning",
                category="high_learning_rate",
                message=f"Learning rate very high: {learning_rate:.2e}",
                details={
                    "step": self.step_counter,
                    "learning_rate": learning_rate
                },
                recommendation="Risk of instability. Consider reducing learning rate if seeing loss spikes."
            )
            self._add_alert(alert)
        else:
            self._clear_alert("high_learning_rate")

        # Very low LR (but not in warmup)
        if self.step_counter > 1000 and learning_rate < 1e-6:
            alert = Alert(
                severity="info",
                category="low_learning_rate",
                message=f"Learning rate very low: {learning_rate:.2e}",
                details={
                    "step": self.step_counter,
                    "learning_rate": learning_rate
                },
                recommendation="Learning may be slow. Training might be near completion or LR schedule very aggressive."
            )
            self._add_alert(alert)
        else:
            self._clear_alert("low_learning_rate")

    def _add_alert(self, alert: Alert):
        """Add or update an active alert."""
        self.active_alerts[alert.category] = alert
        self.alert_history.append(alert)

    def _clear_alert(self, category: str):
        """Clear an active alert (issue resolved)."""
        if category in self.active_alerts:
            del self.active_alerts[category]

    def _check_resolved(self):
        """Check if any active alerts have been resolved."""
        # Alerts are explicitly cleared by check functions
        pass

    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get all active alerts of a specific severity."""
        return [a for a in self.active_alerts.values() if a.severity == severity]

    def get_all_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_summary(self) -> Dict[str, int]:
        """Get counts of active alerts by severity."""
        summary = {"critical": 0, "warning": 0, "info": 0}
        for alert in self.active_alerts.values():
            summary[alert.severity] += 1
        return summary

    def to_dict(self) -> Dict:
        """Export all active alerts as dict for JSON."""
        return {
            "alerts": [a.to_dict() for a in self.active_alerts.values()],
            "summary": self.get_summary()
        }


def create_alerts_manager() -> SmartAlertsManager:
    """Factory function to create alerts manager."""
    return SmartAlertsManager()


if __name__ == "__main__":
    print("Smart Alerts Manager - Test Mode")
    print("This module should be imported and used during training.")

    # Quick test
    manager = create_alerts_manager()

    print("\nSimulating various alert conditions:")

    # Test NaN loss
    alerts = manager.check_all(step=100, loss=float('nan'))
    print(f"\n1. NaN Loss: {len(alerts)} alerts")
    for alert in alerts:
        print(f"   [{alert.severity.upper()}] {alert.message}")

    # Test normal training
    alerts = manager.check_all(step=200, loss=1.5)
    print(f"\n2. Normal Training: {len(alerts)} alerts")

    # Test plateau
    flat_losses = [2.0] * 60
    alerts = manager.check_all(step=300, loss=2.0, recent_losses=flat_losses)
    print(f"\n3. Plateau: {len(alerts)} alerts")
    for alert in alerts:
        print(f"   [{alert.severity.upper()}] {alert.message}")
