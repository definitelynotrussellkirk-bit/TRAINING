"""
Status Monitor - Detects when training status becomes stale.

This module provides utilities to detect when training_status.json stops
updating while training is supposed to be running, which indicates a
callback wiring bug.

Usage:
    from core.status_monitor import StatusMonitor, check_status_staleness

    # One-shot check
    result = check_status_staleness()
    if result.is_stale:
        print(f"Warning: Status stale for {result.stale_seconds}s")

    # Continuous monitoring (in daemon)
    monitor = StatusMonitor()
    monitor.mark_training_started()
    ...
    if monitor.should_warn():
        print(f"Warning: {monitor.get_warning_message()}")
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# POLICY 2: Performance Baselines
@dataclass
class PerformanceResult:
    """Result of a performance check."""
    is_degraded: bool
    current_speed: float
    baseline_speed: Optional[float]
    degradation_pct: float
    message: str
    severity: str  # "ok", "warning", "error"


@dataclass
class StalenessResult:
    """Result of a staleness check."""
    is_stale: bool
    status_exists: bool
    last_update: Optional[datetime]
    stale_seconds: float
    training_status: Optional[str]  # "training", "idle", etc.
    current_step: Optional[int]
    message: str

    @property
    def severity(self) -> str:
        """Return severity level for UI display."""
        if not self.is_stale:
            return "ok"
        if self.stale_seconds < 120:
            return "warning"
        return "error"


def _get_status_file() -> Path:
    """Get path to training_status.json."""
    try:
        from core.paths import get_base_dir
        return get_base_dir() / "status" / "training_status.json"
    except ImportError:
        return Path(__file__).parent.parent / "status" / "training_status.json"


def check_status_staleness(
    max_stale_seconds: float = 60.0,
    status_file: Optional[Path] = None,
) -> StalenessResult:
    """
    Check if training_status.json is stale.

    Args:
        max_stale_seconds: Consider stale if older than this many seconds
        status_file: Path to status file (auto-detected if not provided)

    Returns:
        StalenessResult with staleness information
    """
    status_file = status_file or _get_status_file()

    if not status_file.exists():
        return StalenessResult(
            is_stale=False,  # Can't be stale if doesn't exist
            status_exists=False,
            last_update=None,
            stale_seconds=0,
            training_status=None,
            current_step=None,
            message="Status file does not exist",
        )

    try:
        with open(status_file) as f:
            data = json.load(f)

        # Parse timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            try:
                last_update = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                last_update = datetime.fromtimestamp(status_file.stat().st_mtime)
        else:
            last_update = datetime.fromtimestamp(status_file.stat().st_mtime)

        # Calculate staleness
        now = datetime.now()
        if last_update.tzinfo:
            # Handle timezone-aware datetime
            from datetime import timezone
            now = datetime.now(timezone.utc)

        stale_seconds = (now - last_update).total_seconds()
        training_status = data.get("status", "unknown")
        current_step = data.get("current_step")

        # Determine if stale based on status
        # Only consider stale if status is "training" and file is old
        is_stale = (
            training_status == "training" and
            stale_seconds > max_stale_seconds
        )

        if is_stale:
            message = f"Training status stale: no update for {stale_seconds:.0f}s (step {current_step})"
        elif training_status == "training":
            message = f"Training active at step {current_step}"
        else:
            message = f"Training status: {training_status}"

        return StalenessResult(
            is_stale=is_stale,
            status_exists=True,
            last_update=last_update,
            stale_seconds=stale_seconds,
            training_status=training_status,
            current_step=current_step,
            message=message,
        )

    except Exception as e:
        logger.error(f"Failed to check status staleness: {e}")
        return StalenessResult(
            is_stale=False,
            status_exists=True,
            last_update=None,
            stale_seconds=0,
            training_status=None,
            current_step=None,
            message=f"Error reading status: {e}",
        )


class StatusMonitor:
    """
    Monitors training status for staleness during active training.

    Use this in the training daemon to detect when callbacks stop working.

    Usage:
        monitor = StatusMonitor()

        # When training starts
        monitor.mark_training_started()

        # Periodically during training
        if monitor.should_warn():
            logger.warning(monitor.get_warning_message())

        # When training stops
        monitor.mark_training_stopped()
    """

    def __init__(
        self,
        warn_after_seconds: float = 60.0,
        error_after_seconds: float = 300.0,
    ):
        self.warn_after_seconds = warn_after_seconds
        self.error_after_seconds = error_after_seconds
        self._training_started_at: Optional[float] = None
        self._last_warning_time: float = 0
        self._warning_cooldown: float = 60.0  # Don't spam warnings

    def mark_training_started(self):
        """Mark that training has started."""
        self._training_started_at = time.time()

    def mark_training_stopped(self):
        """Mark that training has stopped."""
        self._training_started_at = None

    @property
    def is_training(self) -> bool:
        """Whether training is currently active."""
        return self._training_started_at is not None

    def check(self) -> StalenessResult:
        """Check current staleness status."""
        return check_status_staleness(max_stale_seconds=self.warn_after_seconds)

    def should_warn(self) -> bool:
        """Whether a warning should be emitted."""
        if not self.is_training:
            return False

        # Check cooldown
        now = time.time()
        if now - self._last_warning_time < self._warning_cooldown:
            return False

        result = self.check()
        if result.is_stale:
            self._last_warning_time = now
            return True

        return False

    def get_warning_message(self) -> str:
        """Get the warning message to display."""
        result = self.check()
        if result.is_stale:
            return (
                f"WARNING: Training status stale for {result.stale_seconds:.0f}s! "
                f"Callbacks may not be wired correctly. "
                f"Step stuck at {result.current_step}."
            )
        return ""


# =============================================================================
# POLICY 2: PERFORMANCE HEALTH CHECKING
# =============================================================================

def get_performance_baseline(model_name: str, batch_size: int) -> Optional[float]:
    """
    Get expected baseline performance for a model/batch_size combination.

    Args:
        model_name: Model identifier (e.g., "qwen3_0.6b")
        batch_size: Batch size being used

    Returns:
        Expected steps/sec, or None if no baseline configured
    """
    try:
        from core.paths import get_base_dir
        config_path = get_base_dir() / "config.json"
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        baselines = config.get("performance_baselines", {})
        speeds = baselines.get("training_speed_it_per_sec", {})
        model_speeds = speeds.get(model_name, {})

        # Try exact match first
        batch_key = f"batch_size_{batch_size}"
        if batch_key in model_speeds:
            return model_speeds[batch_key]

        # No baseline found
        return None

    except Exception as e:
        logger.debug(f"Could not load performance baseline: {e}")
        return None


def check_performance_health(
    current_speed: Optional[float] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None
) -> PerformanceResult:
    """
    Check if training performance is degraded.

    POLICY 2: Performance Baselines & Alerts
    Alerts when speed drops below 50% of baseline.

    Args:
        current_speed: Current steps/sec (auto-detected if None)
        model_name: Model name (auto-detected if None)
        batch_size: Batch size (auto-detected if None)

    Returns:
        PerformanceResult with degradation information
    """
    # Auto-detect current state if not provided
    if current_speed is None or model_name is None or batch_size is None:
        try:
            from core.realm_store import get_training_state
            from core.paths import get_base_dir

            training = get_training_state()
            if current_speed is None:
                current_speed = training.get("speed") or 0

            if model_name is None or batch_size is None:
                config_path = get_base_dir() / "config.json"
                with open(config_path) as f:
                    config = json.load(f)
                if model_name is None:
                    model_name = config.get("model_name", "unknown")
                if batch_size is None:
                    batch_size = config.get("batch_size", 1)

        except Exception as e:
            logger.debug(f"Could not auto-detect performance params: {e}")
            return PerformanceResult(
                is_degraded=False,
                current_speed=0,
                baseline_speed=None,
                degradation_pct=0,
                message="Could not check performance - missing data",
                severity="unknown"
            )

    # Get baseline for this configuration
    baseline = get_performance_baseline(model_name, batch_size)

    if baseline is None:
        return PerformanceResult(
            is_degraded=False,
            current_speed=current_speed,
            baseline_speed=None,
            degradation_pct=0,
            message=f"No baseline configured for {model_name} batch_size={batch_size}",
            severity="ok"
        )

    # Calculate degradation
    if baseline > 0:
        pct_of_baseline = (current_speed / baseline) * 100
        degradation_pct = 100 - pct_of_baseline
    else:
        pct_of_baseline = 0
        degradation_pct = 0

    # Determine if degraded
    threshold_pct = 50  # Alert if <50% of baseline
    is_degraded = pct_of_baseline < threshold_pct

    if is_degraded:
        severity = "error" if pct_of_baseline < 25 else "warning"
        message = (
            f"Performance degraded! "
            f"Speed {current_speed:.2f} it/s is {pct_of_baseline:.0f}% of "
            f"baseline {baseline:.2f} it/s (expected for {model_name} batch={batch_size})"
        )
    else:
        severity = "ok"
        message = f"Performance OK: {current_speed:.2f} it/s ({pct_of_baseline:.0f}% of baseline)"

    return PerformanceResult(
        is_degraded=is_degraded,
        current_speed=current_speed,
        baseline_speed=baseline,
        degradation_pct=degradation_pct,
        message=message,
        severity=severity
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check training status staleness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch continuously")

    args = parser.parse_args()

    if args.watch:
        print("Watching training status (Ctrl+C to stop)...")
        try:
            while True:
                result = check_status_staleness()
                status_icon = "🟢" if not result.is_stale else "🔴"
                print(f"\r{status_icon} {result.message} (age: {result.stale_seconds:.0f}s)    ", end="")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped")
    else:
        result = check_status_staleness()

        if args.verbose:
            print(f"Status file exists: {result.status_exists}")
            print(f"Last update: {result.last_update}")
            print(f"Stale seconds: {result.stale_seconds:.1f}")
            print(f"Training status: {result.training_status}")
            print(f"Current step: {result.current_step}")
            print(f"Is stale: {result.is_stale}")
            print(f"Severity: {result.severity}")
            print()

        status_icon = "🟢" if not result.is_stale else ("🟡" if result.severity == "warning" else "🔴")
        print(f"{status_icon} {result.message}")
