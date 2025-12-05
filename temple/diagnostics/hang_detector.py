"""
Training Hang Detection - Detect Stalls, Not Just Crashes
==========================================================

Many training runs don't crash - they just stall. This is often worse than
a crash because:
1. Resources sit idle burning compute
2. No error message tells you something is wrong
3. You might not notice for hours

This module detects hangs through multiple signals:
1. Step Rate - Steps/sec drops to near zero
2. Loss Staleness - Loss hasn't changed for N steps
3. Heartbeat Timeout - No update in X seconds
4. GPU Utilization - GPU goes idle but process alive
5. Gradient Silence - No gradient updates

Usage:
    from temple.diagnostics import HangDetector

    detector = HangDetector(heartbeat_timeout=60)

    for step, batch in enumerate(dataloader):
        loss = train_step(batch)

        # Check for hang
        status = detector.heartbeat(step=step, loss=loss)
        if status.is_hung:
            print(f"HANG DETECTED: {status.reason}")
            # Take action: restart, alert, etc.

    # Or run as background monitor
    detector.start_monitoring()
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HangType(Enum):
    """Type of hang detected."""
    NONE = "none"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    ZERO_STEP_RATE = "zero_step_rate"
    LOSS_STALE = "loss_stale"
    GPU_IDLE = "gpu_idle"
    GRADIENT_ZERO = "gradient_zero"
    PROCESS_ZOMBIE = "process_zombie"


@dataclass
class HangStatus:
    """Status from hang detection check."""
    is_hung: bool = False
    hang_type: HangType = HangType.NONE
    reason: str = ""
    seconds_since_update: float = 0.0
    steps_per_second: float = 0.0
    last_loss: Optional[float] = None
    gpu_utilization: Optional[float] = None
    confidence: float = 0.0  # 0-1, how confident we are this is a real hang

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_hung": self.is_hung,
            "hang_type": self.hang_type.value,
            "reason": self.reason,
            "seconds_since_update": self.seconds_since_update,
            "steps_per_second": self.steps_per_second,
            "last_loss": self.last_loss,
            "gpu_utilization": self.gpu_utilization,
            "confidence": self.confidence,
        }


@dataclass
class HeartbeatRecord:
    """Single heartbeat record."""
    timestamp: float
    step: int
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    gpu_memory: Optional[float] = None


class HangDetector:
    """
    Detects training hangs through multiple signals.

    Configuration:
        heartbeat_timeout: Seconds without update before hang alert (default 60)
        min_step_rate: Steps/sec below this is suspicious (default 0.01)
        loss_stale_steps: Loss unchanged for this many steps is suspicious (default 100)
        check_gpu: Enable GPU utilization monitoring (default True)
    """

    def __init__(
        self,
        heartbeat_timeout: float = 60.0,
        min_step_rate: float = 0.01,
        loss_stale_steps: int = 100,
        loss_stale_threshold: float = 1e-8,  # Loss change smaller than this = stale
        check_gpu: bool = True,
        on_hang: Optional[Callable[[HangStatus], None]] = None,
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.min_step_rate = min_step_rate
        self.loss_stale_steps = loss_stale_steps
        self.loss_stale_threshold = loss_stale_threshold
        self.check_gpu = check_gpu
        self.on_hang = on_hang

        # State
        self.heartbeats: deque[HeartbeatRecord] = deque(maxlen=1000)
        self.last_heartbeat_time: float = time.time()
        self.last_step: int = 0
        self.last_loss: Optional[float] = None
        self.loss_unchanged_count: int = 0
        self.hang_detected_at: Optional[float] = None
        self.last_hang_status: Optional[HangStatus] = None

        # Background monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    def heartbeat(
        self,
        step: int,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        gpu_memory: Optional[float] = None,
    ) -> HangStatus:
        """
        Record a heartbeat and check for hangs.

        Call this every training step (or periodically).

        Args:
            step: Current training step
            loss: Current loss value
            grad_norm: Current gradient norm
            gpu_memory: Current GPU memory usage (0-1)

        Returns:
            HangStatus indicating if a hang is detected
        """
        now = time.time()

        # Record heartbeat
        record = HeartbeatRecord(
            timestamp=now,
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            gpu_memory=gpu_memory,
        )
        self.heartbeats.append(record)

        # Check for loss staleness
        if loss is not None:
            if self.last_loss is not None:
                if abs(loss - self.last_loss) < self.loss_stale_threshold:
                    self.loss_unchanged_count += 1
                else:
                    self.loss_unchanged_count = 0
            self.last_loss = loss

        # Update state
        self.last_heartbeat_time = now
        self.last_step = step

        # Clear any hang detection since we got an update
        self.hang_detected_at = None

        # Check all hang signals
        status = self._check_hang_signals()
        self.last_hang_status = status

        if status.is_hung and self.on_hang:
            self.on_hang(status)

        return status

    def check(self) -> HangStatus:
        """
        Check for hang without recording a heartbeat.

        Use this for periodic monitoring from a separate thread.
        """
        return self._check_hang_signals()

    def _check_hang_signals(self) -> HangStatus:
        """Check all hang detection signals."""
        now = time.time()

        # 1. Heartbeat timeout
        seconds_since_update = now - self.last_heartbeat_time
        if seconds_since_update > self.heartbeat_timeout:
            if self.hang_detected_at is None:
                self.hang_detected_at = now
            return HangStatus(
                is_hung=True,
                hang_type=HangType.HEARTBEAT_TIMEOUT,
                reason=f"No heartbeat for {seconds_since_update:.0f}s (timeout: {self.heartbeat_timeout}s)",
                seconds_since_update=seconds_since_update,
                last_loss=self.last_loss,
                confidence=min(1.0, seconds_since_update / (self.heartbeat_timeout * 2)),
            )

        # 2. Step rate too low
        step_rate = self._calculate_step_rate()
        if step_rate is not None and step_rate < self.min_step_rate and len(self.heartbeats) > 10:
            # Only alert if we have enough data and rate is consistently low
            if self.hang_detected_at is None:
                self.hang_detected_at = now
            return HangStatus(
                is_hung=True,
                hang_type=HangType.ZERO_STEP_RATE,
                reason=f"Step rate {step_rate:.4f}/s below minimum {self.min_step_rate}/s",
                seconds_since_update=seconds_since_update,
                steps_per_second=step_rate,
                last_loss=self.last_loss,
                confidence=0.8,
            )

        # 3. Loss stale
        if self.loss_unchanged_count >= self.loss_stale_steps:
            return HangStatus(
                is_hung=True,
                hang_type=HangType.LOSS_STALE,
                reason=f"Loss unchanged for {self.loss_unchanged_count} steps",
                seconds_since_update=seconds_since_update,
                steps_per_second=step_rate or 0.0,
                last_loss=self.last_loss,
                confidence=0.7,
            )

        # 4. GPU utilization (if enabled)
        if self.check_gpu:
            gpu_util = self._check_gpu_utilization()
            if gpu_util is not None and gpu_util < 0.1:  # <10% utilization
                # GPU is nearly idle but we're supposedly training
                if self._training_should_be_active():
                    return HangStatus(
                        is_hung=True,
                        hang_type=HangType.GPU_IDLE,
                        reason=f"GPU utilization only {gpu_util:.0%} during training",
                        seconds_since_update=seconds_since_update,
                        gpu_utilization=gpu_util,
                        last_loss=self.last_loss,
                        confidence=0.6,
                    )

        # 5. Gradient zero (if we have grad norms)
        if self.heartbeats and len(self.heartbeats) >= 10:
            recent_grads = [
                h.grad_norm for h in list(self.heartbeats)[-10:]
                if h.grad_norm is not None
            ]
            if recent_grads and all(g == 0 or g < 1e-10 for g in recent_grads):
                return HangStatus(
                    is_hung=True,
                    hang_type=HangType.GRADIENT_ZERO,
                    reason="Gradients are zero for last 10 steps",
                    seconds_since_update=seconds_since_update,
                    last_loss=self.last_loss,
                    confidence=0.9,
                )

        # No hang detected
        return HangStatus(
            is_hung=False,
            hang_type=HangType.NONE,
            seconds_since_update=seconds_since_update,
            steps_per_second=step_rate or 0.0,
            last_loss=self.last_loss,
            gpu_utilization=self._check_gpu_utilization() if self.check_gpu else None,
        )

    def _calculate_step_rate(self) -> Optional[float]:
        """Calculate steps per second from recent heartbeats."""
        if len(self.heartbeats) < 2:
            return None

        recent = list(self.heartbeats)[-10:]  # Last 10 heartbeats

        if len(recent) < 2:
            return None

        time_span = recent[-1].timestamp - recent[0].timestamp
        step_span = recent[-1].step - recent[0].step

        if time_span <= 0:
            return None

        return step_span / time_span

    def _check_gpu_utilization(self) -> Optional[float]:
        """Check GPU utilization if pynvml available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except:
            return None

    def _training_should_be_active(self) -> bool:
        """Check if training should be actively running."""
        # If we've received heartbeats recently, training should be active
        if not self.heartbeats:
            return False

        recent_heartbeat = list(self.heartbeats)[-1]
        age = time.time() - recent_heartbeat.timestamp

        # If last heartbeat was within timeout, training should be active
        return age < self.heartbeat_timeout

    # ========== Background Monitoring ==========

    def start_monitoring(self, check_interval: float = 5.0) -> None:
        """
        Start background monitoring thread.

        Args:
            check_interval: Seconds between hang checks
        """
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Monitoring already running")
            return

        self._stop_monitoring.clear()

        def monitor_loop():
            while not self._stop_monitoring.is_set():
                status = self.check()
                if status.is_hung:
                    logger.warning(f"HANG DETECTED: {status.reason}")
                    if self.on_hang:
                        self.on_hang(status)
                time.sleep(check_interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started hang monitoring (interval: {check_interval}s)")

    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Stopped hang monitoring")

    # ========== Status File Monitoring ==========

    @classmethod
    def from_status_file(
        cls,
        status_path: str = "status/training_status.json",
        heartbeat_timeout: float = 120.0,
    ) -> "HangDetector":
        """
        Create a detector that monitors a status file.

        Usage:
            detector = HangDetector.from_status_file()
            status = detector.check_status_file()
        """
        detector = cls(heartbeat_timeout=heartbeat_timeout)
        detector._status_path = Path(status_path)
        return detector

    def check_status_file(self) -> HangStatus:
        """
        Check for hang by monitoring status file modification time.
        """
        if not hasattr(self, "_status_path"):
            raise RuntimeError("Use from_status_file() to create detector for file monitoring")

        path = self._status_path

        if not path.exists():
            return HangStatus(
                is_hung=False,
                reason="Status file does not exist (training may not have started)",
            )

        # Check file modification time
        mtime = path.stat().st_mtime
        age = time.time() - mtime

        if age > self.heartbeat_timeout:
            return HangStatus(
                is_hung=True,
                hang_type=HangType.HEARTBEAT_TIMEOUT,
                reason=f"Status file not updated for {age:.0f}s",
                seconds_since_update=age,
                confidence=min(1.0, age / (self.heartbeat_timeout * 2)),
            )

        # Try to read current state from file
        try:
            import json
            with open(path) as f:
                data = json.load(f)

            is_training = data.get("is_training", False)
            current_step = data.get("current_step", 0)
            current_loss = data.get("current_loss")

            # Record as heartbeat
            self.heartbeat(step=current_step, loss=current_loss)

            return HangStatus(
                is_hung=False,
                seconds_since_update=age,
                last_loss=current_loss,
            )

        except Exception as e:
            logger.warning(f"Error reading status file: {e}")
            return HangStatus(
                is_hung=False,
                reason=f"Could not read status file: {e}",
            )

    # ========== Reporting ==========

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of hang detector state."""
        step_rate = self._calculate_step_rate()

        return {
            "heartbeat_count": len(self.heartbeats),
            "last_step": self.last_step,
            "last_loss": self.last_loss,
            "seconds_since_update": time.time() - self.last_heartbeat_time,
            "steps_per_second": step_rate,
            "loss_unchanged_count": self.loss_unchanged_count,
            "hang_detected": self.hang_detected_at is not None,
            "last_status": self.last_hang_status.to_dict() if self.last_hang_status else None,
        }


# ========== Convenience Functions ==========

def check_training_hung(
    status_path: str = "status/training_status.json",
    timeout: float = 120.0,
) -> HangStatus:
    """
    Quick check if training is hung.

    Usage:
        from temple.diagnostics.hang_detector import check_training_hung

        status = check_training_hung()
        if status.is_hung:
            print(f"Training hung: {status.reason}")
    """
    detector = HangDetector.from_status_file(status_path, heartbeat_timeout=timeout)
    return detector.check_status_file()


def monitor_training(
    status_path: str = "status/training_status.json",
    check_interval: float = 30.0,
    timeout: float = 120.0,
    on_hang: Optional[Callable[[HangStatus], None]] = None,
) -> HangDetector:
    """
    Start monitoring training for hangs.

    Usage:
        from temple.diagnostics.hang_detector import monitor_training

        def alert(status):
            print(f"ALERT: {status.reason}")
            # Send slack message, etc.

        detector = monitor_training(on_hang=alert)

        # Later
        detector.stop_monitoring()
    """
    detector = HangDetector.from_status_file(status_path, heartbeat_timeout=timeout)
    detector.on_hang = on_hang

    def file_monitor():
        while not detector._stop_monitoring.is_set():
            status = detector.check_status_file()
            if status.is_hung and on_hang:
                on_hang(status)
            time.sleep(check_interval)

    detector._stop_monitoring = threading.Event()
    detector._monitor_thread = threading.Thread(target=file_monitor, daemon=True)
    detector._monitor_thread.start()

    logger.info(f"Started training monitor (check every {check_interval}s, timeout {timeout}s)")
    return detector


class HangWatchdog:
    """
    A watchdog that auto-restarts training on hang.

    Usage:
        watchdog = HangWatchdog(
            restart_command="python3 core/train.py --resume",
            max_restarts=3,
        )
        watchdog.start()
    """

    def __init__(
        self,
        restart_command: Optional[str] = None,
        max_restarts: int = 3,
        restart_cooldown: float = 300.0,  # 5 minutes between restarts
        status_path: str = "status/training_status.json",
        timeout: float = 180.0,
        check_interval: float = 30.0,
    ):
        self.restart_command = restart_command
        self.max_restarts = max_restarts
        self.restart_cooldown = restart_cooldown
        self.status_path = status_path
        self.timeout = timeout
        self.check_interval = check_interval

        self.restart_count = 0
        self.last_restart: Optional[float] = None
        self.detector: Optional[HangDetector] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the watchdog."""
        self._stop.clear()

        def watchdog_loop():
            while not self._stop.is_set():
                detector = HangDetector.from_status_file(
                    self.status_path,
                    heartbeat_timeout=self.timeout,
                )
                status = detector.check_status_file()

                if status.is_hung:
                    self._handle_hang(status)

                time.sleep(self.check_interval)

        self._thread = threading.Thread(target=watchdog_loop, daemon=True)
        self._thread.start()
        logger.info("Started HangWatchdog")

    def stop(self) -> None:
        """Stop the watchdog."""
        self._stop.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5.0)
        logger.info("Stopped HangWatchdog")

    def _handle_hang(self, status: HangStatus) -> None:
        """Handle a detected hang."""
        logger.warning(f"Hang detected: {status.reason}")

        # Check restart limits
        if self.restart_count >= self.max_restarts:
            logger.error(f"Max restarts ({self.max_restarts}) reached, giving up")
            return

        # Check cooldown
        if self.last_restart:
            elapsed = time.time() - self.last_restart
            if elapsed < self.restart_cooldown:
                logger.info(f"In cooldown ({elapsed:.0f}s / {self.restart_cooldown}s)")
                return

        # Attempt restart
        if self.restart_command:
            self._restart()
        else:
            logger.warning("No restart command configured, just alerting")

    def _restart(self) -> None:
        """Execute restart command."""
        self.restart_count += 1
        self.last_restart = time.time()

        logger.info(f"Restarting training (attempt {self.restart_count}/{self.max_restarts})")
        logger.info(f"Command: {self.restart_command}")

        try:
            import subprocess
            subprocess.Popen(
                self.restart_command,
                shell=True,
                start_new_session=True,
            )
        except Exception as e:
            logger.error(f"Restart failed: {e}")
