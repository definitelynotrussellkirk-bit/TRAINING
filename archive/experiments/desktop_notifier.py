#!/usr/bin/env python3
"""
Desktop Notifier - Send native desktop notifications

Sends notifications for:
- Training crashes
- Training completion
- OOM errors
- Critical failures

Works on Linux (notify-send), macOS (osascript), Windows (win10toast).
"""

import subprocess
import sys
import platform
from typing import Optional
from dataclasses import dataclass


@dataclass
class Notification:
    title: str
    message: str
    urgency: str = "normal"  # low, normal, critical


class DesktopNotifier:
    """Cross-platform desktop notification system."""

    def __init__(self):
        self.system = platform.system()
        self._check_availability()

    def _check_availability(self) -> bool:
        """Check if notifications are available on this system."""
        if self.system == "Linux":
            try:
                subprocess.run(["which", "notify-send"],
                             capture_output=True, check=True)
                return True
            except subprocess.CalledProcessError:
                print("⚠️  notify-send not found. Install with: sudo apt install libnotify-bin")
                return False
        elif self.system == "Darwin":  # macOS
            return True  # osascript is built-in
        elif self.system == "Windows":
            try:
                import win10toast
                return True
            except ImportError:
                print("⚠️  win10toast not found. Install with: pip install win10toast")
                return False
        return False

    def send(self, notification: Notification) -> bool:
        """Send a desktop notification."""
        try:
            if self.system == "Linux":
                return self._send_linux(notification)
            elif self.system == "Darwin":
                return self._send_macos(notification)
            elif self.system == "Windows":
                return self._send_windows(notification)
            else:
                print(f"⚠️  Notifications not supported on {self.system}")
                return False
        except Exception as e:
            print(f"⚠️  Failed to send notification: {e}")
            return False

    def _send_linux(self, notif: Notification) -> bool:
        """Send notification on Linux using notify-send."""
        cmd = [
            "notify-send",
            "-u", notif.urgency,
            "-a", "Ultimate Trainer",
            notif.title,
            notif.message
        ]

        subprocess.run(cmd, capture_output=True)
        return True

    def _send_macos(self, notif: Notification) -> bool:
        """Send notification on macOS using osascript."""
        script = f'display notification "{notif.message}" with title "{notif.title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True)
        return True

    def _send_windows(self, notif: Notification) -> bool:
        """Send notification on Windows using win10toast."""
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            notif.title,
            notif.message,
            duration=10,
            threaded=True
        )
        return True

    # Convenience methods for common notifications

    def training_crashed(self, error: str):
        """Notify that training crashed."""
        self.send(Notification(
            title="Training CRASHED!",
            message=f"Error: {error[:100]}",
            urgency="critical"
        ))

    def training_complete(self, duration: str):
        """Notify that training completed successfully."""
        self.send(Notification(
            title="Training Complete!",
            message=f"Training finished in {duration}",
            urgency="normal"
        ))

    def oom_error(self):
        """Notify about out-of-memory error."""
        self.send(Notification(
            title="OUT OF MEMORY!",
            message="Training failed - GPU ran out of memory. Reduce batch size or LoRA rank.",
            urgency="critical"
        ))

    def validation_failed(self, issues: int):
        """Notify about validation failures."""
        self.send(Notification(
            title="Validation Failed",
            message=f"Found {issues} issues in dataset. Check logs.",
            urgency="normal"
        ))


def test_notifications():
    """Test notification system."""
    notifier = DesktopNotifier()

    print("Testing notifications...")

    # Test normal notification
    notifier.send(Notification(
        title="Test Notification",
        message="This is a test from Ultimate Trainer",
        urgency="normal"
    ))
    print("✓ Sent test notification")

    # Test crash notification
    notifier.training_crashed("CUDA out of memory")
    print("✓ Sent crash notification")


if __name__ == "__main__":
    test_notifications()
