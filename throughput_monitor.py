#!/usr/bin/env python3
"""
Throughput Monitoring

Tracks training speed in tokens/second:
- Current throughput
- Average throughput
- Trend detection (slowing down vs speeding up)
- ETA calculation improvements

Helps identify:
- Performance degradation
- Optimal batch sizes
- When to scale up/down resources
"""

import time
from typing import Dict, List, Optional
from collections import deque
import numpy as np


class ThroughputMonitor:
    """Monitor training throughput (tokens/sec)."""

    def __init__(self, window_size: int = 50):
        """
        Initialize throughput monitor.

        Args:
            window_size: Number of recent measurements to track
        """
        self.window_size = window_size

        # Track tokens processed and timestamps
        self.step_timestamps = deque(maxlen=window_size)
        self.step_token_counts = deque(maxlen=window_size)

        # Overall statistics
        self.total_tokens = 0
        self.start_time = None
        self.last_update_time = None

        # Current throughput (smoothed)
        self.current_throughput = 0.0

        print(f"⏱️  Throughput Monitor initialized (window={window_size})")

    def update(self, num_tokens: int, timestamp: Optional[float] = None):
        """
        Update with tokens processed in current step.

        Args:
            num_tokens: Number of tokens processed
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize on first update
        if self.start_time is None:
            self.start_time = timestamp

        # Store this step's data
        self.step_timestamps.append(timestamp)
        self.step_token_counts.append(num_tokens)
        self.total_tokens += num_tokens
        self.last_update_time = timestamp

        # Calculate current throughput if we have enough data
        if len(self.step_timestamps) >= 2:
            # Use recent window for current throughput
            window_start = self.step_timestamps[0]
            window_end = self.step_timestamps[-1]
            window_duration = window_end - window_start

            if window_duration > 0:
                window_tokens = sum(self.step_token_counts)
                self.current_throughput = window_tokens / window_duration

    def get_current_throughput(self) -> float:
        """Get current throughput (tokens/sec) based on recent window."""
        return self.current_throughput

    def get_average_throughput(self) -> float:
        """Get overall average throughput (tokens/sec) since start."""
        if self.start_time is None or self.last_update_time is None:
            return 0.0

        total_duration = self.last_update_time - self.start_time
        if total_duration <= 0:
            return 0.0

        return self.total_tokens / total_duration

    def get_trend(self) -> str:
        """
        Detect throughput trend.

        Returns:
            "improving", "stable", "degrading", or "insufficient_data"
        """
        if len(self.step_timestamps) < 20:
            return "insufficient_data"

        # Compare first half vs second half of window
        mid_idx = len(self.step_timestamps) // 2

        # First half throughput
        first_half_start = self.step_timestamps[0]
        first_half_end = self.step_timestamps[mid_idx - 1]
        first_half_duration = first_half_end - first_half_start
        first_half_tokens = sum(list(self.step_token_counts)[:mid_idx])

        # Second half throughput
        second_half_start = self.step_timestamps[mid_idx]
        second_half_end = self.step_timestamps[-1]
        second_half_duration = second_half_end - second_half_start
        second_half_tokens = sum(list(self.step_token_counts)[mid_idx:])

        if first_half_duration <= 0 or second_half_duration <= 0:
            return "insufficient_data"

        first_throughput = first_half_tokens / first_half_duration
        second_throughput = second_half_tokens / second_half_duration

        # Compare
        relative_change = (second_throughput - first_throughput) / first_throughput

        if relative_change > 0.1:
            return "improving"
        elif relative_change < -0.1:
            return "degrading"
        else:
            return "stable"

    def estimate_time_remaining(
        self,
        remaining_tokens: int,
        use_current: bool = True
    ) -> float:
        """
        Estimate time remaining based on throughput.

        Args:
            remaining_tokens: Number of tokens left to process
            use_current: Use current throughput (True) vs average (False)

        Returns:
            Estimated seconds remaining
        """
        throughput = self.get_current_throughput() if use_current else self.get_average_throughput()

        if throughput <= 0:
            return 0.0

        return remaining_tokens / throughput

    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive throughput statistics."""
        stats = {
            "current_tokens_per_sec": self.get_current_throughput(),
            "average_tokens_per_sec": self.get_average_throughput(),
            "total_tokens_processed": self.total_tokens,
            "trend": self.get_trend(),
        }

        # Calculate efficiency (current vs average)
        if stats["average_tokens_per_sec"] > 0:
            stats["efficiency_ratio"] = stats["current_tokens_per_sec"] / stats["average_tokens_per_sec"]
        else:
            stats["efficiency_ratio"] = 1.0

        # Time elapsed
        if self.start_time and self.last_update_time:
            stats["elapsed_seconds"] = self.last_update_time - self.start_time
        else:
            stats["elapsed_seconds"] = 0.0

        return stats

    def format_throughput(self, tokens_per_sec: float) -> str:
        """Format throughput for human reading."""
        if tokens_per_sec >= 1000:
            return f"{tokens_per_sec/1000:.2f}K tok/s"
        else:
            return f"{tokens_per_sec:.1f} tok/s"

    def format_duration(self, seconds: float) -> str:
        """Format duration for human reading."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def get_summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_statistics()
        current = self.format_throughput(stats["current_tokens_per_sec"])
        average = self.format_throughput(stats["average_tokens_per_sec"])
        trend = stats["trend"]
        elapsed = self.format_duration(stats["elapsed_seconds"])

        return f"Throughput: {current} (avg: {average}, {trend}, elapsed: {elapsed})"


def create_throughput_monitor(window_size: int = 50) -> ThroughputMonitor:
    """
    Factory function to create a throughput monitor.

    Args:
        window_size: Number of recent measurements to track

    Returns:
        ThroughputMonitor instance
    """
    return ThroughputMonitor(window_size=window_size)


if __name__ == "__main__":
    print("Throughput Monitor - Test Mode")
    print("This module should be imported and used during training.")

    # Quick test
    monitor = create_throughput_monitor(window_size=10)

    print("\nSimulating training steps:")
    start = time.time()
    for step in range(20):
        # Simulate processing 1000 tokens per step
        num_tokens = 1000
        timestamp = start + step * 0.5  # 0.5 seconds per step
        monitor.update(num_tokens, timestamp=timestamp)

        if step % 5 == 0:
            print(f"Step {step}: {monitor.get_summary()}")

    print("\nFinal statistics:")
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
