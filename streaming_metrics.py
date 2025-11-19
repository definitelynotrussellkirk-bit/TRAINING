#!/usr/bin/env python3
"""
Streaming Metrics - Smooth Signal from Noisy Training

Provides:
1. Streaming Cross-Entropy: EMA-smoothed loss for cleaner trends
2. Token Entropy: Per-token prediction uncertainty
3. Calibration: Are confidence scores meaningful?

Why streaming?
- Per-batch loss is noisy (depends on batch composition)
- EMA smoothing reveals true trends
- Easier to detect plateaus and overfitting
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class StreamingMetricsTracker:
    """Track smoothed metrics during training."""

    def __init__(self, ema_alpha: float = 0.1, window_size: int = 50):
        """
        Initialize streaming metrics tracker.

        Args:
            ema_alpha: EMA smoothing factor (0.1 = 10% new, 90% old)
            window_size: Window size for moving statistics
        """
        self.ema_alpha = ema_alpha
        self.window_size = window_size

        # Streaming cross-entropy (EMA smoothed)
        self.streaming_ce = None
        self.streaming_ce_history = deque(maxlen=1000)

        # Raw loss history for comparison
        self.raw_loss_history = deque(maxlen=1000)

        # Token entropy tracking
        self.token_entropy_history = deque(maxlen=1000)

        # Windowed statistics
        self.recent_losses = deque(maxlen=window_size)
        self.recent_entropies = deque(maxlen=window_size)

        print(f"📊 Streaming Metrics initialized (α={ema_alpha}, window={window_size})")

    def update_loss(self, loss: float) -> Dict[str, float]:
        """
        Update streaming cross-entropy with new loss value.

        Args:
            loss: Current batch loss

        Returns:
            Dict with smoothed and raw values
        """
        # Update EMA
        if self.streaming_ce is None:
            self.streaming_ce = loss
        else:
            self.streaming_ce = (
                self.ema_alpha * loss +
                (1 - self.ema_alpha) * self.streaming_ce
            )

        # Store in history
        self.streaming_ce_history.append(self.streaming_ce)
        self.raw_loss_history.append(loss)
        self.recent_losses.append(loss)

        return {
            "streaming_ce": float(self.streaming_ce),
            "raw_loss": float(loss),
            "loss_variance": float(np.var(list(self.recent_losses))) if len(self.recent_losses) > 1 else 0.0
        }

    def calculate_token_entropy(
        self,
        logits: torch.Tensor,
        average: bool = True
    ) -> float:
        """
        Calculate entropy of next-token predictions.

        High entropy = model is uncertain
        Low entropy = model is confident

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            average: If True, return mean entropy across tokens

        Returns:
            Entropy value (averaged if requested)
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Calculate entropy: H(p) = -sum(p * log(p))
        # Use torch.clamp to avoid log(0)
        entropy = -torch.sum(
            probs * torch.log(torch.clamp(probs, min=1e-10)),
            dim=-1
        )

        # Average across batch and sequence if requested
        if average:
            entropy = entropy.mean().item()
        else:
            entropy = entropy.cpu().numpy()

        # Store in history
        if isinstance(entropy, float):
            self.token_entropy_history.append(entropy)
            self.recent_entropies.append(entropy)

        return entropy

    def get_trends(self) -> Dict[str, str]:
        """
        Detect trends in metrics (improving, stable, degrading).

        Returns:
            Dict with trend indicators
        """
        trends = {}

        # Loss trend (lower is better)
        if len(self.streaming_ce_history) >= 20:
            recent_avg = np.mean(list(self.streaming_ce_history)[-10:])
            older_avg = np.mean(list(self.streaming_ce_history)[-20:-10])

            if recent_avg < older_avg * 0.95:
                trends["loss_trend"] = "improving"
            elif recent_avg > older_avg * 1.05:
                trends["loss_trend"] = "degrading"
            else:
                trends["loss_trend"] = "stable"
        else:
            trends["loss_trend"] = "insufficient_data"

        # Entropy trend (stable is usually good)
        if len(self.token_entropy_history) >= 20:
            recent_avg = np.mean(list(self.token_entropy_history)[-10:])
            older_avg = np.mean(list(self.token_entropy_history)[-20:-10])

            if abs(recent_avg - older_avg) < older_avg * 0.05:
                trends["entropy_trend"] = "stable"
            elif recent_avg < older_avg:
                trends["entropy_trend"] = "decreasing"  # More confident
            else:
                trends["entropy_trend"] = "increasing"  # Less confident
        else:
            trends["entropy_trend"] = "insufficient_data"

        return trends

    def detect_plateau(self, threshold: float = 0.02, window: int = 20) -> bool:
        """
        Detect if loss has plateaued (not improving).

        Args:
            threshold: Minimum relative improvement to not be plateau
            window: Number of recent steps to check

        Returns:
            True if plateaued, False otherwise
        """
        if len(self.streaming_ce_history) < window:
            return False

        recent_window = list(self.streaming_ce_history)[-window:]
        first_half = np.mean(recent_window[:window//2])
        second_half = np.mean(recent_window[window//2:])

        # Check if improvement is less than threshold
        if first_half == 0:
            return False

        relative_improvement = (first_half - second_half) / first_half

        return relative_improvement < threshold

    def get_statistics(self) -> Dict[str, float]:
        """
        Get current statistics for all tracked metrics.

        Returns:
            Dict with comprehensive statistics
        """
        stats = {}

        # Streaming CE stats
        if self.streaming_ce is not None:
            stats["streaming_ce"] = float(self.streaming_ce)

        # Loss variance (noise level)
        if len(self.recent_losses) > 1:
            stats["loss_variance"] = float(np.var(list(self.recent_losses)))
            stats["loss_std"] = float(np.std(list(self.recent_losses)))
        else:
            stats["loss_variance"] = 0.0
            stats["loss_std"] = 0.0

        # Entropy stats
        if len(self.recent_entropies) > 0:
            stats["mean_entropy"] = float(np.mean(list(self.recent_entropies)))
            stats["entropy_std"] = float(np.std(list(self.recent_entropies))) if len(self.recent_entropies) > 1 else 0.0
        else:
            stats["mean_entropy"] = 0.0
            stats["entropy_std"] = 0.0

        # Trend indicators
        stats.update(self.get_trends())

        # Plateau detection
        stats["is_plateaued"] = self.detect_plateau()

        return stats

    def get_history(self, metric: str = "streaming_ce", last_n: int = 100) -> List[float]:
        """
        Get historical values for a metric.

        Args:
            metric: Metric name ("streaming_ce", "raw_loss", "token_entropy")
            last_n: Number of recent values to return

        Returns:
            List of historical values
        """
        if metric == "streaming_ce":
            history = list(self.streaming_ce_history)
        elif metric == "raw_loss":
            history = list(self.raw_loss_history)
        elif metric == "token_entropy":
            history = list(self.token_entropy_history)
        else:
            return []

        return history[-last_n:]

    def calculate_calibration_error(
        self,
        confidences: torch.Tensor,
        correctness: torch.Tensor,
        num_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures if confidence scores match actual accuracy.
        Example: If model is 80% confident, it should be right 80% of time.

        Args:
            confidences: Model confidence scores [N]
            correctness: Binary correctness (1=correct, 0=wrong) [N]
            num_bins: Number of confidence bins

        Returns:
            ECE value (0 = perfectly calibrated, 1 = worst)
        """
        # Ensure tensors are on CPU
        confidences = confidences.cpu()
        correctness = correctness.cpu()

        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(confidences)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = correctness[in_bin].float().mean()

                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()

                # Add weighted difference to ECE
                ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

        return float(ece)


def create_streaming_tracker(ema_alpha: float = 0.1, window_size: int = 50) -> StreamingMetricsTracker:
    """
    Factory function to create a streaming metrics tracker.

    Args:
        ema_alpha: EMA smoothing factor
        window_size: Window size for moving statistics

    Returns:
        StreamingMetricsTracker instance
    """
    return StreamingMetricsTracker(ema_alpha=ema_alpha, window_size=window_size)


if __name__ == "__main__":
    print("Streaming Metrics Tracker - Test Mode")
    print("This module should be imported and used during training.")

    # Quick test
    tracker = create_streaming_tracker()

    print("\nSimulating training loss updates:")
    for step in range(20):
        # Simulate decreasing loss with noise
        loss = 2.0 * np.exp(-step/10) + np.random.normal(0, 0.1)
        result = tracker.update_loss(loss)
        print(f"Step {step}: Raw={result['raw_loss']:.4f}, Smoothed={result['streaming_ce']:.4f}")

    print("\nStatistics:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
