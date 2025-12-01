"""
THE FORTUNE TELLER - Surprise-Weighted Training

The Fortune Teller predicts what the hero will struggle with,
and focuses gradient updates there.

Core Idea:
- Tokens that are well-predicted (low surprise) contribute less to gradients
- Tokens that are uncertain/surprising contribute more to gradients
- Automatic curriculum: naturally focuses on hard parts as easy parts are mastered

Surprise Metrics:
1. entropy: H = -Σ p(x) log p(x) - high entropy = high uncertainty
2. confidence: 1 - max(p) - low confidence = high surprise
3. perplexity: exp(-log_p(correct)) - high perplexity = surprising
4. margin: p(correct) - p(second_best) - small margin = uncertain

Edge Cases Handled:
- Overconfident wrong predictions: multiply surprise by loss
- All low surprise: normalize to [min_weight, 1.0] range
- All high surprise: clips at max, falls back to normal training
- Batch normalization: normalize surprises within each batch

Usage:
    # In config.json
    {
        "loss": {
            "type": "fortune_teller",
            "surprise_metric": "entropy",
            "min_surprise": 0.1,
            "normalize_batch": true,
            "temperature": 1.0
        }
    }

    # Or directly
    loss_fn = FortuneTellerLoss(
        surprise_metric="entropy",
        min_surprise=0.1,
        normalize_batch=True,
        temperature=1.0
    )

    loss, details = loss_fn(logits, labels, return_details=True)
    print(f"Avg surprise: {details['avg_surprise']:.3f}")
    print(f"Surprise std: {details['surprise_std']:.3f}")
"""

from enum import Enum
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurpriseMetric(str, Enum):
    """Available surprise metrics."""
    ENTROPY = "entropy"
    CONFIDENCE = "confidence"
    PERPLEXITY = "perplexity"
    MARGIN = "margin"


class FortuneTellerLoss(nn.Module):
    """
    Surprise-weighted cross-entropy loss.

    Focuses gradient updates on tokens that surprise the model,
    allowing efficient learning by not wasting updates on already-mastered patterns.
    """

    def __init__(
        self,
        surprise_metric: str = "entropy",
        min_surprise: float = 0.1,
        max_surprise: float = 10.0,
        normalize_batch: bool = True,
        temperature: float = 1.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        """
        Args:
            surprise_metric: Which metric to use for surprise (entropy, confidence, perplexity, margin)
            min_surprise: Minimum surprise weight (prevents vanishing gradients)
            max_surprise: Maximum surprise weight (prevents explosion)
            normalize_batch: Normalize surprises within each batch
            temperature: Temperature for scaling surprise (higher = more uniform)
            ignore_index: Index to ignore in loss calculation (padding)
            reduction: How to reduce loss (mean, sum, none)
        """
        super().__init__()

        if surprise_metric not in [m.value for m in SurpriseMetric]:
            raise ValueError(f"Unknown surprise_metric: {surprise_metric}. Use {[m.value for m in SurpriseMetric]}")

        self.surprise_metric = surprise_metric
        self.min_surprise = min_surprise
        self.max_surprise = max_surprise
        self.normalize_batch = normalize_batch
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.reduction = reduction

        # Track statistics for monitoring
        self.register_buffer("total_tokens", torch.tensor(0.0))
        self.register_buffer("total_surprise", torch.tensor(0.0))
        self.register_buffer("total_loss", torch.tensor(0.0))

    def compute_entropy(self, probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Compute entropy: H = -Σ p(x) log p(x)

        High entropy = model is uncertain = high surprise
        Range: [0, log(vocab_size)]
        """
        log_probs = torch.log(probs + eps)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def compute_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence: 1 - max(p)

        Low max probability = not confident = high surprise
        Range: [0, 1]
        """
        max_prob, _ = probs.max(dim=-1)
        surprise = 1.0 - max_prob
        return surprise

    def compute_perplexity(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute perplexity: exp(-log_p(correct_token))

        High perplexity = surprising
        Range: [1, inf]
        """
        # Gather probabilities of correct tokens
        correct_probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        perplexity = torch.exp(-torch.log(correct_probs + 1e-10))
        return perplexity

    def compute_margin(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute margin: p(correct) - p(second_best)

        Small margin = uncertain between options = high surprise
        We return (1 - margin) so high surprise = high value
        Range: [0, 1] after transformation
        """
        # Get probability of correct token
        correct_probs = torch.gather(probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Get max probability (might be correct token or another)
        max_prob, _ = probs.max(dim=-1)

        # If correct token is max, get second best
        # Set correct token prob to 0 temporarily
        probs_copy = probs.clone()
        probs_copy.scatter_(dim=-1, index=labels.unsqueeze(-1), value=0.0)
        second_best, _ = probs_copy.max(dim=-1)

        # Margin is difference between correct and second best
        margin = correct_probs - second_best

        # Convert to surprise: small margin = high surprise
        surprise = 1.0 - margin
        return surprise.clamp(0.0, 1.0)

    def compute_surprise(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token surprise based on configured metric.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]

        Returns:
            surprise: [batch_size, seq_len] - per-token surprise values
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute surprise based on metric
        if self.surprise_metric == SurpriseMetric.ENTROPY:
            surprise = self.compute_entropy(probs)
        elif self.surprise_metric == SurpriseMetric.CONFIDENCE:
            surprise = self.compute_confidence(probs)
        elif self.surprise_metric == SurpriseMetric.PERPLEXITY:
            surprise = self.compute_perplexity(probs, labels)
        elif self.surprise_metric == SurpriseMetric.MARGIN:
            surprise = self.compute_margin(probs, labels)
        else:
            raise ValueError(f"Unknown surprise metric: {self.surprise_metric}")

        return surprise

    def normalize_surprise(
        self,
        surprise: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize surprise values to prevent vanishing/exploding gradients.

        Args:
            surprise: [batch_size, seq_len] - raw surprise values
            mask: [batch_size, seq_len] - valid token mask

        Returns:
            normalized_surprise: [batch_size, seq_len] - surprise in [min_surprise, max_surprise]
        """
        if not self.normalize_batch:
            # Just clip to min/max
            return surprise.clamp(self.min_surprise, self.max_surprise)

        # Normalize within batch (only over valid tokens)
        if mask.sum() == 0:
            return surprise.clamp(self.min_surprise, self.max_surprise)

        # Get statistics over valid tokens
        valid_surprise = surprise[mask]
        surprise_mean = valid_surprise.mean()
        surprise_std = valid_surprise.std()

        # Standardize
        if surprise_std > 1e-6:
            surprise = (surprise - surprise_mean) / (surprise_std + 1e-6)

        # Apply temperature (higher temp = more uniform weights)
        if self.temperature != 1.0:
            surprise = surprise / self.temperature

        # Scale to [min_surprise, max_surprise] range
        # Use sigmoid to map to [0, 1] then scale
        surprise = torch.sigmoid(surprise)
        surprise = self.min_surprise + surprise * (self.max_surprise - self.min_surprise)

        return surprise

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute surprise-weighted cross-entropy loss.

        Args:
            logits: [batch_size, seq_len, vocab_size] - model predictions
            labels: [batch_size, seq_len] - ground truth tokens
            return_details: if True, return (loss, details_dict) else just loss

        Returns:
            loss: scalar tensor (if reduction != 'none')
            details: dict with surprise statistics (if return_details=True)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Create mask for valid tokens (not padding)
        mask = labels != self.ignore_index

        # Compute per-token surprise
        surprise = self.compute_surprise(logits, labels)

        # Normalize surprise
        surprise_weights = self.normalize_surprise(surprise, mask)

        # Compute standard cross-entropy loss per token
        # Shape: [batch_size, seq_len]
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='none'
        ).view(batch_size, seq_len)

        # Weight loss by surprise
        weighted_loss = ce_loss * surprise_weights

        # Apply mask
        weighted_loss = weighted_loss * mask.float()

        # Reduce
        if self.reduction == "mean":
            loss = weighted_loss.sum() / (mask.sum() + 1e-10)
        elif self.reduction == "sum":
            loss = weighted_loss.sum()
        else:  # none
            loss = weighted_loss

        # Update statistics
        if self.training:
            num_tokens = mask.sum()
            self.total_tokens += num_tokens
            self.total_surprise += (surprise * mask.float()).sum()
            self.total_loss += weighted_loss.sum()

        # Prepare details if requested
        if return_details:
            valid_surprise = surprise[mask]
            valid_weights = surprise_weights[mask]
            valid_loss = ce_loss[mask]

            details = {
                "avg_surprise": valid_surprise.mean().item() if valid_surprise.numel() > 0 else 0.0,
                "surprise_std": valid_surprise.std().item() if valid_surprise.numel() > 0 else 0.0,
                "avg_weight": valid_weights.mean().item() if valid_weights.numel() > 0 else 0.0,
                "weight_std": valid_weights.std().item() if valid_weights.numel() > 0 else 0.0,
                "avg_ce_loss": valid_loss.mean().item() if valid_loss.numel() > 0 else 0.0,
                "max_surprise": valid_surprise.max().item() if valid_surprise.numel() > 0 else 0.0,
                "min_surprise": valid_surprise.min().item() if valid_surprise.numel() > 0 else 0.0,
                "num_tokens": mask.sum().item(),
                "surprise_metric": self.surprise_metric,
            }
            return loss, details

        return loss

    def get_stats(self) -> Dict[str, float]:
        """Get accumulated statistics."""
        if self.total_tokens == 0:
            return {
                "avg_surprise": 0.0,
                "avg_loss": 0.0,
                "total_tokens": 0,
            }

        return {
            "avg_surprise": (self.total_surprise / self.total_tokens).item(),
            "avg_loss": (self.total_loss / self.total_tokens).item(),
            "total_tokens": self.total_tokens.item(),
        }

    def reset_stats(self):
        """Reset accumulated statistics."""
        self.total_tokens.zero_()
        self.total_surprise.zero_()
        self.total_loss.zero_()


class FortuneTellerTracker:
    """
    Tracks Fortune Teller metrics over training for analysis and visualization.

    Usage:
        tracker = FortuneTellerTracker()

        # During training
        _, details = loss_fn(logits, labels, return_details=True)
        tracker.update(step, details)

        # Get statistics
        stats = tracker.get_stats(window=100)
        print(f"Recent avg surprise: {stats['avg_surprise']:.3f}")

        # Save to disk
        tracker.save("fortune_teller_metrics.json")
    """

    def __init__(self):
        self.history = []

    def update(self, step: int, details: Dict):
        """Record metrics from a training step."""
        record = {
            "step": step,
            **details
        }
        self.history.append(record)

    def get_stats(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Get statistics over recent history.

        Args:
            window: Number of recent steps to include (None = all)

        Returns:
            stats: dict with aggregated metrics
        """
        if not self.history:
            return {}

        records = self.history[-window:] if window else self.history

        # Average over records
        keys = ["avg_surprise", "surprise_std", "avg_weight", "weight_std", "avg_ce_loss"]
        stats = {}

        for key in keys:
            values = [r[key] for r in records if key in r]
            if values:
                stats[key] = sum(values) / len(values)

        stats["num_records"] = len(records)
        return stats

    def save(self, path: str):
        """Save history to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str):
        """Load history from JSON file."""
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)
