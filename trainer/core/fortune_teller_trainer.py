"""
Custom Trainer with Fortune Teller Loss Support

Extends HuggingFace Trainer to support surprise-weighted training.
"""

import torch
from typing import Dict, Optional, Any, Tuple, Union
from transformers import Trainer

from trainer.losses import FortuneTellerLoss, FortuneTellerTracker


class FortuneTellerTrainer(Trainer):
    """
    Custom Trainer that uses surprise-weighted loss.

    Instead of treating all tokens equally, focuses gradient updates
    on tokens that surprise the model (high uncertainty/entropy).

    Usage:
        loss_config = {
            "surprise_metric": "entropy",
            "min_surprise": 0.1,
            "normalize_batch": True,
            "temperature": 1.0
        }

        trainer = FortuneTellerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss_config=loss_config,
        )
    """

    def __init__(
        self,
        loss_config: Optional[Dict[str, Any]] = None,
        tracker: Optional[FortuneTellerTracker] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            loss_config: Configuration for FortuneTellerLoss (surprise_metric, min_surprise, etc.)
            tracker: Optional FortuneTellerTracker for metrics logging
            *args, **kwargs: Passed to parent Trainer
        """
        super().__init__(*args, **kwargs)

        # Setup Fortune Teller loss
        loss_config = loss_config or {}
        self.fortune_teller_loss = FortuneTellerLoss(**loss_config)

        # Setup tracker
        self.tracker = tracker or FortuneTellerTracker()

        # Move loss to same device as model
        if hasattr(self.model, 'device'):
            self.fortune_teller_loss = self.fortune_teller_loss.to(self.model.device)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override compute_loss to use Fortune Teller loss.

        Args:
            model: The model to train
            inputs: Dict with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs

        Returns:
            loss (and optionally outputs)
        """
        # Extract labels
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute Fortune Teller loss
        loss, details = self.fortune_teller_loss(
            logits=logits,
            labels=labels,
            return_details=True
        )

        # Log metrics (if tracker provided)
        if self.tracker is not None and self.state.global_step % 10 == 0:
            self.tracker.update(self.state.global_step, details)

        # Also log to HF Trainer logs
        if self.state.global_step % 100 == 0 and details:
            self.log({
                "fortune_teller/avg_surprise": details["avg_surprise"],
                "fortune_teller/surprise_std": details["surprise_std"],
                "fortune_teller/avg_weight": details["avg_weight"],
                "fortune_teller/avg_ce_loss": details["avg_ce_loss"],
            })

        # Restore labels for compatibility
        inputs["labels"] = labels

        if return_outputs:
            return loss, outputs
        return loss

    def get_fortune_teller_stats(self) -> Dict[str, float]:
        """Get accumulated Fortune Teller statistics."""
        return self.fortune_teller_loss.get_stats()

    def reset_fortune_teller_stats(self):
        """Reset accumulated Fortune Teller statistics."""
        self.fortune_teller_loss.reset_stats()

    def save_fortune_teller_history(self, path: str):
        """Save Fortune Teller metrics history to file."""
        if self.tracker:
            self.tracker.save(path)
