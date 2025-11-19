"""
Custom Data Collator for Chat Completion Training

Masks the instruction/prompt portion so the model only learns to predict
the assistant's response, not the entire conversation.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCompletionOnly:
    """
    Data collator that masks everything before the assistant's response.

    For Qwen models using ChatML format, responses start after:
    "<|im_start|>assistant\n"

    Everything before this marker gets label = -100 (ignored in loss).
    Only the assistant's response is trained on.
    """

    tokenizer: PreTrainedTokenizerBase
    response_template: str = "<|im_start|>assistant\n"
    ignore_index: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of already-tokenized examples.

        Args:
            features: List of dicts with 'input_ids' and 'attention_mask' from tokenization

        Returns:
            Dict with 'input_ids', 'attention_mask', and 'labels' tensors
        """
        # Features are already tokenized, just need to pad and create batch
        from transformers import DataCollatorForLanguageModeling

        # Use default collator for padding
        default_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        batch = default_collator(features)

        # batch now has 'input_ids', 'attention_mask', and 'labels'
        # Need to mask the instruction portion in labels

        # Tokenize the response template to find where responses start
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )
        response_template_length = len(response_token_ids)

        # For each example in batch, mask everything before the response
        labels = batch['labels'].clone()

        for idx, input_ids in enumerate(batch['input_ids']):
            # Find where the response template appears in the token sequence
            input_ids_list = input_ids.tolist()

            # Search for the response template token sequence
            response_start_idx = None
            for i in range(len(input_ids_list) - response_template_length + 1):
                if input_ids_list[i:i + response_template_length] == response_token_ids:
                    # Found it! Response starts AFTER the template
                    response_start_idx = i + response_template_length
                    break

            if response_start_idx is not None:
                # Mask everything before the response (including the template itself)
                labels[idx, :response_start_idx] = self.ignore_index
            else:
                # Couldn't find template - this shouldn't happen with proper data
                # but if it does, mask the entire sequence to be safe
                print(f"Warning: Could not find response template in example {idx}")
                labels[idx, :] = self.ignore_index

        batch['labels'] = labels
        return batch
