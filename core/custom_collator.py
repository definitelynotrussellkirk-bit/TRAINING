"""
Custom Data Collator for Chat Completion Training

Masks instruction/prompt tokens so the model trains ONLY on predicting assistant responses.
Critical for chat fine-tuning: prevents model from learning to generate user prompts.

=== CORE RESPONSIBILITY ===
Answer: "Which tokens should contribute to the training loss?"

Masks all tokens before the assistant's response by setting label = -100 (ignored in loss).
Only assistant response tokens are trained on, ensuring the model learns to respond, not prompt.

=== MASKING STRATEGY ===

Input sequence (ChatML format):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

After masking (labels):
```
[-100, -100, -100, ...              # System message masked
 -100, -100, -100, ...              # User message masked
 -100, -100, -100,                  # "<|im_start|>assistant\n" masked
 The, answer, is, 4, ., <|im_end|>] # Only this part trained
```

Result: Model trains on "The answer is 4." but not on prompts/system messages.

=== ALGORITHM ===

For each example in batch:
1. Tokenize response_template ("<|im_start|>assistant\n") → token IDs
2. Search input_ids for response_template token sequence
3. Find response_start_idx = position after template
4. Set labels[:response_start_idx] = -100 (mask everything before response)
5. Keep labels[response_start_idx:] unchanged (train on response)

Edge case: If template not found, mask entire sequence and warn.

=== WHY THIS MATTERS ===

Without masking:
- Model learns to generate both prompts AND responses
- May start outputting "user:" or "assistant:" prefixes
- Wastes compute training on irrelevant tokens

With masking:
- Model learns pure response generation
- Better instruction following
- 2-3x more efficient (train only on relevant tokens)

=== USAGE EXAMPLE ===
```python
from transformers import AutoTokenizer, Trainer
from core.custom_collator import DataCollatorForCompletionOnly

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-0.5B")

# Create collator
collator = DataCollatorForCompletionOnly(
    tokenizer=tokenizer,
    response_template="<|im_start|>assistant\n"
)

# Use in Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator  # Automatically masks instructions
)
```

=== INTEGRATION POINTS ===
- Used by: core/train.py (HuggingFace Trainer)
- Inputs: Tokenized chat messages (input_ids, attention_mask)
- Outputs: Padded batch with masked labels
- Format: ChatML (<|im_start|>assistant\n marker)

=== RESPONSE TEMPLATE ===
Default: "<|im_start|>assistant\n" (ChatML format for Qwen models)

Customizable for other formats:
- GPT-4: "assistant:"
- Llama: "[/INST]"
- Custom: Any string marking response start

Template is tokenized and searched as token IDs (not string search).
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCompletionOnly:
    """
    Data collator that masks instruction tokens, training only on assistant responses.

    Uses response_template ("<|im_start|>assistant\n") to identify where responses begin.
    All tokens before this marker get label=-100 (ignored in loss computation).

    === ATTRIBUTES ===
    tokenizer: HuggingFace tokenizer (used for padding + tokenizing response_template)
    response_template: String marking response start (default: "<|im_start|>assistant\n")
    ignore_index: Label value for masked tokens (default: -100, ignored by CrossEntropyLoss)

    === DATA FLOW ===
    1. Receive batch of tokenized examples from DataLoader
    2. Pad sequences to same length using default collator
    3. For each example:
       a. Tokenize response_template → [token_id_1, token_id_2, ...]
       b. Search input_ids for template token sequence
       c. Find response_start_idx (position after template)
       d. Set labels[:response_start_idx] = -100
    4. Return batch with masked labels

    === MASKING EXAMPLE ===
    Input (ChatML):
        "<|im_start|>user\nWhat is 2+2?<|im_end|><|im_start|>assistant\n4<|im_end|>"

    Tokenized input_ids:
        [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 151644, 77091, 198, 19, 151645]

    Response template tokens:
        [151644, 77091, 198]  # "<|im_start|>assistant\n"

    Template found at index 11, response starts at index 14.

    Masked labels:
        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  # User message
         -100, -100, -100,                                                      # Template
         19, 151645]                                                            # Response: "4<|im_end|>"

    Result: Loss computed only on tokens [19, 151645] (the "4" response).

    === PADDING & BATCHING ===
    Uses DataCollatorForLanguageModeling internally for padding.
    Handles variable-length sequences automatically.
    Pads on right with tokenizer.pad_token_id.

    === EDGE CASES ===
    1. Template not found → Mask entire sequence, print warning
    2. Multiple responses → Only first response trained (by design)
    3. Empty response → Template found but no tokens after → trains on empty sequence
    """

    tokenizer: PreTrainedTokenizerBase
    response_template: str = "<|im_start|>assistant\n"
    ignore_index: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and mask instruction tokens.

        Pads sequences, creates labels, and masks all tokens before assistant's response.
        Called automatically by HuggingFace Trainer for each training batch.

        Args:
            features: List of tokenized examples, each with:
                - 'input_ids': List[int] - Token IDs
                - 'attention_mask': List[int] - Attention mask (1=real token, 0=padding)

        Returns:
            Dict[str, torch.Tensor]: Padded batch with masked labels
            {
                'input_ids': Tensor[batch_size, max_seq_len],
                'attention_mask': Tensor[batch_size, max_seq_len],
                'labels': Tensor[batch_size, max_seq_len]  # -100 for masked tokens
            }

        Algorithm:
            1. Use DataCollatorForLanguageModeling to pad batch
            2. Tokenize response_template once
            3. For each sequence in batch:
               - Search for response_template token sequence
               - Mask labels[:response_start_idx] = -100
            4. Return batch

        Example:
            >>> collator = DataCollatorForCompletionOnly(tokenizer)
            >>> features = [
            ...     {'input_ids': [1, 2, 3, 4, 5], 'attention_mask': [1, 1, 1, 1, 1]},
            ...     {'input_ids': [6, 7, 8], 'attention_mask': [1, 1, 1]}
            ... ]
            >>> batch = collator(features)
            >>> batch.keys()
            dict_keys(['input_ids', 'attention_mask', 'labels'])
            >>> batch['labels'][:, :3]  # First 3 tokens masked
            tensor([[-100, -100, -100, ...], [-100, -100, -100, ...]])

        Side Effects:
            - Prints warning if response_template not found in sequence
            - Clones labels tensor (original batch unchanged)
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

        # Get the <|im_end|> token ID to find response boundaries
        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        im_end_id = im_end_token[0] if im_end_token else None

        for idx, input_ids in enumerate(batch['input_ids']):
            # Find ALL occurrences of response template (handles packed sequences)
            input_ids_list = input_ids.tolist()

            # Find all positions where response template appears
            response_positions = []
            for i in range(len(input_ids_list) - response_template_length + 1):
                if input_ids_list[i:i + response_template_length] == response_token_ids:
                    # Response starts AFTER the template
                    response_positions.append(i + response_template_length)

            if response_positions:
                # Start with all masked
                labels[idx, :] = self.ignore_index

                # Unmask only the response portions (from template to <|im_end|>)
                for response_start in response_positions:
                    # Find the <|im_end|> token that ends this response
                    response_end = len(input_ids_list)  # Default to end
                    if im_end_id is not None:
                        for j in range(response_start, len(input_ids_list)):
                            if input_ids_list[j] == im_end_id:
                                response_end = j + 1  # Include the <|im_end|> token
                                break

                    # Unmask the response portion (set labels to actual token IDs)
                    labels[idx, response_start:response_end] = batch['input_ids'][idx, response_start:response_end]
            else:
                # Couldn't find template - mask entire sequence to be safe
                print(f"Warning: Could not find response template in example {idx}")
                labels[idx, :] = self.ignore_index

        batch['labels'] = labels
        return batch
