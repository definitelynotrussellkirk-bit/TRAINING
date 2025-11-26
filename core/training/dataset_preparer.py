#!/usr/bin/env python3
"""
Dataset Preparer - Load and prepare datasets for training.

This module handles dataset preparation:
- Loading JSONL files
- Train/validation splitting
- Example transformation (system prompt injection)
- Tokenization and formatting
- Fixed validation set loading

Usage:
    from training.dataset_preparer import DatasetPreparer, DatasetConfig

    config = DatasetConfig(
        dataset_path="/path/to/data.jsonl",
        system_prompt="You are a helpful assistant.",
        validation_split=0.05
    )

    preparer = DatasetPreparer(config, tokenizer)
    result = preparer.prepare()

    train_ds = result.train_dataset
    val_ds = result.val_dataset
"""

import json
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    Configuration for dataset preparation.

    Attributes:
        dataset_path: Path to JSONL training data
        system_prompt: Base system prompt to inject
        validation_split: Fraction for validation (default: 0.05)
        max_val_size: Maximum validation examples (default: 100)
        fixed_validation_path: Optional path to fixed validation set
        shuffle: Whether to shuffle examples (default: True)
        seed: Random seed for reproducibility (default: None = random)
    """
    dataset_path: str
    # Default from core.prompts (single source of truth)
    # Literal here to avoid circular import; should match core.prompts.BASE_PROMPT
    system_prompt: str = "You are happy. You enjoy helping others."
    validation_split: float = 0.05
    max_val_size: int = 100
    fixed_validation_path: Optional[str] = None
    shuffle: bool = True
    seed: Optional[int] = None


@dataclass
class PreparedDataset:
    """
    Result of dataset preparation.

    Attributes:
        train_dataset: HuggingFace Dataset for training
        val_dataset: HuggingFace Dataset for validation
        raw_train_examples: Original train examples (for monitoring)
        raw_val_examples: Original validation examples (for monitoring)
        enforced_system_prompt: The system prompt used
        total_examples: Total number of examples loaded
    """
    train_dataset: Dataset
    val_dataset: Dataset
    raw_train_examples: List[Dict]
    raw_val_examples: List[Dict]
    enforced_system_prompt: str
    total_examples: int


class DatasetPreparer:
    """
    Prepares datasets for training with configurable transformations.

    Handles:
    - Loading JSONL training data
    - Train/validation splitting
    - System prompt injection
    - Example transformation (via callback)
    - Formatting for HuggingFace Trainer

    Example:
        config = DatasetConfig(
            dataset_path="data/train.jsonl",
            system_prompt="You are helpful.",
            validation_split=0.05
        )

        preparer = DatasetPreparer(config, tokenizer)
        result = preparer.prepare()

        # Use with HuggingFace Trainer
        trainer = Trainer(
            train_dataset=result.train_dataset,
            eval_dataset=result.val_dataset,
            ...
        )
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        is_vision_model: bool = False,
        transform_fn: Optional[Callable[[Dict, int, str], Dict]] = None
    ):
        """
        Initialize dataset preparer.

        Args:
            config: Dataset configuration
            tokenizer: Tokenizer for formatting
            is_vision_model: Whether model is a vision-language model
            transform_fn: Optional transform function(example, index, system_prompt) -> example
        """
        self.config = config
        self.tokenizer = tokenizer
        self.is_vision_model = is_vision_model
        self.transform_fn = transform_fn

    def prepare(self) -> PreparedDataset:
        """
        Load and prepare datasets.

        Returns:
            PreparedDataset with train/val datasets and metadata

        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If dataset is empty or invalid
        """
        # Build enforced system prompt with date
        current_date = datetime.now().strftime('%Y-%m-%d')
        enforced_prompt = f"Today is {current_date}. {self.config.system_prompt}"
        logger.info(f"System prompt: {enforced_prompt[:50]}...")

        # Load examples
        examples = self._load_jsonl(self.config.dataset_path)
        total = len(examples)

        if total == 0:
            raise ValueError(f"No examples found in {self.config.dataset_path}")

        logger.info(f"Loaded {total:,} examples")

        # Shuffle if configured
        if self.config.shuffle:
            if self.config.seed is not None:
                random.seed(self.config.seed)
            random.shuffle(examples)
            logger.info("Shuffled examples")

        # Split train/val
        val_size = min(self.config.max_val_size, int(total * self.config.validation_split))
        val_examples = examples[:val_size]
        train_examples = examples[val_size:]

        logger.info(f"Split: {len(train_examples):,} train, {len(val_examples):,} val")

        # Transform examples
        train_examples = self._transform_examples(train_examples, enforced_prompt)
        val_examples = self._transform_examples(val_examples, enforced_prompt)

        # Keep raw examples for monitoring
        raw_train = train_examples.copy()
        raw_val = val_examples.copy()

        # Format for training
        train_formatted = [self._format_example(ex) for ex in train_examples]
        val_formatted = [self._format_example(ex) for ex in val_examples]

        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_formatted)
        val_dataset = Dataset.from_list(val_formatted)

        return PreparedDataset(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            raw_train_examples=raw_train,
            raw_val_examples=raw_val,
            enforced_system_prompt=enforced_prompt,
            total_examples=total
        )

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load examples from JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")

        return examples

    def _transform_examples(
        self, examples: List[Dict], system_prompt: str
    ) -> List[Dict]:
        """Transform examples with system prompt injection and optional transform."""
        transformed = []

        for idx, ex in enumerate(examples):
            # Inject system prompt
            ex = self._inject_system_prompt(ex, system_prompt)

            # Apply custom transform if provided
            if self.transform_fn:
                ex = self.transform_fn(ex, idx, system_prompt)

            # Sanitize
            ex = self._sanitize_example(ex)
            transformed.append(ex)

        return transformed

    def _inject_system_prompt(self, example: Dict, system_prompt: str) -> Dict:
        """Inject or replace system prompt in example."""
        new_ex = dict(example)
        msgs = list(example.get('messages', []))

        if not msgs or msgs[0].get('role') != 'system':
            # Add system message at start
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        else:
            # Replace existing system message
            msgs[0] = {"role": "system", "content": system_prompt}

        new_ex['messages'] = msgs
        return new_ex

    def _sanitize_example(self, example: Dict) -> Dict:
        """Sanitize example content (remove NaN, fix encoding, etc.)."""
        new_ex = dict(example)
        msgs = []

        for msg in example.get('messages', []):
            content = msg.get('content', '')

            # Convert non-string content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            # Remove NaN values (can appear from pandas)
            if content == 'nan' or content == 'NaN':
                content = ''

            # Ensure UTF-8
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')

            msgs.append({
                "role": msg.get('role', 'user'),
                "content": content
            })

        new_ex['messages'] = msgs
        return new_ex

    def _format_example(self, example: Dict) -> Dict:
        """Format example for training using tokenizer."""
        messages = example['messages']

        # Format messages based on model type
        formatted_messages = []
        for msg in messages:
            content = msg.get('content', '')

            # Convert non-string content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            # Wrap as list for vision models
            if self.is_vision_model:
                content = [{"type": "text", "text": content}]

            formatted_messages.append({
                "role": msg['role'],
                "content": content
            })

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": text}

    def load_fixed_validation_set(
        self, validation_path: Optional[str] = None
    ) -> Optional[PreparedDataset]:
        """
        Load a fixed validation set (separate from train split).

        Args:
            validation_path: Path to validation JSONL (optional, uses config default)

        Returns:
            PreparedDataset with validation data, or None if not found
        """
        path = validation_path or self.config.fixed_validation_path
        if not path:
            return None

        path = Path(path)
        if not path.exists():
            logger.warning(f"Fixed validation set not found: {path}")
            return None

        try:
            examples = self._load_jsonl(str(path))

            # Sample for efficiency
            if len(examples) > self.config.max_val_size:
                random.seed(42)  # Reproducible
                examples = random.sample(examples, self.config.max_val_size)

            logger.info(f"Loaded fixed validation set: {len(examples)} examples")

            # Use same enforced prompt
            current_date = datetime.now().strftime('%Y-%m-%d')
            enforced_prompt = f"Today is {current_date}. {self.config.system_prompt}"

            examples = self._transform_examples(examples, enforced_prompt)
            formatted = [self._format_example(ex) for ex in examples]

            return PreparedDataset(
                train_dataset=Dataset.from_list([]),  # Empty
                val_dataset=Dataset.from_list(formatted),
                raw_train_examples=[],
                raw_val_examples=examples,
                enforced_system_prompt=enforced_prompt,
                total_examples=len(examples)
            )

        except Exception as e:
            logger.error(f"Failed to load fixed validation: {e}")
            return None


if __name__ == "__main__":
    # Quick test
    import tempfile

    logging.basicConfig(level=logging.INFO)

    # Create test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(10):
            example = {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"}
                ]
            }
            f.write(json.dumps(example) + '\n')
        test_file = f.name

    # Mock tokenizer
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return f"<formatted>{len(messages)} messages</formatted>"

    # Test preparation
    config = DatasetConfig(
        dataset_path=test_file,
        system_prompt="Test prompt",
        validation_split=0.2
    )

    preparer = DatasetPreparer(config, MockTokenizer())
    result = preparer.prepare()

    print(f"\nResults:")
    print(f"  Total: {result.total_examples}")
    print(f"  Train: {len(result.train_dataset)}")
    print(f"  Val: {len(result.val_dataset)}")
    print(f"  System prompt: {result.enforced_system_prompt[:50]}...")

    # Cleanup
    import os
    os.unlink(test_file)

    print("\nDatasetPreparer ready for use!")
