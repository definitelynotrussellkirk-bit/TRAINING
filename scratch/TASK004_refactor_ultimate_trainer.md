# TASK004: Refactor UltimateTrainer

**Priority:** MEDIUM
**Effort:** 6 hours
**Dependencies:** None
**Files:** `core/train.py`

---

## Problem

`UltimateTrainer` in `core/train.py` is a "god object" doing everything:

1. CLI arg parsing + config orchestration
2. Model discovery via ModelDatabase
3. Precision/quantization setup, flash-attention probing
4. Qwen3VL vs CausalLM branching
5. Logit processors (emoji think, stop-emoji, post-stop penalties)
6. Data sanitation & augmentation
7. Status writing, live monitor, evolution tracker, layer monitor
8. Full HF Trainer loop & CLI prompting

This makes it:
- Hard to unit-test in isolation
- Hard to reason about failures
- Hard to reuse components

## Solution

Split into focused collaborators with single responsibilities.

## Target Architecture

```
core/
├── train.py                    # Thin CLI wrapper (~100 lines)
├── training/
│   ├── __init__.py
│   ├── config.py              # TrainingConfig dataclass (from trainer/)
│   ├── model_loader.py        # Model loading, precision, flash-attn
│   ├── dataset_preparer.py    # Data loading, sanitation, augmentation
│   ├── logit_processors.py    # Emoji think, stop penalty, etc.
│   ├── monitoring_bundle.py   # Status writer, live monitor, layer monitor
│   └── trainer_runner.py      # HF Trainer setup and execution
```

## Implementation Steps

### Step 1: Extract ModelLoader

```python
# core/training/model_loader.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class ModelConfig:
    model_path: Path
    precision: str = "bf16"  # bf16, fp16, fp32
    use_flash_attention: bool = True
    trust_remote_code: bool = True

class ModelLoader:
    """Handles model and tokenizer loading with precision/attention setup."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer with configured precision."""
        dtype = self._get_dtype()
        attn_impl = self._get_attention_impl()

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=self.config.trust_remote_code
        )

        return model, tokenizer

    def _get_dtype(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }.get(self.config.precision, torch.bfloat16)

    def _get_attention_impl(self) -> str:
        if self.config.use_flash_attention:
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                pass
        return "sdpa"
```

### Step 2: Extract DatasetPreparer

```python
# core/training/dataset_preparer.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datasets import Dataset

@dataclass
class DatasetConfig:
    dataset_path: Path
    max_length: int = 4096
    validation_split: float = 0.1
    apply_thinking_pattern: bool = False
    thinking_emoji: str = ""

class DatasetPreparer:
    """Handles dataset loading, validation, and preprocessing."""

    def __init__(self, config: DatasetConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def prepare(self) -> Tuple[Dataset, Dataset]:
        """Load and prepare train/val datasets."""
        raw_data = self._load_jsonl()
        processed = self._preprocess(raw_data)
        train, val = self._split(processed)
        return train, val

    def _load_jsonl(self) -> List[dict]:
        ...

    def _preprocess(self, data: List[dict]) -> Dataset:
        ...

    def _split(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        ...
```

### Step 3: Extract LogitProcessors

```python
# core/training/logit_processors.py
from transformers import LogitsProcessor
from typing import List
import torch

class EmojiThinkProcessor(LogitsProcessor):
    """Boosts thinking emoji tokens during generation."""
    ...

class StopEmojiProcessor(LogitsProcessor):
    """Handles stop signal detection."""
    ...

class PostStopPenaltyProcessor(LogitsProcessor):
    """Penalizes tokens after stop signal."""
    ...

def get_processors(config: dict, tokenizer) -> List[LogitsProcessor]:
    """Factory function to create configured processors."""
    processors = []
    if config.get("emoji_think"):
        processors.append(EmojiThinkProcessor(tokenizer, config["thinking_emoji"]))
    if config.get("stop_penalty"):
        processors.append(StopEmojiProcessor(tokenizer, config["stop_emoji"]))
    return processors
```

### Step 4: Extract MonitoringBundle

```python
# core/training/monitoring_bundle.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class MonitoringConfig:
    status_dir: Path
    enable_live_monitor: bool = True
    enable_layer_monitor: bool = False
    enable_evolution_tracker: bool = False

class MonitoringBundle:
    """Manages all monitoring components."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.status_writer = None
        self.live_monitor = None
        self.layer_monitor = None

    def start(self):
        """Initialize monitoring components."""
        ...

    def update(self, metrics: dict):
        """Push metrics to all monitors."""
        ...

    def stop(self):
        """Cleanup monitoring."""
        ...
```

### Step 5: Extract TrainerRunner

```python
# core/training/trainer_runner.py
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    output_dir: Path
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    eval_steps: int = 50
    save_steps: int = 1000
    gradient_accumulation_steps: int = 1

class TrainerRunner:
    """Wraps HuggingFace Trainer with our configuration."""

    def __init__(
        self,
        config: TrainerConfig,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        monitoring: MonitoringBundle
    ):
        self.config = config
        self.monitoring = monitoring
        self.trainer = self._create_trainer(model, tokenizer, train_dataset, eval_dataset)

    def run(self) -> dict:
        """Execute training and return results."""
        self.monitoring.start()
        try:
            result = self.trainer.train()
            return {"status": "success", "metrics": result.metrics}
        finally:
            self.monitoring.stop()

    def _create_trainer(self, model, tokenizer, train_ds, eval_ds) -> Trainer:
        ...
```

### Step 6: Slim down train.py

```python
# core/train.py
"""Training CLI - thin wrapper around training components."""

import argparse
from pathlib import Path
from training.config import load_config
from training.model_loader import ModelLoader, ModelConfig
from training.dataset_preparer import DatasetPreparer, DatasetConfig
from training.monitoring_bundle import MonitoringBundle, MonitoringConfig
from training.trainer_runner import TrainerRunner, TrainerConfig

def main():
    args = parse_args()
    config = load_config(args)

    # Load model
    model_loader = ModelLoader(ModelConfig(
        model_path=config.model_path,
        precision=config.precision
    ))
    model, tokenizer = model_loader.load()

    # Prepare data
    data_preparer = DatasetPreparer(DatasetConfig(
        dataset_path=config.dataset_path,
        max_length=config.max_length
    ), tokenizer)
    train_ds, val_ds = data_preparer.prepare()

    # Setup monitoring
    monitoring = MonitoringBundle(MonitoringConfig(
        status_dir=config.status_dir
    ))

    # Run training
    runner = TrainerRunner(
        TrainerConfig(**config.trainer_args),
        model, tokenizer, train_ds, val_ds, monitoring
    )
    result = runner.run()

    print(f"Training complete: {result}")

if __name__ == "__main__":
    main()
```

## Checkpoints

- [x] Create `core/training/` package directory
- [x] Extract `ModelLoader` class (with tests)
- [x] Extract `DatasetPreparer` class (with tests)
- [x] Extract `MonitoringBundle` class (with tests)
- [ ] Extract logit processors (existing in logit_penalty.py)
- [ ] Extract `TrainerRunner` class (future work)
- [ ] Wire components into train.py (future work)
- [ ] Slim `train.py` to ~100 lines (future work)
- [ ] Add integration tests

## Verification

```bash
# Training should work exactly as before
python core/train.py --dataset data.jsonl --model qwen3 --output outputs

# New components should be importable
python -c "from core.training.model_loader import ModelLoader; print('OK')"
```

## Migration Strategy

1. Create new modules alongside existing code
2. Gradually move functionality, one component at a time
3. Keep `UltimateTrainer` working during migration
4. Once all components extracted, delete old code
5. Each extraction should be a separate commit
