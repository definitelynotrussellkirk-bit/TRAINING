# TrainerEngine Implementation Design

> **ARCHIVE NOTE (2025-11-24):**
> This document describes the pre-implementation state where TrainerEngine
> was still a stub. The engine is now fully implemented in
> `trainer/core/engine.py`. Treat this doc as historical design, not
> current behavior.

**Goal:** Implement full `TrainerEngine.run_job(config)` that orchestrates all training operations.

---

## Current State

**trainer/core/engine.py** - Proof-of-concept stub:
```python
def run_job(self, config: TrainerConfig) -> TrainingResult:
    # Stub - returns error message
    pass
```

**core/train.py** - Production script:
- `UltimateTrainer` class with methods:
  - `load_model()`
  - `prepare_dataset()`
  - `setup_training()`
  - `run()` - orchestrates everything
- ~1900 lines, complex, hard to modify

---

## Design Approach

**Strategy:** Implement TrainerEngine by extracting core logic from UltimateTrainer

### Architecture

```
TrainerEngine
├── run_job(config) -> TrainingResult
│
├── Helper Methods (private):
│   ├── _load_model_and_tokenizer(config)
│   ├── _prepare_dataset(config, profile, tokenizer)
│   ├── _create_trainer(config, model, datasets, callbacks)
│   └── _execute_training(trainer, config)
│
└── Integration:
    ├── Uses ConfigLoader for config
    ├── Uses get_profile() for data transformation
    ├── Uses create_preview_backend() for monitoring
    └── Returns TrainingResult with metrics
```

### run_job() Flow

```python
def run_job(self, config: TrainerConfig) -> TrainingResult:
    """
    Execute training job with given config.

    Flow:
    1. Validate config
    2. Load profile from config.profile.name
    3. Load model & tokenizer
    4. Prepare datasets using profile
    5. Create trainer with callbacks
    6. Execute training
    7. Save final checkpoint
    8. Return TrainingResult
    """

    # 1. Validate
    ConfigLoader.validate_locked_config(config)

    # 2. Load profile
    profile = get_profile(config.profile.name)

    # 3. Load model & tokenizer
    model, tokenizer = self._load_model_and_tokenizer(config)

    # 4. Prepare datasets
    train_dataset, val_dataset = self._prepare_dataset(config, profile, tokenizer)

    # 5. Create trainer
    trainer = self._create_trainer(config, model, train_dataset, val_dataset)

    # 6. Train
    result = trainer.train()

    # 7. Save
    trainer.save_model(config.output.output_dir)

    # 8. Return result
    return TrainingResult(
        success=True,
        global_step=result.global_step,
        runtime_sec=result.metrics['train_runtime'],
        last_checkpoint_path=config.output.output_dir,
        final_loss=result.metrics.get('train_loss', 0.0),
        summary=result.metrics
    )
```

---

## Implementation Plan

### Step 1: Extract _load_model_and_tokenizer

From `UltimateTrainer.load_model()` (lines 487-663):

```python
def _load_model_and_tokenizer(self, config: TrainerConfig):
    """
    Load model and tokenizer from config.

    Args:
        config: TrainerConfig with model settings

    Returns:
        (model, tokenizer) tuple
    """
    # Get model path
    model_path = config.model.model_path

    # Setup quantization (if using LoRA)
    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(...)

    # Set precision
    torch_dtype = self._get_torch_dtype(config.hyperparams.fp_precision)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        ...
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, ...)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable cache for training
    model.config.use_cache = False

    return model, tokenizer
```

### Step 2: Extract _prepare_dataset

From `UltimateTrainer.prepare_dataset()` (lines 665-770):

```python
def _prepare_dataset(self, config: TrainerConfig, profile: DataProfile, tokenizer):
    """
    Load and prepare datasets.

    Args:
        config: TrainerConfig with data settings
        profile: Data profile for transformations
        tokenizer: Tokenizer

    Returns:
        (train_dataset, val_dataset) tuple
    """
    # Load raw data
    with open(config.data.dataset_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # Shuffle
    random.shuffle(examples)

    # Split train/val
    val_size = min(
        config.monitoring.validation_max_samples,
        int(len(examples) * config.monitoring.validation_split)
    )
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    # Build system prompt
    system_prompt = self._build_system_prompt(config)

    # Transform using profile
    train_examples = [
        profile.transform_example(ex, idx, system_prompt)
        for idx, ex in enumerate(train_examples)
    ]
    val_examples = [
        profile.transform_example(ex, idx, system_prompt)
        for idx, ex in enumerate(val_examples)
    ]

    # Tokenize
    def tokenize_function(examples):
        # Apply chat template & tokenize
        ...

    train_dataset = Dataset.from_list(train_examples).map(tokenize_function)
    val_dataset = Dataset.from_list(val_examples).map(tokenize_function)

    return train_dataset, val_dataset
```

### Step 3: Extract _create_trainer

From `UltimateTrainer.setup_training()` (lines 1035-1100):

```python
def _create_trainer(self, config, model, train_dataset, val_dataset):
    """
    Create HuggingFace Trainer with callbacks.

    Args:
        config: TrainerConfig
        model: Loaded model
        train_dataset: Tokenized training dataset
        val_dataset: Tokenized validation dataset

    Returns:
        Trainer instance
    """
    # Training arguments
    training_args = self._build_training_arguments(config)

    # Data collator
    collator = DataCollatorForCompletionOnly(
        tokenizer=self.tokenizer,
        mlm=False
    )

    # Callbacks
    callbacks = self._build_callbacks(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks
    )

    return trainer
```

### Step 4: Helper Methods

```python
def _get_torch_dtype(self, precision: str):
    """Convert precision string to torch dtype"""
    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    elif precision == "fp32":
        return torch.float32
    else:
        return torch.bfloat16  # Default

def _build_system_prompt(self, config: TrainerConfig) -> str:
    """Build system prompt from config"""
    template = config.monitoring.system_prompt_base
    date_str = datetime.now().strftime('%Y-%m-%d')
    return template.format(date=date_str)

def _build_training_arguments(self, config: TrainerConfig):
    """Build HuggingFace TrainingArguments from config"""
    use_fp16 = config.hyperparams.fp_precision == "fp16"
    use_bf16 = config.hyperparams.fp_precision == "bf16"

    return TrainingArguments(
        output_dir=config.output.output_dir,
        per_device_train_batch_size=config.hyperparams.batch_size,
        gradient_accumulation_steps=config.hyperparams.gradient_accumulation,
        learning_rate=config.hyperparams.learning_rate,
        warmup_steps=config.hyperparams.warmup_steps,
        num_train_epochs=config.hyperparams.num_epochs,
        max_steps=config.hyperparams.max_steps,
        save_steps=config.hyperparams.save_steps,
        eval_steps=config.hyperparams.eval_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config.environment.logging_steps,
        report_to=config.environment.report_to,
        ...
    )

def _build_callbacks(self, config: TrainerConfig):
    """Build training callbacks from config"""
    callbacks = []

    # Add monitoring callback if enabled
    if config.monitoring.enable_pattern_tracking:
        # Create LiveMonitorCallback
        # (For now, can be simplified or stubbed)
        pass

    return callbacks
```

---

## Integration with core/train.py

### Option A: Make UltimateTrainer delegate to TrainerEngine

```python
class UltimateTrainer:
    def __init__(self, args, controller=None):
        self.args = args
        self.config = ConfigLoader.from_args_and_json(args)
        self.engine = TrainerEngine(status_writer)

    def run(self):
        # Delegate to engine
        result = self.engine.run_job(self.config)
        return result.success
```

**Pros:** Simple, clean separation
**Cons:** Loses some existing features (live monitor callback, etc.)

### Option B: Keep UltimateTrainer, use engine methods internally

```python
class UltimateTrainer:
    def __init__(self, args, controller=None):
        self.args = args
        self.config = ConfigLoader.from_args_and_json(args)
        self.engine = TrainerEngine(status_writer)

    def load_model(self):
        # Delegate to engine
        self.model, self.tokenizer = self.engine._load_model_and_tokenizer(self.config)
        # ... keep existing monitoring setup

    def prepare_dataset(self):
        # Delegate to engine
        profile = get_profile(self.config.profile.name)
        self.train_dataset, self.val_dataset = self.engine._prepare_dataset(
            self.config, profile, self.tokenizer
        )
        # ... keep existing features
```

**Pros:** Gradual migration, keeps all features
**Cons:** More complex, some duplication

**Recommendation:** Start with Option B, then gradually move to Option A.

---

## Implementation Order

1. ✅ Create preview_backend.py (done)
2. ✅ Update MonitoringConfig (done)
3. **Create full TrainerEngine.run_job()** ← NEXT
4. Extract _load_model_and_tokenizer()
5. Extract _prepare_dataset()
6. Extract _create_trainer()
7. Add helper methods
8. Test with simple dataset
9. Integrate with core/train.py (Option B)
10. Update docs

---

## Testing Strategy

```bash
# Test 1: Direct engine usage
python3 -c "
from trainer.core import TrainerEngine
from trainer.config import create_default_config
from trainer.monitoring import TrainingStatusWriter

config = create_default_config(
    model_path='models/Qwen3-0.6B',
    dataset_path='data/small_test.jsonl',
    output_dir='outputs/engine_test',
    base_model='Qwen/Qwen3-0.6B',
    model_architecture='Qwen3ForCausalLM',
    max_context_length=4096,
    vocab_size=151936
)

status_writer = TrainingStatusWriter('status/training_status.json')
engine = TrainerEngine(status_writer)
result = engine.run_job(config)

print(f'Success: {result.success}')
print(f'Final loss: {result.final_loss}')
"

# Test 2: Via core/train.py (backward compat)
python3 core/train.py --dataset data/small_test.jsonl --model qwen3 --output outputs/test
```

---

## Next Steps

Implement in this order:
1. Create full engine.run_job() skeleton
2. Implement _load_model_and_tokenizer()
3. Implement _prepare_dataset()
4. Implement _create_trainer()
5. Wire everything together
6. Test end-to-end

Ready to proceed!
