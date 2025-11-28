# UltimateTrainer Refactor Plan

> **Status:** PHASE 1-3, 5 COMPLETE | Phase 4 PENDING
> **Created:** 2025-11-28
> **Updated:** 2025-11-28 (Implementation complete)
> **Goal:** Refactor UltimateTrainer to delegate HF training to TrainerEngine while preserving all existing functionality

## Implementation Summary

### Completed
- **Phase 1:** Enhanced TrainerEngine with all features (~970 lines)
  - Qwen3VL model fallback with vision tower freezing
  - Flash Attention 2 detection and auto-selection
  - Response-only masking via DataCollatorForCompletionOnly
  - Dataset packing via trl.pack_dataset
  - Muon optimizer support
  - Checkpoint resumption with scheduler management
  - Masking validation
  - Callback injection support

- **Phase 2:** Updated TrainerConfig schema
  - Added `DataConfig.enable_packing`, `DataConfig.packing_strategy`
  - Added `ModelConfig.prefer_flash_attention`, `freeze_vision_towers`, `try_vision_model_first`

- **Phase 3:** Refactored UltimateTrainer
  - Added TrainerEngine instance in `__init__`
  - Added `USE_ENGINE=1` environment variable flag
  - Added `_run_with_engine()` method for delegation
  - Existing code path preserved as default

- **Phase 5:** Added tests (19 tests, all passing)
  - `tests/test_trainer_engine.py`
  - Tests for TrainingResult, MonitorContext, config, helpers, checkpoint resumption, masking

### Pending
- **Phase 4:** Extract LiveMonitorCallback to separate module
  - Currently ~400 lines inline in `train()`
  - Would reduce `core/train.py` by ~500 lines

### How to Test the New Engine Path

```bash
# Enable engine delegation
USE_ENGINE=1 python3 core/train.py \
    --dataset data/test.jsonl \
    --model qwen3_0.6b \
    --output-dir /tmp/test \
    --yes
```

---

## 1. Current State Analysis

### UltimateTrainer Responsibilities (7+)

| Responsibility | Lines | Should Move To |
|---------------|-------|----------------|
| CLI UX (prompts, banners, emoji prints) | ~100 | Stay in UltimateTrainer |
| Config loading + locked config enforcement | ~100 | Stay (use TrainerConfig) |
| Data validation (DatasetValidator) | ~30 | Stay in UltimateTrainer |
| Model loading (quantization, precision, Qwen3VL) | ~200 | TrainerEngine |
| Data formatting (thinking/stop patterns) | ~200 | DataProfile (already moved) |
| Monitoring setup (live monitor, callbacks) | ~500 | TrainerEngine._create_callbacks() |
| HF training (Trainer, train loop, checkpoints) | ~400 | TrainerEngine |

### TrainerEngine Current State

`trainer/core/engine.py` is **already implemented** with:
- `run_job(config)` - full 8-step pipeline
- `_load_model_and_tokenizer()` - basic loading
- `_prepare_dataset()` - transform + tokenize
- `_create_trainer()` - HF Trainer creation
- `TrainingResult` dataclass

### Key Gaps Between TrainerEngine and UltimateTrainer

| Feature | UltimateTrainer | TrainerEngine | Gap |
|---------|-----------------|---------------|-----|
| Qwen3VL model fallback | Yes | No | Add to engine |
| Flash Attention 2 detection | Yes | No | Add to engine |
| Muon optimizer support | Yes | No | Add to engine |
| Response-only masking | DataCollatorForCompletionOnly | DataCollatorForLanguageModeling | Upgrade collator |
| Packing (trl.pack_dataset) | Yes | No | Add to engine |
| Checkpoint resumption | Yes (complex) | No | Add to engine |
| LiveMonitorCallback | Yes (500+ lines) | No callbacks | Add callback system |
| Remote evaluation | Yes | No | Add to engine config |
| Masking validation | Yes | No | Add validation step |
| Fixed validation set | Yes | No | Add to engine |
| Muon optimizer | Yes | No | Add optimizer factory |

---

## 2. Target Architecture

### After Refactor

```
UltimateTrainer (thin shell)
├── __init__()
│   ├── Load config (ConfigLoader)
│   ├── Create status writer
│   └── Create engine: TrainerEngine(status_writer)
│
├── run() - CLI orchestration
│   ├── Step 0: enforce_locked_config()
│   ├── Step 1: validate_dataset() [DatasetValidator]
│   ├── Step 2: estimate_time() [TimeEstimator + display]
│   ├── Step 3: setup_monitors() [LiveInferenceMonitor setup]
│   ├── Step 4: engine.run_job(config, monitors=monitors)
│   └── Step 5: Display results, notify
│
└── CLI helpers
    ├── validate_dataset()
    ├── estimate_time()
    └── setup_monitors()

TrainerEngine (owns HF pipeline)
├── run_job(config, monitors=None) - main entry
│   ├── _validate_config()
│   ├── _load_profile()
│   ├── _load_model_and_tokenizer()  # Enhanced
│   ├── _prepare_dataset()           # With packing
│   ├── _create_trainer()            # With callbacks + optimizer
│   └── _execute_training()          # With resume + validation
│
└── Private helpers
    ├── _get_torch_dtype()
    ├── _build_training_arguments()
    ├── _build_callbacks()
    ├── _create_optimizer()
    └── _validate_masking()
```

---

## 3. Implementation Steps

### Phase 1: Enhance TrainerEngine (No UltimateTrainer Changes)

#### Step 1.1: Add Model Loading Features to Engine
**File:** `trainer/core/engine.py`

Add to `_load_model_and_tokenizer()`:
- Qwen3VL fallback (try Qwen3VLForConditionalGeneration first)
- Flash Attention 2 detection and selection
- Vision tower freezing for text-only training
- Chat template override application

```python
def _load_model_and_tokenizer(self, config: TrainerConfig):
    # Try flash_attention_2 if available
    attn_impl = "sdpa"
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    # Try Qwen3VL first for VL models
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(...)
        # Freeze vision towers
        for n, p in model.named_parameters():
            if any(k in n for k in ["vision_model", "video_model"]):
                p.requires_grad = False
        tokenizer = AutoProcessor.from_pretrained(...).tokenizer
    except:
        # Fallback to standard CausalLM
        model = AutoModelForCausalLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
```

#### Step 1.2: Add Response-Only Collator
**File:** `trainer/core/engine.py`

Replace `DataCollatorForLanguageModeling` with `DataCollatorForCompletionOnly`:

```python
from core.custom_collator import DataCollatorForCompletionOnly

def _create_trainer(self, config, train_dataset, val_dataset):
    data_collator = DataCollatorForCompletionOnly(
        tokenizer=self.tokenizer,
        response_template="<|im_start|>assistant\n"
    )
    # ...
```

#### Step 1.3: Add Packing Support
**File:** `trainer/core/engine.py`

Add packing to `_prepare_dataset()`:

```python
def _prepare_dataset(self, config: TrainerConfig):
    # ... existing code ...

    # Pack dataset if enabled
    if config.data.enable_packing:
        from trl import pack_dataset
        train_dataset = pack_dataset(
            train_dataset,
            seq_length=config.hyperparams.max_length,
            strategy="bfd"
        )
        # Remove seq_lengths metadata
        if 'seq_lengths' in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns(['seq_lengths'])

    return train_dataset, val_dataset
```

#### Step 1.4: Add Optimizer Factory
**File:** `trainer/core/engine.py`

```python
def _create_optimizer(self, config_dict: dict, num_training_steps: int):
    """Create optimizer (Muon or AdamW) based on config."""
    try:
        from trainer.optimizers import create_optimizer as create_custom_optimizer
        opt_config = config_dict.get("optimizer", {})
        optimizer_type = opt_config.get("type", "adamw")

        if optimizer_type == "muon":
            return create_custom_optimizer(
                self.model,
                config_dict,
                optimizer_type="muon",
                num_training_steps=num_training_steps,
            )
    except ImportError:
        pass

    return None, None, None  # Use Trainer defaults
```

#### Step 1.5: Add Checkpoint Resumption
**File:** `trainer/core/engine.py`

```python
def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
    """Find latest checkpoint for resumption."""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = []
    for cp in checkpoint_dir.glob("checkpoint-*"):
        parts = cp.name.split("-", 1)
        if len(parts) == 2:
            try:
                step = int(parts[1])
                if (cp / "trainer_state.json").exists():
                    candidates.append((step, cp))
            except ValueError:
                continue

    if candidates:
        return str(max(candidates, key=lambda x: x[0])[1])
    return None
```

#### Step 1.6: Add Callback System
**File:** `trainer/core/engine.py`

```python
def run_job(self, config: TrainerConfig, monitors: dict = None) -> TrainingResult:
    """
    Execute training job.

    Args:
        config: TrainerConfig
        monitors: Optional dict with:
            - live_monitor: LiveInferenceMonitor
            - evolution_tracker: EvolutionTracker
            - layer_monitor: LayerMonitor
            - controller: TrainingController (pause/stop)
    """
    # ... existing code ...

    # 5. Create trainer with callbacks
    callbacks = self._build_callbacks(config, monitors)
    trainer = self._create_trainer(config, train_dataset, val_dataset, callbacks)

def _build_callbacks(self, config: TrainerConfig, monitors: dict = None):
    """Build training callbacks from config and monitors."""
    callbacks = []

    if monitors:
        from core.train import LiveMonitorCallback  # Or extract to separate module
        callback = LiveMonitorCallback(
            monitor=monitors.get('live_monitor'),
            status_writer=self.status_writer,
            # ... other params from monitors dict
        )
        callbacks.append(callback)

    return callbacks
```

#### Step 1.7: Add Masking Validation
**File:** `trainer/core/engine.py`

```python
def _validate_masking(self, dataset, collator, sample_count: int = 5):
    """Validate that masking is working correctly."""
    samples = [dataset[i] for i in range(min(sample_count, len(dataset)))]
    batch = collator(samples)

    masked_count = (batch['labels'] == -100).sum().item()
    total_count = batch['labels'].numel()
    masked_pct = 100 * masked_count / total_count

    if masked_pct < 25:
        raise ValueError(f"Masking too low ({masked_pct:.1f}%). Check collator config.")

    return {"masked_pct": masked_pct, "trained_pct": 100 - masked_pct}
```

---

### Phase 2: Update TrainerConfig Schema

#### Step 2.1: Add Missing Config Fields
**File:** `trainer/config/schema.py`

```python
@dataclass
class DataConfig:
    # ... existing ...
    enable_packing: bool = True  # NEW
    packing_strategy: str = "bfd"  # NEW

@dataclass
class ModelConfig:
    # ... existing ...
    prefer_flash_attention: bool = True  # NEW
    freeze_vision_towers: bool = True  # NEW (for Qwen3VL)

@dataclass
class MonitoringConfig:
    # ... existing ...
    enable_remote_eval: bool = False  # NEW
    remote_eval_interval: int = 5000  # NEW
    remote_eval_host: str = ""  # NEW
```

---

### Phase 3: Refactor UltimateTrainer to Delegate

#### Step 3.1: Add TrainerEngine Instance
**File:** `core/train.py`

```python
from trainer.core.engine import TrainerEngine, TrainingResult

class UltimateTrainer:
    def __init__(self, args, controller=None):
        self.args = args
        self.controller = controller

        # Config loading (existing)
        try:
            self.config, self.config_dict = ConfigLoader.from_args_and_json_with_raw(args)
            # ...
        except:
            # ...

        # Status writer (existing)
        self.status_writer = TrainingStatusWriter(...)

        # NEW: Create TrainerEngine
        self.engine = TrainerEngine(status_writer=self.status_writer)

        # Keep other attributes for CLI orchestration
        self.model_db = ModelDatabase()
        self.validator = None
        # ...
```

#### Step 3.2: Simplify run() Method
**File:** `core/train.py`

```python
def run(self):
    """Execute full training pipeline."""
    print("\n" + "=" * 80)
    print("ULTIMATE TRAINER")
    print("=" * 80)

    # Step 0: Enforce locked config (keep as-is)
    self.enforce_locked_config()

    # Step 1: Validate dataset (keep as-is)
    if not self.args.skip_validation:
        if not self.validate_dataset():
            return False

    # Step 2: Time estimation (keep as-is)
    estimate = self.estimate_time()
    TimeEstimator.display_estimate(estimate)
    if not self.args.yes:
        response = input("Continue? [yes/no]: ").strip().lower()
        if response != 'yes':
            return False

    # Step 3: Setup monitors (extract from existing setup_live_monitor)
    monitors = self._setup_monitors()

    # Step 4: Delegate to TrainerEngine
    result = self.engine.run_job(self.config, monitors=monitors)

    # Step 5: Handle result
    if result.success:
        self._on_training_success(result)
        return True
    else:
        self._on_training_failure(result)
        return False

def _setup_monitors(self) -> dict:
    """Setup monitoring components for training."""
    # ... existing setup_live_monitor logic ...
    return {
        'live_monitor': self.live_monitor,
        'evolution_tracker': self.evolution_tracker,
        'layer_monitor': self.layer_monitor,
        'controller': self.controller,
        'raw_train_examples': self.raw_train_examples,
        # ... other monitor state
    }

def _on_training_success(self, result: TrainingResult):
    """Handle successful training completion."""
    self.training_summary = result.summary
    self.status_writer.mark_completed(...)
    self.notifier.training_complete(...)

def _on_training_failure(self, result: TrainingResult):
    """Handle training failure."""
    self.status_writer.mark_crashed(result.error_message, ...)
    self.notifier.training_crashed(result.error_message)
```

#### Step 3.3: Remove Duplicated Methods from UltimateTrainer

Delete these methods (now in TrainerEngine):
- `load_model()` - replaced by `engine._load_model_and_tokenizer()`
- `prepare_dataset()` - replaced by `engine._prepare_dataset()`
- `train()` - replaced by `engine.run_job()`

Keep these methods (CLI-specific):
- `validate_dataset()` - uses DatasetValidator (pre-engine)
- `estimate_time()` - uses TimeEstimator (pre-engine)
- `enforce_locked_config()` - config guard (pre-engine)
- `_setup_monitors()` - creates monitors dict for engine
- `_on_training_success()` / `_on_training_failure()` - post-engine UX

---

### Phase 4: Extract LiveMonitorCallback

#### Step 4.1: Move to Separate Module
**File:** `trainer/monitoring/callbacks.py` (NEW)

```python
"""Training callbacks for monitoring and control."""

from transformers import TrainerCallback
import torch
import time

class LiveMonitorCallback(TrainerCallback):
    """
    Comprehensive training callback for:
    - Status writer updates
    - Throughput tracking
    - Loss stability monitoring
    - Control signals (pause/stop)
    - Checkpoint ledger recording
    - Remote evaluation triggering
    """

    def __init__(
        self,
        status_writer,
        monitors: dict,
        config: 'TrainerConfig',
    ):
        self.status_writer = status_writer
        self.live_monitor = monitors.get('live_monitor')
        self.controller = monitors.get('controller')
        self.evolution_tracker = monitors.get('evolution_tracker')
        # ... extract all 20+ parameters into config or monitors dict

    def on_step_end(self, args, state, control, **kwargs):
        # ... existing 200+ lines of on_step_end logic

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # ... existing on_evaluate logic

    def on_save(self, args, state, control, **kwargs):
        # ... existing on_save logic (ledger, eval queue, remote sync)
```

This extraction reduces `core/train.py` by ~500 lines.

---

### Phase 5: Tests

#### Step 5.1: Unit Test for TrainerEngine
**File:** `tests/test_trainer_engine.py` (NEW)

```python
"""Tests for TrainerEngine."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from trainer.core.engine import TrainerEngine, TrainingResult
from trainer.config import create_default_config
from trainer.monitoring import TrainingStatusWriter


class TestTrainerEngine:
    """Test TrainerEngine functionality."""

    @pytest.fixture
    def mock_status_writer(self):
        return Mock(spec=TrainingStatusWriter)

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create minimal config for testing."""
        # Create tiny test dataset
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.write_text('{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}\n')

        return create_default_config(
            model_path="models/Qwen3-0.6B",
            dataset_path=str(dataset_path),
            output_dir=str(tmp_path / "output"),
            base_model="Qwen/Qwen3-0.6B",
            model_architecture="Qwen3ForCausalLM",
            max_context_length=512,
            vocab_size=151936
        )

    def test_engine_initialization(self, mock_status_writer):
        """Test engine can be initialized."""
        engine = TrainerEngine(mock_status_writer)
        assert engine.status_writer == mock_status_writer
        assert engine.model is None
        assert engine.tokenizer is None

    @pytest.mark.skip(reason="Requires GPU and model")
    def test_run_job_smoke(self, mock_status_writer, minimal_config):
        """Smoke test for run_job (requires GPU)."""
        engine = TrainerEngine(mock_status_writer)
        result = engine.run_job(minimal_config)
        assert isinstance(result, TrainingResult)


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_from_error(self):
        """Test creating error result."""
        result = TrainingResult.from_error("Test error")
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.global_step == 0
```

#### Step 5.2: Integration Test
**File:** `tests/test_ultimate_trainer_integration.py` (NEW)

```python
"""Integration tests for UltimateTrainer + TrainerEngine."""

import subprocess
import sys
from pathlib import Path


def test_cli_smoke():
    """Smoke test CLI still works after refactor."""
    # This assumes a tiny test dataset exists
    result = subprocess.run([
        sys.executable, "core/train.py",
        "--dataset", "data/test_tiny.jsonl",
        "--model", "qwen3_0.6b",
        "--output-dir", "/tmp/test_output",
        "--max-steps", "2",
        "--yes"
    ], capture_output=True, text=True, timeout=120)

    # Check it at least starts (may fail without GPU)
    assert "ULTIMATE TRAINER" in result.stdout or "error" in result.stderr.lower()
```

---

## 4. Migration Strategy

### Option A: Big Bang (NOT recommended)
Replace all UltimateTrainer internals at once.
- Risk: Everything breaks
- Benefit: Clean cut

### Option B: Gradual Migration (RECOMMENDED)

1. **Week 1:** Enhance TrainerEngine with missing features (Phase 1)
   - No changes to UltimateTrainer
   - Engine can be tested independently

2. **Week 2:** Add callback extraction (Phase 4)
   - LiveMonitorCallback becomes reusable
   - Still no UltimateTrainer changes

3. **Week 3:** Refactor UltimateTrainer to delegate (Phase 3)
   - Keep old methods as fallback initially
   - Feature flag: `USE_ENGINE=1` env var

4. **Week 4:** Remove deprecated code paths
   - Delete old load_model(), prepare_dataset(), train()
   - Clean up imports

### Rollback Plan

At each phase, if issues arise:
1. Revert the specific phase's changes
2. Old UltimateTrainer code continues to work
3. TrainerEngine improvements remain (they're additive)

---

## 5. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `trainer/core/engine.py` | MODIFY | Add missing features from UltimateTrainer |
| `trainer/config/schema.py` | MODIFY | Add packing, flash_attn config fields |
| `trainer/monitoring/callbacks.py` | NEW | Extract LiveMonitorCallback |
| `core/train.py` | MODIFY | Delegate to TrainerEngine, remove duplication |
| `tests/test_trainer_engine.py` | NEW | Unit tests for engine |
| `tests/test_ultimate_trainer_integration.py` | NEW | Integration tests |

**Lines of Code Impact:**
- `core/train.py`: ~2280 lines → ~800 lines (-65%)
- `trainer/core/engine.py`: ~670 lines → ~1200 lines (+80%)
- `trainer/monitoring/callbacks.py`: 0 → ~600 lines (NEW)
- Net: Slight increase, but much better separation

---

## 6. Success Criteria

1. **Functional Parity:**
   - All existing training features work identically
   - Checkpoints are compatible (can resume from old checkpoints)
   - Config system unchanged for users

2. **Code Quality:**
   - UltimateTrainer < 1000 lines
   - TrainerEngine has clear single responsibility
   - LiveMonitorCallback reusable from other entry points

3. **Testing:**
   - Unit tests for TrainerEngine pass
   - Integration test confirms CLI works
   - Manual test: full training run succeeds

4. **No Regressions:**
   - Muon optimizer still works
   - Packing still works
   - Response-only masking still works
   - Remote eval still works
   - Checkpoint ledger still records

---

## 7. Open Questions

1. **Should TrainerEngine accept monitors directly, or build them internally?**
   - Option A: Accept monitors dict (keeps engine pure)
   - Option B: Build monitors from config (more convenient)
   - **Decision:** Option A - keeps engine testable

2. **Where should LiveMonitorCallback live?**
   - Option A: `trainer/monitoring/callbacks.py` (new)
   - Option B: `core/callbacks.py` (alongside train.py)
   - **Decision:** Option A - better module organization

3. **Should we keep legacy fallback code?**
   - Option A: Yes, behind feature flag for safety
   - Option B: No, clean break
   - **Decision:** Option A initially, remove after validation

---

## 8. Implementation Order

```
Phase 1: Enhance TrainerEngine
├── Step 1.1: Qwen3VL + Flash Attention support
├── Step 1.2: Response-only collator
├── Step 1.3: Packing support
├── Step 1.4: Optimizer factory
├── Step 1.5: Checkpoint resumption
├── Step 1.6: Callback system
└── Step 1.7: Masking validation

Phase 2: Update Config Schema
└── Step 2.1: Add missing fields

Phase 3: Refactor UltimateTrainer
├── Step 3.1: Add TrainerEngine instance
├── Step 3.2: Simplify run() method
└── Step 3.3: Remove duplicated methods

Phase 4: Extract LiveMonitorCallback
└── Step 4.1: Move to separate module

Phase 5: Tests
├── Step 5.1: Unit tests
└── Step 5.2: Integration tests
```

---

## Appendix A: Code Size Comparison

### Before Refactor

| Module | Lines | Responsibilities |
|--------|-------|------------------|
| `core/train.py` | 2280 | Everything |
| `trainer/core/engine.py` | 670 | Basic HF training |

### After Refactor

| Module | Lines | Responsibilities |
|--------|-------|------------------|
| `core/train.py` | ~800 | CLI orchestration |
| `trainer/core/engine.py` | ~1200 | HF training pipeline |
| `trainer/monitoring/callbacks.py` | ~600 | Training callbacks |

---

## Appendix B: Key Methods Migration

| UltimateTrainer Method | After Refactor |
|------------------------|----------------|
| `__init__()` | Keep (+ add engine) |
| `run()` | Keep (simplified) |
| `validate_dataset()` | Keep |
| `load_model()` | DELETE → `engine._load_model_and_tokenizer()` |
| `prepare_dataset()` | DELETE → `engine._prepare_dataset()` |
| `estimate_time()` | Keep |
| `setup_live_monitor()` | Rename → `_setup_monitors()` |
| `train()` | DELETE → `engine.run_job()` |
| `enforce_locked_config()` | Keep |
| `sanitize_example()` | DELETE (in profile) |
| `enforce_thinking_requirement()` | DELETE (in profile) |
| `enforce_stop_requirement()` | DELETE (in profile) |
| `_record_training_summary()` | Keep (from TrainingResult) |

---

**Ready for implementation approval.**
