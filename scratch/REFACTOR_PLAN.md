# TRAINING SYSTEM REFACTOR - COMPLETE PLAN

**Date:** 2025-11-22
**Goal:** Transform UltimateTrainer into a stable, API-driven service with clean separation of concerns

---

## 🎯 Target Architecture: 3-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│ 🔒 LAYER 1: CORE ENGINE (Stable - Don't Touch Casually)    │
│                                                             │
│  • Orchestration: validate → load → train → save           │
│  • HuggingFace Trainer setup                               │
│  • Checkpoint resume logic                                 │
│  • Data I/O contracts (JSONL → Dataset → tokenized)        │
│  • Safety (NaN/Inf detection, crash handling)              │
│  • Status writer contract                                  │
│                                                             │
│  Changes rarely, in controlled way                         │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 🎚️  LAYER 2: CONFIG & TOGGLES (Tuned Often)                │
│                                                             │
│  • Hyperparameters (batch_size, lr, warmup, etc.)          │
│  • Model config (fp16/bf16, QLoRA toggle)                  │
│  • Data behavior (which regime, system prompts)            │
│  • Monitoring behavior (eval frequency, metrics)           │
│                                                             │
│  Lives in config.json, NOT scattered in code               │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 🧩 LAYER 3: PROFILES / PLUGINS (Edit Frequently)           │
│                                                             │
│  • Data profiles (emoji-thinking, regime-3, plain SFT)     │
│  • Logit penalty profiles                                  │
│  • System prompt templates                                 │
│  • Monitoring plugins (pattern tracker, layer monitor)     │
│                                                             │
│  Playground for experimentation, stable interfaces         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 New Directory Structure

```
trainer/
├── __init__.py
├── cli_main.py                 # Thin CLI wrapper
│
├── core/                       # 🔒 LAYER 1: Core Engine
│   ├── __init__.py
│   ├── engine.py               # TrainerEngine.run_job()
│   ├── model_loader.py         # Model/tokenizer loading
│   ├── dataset_loader.py       # JSONL → HF Dataset
│   └── hf_trainer_factory.py   # TrainingArguments + Trainer setup
│
├── config/                     # 🎚️ LAYER 2: Config System
│   ├── __init__.py
│   ├── schema.py               # TrainerConfig dataclasses
│   ├── loader.py               # Load/merge config.json + CLI
│   └── validator.py            # Config validation
│
├── profiles/                   # 🧩 LAYER 3: Data Profiles
│   ├── __init__.py
│   ├── base.py                 # Base profile interface
│   ├── emoji_think.py          # Current emoji system
│   ├── regime3.py              # Future regime-3 system
│   └── plain_sft.py            # Optional plain fine-tuning
│
├── monitoring/                 # 🧩 LAYER 3: Monitoring
│   ├── __init__.py
│   ├── status_writer.py        # TrainingStatusWriter
│   ├── callbacks.py            # LiveMonitorCallback, etc.
│   └── plugins/
│       ├── pattern_tracker.py
│       ├── layer_monitor.py
│       └── evolution_tracker.py
│
└── utils/                      # Shared utilities
    ├── safety.py               # NaN/Inf detection
    ├── logging.py              # Logging setup
    └── helpers.py              # Misc helpers
```

---

## 📐 Core Interfaces

### 1. TrainerConfig (Layer 2)

```python
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class Hyperparams:
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    warmup_steps: int
    max_length: int
    fp_precision: Literal["fp16", "bf16", "fp32"]
    save_steps: int
    eval_steps: int
    num_epochs: int

@dataclass
class ProfileConfig:
    name: str  # "emoji_think", "regime3", "plain_sft"
    # Profile-specific options (extensible)
    options: dict = None

@dataclass
class MonitoringConfig:
    update_interval: int = 2                    # Status update frequency
    inference_interval: int = 50                # Live inference frequency
    micro_eval_interval: int = 200              # Micro eval frequency
    num_eval_samples: int = 4                   # Samples per eval
    system_prompt: str = ""                     # Base system prompt
    enable_pattern_tracking: bool = True
    enable_layer_monitor: bool = False
    enable_evolution_tracker: bool = True

@dataclass
class LockedConfig:
    """Fields that cannot be overridden via CLI"""
    base_model: str
    max_context_length: int
    model_architecture: str

@dataclass
class TrainerConfig:
    # Paths
    model_path: str
    output_dir: str
    dataset_path: str

    # Core settings
    hyperparams: Hyperparams
    profile: ProfileConfig
    monitoring: MonitoringConfig
    locked: LockedConfig

    # Optional
    resume_from_checkpoint: Optional[str] = None
    validation_split: float = 0.05
```

### 2. Profile Interface (Layer 3)

```python
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import LogitsProcessorList

class DataProfile(ABC):
    """Base interface for data transformation profiles"""

    name: str  # "emoji_think", "regime3", etc.

    @abstractmethod
    def transform_example(
        self,
        example: dict,
        index: int,
        system_prompt: str
    ) -> dict:
        """
        Transform raw example into training format.

        Args:
            example: Raw JSONL example {"messages": [...]}
            index: Example index (for random seeds)
            system_prompt: Base system prompt

        Returns:
            Transformed example ready for tokenization
        """
        pass

    @abstractmethod
    def build_logits_processors(
        self,
        tokenizer
    ) -> LogitsProcessorList:
        """
        Build logits processors for this profile.

        Args:
            tokenizer: Model tokenizer

        Returns:
            LogitsProcessorList for training
        """
        pass

    @abstractmethod
    def get_system_prompt_template(self) -> str:
        """Get system prompt template for this profile"""
        pass
```

### 3. TrainerEngine API (Layer 1)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingResult:
    """Result of a training job"""
    success: bool
    global_step: int
    runtime_sec: float
    last_checkpoint_path: Optional[str]
    final_loss: float
    summary: dict

class TrainerEngine:
    """Core training engine - stable API surface"""

    def __init__(self, status_writer: TrainingStatusWriter):
        self.status_writer = status_writer

    def run_job(self, config: TrainerConfig) -> TrainingResult:
        """
        Execute a training job.

        This is the ONLY public method. All training goes through here.

        Args:
            config: Complete training configuration

        Returns:
            TrainingResult with success status and metrics
        """
        # 1. Validate config
        # 2. Load profile
        # 3. Load model & tokenizer
        # 4. Load & transform dataset
        # 5. Setup HF Trainer
        # 6. Execute training
        # 7. Save final checkpoint
        # 8. Return result
        pass
```

---

## 🔄 Refactor Steps (Safe Order)

### ⭐ STEP 0: Freeze Current Behavior

```bash
# Create safety tag
git add -A
git commit -m "Checkpoint before refactor - working emoji system"
git tag trainer_v1_emoji_baseline

# Backup current file
cp core/train.py core/train_v1_backup.py
```

**Deliverable:** Safe rollback point

---

### ⭐ STEP 1: Extract Config into Single Object

**Goal:** Replace scattered `args.X` and `config.json` reads with unified config object

**Tasks:**
1. Create `trainer/config/schema.py` with dataclasses
2. Create `trainer/config/loader.py` for config loading
3. Replace `args` access in `train.py` with `config.X`
4. Test: Behavior identical to before

**File Changes:**
- NEW: `trainer/config/schema.py`
- NEW: `trainer/config/loader.py`
- MODIFY: `core/train.py` (use config object)

**Validation:**
```bash
# Run with same inputs, verify identical output
python3 core/train.py --dataset test.jsonl --model qwen3_0.6b
```

---

### ⭐ STEP 2: Extract Emoji/Stop Logic into Profile

**Goal:** Move all emoji contract logic to `profiles/emoji_think.py`

**Current Code Locations:**
```
train.py line 75-91:   THINKING_EMOJIS, STOP_EMOJI_POOL
train.py line 95-115:  get_random_stop_emoji/count
train.py line 117-145: inject_system_prompt
train.py line 147-193: enforce_thinking_requirement
train.py line 195-238: enforce_stop_requirement
train.py line 240-270: sanitize_example
train.py line 850-870: build_think_penalty_processor
train.py line 872-895: build_post_stop_penalty_processor
```

**New Structure:**
```python
# profiles/emoji_think.py

class EmojiThinkProfile(DataProfile):
    name = "emoji_think"

    THINKING_EMOJIS = ["🤔", "💭", "🧠", ...]
    STOP_EMOJI_POOL = ["🛑", "⛔", "🚫", ...]

    def transform_example(self, example, index, system_prompt):
        # Combine: inject_system + enforce_thinking +
        #          enforce_stop + sanitize
        ...

    def build_logits_processors(self, tokenizer):
        # build_think_penalty + build_post_stop_penalty
        ...

    def get_system_prompt_template(self):
        return "Current date: {date}. Respond naturally..."
```

**In engine:**
```python
# Load profile
profile = ProfileRegistry.get(config.profile.name)

# Transform dataset
train_examples = [
    profile.transform_example(ex, idx, config.monitoring.system_prompt)
    for idx, ex in enumerate(train_examples)
]

# Setup penalties
logits_processor = profile.build_logits_processors(tokenizer)
```

**Validation:**
- Training output identical
- Same emoji patterns in generated examples
- Same penalty behavior

---

### ⭐ STEP 3: Extract LiveMonitorCallback to Monitoring Module

**Goal:** Move callback logic out of `train()` into `monitoring/callbacks.py`

**Current Code:**
```
train.py line 1050-1450: LiveMonitorCallback (entire class)
```

**New Structure:**
```python
# monitoring/callbacks.py

class LiveMonitorCallback(TrainerCallback):
    def __init__(
        self,
        status_writer,
        config: MonitoringConfig,
        tokenizer,
        model,
        validation_examples,
        ...
    ):
        self.config = config
        # ... existing init logic

    # All existing methods (on_step_end, etc.)
```

**In engine:**
```python
# monitoring/__init__.py
def build_callbacks(
    config: MonitoringConfig,
    status_writer,
    model,
    tokenizer,
    ...
) -> List[TrainerCallback]:
    callbacks = []

    callbacks.append(LiveMonitorCallback(
        status_writer=status_writer,
        config=config,
        ...
    ))

    # Future: Add other callbacks

    return callbacks
```

**Validation:**
- Live monitor updates at same frequency
- Metrics identical
- Web UI works unchanged

---

### ⭐ STEP 4: Create TrainerEngine (API-Style)

**Goal:** Turn `UltimateTrainer` into clean `TrainerEngine.run_job()` API

**Current Code:**
```
train.py line 270-1500: UltimateTrainer class
```

**New Structure:**
```python
# trainer/core/engine.py

class TrainerEngine:
    def __init__(self, status_writer):
        self.status_writer = status_writer

    def run_job(self, config: TrainerConfig) -> TrainingResult:
        # Refactored from UltimateTrainer.run()
        # 1. Validate config
        # 2. Load profile
        # 3. Load model (via model_loader.py)
        # 4. Load dataset (via dataset_loader.py)
        # 5. Transform with profile
        # 6. Setup Trainer (via hf_trainer_factory.py)
        # 7. Train
        # 8. Save
        # 9. Return result
```

**CLI Wrapper:**
```python
# trainer/cli_main.py

def main():
    args = parse_args()
    config = ConfigLoader.from_args_and_json(args)

    status_writer = TrainingStatusWriter(
        status_file="status/training_status.json"
    )

    engine = TrainerEngine(status_writer)
    result = engine.run_job(config)

    sys.exit(0 if result.success else 1)

if __name__ == "__main__":
    main()
```

**Daemon Integration:**
```python
# core/training_daemon.py (updated)

from trainer.core.engine import TrainerEngine
from trainer.config.loader import ConfigLoader

engine = TrainerEngine(status_writer)

for file in watch_queue():
    config = ConfigLoader.from_file_and_defaults(
        dataset_path=file,
        base_config="config.json"
    )
    result = engine.run_job(config)
    handle_result(result, file)
```

**Validation:**
- Same training behavior via CLI
- Daemon can call engine directly
- Future: HTTP API can call engine

---

### ⭐ STEP 5: Add Regime-3 Profile (After Refactor)

**Goal:** Plug in regime-3 WITHOUT touching core engine

**New Profile:**
```python
# profiles/regime3.py

class Regime3Profile(DataProfile):
    name = "regime3_symbolic"

    def transform_example(self, example, index, system_prompt):
        # Assume messages already encoded
        # Just enforce regime-3 system prompt
        # with <<ANS_START>> / <<ANS_END>> markers
        ...

    def build_logits_processors(self, tokenizer):
        # Lighter penalties for regime-3
        # Maybe enforce <<ANS_START>>/<<ANS_END>> tokens
        ...

    def get_system_prompt_template(self):
        return """
        Current date: {date}.

        Contract:
        - Use canonical form: (op arg1 arg2)
        - Wrap answer with <<ANS_START>> ... <<ANS_END>>
        """
```

**Usage:**
```bash
# Run with regime-3 profile
python3 trainer/cli_main.py \
    --dataset regime3_data.jsonl \
    --profile regime3_symbolic \
    --output-dir models/regime3_checkpoint
```

**Engine stays unchanged** - just loads different profile!

---

## 📊 Migration Mapping

### Current `train.py` → New Structure

| Current Code (train.py) | New Location | Layer |
|-------------------------|--------------|-------|
| `THINKING_EMOJIS`, `STOP_EMOJI_POOL` | `profiles/emoji_think.py` | 3 |
| `get_random_stop_emoji()` | `profiles/emoji_think.py` | 3 |
| `inject_system_prompt()` | `profiles/emoji_think.py` | 3 |
| `enforce_thinking_requirement()` | `profiles/emoji_think.py` | 3 |
| `enforce_stop_requirement()` | `profiles/emoji_think.py` | 3 |
| `sanitize_example()` | `profiles/emoji_think.py` | 3 |
| `build_think_penalty_processor()` | `profiles/emoji_think.py` | 3 |
| `build_post_stop_penalty_processor()` | `profiles/emoji_think.py` | 3 |
| | | |
| `UltimateTrainer.run()` orchestration | `core/engine.py` | 1 |
| `UltimateTrainer.load_model()` | `core/model_loader.py` | 1 |
| Tokenization logic | `core/dataset_loader.py` | 1 |
| `TrainingArguments` setup | `core/hf_trainer_factory.py` | 1 |
| Checkpoint resume logic | `core/engine.py` | 1 |
| NaN/Inf detection | `utils/safety.py` | 1 |
| | | |
| Hardcoded hyperparams | `config/schema.py` defaults | 2 |
| `config.json` loading | `config/loader.py` | 2 |
| CLI arg parsing | `config/loader.py` | 2 |
| Locked config enforcement | `config/validator.py` | 2 |
| | | |
| `LiveMonitorCallback` | `monitoring/callbacks.py` | 3 |
| `TrainingStatusWriter` | `monitoring/status_writer.py` | 3 |
| Pattern tracking | `monitoring/plugins/pattern_tracker.py` | 3 |
| Layer monitoring | `monitoring/plugins/layer_monitor.py` | 3 |
| Evolution tracking | `monitoring/plugins/evolution_tracker.py` | 3 |

---

## ✅ Success Criteria

After refactor is complete:

**Functional:**
- [ ] Emoji training works identically to before
- [ ] CLI interface unchanged (backward compatible)
- [ ] Daemon can use new engine
- [ ] Web UI receives same status updates
- [ ] All tests pass

**Structural:**
- [ ] Core engine < 500 lines
- [ ] Config is single source of truth
- [ ] Profiles are pluggable (can add regime-3 without touching core)
- [ ] Monitoring is pluggable (can add metrics without touching core)

**Documentation:**
- [ ] Each layer has README explaining purpose
- [ ] Profile interface documented with examples
- [ ] Config schema documented with all fields

---

## 🚧 Risk Mitigation

**Risks:**
1. Breaking existing training behavior
2. Performance regression
3. Daemon integration issues

**Mitigation:**
1. **Git tags** at each step
2. **Parallel testing** - run old vs new on same data
3. **Gradual rollout** - keep old train.py until new is proven
4. **Comprehensive validation** - check metrics, checkpoints, status files

---

## 📅 Timeline Estimate

| Step | Estimated Time | Validation Time |
|------|----------------|-----------------|
| Step 0: Freeze | 5 min | - |
| Step 1: Config | 2 hours | 30 min |
| Step 2: Profile | 3 hours | 1 hour |
| Step 3: Callbacks | 2 hours | 30 min |
| Step 4: Engine API | 4 hours | 2 hours |
| Step 5: Regime-3 | 4 hours | 1 hour |
| **TOTAL** | **~16 hours** | **~5 hours** |

**Recommendation:** Do steps 1-2 first, validate thoroughly, then continue.

---

## 🎯 Next Actions

1. **Review this plan** - Any concerns? Adjustments needed?
2. **Start Step 0** - Create baseline tag
3. **Begin Step 1** - Extract config (lowest risk)
4. **Validate early, validate often**

Once Steps 1-4 are complete, regime-3 can be added as a clean plugin without touching the stable core.

---

**Questions? Ready to begin?**
