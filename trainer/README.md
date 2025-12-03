# trainer/ - Refactored Training Architecture

**Status:** Production-ready (integrated into core/train.py as of 2025-11-22)
**Version:** 1.0.0
**Architecture:** 3-layer modular system (config, profiles, monitoring)

---

## 📋 OVERVIEW

The `trainer/` module is a clean extraction of training system components from the monolithic `core/train.py` (~1000 lines) into a modular, testable, type-safe architecture (~3,400 lines across 14 modules).

**Purpose:**
- Separate configuration from execution
- Make data profiles pluggable (emoji_think, regime3, custom)
- Enable testing without running full training
- Support multiple training strategies without code duplication

**NOT a replacement:** core/train.py still works and is now enhanced by using trainer/ modules internally.

---

## 🏗️ ARCHITECTURE

3-Layer Design:

```
Layer 1: Engine API          (trainer/core/engine.py)
         TrainerEngine.run_job()
                 ↓
Layer 2: Configuration       (trainer/config/)
         ConfigLoader, TrainerConfig, 8 dataclasses
                 ↓
Layer 3: Plugins             (trainer/profiles/, trainer/monitoring/)
         DataProfiles, Callbacks, Status Writers
```

### Layer 1: Engine API

**File:** `trainer/core/engine.py`
**Purpose:** High-level orchestration
**Status:** Fully implemented, API-stable, currently used via trainer/cli_main.py (daemon still uses UltimateTrainer for now)

```python
from trainer.core import TrainerEngine
from trainer.config import create_default_config
from trainer.monitoring import TrainingStatusWriter

# Create configuration
config = create_default_config(
    model_path="models/Qwen3-0.6B",
    dataset_path="data/train.jsonl",
    output_dir="outputs/engine_run",
    base_model="Qwen/Qwen3-0.6B",
    model_architecture="Qwen3ForCausalLM",
    max_context_length=4096,
    vocab_size=151936,
)

# Initialize engine with status writer
status_writer = TrainingStatusWriter("status/training_status.json")
engine = TrainerEngine(status_writer)

# Run training job
result = engine.run_job(config)
```

### Layer 2: Configuration

**Files:**
- `trainer/config/schema.py` - 8 dataclasses (type-safe config)
- `trainer/config/loader.py` - ConfigLoader (JSON + CLI merging)

**Key Dataclasses:**
```python
@dataclass
class TrainerConfig:
    hyperparams: Hyperparams          # batch_size, lr, epochs, etc.
    model: ModelConfig                # model_path, precision, etc.
    data: DataConfig                  # max_length, packing, etc.
    profile: ProfileConfig            # name, stop_sequences, etc.
    inference: InferenceConfig        # eval_steps, num_samples, etc.
    output: OutputConfig              # output_dir, save_steps, etc.
    system: SystemConfig              # base_dir, status_file, etc.
    debug: DebugConfig                # verbose, dry_run, etc.
```

**Usage:**
```python
from trainer.config.loader import ConfigLoader

# Load from JSON + CLI args
loader = ConfigLoader("config.json")
config = loader.merge_with_cli_args(cli_args)

# Or from config.json alone
loader = ConfigLoader("config.json")
config = loader.load()
```

### Layer 3: Plugins

#### Profiles (Data Processing)

**Files:**
- `trainer/profiles/base.py` - DataProfile ABC (interface)
- `trainer/profiles/emoji_think.py` - Emoji thinking profile
- `trainer/profiles/regime3.py` - Symbolic reasoning profile

**Interface:**
```python
class DataProfile(ABC):
    @abstractmethod
    def prepare_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert example to ChatML format"""

    @abstractmethod
    def get_stop_sequences(self) -> List[str]:
        """Return stop tokens for this profile"""

    @abstractmethod
    def format_system_prompt(self, base_prompt: str) -> str:
        """Enhance system prompt"""
```

**Usage:**
```python
from trainer.profiles import get_profile

# Get profile by name
profile = get_profile("emoji_think")
messages = profile.prepare_messages(example)
stop_sequences = profile.get_stop_sequences()
```

#### Monitoring (Status Updates)

**Files:**
- `trainer/monitoring/status_writer.py` - Extracted from core/training_status.py
- `trainer/monitoring/callbacks.py` - HuggingFace Trainer callbacks

---

## 🚀 USAGE

### Production Usage (core/train.py)

The trainer/ modules are integrated into `core/train.py`:

```python
# core/train.py now uses:
from trainer.config.loader import ConfigLoader
from trainer.profiles import get_profile

# Load config
loader = ConfigLoader("config.json")
config = loader.merge_with_cli_args(args)

# Get profile
profile = get_profile(config.profile.name)
stop_sequences = profile.get_stop_sequences()
```

### Standalone Usage (CLI)

**Demo script:** `trainer/cli_main.py`

```bash
# Run with profile
python3 trainer/cli_main.py --dataset data.jsonl --profile emoji_think

# Override config
python3 trainer/cli_main.py --dataset data.jsonl --batch-size 32 --lr 1e-4
```

### Programmatic Usage

```python
from pathlib import Path
from trainer.config.loader import ConfigLoader
from trainer.profiles import get_profile

# Load configuration
loader = ConfigLoader(Path("config.json"))
config = loader.load()

# Switch profile dynamically
config.profile.name = "regime3"
profile = get_profile(config.profile.name)

# Process data with profile
for example in dataset:
    messages = profile.prepare_messages(example)
    # ... train on messages
```

---

## 📦 MODULE STRUCTURE

```
trainer/
├── __init__.py                    # Package exports
├── cli_main.py                    # CLI demonstration script
│
├── config/                        # Layer 2: Configuration
│   ├── __init__.py
│   ├── schema.py                  # 8 dataclasses (TrainerConfig, etc.)
│   └── loader.py                  # ConfigLoader (JSON + CLI merge)
│
├── profiles/                      # Layer 3: Data profiles
│   ├── __init__.py                # get_profile() factory
│   ├── base.py                    # DataProfile ABC
│   ├── emoji_think.py             # Emoji thinking profile
│   └── regime3.py                 # Symbolic reasoning profile
│
├── monitoring/                    # Layer 3: Monitoring
│   ├── __init__.py
│   ├── status_writer.py           # TrainingStatusWriter (extracted)
│   ├── callbacks.py               # HuggingFace Trainer callbacks
│   └── preview_backend.py         # Inference preview backend
│
└── core/                          # Layer 1: Engine (experimental)
    ├── __init__.py
    └── engine.py                  # TrainerEngine.run_job()
```

---

## 🔧 PROFILES

### emoji_think (Default)

**Purpose:** Teach model to think before answering using emoji markers

**Features:**
- Prepends `🤔🤔🤔🤔` thinking marker to outputs
- Adds thinking instruction to prompts
- Stop sequences: `["\n\n", "<think>"]`

**Example:**
```
User: What is 2+2?
Model: 🤔🤔🤔🤔
Let me calculate: 2 + 2 = 4

The answer is 4
```

### regime3 (Symbolic Reasoning)

**Purpose:** Structured symbolic reasoning with explicit steps

**Features:**
- Structured output format (steps, reasoning, answer)
- Enhanced reasoning instructions
- Stop sequences: `["</reasoning>", "\n\n"]`

**Example:**
```
User: What is 2+2?
Model:
Step 1: Identify the operation (addition)
Step 2: Apply the rule: 2 + 2
Step 3: Calculate the result: 4

Answer: 4
```

### Custom Profiles

Create your own by subclassing `DataProfile`:

```python
from trainer.profiles.base import DataProfile

class MyProfile(DataProfile):
    def prepare_messages(self, example):
        # Your logic here
        pass

    def get_stop_sequences(self):
        return ["\n\n", "DONE"]

    def format_system_prompt(self, base):
        return base + "\n\nUse my custom format."
```

---

## 🧪 TESTING

The trainer/ modules are fully tested:

```bash
# Test emoji_think profile (6 tests)
python3 scratch/test_emoji_profile.py

# Test regime3 profile (7 tests)
python3 scratch/test_regime3_profile.py

# Test config loader
python3 trainer/config/loader.py  # Has __main__ with demo

# Test profile loading
python3 trainer/profiles/__init__.py  # Has __main__ with demo
```

**Test Coverage:**
- 13/13 tests passing
- Profiles: emoji_think (6 tests), regime3 (7 tests)
- Config loading, CLI merging, profile registration
- Backward compatibility with core/train.py

---

## 📊 COMPARISON: Before vs After

### Before Refactor (core/train.py)

```python
# Monolithic core/train.py (~1000 lines)
# - Config hardcoded or scattered across file
# - Profile logic embedded in main loop
# - Hard to test, hard to extend
# - No type safety

# Example: Change profile
if args.profile == "emoji":
    # Inline emoji logic here
    prompt = add_emoji_instruction(prompt)
    output = prepend_emoji(output)
elif args.profile == "regime3":
    # Inline regime3 logic here
    # ... different logic ...
```

### After Refactor (trainer/ modules)

```python
# Modular architecture (~3,400 lines, 14 modules)
# - Config in dataclasses (type-safe)
# - Profiles are plugins
# - Each module testable independently
# - 100% backward compatible

# Example: Change profile
from trainer.profiles import get_profile
profile = get_profile(config.profile.name)
messages = profile.prepare_messages(example)
```

**Benefits:**
- **Testability:** Unit test each profile without running training
- **Extensibility:** Add new profiles without modifying core code
- **Type Safety:** Catch errors at load time, not runtime
- **Maintainability:** Changes isolated to single module
- **Documentation:** Each module self-contained

---

## 🔄 MIGRATION GUIDE

### For Users (Using core/train.py)

**No changes needed!** core/train.py still works exactly as before.

**To use new profiles:**
1. Edit config.json: `{"profile": {"name": "regime3"}}`
2. Run training: `python3 core/train.py --dataset data.jsonl`

### For Developers (Extending System)

**Adding a new profile:**

1. Create profile file: `trainer/profiles/my_profile.py`
```python
from trainer.profiles.base import DataProfile

class MyProfile(DataProfile):
    # Implement 3 abstract methods
    pass
```

2. Register in `trainer/profiles/__init__.py`:
```python
from .my_profile import MyProfile
PROFILES["my_profile"] = MyProfile
```

3. Use it: `{"profile": {"name": "my_profile"}}`

---

## 🐛 GOTCHAS

### 1. Config vs Args Priority

CLI args override config.json:
```bash
# config.json has batch_size=16
python3 trainer/cli_main.py --batch-size 32  # Uses 32 (CLI wins)
```

### 2. Profile Not Found

If profile name is invalid, falls back to "emoji_think":
```python
config.profile.name = "invalid"
profile = get_profile(config.profile.name)  # Returns emoji_think + warning
```

### 3. Precision String vs Enum

Precision can be string or torch dtype:
```python
# Both valid:
config.model.precision = "bfloat16"  # String
config.model.precision = torch.bfloat16  # Dtype
```

### 4. Status Writer Compatibility

trainer/monitoring/status_writer.py is a copy of core/training_status.py:
- Keep in sync manually
- Or import from core/ directly

---

## 📚 DOCUMENTATION

Each module has comprehensive docstrings. Read them for details:

```bash
# View module docs
python3 -m pydoc trainer.config.schema
python3 -m pydoc trainer.profiles.base
python3 -m pydoc trainer.core.engine
```

---

## 🎯 ROADMAP

### Completed (2025-11-22)

- ✅ Step 1: Extract config system (schema.py, loader.py)
- ✅ Step 2: Extract emoji_think profile
- ✅ Step 3: Create DataProfile ABC
- ✅ Step 4: Create regime3 profile
- ✅ Step 5: Extract monitoring components
- ✅ Step 6: Production integration (core/train.py)

### Future Work

- [ ] Step 7: Migrate core/train.py to use TrainerEngine.run_job()
- [ ] Step 8: Add more profiles (chain-of-thought, tool-use, etc.)
- [ ] Step 9: Profile composition (emoji_think + regime3)
- [ ] Step 10: Auto-profile selection (analyze dataset → suggest profile)

---

## 🤝 RELATED MODULES

- **core/train.py** - Production training script (uses trainer/ internally)
- **arena/hero_loop.py** - Autonomous training daemon
- **core/training_status.py** - Status tracking (copied to trainer/monitoring/)
- **core/custom_collator.py** - ChatML data collation
- **core/logit_penalty.py** - Penalty system (used by profiles)

---

## 📝 HISTORY

- **2025-11-21:** Refactor planning (REFACTOR_PLAN.md)
- **2025-11-22 (Morning):** Steps 1-5 completed (~3 hours)
  - Created config system
  - Extracted emoji_think profile
  - Created DataProfile ABC + regime3
  - Extracted monitoring components
  - 13/13 tests passing
- **2025-11-22 (Afternoon):** Step 6 completed (production integration)
  - core/train.py now uses ConfigLoader
  - Profiles active (emoji_think, regime3)
  - 100% backward compatible
  - Pushed to GitHub with 6 git tags

---

**For detailed information on specific modules, see their inline documentation.**
