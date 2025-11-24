# Module Contracts Standard

**Purpose:** Define clear contracts for all Python modules in the TRAINING system

**Last Updated:** 2025-11-24
**Status:** Standard established, rolling out to all modules

---

## Overview

Every module in this codebase should have clear, well-documented contracts that specify:
- What inputs it accepts (types, formats, validation)
- What outputs it produces (types, formats, guarantees)
- What side effects it has (file I/O, network, state changes)
- What errors it can raise (exception types, conditions)

---

## Contract Requirements

### 1. Type Hints (REQUIRED)

All public functions and methods must have full type annotations:

```python
from typing import Optional, Dict, List, Any
from pathlib import Path

def process_data(
    input_file: Path,
    config: Dict[str, Any],
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process training data from file."""
    pass
```

**Requirements:**
- ✅ All parameters typed
- ✅ Return value typed
- ✅ Use specific types (not just `Any` when possible)
- ✅ Use `Optional[T]` for nullable values
- ✅ Import types from `typing` module

### 2. Module Docstring (REQUIRED)

Every module file must start with a module-level docstring:

```python
"""
Training data validator module.

This module provides validation for training datasets in JSONL format.
It checks for format errors, data leakage, and quality issues.

Key Components:
    - DataValidator: Main validation class
    - ValidationIssue: Dataclass for validation errors
    - TokenStats: Statistics about token usage

Usage:
    from core.validator import DataValidator

    validator = DataValidator(tokenizer)
    issues = validator.validate_file("data.jsonl")
"""
```

**Requirements:**
- ✅ One-line summary
- ✅ Detailed purpose description
- ✅ List of key components
- ✅ Usage example

### 3. Class Docstrings (REQUIRED)

Every class must have a comprehensive docstring:

```python
class TrainerEngine:
    """
    High-level training engine that orchestrates complete training jobs.

    Responsibilities:
        - Load and validate training data
        - Setup model and tokenizer
        - Apply data profile transformations
        - Execute training with monitoring
        - Handle checkpointing

    Data Flow:
        1. Load JSONL → validate format
        2. Apply profile transforms (emoji, regime3, etc.)
        3. Create train/eval splits
        4. Setup HuggingFace Trainer with callbacks
        5. Execute training loop
        6. Save checkpoints and status

    Attributes:
        config: TrainerConfig with all parameters
        model: Loaded HuggingFace model
        tokenizer: Tokenizer instance
        profile: Active DataProfile (emoji_think, regime3, etc.)
        callbacks: List of training callbacks

    Example:
        config = TrainerConfig(...)
        engine = TrainerEngine()
        result = engine.run_job(config)
        print(f"Trained for {result.global_step} steps")
    """
```

**Requirements:**
- ✅ Purpose statement
- ✅ Responsibilities list
- ✅ Data flow (if applicable)
- ✅ Attributes documentation
- ✅ Usage example

### 4. Method Docstrings (REQUIRED)

Every public method must document parameters, returns, raises, and side effects:

```python
def run_job(self, config: TrainerConfig) -> TrainingResult:
    """
    Execute a complete training job with the given configuration.

    Args:
        config: TrainerConfig containing:
            - data: DataConfig (dataset path, validation split)
            - hyperparams: Hyperparams (batch_size, learning_rate, etc.)
            - profile: ProfileConfig (data transformation profile)
            - output: OutputConfig (checkpoint directory, save frequency)
            - monitoring: MonitoringConfig (status file, callbacks)

    Returns:
        TrainingResult containing:
            - success: bool (True if training completed)
            - global_step: int (total training steps executed)
            - runtime_sec: float (wall-clock time in seconds)
            - last_checkpoint_path: Optional[str] (path to final checkpoint)
            - final_loss: float (last recorded training loss)
            - summary: Dict[str, Any] (additional metrics)
            - error_message: Optional[str] (if failed)

    Raises:
        ValueError: If config validation fails (invalid paths, etc.)
        RuntimeError: If training fails (OOM, model load error, etc.)
        FileNotFoundError: If dataset file doesn't exist

    Side Effects:
        - Creates checkpoint files in config.output.output_dir
        - Writes training status to config.monitoring.status_file
        - May write profile files to disk (stop signals, etc.)
        - Logs training progress to stdout/stderr

    Example:
        config = TrainerConfig(
            data=DataConfig(dataset="data.jsonl"),
            hyperparams=Hyperparams(batch_size=16)
        )
        result = engine.run_job(config)
        if result.success:
            print(f"Success! Final loss: {result.final_loss}")
    """
```

**Requirements:**
- ✅ One-line summary
- ✅ Args section with full parameter documentation
- ✅ Returns section with structure documentation
- ✅ Raises section with exception types and conditions
- ✅ Side Effects section (file I/O, network, state changes)
- ✅ Example usage

### 5. Dataclass Documentation (REQUIRED)

All dataclasses must document fields and purpose:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingResult:
    """
    Result of a training job execution.

    Contains all relevant information about training outcome including
    success status, performance metrics, and error information if failed.

    Used as return value from TrainerEngine.run_job() and similar methods.
    """
    success: bool  # True if training completed without errors
    global_step: int  # Total number of training steps executed
    runtime_sec: float  # Wall-clock time in seconds
    last_checkpoint_path: Optional[str]  # Path to final checkpoint or None
    final_loss: float  # Last recorded training loss value
    summary: Dict[str, Any]  # Additional metrics (accuracy, etc.)
    error_message: Optional[str] = None  # Error description if failed
```

**Requirements:**
- ✅ Class docstring with purpose
- ✅ Inline comments for each field
- ✅ Type annotations for all fields
- ✅ Default values where appropriate

### 6. Data Format Documentation (REQUIRED for I/O)

Any module that reads/writes files must document formats:

```python
"""
Training Status JSON Format
---------------------------

The training_status.json file has the following structure:

{
    "current_step": 1000,           // int: Current training step
    "total_steps": 10000,           // int: Total steps to train
    "loss": 0.45,                   // float: Current training loss
    "validation_loss": 0.52,        // float: Current validation loss
    "val_train_gap": 0.07,          // float: Gap between val and train loss
    "learning_rate": 0.0002,        // float: Current learning rate
    "epoch": 1.5,                   // float: Current epoch (fractional)
    "examples_seen": 16000,         // int: Total examples processed
    "timestamp": "2025-11-24T10:30:00",  // ISO string: Last update time
    "recent_examples": [            // List: Recent inference examples
        {
            "prompt": "What is 2+2?",         // str: Input prompt
            "golden_answer": "4",             // str: Expected answer
            "model_answer": "4",              // str: Model's answer
            "correct": true                   // bool: Whether correct
        }
    ],
    "metrics": {                    // Dict: Additional metrics
        "train_loss_history": [0.5, 0.48, 0.45],  // List[float]
        "accuracy_history": [0.7, 0.75, 0.8],     // List[float]
        "tokens_per_second": 1250.5               // float
    }
}
"""
```

**Requirements:**
- ✅ Format name and version (if applicable)
- ✅ Complete structure with types
- ✅ Field descriptions with units
- ✅ Example values

---

## Exemplary Modules (Use as Templates)

These modules have excellent contracts and should be used as reference:

### 1. **trainer/config/schema.py** (10/10)
**Why:** Perfect dataclass patterns with comprehensive field documentation

```python
@dataclass
class Hyperparams:
    """
    Training hyperparameters.

    All batch-related parameters are per-device. The effective batch size
    is: batch_size * gradient_accumulation_steps * num_gpus.
    """
    batch_size: int = 16
    eval_batch_size: int = 32
    # ... all fields documented
```

**Key Features:**
- Clear class purpose
- Inline comments on every field
- Relationships explained (effective batch size)
- Default values provided

### 2. **trainer/config/loader.py** (9.5/10)
**Why:** Comprehensive method documentation with precedence rules

```python
def merge_configs(
    self,
    cli_args: Namespace,
    json_config: Dict[str, Any]
) -> TrainerConfig:
    """
    Merge CLI arguments with JSON config, respecting precedence.

    Precedence (highest to lowest):
        1. CLI arguments (if explicitly provided)
        2. JSON config values
        3. Schema defaults

    Args:
        cli_args: Parsed command-line arguments
        json_config: Configuration from config.json

    Returns:
        TrainerConfig with merged values

    Raises:
        ValueError: If conflicting locked parameters detected
    """
```

**Key Features:**
- Explicit precedence rules
- Clear input/output contracts
- Exception conditions documented

### 3. **trainer/profiles/base.py** (9.5/10)
**Why:** Clean ABC interface with example usage

```python
from abc import ABC, abstractmethod

class DataProfile(ABC):
    """
    Abstract base class for data transformation profiles.

    A profile defines how to transform training examples,
    including adding special tokens, stop signals, and
    custom logits processing.

    Subclasses must implement:
        - transform_example()
        - build_logits_processors()
        - get_metadata()
    """

    @abstractmethod
    def transform_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single training example.

        Args:
            example: Dict with 'prompt' and 'completion' keys

        Returns:
            Transformed example with same structure

        Example:
            >>> profile = EmojiThinkProfile()
            >>> result = profile.transform_example({
            ...     "prompt": "What is 2+2?",
            ...     "completion": "4"
            ... })
            >>> print(result["completion"])
            "🤔 Let me think... 4 ✓"
        """
        pass
```

**Key Features:**
- Abstract interface clearly defined
- Required methods listed upfront
- Example usage in docstring
- Input/output structure specified

### 4. **core/atomic_ops.py** (9/10)
**Why:** Focused utility module with clear guarantees

```python
def write_json_atomic(data: Dict[str, Any], path: Path) -> None:
    """
    Atomically write JSON data to file.

    Uses write-to-temp + atomic rename pattern to ensure file
    is never left in partial state. Either write succeeds completely
    or file is unchanged.

    Args:
        data: Dictionary to serialize as JSON
        path: Destination file path

    Returns:
        None

    Raises:
        OSError: If write or rename fails
        JSONEncodeError: If data not JSON-serializable

    Guarantees:
        - Atomic operation (no partial writes visible)
        - Original file unchanged if operation fails
        - Temp files cleaned up on error
    """
```

**Key Features:**
- Operation guarantees documented
- Error handling explicit
- Implementation pattern explained

---

## Common Patterns

### Pattern 1: Configuration Classes

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProcessorConfig:
    """Configuration for data processing."""
    input_dir: Path
    output_dir: Path
    max_workers: int = 4
    batch_size: int = 1000

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {self.input_dir}")
```

### Pattern 2: Result Objects

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ValidationResult:
    """Result of data validation operation."""
    valid: bool  # True if all checks passed
    issues: List[str]  # List of validation errors found
    warnings: List[str]  # Non-fatal issues
    stats: Dict[str, Any]  # Statistics about validated data
    error_message: Optional[str] = None  # Fatal error if validation failed
```

### Pattern 3: Context Managers

```python
from typing import Optional
from pathlib import Path

class SafeFileWriter:
    """
    Context manager for safe file writing with atomic rename.

    Usage:
        with SafeFileWriter(path) as f:
            f.write("data")
        # File atomically written here
    """

    def __init__(self, target_path: Path):
        """
        Initialize safe file writer.

        Args:
            target_path: Final destination for file
        """
        self.target_path = target_path
        self.temp_path: Optional[Path] = None

    def __enter__(self) -> TextIO:
        """Open temporary file for writing."""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Atomically rename temp file to target on success."""
        pass
```

---

## Checklist for New/Updated Modules

Before considering a module "contract complete", verify:

- [ ] Module docstring with purpose and usage
- [ ] All classes have comprehensive docstrings
- [ ] All public methods have docstrings (Args/Returns/Raises)
- [ ] All parameters have type hints
- [ ] All return values have type hints
- [ ] Dataclasses have inline field comments
- [ ] File formats documented (if I/O module)
- [ ] Side effects documented
- [ ] Example usage provided
- [ ] Error conditions explicit

---

## Priority Rollout Plan

### Phase 1: Critical Modules (CURRENT)
1. core/train.py - Main training orchestrator
2. core/training_status.py - Status data format
3. core/training_controller.py - Control interface
4. trainer/core/engine.py - Public training API
5. core/logit_penalty.py - Penalty system

### Phase 2: Core Infrastructure
6. core/training_daemon.py
7. core/training_queue.py
8. core/custom_collator.py
9. core/time_estimator.py
10. core/model_db.py

### Phase 3: Trainer Modules
11. trainer/profiles/emoji_think.py
12. trainer/profiles/regime3.py
13. trainer/monitoring/callbacks.py

### Phase 4: Monitoring & Management
14. monitoring/api/aggregator.py
15. monitoring/curriculum_optimizer.py
16. (remaining monitoring modules)
17. (management modules)

---

## Enforcement

**Manual Review:** All PRs require contract checklist completion

**Future:** Consider automated enforcement via:
- mypy for type hint coverage
- pydocstyle for docstring coverage
- Custom linter for contract completeness

---

## Questions?

See exemplary modules:
- trainer/config/schema.py
- trainer/config/loader.py
- trainer/profiles/base.py
- core/atomic_ops.py

Or ask for clarification on specific contract requirements.
