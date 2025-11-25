# TASK008: Consolidate Data Validation

**Priority:** LOW
**Effort:** 2 hours
**Dependencies:** TASK004, TASK005
**Files:** `core/validator.py`, `core/training_daemon.py`

---

## Problem

Two validation layers with overlapping responsibilities:

1. **DatasetValidator** (`core/validator.py`)
   - Used by `UltimateTrainer`
   - Checks: JSONL structure, roles, token lengths, leakage, duplicates
   - Runs before training, asks user to proceed

2. **Inline validation** (`training_daemon.py`)
   - `validate_data_before_training()` method
   - Loads tokenizer, samples 100 lines
   - Computes token length stats (p95/p99 vs max_length)
   - Can quarantine failing files
   - Runs synchronously, can block daemon

This causes:
- Duplicated logic (both check schema and lengths)
- Inconsistent behavior
- Blocking operations in daemon

## Solution

Create a unified validation service used by both trainer and daemon.

## Target Architecture

```python
# core/validation/
├── __init__.py
├── validator.py         # Core validation logic
├── schema_checks.py     # JSONL structure validation
├── content_checks.py    # Leakage, duplicates, quality
└── length_checks.py     # Token length analysis
```

## Implementation Steps

### Step 1: Define validation levels

```python
# core/validation/validator.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class ValidationLevel(Enum):
    QUICK = "quick"      # Schema only, fast
    STANDARD = "standard"  # Schema + lengths
    DEEP = "deep"        # Schema + lengths + content quality

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: dict  # Token length stats, etc.

    def should_proceed(self) -> bool:
        return self.valid and len(self.errors) == 0
```

### Step 2: Create unified validator

```python
# core/validation/validator.py
class DataValidator:
    """
    Unified data validation for trainer and daemon.
    """

    def __init__(self, tokenizer=None, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def validate(
        self,
        file_path: Path,
        level: ValidationLevel = ValidationLevel.STANDARD,
        sample_size: int = 100
    ) -> ValidationResult:
        """
        Validate a dataset file.

        Args:
            file_path: Path to JSONL file
            level: How thorough to validate
            sample_size: Max samples for length analysis

        Returns:
            ValidationResult with errors, warnings, stats
        """
        errors = []
        warnings = []
        stats = {}

        # Always check schema
        schema_result = self._check_schema(file_path)
        errors.extend(schema_result.errors)

        if level in [ValidationLevel.STANDARD, ValidationLevel.DEEP]:
            # Check token lengths
            if self.tokenizer:
                length_result = self._check_lengths(file_path, sample_size)
                warnings.extend(length_result.warnings)
                stats.update(length_result.stats)

        if level == ValidationLevel.DEEP:
            # Check content quality
            content_result = self._check_content(file_path)
            warnings.extend(content_result.warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def _check_schema(self, file_path: Path) -> ValidationResult:
        """Check JSONL structure and required fields."""
        # Moved from core/validator.py
        ...

    def _check_lengths(self, file_path: Path, sample_size: int) -> ValidationResult:
        """Analyze token lengths."""
        # Moved from training_daemon.py.validate_data_before_training
        ...

    def _check_content(self, file_path: Path) -> ValidationResult:
        """Check for leakage, duplicates, quality issues."""
        # Moved from core/validator.py
        ...
```

### Step 3: Update UltimateTrainer

```python
# core/train.py
from validation.validator import DataValidator, ValidationLevel

class UltimateTrainer:
    def validate_dataset(self, path: Path) -> bool:
        validator = DataValidator(self.tokenizer, self.max_length)
        result = validator.validate(path, level=ValidationLevel.DEEP)

        if result.errors:
            logger.error(f"Validation errors: {result.errors}")
            return False

        if result.warnings:
            logger.warning(f"Validation warnings: {result.warnings}")
            # Ask user to proceed
            if not self._confirm_proceed():
                return False

        logger.info(f"Validation stats: {result.stats}")
        return True
```

### Step 4: Update TrainingDaemon

```python
# core/training_daemon.py
from validation.validator import DataValidator, ValidationLevel

class TrainingDaemon:
    def __init__(self, ...):
        # Lazy tokenizer loading for validation
        self._tokenizer = None
        self.validator = None

    def validate_data_before_training(self, file_path: Path) -> bool:
        """Quick validation before adding to queue."""
        if self.validator is None:
            self.validator = DataValidator(
                tokenizer=self._get_tokenizer(),
                max_length=self.config.get("max_length", 4096)
            )

        # Use QUICK for daemon (non-blocking)
        result = self.validator.validate(
            file_path,
            level=ValidationLevel.QUICK,
            sample_size=50  # Smaller sample for speed
        )

        if not result.valid:
            logger.error(f"Validation failed for {file_path}: {result.errors}")
            self._quarantine(file_path)
            return False

        return True
```

### Step 5: Delete old validator code

- Remove duplicated validation from `training_daemon.py`
- Keep `core/validator.py` as thin wrapper or remove entirely
- Update imports throughout codebase

## Checkpoints

- [ ] Create `core/validation/` package
- [ ] Implement `DataValidator` with level support
- [ ] Migrate schema checks from old validator
- [ ] Migrate length checks from daemon
- [ ] Update `UltimateTrainer` to use new validator
- [ ] Update `TrainingDaemon` to use new validator
- [ ] Remove old duplicated code
- [ ] Add unit tests for validator

## Verification

```bash
# Validator should work standalone
python -c "
from core.validation.validator import DataValidator, ValidationLevel
v = DataValidator(max_length=4096)
result = v.validate(Path('data/test.jsonl'), ValidationLevel.QUICK)
print(result)
"

# Training should still validate
python core/train.py --dataset data.jsonl --validate-only

# Daemon should validate incoming files
# (Check logs for validation messages)
```

## Benefits

1. **Single source of truth** for validation logic
2. **Configurable depth** - quick for daemon, deep for trainer
3. **Non-blocking** - daemon uses quick validation
4. **Testable** - can unit test validator in isolation
5. **Consistent** - same rules applied everywhere
