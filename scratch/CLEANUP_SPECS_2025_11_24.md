# Cleanup Specs - 2025-11-24

**Status:** Implementation in progress
**Goal:** Fix validator integration, packaging, and documentation accuracy

---

## Executive Summary

The exploration revealed that the new `core/validation/validator.py` module (created in TASK008) was never integrated into production code. The daemon has its own inline validation at `training_daemon.py:608-771`. This spec defines a **hybrid approach** that preserves all existing safety features while enabling the new architecture.

---

## 1. Validator Integration Strategy

### Current State

| Component | Validator Used | Notes |
|-----------|---------------|-------|
| `core/training_daemon.py` | Inline `validate_data_before_training()` | 164 lines, comprehensive |
| `core/train.py` | Legacy `DatasetValidator` | From `core/validator.py` |
| `core/validation/validator.py` | **ORPHANED** | Never imported |

### Decision: Hybrid Approach

**Rationale:**
- The daemon's inline validation is comprehensive and battle-tested
- The new `DataValidator` has better architecture (levels, programmatic API)
- We don't want to regress safety features

**Plan:**
1. Wire `DataValidator.QUICK` into daemon's `flatten_inbox()` for early rejection
2. Keep inline `validate_data_before_training()` for comprehensive pre-training checks
3. Add deprecation notice to `core/validator.py`
4. Update CLAUDE.md to reflect actual state

### Implementation

**Step 1: Add QUICK validation to daemon file processing**

Location: `core/training_daemon.py` around line 604 (`get_inbox_files` area)

```python
# After flatten_inbox(), add quick validation pass
from core.validation.validator import DataValidator, ValidationLevel

def quick_validate_inbox_file(self, file_path: Path) -> bool:
    """Quick schema validation to reject obviously bad files early."""
    validator = DataValidator()  # No tokenizer needed for QUICK
    result = validator.validate(file_path, ValidationLevel.QUICK)
    if not result.should_proceed():
        self.logger.warning(f"Quick validation failed for {file_path.name}: {result.errors}")
        return False
    return True
```

**Step 2: Integrate into process_inbox flow**

The daemon calls `self.queue.process_inbox()` at line 1321. Add a validation hook:

```python
# In run() loop, after flatten_inbox():
self.flatten_inbox()

# NEW: Quick validate inbox files before queueing
for inbox_file in self.get_inbox_files():
    if not self.quick_validate_inbox_file(inbox_file):
        # Move to failed queue immediately
        self.queue.mark_failed(inbox_file, error="Quick validation failed")
        continue

# Then process remaining valid files
self.queue.process_inbox(default_priority="normal")
```

**Step 3: Add deprecation warning to legacy validator**

```python
# core/validator.py - add at top after imports
import warnings
warnings.warn(
    "core.validator.DatasetValidator is deprecated. "
    "Use core.validation.validator.DataValidator instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)
```

---

## 2. pyproject.toml Restructuring

### Current Problem

```toml
dependencies = [
    "torch>=2.0",           # Heavy GPU package
    "transformers>=4.35",   # Heavy GPU package
    "accelerate>=0.25",     # Heavy GPU package
    ...
]
```

This makes `pip install llm-training` require CUDA even for CI/testing.

### Solution

```toml
[project]
dependencies = [
    # Light dependencies only
    "psutil>=5.9",
    "requests>=2.31",
]

[project.optional-dependencies]
# GPU training dependencies
training = [
    "torch>=2.0",
    "transformers>=4.35",
    "datasets>=2.14",
    "accelerate>=0.25",
    "safetensors>=0.4",
]

# Inference server dependencies
inference = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "pydantic>=2.0",
]

# Development dependencies
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
]

# Full installation (everything)
all = [
    "llm-training[training,inference,dev]",
]
```

### Install Commands

```bash
# CI/testing (light, no GPU)
pip install -e .

# Training machine (full GPU)
pip install -e ".[training]"

# Development (all features)
pip install -e ".[all]"
```

---

## 3. paths.py Logging Enhancement

### Current Problem

`get_base_dir()` silently resolves paths. When debugging path issues, there's no visibility into which resolution method was used.

### Solution

Add logging that fires once at first call:

```python
import logging

logger = logging.getLogger(__name__)
_logged_resolution = False

@lru_cache(maxsize=1)
def get_base_dir() -> Path:
    global _logged_resolution

    # Resolution logic...

    # Log resolution method (once)
    if not _logged_resolution:
        if source == "env":
            logger.info(f"Base dir from $TRAINING_BASE_DIR: {path}")
        elif source == "auto":
            logger.info(f"Base dir auto-detected: {path}")
        elif source == "fallback":
            logger.info(f"Base dir from fallback: {path}")
        _logged_resolution = True

    return path
```

---

## 4. BackgroundWorker Documentation

### Current Limitation

The `timeout` parameter in `Task` is metadata only - it's not enforced. Tasks can run indefinitely.

### Solution

Document the limitation clearly in the docstring:

```python
@dataclass
class Task:
    """
    A task to be executed by BackgroundWorker.

    Note: The `timeout` parameter is metadata only and is NOT enforced.
    Tasks will run to completion regardless of timeout value. If a task
    hangs, it cannot be forcibly terminated - the daemon would need to
    be restarted.

    For critical operations, implement timeout logic within the task itself.
    """
```

---

## 5. CLAUDE.md Updates

### Changes Required

1. **Validator section**: Update to reflect actual state (hybrid approach, not "unified")
2. **Recent updates**: Add this cleanup session
3. **Remove false claims**: TASK008 did NOT integrate the validator into production

### Specific Text Changes

**Old (incorrect):**
```
- TASK008: Created unified DataValidator with QUICK/STANDARD/DEEP levels
```

**New (accurate):**
```
- TASK008: Created DataValidator module (core/validation/validator.py)
  - Supports QUICK/STANDARD/DEEP levels
  - Status: Module exists but not yet integrated into daemon
  - Training daemon uses inline validation (validate_data_before_training)
```

---

## 6. Implementation Order

1. [ ] **pyproject.toml** - Move GPU deps to `[training]` extra
2. [ ] **core/validator.py** - Add deprecation warning
3. [ ] **core/paths.py** - Add resolution logging
4. [ ] **core/daemon/background_worker.py** - Document timeout limitation
5. [ ] **core/training_daemon.py** - Add quick validation hook
6. [ ] **CLAUDE.md** - Update to reflect actual state
7. [ ] **Verify CLI entry points** - Check main() functions exist

---

## 7. Testing Plan

After implementation:

```bash
# Test package installs
pip install -e .                    # Should work without torch
pip install -e ".[training]"        # Should install torch

# Test path resolution logging
python3 -c "from core.paths import get_base_dir; get_base_dir()"

# Test quick validation
python3 -c "from core.validation.validator import DataValidator, ValidationLevel; print('OK')"

# Test daemon starts
python3 core/training_daemon.py --help

# Test deprecation warning
python3 -c "from core.validator import DatasetValidator" 2>&1 | grep -i deprecat
```

---

## Appendix: Files Modified

| File | Change |
|------|--------|
| `pyproject.toml` | Restructure dependencies |
| `core/validator.py` | Add deprecation warning |
| `core/paths.py` | Add resolution logging |
| `core/daemon/background_worker.py` | Document timeout limitation |
| `core/training_daemon.py` | Add quick validation hook |
| `CLAUDE.md` | Update documentation accuracy |
