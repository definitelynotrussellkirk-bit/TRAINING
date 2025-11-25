# TASK007: Package Structure

**Priority:** LOW
**Effort:** 4 hours
**Dependencies:** TASK004, TASK005
**Files:** Project-wide restructure

---

## Problem

Current code uses `sys.path.insert(...)` hacks to import between modules:

```python
# core/train.py
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "management"))
sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring/servers"))
```

This causes:
- Can't `pip install` and import normally
- Surprise import order issues
- Cross-repo dependencies not expressed in requirements
- Harder packaging for deployment

## Solution

Restructure as a proper Python package with `pyproject.toml`.

## Target Structure

```
TRAINING/
├── pyproject.toml
├── src/
│   └── llm_training/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── training_daemon.py
│       │   └── training/          # From TASK004
│       ├── management/
│       │   ├── __init__.py
│       │   ├── retention_manager.py
│       │   └── backup_manager.py
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── model_comparison_engine.py
│       │   └── deployment_orchestrator.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── server.py          # Was main.py
│       │   └── worker.py          # Was inference_worker.py
│       └── utils/
│           ├── __init__.py
│           └── paths.py           # From TASK006
├── tests/
├── scripts/
├── data/
├── models/                        # Not in package
└── configs/
```

## Implementation Steps

### Step 1: Create pyproject.toml

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-training"
version = "0.1.0"
description = "LLM Training Platform"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.35",
    "datasets>=2.14",
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "pydantic>=2.0",
    "psutil>=5.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
]

[project.scripts]
llm-train = "llm_training.core.train:main"
llm-daemon = "llm_training.core.training_daemon:main"
llm-inference = "llm_training.inference.server:main"

[tool.setuptools.packages.find]
where = ["src"]
```

### Step 2: Create src/ structure

```bash
mkdir -p src/llm_training/{core,management,monitoring,inference,utils}
touch src/llm_training/__init__.py
touch src/llm_training/core/__init__.py
# etc.
```

### Step 3: Move and update imports

```python
# Before (in core/train.py):
sys.path.insert(0, str(Path(__file__).parent.parent / "management"))
from retention_manager import RetentionManager

# After (in src/llm_training/core/train.py):
from llm_training.management.retention_manager import RetentionManager
```

### Step 4: Install in editable mode

```bash
pip install -e ".[dev]"
```

### Step 5: Update entry points

```python
# src/llm_training/core/train.py
def main():
    """CLI entry point."""
    # ... existing main logic ...

if __name__ == "__main__":
    main()
```

Now can run as:
```bash
llm-train --dataset data.jsonl --model qwen3
# or
python -m llm_training.core.train --dataset data.jsonl --model qwen3
```

### Step 6: Handle external dependencies

For `skill_syllo_variant` integration:

```python
# Option A: Make it optional
try:
    from skill_syllo_variant.scripts import export_training_data
    HAS_SYLLO = True
except ImportError:
    HAS_SYLLO = False

# Option B: Add as dependency
# In pyproject.toml:
[project.optional-dependencies]
syllo = ["skill-syllo-variant @ git+https://github.com/..."]
```

## Checkpoints

- [x] Create `pyproject.toml` (flat layout, no src/)
- [x] Create `__init__.py` files for all packages
- [x] Add tool configurations (pytest, black, ruff)
- [x] Define CLI entry points
- [ ] Create `src/llm_training/` directory structure (deferred - using flat layout)
- [ ] Update all imports (remove sys.path hacks) - future work
- [ ] `pip install -e .` works (requires venv)
- [ ] CLI entry points work (`llm-train`, `llm-daemon`)
- [ ] Tests pass with new structure

## Verification

```bash
# Install
pip install -e ".[dev]"

# Imports work
python -c "from llm_training.core.train import main; print('OK')"
python -c "from llm_training.management.retention_manager import RetentionManager; print('OK')"

# CLI works
llm-train --help
llm-daemon --help

# Tests pass
pytest tests/ -v
```

## Migration Strategy

1. Create new structure alongside existing
2. Copy files (don't move yet)
3. Update imports in new location
4. Test new structure works
5. Update scripts/docs to use new paths
6. Delete old locations
7. Each step = separate commit

## Backward Compatibility

During migration, keep old paths working:

```python
# core/__init__.py (temporary)
import warnings
warnings.warn(
    "Importing from core/ is deprecated. Use llm_training.core instead.",
    DeprecationWarning
)
from llm_training.core import *
```
