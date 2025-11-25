# TASK006: Remove Hardcoded Paths

**Priority:** MEDIUM
**Effort:** 3 hours
**Dependencies:** None
**Files:** Multiple (tests, scripts, CLI defaults)

---

## Problem

Hardcoded paths like `/path/to/training/...` appear throughout:

- `tests/test_model.py` - default model paths
- `tests/test_specific.py` - hardcoded checkpoint paths
- `core/training_queue.py` - CLI default `--base-dir`
- Various scripts and docs

This causes:
- Failures on other machines/CI
- Confusion for contributors
- Reproducibility issues

## Solution

Replace hardcoded paths with:
1. Environment variables
2. Repository-relative paths
3. CLI arguments with sensible defaults
4. Test fixtures with temp directories

## Implementation Steps

### Step 1: Create central config module

```python
# core/paths.py
"""Central path configuration."""
import os
from pathlib import Path

def get_base_dir() -> Path:
    """Get base directory from env or default to repo root."""
    if "TRAINING_BASE_DIR" in os.environ:
        return Path(os.environ["TRAINING_BASE_DIR"])

    # Default: find repo root by looking for CLAUDE.md
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "CLAUDE.md").exists():
            return current
        current = current.parent

    raise RuntimeError("Could not find repository root")

def get_models_dir() -> Path:
    return get_base_dir() / "models"

def get_current_model_dir() -> Path:
    return get_models_dir() / "current_model"

def get_status_dir() -> Path:
    return get_base_dir() / "status"

# etc.
```

### Step 2: Update CLI defaults

```python
# core/training_queue.py
from core.paths import get_base_dir

parser.add_argument(
    "--base-dir",
    type=Path,
    default=None,  # Will use get_base_dir() if not provided
    help="Base directory (default: auto-detect or $TRAINING_BASE_DIR)"
)

# In main:
base_dir = args.base_dir or get_base_dir()
```

### Step 3: Update tests to use fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def base_dir(tmp_path):
    """Provides a temporary base directory matching repo structure."""
    # Create expected subdirectories
    (tmp_path / "models/current_model").mkdir(parents=True)
    (tmp_path / "queue/high").mkdir(parents=True)
    (tmp_path / "queue/normal").mkdir(parents=True)
    (tmp_path / "queue/low").mkdir(parents=True)
    (tmp_path / "status").mkdir()
    (tmp_path / "inbox").mkdir()
    return tmp_path

@pytest.fixture
def mock_model(base_dir):
    """Creates a minimal mock model."""
    model_dir = base_dir / "models/current_model/checkpoint-1000"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen2"}')
    (model_dir / "tokenizer.json").write_text('{}')
    return model_dir
```

### Step 4: Search and replace hardcoded paths

```bash
# Find all hardcoded paths
grep -r "/home/user" --include="*.py" --include="*.md" --include="*.sh"
```

Replace patterns:

| Pattern | Replace With |
|---------|--------------|
| `/path/to/training` | `get_base_dir()` or `$TRAINING_BASE_DIR` |
| `Path("/home/user/...")` | `get_base_dir() / "..."` |
| Default paths in argparse | `default=None` + runtime resolution |

### Step 5: Update documentation

```markdown
# In README.md or DEVELOPMENT.md

## Configuration

Set `TRAINING_BASE_DIR` environment variable to override the default base directory:

```bash
export TRAINING_BASE_DIR=/path/to/your/training
```

If not set, the system auto-detects the repository root.
```

### Step 6: Files to update

| File | Change |
|------|--------|
| `core/training_queue.py` | Use `get_base_dir()` for default |
| `core/training_daemon.py` | Use `get_base_dir()` for default |
| `core/train.py` | Use `get_base_dir()` for paths |
| `tests/test_specific.py` | Move to experiments or use fixtures |
| `tests/test_model.py` | Move to experiments or use fixtures |
| `monitoring/*.py` | Use `get_base_dir()` where needed |
| `scripts/*.sh` | Use `$TRAINING_BASE_DIR` |

## Checkpoints

- [x] Create `core/paths.py` with path utilities (supports TRAINING_BASE_DIR env var)
- [x] Update `training_controller.py` CLI defaults (uses get_base_dir())
- [x] Update `training_queue.py` CLI default (auto-detects or uses $TRAINING_BASE_DIR)
- [x] Update `training_daemon.py` CLI default (auto-detects or uses $TRAINING_BASE_DIR)
- [x] Update `train.py` remote path fallback (uses REMOTE_MODELS_DIR constant)
- [x] Create test fixtures in `conftest.py` (done in TASK002)
- [x] Move hardcoded test files to experiments/ (done in TASK002)
- [ ] Update shell scripts to use env var (future work)
- [ ] Update remaining 130+ files in monitoring/, tools/, etc. (future work)
- [ ] Verify: `grep -r "/home/user" --include="*.py"` returns nothing (future work)

## Verification

```bash
# Should find no hardcoded paths in Python
grep -r "/home/user" --include="*.py" | wc -l
# Expected: 0

# Test with different base dir
export TRAINING_BASE_DIR=/tmp/test_training
python -c "from core.paths import get_base_dir; print(get_base_dir())"
# Expected: /tmp/test_training

# Tests should pass with temp dirs
pytest tests/ -v
```
