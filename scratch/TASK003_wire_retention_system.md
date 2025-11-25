# TASK003: Wire Retention System

**Priority:** HIGH
**Effort:** 2 hours
**Dependencies:** None
**Files:** `core/training_daemon.py`, `management/retention_manager.py`

---

## Problem

Two retention systems exist:

1. **Old system:** `management/checkpoint_retention.py` with `enforce_retention()`
   - Currently wired into `training_daemon.py`
   - Simpler logic

2. **New system:** `management/retention_manager.py` with `RetentionManager`
   - 36-hour rule, 150GB limit
   - Protected latest/best/today/yesterday
   - Full pytest suite in `tests/test_retention_manager.py`
   - NOT wired into daemon (per design doc checklist)

The daemon still uses the old system, but the new system is better tested and documented.

## Solution

Wire `RetentionManager` into `training_daemon.py` and retire the old `checkpoint_retention.py`.

## Implementation Steps

### Step 1: Audit current daemon retention calls

```python
# In training_daemon.py, find:
from management.checkpoint_retention import enforce_retention
# and calls like:
enforce_retention(self.checkpoints_dir, max_checkpoints=10)
```

### Step 2: Create adapter for RetentionManager

```python
# management/retention_service.py
from pathlib import Path
from retention_manager import RetentionManager, RetentionConfig

class RetentionService:
    """
    High-level service wrapping RetentionManager for daemon use.
    """
    def __init__(self, base_dir: Path, config: dict = None):
        retention_config = RetentionConfig(
            max_total_size_gb=config.get("max_retention_gb", 150),
            min_checkpoint_age_hours=config.get("min_checkpoint_age_hours", 36),
            protected_keywords=["best", "latest", "today", "yesterday"]
        )
        self.manager = RetentionManager(base_dir, retention_config)

    def enforce(self) -> dict:
        """
        Run retention policy and return summary.

        Returns:
            {
                'deleted': ['checkpoint-1000', ...],
                'protected': ['checkpoint-2000', ...],
                'freed_gb': 12.5
            }
        """
        result = self.manager.enforce()
        return {
            'deleted': [str(p) for p in result.deleted],
            'protected': [str(p) for p in result.protected],
            'freed_gb': result.freed_bytes / 1e9
        }
```

### Step 3: Update training_daemon.py

```python
# Before:
from management.checkpoint_retention import enforce_retention

# After:
from management.retention_service import RetentionService

# In __init__:
self.retention_service = RetentionService(
    self.checkpoints_dir,
    config=self.config.get("retention", {})
)

# Replace enforce_retention calls:
# Before:
enforce_retention(self.checkpoints_dir, max_checkpoints=10)

# After:
result = self.retention_service.enforce()
logger.info(f"Retention: deleted {len(result['deleted'])}, freed {result['freed_gb']:.1f}GB")
```

### Step 4: Add retention config to config.json

```json
{
  "retention": {
    "max_retention_gb": 150,
    "min_checkpoint_age_hours": 36,
    "enabled": true
  }
}
```

### Step 5: Deprecate old system

```python
# management/checkpoint_retention.py
import warnings

def enforce_retention(*args, **kwargs):
    warnings.warn(
        "checkpoint_retention.enforce_retention is deprecated. "
        "Use retention_service.RetentionService instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Optionally forward to new system or raise
    from .retention_service import RetentionService
    ...
```

### Step 6: Update ARCHITECTURE.md

Add section:
```markdown
## Checkpoint Retention

The system uses `RetentionManager` (management/retention_manager.py) with:
- 150GB total size limit
- 36-hour minimum age before deletion
- Protection for: latest, best, today, yesterday checkpoints

Configuration in `config.json`:
```json
{"retention": {"max_retention_gb": 150, "min_checkpoint_age_hours": 36}}
```
```

## Checkpoints

- [x] Create `management/retention_service.py` adapter
- [x] Update `training_daemon.py` imports
- [x] Replace `enforce_retention()` calls with service
- [x] Add retention config to `config.json`
- [x] Add deprecation warning to old `checkpoint_retention.py`
- [ ] Update ARCHITECTURE.md with retention docs (deferred)
- [x] Test: syntax verification passed
- [ ] Test: daemon starts and retention runs without errors (requires restart)

## Verification

```bash
# Check daemon logs for new retention output
tail -f logs/training_output.log | grep -i retention

# Manually trigger retention
python -c "
from management.retention_service import RetentionService
from pathlib import Path
svc = RetentionService(Path('models/current_model'))
print(svc.enforce())
"
```

## Rollback

If new retention misbehaves:
1. Revert `training_daemon.py` to use old `enforce_retention`
2. Restart daemon
3. Debug `RetentionService` separately
