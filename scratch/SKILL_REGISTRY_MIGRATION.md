# Skill Registry Migration Audit

**Created:** 2025-11-27
**Status:** AUDIT COMPLETE - Migration Pending

## The Problem

Hardcoded skill names (`"syllo"`, `"binary"`, `"SYLLO"`, `"BINARY"`) are scattered throughout the codebase instead of reading from the central `SKILL_REGISTRY` in `guild/skills/loader.py`.

## Source of Truth

```
configs/skills/sy.yaml     # Full skill definition
configs/skills/bin.yaml    # Full skill definition
guild/skills/loader.py     # SKILL_REGISTRY - runtime config
```

## Files Requiring Migration

### CRITICAL: Core Training Pipeline

| File | Lines | Issue |
|------|-------|-------|
| `data_manager/curriculum_manager.py` | 28-142, 196, 325, 391-420 | Hardcoded `"syllo"`, `"binary"` in SKILL_LEVELS, defaults, CLI |
| `data_manager/manager.py` | 59, 94, 156, 174, 196-199, 265, 419, 430 | `"syllo"` defaults everywhere |
| `data_manager/skill_api_client.py` | 33-39, 125, 153, 200-203, 266, 288 | Hardcoded skill configs, CLI choices |
| `monitoring/curriculum_eval_loop.py` | 107, 366, 511, 534, 541 | `"syllo"` as default skill |
| `core/training_daemon.py` | ? | Skill references |

### CRITICAL: Tavern UI

| File | Lines | Issue |
|------|-------|-------|
| `tavern/static/js/game.js` | 32, 45-50, 178-196, 381-408 | Hardcoded `SYLLO`, `BINARY`, `sylloLevel`, `binaryLevel` |
| `tavern/templates/game.html` | 65, 139+ | Hardcoded skill names in HTML |

### HIGH: Evaluation System

| File | Lines | Issue |
|------|-------|-------|
| `monitoring/skill_evaluators.py` | 76, 166, 257-276, 303, 325 | Hardcoded evaluator configs |
| `monitoring/discrimination_generator.py` | 102-103 | Reads `syllo` from state |
| `monitoring/syllo_l1_generator.py` | 146 | Hardcoded `syllo` state access |
| `monitoring/baseline_eval_10level.py` | ? | Syllo-specific eval |
| `monitoring/data_generation_automation.py` | 26-34, 265 | Hardcoded skill configs |

### HIGH: Guild System

| File | Lines | Issue |
|------|-------|-------|
| `guild/integration/curriculum_adapter.py` | 58, 80, 99-105, 493 | Hardcoded defaults and configs |
| `guild/integration/__init__.py` | 39-40 | Example uses `"syllo"` |
| `guild/skills/contract.py` | 11, 46, 50, 155, 318, 327 | Hardcoded examples |
| `guild/dispatch/advisor.py` | ? | Skill references |

### MEDIUM: API & Monitoring

| File | Lines | Issue |
|------|-------|-------|
| `monitoring/api/server.py` | 433-436 | Default skill state |
| `monitoring/api/plugins/curriculum.py` | 113 | Hardcoded max_level |
| `monitoring/api/plugins/skill_metrics.py` | 19, 77, 263 | `'syllable'`, `'binary'` lists |
| `monitoring/gpu_task_scheduler.py` | 281 | Default skill param |
| `monitoring/task_client.py` | 11 | Example uses `"syllo"` |

### MEDIUM: Lineage & Validation

| File | Lines | Issue |
|------|-------|-------|
| `core/lineage.py` | 251, 261 | Filename detection for skill |
| `core/lineage_tracker.py` | ? | Skill tracking |
| `core/validation/spec.py` | ? | Skill validation |
| `core/validation/validator.py` | ? | Skill validation |

### LOW: Tests

| File | Lines | Issue |
|------|-------|-------|
| `tests/guild/test_integration.py` | 469-705 | Uses `"syllo"` in all tests |
| `tests/guild/test_config.py` | 182 | Expects `"syllo"` tag |
| `tests/guild/test_types.py` | ? | Skill type tests |

### LOW: State Files (Will Auto-Update)

| File | Issue |
|------|-------|
| `config.json` | `"current_skill": "syllo"` |
| `data_manager/curriculum_state.json` | Skill state storage |
| `guild/dispatch/progression_state.json` | Skill progression |
| `control/state.json` | Current training state |

### REFERENCE ONLY: Docs & Data

| File | Issue |
|------|-------|
| `CLAUDE.md` | Documentation (update after migration) |
| `ARCHITECTURE.md` | Documentation |
| `CHANGELOG.md` | Historical references (keep) |
| `data/validation/syllo_10level/` | Eval data (keep, rename later) |
| `data/validation/binary/` | Eval data (keep) |
| `configs/quests/reasoning/syllo_puzzle.yaml` | Quest config |

---

## Migration Strategy

### Phase 1: Central Registry Functions

Add utility functions to `guild/skills/loader.py`:

```python
def get_all_skill_ids() -> list[str]:
    """Get all registered skill IDs."""
    return list(SKILL_REGISTRY.keys())

def get_default_skill() -> str:
    """Get the default/primary skill ID."""
    return list(SKILL_REGISTRY.keys())[0]  # First registered = default

def get_skill_config(skill_id: str) -> dict:
    """Get full skill config from YAML + registry."""
    ...
```

### Phase 2: Update Core Pipeline

1. `data_manager/curriculum_manager.py` - Use registry for SKILL_LEVELS
2. `data_manager/manager.py` - Use `get_default_skill()`
3. `data_manager/skill_api_client.py` - Use registry for API URLs
4. `monitoring/curriculum_eval_loop.py` - Use registry

### Phase 3: Update UI

1. `tavern/static/js/game.js` - Fetch skill list from API
2. `tavern/templates/game.html` - Generate skill cards dynamically
3. Add `/api/skills` endpoint to return registered skills

### Phase 4: Update Evaluation

1. `monitoring/skill_evaluators.py` - Load configs from registry
2. Skill APIs provide their own eval sets via `/eval` endpoint

### Phase 5: Update Tests

1. Update test fixtures to use registry
2. Remove hardcoded skill assumptions

---

## API Contract: Skill Server Endpoints

Each skill API (port 8080, 8090, etc.) should provide:

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/health` | GET | `{"status": "ok"}` |
| `/info` | GET | Skill metadata (name, levels, description) |
| `/levels` | GET | Level progression config |
| `/generate` | POST | Training samples |
| `/eval` | GET | Download eval set for current level |
| `/eval/{level}` | GET | Download eval set for specific level |

---

## Naming Migration

| Old | New |
|-----|-----|
| `syllo` | `sy` |
| `SYLLO` | `SY` |
| `binary` | `bin` |
| `BINARY` | `BIN` |

**Note:** Files in `data/` directories keep old names for backward compatibility.
State files will auto-migrate as training continues.

---

## Files NOT to Change

- Historical data in `data/validation/`, `data/evolution_snapshots/`
- Committed state files (will update naturally)
- CHANGELOG entries (historical accuracy)
- Scratch/planning docs (reference only)
