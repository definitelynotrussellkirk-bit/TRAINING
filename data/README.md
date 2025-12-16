# Data Directory Lifecycle

This directory contains training data at various stages of the pipeline.

## Directory Structure

| Directory | Status | Purpose | Gitignored? |
|-----------|--------|---------|-------------|
| `raw/` | Scaffold | Original source data | No |
| `generated/` | Active | Training files created by curriculum | No |
| `validation/` | Active | Evaluation/benchmark datasets | No |
| `staging/` | Scaffold | Temp staging for data prep | No |
| `registry/` | Scaffold | Data asset registry | No |
| `eval_banks/` | Used | Evaluation sample banks | No |
| `predictions/` | Scaffold | Model prediction outputs | No |
| `prediction_history/` | Generated | Historical predictions | **Yes** |
| `preview_history/` | Generated | Preview outputs | No |
| `evolution_snapshots/` | Generated | Training evolution data | **Yes** |
| `self_correction/` | Used | Self-correction training data | No |
| `validated/` | Scaffold | Post-validation data | No |
| `reports/` | Scaffold | Data analysis reports | No |
| `aimo/` | Active | AIMO competition data | No |

## Lifecycle

```
raw/ → generated/ → staging/ → queue/inbox/ → training
                 ↓
            validation/  (evaluation sets)
```

## What to Ignore

- `prediction_history/` - Large, machine-specific
- `evolution_snapshots/` - Large, generated

## Scaffold Directories

Empty directories marked "Scaffold" are placeholders for features that may be:
- Partially implemented
- Planned for future use
- Used by specific code paths

Don't delete them - code may expect them to exist.
