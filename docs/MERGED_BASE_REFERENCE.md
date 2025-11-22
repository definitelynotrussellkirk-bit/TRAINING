#!/usr/bin/env markdown

# Current Merged Base Reference

- **Active base path:** `/path/to/training/consolidated_models/20251119_152444`
- **Created via:** `python3 consolidate_model.py --base-dir /path/to/training --current-dir current_model_small --description "MERGED 2025-11-19"`
- **Version archive:** `models/versions/v001_20251119_152225_MERGED_2025-11-19`

## Files that explicitly reference the merged base

| File | Purpose |
|------|---------|
| `config.json` | Active daemon/training settings |
| `config_qwen3_06b.json (short/med/long)` | Preset variants used by `toggle_preset.sh` |
| `AGENTS.md` | Repository-wide instructions |
| `README.md` | Quick-start model location |
| `CLAUDE.md` | Summary of current base |
| `SYSTEM_HEALTH_REPORT_2025-11-17.md` | Status checks and config snapshot |
| `PRODUCTION_READINESS_REPORT.md` | Locked base reference |
| `verify_checkpoint_resume.py` | Config guard to ensure base matches |
| Tests (`test_model.py`, `test_minimal_training.py`, `test_callback_minimal.py`) | Smoke tests load the merged base |

## How to find all references

Run:

```bash
rg -n "/path/to/training/consolidated_models/20251119_152444"
```

Update all occurrences whenever a new merged base is created so automation and documentation stay consistent.

## Updating to a new merged base

1. Run `consolidate_model.py` with `--current-dir` pointing to the adapter you want to merge.
2. Update every config/preset (`config*.json`) and re-run `rg` to confirm only the new path remains.
3. Add the new base path to this document and remove the old entry.
