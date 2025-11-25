# Monitoring Tracking Improvements Plan

**Created:** 2025-11-25
**Status:** Proposed
**Related Systems:** Self-Correction Loop, Adversarial Miner

## Problem Summary

Two critical monitoring gaps exist:

### 1. Self-Correction Loop
- **Never writes `status/self_correction.json`** - Plugin expects this file but it doesn't exist
- **No impact tracking** - No way to measure if corrections improve model behavior
- Stats exist only in memory during runtime, lost after process ends

### 2. Adversarial Miner
- **Format mismatch** - Expects `text` field, validation data uses `messages[]` format
- **Status structure mismatch** - Plugin expects `total_examples_mined`, `categories`; miner writes `stats`, `adversarial_count`
- **Effectively a no-op** with current validation data format

## Proposed Solution

Four separate tasks to fix both systems:

| Task | File | Scope | Effort |
|------|------|-------|--------|
| TASK010 | Self-correction status writer | Add `update_status()` method | Small |
| TASK011 | Self-correction impact monitor | New module for impact tracking | Medium |
| TASK012 | Adversarial miner format support | Handle `messages[]` format | Small |
| TASK013 | Adversarial miner status alignment | Match plugin expectations | Small |

## Implementation Order

```
TASK010 → TASK011 (self-correction)
TASK012 → TASK013 (adversarial)
```

Tasks are independent between groups - can parallelize.

## Files Changed

```
Modified:
  monitoring/self_correction_loop.py      # TASK010
  monitoring/adversarial_miner.py         # TASK012, TASK013

New:
  monitoring/self_correction_impact.py    # TASK011
```

## Success Criteria

1. **Self-correction**:
   - `status/self_correction.json` exists and updates after each run
   - Plugin returns meaningful `latest_summary` and `total_corrections`
   - Impact monitor shows error rate changes over time

2. **Adversarial miner**:
   - Successfully mines examples from `messages[]` format data
   - `stats.total_tested > 0` in status file
   - Plugin returns meaningful `latest_summary.total_examples`

## Risk Assessment

- **Low risk** - These are additive changes, no modification to core training logic
- **Testing** - Each task includes its own test verification steps
- **Rollback** - If issues occur, simply stop the monitoring processes
