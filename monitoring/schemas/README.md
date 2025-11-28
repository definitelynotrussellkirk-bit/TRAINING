# Data Schemas - Master Monitoring System

**Phase 1, Task 1.3 Complete**
**Generated:** 2025-11-23 23:35

## Overview

This directory contains JSON Schema definitions for all data sources in the Master Monitoring System. These schemas define the expected structure, types, and validation rules for each data endpoint.

## Schema Files

### Local (4090 RTX Training Machine)

1. **training_status.schema.json**
   - Source: `status/training_status.json`
   - Refresh: Real-time (every 1-5 seconds)
   - Critical Fields: `status`, `current_step`, `loss`, `validation_loss`
   - Size: ~40KB per update

### Remote (3090 RTX Intelligence Machine)

2. **curriculum_optimization.schema.json**
   - Source: `ssh://inference.local/~/TRAINING/status/curriculum_optimization.json`
   - Refresh: Every 5 minutes
   - Critical Fields: `evaluations[].difficulties.{easy,medium,hard}.accuracy`
   - Size: ~2KB per update

3. **adversarial_mining.schema.json**
   - Source: `ssh://inference.local/~/TRAINING/status/adversarial_mining.json`
   - Refresh: Every 5 minutes
   - Critical Fields: `mining_runs[].stats.incorrect_predictions`
   - Size: ~2KB per update

## Additional Schemas (To Be Created)

Phase 2 will add schemas for:

- `regression_monitoring.schema.json` - Regression detection
- `model_comparisons.schema.json` - Checkpoint rankings
- `confidence_calibration.schema.json` - Confidence bins
- `automated_testing.schema.json` - Validation results
- `checkpoint_sync.schema.json` - Sync status
- `gpu_stats.schema.json` - GPU memory/utilization
- `queue_status.schema.json` - Training queue state

## Usage

### Validation Example

```python
import json
import jsonschema

# Load schema
with open('monitoring/schemas/training_status.schema.json') as f:
    schema = json.load(f)

# Load data
with open('status/training_status.json') as f:
    data = json.load(f)

# Validate
try:
    jsonschema.validate(instance=data, schema=schema)
    print("✓ Valid")
except jsonschema.ValidationError as e:
    print(f"✗ Validation failed: {e.message}")
```

### Plugin Integration

When creating data source plugins (Phase 2), use these schemas to:

1. **Validate incoming data** - Ensure data matches expected format
2. **Handle missing fields** - Gracefully degrade when optional fields absent
3. **Type checking** - Convert types as needed (e.g., string timestamps to datetime)
4. **Documentation** - Auto-generate API docs from schemas

## Data Characteristics

### Update Frequencies

- **Real-time (< 5s):** training_status
- **Fast (5min):** curriculum, adversarial, regression, model_comparison
- **Medium (10min):** confidence, testing
- **Slow (30min+):** checkpoint_sync

### Data Sizes

- **Small (< 5KB):** Most status files
- **Medium (5-50KB):** training_status, model_comparisons
- **Large (> 50KB):** Historical logs, full validation results

### Critical vs Optional

**Critical** (system breaks without):
- training_status.{status, current_step, loss}
- curriculum.evaluations[].difficulties
- gpu_stats.{vram_used, vram_total}

**Optional** (nice to have):
- training_status.recent_examples
- curriculum.evaluations[].timestamp
- Visualization metadata

## Schema Evolution

### Version Strategy

- Schemas use JSON Schema Draft 7
- Breaking changes = new schema file (e.g., `training_status.v2.schema.json`)
- Non-breaking additions = update existing schema
- Mark deprecated fields with `"deprecated": true` in description

### Compatibility

Plugins must handle:
- Missing optional fields (use defaults)
- Extra fields (ignore gracefully)
- Type variations (e.g., `null` for infinity in loss values)

## Testing

Schema files validated with:
```bash
python3 monitoring/tests/test_all_endpoints.py
```

This test suite:
- Loads each schema
- Fetches live data from each source
- Validates data against schema
- Reports pass/fail per source

## Next Steps (Phase 2)

1. Create remaining schema files for 7 intelligence systems
2. Add GPU stats schemas (local + remote)
3. Add queue/storage schemas
4. Create skill manifest schema
5. Build schema validation into plugins
