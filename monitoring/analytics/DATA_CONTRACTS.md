# Training Analytics Data Contracts

This document defines the data structures and calculations for the training analytics system.

**Last Updated:** 2025-11-25
**Update Frequency:** Daemons run every 10 minutes on 3090

---

## Overview

Two analytics daemons run continuously on the 3090, analyzing model weights:
1. **Layer Drift Monitor** - Compares current model weights to base model
2. **Parameter Stability Monitor** - Checks for weight explosion/vanishing

Data is written to `status/` JSON files and served via the unified API.

---

## 1. Layer Drift Analysis

### Purpose
Tracks how much each transformer layer has changed from the base model during training. Helps understand:
- Which layers are learning task-specific features
- Whether training is affecting all layers or concentrated in specific regions
- Embedding adaptation for vocabulary changes

### Source Files
- **Generator:** `monitoring/analytics/layer_drift_monitor.py`
- **Output:** `status/layer_drift.json`
- **History:** `status/layer_drift_history.json`

### Data Schema

```json
{
  "timestamp": "2025-11-25T02:54:38.597583",
  "reference_checkpoint": "Qwen3-0.6B",
  "current_checkpoint": "deployed",
  "current_step": null,
  "total_params": 596049920,
  "total_relative_change": 0.277703,
  "summary": {
    "num_transformer_layers": 28,
    "avg_relative_change": 0.285406,
    "max_drift_layer": 14,
    "max_drift_value": 0.351597,
    "min_drift_layer": 27,
    "min_drift_value": 0.113497,
    "top_layers_avg": 0.190713,
    "bottom_layers_avg": 0.288328,
    "pattern": "uniform"
  },
  "layers": [
    {
      "layer_idx": 0,
      "relative_change": 0.303384,
      "delta_norm": 49.6932,
      "reference_norm": 163.8,
      "current_norm": 169.27
    }
    // ... more layers
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `reference_checkpoint` | string | Base model name (e.g., "Qwen3-0.6B") |
| `current_checkpoint` | string | Trained model checkpoint being analyzed |
| `current_step` | int\|null | Training step if from checkpoint-N format |
| `total_params` | int | Total parameter count in model |
| `total_relative_change` | float | Overall weight change (0-1 scale, multiply by 100 for %) |

**Per-Layer Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `layer_idx` | int | Layer index (-1 for embeddings, 0-27 for transformers) |
| `relative_change` | float | `delta_norm / reference_norm` - fraction of original weights changed |
| `delta_norm` | float | L2 norm of (current_weights - reference_weights) |
| `reference_norm` | float | L2 norm of base model weights for this layer |
| `current_norm` | float | L2 norm of trained model weights for this layer |

**Summary Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | string | Distribution pattern: "uniform", "top_heavy", or "bottom_heavy" |
| `top_layers_avg` | float | Average relative change in top 7 layers (21-27) |
| `bottom_layers_avg` | float | Average relative change in bottom 7 layers (0-6) |
| `max_drift_layer` | int | Layer with highest relative change |
| `min_drift_layer` | int | Layer with lowest relative change |

### Calculation Details

**Relative Change:**
```python
delta = current_weights - reference_weights
relative_change = torch.norm(delta) / torch.norm(reference_weights)
```

**Pattern Classification:**
- If `top_layers_avg > bottom_layers_avg * 1.5`: "top_heavy"
- If `bottom_layers_avg > top_layers_avg * 1.5`: "bottom_heavy"
- Otherwise: "uniform"

### Interpretation Guide

| Total Change | Meaning |
|--------------|---------|
| < 10% | Minimal training effect |
| 10-30% | Normal training adaptation |
| > 30% | Significant model modification |

| Pattern | Typical Cause |
|---------|---------------|
| uniform | Full fine-tuning, task affects whole model |
| top_heavy | Task-specific adaptation (typical for instruction tuning) |
| bottom_heavy | Unusual - may indicate low-level feature learning |

---

## 2. Parameter Stability Analysis

### Purpose
Monitors weight norms to detect:
- Exploding weights (training instability)
- Vanishing weights (dying layers)
- Overall model health

### Source Files
- **Generator:** `monitoring/analytics/parameter_stability.py`
- **Output:** `status/parameter_stability.json`
- **History:** `status/parameter_stability_history.json`

### Data Schema

```json
{
  "timestamp": "2025-11-25T02:51:43.024363",
  "checkpoint": "deployed",
  "step": null,
  "summary": {
    "num_layers": 28,
    "avg_weight_norm": 222.3419,
    "std_weight_norm": 143.9439,
    "max_weight_norm": 649.2462,
    "min_weight_norm": 126.7223,
    "health_status": "warning",
    "total_alerts": 2,
    "warning_alerts": 2,
    "critical_alerts": 0
  },
  "alerts": [
    {
      "layer": 26,
      "type": "exploding",
      "severity": "warning",
      "value": 106.5,
      "threshold": 100.0,
      "message": "Layer 26 has extreme weight values"
    }
  ],
  "layers": [
    {
      "layer_idx": 0,
      "weight_norm": 169.27,
      "max_abs_weight": 96.5,
      "min_abs_weight": 1.28e-8
    }
    // ... more layers
  ]
}
```

### Field Definitions

**Summary Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `health_status` | string | "healthy", "warning", or "critical" |
| `avg_weight_norm` | float | Mean L2 norm across all layers |
| `std_weight_norm` | float | Standard deviation of norms |
| `total_alerts` | int | Number of stability alerts triggered |

**Per-Layer Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `weight_norm` | float | L2 norm of all weights in layer |
| `max_abs_weight` | float | Maximum absolute value of any single weight |
| `min_abs_weight` | float | Minimum absolute value of any non-zero weight |

**Alert Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | "exploding" or "vanishing" |
| `severity` | string | "warning" or "critical" |
| `value` | float | The problematic weight value |
| `threshold` | float | Threshold that was exceeded |

### Alert Thresholds

| Alert Type | Warning | Critical |
|------------|---------|----------|
| Exploding (max_abs_weight) | > 100 | > 150 |
| Vanishing (min_abs_weight) | < 1e-10 | < 1e-12 |

### Health Status Criteria

- **healthy:** No alerts
- **warning:** Has warning-level alerts only
- **critical:** Has any critical-level alerts

### Interpretation Guide

| Observation | Possible Cause | Action |
|-------------|----------------|--------|
| High max_abs_weight in top layers | Normal for LMs (residual accumulation) | Monitor trend |
| Rapidly increasing norms | Learning rate too high | Consider reducing LR |
| Very small weights in early layers | Dying neurons | Check gradient flow |

---

## API Endpoint

The data is served via the unified monitoring API:

**Endpoint:** `GET /api/unified`

**Response Structure:**
```json
{
  "sources": {
    "training_analytics": {
      "status": "ok",
      "data": {
        "layer_drift": { ... },
        "parameter_stability": { ... },
        "data_file_impact": { ... }
      }
    }
  }
}
```

**Refresh Interval:** 60 seconds (data updates every 10 minutes)

---

## Dashboard Visualization

Two cards on the master dashboard display this data:

### Layer Drift Card
- **Total Change:** Overall weight modification percentage
- **Pattern:** Distribution of changes (uniform/top_heavy/bottom_heavy)
- **Max Drift:** Layer with most change
- **Bar Chart:** Per-layer drift visualization
  - Green: < 20% change
  - Yellow: 20-35% change
  - Red: > 35% change

### Parameter Stability Card
- **Health:** Overall status (healthy/warning/critical)
- **Avg Norm:** Mean weight norm across layers
- **Alerts:** Count and list of stability warnings
- **Bar Chart:** Per-layer norm visualization with alert highlighting

---

## Data Flow

```
3090: Checkpoint appears
  ↓
3090: layer_drift_monitor.py (every 10 min)
  ↓
3090: status/layer_drift.json
  ↓
4090: TrainingAnalyticsPlugin (SSH fetch, every 60s)
  ↓
4090: /api/unified endpoint
  ↓
Browser: master_dashboard.html (fetch every 5s)
```

---

## Changelog

- **2025-11-25:** Initial data contracts document
- **2025-11-25:** Added API plugin for remote SSH fetch
- **2025-11-25:** Added dashboard visualization cards
