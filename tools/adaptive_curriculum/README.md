# Adaptive Curriculum Learning System

Automatically adjusts training data difficulty to keep your model at ~80% accuracy through adaptive curriculum learning.

## Overview

This system implements curriculum learning by:
1. **Generating** training data at different difficulty levels
2. **Evaluating** model performance on each difficulty level
3. **Adapting** difficulty to maintain target accuracy (~80%)
4. **Tracking** performance metrics over time

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Orchestrator                       │
│  ┌──────────────────────────────────────────────┐  │
│  │  1. Check queue depth                        │  │
│  │  2. For each generator:                      │  │
│  │     - Get stats (rolling accuracy)           │  │
│  │     - Choose difficulty (controller)         │  │
│  │     - Generate batch                         │  │
│  │  3. Periodically evaluate                    │  │
│  │  4. Update stats & adjust difficulty         │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         │                │               │
         ▼                ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │Generator │    │  Stats   │    │Controller│
   │ Registry │    │ Manager  │    │          │
   └──────────┘    └──────────┘    └──────────┘
```

### Components

1. **Stats Manager** (`stats.py`)
   - Tracks rolling window accuracy per (generator, difficulty)
   - Maintains evaluation history
   - Persists to disk

2. **Difficulty Controller** (`controller.py`)
   - Adjusts difficulty to maintain ~80% accuracy
   - If accuracy > 85%: increase difficulty
   - If accuracy < 75%: decrease difficulty
   - Supports difficulty mixtures (advanced)

3. **Evaluator** (`evaluator.py`)
   - Calls inference API to test model
   - Computes accuracy on eval sets
   - Supports custom judging functions

4. **Generator Registry** (`registry.py`)
   - Manages multiple data generators
   - Maps difficulty levels → toggle values
   - Calls generator APIs

5. **Orchestrator** (`orchestrator.py`)
   - Main control loop
   - Coordinates all components
   - Manages generation/evaluation cycles

## Quick Start

### 1. Initialize Generator Config

```bash
cd /path/to/training
python3 -m tools.adaptive_curriculum.cli init-config --output generators.json
```

This creates `generators.json` with default SYLLO generator config.

### 2. Edit Generator Config (Optional)

```json
{
  "generators": [
    {
      "generator_id": "syllo",
      "api_url": "http://127.0.0.1",
      "api_port": 8765,
      "difficulty_levels": {
        "0": {
          "difficulty": "EASY",
          "min_words": 4,
          "max_words": 5,
          "red_herring_count": 0
        },
        "1": {
          "difficulty": "MEDIUM",
          "min_words": 5,
          "max_words": 6,
          "red_herring_count": 1
        },
        "2": {
          "difficulty": "HARD",
          "min_words": 6,
          "max_words": 8,
          "red_herring_count": 3
        }
      },
      "default_count": 1000,
      "default_priority": "normal"
    }
  ]
}
```

### 3. Start Orchestrator

```bash
# Using default SYLLO generator
python3 -m tools.adaptive_curriculum.cli start

# Or with custom config
python3 -m tools.adaptive_curriculum.cli start --generators generators.json

# Customize parameters
python3 -m tools.adaptive_curriculum.cli start \
  --batch-size 500 \
  --queue-threshold 1 \
  --check-interval 180 \
  --target-accuracy 0.8 \
  --accuracy-band 0.05
```

### 4. Monitor Progress

```bash
# Check status
python3 -m tools.adaptive_curriculum.cli status

# JSON output
python3 -m tools.adaptive_curriculum.cli status --json
```

## Usage Examples

### Generate Single Batch

```bash
python3 -m tools.adaptive_curriculum.cli generate \
  --generator syllo \
  --count 1000
```

### Run Evaluations

```bash
python3 -m tools.adaptive_curriculum.cli evaluate

# JSON output
python3 -m tools.adaptive_curriculum.cli evaluate --json
```

### Background Daemon

```bash
cd /path/to/training
nohup python3 -m tools.adaptive_curriculum.cli start \
  --generators generators.json \
  > logs/curriculum.log 2>&1 &

echo $! > control/curriculum.pid
```

Stop daemon:
```bash
kill $(cat control/curriculum.pid)
```

## Configuration

### Orchestrator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 1000 | Examples per generation batch |
| `--queue-threshold` | 2 | Generate when queue depth < threshold |
| `--check-interval` | 300 | Seconds between queue checks |
| `--eval-interval` | 500 | Evaluate every N training steps |
| `--eval-sample-size` | 100 | Examples per eval set |
| `--target-accuracy` | 0.8 | Target accuracy (80%) |
| `--accuracy-band` | 0.05 | Tolerance band (±5%) |
| `--window-size` | 200 | Rolling window for accuracy |
| `--min-samples` | 20 | Min samples before adjusting |

### Generator Config

Each generator has:
- `generator_id`: Unique identifier
- `api_url` / `api_port`: Generator API endpoint
- `difficulty_levels`: Map of level → toggles
- `default_count`: Examples per batch
- `default_priority`: Queue priority

## How It Works

### 1. Generation Phase

```python
# For each generator:
stats = stats_manager.get_stats("syllo")
difficulty = controller.choose_difficulty("syllo", stats)

# difficulty=0 if accuracy=None (bootstrap)
# difficulty++ if accuracy > 85%
# difficulty-- if accuracy < 75%

examples = registry.generate("syllo", difficulty, count=1000)
# Writes to inbox/syllo_diff0_20251122_120000.jsonl
```

### 2. Training Phase

Training daemon picks up file from inbox and trains as usual.

### 3. Evaluation Phase

```python
# Every N generations:
for eval_set in eval_sets:
    batch = load_eval_batch(eval_set)
    results = evaluator.evaluate_batch(batch)

    stats_manager.update(
        generator_id="syllo",
        difficulty=0,
        correct_count=80,
        total_count=100
    )
    # Accuracy = 80%
```

### 4. Adaptation Phase

```python
# Next generation uses updated stats:
stats.accuracy(0) = 0.80  # Still in range [0.75, 0.85]
difficulty = 0  # Stay at level 0

stats.accuracy(0) = 0.90  # Too easy!
difficulty = 1  # Bump to medium
```

## Metadata Format

All generated examples include metadata:

```json
{
  "messages": [...],
  "metadata": {
    "generator_id": "syllo",
    "difficulty_level": 0,
    "toggles": {
      "difficulty": "EASY",
      "min_words": 4,
      "max_words": 5,
      "red_herring_count": 0
    },
    "dataset": "syllo_api_autogen",
    "puzzle_id": "syllo_12345"
  }
}
```

This enables:
- Filtering by difficulty
- Toggle analysis
- Performance debugging
- Curriculum tracking

## Adding New Generators

### 1. Create Generator API

Your API must have `/generate` endpoint that accepts:
```json
{
  "count": 1000,
  "difficulty": "EASY",
  ...toggles...
}
```

And returns:
```json
{
  "examples": [
    {
      "messages": [...],
      "metadata": {...}
    }
  ]
}
```

### 2. Define Difficulty Levels

```python
from tools.adaptive_curriculum.registry import GeneratorConfig

my_generator = GeneratorConfig(
    generator_id="my_gen",
    api_url="http://localhost",
    api_port=9000,
    difficulty_levels={
        0: {"difficulty": "easy", "param1": 10},
        1: {"difficulty": "medium", "param1": 20},
        2: {"difficulty": "hard", "param1": 30}
    }
)
```

### 3. Register

```python
orchestrator.register_generator(my_generator)
```

Or add to `generators.json`:
```json
{
  "generators": [
    {
      "generator_id": "my_gen",
      "api_url": "http://localhost",
      "api_port": 9000,
      "difficulty_levels": {
        "0": {"difficulty": "easy"},
        "1": {"difficulty": "medium"},
        "2": {"difficulty": "hard"}
      }
    }
  ]
}
```

## Advanced Features

### Custom Judging Functions

```python
from tools.adaptive_curriculum.evaluator import ModelEvaluator

def my_custom_judge(prediction: str, expected: str) -> bool:
    """Custom correctness check."""
    # Parse JSON
    try:
        pred = json.loads(prediction)
        exp = json.loads(expected)
        return pred["answer"] == exp["answer"]
    except:
        return False

evaluator = ModelEvaluator()
results = evaluator.evaluate_batch(batch, custom_judge=my_custom_judge)
```

### Difficulty Mixtures

```python
from tools.adaptive_curriculum.controller import MixedDifficultyController

controller = MixedDifficultyController()
mixture = controller.choose_mixture("syllo", stats)
# Returns: {0: 0.6, 1: 0.4} = 60% easy + 40% medium

counts = controller.apply_mixture(1000, mixture)
# Returns: {0: 600, 1: 400}
```

### Per-Toggle Analysis

```python
# Filter stats by toggle values
stats = stats_manager.get_stats("syllo")

for eval_result in stats.recent_evals():
    toggles = eval_result.toggles
    if toggles.get("red_herring_count", 0) > 0:
        print(f"Red herring accuracy: {eval_result.accuracy}")
```

## Troubleshooting

### No evaluations running

**Check:** Are eval sets being created?
```bash
ls -lh eval_sets/
```

**Fix:** Ensure generation is creating eval sets:
```python
eval_path = eval_builder.create_eval_set(...)
```

### Accuracy always None

**Cause:** Not enough samples yet (< min_samples)

**Fix:** Wait for more evaluations, or lower `--min-samples`

### Difficulty not changing

**Check:** Is accuracy outside tolerance band?
- If 75% < accuracy < 85%, difficulty stays same (by design)

**Fix:** Adjust `--accuracy-band` for more sensitivity

### Generator API errors

**Check:** Is generator API running?
```bash
curl http://127.0.0.1:8765/health
```

**Fix:** Start generator API:
```bash
cd /path/to/generator
python3 api_server.py --port 8765
```

## Files Created

```
/path/to/training/
├── inbox/                          # Generated batches
│   └── syllo_diff0_*.jsonl
├── eval_sets/                      # Evaluation sets
│   └── syllo_diff0_eval.jsonl
├── eval_results/                   # Evaluation results
│   └── eval_results_*.json
└── status/
    └── curriculum_stats.json       # Stats persistence
```

## Integration with Existing System

The adaptive curriculum system integrates seamlessly:

1. **Generates** to `inbox/` (training daemon picks up automatically)
2. **Uses** remote inference API (same as current setup)
3. **Respects** queue threshold (won't overwhelm training)
4. **Logs** metadata (compatible with existing JSONL format)

No changes needed to training system!

## Performance Tips

1. **Start with small batches** (--batch-size 500) for faster iteration
2. **Use lower eval intervals** during early training (--eval-interval 100)
3. **Increase window size** for more stable adjustments (--window-size 500)
4. **Run evaluations in parallel** (add threading in future)

## Future Enhancements

- [ ] Multi-armed bandit for optimal difficulty selection
- [ ] Per-skill tracking (not just per-generator)
- [ ] Automatic toggle discovery (learn what affects difficulty)
- [ ] Web UI for monitoring
- [ ] A/B testing different curriculum strategies
- [ ] Export curriculum as training schedule

## Credits

Implements curriculum learning ideas from:
- Bengio et al. (2009) "Curriculum Learning"
- Graves et al. (2017) "Automated Curriculum Learning"
- Your original plan: "keep them hovering around ~80% accuracy"
