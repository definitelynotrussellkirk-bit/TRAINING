# THE FORTUNE TELLER

**Surprise-Weighted Training for Efficient Learning**

---

## 🔮 Overview

THE FORTUNE TELLER is a training enhancement that focuses gradient updates on tokens that **surprise** the model, rather than treating all tokens equally.

**Core Idea**: Tokens the model predicts confidently and correctly shouldn't receive the same gradient weight as uncertain predictions. By focusing on surprises, training becomes more efficient and naturally implements a form of curriculum learning.

---

## 🎯 The Problem

Standard cross-entropy loss treats all tokens equally:
- Model predicts "the" correctly with 99% confidence → Full gradient update
- Model struggles between "their" vs "there" with 50/50 confidence → Same gradient update

This is wasteful. The model has already mastered "the" - why spend gradient budget on it?

---

## 💡 The Solution

**Surprise-weighted loss**: Modulate gradient contributions by how "surprised" the model is:

```
weighted_loss = cross_entropy_loss × surprise_weight
```

Where `surprise_weight` is computed per-token based on:
1. **Entropy**: How uncertain is the distribution?
2. **Confidence**: How low is the max probability?
3. **Perplexity**: How unexpected is the correct token?
4. **Margin**: How close is the second-best option?

---

## 📊 Surprise Metrics

### 1. Entropy (Default)

```python
H = -Σ p(x) log p(x)
```

- **High entropy** = Model is uncertain = High surprise
- **Low entropy** = Model is confident = Low surprise
- **Range**: [0, log(vocab_size)]
- **Best for**: General-purpose surprise detection

### 2. Confidence

```python
surprise = 1 - max(p)
```

- **Low max prob** = Not confident = High surprise
- **High max prob** = Confident = Low surprise
- **Range**: [0, 1]
- **Best for**: Simple, interpretable weighting

### 3. Perplexity

```python
perplexity = exp(-log p(correct_token))
```

- **High perplexity** = Correct token was unexpected = High surprise
- **Low perplexity** = Correct token was expected = Low surprise
- **Range**: [1, ∞]
- **Best for**: Emphasizing truly unexpected correct tokens

### 4. Margin

```python
margin = p(correct) - p(second_best)
surprise = 1 - margin
```

- **Small margin** = Close competition = High surprise
- **Large margin** = Clear winner = Low surprise
- **Range**: [0, 1]
- **Best for**: Multi-way classification uncertainty

---

## 🎮 RPG Integration

**In-game lore**: The Fortune Teller is an oracle who predicts what will challenge DIO most, guiding training effort to where it's needed.

```
┌─────────────────────────────────────┐
│  🔮 THE FORTUNE TELLER              │
├─────────────────────────────────────┤
│  "I see... uncertainty ahead."      │
│                                     │
│  Current Reading:                   │
│  • Avg Surprise:    2.34            │
│  • Surprise Std:    1.87            │
│  • Training Focus:  HIGH            │
│                                     │
│  Prediction: DIO will struggle      │
│  with syllogistic reasoning at      │
│  Level 15. Recommend extra XP.      │
└─────────────────────────────────────┘
```

---

## ⚙️ Configuration

### Enable in `config.json`

```json
{
  "fortune_teller": {
    "enabled": true,
    "surprise_metric": "entropy",
    "min_surprise": 0.1,
    "max_surprise": 10.0,
    "normalize_batch": true,
    "temperature": 1.0,
    "save_history": true,
    "history_path": null
  }
}
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable Fortune Teller loss |
| `surprise_metric` | `"entropy"` | Which metric to use (entropy, confidence, perplexity, margin) |
| `min_surprise` | `0.1` | Minimum weight (prevents vanishing gradients) |
| `max_surprise` | `10.0` | Maximum weight (prevents explosion) |
| `normalize_batch` | `true` | Normalize surprises within each batch |
| `temperature` | `1.0` | Temperature for scaling (higher = more uniform weights) |
| `save_history` | `true` | Save surprise metrics to JSON |
| `history_path` | `null` | Path to save history (default: output_dir/fortune_teller_history.json) |

---

## 🧪 Usage Examples

### 1. Basic Usage

```python
from trainer.losses import FortuneTellerLoss

loss_fn = FortuneTellerLoss(
    surprise_metric="entropy",
    min_surprise=0.1,
    normalize_batch=True,
)

# During training
loss, details = loss_fn(logits, labels, return_details=True)
print(f"Loss: {loss.item():.4f}")
print(f"Avg surprise: {details['avg_surprise']:.4f}")
```

### 2. With TrainerEngine

```python
from trainer.core import TrainerEngine
from trainer.config import TrainerConfig

# Create config with Fortune Teller enabled
config = TrainerConfig(...)
config.fortune_teller.enabled = True
config.fortune_teller.surprise_metric = "entropy"

# Run training
engine = TrainerEngine()
result = engine.run_job(config)

# History saved automatically to output_dir/fortune_teller_history.json
```

### 3. Tracking and Analysis

```python
from trainer.losses import FortuneTellerTracker

tracker = FortuneTellerTracker()

# During training
_, details = loss_fn(logits, labels, return_details=True)
tracker.update(step, details)

# Get statistics
stats = tracker.get_stats(window=100)
print(f"Recent avg surprise: {stats['avg_surprise']:.3f}")

# Save to disk
tracker.save("fortune_teller_history.json")
```

### 4. Visualization

```bash
# Test metrics on synthetic data
python3 scripts/test_fortune_teller.py --test-metrics

# Visualize training history
python3 scripts/test_fortune_teller.py --visualize results/fortune_teller_history.json
```

---

## 🎯 Expected Behavior

### During Training

**Early stages** (high surprise everywhere):
- All tokens are surprising
- Weights are relatively uniform
- Falls back to near-standard training
- **This is expected and good!**

**Mid stages** (differentiation):
- Easy patterns (grammar, common words) → Low surprise → Low weight
- Hard patterns (reasoning, rare words) → High surprise → High weight
- **Automatic curriculum emerges!**

**Late stages** (mastery):
- Most tokens have low surprise
- Gradient focus on remaining difficult cases
- **Efficient fine-tuning**

### Surprise Evolution

Typical surprise curve over training:

```
Surprise
   ^
   │     ╱‾‾‾╲
 3 │    ╱      ╲___
   │   ╱           ╲___
 2 │  ╱                ╲___
   │ ╱                     ╲___
 1 │╱                           ╲___
   └────────────────────────────────> Steps
   0    5k    10k   15k   20k   25k
```

1. **Initial rise**: Model learns basic patterns, surprise increases as it becomes aware of what it doesn't know
2. **Plateau**: Steady learning on core curriculum
3. **Decline**: As patterns are mastered, surprise decreases
4. **Long tail**: Remaining surprise focused on hardest cases

---

## ⚠️ Edge Cases & Mitigations

### 1. Overconfident Wrong Predictions

**Problem**: Model confidently predicts wrong token → Low surprise but high loss

**Mitigation**: Surprise is multiplied by loss, so high loss still gets gradient signal

**Math**:
```python
weighted_loss = cross_entropy_loss × surprise_weight
```
If CE loss is high (wrong prediction), weighted loss is still high regardless of surprise.

### 2. Vanishing Gradients (All Low Surprise)

**Problem**: Everything becomes "easy", surprise drops to zero, no learning

**Mitigation**: `min_surprise` parameter ensures minimum weight

**Config**:
```json
{"min_surprise": 0.1}  // At least 10% of standard gradient
```

### 3. Exploding Gradients (All High Surprise)

**Problem**: Everything is surprising, weights blow up

**Mitigation**: `max_surprise` clipping and batch normalization

**Config**:
```json
{
  "max_surprise": 10.0,
  "normalize_batch": true
}
```

### 4. Batch-Level Variance

**Problem**: Some batches have all easy examples, some all hard → inconsistent gradients

**Mitigation**: Batch normalization standardizes surprise distribution per batch

**Effect**: Each batch gets a balanced distribution of weights, regardless of absolute difficulty

### 5. Temperature Scaling

**Problem**: Need to control how sharply surprise affects weights

**Solution**: Temperature parameter smooths or sharpens the distribution

```python
surprise = surprise / temperature

# temperature = 0.5 → Sharper focus on high-surprise tokens
# temperature = 1.0 → Standard scaling (default)
# temperature = 2.0 → More uniform, gentler focusing
```

---

## 📈 Predicted Effects

### Positive

1. **Efficient Learning**: Don't waste gradients on mastered patterns
2. **Automatic Curriculum**: Naturally focuses on progressively harder content
3. **Reduced Forgetting**: Confident correct predictions have low gradient → less likely to be unlearned
4. **Better Generalization**: Focus on uncertain/novel patterns improves robustness
5. **Faster Convergence**: Gradient budget spent where it matters

### Potential Issues

1. **Confidence Calibration**: If model becomes overconfident incorrectly, surprise drops but performance doesn't improve
2. **Metric Sensitivity**: Different surprise metrics may behave differently for your data
3. **Hyperparameter Tuning**: min/max surprise, temperature need tuning
4. **Computational Cost**: Extra forward pass operations (entropy, softmax, etc.) per token

---

## 🧪 Experiments & Validation

### Recommended Experiments

1. **Baseline Comparison**:
   - Train same model with/without Fortune Teller
   - Compare: final loss, convergence speed, eval metrics

2. **Metric Ablation**:
   - Try all 4 surprise metrics (entropy, confidence, perplexity, margin)
   - Find which works best for your domain

3. **Hyperparameter Sweep**:
   - `min_surprise`: [0.01, 0.1, 0.3]
   - `temperature`: [0.5, 1.0, 2.0]
   - `normalize_batch`: [true, false]

4. **Curriculum Analysis**:
   - Track which tokens have high surprise over time
   - Verify automatic progression from easy → hard

5. **Confidence Calibration**:
   - Measure model calibration (are probabilities accurate?)
   - Compare standard vs Fortune Teller calibration

### Metrics to Track

- **Training**: Loss, surprise (mean, std, max, min), gradient norms
- **Evaluation**: Accuracy, perplexity, calibration error
- **Efficiency**: Steps to convergence, total FLOPs
- **Curriculum**: Surprise distribution evolution, easy vs hard token accuracy

---

## 🗂️ File Structure

```
trainer/
├── losses/
│   ├── __init__.py                      # Export FortuneTellerLoss
│   └── fortune_teller.py                # Core implementation
├── core/
│   ├── engine.py                        # Integration with TrainerEngine
│   └── fortune_teller_trainer.py        # Custom Trainer class
└── config/
    └── schema.py                        # FortuneTellerConfig dataclass

scripts/
└── test_fortune_teller.py               # Testing and visualization

docs/
└── FORTUNE_TELLER.md                    # This file
```

---

## 🚀 Quick Start

### 1. Test the Implementation

```bash
# Test all surprise metrics
python3 scripts/test_fortune_teller.py --test-metrics
```

### 2. Enable for Training

Edit `config.json`:

```json
{
  "fortune_teller": {
    "enabled": true,
    "surprise_metric": "entropy"
  }
}
```

### 3. Run Training

```bash
USE_ENGINE=1 python3 core/train.py --dataset data/train.jsonl --yes
```

### 4. Analyze Results

```bash
# Visualize surprise over training
python3 scripts/test_fortune_teller.py --visualize models/current_model/fortune_teller_history.json
```

---

## 📚 Theory & Background

### Relation to Curriculum Learning

Fortune Teller implements **automatic curriculum learning**:
- Standard curriculum: Manually design easy → hard progression
- Fortune Teller: Model discovers its own curriculum via surprise

As training progresses:
- Initially mastered patterns become "easy" (low surprise)
- Remaining difficult patterns stay "hard" (high surprise)
- Gradient budget automatically shifts to hard cases

### Relation to Importance Sampling

Similar to importance sampling in RL:
- Sample high-reward experiences more often
- Fortune Teller: Weight high-surprise tokens more heavily

### Relation to Active Learning

Active learning: Query labels for most uncertain examples
Fortune Teller: Weight gradients by uncertainty (surprise)

Both focus compute on informative data points.

---

## 🎓 Future Directions

### Token-Level Primitives

Track surprise per primitive pattern:
- High surprise on syllogisms → Need more L15-L20 training
- Low surprise on binary arithmetic → Can reduce BIN training

### Dynamic Curriculum Adjustment

Use surprise to auto-adjust difficulty:
```python
if avg_surprise < 0.5:
    level_up()  # Too easy, increase difficulty
elif avg_surprise > 3.0:
    level_down()  # Too hard, decrease difficulty
```

### Meta-Learning

Learn the surprise metric itself:
- Train a small network to predict optimal weights
- Use surprise history to improve weighting strategy

### Multi-Task Weighting

Different skills have different surprise profiles:
- Weight tasks by their current surprise
- Focus on skills that need work

---

## 📞 Troubleshooting

### Training Loss Doesn't Decrease

**Check**:
- Is `min_surprise` too high? (Try 0.01)
- Is `normalize_batch` enabled? (Try disabling)
- Are all tokens unsurprising? (Check surprise history)

### Gradients Explode

**Check**:
- Is `max_surprise` set? (Try 10.0)
- Is `normalize_batch` enabled? (Should be)
- Are there NaN values in logits?

### Surprise Doesn't Decrease Over Time

**Check**:
- Is the model learning? (Check standard loss)
- Is the metric appropriate? (Try different surprise_metric)
- Is data too diverse/hard? (Expected for complex curricula)

### Fortune Teller Performs Worse Than Standard

**Possible causes**:
1. Hyperparameters not tuned (try different min/max/temperature)
2. Metric not suited to task (try different surprise_metric)
3. Data already well-curated (Fortune Teller helps most with mixed difficulty)
4. Insufficient training time (benefits compound over time)

---

## 🎯 Summary

**THE FORTUNE TELLER** is a training enhancement that:
- ✅ Focuses gradients where needed (high surprise)
- ✅ Reduces wasted updates (low surprise on mastered patterns)
- ✅ Implements automatic curriculum (easy → hard progression)
- ✅ Integrates seamlessly with TrainerEngine
- ✅ Tracks metrics for analysis and visualization

**When to use**:
- Training on mixed-difficulty data
- Want automatic curriculum learning
- Need efficient use of compute budget
- Researching uncertainty-weighted learning

**When NOT to use**:
- Data is already well-curated (single difficulty level)
- Need exact reproduction of baseline (different gradient trajectory)
- Training time is extremely limited (adds small overhead)

---

**Built with PyTorch • Compatible with HuggingFace Transformers • MIT Licensed**
