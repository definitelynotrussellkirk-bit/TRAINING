# Custom Loss Functions

## Fortune Teller Loss

Surprise-weighted cross-entropy that focuses gradients on uncertain predictions.

### Quick Start

```python
from trainer.losses import FortuneTellerLoss

# Create loss function
loss_fn = FortuneTellerLoss(
    surprise_metric="entropy",  # entropy, confidence, perplexity, margin
    min_surprise=0.1,
    max_surprise=10.0,
    normalize_batch=True,
    temperature=1.0,
)

# Use in training
loss, details = loss_fn(logits, labels, return_details=True)
```

### Enable in Config

```json
{
  "fortune_teller": {
    "enabled": true,
    "surprise_metric": "entropy",
    "min_surprise": 0.1,
    "max_surprise": 10.0,
    "normalize_batch": true,
    "temperature": 1.0
  }
}
```

### Test

```bash
python3 scripts/test_fortune_teller.py --test-metrics
```

See `docs/FORTUNE_TELLER.md` for full documentation.
