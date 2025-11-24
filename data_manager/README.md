# Data Manager System

**Unified coordinator for data generation, quality testing, and queue management**

## Overview

The Data Manager acts as both a **data pipeline orchestrator** and **test suite** to ensure only high-quality data enters the training queue.

```
┌──────────────────────────────────────────────────────────────┐
│                      DATA MANAGER                            │
│                                                              │
│  ┌────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │  Generate  │───►│  Test       │───►│  Queue         │  │
│  │  (Remote)  │    │  (Quality)  │    │  (Training)    │  │
│  └────────────┘    └─────────────┘    └────────────────┘  │
│       │                   │                     │          │
│       ▼                   ▼                     ▼          │
│  RTX 3090           5 Test Suites      Priority Queue     │
│  192.168.x.x     - Format                              │
│                     - Length                              │
│                     - Diversity                           │
│                     - Balance                             │
│                     - Content Quality                     │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. **RemoteGPUClient** (`remote_client.py`)
- Communicates with RTX 3090 inference server (192.168.x.x:8765)
- Health checks and GPU monitoring
- Data generation requests
- Inference API

### 2. **QualityChecker** (`quality_checker.py`)
- 5 test suites:
  - **Format Check**: Validates conversation structure
  - **Length Check**: Ensures tokens within min/max bounds
  - **Diversity Check**: Measures response uniqueness (≥70%)
  - **Balance Check**: User/assistant ratio (~1:1)
  - **Content Quality**: Detects empty, malformed, or suspicious content
- Generates detailed quality reports
- Approval/rejection recommendations

### 3. **DataManager** (`manager.py`)
- Main orchestrator
- Workflow: Generate → Test → Queue
- Tracks statistics and approval rates
- Automatic cooldown management
- Saves rejected data for analysis

## Usage

### CLI Commands

**Check Status:**
```bash
python3 data_manager/manager.py status
```

**Generate Data:**
```bash
# Auto (uses config.json settings)
python3 data_manager/manager.py generate

# Force generation (ignore checks)
python3 data_manager/manager.py generate --force

# Custom count and seed
python3 data_manager/manager.py generate --count 50000 --seed 12345
```

**View Statistics:**
```bash
python3 data_manager/manager.py stats
```

### Programmatic Usage

```python
from data_manager import DataManager
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

# Create manager
manager = DataManager('/path/to/training', config)

# Check remote status
manager.check_remote_status()

# Generate, test, and queue
success = manager.generate_and_queue()

# Get stats
stats = manager.get_stats()
print(f"Approval rate: {stats['approval_rate']*100:.1f}%")
```

## Integration with Training Daemon

The Data Manager is integrated into `core/training_daemon.py`:

```python
# In training_daemon.py
from data_manager import DataManager

manager = DataManager(base_dir, config)

# Called every poll cycle when queue is empty
if queue_is_empty:
    manager.generate_and_queue()
```

## Configuration

In `config.json`:

```json
{
  "auto_generate": {
    "enabled": true,
    "host": "192.168.x.x",
    "port": 8765,
    "count": 100000,
    "priority": "normal",
    "threshold": 0,
    "seed": 93001,
    "cooldown_sec": 180,
    "payload": {}
  }
}
```

**Parameters:**
- `enabled`: Enable/disable auto-generation
- `host`: Remote GPU server IP
- `port`: API port
- `count`: Examples per batch
- `priority`: Queue priority (high/normal/low)
- `threshold`: Queue depth threshold (generate when below this)
- `seed`: Random seed for reproducibility
- `cooldown_sec`: Minimum seconds between generations
- `payload`: Additional parameters for generation API

## Quality Reports

Reports are saved to `data_manager/reports/`:

```json
{
  "total_checks": 5,
  "passed_checks": 5,
  "failed_checks": 0,
  "overall_pass": true,
  "recommendation": "✅ APPROVED: Data quality is excellent. Safe to queue for training.",
  "results": {
    "format": {
      "passed": true,
      "details": {
        "valid_count": 100,
        "pass_rate": 1.0
      }
    },
    "length": {...},
    "diversity": {...},
    "balance": {...},
    "content": {...}
  }
}
```

## Statistics Tracking

The manager tracks:
- Total examples generated
- Total examples approved/rejected
- Approval rate
- Average quality scores
- Queue status
- Remote GPU memory usage

## Directory Structure

```
data_manager/
├── __init__.py
├── manager.py              # Main orchestrator
├── remote_client.py        # RTX 3090 client
├── quality_checker.py      # Test suites
├── README.md               # This file
├── reports/                # Quality reports
├── rejected/               # Rejected data for analysis
├── tests/                  # Unit tests
├── generators/             # Custom data generators
└── evaluators/             # Custom evaluators
```

## Extending

### Add Custom Generator

```python
# data_manager/generators/my_generator.py
def generate_custom_data(count):
    # Your generation logic
    return data
```

### Add Custom Test

```python
# In quality_checker.py
def _check_my_metric(self, data):
    # Your test logic
    passed = True
    details = {...}
    return passed, details

# Add to check_all()
checks.append(("my_metric", self._check_my_metric))
```

## Workflow Example

```
1. Queue drops to 0
   ↓
2. DataManager.should_generate() → True
   ↓
3. RemoteGPUClient.generate_data(100000)
   ↓
4. QualityChecker.check_all(data)
   ├─ Format: ✅ PASS
   ├─ Length: ✅ PASS
   ├─ Diversity: ✅ PASS (78%)
   ├─ Balance: ✅ PASS (0.98:1)
   └─ Content: ✅ PASS
   ↓
5. Report: "✅ APPROVED"
   ↓
6. Queue data → queue/normal/syllo_autogen_20251122_130000_count100000.jsonl
   ↓
7. Training daemon picks up file
```

## Troubleshooting

**Remote GPU unreachable:**
```bash
# Check network
ping 192.168.x.x

# Check API
curl http://192.168.x.x:8765/health
```

**Quality tests failing:**
- Check `data_manager/reports/` for detailed reports
- Review rejected data in `data_manager/rejected/`
- Adjust thresholds in `QualityChecker` if needed

**No data generated:**
- Check cooldown timer
- Verify queue threshold
- Check remote GPU availability
