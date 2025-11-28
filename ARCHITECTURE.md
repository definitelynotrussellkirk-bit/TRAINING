# System Architecture

**Last Updated:** 2025-11-28

## System Split

Two-GPU architecture with distinct responsibilities:

| GPU | Host ID | Role | Key Processes |
|-----|---------|------|---------------|
| **RTX 4090** | `4090` (trainer) | Training, evaluation, orchestration | training_daemon, model_comparison_engine, deployment_orchestrator |
| **RTX 3090** | `3090` (inference) | Inference only | FastAPI server (port 8765) |

**Note:** Concrete IPs, ports, and SSH users are defined in `config/hosts.json` and must not be hardcoded in code.

## Data Flow

```
                          4090 (Training)                           3090 (Inference)
┌────────────────────────────────────────────────────────────┐    ┌─────────────────┐
│                                                            │    │                 │
│  inbox/ ─► SpecValidator ─► DataValidator ─► queue/       │    │   /v1/chat/     │
│                 │                 │                        │    │   completions   │
│            (schema)          (content)                     │    │        │        │
│                                  │                         │    │        ▼        │
│                                  ▼                         │    │  Deployed Model │
│                         training_daemon                    │    │  (checkpoint-N) │
│                              │                             │    │                 │
│                              ▼                             │    └────────▲────────┘
│                    checkpoints/checkpoint-N                │             │
│                              │                             │             │
│                              ▼                             │     rsync   │
│                  model_comparison_engine                   │      (8s)   │
│                              │                             │             │
│                              ▼                             │             │
│                  deployment_orchestrator ─────────────────────────────────┘
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Deployment latency:** < 15 minutes from checkpoint creation to serving

## Validation System (Two-Layer)

Training data passes through two validation gates before training:

### Layer 1: SpecValidator (Schema Gate)
- **Location:** `core/validation/spec.py`
- **Purpose:** Deny-by-default schema enforcement
- **Checks:** Job must map to known spec (chat_sft_v1, syllo_v1, completion_v1)
- **Outcome:** Unknown schemas rejected before reaching queue

### Layer 2: DataValidator (Content Gate)
- **Location:** `core/validation/validator.py`
- **Purpose:** Content quality validation at three levels
- **Levels:**
  - `QUICK`: Basic format checks (< 1 sec)
  - `STANDARD`: Token stats, balance analysis (< 30 sec)
  - `DEEP`: Answer leakage detection, full quality audit (minutes)

```python
# Usage in daemon
from core.validation import SpecValidator, DataValidator, DATASET_SPECS

spec_validator = SpecValidator(DATASET_SPECS)
data_validator = DataValidator(tokenizer)

# On file arrival
spec = spec_validator.validate_job(job_config, metadata)  # Raises on unknown schema
issues = data_validator.validate_file(file_path, level='QUICK')  # Raises on bad data
```

## Data Lineage System

Tracks generator and validator provenance for all training data, enabling debugging of data quality issues.

### Components
| Module | Location | Purpose |
|--------|----------|---------|
| Generator Registry | `core/lineage.py` | Known generators + versions |
| FileLineage | `core/lineage.py` | Metadata dataclass for sidecar files |
| LineageTracker | `core/lineage_tracker.py` | Aggregates validation stats per generator/validator |

### Data Flow
```
Generator creates JSONL  →  Writes .meta.json sidecar  →  Daemon validates
                                                              ↓
Dashboard ← /api/lineage ← status/data_lineage.json ← LineageTracker records
```

### Sidecar Files
Each training file can have a `.meta.json` sidecar:
```
train_SYLLO_level3_100_20251126.jsonl
train_SYLLO_level3_100_20251126.jsonl.meta.json  ← lineage metadata
```

### Generator Versioning
```python
# In each generator file
GENERATOR_ID = "discrimination"
GENERATOR_VERSION = "1.0.0"  # Bump when logic changes
```

Registered generators: `discrimination`, `syllo_local`, `syllo_api`, `curriculum`, `manual`

### Validator Versioning
```python
# In each validator file
VALIDATOR_NAME = "data_validator"
VALIDATOR_VERSION = "1.0.0"  # Bump when logic changes
```

### Stats Aggregation
LineageTracker aggregates per-generator and per-validator:
- Total validations, passed, failed
- Fail rate percentage
- Top error reasons
- Worst offenders (generators/validators with >5% rejection)

### API Access
```bash
curl http://localhost:8081/api/lineage | jq .summary
```

## Key Modules

### Training Core
| Module | Location | Responsibility |
|--------|----------|----------------|
| TrainerEngine | `trainer/core/engine.py` | High-level training API |
| ConfigLoader | `trainer/config/loader.py` | JSON + CLI config merging |
| DataProfile | `trainer/profiles/base.py` | Data transformation interface |
| UltimateTrainer | `core/train.py` | Main training script |
| training_daemon | `core/training_daemon.py` | Queue processing orchestrator |

### Daemon Services (Extracted)
| Module | Location | Responsibility |
|--------|----------|----------------|
| PIDManager | `core/daemon/pid_manager.py` | Single-instance enforcement |
| FileWatcher | `core/daemon/file_watcher.py` | Inbox monitoring |
| SnapshotService | `core/daemon/snapshot_service.py` | Checkpoint snapshots |
| BackgroundWorker | `core/daemon/background_worker.py` | Non-blocking tasks |

### Monitoring API
| Module | Location | Responsibility |
|--------|----------|----------------|
| aggregator.py | `monitoring/api/aggregator.py` | Plugin-based data aggregation |
| skill_metrics.py | `monitoring/api/plugins/skill_metrics.py` | Baseline test results |
| model_comparison.py | `monitoring/api/plugins/model_comparison.py` | Checkpoint rankings |

### Data Lineage
| Module | Location | Responsibility |
|--------|----------|----------------|
| lineage.py | `core/lineage.py` | Generator registry + FileLineage dataclass |
| lineage_tracker.py | `core/lineage_tracker.py` | Validation stats aggregation |

## Configuration

Primary config: `config.json` (source of truth for all training parameters)

**Structure:**
```json
{
  "model_name": "qwen3_0.6b",
  "model_path": "models/Qwen3-0.6B",
  "profile": {"name": "emoji_think"},
  "hyperparams": {
    "fp_precision": "bf16",
    "max_length": 2048,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "learning_rate": 0.0004
  },
  "auto_generate": {
    "enabled": true,
    "host": "localhost",
    "port": 8080
  },
  "locked": {
    "base_model": "Qwen/Qwen3-0.6B",
    "model_architecture": "Qwen3ForCausalLM"
  }
}
```

See `trainer/config/schema.py` for full TrainerConfig dataclass.

## Ports Reference

| Port | Host | Service | Description |
|------|------|---------|-------------|
| 8080 | trainer | Live Monitor UI | Training metrics dashboard |
| 8080 | trainer | SYLLO Skill API | Local skill API for curriculum data |
| 8081 | trainer | Unified Monitoring API | Aggregated metrics via plugins |
| 8090 | trainer | Binary Skill API | Local skill API for binary math |
| 8765 | inference | Inference Server | Primary model inference |
| 8766 | inference | GPU Task Scheduler | Task queue coordinator |
| 8767 | trainer | VaultKeeper | Asset registry and ledger API |
| 8888 | trainer | Tavern UI | Main game interface |

## Module Contracts

All modules follow contract standards defined in `MODULE_CONTRACTS.md`:
- Type hints required on all public functions
- Docstrings with Args/Returns/Raises/Side Effects
- Data formats documented for I/O modules

## Profiles

Data transformation profiles in `trainer/profiles/`:
- `emoji_think.py`: Emoji thinking patterns + stop signals
- `regime3.py`: Symbolic reasoning format

Switch via config.json:
```json
{"profile": {"name": "emoji_think"}}
```

## Thinking Tokens / Chat Templates

Qwen3 models ship with a chat template that auto-injects `<think></think>` tags around all assistant content. This conflicts with custom thinking paradigms like `emoji_think`.

**Solution:** `core/chat_templates.py` detects and overrides Qwen3's template:

1. After tokenizer loads, detects `<think>` injection pattern in template
2. If `emoji_think` or `regime3` profile is active, replaces with clean ChatML template
3. Training data formatted WITHOUT `<think></think>` blocks
4. Logit penalties in profiles further suppress `<think>` token generation

**Result:** Training uses only the emoji thinking paradigm (💭...🔚) without competing systems.

See `core/chat_templates.py` for implementation and `trainer/profiles/emoji_think.py` for logit penalties.

## Data Generation (Curriculum-Based)

Auto-generation uses local singleSKILL APIs with curriculum-based difficulty:

```
                 DataManager
                      │
                      ▼
              CurriculumManager
              (active_skill, level)
                      │
         ┌───────────┴───────────┐
         ▼                       ▼
   SkillAPIClient          SkillAPIClient
   (SYLLO :8080)           (Binary :8090)
         │                       │
         ▼                       ▼
   singleSKILL/            singleSKILL/
   skill_syllo_variant/    skill_basic_math/
```

### Components
| Module | Location | Responsibility |
|--------|----------|----------------|
| DataManager | `data_manager/manager.py` | Orchestrates generation + quality + queueing |
| CurriculumManager | `data_manager/curriculum_manager.py` | Tracks skill levels, progression |
| SkillAPIClient | `data_manager/skill_api_client.py` | HTTP client for skill APIs |

### Curriculum Levels
- **SYLLO:** 5 levels (4-8 word puzzles)
- **Binary:** 7 levels (magnitude 1-10 to 10K-100K)
- **Progression:** 80% accuracy over 3 evaluations

### State Files
- `data_manager/curriculum_state.json` - Current levels, accuracy history
- `config.json` → `auto_generate` section - Enable/disable, count, cooldown

### Usage
```bash
# Check status
python3 data_manager/manager.py status

# Generate manually (100 examples at current level)
python3 data_manager/manager.py generate --force --count 100

# Check curriculum progress
python3 data_manager/curriculum_manager.py status
```

## Host Configuration

All host-specific settings (IPs, ports, SSH users, paths) are in `config/hosts.json`:

```json
{
  "hosts": {
    "4090": {
      "name": "Training Server",
      "host": "<trainer-ip>",
      "role": "trainer",
      "services": { ... }
    },
    "3090": {
      "name": "Inference Server",
      "host": "<inference-ip>",
      "role": "inference",
      "services": { ... }
    }
  }
}
```

Copy `config/hosts.example.json` to `config/hosts.json` and customize for your setup.

## Remote Services (ServiceClient)

All HTTP communication with remote services goes through a unified abstraction in `core/services.py`.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Domain Clients                               │
│  PredictionClient  │  VaultClient  │  TaskClient  │  OracleClient│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ServiceClient                               │
│  - Retry with exponential backoff                                │
│  - Configurable timeouts                                         │
│  - Standard exception hierarchy                                  │
│  - API key authentication                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      core.hosts                                  │
│  get_service_url("inference") → http://inference.local:8765     │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from core.services import get_service_client, ServiceError, ServiceUnavailable

# Get a client for any registered service
client = get_service_client("inference")

# Make requests with automatic retry/timeout handling
try:
    result = client.post_json("/v1/chat/completions", json=payload)
except ServiceUnavailable:
    # Service down - degrade gracefully
    pass
except ServiceHttpError as e:
    # HTTP 4xx/5xx
    logger.error(f"HTTP {e.status}: {e.body}")
```

### Exception Hierarchy

| Exception | When Raised | Retried? |
|-----------|-------------|----------|
| `ServiceUnavailable` | Connection error, timeout, or retries exhausted | Yes (until limit) |
| `ServiceHttpError` | HTTP 4xx/5xx response | 5xx only |
| `ServiceAuthError` | HTTP 401/403 | No |
| `ServiceDecodeError` | Invalid JSON response | No |

### Domain Clients

High-level clients wrap `ServiceClient` with domain-specific methods:

| Client | Service | Location |
|--------|---------|----------|
| `PredictionClient` | inference | `monitoring/prediction_client.py` |
| `VaultKeeperClient` | vault | `vault/client.py` |
| `TaskClient` | scheduler | `monitoring/task_client.py` |
| `OracleClient` | inference | `watchtower/oracle_client.py` |

### Configuration

Service URLs are resolved from `config/hosts.json` via `core.hosts`.
Per-service overrides via environment variables:

```bash
INFERENCE_TIMEOUT_S=120
INFERENCE_MAX_RETRIES=5
INFERENCE_API_KEY=secret123
```
