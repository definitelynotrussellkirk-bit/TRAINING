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

## Training Flow Architecture

**Canonical Rule:** There is exactly ONE training executor. Everything else schedules or wraps.

### Layer Model

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3: INTERFACES (thin wrappers, no training logic)         │
│                                                                  │
│   training/cli.py     tavern/api/*     arena/hero_loop.py       │
│   (human CLI)         (HTTP API)       (campaign runner)         │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ▼                                     │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: ORCHESTRATION (decide what/when, never how)           │
│                                                                  │
│   core/training_daemon.py          core/training_queue.py       │
│   (long-running scheduler)         (priority queue)              │
│         │                                │                       │
│         └────────────────┬───────────────┘                       │
│                          ▼                                       │
│                     RunConfig                                    │
│                    (dataclass)                                   │
│                          │                                       │
├──────────────────────────┼───────────────────────────────────────┤
│  Layer 1: ENGINE (the boss - owns the training loop)            │
│                          │                                       │
│                          ▼                                       │
│               trainer/core/engine.py                             │
│                   TrainerEngine                                  │
│                          │                                       │
│         ┌────────────────┼─────────────────┐                     │
│         ▼                ▼                 ▼                     │
│    Model Load      Dataset Prep      HF Trainer                  │
│    (Flash Attn,    (Profile,         (callbacks,                 │
│     Qwen3VL,       tokenize,         optimizer,                  │
│     precision)     packing)          collator)                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Layer 1: TrainerEngine (THE BOSS)

**Location:** `trainer/core/engine.py`

**Responsibilities:**
- Load model + tokenizer (Flash Attention, Qwen3VL, precision)
- Prepare datasets (profile transforms, tokenization, packing)
- Create HF Trainer with callbacks, optimizer, collator
- Execute training loop
- Return structured `TrainingResult`

**API:**
```python
from trainer.core.engine import TrainerEngine, TrainingResult

@dataclass
class TrainingResult:
    success: bool
    global_step: int
    runtime_sec: float
    last_checkpoint_path: Optional[str]
    final_loss: float
    summary: Dict[str, Any]
    error_message: Optional[str] = None

class TrainerEngine:
    def __init__(self, status_writer: TrainingStatusWriter): ...

    def run_job(
        self,
        config: TrainerConfig,
        config_dict: Optional[Dict] = None,
        monitors: Optional[MonitorContext] = None,
        callbacks: Optional[List[TrainerCallback]] = None
    ) -> TrainingResult:
        """Execute complete training job. ONLY public method."""
```

**Rule:** All training logic lives here. If you're touching model APIs, you're in the engine.

### Layer 2: Orchestration (Schedule Only)

**Location:** `core/training_daemon.py`, `core/training_queue.py`

**Responsibilities:**
- Monitor inbox for new files
- Manage priority queues (high/normal/low)
- Handle pause/resume/stop signals
- Auto-generate data when queue empty
- Create daily snapshots
- Enforce checkpoint retention

**Rule:** Orchestration NEVER calls model APIs directly. It only:
1. Creates `TrainerConfig` from job metadata
2. Calls `TrainerEngine.run_job(config)`
3. Records results

**Pattern:**
```python
# In daemon - CORRECT
job = queue.get_next_job()
config = job_to_trainer_config(job)
engine = TrainerEngine(status_writer)
result = engine.run_job(config)
record_result(job.id, result)

# In daemon - WRONG (leaking training logic)
model = AutoModelForCausalLM.from_pretrained(...)  # NO!
trainer = Trainer(model=model, ...)  # NO!
```

### Layer 3: Interfaces (Thin Wrappers)

**Locations:**
- `training/cli.py` - Human CLI
- `tavern/api/*` - HTTP API for UI
- `arena/hero_loop.py` - Campaign-based runner
- `arena/trainers/*.py` - Trainer wrappers

**Responsibilities:**
- Parse user input (CLI args, HTTP requests)
- Translate to orchestration calls
- Format output for humans/APIs

**Rule:** Interfaces NEVER import training libraries. They only:
1. Parse input → `RunConfig` or job submission
2. Call Layer 2 (daemon/queue) or Layer 1 (engine)
3. Format output

**Pattern:**
```python
# In CLI - CORRECT
def cmd_train(args):
    config = config_from_args(args)
    engine = TrainerEngine(status_writer)
    result = engine.run_job(config)
    print_summary(result)

# In CLI - WRONG
from transformers import Trainer  # NO! Don't import HF here
```

### Migration Status

| Component | Current State | Target State |
|-----------|---------------|--------------|
| `trainer/core/engine.py` | ✅ Canonical engine | ✅ Done |
| `core/train.py` | ✅ Delegates to engine (legacy deprecated) | ✅ Done |
| `core/training_daemon.py` | ✅ Uses TrainerEngine directly | ✅ Done |
| `training/cli.py` | ✅ Spawns hero loops (clean interface) | ✅ Done |
| `arena/hero_loop.py` | ✅ Uses factory → engine trainers | ✅ Done |
| `arena/trainers/*.py` | ✅ Delegates to TrainerEngine | ✅ Done |

### Verification Checklist

You've "done it right" when:

1. ✅ Exactly one file implements the training loop: `trainer/core/engine.py`
2. ✅ `training/cli.py` does not import model/dataset libs directly (spawns hero loops)
3. ✅ `core/training_daemon.py` uses `TrainerEngine.run_job()` directly (no training inner loops)
4. ✅ Arena trainers (`arena/trainers/*.py`) delegate to TrainerEngine
5. ⬜ Swapping optimizers or models requires editing only `trainer/core/engine.py`

**Updated:** 2025-11-29 - All training paths now route through TrainerEngine

## Key Modules

### Training Core
| Module | Location | Responsibility |
|--------|----------|----------------|
| TrainerEngine | `trainer/core/engine.py` | **THE** training executor (Layer 1) |
| ConfigLoader | `trainer/config/loader.py` | JSON + CLI config merging |
| DataProfile | `trainer/profiles/base.py` | Data transformation interface |
| UltimateTrainer | `core/train.py` | Legacy wrapper → delegates to engine |
| training_daemon | `core/training_daemon.py` | Queue processing orchestrator (Layer 2) |

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
