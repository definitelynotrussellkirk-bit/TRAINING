# STORAGE + DEVICE REGISTRY IMPLEMENTATION PLAN

**Created:** 2025-11-28
**Status:** Design Complete - Ready for Implementation

---

## EXECUTIVE SUMMARY

This plan unifies two complementary systems:

1. **Device Registry** - "What can this machine do?" (roles + capabilities)
2. **Storage System** - "Where do things live?" (zones + handles + path resolution)

Together, they enable:
- Job dispatch to appropriate workers (evals to eval_workers, data gen to data_forge nodes)
- Storage-aware decisions (keep HOT on NVMe, archive to COLD on NAS)
- Multi-host coordination without hardcoded paths

---

## CURRENT STATE ANALYSIS

### Existing Infrastructure

| System | File | Purpose | Status |
|--------|------|---------|--------|
| **Host Registry** | `core/hosts.py` | Services, SSH, networking | ✅ Works |
| **Storage Registry** | `vault/storage_registry.py` | Strongholds, profiles, sync | ✅ Works |
| **Zone Registry** | `vault/zones.py` | Asset federation, transfers | ✅ Works |

### What's Missing

| Concept | Gap |
|---------|-----|
| **Device Roles** | Hosts have `role` (trainer/inference) but no multi-role support (eval_worker + data_forge) |
| **Storage Temperatures** | Strongholds exist but no HOT/WARM/COLD zone abstraction |
| **StorageHandle** | No unified way to say "get me checkpoint-12345" without knowing which host |
| **Job Dispatch** | No mechanism to find "all devices with eval_worker role" |
| **GPU/CPU Capabilities** | Hosts don't declare hardware specs |

---

## DESIGN

### Part 1: Device Registry

A device is a machine in your lab with declared **roles** and **capabilities**.

#### 1.1 Device Roles (DeviceRole enum)

```python
class DeviceRole(str, Enum):
    # Compute roles
    TRAINER = "trainer"           # GPU training (4090)
    INFERENCE = "inference"       # Model serving (3090)
    EVAL_WORKER = "eval_worker"   # Runs skill evals (passives)
    DATA_FORGE = "data_forge"     # Generates/filters training data
    VAULT_WORKER = "vault_worker" # Archive/retention operations
    ANALYTICS = "analytics"       # Dashboards, metrics aggregation

    # Storage roles
    STORAGE_HOT = "storage_hot"   # Fast local NVMe
    STORAGE_WARM = "storage_warm" # NAS primary
    STORAGE_COLD = "storage_cold" # Archive NAS

    # Control roles
    CONTROL_PLANE = "control_plane"  # Orchestration services
```

#### 1.2 Device Config Schema (`config/devices.json`)

```json
{
  "schema_version": 1,
  "devices": {
    "trainer4090": {
      "hostname": "192.168.x.x",
      "roles": ["trainer", "eval_worker", "storage_hot", "control_plane"],
      "gpus": [
        {"name": "RTX 4090", "count": 1, "vram_gb": 24}
      ],
      "cpu": {"cores": 16, "threads": 32},
      "memory_gb": 64,
      "storage_zones": ["hot"],
      "network": {"speed_gbps": 10, "tags": ["lan_core"]}
    },
    "inference3090": {
      "hostname": "192.168.x.x",
      "roles": ["inference", "eval_worker", "storage_hot"],
      "gpus": [
        {"name": "RTX 3090", "count": 1, "vram_gb": 24}
      ],
      "cpu": {"cores": 8, "threads": 16},
      "memory_gb": 32,
      "storage_zones": ["hot"],
      "network": {"speed_gbps": 10, "tags": ["lan_core"]}
    },
    "macmini_eval_1": {
      "hostname": "macmini-eval-1.local",
      "roles": ["eval_worker", "data_forge", "vault_worker", "analytics"],
      "gpus": [],
      "cpu": {"cores": 8, "threads": 8},
      "memory_gb": 16,
      "storage_zones": [],
      "network": {"speed_gbps": 1, "tags": ["lan_edge"]}
    },
    "synology_data": {
      "hostname": "192.168.x.x",
      "roles": ["storage_warm"],
      "gpus": [],
      "cpu": {"cores": 4, "threads": 8},
      "memory_gb": 8,
      "storage_zones": ["warm"],
      "network": {"speed_gbps": 10, "tags": ["nas"]}
    },
    "synology_archive": {
      "hostname": "192.168.x.x",
      "roles": ["storage_cold"],
      "gpus": [],
      "cpu": {"cores": 4, "threads": 8},
      "memory_gb": 8,
      "storage_zones": ["cold"],
      "network": {"speed_gbps": 1, "tags": ["nas_archive"]}
    }
  }
}
```

#### 1.3 DeviceInfo Dataclass

```python
@dataclass
class GPUInfo:
    name: str
    count: int
    vram_gb: float

@dataclass
class CPUInfo:
    cores: int
    threads: int

@dataclass
class NetworkInfo:
    speed_gbps: float
    tags: List[str]

@dataclass
class DeviceInfo:
    device_id: str
    hostname: str
    roles: List[DeviceRole]
    gpus: List[GPUInfo]
    cpu: CPUInfo
    memory_gb: int
    storage_zones: List[str]  # ["hot", "warm", "cold"]
    network: NetworkInfo

    def has_role(self, role: DeviceRole) -> bool:
        return role in self.roles

    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    def total_vram(self) -> float:
        return sum(g.vram_gb * g.count for g in self.gpus)
```

#### 1.4 DeviceRegistry Class

```python
class DeviceRegistry:
    def __init__(self, config_path: Path = None):
        # Load from config/devices.json
        pass

    def get(self, device_id: str) -> DeviceInfo: ...
    def all_devices(self) -> List[DeviceInfo]: ...
    def devices_with_role(self, role: DeviceRole) -> List[DeviceInfo]: ...
    def devices_with_storage_zone(self, zone: str) -> List[DeviceInfo]: ...
    def devices_with_gpu(self) -> List[DeviceInfo]: ...
```

---

### Part 2: Storage System

Storage is organized into **zones** (temperature) with **handles** (logical references).

#### 2.1 Storage Zones (StorageZone enum)

```python
class StorageZone(str, Enum):
    HOT = "hot"    # Local NVMe (fast, limited, ephemeral)
    WARM = "warm"  # Primary NAS (networked, durable)
    COLD = "cold"  # Archive NAS (cheap, slow, long-term)
```

#### 2.2 Storage Kinds (StorageKind enum)

```python
class StorageKind(str, Enum):
    BASE_MODEL = "base_model"       # Pretrained models (Qwen3-0.6B)
    CURRENT_MODEL = "current_model" # Active training directory
    CHECKPOINT = "checkpoint"       # HF checkpoint-XXXX directories
    SNAPSHOT = "snapshot"           # Promoted/blessed checkpoints
    DATASET = "dataset"             # Training/validation data
    BENCHMARK = "benchmark"         # Eval benchmarks
    QUEUE = "queue"                 # Training queue files
    LOG = "log"                     # Logs
    META = "meta"                   # Guild states, configs
```

#### 2.3 StorageHandle (Logical Reference)

```python
@dataclass(frozen=True)
class StorageHandle:
    kind: StorageKind    # What type of thing
    key: str             # Unique identifier (e.g., "checkpoint-182000")
    zone: StorageZone    # Where it lives

    @property
    def handle_id(self) -> str:
        """Unique string identifier."""
        return f"{self.kind.value}:{self.key}@{self.zone.value}"
```

This is the **lingua franca** - code talks in handles, not paths:

```python
# OLD: Hardcoded path
path = "/path/to/training/models/current_model/checkpoint-182000"

# NEW: Ask for handle, get resolved path
handle = StorageHandle(
    kind=StorageKind.CHECKPOINT,
    key="checkpoint-182000",
    zone=StorageZone.HOT
)
path = storage_registry.resolve(handle)  # Returns actual filesystem path
```

#### 2.4 Zone Path Resolution Config

Extend `config/storage_registry.json` (or create new `config/storage_zones.json`):

```json
{
  "schema_version": 2,
  "zones": {
    "hot": {
      "devices": ["trainer4090", "inference3090"],
      "roots": {
        "trainer4090": "/path/to/training",
        "inference3090": "/home/user/llm"
      }
    },
    "warm": {
      "devices": ["synology_data"],
      "roots": {
        "synology_data": "/volume1/data/llm_training"
      }
    },
    "cold": {
      "devices": ["synology_archive"],
      "roots": {
        "synology_archive": "/volume1/archive/llm_training"
      }
    }
  },
  "kind_patterns": {
    "base_model": {
      "default_zone": "warm",
      "subdir": "models/base/{key}"
    },
    "current_model": {
      "default_zone": "hot",
      "subdir": "models/current_model"
    },
    "checkpoint": {
      "default_zone": "hot",
      "subdir": "models/current_model/{key}"
    },
    "snapshot": {
      "default_zone": "warm",
      "subdir": "snapshots/{key}"
    },
    "dataset": {
      "default_zone": "warm",
      "subdir": "data/{key}"
    },
    "benchmark": {
      "default_zone": "warm",
      "subdir": "data/validation/{key}"
    },
    "queue": {
      "default_zone": "hot",
      "subdir": "queue/{key}"
    },
    "log": {
      "default_zone": "hot",
      "subdir": "logs/{key}"
    },
    "meta": {
      "default_zone": "warm",
      "subdir": "status/{key}"
    }
  }
}
```

#### 2.5 StorageResolver Class

```python
class StorageResolver:
    """Resolves StorageHandles to filesystem paths."""

    def __init__(self, zones_config: Path, device_id: str):
        self.config = json.loads(zones_config.read_text())
        self.device_id = device_id

    def resolve(self, handle: StorageHandle) -> Path:
        """Resolve handle to local path."""
        zone_cfg = self.config["zones"][handle.zone.value]

        # Get root for this device in this zone
        root = zone_cfg["roots"].get(self.device_id)
        if not root:
            raise ValueError(f"Device {self.device_id} has no root in zone {handle.zone}")

        # Get pattern for this kind
        kind_cfg = self.config["kind_patterns"][handle.kind.value]
        subdir = kind_cfg["subdir"].format(key=handle.key)

        return Path(root) / subdir

    def default_handle(self, kind: StorageKind, key: str) -> StorageHandle:
        """Create handle with default zone for this kind."""
        kind_cfg = self.config["kind_patterns"][kind.value]
        default_zone = StorageZone(kind_cfg["default_zone"])
        return StorageHandle(kind=kind, key=key, zone=default_zone)

    def locate(self, kind: StorageKind, key: str) -> Optional[Path]:
        """Find where an asset exists (check all zones)."""
        for zone in StorageZone:
            handle = StorageHandle(kind=kind, key=key, zone=zone)
            try:
                path = self.resolve(handle)
                if path.exists():
                    return path
            except ValueError:
                continue
        return None
```

---

### Part 3: Integration Points

#### 3.1 Relationship Between Systems

```
┌─────────────────────────────────────────────────────────────┐
│                      Device Registry                         │
│  "What machines exist and what can they do?"                │
│                                                             │
│  devices.json → DeviceRegistry → DeviceInfo                 │
│  - roles: [trainer, eval_worker, storage_hot, ...]         │
│  - gpus, cpu, memory                                        │
│  - storage_zones: ["hot", "warm", "cold"]                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ cross-references
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Storage System                          │
│  "Where do things live and how to find them?"               │
│                                                             │
│  storage_zones.json → StorageResolver → Path                │
│  - zones: hot/warm/cold with device roots                   │
│  - kind_patterns: checkpoint, dataset, etc.                 │
│  - handles: logical references                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ integrates with existing
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Existing Systems                           │
│                                                             │
│  core/hosts.py     - SSH, services, networking              │
│  vault/zones.py    - Zone federation, transfers             │
│  vault/storage_registry.py - Strongholds, sync              │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 How Host Registry vs Device Registry Relate

| Concept | Host Registry (`hosts.json`) | Device Registry (`devices.json`) |
|---------|------------------------------|----------------------------------|
| Focus | Network services | Compute capabilities |
| Purpose | "Where is the inference API?" | "What machine can run evals?" |
| Roles | trainer/inference/storage | trainer/inference/eval_worker/data_forge/... |
| Services | Port numbers, health checks | N/A |
| Hardware | N/A | GPUs, CPU, memory |

**Decision**: Keep both. They answer different questions:
- `hosts.json`: Network topology, service endpoints
- `devices.json`: Compute capabilities, job routing

Both share `hostname` as the link. A device's `hostname` matches a host's `host` field.

#### 3.3 How Storage Registry vs Storage System Relate

| Concept | Storage Registry (current) | Storage System (new) |
|---------|---------------------------|---------------------|
| Focus | Physical locations | Logical references |
| Purpose | "Sync to Synology" | "Get checkpoint path" |
| Abstraction | Strongholds | Zones + Handles |
| Operations | sync_treasure() | resolve(handle) |

**Decision**: The new Storage System is a **layer on top**. It uses StorageResolver for path logic, but can call existing `storage_registry.sync_treasure()` for actual transfers.

---

### Part 4: Job Dispatch System

With DeviceRegistry, we can route jobs to appropriate workers.

#### 4.1 Job Types

```python
class JobType(str, Enum):
    EVAL = "eval"              # Skill evaluation (needs inference)
    DATA_GEN = "data_gen"      # Generate training data
    SPARRING = "sparring"      # Sparring sessions (needs inference)
    ARCHIVE = "archive"        # Archive/compress checkpoints
    ANALYTICS = "analytics"    # Metrics aggregation
```

#### 4.2 JobRouter

```python
class JobRouter:
    def __init__(self, device_registry: DeviceRegistry):
        self.devices = device_registry

    def find_workers(self, job_type: JobType) -> List[DeviceInfo]:
        """Find devices capable of running this job type."""
        role_map = {
            JobType.EVAL: DeviceRole.EVAL_WORKER,
            JobType.DATA_GEN: DeviceRole.DATA_FORGE,
            JobType.SPARRING: DeviceRole.EVAL_WORKER,  # Needs inference too
            JobType.ARCHIVE: DeviceRole.VAULT_WORKER,
            JobType.ANALYTICS: DeviceRole.ANALYTICS,
        }
        required_role = role_map.get(job_type)
        if not required_role:
            return []

        candidates = self.devices.devices_with_role(required_role)

        # Additional filters
        if job_type in (JobType.EVAL, JobType.SPARRING):
            # Need GPU for inference
            candidates = [d for d in candidates if d.has_gpu() or
                          self._can_reach_inference(d)]

        return candidates

    def _can_reach_inference(self, device: DeviceInfo) -> bool:
        """Check if device can reach inference service."""
        # Use host registry to check
        return True  # Simplified
```

#### 4.3 Example: Distributing Skill Evals

```python
# In guild/task_master.py or new guild/job_dispatcher.py

def dispatch_eval_jobs(skill_id: str, batch_count: int = 10):
    """Distribute eval batches across available workers."""

    device_reg = get_device_registry()
    router = JobRouter(device_reg)

    workers = router.find_workers(JobType.EVAL)
    if not workers:
        raise RuntimeError("No eval workers available")

    # Distribute batches round-robin
    jobs = []
    for i in range(batch_count):
        worker = workers[i % len(workers)]
        jobs.append({
            "worker": worker.device_id,
            "hostname": worker.hostname,
            "skill_id": skill_id,
            "batch_index": i,
        })

    # Submit jobs (via HTTP to workers, or queue system)
    for job in jobs:
        submit_to_worker(job)

    return jobs
```

---

## IMPLEMENTATION PHASES

### Phase 1: Core Types + Config (Day 1)

**Goal**: Define types and config schemas, no behavior changes yet.

#### Files to Create:

```
core/
├── devices.py              # DeviceRole, DeviceInfo, DeviceRegistry
└── storage_types.py        # StorageZone, StorageKind, StorageHandle

config/
├── devices.json            # Device definitions (copy example)
└── storage_zones.json      # Zone definitions (new)
```

#### Tasks:

1. **Create `core/devices.py`**
   - `DeviceRole` enum (10 roles)
   - `GPUInfo`, `CPUInfo`, `NetworkInfo` dataclasses
   - `DeviceInfo` dataclass with role checking methods
   - `DeviceRegistry` class that loads from `config/devices.json`
   - `get_device_registry()` singleton helper
   - `get_current_device()` - reads `TRAINING_DEVICE_ID` env var

2. **Create `core/storage_types.py`**
   - `StorageZone` enum (hot/warm/cold)
   - `StorageKind` enum (10 kinds)
   - `StorageHandle` frozen dataclass
   - Basic validation

3. **Create `config/devices.json`**
   - Start with: trainer4090, inference3090, synology_data, synology_archive
   - Leave Mac minis as placeholders for now

4. **Create `config/storage_zones.json`**
   - Define zone roots for each device
   - Define kind patterns (subdir templates)

#### Tests:
```python
# tests/test_device_registry.py
def test_load_devices():
    reg = DeviceRegistry(Path("config/devices.json"))
    assert reg.get("trainer4090") is not None

def test_filter_by_role():
    reg = DeviceRegistry(...)
    workers = reg.devices_with_role(DeviceRole.EVAL_WORKER)
    assert len(workers) >= 1
```

---

### Phase 2: Storage Resolver (Day 2)

**Goal**: StorageResolver can turn handles into paths.

#### Files to Create/Modify:

```
vault/
├── storage_resolver.py     # NEW: StorageResolver class
└── storage_registry.py     # ADD: Integration with resolver
```

#### Tasks:

1. **Create `vault/storage_resolver.py`**
   - `StorageResolver` class
   - `resolve(handle) -> Path`
   - `default_handle(kind, key) -> StorageHandle`
   - `locate(kind, key) -> Optional[Path]` (search all zones)
   - `get_resolver()` singleton

2. **Add `ask_storage()` convenience function**
   ```python
   def ask_storage(kind: StorageKind, key: str) -> Path:
       """One-liner to get path for a storage item."""
       resolver = get_resolver()
       handle = resolver.default_handle(kind, key)
       return resolver.resolve(handle)
   ```

3. **Integration with existing StorageRegistry**
   - `StorageRegistry.resolve_handle(handle)` method
   - Uses `StorageResolver` internally

#### Tests:
```python
def test_resolve_checkpoint():
    resolver = StorageResolver(...)
    handle = StorageHandle(
        kind=StorageKind.CHECKPOINT,
        key="checkpoint-182000",
        zone=StorageZone.HOT
    )
    path = resolver.resolve(handle)
    assert "models/current_model/checkpoint-182000" in str(path)
```

---

### Phase 3: Host + Device Integration (Day 3)

**Goal**: Unify host registry and device registry access patterns.

#### Files to Modify:

```
core/
├── hosts.py                # ADD: device_id property, cross-reference
└── devices.py              # ADD: host lookup method
```

#### Tasks:

1. **Add device_id to HostConfig**
   ```python
   @dataclass
   class HostConfig:
       # ... existing fields ...
       device_id: Optional[str] = None  # Link to devices.json
   ```

2. **Add cross-reference methods**
   ```python
   # In devices.py
   def get_host_for_device(device_id: str) -> Optional[HostConfig]:
       """Get host config for a device (by matching hostname)."""
       device = get_device_registry().get(device_id)
       if not device:
           return None
       # Match by hostname
       for host in get_registry().list_all():
           if host.host == device.hostname:
               return host
       return None
   ```

3. **Update hosts.json** to include `device_id` links
   ```json
   "4090": {
       "device_id": "trainer4090",
       ...
   }
   ```

---

### Phase 4: Migration - Training Daemon (Day 4-5)

**Goal**: Training daemon uses storage handles instead of hardcoded paths.

#### Files to Modify:

```
core/
├── train.py                # Use ask_storage() for checkpoint paths
├── training_daemon.py      # Use handles for queue paths
└── training_status.py      # Use handles for status paths

trainer/
├── core/engine.py          # Use handles for model/checkpoint paths
└── monitoring/callbacks.py # Use handles for ledger paths
```

#### Tasks:

1. **Refactor checkpoint path construction**

   Before:
   ```python
   ckpt_path = self.base_dir / "models" / "current_model" / f"checkpoint-{step}"
   ```

   After:
   ```python
   from vault.storage_resolver import ask_storage
   from core.storage_types import StorageKind

   ckpt_path = ask_storage(StorageKind.CHECKPOINT, f"checkpoint-{step}")
   ```

2. **Refactor queue paths**

   Before:
   ```python
   queue_dir = self.base_dir / "queue" / priority
   ```

   After:
   ```python
   queue_path = ask_storage(StorageKind.QUEUE, priority)
   ```

3. **Backward compatibility**
   - Keep existing paths working during migration
   - Add `STORAGE_V2=1` env var to enable new paths

---

### Phase 5: Retention Engine (Day 6)

**Goal**: Retention policies work with storage handles.

#### Files to Create/Modify:

```
management/
├── retention_engine.py     # NEW: Unified retention policy engine
├── checkpoint_retention.py # REFACTOR: Use handles
└── safe_checkpoint_cleanup.py  # REFACTOR: Use handles
```

#### Tasks:

1. **Create `management/retention_engine.py`**
   ```python
   class RetentionEngine:
       def __init__(self):
           self.resolver = get_resolver()
           self.device = get_current_device()

       def list_checkpoints(self, zone: StorageZone) -> List[StorageHandle]:
           """List all checkpoints in a zone."""
           pass

       def apply_hot_policy(self, keep_recent: int = 50, keep_promoted: int = 3):
           """Apply retention policy to HOT zone."""
           pass

       def archive_to_cold(self, handles: List[StorageHandle]):
           """Archive checkpoints to COLD zone."""
           pass
   ```

2. **Define retention rules in config**
   ```json
   {
     "retention_policies": {
       "hot": {
         "keep_recent_checkpoints": 50,
         "keep_promoted_snapshots": 3,
         "max_age_hours": 168
       },
       "warm": {
         "keep_all_snapshots": true,
         "max_age_days": 90
       },
       "cold": {
         "keep_forever": true,
         "compress": true
       }
     }
   }
   ```

---

### Phase 6: Job Dispatcher (Day 7-8)

**Goal**: Jobs can be routed to appropriate workers.

#### Files to Create:

```
guild/
├── job_types.py            # JobType enum
├── job_router.py           # JobRouter class
└── job_dispatcher.py       # Dispatch jobs to workers

workers/                    # NEW directory
├── __init__.py
├── base_worker.py          # BaseWorker ABC
├── eval_worker.py          # EvalWorker implementation
└── data_forge_worker.py    # DataForgeWorker implementation
```

#### Tasks:

1. **Create job routing infrastructure**
   ```python
   # guild/job_router.py
   class JobRouter:
       def find_workers(self, job_type: JobType) -> List[DeviceInfo]: ...
       def best_worker(self, job_type: JobType) -> DeviceInfo: ...
   ```

2. **Create worker base class**
   ```python
   # workers/base_worker.py
   class BaseWorker(ABC):
       @abstractmethod
       def handle_job(self, job: Dict) -> Dict: ...

       def run_server(self, port: int): ...
   ```

3. **Create eval worker**
   ```python
   # workers/eval_worker.py
   class EvalWorker(BaseWorker):
       def handle_job(self, job: Dict) -> Dict:
           skill_id = job["skill_id"]
           # Run evaluation
           return {"success": True, "results": ...}
   ```

4. **Create dispatcher**
   ```python
   # guild/job_dispatcher.py
   class JobDispatcher:
       def submit(self, job_type: JobType, payload: Dict) -> str:
           """Submit job, return job_id."""
           worker = self.router.best_worker(job_type)
           # POST to worker HTTP endpoint
           return job_id

       def wait(self, job_id: str, timeout: int = 300) -> Dict:
           """Wait for job completion."""
           pass
   ```

---

### Phase 7: Mac Mini Integration (Day 9-10)

**Goal**: Actually run workers on Mac minis.

#### Tasks:

1. **Setup script for Mac minis**
   ```bash
   # scripts/setup_worker.sh
   # Runs on each Mac mini to:
   # - Install Python dependencies
   # - Copy worker code
   # - Setup systemd/launchd service
   ```

2. **Worker deployment config**
   ```json
   {
     "workers": {
       "macmini_eval_1": {
         "roles": ["eval_worker", "data_forge"],
         "port": 8900,
         "env": {
           "TRAINING_DEVICE_ID": "macmini_eval_1",
           "INFERENCE_URL": "http://192.168.x.x:8765"
         }
       }
     }
   }
   ```

3. **Test end-to-end job submission**
   - Submit eval job from 4090
   - Worker on Mac mini executes
   - Results returned to 4090

---

## FILE INVENTORY

### New Files

| File | Purpose | Priority |
|------|---------|----------|
| `core/devices.py` | Device registry + types | P0 |
| `core/storage_types.py` | Storage handle types | P0 |
| `vault/storage_resolver.py` | Handle → Path resolution | P0 |
| `config/devices.json` | Device definitions | P0 |
| `config/storage_zones.json` | Zone definitions | P0 |
| `management/retention_engine.py` | Unified retention | P1 |
| `guild/job_types.py` | Job type definitions | P1 |
| `guild/job_router.py` | Find workers for jobs | P1 |
| `guild/job_dispatcher.py` | Submit/track jobs | P2 |
| `workers/base_worker.py` | Worker base class | P2 |
| `workers/eval_worker.py` | Eval worker impl | P2 |

### Modified Files

| File | Change | Priority |
|------|--------|----------|
| `core/hosts.py` | Add device_id link | P1 |
| `config/hosts.json` | Add device_id field | P1 |
| `core/train.py` | Use storage handles | P2 |
| `trainer/core/engine.py` | Use storage handles | P2 |

---

## ENVIRONMENT VARIABLES

Each node sets these in its environment:

```bash
# On trainer4090
export TRAINING_DEVICE_ID=trainer4090
export STORAGE_ZONES_PATH=/path/to/training/config/storage_zones.json
export DEVICES_CONFIG_PATH=/path/to/training/config/devices.json

# On inference3090
export TRAINING_DEVICE_ID=inference3090
# ... same config paths (could be on NAS or synced)

# On macmini_eval_1
export TRAINING_DEVICE_ID=macmini_eval_1
export INFERENCE_URL=http://192.168.x.x:8765
```

---

## TESTING STRATEGY

### Unit Tests

```python
# tests/test_device_registry.py
# tests/test_storage_types.py
# tests/test_storage_resolver.py
# tests/test_job_router.py
```

### Integration Tests

```python
# tests/integration/test_cross_device.py
def test_locate_checkpoint_across_zones(): ...
def test_job_dispatch_to_worker(): ...
```

### Manual Validation

1. Set `TRAINING_DEVICE_ID=trainer4090` and verify paths resolve correctly
2. Test `ask_storage(CHECKPOINT, "checkpoint-182000")` returns valid path
3. Verify retention engine identifies correct checkpoints to archive

---

## ROLLOUT PLAN

1. **Week 1**: Phase 1-3 (types, resolver, integration)
2. **Week 2**: Phase 4-5 (migration, retention)
3. **Week 3**: Phase 6-7 (job dispatch, Mac mini setup)

Each phase is independently testable and doesn't break existing functionality.

---

## RISKS + MITIGATIONS

| Risk | Mitigation |
|------|------------|
| Config file out of sync across hosts | Use NAS for shared config, or sync script |
| Mac minis offline | Job dispatcher has timeout + retry logic |
| Path resolution fails | Fallback to hardcoded paths during migration |
| Too much abstraction | Keep `ask_storage()` simple, document well |

---

## FUTURE CONSIDERATIONS

### Distributed Training (Not Now, Maybe Later)

The device registry is designed to support multi-GPU distributed training in the future:

- **Network is solid**: 10Gbps infrastructure can handle gradient sync
- **Potential setup**: 4090 + 3090 via DeepSpeed ZeRO-3 or FSDP
- **When ready**: Add `DISTRIBUTED_TRAINER` role to device registry

**Why not now:**
- Single 4090 is sufficient for current model sizes (0.6B-4B)
- Complexity overhead not justified yet
- Focus on the side-task workers first (evals, data gen)

**When it makes sense:**
- Training larger models (8B+) that don't fit on single GPU
- Need faster iteration on existing model sizes
- Have stable worker infrastructure as foundation

---

## SUCCESS CRITERIA

1. `get_device_registry().devices_with_role(DeviceRole.EVAL_WORKER)` returns list
2. `ask_storage(StorageKind.CHECKPOINT, "checkpoint-182000")` returns valid path
3. Retention engine can list all checkpoints in HOT zone
4. Job dispatcher can submit eval job to Mac mini and get result back
5. All existing training functionality continues to work

---

## APPENDIX: Example Usage

### A. Finding a Checkpoint

```python
from vault.storage_resolver import ask_storage, get_resolver
from core.storage_types import StorageKind, StorageZone, StorageHandle

# Simple: get default path
path = ask_storage(StorageKind.CHECKPOINT, "checkpoint-182000")

# Explicit: specify zone
resolver = get_resolver()
handle = StorageHandle(
    kind=StorageKind.CHECKPOINT,
    key="checkpoint-182000",
    zone=StorageZone.WARM  # Get from NAS instead of local
)
nas_path = resolver.resolve(handle)

# Search: find wherever it exists
found_path = resolver.locate(StorageKind.CHECKPOINT, "checkpoint-182000")
```

### B. Dispatching Eval Job

```python
from guild.job_dispatcher import JobDispatcher, JobType

dispatcher = JobDispatcher()

# Submit eval job
job_id = dispatcher.submit(
    JobType.EVAL,
    {
        "skill_id": "bin",
        "level": 5,
        "batch_size": 100,
    }
)

# Wait for result
result = dispatcher.wait(job_id, timeout=300)
print(f"Accuracy: {result['accuracy']}")
```

### C. Retention Policy

```python
from management.retention_engine import RetentionEngine

engine = RetentionEngine()

# Run daily cleanup
engine.apply_hot_policy(
    keep_recent=50,      # Keep last 50 checkpoints
    keep_promoted=3,     # Keep 3 promoted snapshots
)

# Archive old snapshots to cold storage
old_snapshots = engine.list_expired(zone=StorageZone.WARM, max_age_days=90)
engine.archive_to_cold(old_snapshots)
```
