#!/usr/bin/env python3
"""
Model Database & Scanner

Discovers, tracks, and manages base models and LoRA adapters across the filesystem.
Maintains a persistent database of available models with their metadata and compatibility info.

=== CORE RESPONSIBILITY ===
Answer the question: "What models do I have and where are they?"

Scans directories for:
- Base models (HuggingFace format with config.json)
- LoRA adapters (adapter_config.json files)
- Model metadata (size, architecture, layers, vocab)
- Adapter compatibility (which adapters work with which base models)

=== DATA FLOW ===

Discovery Workflow:
1. User calls scan_for_models(search_paths) or scan_for_adapters(search_paths)
2. Scanner walks directory tree looking for:
   - Models: directories with config.json
   - Adapters: directories with adapter_config.json
3. For each found item:
   - Parse config JSON
   - Calculate disk size
   - Extract metadata (hidden_size, num_layers, vocab_size, lora_r, etc.)
   - Create ModelInfo or AdapterInfo object
4. Store in memory (self.models dict, self.adapters dict)
5. Save to persistent database (~/.ultimate_trainer/models.json)

Query Workflow:
- list_models() → All discovered models
- list_adapters(base_model) → Adapters compatible with base_model
- get_model(name) → Single model by name
- display_models() → Pretty-printed model list with adapter counts

=== DATABASE FORMAT ===
Stored at: ~/.ultimate_trainer/models.json

```json
{
  "models": {
    "Qwen3-0.6B": {
      "name": "Qwen3-0.6B",
      "path": "/path/to/model",
      "size_gb": 1.5,
      "hidden_size": 1024,
      "num_layers": 28,
      "vocab_size": 151936,
      "model_type": "qwen2",
      "last_seen": "2025-11-24T12:00:00",
      "tested": false
    }
  },
  "adapters": {
    "/path/to/adapter": {
      "name": "my_adapter",
      "path": "/path/to/adapter",
      "base_model": "Qwen3-0.6B",
      "lora_r": 8,
      "lora_alpha": 16,
      "trainable_params": 4194304,
      "size_mb": 15.2,
      "created": "2025-11-24T12:00:00",
      "last_checkpoint": null,
      "training_complete": true
    }
  },
  "updated": "2025-11-24T12:00:00"
}
```

=== USAGE EXAMPLE ===
```python
from core.model_db import ModelDatabase
from pathlib import Path

# Initialize (auto-loads existing database)
db = ModelDatabase()

# Scan for models
search_paths = [
    Path("/media/user/ST/aiPROJECT/models"),
    Path.home() / ".cache/huggingface/hub"
]
db.scan_for_models(search_paths)

# List available models
for model in db.list_models():
    print(f"{model.name}: {model.size_gb} GB, {model.num_layers} layers")

# Find adapters for a specific model
adapters = db.list_adapters(base_model="Qwen3-0.6B")
print(f"Found {len(adapters)} adapters for Qwen3-0.6B")

# Pretty display
db.display_models()  # Shows models + compatible adapter counts
```

=== INTEGRATION POINTS ===
- Used by: CLI tools, monitoring UI (model selection dropdowns)
- Inputs: Directory paths to scan
- Outputs: models.json database, ModelInfo/AdapterInfo objects
- Persistence: ~/.ultimate_trainer/models.json

=== ADAPTER COMPATIBILITY ===
Matches adapters to base models by:
1. adapter.base_model contains model.name, OR
2. model.name contains adapter.base_model

Examples:
- Adapter base_model: "/path/to/Qwen3-0.6B" → Matches model name "Qwen3-0.6B"
- Adapter base_model: "qwen3" → Matches any model with "qwen3" in name

=== FILE DISCOVERY ===
Models:
- Must have: config.json in root directory
- Size calculated: sum(all file sizes in directory tree)
- Metadata from config.json: hidden_size, num_hidden_layers, vocab_size, model_type

Adapters:
- Must have: adapter_config.json
- Size calculated: sum(non-.pt files)
- Metadata from adapter_config.json: base_model_name_or_path, r, lora_alpha
- Trainable params estimated: 7 modules × 2 matrices × r × 4096 (typical hidden size)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class ModelInfo:
    """
    Metadata for a discovered base model.

    Extracted from model's config.json file and filesystem metadata.
    Stored in models.json database.

    Attributes:
        name: Model directory name (e.g., "Qwen3-0.6B")
        path: Absolute path to model directory
        size_gb: Total disk size in GB (sum of all files)
        hidden_size: Model hidden dimension from config (e.g., 1024)
        num_layers: Number of transformer layers from config (e.g., 28)
        vocab_size: Vocabulary size from config (e.g., 151936)
        model_type: Model architecture type from config (e.g., "qwen2")
        last_seen: ISO timestamp when last discovered
        tested: Whether model has been tested/validated (default: False)

    Example:
        model = ModelInfo(
            name="Qwen3-0.6B",
            path="/path/to/model",
            size_gb=1.5,
            hidden_size=1024,
            num_layers=28,
            vocab_size=151936,
            model_type="qwen2",
            last_seen="2025-11-24T12:00:00",
            tested=False
        )
    """
    name: str
    path: str
    size_gb: float
    hidden_size: int
    num_layers: int
    vocab_size: int
    model_type: str
    last_seen: str
    tested: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class AdapterInfo:
    """
    Metadata for a discovered LoRA adapter.

    Extracted from adapter_config.json and filesystem metadata.
    Stored in models.json database.

    Attributes:
        name: Adapter directory name (e.g., "my_adapter")
        path: Absolute path to adapter directory
        base_model: Base model name/path from adapter_config.json
        lora_r: LoRA rank (r parameter, e.g., 8)
        lora_alpha: LoRA alpha scaling parameter (e.g., 16)
        trainable_params: Estimated trainable parameters (7×2×r×4096)
        size_mb: Disk size in MB (sum of non-.pt files)
        created: ISO timestamp from file creation time
        last_checkpoint: Path to checkpoint if in-progress training, else None
        training_complete: True if final adapter, False if checkpoint

    Trainable Params Formula:
        trainable_params = 7 modules × 2 matrices × r × hidden_size
        Assumes 7 target modules (typical: q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj) and hidden_size=4096

    Example:
        adapter = AdapterInfo(
            name="my_adapter",
            path="/path/to/adapter",
            base_model="Qwen3-0.6B",
            lora_r=8,
            lora_alpha=16,
            trainable_params=4194304,  # 7×2×8×4096
            size_mb=15.2,
            created="2025-11-24T12:00:00",
            last_checkpoint=None,
            training_complete=True
        )
    """
    name: str
    path: str
    base_model: str
    lora_r: int
    lora_alpha: int
    trainable_params: int
    size_mb: float
    created: str
    last_checkpoint: Optional[str] = None
    training_complete: bool = False

    def to_dict(self):
        return asdict(self)


class ModelDatabase:
    """
    Persistent database of discovered models and adapters.

    Scans filesystem for HuggingFace models and LoRA adapters, extracts metadata,
    and maintains a persistent JSON database for quick lookups.

    === RESPONSIBILITIES ===
    1. Discovery - Scan directories for models (config.json) and adapters (adapter_config.json)
    2. Metadata Extraction - Parse config files, calculate sizes, extract architecture info
    3. Persistence - Save/load database to ~/.ultimate_trainer/models.json
    4. Querying - Fast lookups by model name, adapter compatibility
    5. Display - Pretty-print model/adapter listings with metadata

    === DATA STRUCTURES ===
    self.models: Dict[str, ModelInfo]
        - Key: Model name (e.g., "Qwen3-0.6B")
        - Value: ModelInfo object with metadata

    self.adapters: Dict[str, AdapterInfo]
        - Key: Adapter path (absolute path to adapter directory)
        - Value: AdapterInfo object with metadata

    === DISCOVERY ALGORITHM ===
    scan_for_models(search_paths):
        For each search_path:
            For each directory in search_path:
                If config.json exists:
                    1. Parse config.json (hidden_size, num_layers, vocab, model_type)
                    2. Calculate directory size (sum all files)
                    3. Create ModelInfo object
                    4. Store in self.models[name]
        Save to disk

    scan_for_adapters(search_paths):
        For each search_path:
            For each adapter_config.json in tree:
                1. Parse adapter_config.json (base_model, r, lora_alpha)
                2. Calculate adapter size (sum non-.pt files)
                3. Estimate trainable params: 7×2×r×4096
                4. Create AdapterInfo object
                5. Store in self.adapters[path]
        Save to disk

    === USAGE EXAMPLE ===
    ```python
    # Initialize and load existing database
    db = ModelDatabase()

    # Scan for new models
    db.scan_for_models([Path("/models"), Path.home()/"huggingface"])

    # Query models
    all_models = db.list_models()
    qwen = db.get_model("Qwen3-0.6B")
    adapters = db.list_adapters(base_model="Qwen3-0.6B")

    # Display
    db.display_models()  # Pretty table with adapter counts
    ```

    === PERSISTENCE ===
    - Database path: ~/.ultimate_trainer/models.json (customizable)
    - Format: JSON with models, adapters, updated timestamp
    - Auto-loads on __init__, auto-saves after scan operations
    - Thread-safe: Each operation completes atomically
    """

    def __init__(self, db_path: Path = Path.home() / ".ultimate_trainer" / "models.json"):
        """
        Initialize database and load existing data.

        Args:
            db_path: Path to database file (default: ~/.ultimate_trainer/models.json)

        Side Effects:
            - Creates db_path parent directory if needed
            - Loads existing database from disk (if exists)
            - Initializes empty models/adapters dicts

        Example:
            db = ModelDatabase()  # Uses default path
            db = ModelDatabase(Path("/custom/db.json"))  # Custom path
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelInfo] = {}
        self.adapters: Dict[str, AdapterInfo] = {}
        self.load()

    def load(self):
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    data = json.load(f)
                    self.models = {k: ModelInfo(**v) for k, v in data.get('models', {}).items()}
                    self.adapters = {k: AdapterInfo(**v) for k, v in data.get('adapters', {}).items()}
                print(f"📂 Loaded database: {len(self.models)} models, {len(self.adapters)} adapters")
            except Exception as e:
                print(f"⚠️  Error loading database: {e}")

    def save(self):
        """Save database to disk."""
        data = {
            'models': {k: v.to_dict() for k, v in self.models.items()},
            'adapters': {k: v.to_dict() for k, v in self.adapters.items()},
            'updated': datetime.now().isoformat()
        }
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def scan_for_models(self, search_paths: List[Path]) -> List[ModelInfo]:
        """Scan directories for models."""
        print("\n🔍 Scanning for models...")
        found = []

        for search_path in search_paths:
            if not search_path.exists():
                print(f"   ⚠️  Path not found: {search_path}")
                continue

            print(f"   Scanning: {search_path}")

            for item in search_path.iterdir():
                if not item.is_dir():
                    continue

                # Check if it's a model (has config.json)
                config_file = item / "config.json"
                try:
                    if not config_file.exists():
                        continue
                except PermissionError:
                    # Skip directories we can't access
                    continue

                try:
                    with open(config_file) as f:
                        config = json.load(f)

                    # Calculate size
                    size_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    size_gb = size_bytes / (1024**3)

                    model_info = ModelInfo(
                        name=item.name,
                        path=str(item),
                        size_gb=round(size_gb, 2),
                        hidden_size=config.get("hidden_size", 0),
                        num_layers=config.get("num_hidden_layers", 0),
                        vocab_size=config.get("vocab_size", 0),
                        model_type=config.get("model_type", "unknown"),
                        last_seen=datetime.now().isoformat(),
                        tested=self.models.get(item.name, ModelInfo(
                            name="", path="", size_gb=0, hidden_size=0,
                            num_layers=0, vocab_size=0, model_type="",
                            last_seen="", tested=False
                        )).tested
                    )

                    self.models[item.name] = model_info
                    found.append(model_info)
                    print(f"      ✓ {item.name} ({size_gb:.1f} GB)")

                except Exception as e:
                    print(f"      ✗ {item.name}: {e}")

        self.save()
        print(f"\n   Found {len(found)} models")
        return found

    def scan_for_adapters(self, search_paths: List[Path]) -> List[AdapterInfo]:
        """Scan for LoRA adapters."""
        print("\n🔍 Scanning for adapters...")
        found = []

        for search_path in search_paths:
            if not search_path.exists():
                continue

            print(f"   Scanning: {search_path}")

            for item in search_path.rglob("adapter_config.json"):
                adapter_dir = item.parent

                try:
                    with open(item) as f:
                        config = json.load(f)

                    # Get base model name
                    base_model = config.get("base_model_name_or_path", "unknown")
                    base_model_name = Path(base_model).name if base_model else "unknown"

                    # Calculate size
                    size_bytes = sum(f.stat().st_size for f in adapter_dir.rglob("*")
                                   if f.is_file() and not f.name.endswith('.pt'))
                    size_mb = size_bytes / (1024**2)

                    # Estimate trainable params
                    r = config.get("r", 0)
                    # Rough estimate: 7 target modules × 2 matrices × r × hidden_size
                    hidden_size = 4096  # Typical
                    trainable_params = 7 * 2 * r * hidden_size

                    adapter_info = AdapterInfo(
                        name=adapter_dir.name,
                        path=str(adapter_dir),
                        base_model=base_model_name,
                        lora_r=config.get("r", 0),
                        lora_alpha=config.get("lora_alpha", 0),
                        trainable_params=trainable_params,
                        size_mb=round(size_mb, 1),
                        created=datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                        last_checkpoint=str(adapter_dir) if "checkpoint" in adapter_dir.name else None,
                        training_complete="checkpoint" not in adapter_dir.name
                    )

                    self.adapters[str(adapter_dir)] = adapter_info
                    found.append(adapter_info)
                    print(f"      ✓ {adapter_dir.name} (r={r}, {size_mb:.0f} MB)")

                except Exception as e:
                    print(f"      ✗ {adapter_dir}: {e}")

        self.save()
        print(f"\n   Found {len(found)} adapters")
        return found

    def list_models(self) -> List[ModelInfo]:
        """List all known models."""
        return list(self.models.values())

    def list_adapters(self, base_model: Optional[str] = None) -> List[AdapterInfo]:
        """List adapters, optionally filtered by base model."""
        adapters = list(self.adapters.values())
        if base_model:
            adapters = [a for a in adapters if base_model in a.base_model or a.base_model in base_model]
        return adapters

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model by name."""
        return self.models.get(name)

    def display_models(self):
        """Display models in a nice format."""
        models = sorted(self.models.values(), key=lambda m: m.size_gb, reverse=True)

        if not models:
            print("\n❌ No models found. Run scan first.")
            return

        print("\n" + "=" * 80)
        print("AVAILABLE MODELS")
        print("=" * 80)

        for i, model in enumerate(models, 1):
            status = "✓ Tested" if model.tested else "○ Untested"

            print(f"\n{i}. {model.name}")
            print(f"   Path: {model.path}")
            print(f"   Size: {model.size_gb} GB ({model.num_layers} layers, {model.hidden_size} hidden)")
            print(f"   Type: {model.model_type}")
            print(f"   Status: {status}")

            # Find compatible adapters
            adapters = self.list_adapters(model.name)
            if adapters:
                print(f"   Adapters: {len(adapters)} found")
                for adapter in adapters[:3]:  # Show first 3
                    checkpoint = " [checkpoint]" if adapter.last_checkpoint else ""
                    print(f"      • {adapter.name} (r={adapter.lora_r}){checkpoint}")
                if len(adapters) > 3:
                    print(f"      ... and {len(adapters) - 3} more")

        print("\n" + "=" * 80)

    def display_adapters(self, base_model: Optional[str] = None):
        """Display adapters."""
        adapters = self.list_adapters(base_model)

        if not adapters:
            print("\n❌ No adapters found.")
            return

        print("\n" + "=" * 80)
        print(f"AVAILABLE ADAPTERS{' for ' + base_model if base_model else ''}")
        print("=" * 80)

        for i, adapter in enumerate(sorted(adapters, key=lambda a: a.created, reverse=True), 1):
            status = "✓ Complete" if adapter.training_complete else "⚙️  Checkpoint"

            print(f"\n{i}. {adapter.name}")
            print(f"   Path: {adapter.path}")
            print(f"   Base Model: {adapter.base_model}")
            print(f"   LoRA: r={adapter.lora_r}, alpha={adapter.lora_alpha}")
            print(f"   Size: {adapter.size_mb} MB")
            print(f"   Trainable Params: {adapter.trainable_params:,}")
            print(f"   Status: {status}")
            print(f"   Created: {adapter.created[:10]}")

        print("\n" + "=" * 80)


def main():
    """Test model scanner."""
    db = ModelDatabase()

    # Common search paths
    search_paths = [
        Path("/media/user/ST/aiPROJECT/models"),
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/tmp")
    ]

    print("=" * 80)
    print("MODEL DATABASE & SCANNER")
    print("=" * 80)

    # Scan for models
    db.scan_for_models(search_paths)

    # Scan for adapters
    adapter_paths = [
        Path("/tmp"),
        Path.home() / ".cache" / "huggingface",
    ]
    db.scan_for_adapters(adapter_paths)

    # Display
    db.display_models()

    print("\n💾 Database saved to:", db.db_path)


if __name__ == "__main__":
    main()
