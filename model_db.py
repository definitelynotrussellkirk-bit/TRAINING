#!/usr/bin/env python3
"""
Model Database & Scanner

Discovers models, tracks what's available, finds compatible adapters.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class ModelInfo:
    """Information about a base model."""
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
    """Information about a LoRA adapter."""
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
    """Database of available models and adapters."""

    def __init__(self, db_path: Path = Path.home() / ".ultimate_trainer" / "models.json"):
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
