"""
Techniques - Named training method configurations.

A Technique is a complete training stack: optimizer choice, precision,
schedule, and stability settings. Think of it as the "Discipline" or
"School of Training" that a Hero studies under.

Usage:
    from trainer.techniques import Technique, get_technique, list_techniques

    # Get a technique by name
    muon = get_technique("muon")
    print(muon.display_name)  # "Muon Physics"

    # List all available techniques
    for tech in list_techniques():
        print(f"{tech.id}: {tech.rpg_name}")

    # Check technique properties
    if muon.optimizer_type == "muon":
        print("Using orthogonalized momentum!")

Techniques are defined in configs/physics/*.yaml and loaded at runtime.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class TechniqueType(str, Enum):
    """
    Core technique/optimizer types.

    These are the fundamental training methods available.
    Each maps to a Physics configuration file.
    """
    MUON = "muon"        # Orthogonalized momentum (experimental)
    ADAMW = "adamw"      # Classic adaptive optimizer
    SGD = "sgd"          # Stochastic gradient descent (rare for LLMs)

    @property
    def display_name(self) -> str:
        """Human-friendly name."""
        names = {
            TechniqueType.MUON: "Muon Physics",
            TechniqueType.ADAMW: "AdamW Physics",
            TechniqueType.SGD: "SGD Physics",
        }
        return names.get(self, self.value)

    @property
    def rpg_name(self) -> str:
        """RPG/lore name."""
        names = {
            TechniqueType.MUON: "The Orthogonal Way",
            TechniqueType.ADAMW: "The Classical Path",
            TechniqueType.SGD: "The Steep Descent",
        }
        return names.get(self, self.value)

    @property
    def icon(self) -> str:
        """Icon for the technique."""
        icons = {
            TechniqueType.MUON: "⚛️",
            TechniqueType.ADAMW: "📐",
            TechniqueType.SGD: "⛷️",
        }
        return icons.get(self, "🔧")


@dataclass
class TechniqueConfig:
    """
    Full technique configuration loaded from YAML.

    Contains all settings needed to configure training physics.
    """
    id: str
    name: str
    rpg_name: str
    version: str
    description: str

    # Display
    icon: str = "🔧"
    color: str = "#6B7280"

    # Optimizer config
    optimizer_type: str = "adamw"
    optimizer_config: Dict[str, Any] = field(default_factory=dict)

    # Precision config
    precision_config: Dict[str, Any] = field(default_factory=dict)

    # Gradient config
    gradient_config: Dict[str, Any] = field(default_factory=dict)

    # Schedule config
    schedule_config: Dict[str, Any] = field(default_factory=dict)

    # Stability config
    stability_config: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    suited_for: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "TechniqueConfig":
        """Load technique config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            id=data.get("id", path.stem),
            name=data.get("name", path.stem),
            rpg_name=data.get("rpg_name", data.get("name", path.stem)),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            icon=data.get("icon", "🔧"),
            color=data.get("color", "#6B7280"),
            optimizer_type=data.get("optimizer", {}).get("type", "adamw"),
            optimizer_config=data.get("optimizer", {}),
            precision_config=data.get("precision", {}),
            gradient_config=data.get("gradients", {}),
            schedule_config=data.get("schedule", {}),
            stability_config=data.get("stability", {}),
            suited_for=data.get("recommendations", {}).get("suited_for", []),
            domains=data.get("recommendations", {}).get("domains", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "rpg_name": self.rpg_name,
            "version": self.version,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "optimizer_type": self.optimizer_type,
            "optimizer_config": self.optimizer_config,
            "precision_config": self.precision_config,
            "gradient_config": self.gradient_config,
            "schedule_config": self.schedule_config,
            "stability_config": self.stability_config,
            "suited_for": self.suited_for,
            "domains": self.domains,
        }


# =============================================================================
# TECHNIQUE REGISTRY
# =============================================================================

_technique_cache: Dict[str, TechniqueConfig] = {}


def _get_physics_dir() -> Path:
    """Get the physics config directory."""
    # Try to find via core.paths
    try:
        from core.paths import get_base_dir
        return get_base_dir() / "configs" / "physics"
    except ImportError:
        # Fallback to relative path
        return Path(__file__).parent.parent / "configs" / "physics"


def load_techniques() -> Dict[str, TechniqueConfig]:
    """Load all technique configurations from disk."""
    global _technique_cache

    if _technique_cache:
        return _technique_cache

    physics_dir = _get_physics_dir()
    if not physics_dir.exists():
        return {}

    for yaml_file in physics_dir.glob("*.yaml"):
        try:
            config = TechniqueConfig.from_yaml(yaml_file)
            _technique_cache[config.id] = config
        except Exception as e:
            print(f"Warning: Failed to load technique {yaml_file}: {e}")

    return _technique_cache


def get_technique(technique_id: str) -> Optional[TechniqueConfig]:
    """Get a specific technique by ID."""
    techniques = load_techniques()
    return techniques.get(technique_id)


def list_techniques() -> List[TechniqueConfig]:
    """List all available techniques."""
    techniques = load_techniques()
    return list(techniques.values())


def get_technique_ids() -> List[str]:
    """Get all technique IDs."""
    techniques = load_techniques()
    return list(techniques.keys())


def reload_techniques() -> Dict[str, TechniqueConfig]:
    """Force reload of techniques from disk."""
    global _technique_cache
    _technique_cache = {}
    return load_techniques()


# =============================================================================
# TECHNIQUE HELPERS
# =============================================================================

def get_optimizer_config(technique_id: str) -> Dict[str, Any]:
    """
    Get optimizer configuration for a technique.

    Returns a dict that can be passed to optimizer construction.
    """
    technique = get_technique(technique_id)
    if not technique:
        raise ValueError(f"Unknown technique: {technique_id}")

    return technique.optimizer_config


def technique_for_domain(domain_id: str) -> List[TechniqueConfig]:
    """
    Get techniques recommended for a domain.

    Returns list of techniques sorted by suitability.
    """
    techniques = load_techniques()
    matching = [
        t for t in techniques.values()
        if domain_id in t.domains
    ]
    return matching


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "--list":
            for tech in list_techniques():
                print(f"{tech.icon} {tech.id}: {tech.rpg_name}")
                print(f"   {tech.description[:60]}...")
                print()

        elif cmd == "--json":
            techniques = {t.id: t.to_dict() for t in list_techniques()}
            print(json.dumps(techniques, indent=2))

        elif cmd in get_technique_ids():
            tech = get_technique(cmd)
            if tech:
                print(f"{tech.icon} {tech.name}")
                print(f"RPG Name: {tech.rpg_name}")
                print(f"Optimizer: {tech.optimizer_type}")
                print(f"Description: {tech.description}")
                print(f"Domains: {', '.join(tech.domains)}")

        else:
            print(f"Unknown command or technique: {cmd}")
            print(f"Available: {', '.join(get_technique_ids())}")

    else:
        print("Techniques - Training Physics Configurations")
        print("=" * 50)
        print(f"Physics directory: {_get_physics_dir()}")
        print(f"Loaded: {len(list_techniques())} techniques")
        print("\nUsage:")
        print("  python3 -m trainer.techniques --list")
        print("  python3 -m trainer.techniques --json")
        print("  python3 -m trainer.techniques <technique_id>")
