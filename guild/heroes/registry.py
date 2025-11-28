"""
Hero Registry - Load and manage hero profiles from YAML configs.

The registry provides a single point of access to all hero definitions.
Heroes are loaded lazily and cached for performance.

Usage:
    from guild.heroes import get_hero, list_heroes, HeroRegistry

    # List available heroes
    heroes = list_heroes()  # ['dio-qwen3-0.6b', 'muon-0.6b']

    # Get hero profile
    hero = get_hero('dio-qwen3-0.6b')
    print(hero.model.hf_name)  # "Qwen/Qwen3-0.6B"

    # Or use registry directly
    registry = HeroRegistry(base_dir)
    hero = registry.get("dio-qwen3-0.6b")

RPG Flavor:
    The Registry is the Guild's tome of champions - every hero who has
    walked these halls is recorded here. Their strengths, their training
    regimens, their very essence captured in ancient scrolls (YAML files).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .types import (
    HeroProfile,
    ModelSpec,
    TrainingDefaults,
    LigerKernelConfig,
    QLoRAConfig,
    ChatTemplate,
    VRAMProfile,
    DisplayConfig,
)

logger = logging.getLogger("hero_registry")


class HeroNotFoundError(Exception):
    """Raised when a hero ID is not found in the registry."""
    pass


class HeroConfigError(Exception):
    """Raised when a hero config file is invalid."""
    pass


class HeroRegistry:
    """
    Registry of all available heroes.

    Loads hero profiles from YAML configs in configs/heroes/.
    Caches loaded profiles for performance.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            base_dir: Base directory containing configs/heroes/.
                      If None, auto-detects from file location.
        """
        if base_dir is None:
            # Auto-detect: guild/heroes/registry.py -> TRAINING/
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = Path(base_dir)
        self.heroes_dir = self.base_dir / "configs" / "heroes"
        self._cache: Dict[str, HeroProfile] = {}

    def list(self) -> List[str]:
        """
        List all available hero IDs.

        Returns:
            List of hero IDs (filenames without .yaml extension)
        """
        if not self.heroes_dir.exists():
            logger.warning(f"Heroes directory not found: {self.heroes_dir}")
            return []

        return [
            f.stem for f in self.heroes_dir.glob("*.yaml")
            if not f.name.startswith("_")  # Skip templates
        ]

    def get(self, hero_id: str) -> HeroProfile:
        """
        Get a hero profile by ID.

        Args:
            hero_id: Hero identifier (e.g., "dio-qwen3-0.6b")

        Returns:
            HeroProfile instance

        Raises:
            HeroNotFoundError: If hero ID not found
            HeroConfigError: If config file is invalid
        """
        if hero_id in self._cache:
            return self._cache[hero_id]

        profile = self._load(hero_id)
        self._cache[hero_id] = profile
        return profile

    def _load(self, hero_id: str) -> HeroProfile:
        """Load a hero profile from YAML."""
        config_path = self.heroes_dir / f"{hero_id}.yaml"

        if not config_path.exists():
            available = self.list()
            raise HeroNotFoundError(
                f"Hero '{hero_id}' not found. Available: {available}"
            )

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise HeroConfigError(f"Invalid YAML in {config_path}: {e}")

        return self._parse(data, hero_id)

    def _parse(self, data: Dict, hero_id: str) -> HeroProfile:
        """Parse YAML data into HeroProfile."""
        try:
            # Parse model spec
            model_data = data.get("model", {})
            model = ModelSpec(
                hf_name=model_data.get("hf_name", ""),
                family=model_data.get("family", "unknown"),
                architecture=model_data.get("architecture", ""),
                size_b=float(model_data.get("size_b", 0.0)),
                vocab_size=int(model_data.get("vocab_size", 0)),
                context_length=int(model_data.get("context_length", 4096)),
                rope_scaling=model_data.get("rope_scaling"),
            )

            # Parse training defaults
            train_data = data.get("training_defaults", {})

            # Parse liger kernel config if present
            liger_data = train_data.get("liger_kernel", {})
            liger_kernel = None
            if liger_data and liger_data.get("enabled", False):
                liger_kernel = LigerKernelConfig(
                    enabled=True,
                    fused_linear_cross_entropy=liger_data.get("fused_linear_cross_entropy", True),
                    fused_rms_norm=liger_data.get("fused_rms_norm", True),
                    fused_swiglu=liger_data.get("fused_swiglu", True),
                    fused_rope=liger_data.get("fused_rope", True),
                )

            training_defaults = TrainingDefaults(
                precision=train_data.get("precision", "bf16"),
                load_in_4bit=train_data.get("load_in_4bit", False),
                batch_size=int(train_data.get("batch_size", 1)),
                gradient_accumulation=int(train_data.get("gradient_accumulation", 16)),
                learning_rate=float(train_data.get("learning_rate", 0.0004)),
                warmup_steps=int(train_data.get("warmup_steps", 100)),
                lr_scheduler=train_data.get("lr_scheduler", "cosine"),
                max_length=int(train_data.get("max_length", 2048)),
                gradient_checkpointing=train_data.get("gradient_checkpointing", True),
                optimizer=train_data.get("optimizer", "adamw"),
                liger_kernel=liger_kernel,
                save_steps=int(train_data.get("save_steps", 10000)),
                save_total_limit=int(train_data.get("save_total_limit", 40)),
            )

            # Parse QLoRA config
            qlora_data = data.get("qlora", {})
            qlora = QLoRAConfig(
                enabled=qlora_data.get("enabled", False),
                r=int(qlora_data.get("r", 64)),
                alpha=int(qlora_data.get("alpha", 16)),
                dropout=float(qlora_data.get("dropout", 0.05)),
                target_modules=qlora_data.get("target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj"
                ]),
            )

            # Parse chat template
            chat_data = data.get("chat", {})
            chat = ChatTemplate(
                template=chat_data.get("template", "default"),
                system_token=chat_data.get("system_token", ""),
                user_token=chat_data.get("user_token", ""),
                assistant_token=chat_data.get("assistant_token", ""),
                end_token=chat_data.get("end_token", ""),
                supports_system=chat_data.get("supports_system", True),
            )

            # Parse VRAM profile
            vram_data = data.get("vram", {})
            vram = VRAMProfile(
                base_memory_gb=float(vram_data.get("base_memory_gb", 1.0)),
                per_batch_gb=float(vram_data.get("per_batch_gb", 0.5)),
                optimizer_overhead_gb=float(vram_data.get("optimizer_overhead_gb", 0.5)),
            )

            # Parse display config
            display_data = data.get("display", {})
            display = DisplayConfig(
                portrait=display_data.get("portrait"),
                icon=display_data.get("icon"),
                color=display_data.get("color", "#888888"),
                emoji=display_data.get("emoji", "emoji"),
            )

            # Build HeroProfile
            return HeroProfile(
                id=data.get("id", hero_id),
                name=data.get("name", hero_id),
                rpg_name=data.get("rpg_name", "The Unknown"),
                description=data.get("description", ""),
                model=model,
                training_defaults=training_defaults,
                qlora=qlora,
                chat=chat,
                vram=vram,
                display=display,
                skills_affinity=data.get("skills_affinity", []),
                notes=data.get("notes", ""),
            )

        except (KeyError, TypeError, ValueError) as e:
            raise HeroConfigError(f"Error parsing hero config: {e}")

    def reload(self, hero_id: Optional[str] = None) -> None:
        """
        Reload hero profile(s) from disk.

        Args:
            hero_id: Specific hero to reload, or None to clear all cache
        """
        if hero_id:
            self._cache.pop(hero_id, None)
        else:
            self._cache.clear()


# Module-level singleton
_registry: Optional[HeroRegistry] = None


def get_registry(base_dir: Optional[Path] = None) -> HeroRegistry:
    """Get the global hero registry singleton."""
    global _registry
    if _registry is None:
        _registry = HeroRegistry(base_dir)
    return _registry


def list_heroes(base_dir: Optional[Path] = None) -> List[str]:
    """
    List all available hero IDs.

    Args:
        base_dir: Optional base directory override

    Returns:
        List of hero IDs
    """
    return get_registry(base_dir).list()


def get_hero(hero_id: str, base_dir: Optional[Path] = None) -> HeroProfile:
    """
    Get a hero profile by ID.

    Args:
        hero_id: Hero identifier (e.g., "dio-qwen3-0.6b")
        base_dir: Optional base directory override

    Returns:
        HeroProfile instance

    Raises:
        HeroNotFoundError: If hero ID not found
    """
    return get_registry(base_dir).get(hero_id)
