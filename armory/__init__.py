"""
Armory - Equipment and configurations for training.

The Armory contains all the equipment (configurations, profiles) that
the hero needs for training battles:

    Equipment   - Training profiles (emoji_think, regime3)
    Smithy      - Configuration loading and merging
    BattleGear  - Hyperparameter sets

RPG Mapping:
    trainer/profiles/     → Equipment (battle loadouts)
    trainer/config/       → Smithy (forge configurations)
    trainer/core/engine   → BattleEngine

Quick Start:
    from armory import get_equipment, load_battle_config

    # Get a training profile
    equipment = get_equipment("emoji_think")
    transformed = equipment.transform_example(example)

    # Load configuration
    config = load_battle_config("config.json", cli_args)
"""

__version__ = "0.1.0"

# Re-export from trainer for convenience
try:
    from trainer.profiles.base import DataProfile
    from trainer.profiles.emoji_think import EmojiThinkProfile
    from trainer.config.loader import ConfigLoader
    from trainer.config.schema import (
        TrainerConfig,
        Hyperparams,
        ProfileConfig,
    )

    # Equipment aliases
    Equipment = DataProfile
    EmojiEquipment = EmojiThinkProfile

    # Smithy alias
    Smithy = ConfigLoader

    def get_equipment(profile_name: str) -> DataProfile:
        """
        Get equipment (training profile) by name.

        Args:
            profile_name: "emoji_think", "regime3", etc.

        Returns:
            DataProfile instance
        """
        if profile_name == "emoji_think":
            return EmojiThinkProfile()
        elif profile_name == "regime3":
            from trainer.profiles.regime3 import Regime3Profile
            return Regime3Profile()
        else:
            raise ValueError(f"Unknown equipment: {profile_name}")

    def load_battle_config(config_path: str, cli_args: dict = None) -> TrainerConfig:
        """
        Load and merge battle configuration.

        Args:
            config_path: Path to config.json
            cli_args: CLI argument overrides

        Returns:
            TrainerConfig
        """
        loader = ConfigLoader(config_path)
        return loader.load(cli_args or {})

    __all__ = [
        # Equipment
        "Equipment",
        "EmojiEquipment",
        "DataProfile",
        "EmojiThinkProfile",
        "get_equipment",
        # Smithy
        "Smithy",
        "ConfigLoader",
        "load_battle_config",
        # Config types
        "TrainerConfig",
        "Hyperparams",
        "ProfileConfig",
    ]

except ImportError:
    # trainer module not available
    __all__ = []


# =============================================================================
# RPG TERMINOLOGY GUIDE
# =============================================================================

"""
ARMORY GLOSSARY
===============

The Armory uses RPG terminology for training configuration:

EQUIPMENT (Profiles)
-------------------
Equipment       = Training profile (data transformation)
EmojiEquipment  = emoji_think profile (thinking tokens)
Regime3Equipment = regime3 profile (symbolic reasoning)
Loadout         = Active profile configuration

SMITHY (Config)
--------------
Smithy          = Configuration loader
Forge           = Merge configs from multiple sources
BattleGear      = Hyperparameters
Blueprint       = Config schema/template

CONFIGURATION
-------------
BattleConfig    = TrainerConfig (full training config)
Hyperparams     = Learning rate, batch size, etc.
ProfileConfig   = Profile-specific settings

FORGE OPERATIONS
----------------
load_battle_config  = Load and merge configuration
get_equipment       = Get profile by name
"""
