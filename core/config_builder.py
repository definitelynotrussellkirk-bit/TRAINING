"""
Config Builder - Merge layered configurations for training.

The Config Builder assembles the final training configuration by merging:
1. Hero defaults (from configs/heroes/{hero_id}.yaml)
2. Campaign overrides (from campaigns/{hero_id}/{campaign_id}/campaign.json)
3. Machine settings (from configs/hosts.json for local hardware)

This layered approach allows:
- Heroes to define sensible defaults for their architecture
- Campaigns to override for specific experiments
- Machines to adjust for local hardware constraints

RPG Flavor (Pathfinder-inspired):
    The Config Builder is like the Game Master preparing a session:
    - The Hero sheet defines base stats and abilities
    - The Campaign book adds story-specific modifiers
    - The Table rules adjust for the local playing environment

Usage:
    from core.config_builder import build_config, get_active_config

    # Get config for active campaign
    config = get_active_config()

    # Build config for specific hero/campaign
    config = build_config(hero_id="titan-qwen3-4b", campaign_id="campaign-001")

    # Access merged settings
    print(config.optimizer)  # "paged_adamw_8bit"
    print(config.learning_rate)  # 0.00002
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from guild.heroes import get_hero, HeroProfile
from guild.heroes.types import LigerKernelConfig

logger = logging.getLogger("config_builder")


@dataclass
class TrainingConfig:
    """
    Final merged training configuration.

    This is what the training daemon actually uses - all layers merged.
    """
    # Identity
    hero_id: str
    campaign_id: str
    campaign_name: str

    # Model
    model_path: str
    model_family: str
    model_size_b: float

    # Precision & Quantization
    precision: str = "bf16"
    load_in_4bit: bool = False

    # Batch settings
    batch_size: int = 1
    gradient_accumulation: int = 16
    max_length: int = 2048

    # Optimizer
    optimizer: str = "adamw"  # adamw, muon, paged_adamw_8bit, adamw_bnb_8bit
    learning_rate: float = 0.0004
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = True
    liger_kernel: Optional[LigerKernelConfig] = None

    # Checkpointing
    save_steps: int = 10000
    save_total_limit: int = 40
    output_dir: str = ""

    # QLoRA (if enabled)
    qlora_enabled: bool = False
    qlora_r: int = 64
    qlora_alpha: int = 16
    qlora_dropout: float = 0.05
    qlora_target_modules: List[str] = field(default_factory=list)

    # Chat template
    chat_template: str = "default"

    # Campaign-specific
    skills_focus: List[str] = field(default_factory=list)
    starting_checkpoint: Optional[str] = None
    current_step: int = 0

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Handle LigerKernelConfig
        if self.liger_kernel:
            d['liger_kernel'] = asdict(self.liger_kernel)
        return d

    def to_hf_training_args(self) -> Dict[str, Any]:
        """Convert to HuggingFace TrainingArguments format."""
        args = {
            'output_dir': self.output_dir,
            'per_device_train_batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'lr_scheduler_type': self.lr_scheduler,
            'bf16': self.precision == 'bf16',
            'fp16': self.precision == 'fp16',
            'gradient_checkpointing': self.gradient_checkpointing,
            'save_steps': self.save_steps,
            'save_total_limit': self.save_total_limit,
            'optim': self.optimizer if self.optimizer != 'muon' else 'adamw_torch',
        }
        return args

    def to_trainer_config_dict(self) -> Dict[str, Any]:
        """
        Convert to dict format compatible with trainer.config.TrainerConfig.from_dict().

        This bridges the campaign system with the existing trainer config schema.
        Also includes extra fields (optimizer, campaign) for train.py compatibility.
        """
        # Build optimizer config (format expected by train.py)
        optimizer_config = {
            'type': self.optimizer,
        }
        # Add optimizer-specific settings based on type
        if self.optimizer == 'muon':
            optimizer_config['muon'] = {
                'hidden_lr': 0.02,
                'aux_lr': self.learning_rate,
                'momentum': 0.95,
            }
        elif self.optimizer in ('adamw', 'paged_adamw_8bit', 'adamw_bnb_8bit'):
            optimizer_config['adamw'] = {
                'lr': self.learning_rate,
                'betas': [0.9, 0.999],
                'eps': 1e-8,
                'weight_decay': 0.01,
            }

        return {
            # Hyperparams
            'hyperparams': {
                'batch_size': self.batch_size,
                'gradient_accumulation': self.gradient_accumulation,
                'learning_rate': self.learning_rate,
                'warmup_steps': self.warmup_steps,
                'max_length': self.max_length,
                'fp_precision': self.precision,
                'save_steps': self.save_steps,
                'save_total_limit': self.save_total_limit,
                'use_gradient_checkpointing': self.gradient_checkpointing,
            },
            # Profile
            'profile': {
                'name': 'emoji_think',  # Default profile
                'options': {},
            },
            # Monitoring (use defaults)
            'monitoring': {},
            # Locked config (critical architecture parameters)
            'locked': {
                'base_model': self.model_path,
                'model_architecture': 'Qwen3ForCausalLM',  # TODO: Make configurable
                'max_context_length': self.max_length,
                'vocab_size': 151936,  # Qwen3 vocab size
                'model_version': f'campaign-{self.campaign_id}',
            },
            # Data (filled in by training daemon with actual file)
            'data': {
                'dataset_path': '',  # Set by training daemon
            },
            # Model
            'model': {
                'model_path': self.model_path,
                'load_in_4bit': self.load_in_4bit,
                'gradient_checkpointing': self.gradient_checkpointing,
            },
            # Output
            'output': {
                'output_dir': self.output_dir,
                'resume_from_checkpoint': self.starting_checkpoint,
            },
            # Environment (use defaults)
            'environment': {},
            # Optimizer config (train.py expects this at top level)
            'optimizer': optimizer_config,
            # Liger kernel config (train.py may check this)
            'liger_kernel': asdict(self.liger_kernel) if self.liger_kernel else None,
            # Campaign metadata (extra fields for reference)
            'campaign': {
                'hero_id': self.hero_id,
                'campaign_id': self.campaign_id,
                'campaign_name': self.campaign_name,
                'skills_focus': self.skills_focus,
                'current_step': self.current_step,
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler,
                'liger_kernel': asdict(self.liger_kernel) if self.liger_kernel else None,
            },
        }


class ConfigBuilder:
    """
    Builds training configurations by merging layers.

    Layer precedence (later overrides earlier):
    1. Hero defaults (base)
    2. Campaign overrides
    3. Machine settings (top)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.campaigns_dir = self.base_dir / "campaigns"
        self.control_dir = self.base_dir / "control"

    def get_active_campaign(self) -> Optional[Dict]:
        """Get the currently active campaign pointer."""
        # Check symlink first
        active_link = self.campaigns_dir / "active"
        if active_link.exists() and active_link.is_symlink():
            campaign_path = active_link.resolve()
            campaign_json = campaign_path / "campaign.json"
            if campaign_json.exists():
                with open(campaign_json) as f:
                    return json.load(f)

        # Fall back to control file
        pointer_path = self.control_dir / "active_campaign.json"
        if pointer_path.exists():
            with open(pointer_path) as f:
                pointer = json.load(f)
            campaign_path = self.base_dir / pointer.get("campaign_path", "")
            campaign_json = campaign_path / "campaign.json"
            if campaign_json.exists():
                with open(campaign_json) as f:
                    return json.load(f)

        return None

    def load_campaign(self, hero_id: str, campaign_id: str) -> Dict:
        """Load a specific campaign's metadata."""
        campaign_path = self.campaigns_dir / hero_id / campaign_id / "campaign.json"
        if not campaign_path.exists():
            raise FileNotFoundError(f"Campaign not found: {campaign_path}")
        with open(campaign_path) as f:
            return json.load(f)

    def build(
        self,
        hero_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> TrainingConfig:
        """
        Build merged training configuration.

        Args:
            hero_id: Hero ID (if None, uses active campaign's hero)
            campaign_id: Campaign ID (if None, uses active campaign)

        Returns:
            TrainingConfig with all layers merged
        """
        # Get campaign metadata
        if hero_id is None or campaign_id is None:
            campaign_data = self.get_active_campaign()
            if campaign_data is None:
                raise ValueError("No active campaign and no hero_id/campaign_id provided")
            hero_id = campaign_data.get("hero_id")
            campaign_id = campaign_data.get("id")
        else:
            campaign_data = self.load_campaign(hero_id, campaign_id)

        # Load hero profile
        hero = get_hero(hero_id)

        # Start with hero defaults
        config_dict = self._hero_to_dict(hero)

        # Apply campaign overrides
        overrides = campaign_data.get("config_overrides", {})
        config_dict = self._merge(config_dict, overrides)

        # Set campaign-specific fields
        config_dict["hero_id"] = hero_id
        config_dict["campaign_id"] = campaign_id
        config_dict["campaign_name"] = campaign_data.get("name", campaign_id)
        config_dict["skills_focus"] = campaign_data.get("skills_focus", [])
        config_dict["starting_checkpoint"] = campaign_data.get("starting_checkpoint")
        config_dict["current_step"] = campaign_data.get("current_step", 0)

        # Set output directory
        config_dict["output_dir"] = str(
            self.campaigns_dir / hero_id / campaign_id / "checkpoints"
        )

        # Build TrainingConfig
        return self._build_config(config_dict, hero)

    def _hero_to_dict(self, hero: HeroProfile) -> Dict[str, Any]:
        """Extract training settings from hero profile."""
        td = hero.training_defaults
        return {
            # Model
            "model_path": hero.model.hf_name,
            "model_family": hero.model.family,
            "model_size_b": hero.model.size_b,
            # Precision
            "precision": td.precision,
            "load_in_4bit": td.load_in_4bit,
            # Batch
            "batch_size": td.batch_size,
            "gradient_accumulation": td.gradient_accumulation,
            "max_length": td.max_length,
            # Optimizer
            "optimizer": td.optimizer,
            "learning_rate": td.learning_rate,
            "warmup_steps": td.warmup_steps,
            "lr_scheduler": td.lr_scheduler,
            # Memory
            "gradient_checkpointing": td.gradient_checkpointing,
            "liger_kernel": td.liger_kernel,
            # Checkpointing
            "save_steps": td.save_steps,
            "save_total_limit": td.save_total_limit,
            # QLoRA
            "qlora_enabled": hero.qlora.enabled,
            "qlora_r": hero.qlora.r,
            "qlora_alpha": hero.qlora.alpha,
            "qlora_dropout": hero.qlora.dropout,
            "qlora_target_modules": hero.qlora.target_modules,
            # Chat
            "chat_template": hero.chat.template,
        }

    def _merge(self, base: Dict, overrides: Dict) -> Dict:
        """Merge override dict into base dict."""
        result = base.copy()
        for key, value in overrides.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge(result[key], value)
            else:
                result[key] = value
        return result

    def _build_config(self, config_dict: Dict, hero: HeroProfile) -> TrainingConfig:
        """Build TrainingConfig from merged dict."""
        return TrainingConfig(
            hero_id=config_dict["hero_id"],
            campaign_id=config_dict["campaign_id"],
            campaign_name=config_dict["campaign_name"],
            model_path=config_dict["model_path"],
            model_family=config_dict["model_family"],
            model_size_b=config_dict["model_size_b"],
            precision=config_dict.get("precision", "bf16"),
            load_in_4bit=config_dict.get("load_in_4bit", False),
            batch_size=config_dict.get("batch_size", 1),
            gradient_accumulation=config_dict.get("gradient_accumulation", 16),
            max_length=config_dict.get("max_length", 2048),
            optimizer=config_dict.get("optimizer", "adamw"),
            learning_rate=config_dict.get("learning_rate", 0.0004),
            warmup_steps=config_dict.get("warmup_steps", 100),
            lr_scheduler=config_dict.get("lr_scheduler", "cosine"),
            gradient_checkpointing=config_dict.get("gradient_checkpointing", True),
            liger_kernel=config_dict.get("liger_kernel"),
            save_steps=config_dict.get("save_steps", 10000),
            save_total_limit=config_dict.get("save_total_limit", 40),
            output_dir=config_dict.get("output_dir", ""),
            qlora_enabled=config_dict.get("qlora_enabled", False),
            qlora_r=config_dict.get("qlora_r", 64),
            qlora_alpha=config_dict.get("qlora_alpha", 16),
            qlora_dropout=config_dict.get("qlora_dropout", 0.05),
            qlora_target_modules=config_dict.get("qlora_target_modules", []),
            chat_template=config_dict.get("chat_template", "default"),
            skills_focus=config_dict.get("skills_focus", []),
            starting_checkpoint=config_dict.get("starting_checkpoint"),
            current_step=config_dict.get("current_step", 0),
        )


# Module-level functions for convenience
_builder: Optional[ConfigBuilder] = None


def get_builder(base_dir: Optional[Path] = None) -> ConfigBuilder:
    """Get the global config builder singleton."""
    global _builder
    if _builder is None:
        _builder = ConfigBuilder(base_dir)
    return _builder


def build_config(
    hero_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> TrainingConfig:
    """
    Build training config for a hero/campaign.

    Args:
        hero_id: Hero ID (if None, uses active campaign)
        campaign_id: Campaign ID (if None, uses active campaign)
        base_dir: Base directory override

    Returns:
        TrainingConfig with merged settings
    """
    return get_builder(base_dir).build(hero_id, campaign_id)


def get_active_config(base_dir: Optional[Path] = None) -> TrainingConfig:
    """
    Get training config for the active campaign.

    Returns:
        TrainingConfig for current active campaign

    Raises:
        ValueError: If no active campaign
    """
    return get_builder(base_dir).build()
