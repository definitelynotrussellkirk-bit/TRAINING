"""
Memory Profile System - Configurable VRAM optimization for training.

Memory profiles define how to optimize training for specific VRAM budgets.
They bundle together:
- PEFT method (QLoRA, LoRA, GaLore, or full training)
- Quantization settings
- Memory-efficient implementations (gradient checkpointing, flash attention)
- Optimizer selection (paged, 8-bit, etc.)

Usage:
    from trainer.config.memory_profiles import (
        get_profile,
        list_profiles,
        suggest_profile,
        MemoryProfile,
    )

    # Load a profile
    profile = get_profile("24gb_qlora")

    # Suggest based on model size and VRAM
    profile = suggest_profile(model_size_b=8.0, vram_gb=24)

    # Apply to training config
    training_args = profile.get_training_args()
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import logging

logger = logging.getLogger(__name__)


class PEFTMethod(Enum):
    """Parameter-Efficient Fine-Tuning methods."""
    NONE = "none"          # Full training
    LORA = "lora"          # Low-Rank Adaptation
    QLORA = "qlora"        # Quantized LoRA
    GALORE = "galore"      # Gradient Low-Rank Projection


class OptimizerType(Enum):
    """Available optimizer types."""
    ADAMW_FUSED = "adamw_torch_fused"      # Fast, high memory
    ADAMW_8BIT = "adamw_8bit"              # 8-bit Adam (bitsandbytes)
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"  # Paged 32-bit (CPU offload)
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"    # Paged 8-bit (max efficiency)
    MUON = "muon"                           # Geometry-aware optimizer
    GALORE_ADAMW = "galore_adamw"           # AdamW with GaLore
    GALORE_ADAMW_8BIT = "galore_adamw_8bit" # 8-bit with GaLore


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class GaLoreConfig:
    """GaLore projection configuration."""
    rank: int = 1024
    update_proj_gap: int = 200
    scale: float = 0.25
    proj_type: str = "std"


@dataclass
class QuantizationConfig:
    """Quantization settings for model loading."""
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class MemoryConfig:
    """Memory-efficient implementation settings."""
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    paged_optimizer: bool = False
    cpu_offload: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: OptimizerType = OptimizerType.ADAMW_FUSED
    lr: float = 5e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Muon-specific
    hidden_lr: float = 0.02
    aux_lr: float = 3e-4
    momentum: float = 0.95


@dataclass
class TrainingConstraints:
    """Recommended training limits for this profile."""
    max_batch_size: int = 1
    recommended_grad_accum: int = 8
    max_sequence_length: int = 1024
    precision: str = "bf16"


@dataclass
class MemoryProfile:
    """
    Complete memory optimization profile.

    Bundles all settings needed to train within a VRAM budget.
    """
    id: str
    name: str
    rpg_name: str = ""
    icon: str = "🧠"
    description: str = ""
    target_vram_gb: int = 24

    # PEFT configuration
    peft_method: PEFTMethod = PEFTMethod.NONE
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    galore: GaLoreConfig = field(default_factory=GaLoreConfig)

    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Memory implementations
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Constraints
    constraints: TrainingConstraints = field(default_factory=TrainingConstraints)

    # Recommendations
    max_model_size_b: float = 8.0
    optimal_model_sizes: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "MemoryProfile":
        """Load profile from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Parse PEFT section
        peft_data = data.get("peft", {})
        peft_method = PEFTMethod(peft_data.get("method", "none"))

        lora_data = peft_data.get("lora", {})
        lora = LoRAConfig(
            r=lora_data.get("r", 64),
            alpha=lora_data.get("alpha", 128),
            dropout=lora_data.get("dropout", 0.05),
            target_modules=lora_data.get("target_modules", LoRAConfig().target_modules),
            bias=lora_data.get("bias", "none"),
            task_type=lora_data.get("task_type", "CAUSAL_LM"),
        )

        galore_data = peft_data.get("galore", {})
        galore = GaLoreConfig(
            rank=galore_data.get("rank", 1024),
            update_proj_gap=galore_data.get("update_proj_gap", 200),
            scale=galore_data.get("scale", 0.25),
            proj_type=galore_data.get("proj_type", "std"),
        )

        # Parse quantization
        quant_data = data.get("quantization", {})
        quantization = QuantizationConfig(
            load_in_4bit=quant_data.get("load_in_4bit", False),
            load_in_8bit=quant_data.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=quant_data.get("bnb_4bit_compute_dtype", "bfloat16"),
            bnb_4bit_quant_type=quant_data.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_data.get("bnb_4bit_use_double_quant", True),
        )

        # Parse memory config
        mem_data = data.get("memory", {})
        memory = MemoryConfig(
            gradient_checkpointing=mem_data.get("gradient_checkpointing", True),
            flash_attention=mem_data.get("flash_attention", True),
            paged_optimizer=mem_data.get("paged_optimizer", False),
            cpu_offload=mem_data.get("cpu_offload", False),
        )

        # Parse optimizer config
        opt_data = data.get("optimizer", {})
        adamw_data = opt_data.get("adamw", {})
        muon_data = opt_data.get("muon", {})

        opt_type_str = opt_data.get("type", "adamw_torch_fused")
        try:
            opt_type = OptimizerType(opt_type_str)
        except ValueError:
            logger.warning(f"Unknown optimizer type: {opt_type_str}, defaulting to adamw_torch_fused")
            opt_type = OptimizerType.ADAMW_FUSED

        optimizer = OptimizerConfig(
            type=opt_type,
            lr=adamw_data.get("lr", 5e-5),
            betas=tuple(adamw_data.get("betas", [0.9, 0.999])),
            eps=adamw_data.get("eps", 1e-8),
            weight_decay=adamw_data.get("weight_decay", 0.01),
            hidden_lr=muon_data.get("hidden_lr", 0.02),
            aux_lr=muon_data.get("aux_lr", 3e-4),
            momentum=muon_data.get("momentum", 0.95),
        )

        # Parse constraints
        const_data = data.get("constraints", {})
        constraints = TrainingConstraints(
            max_batch_size=const_data.get("max_batch_size", 1),
            recommended_grad_accum=const_data.get("recommended_grad_accum", 8),
            max_sequence_length=const_data.get("max_sequence_length", 1024),
            precision=const_data.get("precision", "bf16"),
        )

        # Parse recommendations
        rec_data = data.get("recommendations", {})

        return cls(
            id=data.get("id", path.stem),
            name=data.get("name", path.stem),
            rpg_name=data.get("rpg_name", ""),
            icon=data.get("icon", "🧠"),
            description=data.get("description", ""),
            target_vram_gb=data.get("target_vram_gb", 24),
            peft_method=peft_method,
            lora=lora,
            galore=galore,
            quantization=quantization,
            memory=memory,
            optimizer=optimizer,
            constraints=constraints,
            max_model_size_b=rec_data.get("max_model_size_b", 8.0),
            optimal_model_sizes=rec_data.get("optimal_model_sizes", []),
            notes=rec_data.get("notes", ""),
        )

    def get_optimizer_name(self) -> str:
        """Get the HuggingFace optimizer name."""
        return self.optimizer.type.value

    def get_training_mode(self) -> str:
        """Get training mode string (full, lora, qlora)."""
        if self.peft_method == PEFTMethod.QLORA:
            return "qlora"
        elif self.peft_method == PEFTMethod.LORA:
            return "lora"
        elif self.peft_method == PEFTMethod.GALORE:
            return "galore"
        else:
            return "full"

    def should_use_paged_optimizer(self) -> bool:
        """Check if paged optimizer should be used."""
        return self.memory.paged_optimizer or self.optimizer.type in (
            OptimizerType.PAGED_ADAMW_32BIT,
            OptimizerType.PAGED_ADAMW_8BIT,
        )

    def to_training_args_overrides(self) -> Dict[str, Any]:
        """
        Get dictionary of TrainingArguments overrides.

        Returns settings that should override HuggingFace TrainingArguments.
        """
        return {
            "optim": self.get_optimizer_name(),
            "per_device_train_batch_size": self.constraints.max_batch_size,
            "gradient_accumulation_steps": self.constraints.recommended_grad_accum,
            "gradient_checkpointing": self.memory.gradient_checkpointing,
            "bf16": self.constraints.precision == "bf16",
            "fp16": self.constraints.precision == "fp16",
            "learning_rate": self.optimizer.lr,
            "weight_decay": self.optimizer.weight_decay,
            "adam_beta1": self.optimizer.betas[0],
            "adam_beta2": self.optimizer.betas[1],
            "adam_epsilon": self.optimizer.eps,
        }

    def __str__(self) -> str:
        return f"{self.icon} {self.name} ({self.id})"


# Module-level cache
_profiles_cache: Dict[str, MemoryProfile] = {}


def _get_profiles_dir() -> Path:
    """Get the memory profiles directory."""
    from core.paths import get_base_dir
    return get_base_dir() / "configs" / "memory_profiles"


def list_profiles() -> List[str]:
    """List available memory profile IDs."""
    profiles_dir = _get_profiles_dir()
    if not profiles_dir.exists():
        return []
    return [p.stem for p in profiles_dir.glob("*.yaml")]


def get_profile(profile_id: str) -> MemoryProfile:
    """
    Load a memory profile by ID.

    Args:
        profile_id: Profile ID (e.g., "24gb_qlora")

    Returns:
        MemoryProfile instance

    Raises:
        FileNotFoundError: If profile doesn't exist
    """
    if profile_id in _profiles_cache:
        return _profiles_cache[profile_id]

    profiles_dir = _get_profiles_dir()
    profile_path = profiles_dir / f"{profile_id}.yaml"

    if not profile_path.exists():
        available = list_profiles()
        raise FileNotFoundError(
            f"Memory profile '{profile_id}' not found. "
            f"Available: {available}"
        )

    profile = MemoryProfile.from_yaml(profile_path)
    _profiles_cache[profile_id] = profile
    return profile


def suggest_profile(
    model_size_b: float,
    vram_gb: int = 24,
    prefer_full_training: bool = False,
) -> MemoryProfile:
    """
    Suggest the best memory profile for a given model size and VRAM.

    Args:
        model_size_b: Model size in billions of parameters
        vram_gb: Available VRAM in GB
        prefer_full_training: If True, prefer full training over PEFT

    Returns:
        Recommended MemoryProfile
    """
    if vram_gb >= 48:
        # High VRAM - can do full training for most models
        if model_size_b <= 8:
            return get_profile("24gb_full_small")  # Overkill but works
        else:
            return get_profile("24gb_qlora")  # Still need QLoRA for 13B+

    elif vram_gb >= 24:
        if model_size_b <= 4 and prefer_full_training:
            return get_profile("24gb_full_small")
        elif model_size_b <= 8:
            return get_profile("24gb_qlora")
        else:
            # Model too large for 24GB even with QLoRA
            logger.warning(
                f"Model size {model_size_b}B may not fit in {vram_gb}GB VRAM. "
                f"Consider using DeepSpeed ZeRO-3 or multi-GPU."
            )
            return get_profile("24gb_qlora")

    else:
        # Low VRAM - always use QLoRA with aggressive settings
        logger.warning(
            f"VRAM {vram_gb}GB is limited. Using aggressive memory settings."
        )
        return get_profile("24gb_qlora")


def get_all_profiles() -> Dict[str, MemoryProfile]:
    """Load all available memory profiles."""
    profiles = {}
    for profile_id in list_profiles():
        try:
            profiles[profile_id] = get_profile(profile_id)
        except Exception as e:
            logger.error(f"Failed to load profile {profile_id}: {e}")
    return profiles


# Quick reference for optimizer selection
OPTIMIZER_MEMORY_RANKING = {
    # Lowest memory first
    OptimizerType.PAGED_ADAMW_8BIT: 1,
    OptimizerType.GALORE_ADAMW_8BIT: 2,
    OptimizerType.PAGED_ADAMW_32BIT: 3,
    OptimizerType.GALORE_ADAMW: 4,
    OptimizerType.ADAMW_8BIT: 5,
    OptimizerType.MUON: 6,
    OptimizerType.ADAMW_FUSED: 7,  # Highest memory
}


def get_optimizer_for_vram(vram_gb: int, model_size_b: float) -> OptimizerType:
    """
    Suggest optimizer based on VRAM budget and model size.

    Args:
        vram_gb: Available VRAM
        model_size_b: Model size in billions

    Returns:
        Recommended OptimizerType
    """
    if vram_gb < 16:
        return OptimizerType.PAGED_ADAMW_8BIT
    elif vram_gb < 24:
        return OptimizerType.PAGED_ADAMW_32BIT
    elif vram_gb < 48:
        if model_size_b > 4:
            return OptimizerType.PAGED_ADAMW_32BIT
        else:
            return OptimizerType.ADAMW_FUSED
    else:
        return OptimizerType.ADAMW_FUSED
