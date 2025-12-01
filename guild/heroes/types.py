"""
Hero Types - Dataclasses for hero profiles.

These types represent the structure of hero YAML configs.
Each hero is a model architecture with training defaults and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ModelSpec:
    """
    Core model specifications - immutable facts about the architecture.

    Attributes:
        hf_name: HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")
        family: Model family (qwen3, llama, mistral, phi, gemma)
        architecture: Transformers architecture class name
        size_b: Size in billions of parameters
        vocab_size: Vocabulary size
        context_length: Maximum context window
        rope_scaling: RoPE scaling style (dynamic, linear, None)
    """
    hf_name: str
    family: str
    architecture: str
    size_b: float
    vocab_size: int
    context_length: int
    rope_scaling: Optional[str] = None


@dataclass
class LigerKernelConfig:
    """
    Liger Kernel optimization settings.

    Liger provides fused CUDA kernels for memory efficiency.
    Critical for large models (4B+) on consumer GPUs.
    """
    enabled: bool = False
    fused_linear_cross_entropy: bool = True  # Key memory saver - logits not materialized
    fused_rms_norm: bool = True
    fused_swiglu: bool = True
    fused_rope: bool = True


@dataclass
class TrainingDefaults:
    """
    Default training hyperparameters for this hero.

    These can be overridden by campaign config_overrides.

    Attributes:
        precision: Floating point precision (fp16, bf16, fp32)
        load_in_4bit: Enable 4-bit quantization
        batch_size: Micro-batch size (VRAM-bound)
        gradient_accumulation: Steps to accumulate gradients
        learning_rate: Base learning rate
        warmup_steps: LR warmup steps
        lr_scheduler: LR scheduler type (cosine, linear, constant)
        max_length: Training sequence length
        gradient_checkpointing: Enable activation checkpointing
        optimizer: Optimizer type (adamw, muon, paged_adamw_8bit, adamw_bnb_8bit)
        liger_kernel: Liger kernel optimization config
        save_steps: Checkpoint save frequency
        save_total_limit: Max checkpoints to keep
    """
    precision: str = "bf16"
    load_in_4bit: bool = False
    batch_size: int = 1
    gradient_accumulation: int = 16
    learning_rate: float = 0.0004
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    max_length: int = 2048
    gradient_checkpointing: bool = True
    optimizer: str = "adamw"  # adamw, muon, paged_adamw_8bit, adamw_bnb_8bit
    liger_kernel: Optional[LigerKernelConfig] = None
    save_steps: int = 10000
    save_total_limit: int = 40

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation


@dataclass
class QLoRAConfig:
    """
    QLoRA/LoRA adapter configuration.

    Attributes:
        enabled: Whether to use LoRA training
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout rate
        target_modules: Which modules to apply LoRA to
    """
    enabled: bool = False
    r: int = 64
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class ChatTemplate:
    """
    Chat template configuration for tokenization.

    Attributes:
        template: Template identifier or path
        system_token: System message start token
        user_token: User message start token
        assistant_token: Assistant message start token
        end_token: Message end token
        supports_system: Whether model supports system messages
    """
    template: str = "default"
    system_token: str = ""
    user_token: str = ""
    assistant_token: str = ""
    end_token: str = ""
    supports_system: bool = True


@dataclass
class VRAMProfile:
    """
    VRAM usage estimates for the memory calculator.

    All values calibrated at max_length=2048, bf16 precision.

    Attributes:
        base_memory_gb: Model weights in memory (bf16/fp16 at 2048 tokens)
        per_batch_gb: Additional memory per micro-batch sample (activations)
        optimizer_overhead_gb: Optimizer + gradient states (AdamW-style)
    """
    base_memory_gb: float = 1.0
    per_batch_gb: float = 0.5
    optimizer_overhead_gb: float = 0.5

    def estimate_breakdown(
        self,
        batch_size: int = 1,
        max_length: int = 2048,
        precision: str = "bf16",
        gradient_checkpointing: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate VRAM breakdown in GB.

        Uses the documented formula from CHANGELOG:

            activation_memory = batch_size × per_batch_gb × (max_length / 2048) × checkpoint_factor
            checkpoint_factor = 0.35 if gradient_checkpointing else 1.0

        Precision scaling (approximate):
            - bf16 / fp16: 1.0 × baseline
            - fp32:        2.0 × baseline

        Args:
            batch_size: Micro-batch size
            max_length: Maximum sequence length
            precision: Floating point precision (bf16, fp16, fp32)
            gradient_checkpointing: Whether checkpointing is enabled

        Returns:
            Dict with keys: model, optimizer, gradients, activations, total
        """
        # 1) Precision factor
        prec = (precision or "").lower()
        if prec in ("bf16", "fp16"):
            precision_factor = 1.0
        elif prec == "fp32":
            precision_factor = 2.0
        else:
            precision_factor = 1.0

        # 2) Length & checkpoint factors
        length_factor = max(max_length, 1) / 2048.0
        checkpoint_factor = 0.35 if gradient_checkpointing else 1.0

        # 3) Components
        model_mem = self.base_memory_gb * precision_factor
        optimizer_mem = self.optimizer_overhead_gb * precision_factor

        activation_total = (
            self.per_batch_gb
            * batch_size
            * length_factor
            * checkpoint_factor
            * precision_factor
        )

        # Split activations into "gradients" vs "activations" for UI
        # This is approximate - real split depends on model architecture
        gradients_mem = activation_total * 0.4
        activations_mem = activation_total * 0.6

        total = model_mem + optimizer_mem + gradients_mem + activations_mem

        return {
            "model": round(model_mem, 2),
            "optimizer": round(optimizer_mem, 2),
            "gradients": round(gradients_mem, 2),
            "activations": round(activations_mem, 2),
            "total": round(total, 2),
        }

    def estimate_total(
        self,
        batch_size: int = 1,
        max_length: int = 2048,
        precision: str = "bf16",
        gradient_checkpointing: bool = True,
    ) -> float:
        """
        Backwards-compatible helper returning total VRAM in GB.

        Older callers can continue using estimate_total(batch_size, gc)
        if they don't care about max_length/precision.

        Args:
            batch_size: Micro-batch size
            max_length: Maximum sequence length
            precision: Floating point precision (bf16, fp16, fp32)
            gradient_checkpointing: Whether checkpointing is enabled

        Returns:
            Estimated total VRAM in GB
        """
        breakdown = self.estimate_breakdown(
            batch_size=batch_size,
            max_length=max_length,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
        )
        return breakdown["total"]


@dataclass
class DisplayConfig:
    """
    UI display configuration.

    Attributes:
        portrait: Path to portrait image (256x256)
        icon: Path to icon image (32x32)
        color: Theme color (hex)
        emoji: Fallback emoji if no images
    """
    portrait: Optional[str] = None
    icon: Optional[str] = None
    color: str = "#888888"
    emoji: str = "emoji"


@dataclass
class HeroProfile:
    """
    Complete hero profile loaded from YAML config.

    A hero represents a model architecture with all its training defaults,
    chat template, and display settings.

    Usage:
        from guild.heroes import get_hero

        hero = get_hero("dio-qwen3-0.6b")
        print(hero.name)  # "DIO"
        print(hero.model.hf_name)  # "Qwen/Qwen3-0.6B"
        print(hero.training_defaults.learning_rate)  # 0.0004
    """
    # Identity
    id: str
    name: str
    rpg_name: str
    description: str

    # Model specs
    model: ModelSpec

    # Training configuration
    training_defaults: TrainingDefaults
    qlora: QLoRAConfig

    # Chat template
    chat: ChatTemplate

    # VRAM profile
    vram: VRAMProfile

    # Display
    display: DisplayConfig

    # Skills affinity
    skills_affinity: List[str] = field(default_factory=list)

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        from dataclasses import asdict
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        """Shortcut to training_defaults.effective_batch_size."""
        return self.training_defaults.effective_batch_size

    @property
    def hf_name(self) -> str:
        """Shortcut to model.hf_name."""
        return self.model.hf_name

    @property
    def size_b(self) -> float:
        """Shortcut to model.size_b."""
        return self.model.size_b

    def estimate_vram(
        self,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        precision: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Estimate VRAM breakdown using hero's VRAM profile and training defaults.

        Args are optional - if None, uses training_defaults values.

        Args:
            batch_size: Micro-batch size (default: training_defaults.batch_size)
            max_length: Maximum sequence length (default: training_defaults.max_length)
            precision: Floating point precision (default: training_defaults.precision)
            gradient_checkpointing: Whether checkpointing is enabled (default: training_defaults)

        Returns:
            Dict with keys: model, optimizer, gradients, activations, total
        """
        td = self.training_defaults
        return self.vram.estimate_breakdown(
            batch_size=batch_size if batch_size is not None else td.batch_size,
            max_length=max_length if max_length is not None else td.max_length,
            precision=precision if precision is not None else td.precision,
            gradient_checkpointing=(
                td.gradient_checkpointing if gradient_checkpointing is None
                else gradient_checkpointing
            ),
        )
