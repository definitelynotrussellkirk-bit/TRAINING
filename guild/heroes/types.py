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

    All values calibrated at max_length=2048, bf16 precision, full fine-tuning.

    Attributes:
        base_memory_gb: Model weights in memory (bf16/fp16 at 2048 tokens)
        per_batch_gb: Additional memory per micro-batch sample (activations)
        optimizer_overhead_gb: Optimizer + gradient states (AdamW-style full FT)
    """
    base_memory_gb: float = 1.0
    per_batch_gb: float = 0.5
    optimizer_overhead_gb: float = 0.5

    # Optimizer memory multipliers relative to full AdamW (optimizer_overhead_gb)
    # These are approximate based on empirical measurements
    OPTIMIZER_FACTORS = {
        "adamw": 1.0,           # Full 32-bit Adam states
        "adamw_8bit": 0.25,     # ~4x reduction from 8-bit quantization
        "galore": 0.25,         # ~4x reduction from low-rank projection
        "galore_8bit": 0.0625,  # ~16x reduction (GaLore + 8-bit combined)
        "muon": 0.5,            # Momentum only (no second moment)
    }

    # Training mode factors
    # LoRA/QLoRA train ~1-5% of params, dramatically reducing optimizer states
    TRAINING_MODE_FACTORS = {
        "full": {"model": 1.0, "optimizer": 1.0, "gradients": 1.0},
        "lora": {"model": 1.0, "optimizer": 0.05, "gradients": 0.05},  # ~5% params trained
        "qlora": {"model": 0.5, "optimizer": 0.03, "gradients": 0.03},  # 4-bit model + LoRA
    }

    def estimate_breakdown(
        self,
        batch_size: int = 1,
        max_length: int = 2048,
        precision: str = "bf16",
        gradient_checkpointing: bool = True,
        training_mode: str = "full",
        optimizer_type: str = "adamw",
        deepspeed_stage: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Estimate VRAM breakdown in GB.

        Uses the documented formula from CHANGELOG with extensions for:
        - training_mode: full, lora, qlora
        - optimizer_type: adamw, adamw_8bit, galore, galore_8bit, muon
        - deepspeed_stage: None, 2, or 3

        Memory formula:
            activation_memory = batch_size × per_batch_gb × (max_length / 2048) × checkpoint_factor
            checkpoint_factor = 0.35 if gradient_checkpointing else 1.0

        Precision scaling:
            - bf16 / fp16: 1.0 × baseline
            - fp32:        2.0 × baseline

        Training mode scaling:
            - full: Full model + optimizer memory
            - lora: Full model, ~5% optimizer/gradients
            - qlora: ~50% model (4-bit), ~3% optimizer/gradients

        Optimizer scaling (relative to AdamW):
            - adamw: 1.0x (baseline)
            - adamw_8bit: 0.25x
            - galore: 0.25x
            - galore_8bit: 0.0625x (~16x reduction)
            - muon: 0.5x

        DeepSpeed ZeRO:
            - Stage 2: Optimizer offloaded to CPU (0% GPU optimizer)
            - Stage 3: Everything offloaded (minimal GPU, ~30% model)

        Args:
            batch_size: Micro-batch size
            max_length: Maximum sequence length
            precision: Floating point precision (bf16, fp16, fp32)
            gradient_checkpointing: Whether checkpointing is enabled
            training_mode: Training mode (full, lora, qlora)
            optimizer_type: Optimizer type (adamw, adamw_8bit, galore, galore_8bit, muon)
            deepspeed_stage: DeepSpeed ZeRO stage (None, 2, or 3)

        Returns:
            Dict with keys: model, optimizer, gradients, activations, total, mode_info
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

        # 3) Training mode factors
        mode_factors = self.TRAINING_MODE_FACTORS.get(
            training_mode, self.TRAINING_MODE_FACTORS["full"]
        )

        # 4) Optimizer type factor
        opt_factor = self.OPTIMIZER_FACTORS.get(optimizer_type, 1.0)

        # 5) DeepSpeed factors
        ds_model_factor = 1.0
        ds_optimizer_factor = 1.0
        ds_gradient_factor = 1.0

        if deepspeed_stage == 2:
            # ZeRO-2: Optimizer states offloaded to CPU
            ds_optimizer_factor = 0.0
        elif deepspeed_stage == 3:
            # ZeRO-3: Everything partitioned/offloaded
            ds_model_factor = 0.3  # Only active shard on GPU
            ds_optimizer_factor = 0.0
            ds_gradient_factor = 0.3

        # 6) Calculate components
        model_mem = (
            self.base_memory_gb
            * precision_factor
            * mode_factors["model"]
            * ds_model_factor
        )

        optimizer_mem = (
            self.optimizer_overhead_gb
            * precision_factor
            * mode_factors["optimizer"]
            * opt_factor
            * ds_optimizer_factor
        )

        activation_total = (
            self.per_batch_gb
            * batch_size
            * length_factor
            * checkpoint_factor
            * precision_factor
        )

        # Gradients scale with training mode (LoRA trains fewer params)
        gradients_mem = (
            activation_total
            * 0.4
            * mode_factors["gradients"]
            * ds_gradient_factor
        )
        activations_mem = activation_total * 0.6

        total = model_mem + optimizer_mem + gradients_mem + activations_mem

        # Build mode info string for UI
        mode_info = []
        if training_mode != "full":
            mode_info.append(training_mode.upper())
        if optimizer_type != "adamw":
            mode_info.append(optimizer_type)
        if deepspeed_stage:
            mode_info.append(f"ZeRO-{deepspeed_stage}")

        return {
            "model": round(model_mem, 2),
            "optimizer": round(optimizer_mem, 2),
            "gradients": round(gradients_mem, 2),
            "activations": round(activations_mem, 2),
            "total": round(total, 2),
            "mode_info": " + ".join(mode_info) if mode_info else "Full FT",
        }

    def estimate_total(
        self,
        batch_size: int = 1,
        max_length: int = 2048,
        precision: str = "bf16",
        gradient_checkpointing: bool = True,
        training_mode: str = "full",
        optimizer_type: str = "adamw",
        deepspeed_stage: Optional[int] = None,
    ) -> float:
        """
        Helper returning total VRAM in GB.

        Args:
            batch_size: Micro-batch size
            max_length: Maximum sequence length
            precision: Floating point precision (bf16, fp16, fp32)
            gradient_checkpointing: Whether checkpointing is enabled
            training_mode: Training mode (full, lora, qlora)
            optimizer_type: Optimizer type (adamw, adamw_8bit, galore, galore_8bit, muon)
            deepspeed_stage: DeepSpeed ZeRO stage (None, 2, or 3)

        Returns:
            Estimated total VRAM in GB
        """
        breakdown = self.estimate_breakdown(
            batch_size=batch_size,
            max_length=max_length,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
            training_mode=training_mode,
            optimizer_type=optimizer_type,
            deepspeed_stage=deepspeed_stage,
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
        training_mode: Optional[str] = None,
        optimizer_type: Optional[str] = None,
        deepspeed_stage: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Estimate VRAM breakdown using hero's VRAM profile and training defaults.

        Args are optional - if None, uses training_defaults values.

        Args:
            batch_size: Micro-batch size (default: training_defaults.batch_size)
            max_length: Maximum sequence length (default: training_defaults.max_length)
            precision: Floating point precision (default: training_defaults.precision)
            gradient_checkpointing: Whether checkpointing is enabled (default: training_defaults)
            training_mode: Training mode - full, lora, qlora (default: "full")
            optimizer_type: Optimizer type - adamw, adamw_8bit, galore, galore_8bit, muon
            deepspeed_stage: DeepSpeed ZeRO stage (None, 2, or 3)

        Returns:
            Dict with keys: model, optimizer, gradients, activations, total, mode_info
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
            training_mode=training_mode if training_mode is not None else "full",
            optimizer_type=optimizer_type if optimizer_type is not None else td.optimizer,
            deepspeed_stage=deepspeed_stage,
        )
