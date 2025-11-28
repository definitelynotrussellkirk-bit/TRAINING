#!/usr/bin/env python3
"""
Configuration Schema for Training System

Defines the complete configuration structure using dataclasses.
Single source of truth for all training parameters.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List


def _get_default_inference_host() -> str:
    """Get default inference host from hosts.json or fallback to localhost."""
    try:
        from core.hosts import get_host
        inference_host = get_host("3090")
        if inference_host:
            return inference_host.host
    except (ImportError, Exception):
        pass
    return "localhost"


def _get_default_inference_port() -> int:
    """Get default inference port from hosts.json or fallback to 8765."""
    try:
        from core.hosts import get_host
        inference_host = get_host("3090")
        if inference_host and "inference" in inference_host.services:
            return inference_host.services["inference"].port
    except (ImportError, Exception):
        pass
    return 8765


@dataclass
class Hyperparams:
    """Core hyperparameters for training"""

    # Batch configuration
    batch_size: int = 1              # Safe default for 24GB GPU
    gradient_accumulation: int = 16  # Effective batch = 16
    effective_batch_size: int = field(init=False)

    # Learning rate
    learning_rate: float = 0.0002
    warmup_steps: int = 100

    # Training duration
    num_epochs: int = 1
    max_steps: Optional[int] = None  # If set, overrides epochs

    # Context & generation
    max_length: int = 2048           # Safe training context for 24GB
    max_new_tokens: int = 2048       # Max tokens for generation

    # Precision
    fp_precision: Literal["fp16", "bf16", "fp32"] = "bf16"  # bf16 for stability

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3        # Keep last N checkpoints

    # Evaluation
    eval_steps: int = 50
    eval_strategy: Literal["steps", "epoch", "no"] = "steps"

    # Memory optimization
    use_gradient_checkpointing: bool = True  # Essential for 24GB GPU

    def __post_init__(self):
        """Calculate derived values"""
        self.effective_batch_size = self.batch_size * self.gradient_accumulation


@dataclass
class ProfileConfig:
    """Data profile configuration"""

    name: str = "emoji_think"  # "emoji_think", "regime3", "plain_sft"

    # Profile-specific options (extensible)
    options: Dict[str, Any] = field(default_factory=dict)

    # System prompt template (can be overridden by profile)
    system_prompt_template: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""

    # Update frequencies
    status_update_interval: int = 2          # Status JSON update (steps)
    inference_interval: int = 50             # Live inference preview (steps)
    micro_eval_interval: int = 200           # Micro evaluation (steps)

    # Evaluation samples
    num_eval_samples: int = 4                # Samples per inference preview
    num_micro_eval_samples: int = 20         # Samples for micro eval

    # Fixed validation set
    validation_split: float = 0.05           # % of data for validation
    validation_max_samples: int = 500        # Max samples in validation set

    # System prompt (default from core.prompts - single source of truth)
    # Import at module level would cause circular import, so we use the literal here
    # but it should match core.prompts.BASE_PROMPT_TEMPLATE
    system_prompt_base: str = "Today is {date}. You are happy. You enjoy helping others."

    # Feature toggles
    enable_pattern_tracking: bool = True
    enable_layer_monitor: bool = False       # Expensive, disable by default
    enable_evolution_tracker: bool = True
    enable_penalty_heatmap: bool = True

    # Output control
    max_output_samples: int = 5              # Max examples to store
    prompt_snapshot_interval: int = 20       # Snapshot prompts every N steps

    # Generation limits
    max_output_tokens: int = 2048            # Max tokens for status writer output
    max_eval_tokens: int = 2048              # Max tokens for evaluation/preview

    # Preview backend configuration
    preview_backend: str = "local"           # "local" or "remote_3090"
    preview_max_tokens: int = 256            # Max tokens for live preview
    preview_temperature: float = 0.7         # Sampling temperature for preview

    # Remote 3090 backend settings (used if preview_backend == "remote_3090")
    remote_3090_host: str = field(default_factory=_get_default_inference_host)
    remote_3090_port: int = field(default_factory=_get_default_inference_port)
    remote_3090_timeout: int = 30            # Request timeout (seconds)
    remote_3090_model_id: Optional[str] = None  # Model ID on 3090 (None = use active)


@dataclass
class LockedConfig:
    """
    Fields that cannot be overridden via CLI.
    These define the fundamental architecture and must be consistent.
    """

    base_model: str
    model_architecture: str
    max_context_length: int
    vocab_size: int

    # Model identification
    model_version: str = "v1"
    created_at: str = ""


@dataclass
class DataConfig:
    """Data loading and processing configuration"""

    # Paths
    dataset_path: str = ""
    validation_dataset_path: Optional[str] = None

    # Processing
    shuffle: bool = True
    seed: int = 42

    # Filtering
    min_length: int = 10                     # Min tokens
    max_length_override: Optional[int] = None  # Override max_length if needed

    # Data augmentation (future)
    augmentation: bool = False

    # Packing (combines short sequences into full-length blocks)
    enable_packing: bool = True              # Pack sequences for efficiency
    packing_strategy: str = "bfd"            # "bfd" (Best Fit Decreasing)


@dataclass
class ModelConfig:
    """Model loading configuration"""

    # Paths
    model_path: str = ""
    tokenizer_path: Optional[str] = None     # Defaults to model_path

    # Loading options
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: str = "auto"
    torch_dtype: Optional[str] = None        # "auto", "float16", "bfloat16"

    # Model modifications
    use_cache: bool = False                  # Disable for training
    gradient_checkpointing: bool = False

    # Attention implementation
    prefer_flash_attention: bool = True      # Use flash_attention_2 if available

    # Vision model handling (Qwen3VL)
    freeze_vision_towers: bool = True        # Freeze vision/video for text-only training
    try_vision_model_first: bool = True      # Try loading as Qwen3VL first


@dataclass
class OutputConfig:
    """Output and checkpointing configuration"""

    # Directories
    output_dir: str = ""
    logging_dir: Optional[str] = None

    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    overwrite_output_dir: bool = True

    # Saving options
    save_safetensors: bool = True
    save_only_model: bool = False


@dataclass
class EnvironmentConfig:
    """Environment and resource configuration"""

    # Compute
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Distributed (future)
    local_rank: int = -1
    world_size: int = 1

    # Logging
    report_to: List[str] = field(default_factory=lambda: ["none"])
    logging_steps: int = 10

    # Safety
    max_grad_norm: float = 1.0
    nan_detection: bool = True


@dataclass
class TrainerConfig:
    """
    Complete training configuration.

    Single source of truth for all training parameters.
    Combines all sub-configurations.
    """

    # Core configurations
    hyperparams: Hyperparams
    profile: ProfileConfig
    monitoring: MonitoringConfig
    locked: LockedConfig
    data: DataConfig
    model: ModelConfig
    output: OutputConfig
    environment: EnvironmentConfig

    # Metadata
    config_version: str = "2.0"
    description: str = ""

    def __post_init__(self):
        """Validate configuration"""
        # Ensure output_dir is set
        if not self.output.output_dir:
            raise ValueError("output_dir must be set")

        # Ensure dataset_path is set
        if not self.data.dataset_path:
            raise ValueError("dataset_path must be set")

        # Ensure model_path is set
        if not self.model.model_path:
            raise ValueError("model_path must be set")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainerConfig':
        """Create from dictionary"""
        # Extract sub-configs
        hyperparams = Hyperparams(**data.get('hyperparams', {}))
        profile = ProfileConfig(**data.get('profile', {}))
        monitoring = MonitoringConfig(**data.get('monitoring', {}))
        locked = LockedConfig(**data['locked'])  # Required
        data_cfg = DataConfig(**data.get('data', {}))
        model = ModelConfig(**data.get('model', {}))
        output = OutputConfig(**data.get('output', {}))
        environment = EnvironmentConfig(**data.get('environment', {}))

        return cls(
            hyperparams=hyperparams,
            profile=profile,
            monitoring=monitoring,
            locked=locked,
            data=data_cfg,
            model=model,
            output=output,
            environment=environment,
            config_version=data.get('config_version', '2.0'),
            description=data.get('description', '')
        )


# Default configuration factory
def create_default_config(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    base_model: str,
    model_architecture: str,
    max_context_length: int,
    vocab_size: int
) -> TrainerConfig:
    """Create default configuration with required parameters"""

    locked = LockedConfig(
        base_model=base_model,
        model_architecture=model_architecture,
        max_context_length=max_context_length,
        vocab_size=vocab_size
    )

    data_cfg = DataConfig(dataset_path=dataset_path)
    model_cfg = ModelConfig(model_path=model_path)
    output_cfg = OutputConfig(output_dir=output_dir)

    return TrainerConfig(
        hyperparams=Hyperparams(),
        profile=ProfileConfig(),
        monitoring=MonitoringConfig(),
        locked=locked,
        data=data_cfg,
        model=model_cfg,
        output=output_cfg,
        environment=EnvironmentConfig()
    )


if __name__ == "__main__":
    # Example usage
    config = create_default_config(
        model_path="models/Qwen3-0.6B",
        dataset_path="data/train.jsonl",
        output_dir="outputs/run_001",
        base_model="Qwen/Qwen3-0.6B",
        model_architecture="Qwen3ForCausalLM",
        max_context_length=4096,
        vocab_size=151936
    )

    print("Default config created:")
    print(f"  Batch size: {config.hyperparams.batch_size}")
    print(f"  Learning rate: {config.hyperparams.learning_rate}")
    print(f"  Profile: {config.profile.name}")
    print(f"  Output dir: {config.output.output_dir}")
