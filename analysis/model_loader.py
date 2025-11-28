"""
Efficient model loading for analysis.

Loads models in bfloat16 precision for memory efficiency while
maintaining accuracy for weight norm calculations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

import torch

logger = logging.getLogger("analysis.model_loader")


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "qwen3-0.6b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "trust_remote_code": True,
    },
    "qwen3-4b": {
        "base_model": "Qwen/Qwen3-4B",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "trust_remote_code": True,
    },
    "qwen3-4b-instruct": {
        "base_model": "Qwen/Qwen3-4B-Instruct",
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "trust_remote_code": True,
    },
}


def get_model_config(model_ref: str) -> Dict[str, Any]:
    """Get configuration for a model reference."""
    if model_ref in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_ref]

    # Try to infer from name
    model_ref_lower = model_ref.lower()
    if "qwen" in model_ref_lower:
        if "4b" in model_ref_lower:
            return MODEL_CONFIGS["qwen3-4b"]
        return MODEL_CONFIGS["qwen3-0.6b"]

    # Default config
    return {
        "base_model": model_ref,
        "model_class": "AutoModelForCausalLM",
        "tokenizer_class": "AutoTokenizer",
        "trust_remote_code": True,
    }


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_for_analysis(
    checkpoint_path: str,
    model_ref: str = "qwen3-0.6b",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """
    Load model for analysis with memory efficiency.

    Uses bf16 precision to reduce memory footprint.
    Does NOT use 8-bit quantization to ensure accurate weight norms.

    Args:
        checkpoint_path: Path to checkpoint directory
        model_ref: Model reference for config lookup
        device: Device to load on ('cuda', 'cpu')
        dtype: Data type (default: bfloat16)

    Returns:
        Loaded model in eval mode
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = get_model_config(model_ref)

    logger.info(f"Loading model from {checkpoint_path}")
    logger.info(f"  Device: {device}, dtype: {dtype}")

    # Check for config.json in checkpoint
    config_file = ckpt_path / "config.json"
    if config_file.exists():
        model_config = AutoConfig.from_pretrained(
            str(ckpt_path),
            trust_remote_code=config.get("trust_remote_code", True),
        )
    else:
        # Fall back to base model config
        model_config = AutoConfig.from_pretrained(
            config["base_model"],
            trust_remote_code=config.get("trust_remote_code", True),
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path),
        config=model_config,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=config.get("trust_remote_code", True),
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  Total params: {param_count/1e9:.2f}B")
    logger.info(f"  Trainable: {trainable_count/1e9:.2f}B")

    return model


def load_tokenizer(model_ref: str = "qwen3-0.6b"):
    """
    Load tokenizer for the model.

    Args:
        model_ref: Model reference for config lookup

    Returns:
        Tokenizer instance
    """
    from transformers import AutoTokenizer

    config = get_model_config(model_ref)

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=config.get("trust_remote_code", True),
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_reference_state_dict(
    checkpoint_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load only the state dict from a checkpoint (memory efficient).

    This is faster and uses less memory than loading the full model
    when we only need weights for drift comparison.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load tensors to (default: cpu)

    Returns:
        State dict mapping parameter names to tensors
    """
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Try different file patterns
    patterns = [
        "model.safetensors",
        "pytorch_model.bin",
        "model-00001-of-*.safetensors",
        "pytorch_model-00001-of-*.bin",
    ]

    found_files = []
    for pattern in patterns:
        matches = list(ckpt_path.glob(pattern))
        if matches:
            found_files = sorted(matches)
            break

    if not found_files:
        raise FileNotFoundError(
            f"No model files found in {checkpoint_path}. "
            f"Expected one of: {patterns}"
        )

    state_dict = {}

    for model_file in found_files:
        logger.info(f"Loading {model_file.name}...")

        if model_file.suffix == ".safetensors":
            # Use safetensors for faster loading
            try:
                from safetensors.torch import load_file
                partial = load_file(str(model_file), device=device)
            except ImportError:
                # Fall back to torch
                partial = torch.load(str(model_file), map_location=device)
        else:
            # PyTorch format
            partial = torch.load(str(model_file), map_location=device)

        state_dict.update(partial)

    logger.info(f"Loaded state dict with {len(state_dict)} parameters")
    return state_dict


def estimate_model_memory(model_ref: str, dtype: torch.dtype = torch.bfloat16) -> float:
    """
    Estimate memory required for a model.

    Args:
        model_ref: Model reference
        dtype: Data type

    Returns:
        Estimated memory in GB
    """
    # Rough parameter counts
    param_counts = {
        "qwen3-0.6b": 0.6e9,
        "qwen3-4b": 4e9,
        "qwen3-4b-instruct": 4e9,
    }

    params = param_counts.get(model_ref, 1e9)

    # Bytes per parameter
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }

    bpp = bytes_per_param.get(dtype, 2)

    # Model weights + ~30% overhead for activations during forward pass
    memory_gb = (params * bpp * 1.3) / (1024**3)

    return memory_gb


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument("checkpoint", help="Checkpoint path")
    parser.add_argument("--model-ref", default="qwen3-0.6b")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Estimated memory: {estimate_model_memory(args.model_ref):.1f} GB")

    model = load_model_for_analysis(
        args.checkpoint,
        model_ref=args.model_ref,
        device=args.device,
    )

    print(f"\nModel architecture:")
    print(model)
