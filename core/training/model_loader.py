#!/usr/bin/env python3
"""
Model Loader - Load models with precision and attention configuration.

This module handles model loading with:
- Precision configuration (bf16/fp16/fp32)
- Flash attention detection and fallback
- 4-bit quantization support
- Qwen3VL vs CausalLM auto-detection
- Vision tower freezing for VL models

Usage:
    from training.model_loader import ModelLoader, ModelConfig

    config = ModelConfig(
        model_path="/path/to/model",
        precision="bf16",
        use_flash_attention=True
    )

    loader = ModelLoader(config)
    result = loader.load()

    model = result.model
    tokenizer = result.tokenizer
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    LogitsProcessorList,
)

# Try importing Qwen3VL (may not be available)
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    Qwen3VLForConditionalGeneration = None

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for model loading.

    Attributes:
        model_path: Path to model directory or HuggingFace model ID
        precision: Floating point precision ("bf16", "fp16", "fp32")
        use_flash_attention: Whether to try using flash attention 2
        load_in_4bit: Whether to use 4-bit quantization
        trust_remote_code: Whether to trust remote code
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        freeze_vision_towers: Whether to freeze vision/video towers (for VL models)
    """
    model_path: str
    precision: str = "bf16"
    use_flash_attention: bool = True
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    use_gradient_checkpointing: bool = False
    freeze_vision_towers: bool = True


@dataclass
class LoadedModel:
    """
    Result of model loading.

    Attributes:
        model: The loaded model
        tokenizer: The loaded tokenizer
        processor: Optional processor (for VL models)
        is_vision_model: Whether this is a vision-language model
        attention_impl: Attention implementation used ("flash_attention_2" or "sdpa")
        precision: Actual precision used
        logits_processor: Optional logits processor list
    """
    model: Any
    tokenizer: Any
    processor: Optional[Any] = None
    is_vision_model: bool = False
    attention_impl: str = "sdpa"
    precision: str = "bf16"
    logits_processor: Optional[LogitsProcessorList] = None


class ModelLoader:
    """
    Loads models with configurable precision and attention.

    Handles:
    - Precision setup (bf16/fp16/fp32)
    - Flash attention detection and fallback to SDPA
    - 4-bit quantization via BitsAndBytes
    - Qwen3VL vs standard CausalLM detection
    - Vision tower freezing for text-only training

    Example:
        config = ModelConfig(
            model_path="Qwen/Qwen2.5-0.5B",
            precision="bf16",
            use_flash_attention=True
        )

        loader = ModelLoader(config)
        result = loader.load()

        # Access loaded components
        model = result.model
        tokenizer = result.tokenizer
        print(f"Loaded with {result.attention_impl}, {result.precision}")
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model loader.

        Args:
            config: Model configuration
        """
        self.config = config

    def load(self) -> LoadedModel:
        """
        Load model and tokenizer with configured settings.

        Returns:
            LoadedModel with model, tokenizer, and metadata

        Raises:
            RuntimeError: If model loading fails
        """
        model_path = self.config.model_path

        # Detect attention implementation
        attn_impl = self._detect_attention_impl()
        logger.info(f"Using attention: {attn_impl}")

        # Build model kwargs
        model_kwargs = self._build_model_kwargs(attn_impl)

        # Try loading model
        model, tokenizer, processor, is_vision = self._load_model_and_tokenizer(
            model_path, model_kwargs
        )

        # Configure model for training
        self._configure_for_training(model, is_vision)

        # Get actual precision used
        precision = self._get_precision_name(model_kwargs.get("torch_dtype"))

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            is_vision_model=is_vision,
            attention_impl=attn_impl,
            precision=precision
        )

    def _detect_attention_impl(self) -> str:
        """Detect best available attention implementation."""
        if not self.config.use_flash_attention:
            return "sdpa"

        try:
            import flash_attn
            logger.info("Flash Attention 2 detected")
            return "flash_attention_2"
        except ImportError:
            logger.info("Flash Attention not installed, using SDPA")
            return "sdpa"

    def _build_model_kwargs(self, attn_impl: str) -> dict:
        """Build kwargs for model loading."""
        kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "attn_implementation": attn_impl
        }

        # Quantization config (for QLoRA)
        if self.config.load_in_4bit:
            logger.info("Enabling 4-bit quantization")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            # Set precision
            kwargs["torch_dtype"] = self._get_torch_dtype()

        return kwargs

    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from precision string."""
        precision_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = precision_map.get(self.config.precision, torch.bfloat16)
        logger.info(f"Using {self.config.precision} precision")
        return dtype

    def _get_precision_name(self, dtype: Optional[torch.dtype]) -> str:
        """Get precision name from torch dtype."""
        if dtype is None:
            return "4bit"  # Quantized
        dtype_map = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }
        return dtype_map.get(dtype, "unknown")

    def _load_model_and_tokenizer(
        self, model_path: str, model_kwargs: dict
    ) -> Tuple[Any, Any, Optional[Any], bool]:
        """
        Load model and tokenizer, auto-detecting model type.

        Returns:
            Tuple of (model, tokenizer, processor, is_vision_model)
        """
        is_vision = False
        processor = None

        # Try Qwen3VL first (for vision-language models)
        if QWEN3VL_AVAILABLE:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, **model_kwargs
                )
                logger.info("Loaded as Qwen3VLForConditionalGeneration")
                is_vision = True

                # Load processor for VL models
                processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=self.config.trust_remote_code
                )
                tokenizer = processor.tokenizer
                logger.info("Loaded AutoProcessor for VL model")

                return model, tokenizer, processor, is_vision

            except Exception as e:
                logger.debug(f"Qwen3VL failed: {e}, trying CausalLM")

        # Fallback to standard CausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        logger.info("Loaded as AutoModelForCausalLM")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=self.config.trust_remote_code
        )

        return model, tokenizer, None, False

    def _configure_for_training(self, model: Any, is_vision: bool) -> None:
        """Configure model for training."""
        # Set pad token if not set
        # (Done on tokenizer in caller, not here since we return tokenizer)

        # Disable KV cache for training
        model.config.use_cache = False
        logger.info("Disabled KV cache")

        # Enable gradient checkpointing if configured
        if self.config.use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            else:
                logger.warning("Model doesn't support gradient checkpointing")

        # Freeze vision towers for VL models (for text-only training)
        if is_vision and self.config.freeze_vision_towers:
            frozen = self._freeze_vision_towers(model)
            logger.info(f"Froze {frozen} vision/video parameters")

    def _freeze_vision_towers(self, model: Any) -> int:
        """Freeze vision and video tower parameters."""
        frozen_count = 0
        vision_keywords = ["vision_model", "video_model", "visual"]

        for name, param in model.named_parameters():
            if any(kw in name for kw in vision_keywords):
                param.requires_grad = False
                frozen_count += 1

        return frozen_count


if __name__ == "__main__":
    # Quick test (requires actual model)
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test config creation
    config = ModelConfig(
        model_path="Qwen/Qwen2.5-0.5B",
        precision="bf16",
        use_flash_attention=True
    )
    print(f"Config: {config}")

    # Test without actual model loading
    loader = ModelLoader(config)
    print(f"Attention impl: {loader._detect_attention_impl()}")
    print(f"Torch dtype: {loader._get_torch_dtype()}")

    print("\nModelLoader ready for use!")
