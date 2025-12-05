"""
Config Validator - Pre-validate config.json
=============================================

Catches common configuration mistakes BEFORE training starts:
- Invalid paths (model doesn't exist)
- Incompatible settings (bf16 on non-Ampere GPU)
- Out of range values (batch_size too large)
- Missing required fields
- Typos in field names

Usage:
    from temple.diagnostics import ConfigValidator

    validator = ConfigValidator()
    report = validator.validate("config.json")

    if not report.is_valid:
        print("Config has problems:")
        for error in report.errors:
            print(f"  ❌ {error}")
        for warning in report.warnings:
            print(f"  ⚠️ {warning}")
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of config validation."""
    is_valid: bool = True  # No errors (warnings OK)
    errors: List[str] = field(default_factory=list)  # Blocking issues
    warnings: List[str] = field(default_factory=list)  # Non-blocking concerns
    suggestions: List[str] = field(default_factory=list)  # Helpful tips
    fixed_config: Optional[Dict[str, Any]] = None  # Auto-fixed config if requested

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_suggestion(self, msg: str) -> None:
        self.suggestions.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []

        if self.is_valid:
            lines.append("✅ Config is valid!")
        else:
            lines.append("❌ Config has errors:")

        for error in self.errors:
            lines.append(f"  ❌ {error}")

        for warning in self.warnings:
            lines.append(f"  ⚠️ {warning}")

        for suggestion in self.suggestions[:3]:
            lines.append(f"  💡 {suggestion}")

        return "\n".join(lines)


class ConfigValidator:
    """
    Validates training configuration files.

    Checks:
    1. Required fields present
    2. Paths exist
    3. Values in valid ranges
    4. Type correctness
    5. Compatibility checks (e.g., bf16 requires Ampere+)
    6. Typo detection
    """

    # Known valid field names for typo detection
    KNOWN_FIELDS = {
        # Top level
        "model_name", "base_model", "training_mode", "dataset",
        "max_length", "batch_size", "gradient_accumulation_steps",
        "num_epochs", "max_steps", "learning_rate", "lr",
        "warmup_steps", "warmup_ratio", "weight_decay",
        "max_grad_norm", "gradient_checkpointing", "fp16", "bf16",
        "seed", "logging_steps", "save_steps", "eval_steps",
        "output_dir", "save_total_limit", "load_best_model_at_end",
        "resume_from_checkpoint", "starting_checkpoint",

        # Optimizer
        "optimizer", "type", "muon", "hidden_lr", "aux_lr", "lr_scheduler",

        # LoRA/QLoRA
        "lora", "r", "lora_alpha", "lora_dropout", "target_modules",
        "load_in_4bit", "load_in_8bit", "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_type", "use_nested_quant",

        # DeepSpeed
        "deepspeed", "deepspeed_config",

        # Environment
        "environment", "device", "device_map", "attn_implementation",
        "trust_remote_code", "torch_dtype",

        # Data
        "data", "train_file", "eval_file", "validation_split_percentage",
        "max_train_samples", "max_eval_samples", "preprocessing_num_workers",
        "dataloader_num_workers",

        # Misc
        "campaign_id", "hero", "skill", "curriculum",
    }

    # Valid values for certain fields
    VALID_VALUES = {
        "training_mode": {"full", "lora", "qlora"},
        "optimizer.type": {"adamw", "adamw_8bit", "galore", "galore_8bit", "muon", "sgd"},
        "torch_dtype": {"float32", "float16", "bfloat16", "auto"},
        "bnb_4bit_quant_type": {"nf4", "fp4"},
        "attn_implementation": {"eager", "sdpa", "flash_attention_2"},
    }

    # Required fields
    REQUIRED_FIELDS = {
        "base_model",  # or model_name
    }

    # Value ranges
    VALUE_RANGES = {
        "learning_rate": (1e-8, 1.0),
        "lr": (1e-8, 1.0),
        "batch_size": (1, 256),
        "gradient_accumulation_steps": (1, 256),
        "max_length": (1, 131072),
        "max_grad_norm": (0.0, 100.0),
        "warmup_ratio": (0.0, 1.0),
        "weight_decay": (0.0, 1.0),
        "lora.r": (1, 256),
        "lora.lora_alpha": (1, 512),
        "lora.lora_dropout": (0.0, 1.0),
        "num_epochs": (1, 100),
    }

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def validate(
        self,
        config_path: str = "config.json",
        auto_fix: bool = False,
    ) -> ValidationResult:
        """
        Validate a configuration file.

        Args:
            config_path: Path to config.json
            auto_fix: If True, include a fixed config in result

        Returns:
            ValidationResult with errors, warnings, suggestions
        """
        result = ValidationResult()

        path = Path(config_path)
        if not path.is_absolute():
            path = self.base_dir / path

        # Check file exists
        if not path.exists():
            result.add_error(f"Config file not found: {path}")
            return result

        # Load and parse JSON
        try:
            with open(path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON: {e}")
            return result
        except Exception as e:
            result.add_error(f"Error reading config: {e}")
            return result

        # Run all checks
        self._check_required_fields(config, result)
        self._check_typos(config, result)
        self._check_paths(config, result)
        self._check_value_ranges(config, result)
        self._check_valid_values(config, result)
        self._check_compatibility(config, result)
        self._check_common_mistakes(config, result)

        # Generate suggestions
        self._generate_suggestions(config, result)

        return result

    def _check_required_fields(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check that required fields are present."""
        # Need either base_model or model_name
        if "base_model" not in config and "model_name" not in config:
            result.add_error("Missing required field: 'base_model' or 'model_name'")

    def _check_typos(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
        prefix: str = "",
    ) -> None:
        """Check for possible typos in field names."""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Check if this looks like a typo
            if key not in self.KNOWN_FIELDS:
                matches = get_close_matches(key, self.KNOWN_FIELDS, n=1, cutoff=0.8)
                if matches:
                    result.add_warning(f"Possible typo: '{key}' - did you mean '{matches[0]}'?")

            # Recurse into nested dicts
            if isinstance(value, dict):
                self._check_typos(value, result, prefix=full_key)

    def _check_paths(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check that paths exist."""
        path_fields = [
            "base_model",
            "model_name",
            "dataset",
            "data.train_file",
            "data.eval_file",
            "resume_from_checkpoint",
            "starting_checkpoint",
            "output_dir",
            "deepspeed_config",
            "environment.deepspeed_config",
        ]

        for field in path_fields:
            value = self._get_nested(config, field)
            if value is None:
                continue

            # Skip if it looks like a HuggingFace model ID
            if "/" in str(value) and not str(value).startswith("/"):
                # Likely a HF model ID like "Qwen/Qwen3-0.6B"
                continue

            path = Path(value)
            if not path.is_absolute():
                path = self.base_dir / path

            # Check existence
            if field in ("output_dir",):
                # Output dir doesn't need to exist
                continue
            elif field.endswith("_checkpoint"):
                # Checkpoints optional
                if value and not path.exists():
                    result.add_warning(f"Checkpoint path doesn't exist: {value}")
            elif not path.exists():
                # Check models directory too
                alt_path = self.base_dir / "models" / value
                if not alt_path.exists():
                    result.add_error(f"Path doesn't exist: {value}")

    def _check_value_ranges(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check that values are in valid ranges."""
        for field, (min_val, max_val) in self.VALUE_RANGES.items():
            value = self._get_nested(config, field)
            if value is None:
                continue

            if not isinstance(value, (int, float)):
                continue

            if value < min_val:
                result.add_error(f"{field}={value} is below minimum ({min_val})")
            elif value > max_val:
                result.add_error(f"{field}={value} is above maximum ({max_val})")

        # Special checks
        lr = config.get("learning_rate") or config.get("lr")
        if lr and lr > 0.01:
            result.add_warning(f"Learning rate {lr} seems very high - typical is 1e-5 to 1e-4")

        batch_size = config.get("batch_size")
        if batch_size and batch_size > 64:
            result.add_warning(f"Batch size {batch_size} is large - ensure enough VRAM")

    def _check_valid_values(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check that values are from valid sets."""
        for field, valid_values in self.VALID_VALUES.items():
            value = self._get_nested(config, field)
            if value is None:
                continue

            if value not in valid_values:
                result.add_error(
                    f"{field}='{value}' is invalid. Valid options: {sorted(valid_values)}"
                )

    def _check_compatibility(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check for incompatible settings."""

        # bf16 requires Ampere or newer
        if config.get("bf16") or config.get("torch_dtype") == "bfloat16":
            try:
                import torch
                if torch.cuda.is_available():
                    cap = torch.cuda.get_device_capability()
                    if cap[0] < 8:  # Pre-Ampere
                        result.add_warning(
                            f"bf16 requires Ampere+ GPU (compute capability 8.0+), "
                            f"but detected {cap[0]}.{cap[1]}"
                        )
            except:
                pass

        # flash_attention_2 has requirements
        attn_impl = self._get_nested(config, "environment.attn_implementation")
        if attn_impl == "flash_attention_2":
            try:
                import flash_attn
            except ImportError:
                result.add_warning("flash_attention_2 requires flash-attn package to be installed")

        # QLora requires 4bit loading
        if config.get("training_mode") == "qlora":
            if not config.get("load_in_4bit"):
                result.add_error("training_mode='qlora' requires load_in_4bit=true")

        # Lora requires lora config
        if config.get("training_mode") in ("lora", "qlora"):
            if "lora" not in config:
                result.add_warning("training_mode='lora' without 'lora' config section")

        # DeepSpeed with certain optimizers
        if config.get("deepspeed_config") or config.get("environment", {}).get("deepspeed_config"):
            opt_type = self._get_nested(config, "optimizer.type")
            if opt_type in ("galore", "galore_8bit", "muon"):
                result.add_warning(f"DeepSpeed with {opt_type} optimizer may have compatibility issues")

        # 8-bit optimizers require bitsandbytes
        opt_type = self._get_nested(config, "optimizer.type")
        if opt_type and "8bit" in opt_type:
            try:
                import bitsandbytes
            except ImportError:
                result.add_error(f"{opt_type} requires bitsandbytes package")

    def _check_common_mistakes(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check for common configuration mistakes."""

        # max_steps and num_epochs both set
        if config.get("max_steps") and config.get("num_epochs"):
            result.add_warning(
                "Both max_steps and num_epochs set - max_steps takes precedence"
            )

        # Gradient checkpointing without large model
        if config.get("gradient_checkpointing"):
            result.add_suggestion(
                "gradient_checkpointing is enabled - this saves VRAM but slows training ~20%"
            )

        # Very small max_length
        max_length = config.get("max_length")
        if max_length and max_length < 128:
            result.add_warning(f"max_length={max_length} is very short for most tasks")

        # Eval steps but no eval file
        if config.get("eval_steps"):
            eval_file = self._get_nested(config, "data.eval_file")
            if not eval_file:
                result.add_warning("eval_steps set but no eval_file specified")

        # Resume from checkpoint that doesn't match model
        resume = config.get("resume_from_checkpoint")
        model = config.get("base_model") or config.get("model_name")
        if resume and model:
            # Check if checkpoint is for same model
            pass  # Would need to inspect checkpoint

    def _generate_suggestions(
        self,
        config: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Generate helpful suggestions based on config."""

        # Suggest gradient checkpointing for large batches
        batch_size = config.get("batch_size", 1)
        grad_acc = config.get("gradient_accumulation_steps", 1)
        effective_batch = batch_size * grad_acc

        if effective_batch > 32 and not config.get("gradient_checkpointing"):
            result.add_suggestion(
                f"With effective batch size {effective_batch}, consider gradient_checkpointing=true"
            )

        # Suggest save frequency
        save_steps = config.get("save_steps")
        max_steps = config.get("max_steps", 10000)
        if save_steps and max_steps:
            checkpoints = max_steps // save_steps
            if checkpoints > 50:
                result.add_suggestion(
                    f"This will create ~{checkpoints} checkpoints. Consider increasing save_steps"
                )
            elif checkpoints < 3:
                result.add_suggestion(
                    f"Only ~{checkpoints} checkpoints will be saved. Consider decreasing save_steps"
                )

        # Learning rate scheduler
        if not config.get("lr_scheduler"):
            result.add_suggestion(
                "No lr_scheduler specified - consider 'cosine' or 'linear' for better convergence"
            )

    def _get_nested(self, config: Dict[str, Any], key: str) -> Any:
        """Get a nested key like 'optimizer.type'."""
        parts = key.split(".")
        value = config
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def validate_and_print(self, config_path: str = "config.json") -> bool:
        """Validate and print results. Returns True if valid."""
        result = self.validate(config_path)
        print(result.summary())
        return result.is_valid


# ========== CLI ==========

def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate training config")
    parser.add_argument("config", nargs="?", default="config.json", help="Config file path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    validator = ConfigValidator()
    result = validator.validate(args.config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    exit(main())
