#!/usr/bin/env python3
"""
Configuration Loader

Loads and merges configuration from multiple sources:
1. Campaign system (hero + campaign overrides) - NEW
2. config.json (base)
3. CLI arguments (overrides)
4. Defaults (fallback)

Handles precedence and validation.
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .schema import (
    TrainerConfig,
    Hyperparams,
    ProfileConfig,
    MonitoringConfig,
    LockedConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    EnvironmentConfig,
)

# Campaign system integration
try:
    from core.config_builder import get_active_config as get_campaign_config, ConfigBuilder
    CAMPAIGN_SYSTEM_AVAILABLE = True
except ImportError:
    CAMPAIGN_SYSTEM_AVAILABLE = False
    get_campaign_config = None
    ConfigBuilder = None


class ConfigLoader:
    """Loads and merges training configuration"""

    @staticmethod
    def from_json_file(config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def from_campaign(use_campaign: bool = True) -> Optional[Dict[str, Any]]:
        """
        Try to load configuration from the campaign system.

        The campaign system provides hero defaults + campaign overrides,
        giving a cleaner separation than config.json.

        Args:
            use_campaign: If False, skip campaign system entirely

        Returns:
            Config dict compatible with TrainerConfig.from_dict(), or None if unavailable
        """
        if not use_campaign or not CAMPAIGN_SYSTEM_AVAILABLE:
            return None

        try:
            campaign_config = get_campaign_config()
            config_dict = campaign_config.to_trainer_config_dict()
            print(f"✓ Using campaign config: {campaign_config.hero_id}/{campaign_config.campaign_id}")
            return config_dict
        except Exception as e:
            # No active campaign or error - fall back to config.json
            print(f"ℹ️  Campaign system unavailable ({e}), using config.json")
            return None

    @staticmethod
    def from_args_and_json(
        args: argparse.Namespace,
        base_config_path: Optional[Path] = None,
        validate_lock: bool = True,
        use_campaign: bool = True
    ) -> TrainerConfig:
        """
        Create configuration from CLI args and config.json

        Precedence (highest to lowest):
        1. CLI arguments (--batch-size, etc.)
        2. Campaign system (if active) - NEW
        3. config.json
        4. Schema defaults

        Args:
            args: Parsed CLI arguments
            base_config_path: Path to config.json (default: config.json)
            validate_lock: If True, validate against .config_lock.json
            use_campaign: If True, try campaign system before config.json

        Returns:
            TrainerConfig instance
        """
        # Try campaign system first
        base_config = ConfigLoader.from_campaign(use_campaign)

        # Fall back to config.json if no campaign
        if base_config is None:
            if base_config_path is None:
                base_config_path = Path("config.json")

            base_config = {}
            if base_config_path.exists():
                base_config = ConfigLoader.from_json_file(base_config_path)

        # Build configuration with precedence
        config_dict = ConfigLoader._merge_config(base_config, args)

        # Create TrainerConfig
        config = TrainerConfig.from_dict(config_dict)

        # Validate locked config
        if validate_lock:
            ConfigLoader.validate_locked_config(config, strict=True)

        return config

    @staticmethod
    def from_args_and_json_with_raw(
        args: argparse.Namespace,
        base_config_path: Optional[Path] = None,
        validate_lock: bool = True,
        use_campaign: bool = True
    ) -> tuple:
        """
        Create configuration and return both TrainerConfig and raw dict.

        The raw dict includes extra fields not in TrainerConfig schema:
        - optimizer: Optimizer type and settings
        - liger_kernel: Liger kernel configuration
        - campaign: Campaign metadata (hero_id, campaign_id, etc.)

        Args:
            args: Parsed CLI arguments
            base_config_path: Path to config.json (default: config.json)
            validate_lock: If True, validate against .config_lock.json
            use_campaign: If True, try campaign system before config.json

        Returns:
            Tuple of (TrainerConfig, raw_config_dict)
        """
        # Try campaign system first
        base_config = ConfigLoader.from_campaign(use_campaign)

        # Fall back to config.json if no campaign
        if base_config is None:
            if base_config_path is None:
                base_config_path = Path("config.json")

            base_config = {}
            if base_config_path.exists():
                base_config = ConfigLoader.from_json_file(base_config_path)

        # Build configuration with precedence
        config_dict = ConfigLoader._merge_config(base_config, args)

        # Create TrainerConfig
        config = TrainerConfig.from_dict(config_dict)

        # Validate locked config
        if validate_lock:
            ConfigLoader.validate_locked_config(config, strict=True)

        # Return both TrainerConfig and raw dict (for extra fields)
        return config, config_dict

    @staticmethod
    def _merge_config(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        """
        Merge base config with CLI args

        CLI args override base config where provided.
        """
        merged = base.copy()

        # Helper to update nested dict
        def set_nested(d: dict, path: list, value):
            """Set value in nested dict via path list"""
            for key in path[:-1]:
                d = d.setdefault(key, {})
            d[path[-1]] = value

        # Map CLI args to config paths
        # Format: (arg_name, config_path)
        arg_mappings = [
            # Data paths
            ('dataset', ['data', 'dataset_path']),
            ('output_dir', ['output', 'output_dir']),
            ('model', ['model', 'model_path']),

            # Hyperparams
            ('batch_size', ['hyperparams', 'batch_size']),
            ('learning_rate', ['hyperparams', 'learning_rate']),
            ('warmup_steps', ['hyperparams', 'warmup_steps']),
            ('epochs', ['hyperparams', 'num_epochs']),
            ('max_steps', ['hyperparams', 'max_steps']),
            ('save_steps', ['hyperparams', 'save_steps']),
            ('eval_steps', ['hyperparams', 'eval_steps']),
            ('max_length', ['hyperparams', 'max_length']),

            # Profile
            ('profile', ['profile', 'name']),

            # Monitoring
            ('num_eval_samples', ['monitoring', 'num_eval_samples']),

            # Precision
            ('fp16', ['hyperparams', 'fp_precision']),  # Special handling
            ('bf16', ['hyperparams', 'fp_precision']),  # Special handling

            # Output
            ('resume_from_checkpoint', ['output', 'resume_from_checkpoint']),
        ]

        # Apply CLI overrides
        for arg_name, config_path in arg_mappings:
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    # Special handling for precision flags
                    if arg_name == 'fp16' and value:
                        set_nested(merged, config_path, 'fp16')
                    elif arg_name == 'bf16' and value:
                        set_nested(merged, config_path, 'bf16')
                    elif arg_name not in ['fp16', 'bf16']:
                        set_nested(merged, config_path, value)

        # Ensure required nested structures exist
        merged.setdefault('hyperparams', {})
        merged.setdefault('profile', {})
        merged.setdefault('monitoring', {})
        merged.setdefault('data', {})
        merged.setdefault('model', {})
        merged.setdefault('output', {})
        merged.setdefault('environment', {})

        # Ensure locked config exists (required)
        if 'locked' not in merged:
            # Use build_locked_config for consistent locked config generation
            # In config.json path (no hero), we use fallback values
            from .locked import build_locked_config
            model_path = merged.get('model', {}).get('model_path', '')
            max_length = merged.get('hyperparams', {}).get('max_length', 4096)
            merged['locked'] = build_locked_config(
                hero=None,  # No hero in config.json-only path
                model_path=base.get('base_model', model_path),
                max_length=max_length,
                model_version='v1',
            )
            merged['locked']['created_at'] = datetime.now().isoformat()

        return merged

    @staticmethod
    def from_file_and_defaults(
        dataset_path: str,
        base_config: str = "config.json",
        validate_lock: bool = True,
        **overrides
    ) -> TrainerConfig:
        """
        Create config from file with specific dataset and overrides.

        Used by daemon when processing queue files.

        Args:
            dataset_path: Path to training dataset
            base_config: Path to base config.json
            validate_lock: If True, validate against .config_lock.json
            **overrides: Additional overrides (e.g., output_dir="outputs/run_001")

        Returns:
            TrainerConfig instance
        """
        # Load base
        base = {}
        if Path(base_config).exists():
            base = ConfigLoader.from_json_file(Path(base_config))

        # Apply overrides
        base.setdefault('data', {})['dataset_path'] = dataset_path

        for key, value in overrides.items():
            # Simple dot-notation support: "output.output_dir" -> nested
            keys = key.split('.')
            d = base
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        # Create config
        config = TrainerConfig.from_dict(base)

        # Validate locked config
        if validate_lock:
            ConfigLoader.validate_locked_config(config, strict=True)

        return config

    @staticmethod
    def validate_locked_config(
        config: TrainerConfig,
        strict: bool = True,
        lock_file: Path = Path(".config_lock.json")
    ):
        """
        Validate locked configuration hasn't been changed across training runs.

        On first run: Creates lock file from current config.locked
        On subsequent runs: Compares current config against lock file

        Args:
            config: TrainerConfig to validate
            strict: If True, raise on mismatch. If False, just warn.
            lock_file: Path to lock file (default: .config_lock.json in cwd)

        Raises:
            ValueError: If locked fields changed and strict=True
        """
        # Build current locked snapshot
        current = {
            "base_model": config.locked.base_model,
            "model_architecture": config.locked.model_architecture,
            "max_context_length": config.locked.max_context_length,
            "vocab_size": config.locked.vocab_size,
            "model_version": config.locked.model_version,
        }

        # Validate fields are present
        for key, value in current.items():
            if not value or (isinstance(value, int) and value <= 0):
                raise ValueError(f"Locked config invalid: {key}={value}")

        # If lock file doesn't exist, create it (first run)
        if not lock_file.exists():
            lock_data = {
                **current,
                "locked_at": datetime.now().isoformat(),
                "note": "DO NOT MODIFY - This file locks critical model architecture parameters"
            }
            lock_file.write_text(json.dumps(lock_data, indent=2))
            print(f"✓ Created lock file: {lock_file}")
            print(f"  Locked: base_model={current['base_model']}, "
                  f"max_length={current['max_context_length']}")
            return

        # Load existing lock
        try:
            with open(lock_file, 'r') as f:
                locked = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read lock file {lock_file}: {e}")

        # Compare current vs locked
        diffs = []
        for key in current.keys():
            if key in locked and locked[key] != current[key]:
                diffs.append(
                    f"{key}: locked={locked[key]}, current={current[key]}"
                )

        # Handle mismatches
        if diffs:
            msg = (
                f"❌ Locked configuration changed!\n"
                f"   Lock file: {lock_file}\n"
                f"   Changes:\n"
            )
            for diff in diffs:
                msg += f"     - {diff}\n"
            msg += "\n"
            msg += "   These parameters CANNOT be changed during training:\n"
            msg += "     - base_model (changing breaks checkpoint compatibility)\n"
            msg += "     - max_context_length (requires model re-initialization)\n"
            msg += "     - vocab_size (incompatible tokenizer)\n"
            msg += "\n"
            msg += "   To proceed:\n"
            msg += f"     1. Delete {lock_file} to start fresh (loses checkpoint compatibility)\n"
            msg += "     2. Revert config.json to match lock file\n"

            if strict:
                raise ValueError(msg)
            else:
                print(f"⚠️  WARNING: {msg}")

        # Success - config matches lock
        print(f"✓ Locked config validated: {lock_file}")
        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Training System - Configurable LLM Training"
    )

    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to training dataset (JSONL)'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='Model name or path (e.g., qwen3_0.6b or /path/to/model)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=False,
        help='Output directory for checkpoints'
    )

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, help='Warmup steps')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--max-steps', type=int, help='Max training steps')
    parser.add_argument('--save-steps', type=int, help='Save checkpoint every N steps')
    parser.add_argument('--eval-steps', type=int, help='Evaluate every N steps')
    parser.add_argument('--max-length', type=int, help='Max sequence length')

    # Precision
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fp16', action='store_true', help='Use FP16 training')
    group.add_argument('--bf16', action='store_true', help='Use BF16 training')

    # Profile
    parser.add_argument(
        '--profile',
        type=str,
        choices=['emoji_think', 'regime3'],
        help='Data profile to use'
    )

    # Monitoring
    parser.add_argument('--num-eval-samples', type=int, help='Number of evaluation samples')

    # Checkpointing
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        help='Resume from checkpoint path'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Base config file (default: config.json)'
    )

    return parser.parse_args()
