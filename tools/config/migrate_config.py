#!/usr/bin/env python3
"""
Config Migration Script

Migrates old flat config.json to new nested TrainerConfig format.

Usage:
    python3 migrate_config.py [--config config.json] [--backup] [--dry-run]

What it does:
    1. Reads existing config.json (flat or hybrid format)
    2. Converts to nested TrainerConfig structure
    3. Creates backup of old config (if --backup)
    4. Writes new nested config.json
    5. Creates .config_lock.json for locked fields
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add trainer module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trainer.config import TrainerConfig, ConfigLoader


def migrate_flat_to_nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat config to nested TrainerConfig structure

    Handles hybrid configs that have some nested fields already.
    """
    # Start with any existing nested structure
    nested = {}

    # Copy over existing nested sections
    for section in ['hyperparams', 'profile', 'monitoring', 'locked', 'model',
                    'data', 'output', 'environment']:
        if section in flat:
            nested[section] = flat[section].copy()

    # Initialize sections if they don't exist
    nested.setdefault('hyperparams', {})
    nested.setdefault('model', {})
    nested.setdefault('output', {})
    nested.setdefault('monitoring', {})
    nested.setdefault('profile', {})
    nested.setdefault('data', {})
    nested.setdefault('environment', {})

    # Set defaults for required fields that daemon doesn't use
    # (Daemon sets these per-job, but config needs valid defaults)
    if 'dataset_path' not in nested['data'] or not nested['data']['dataset_path']:
        nested['data']['dataset_path'] = ""  # Empty string = not set, daemon will override

    if 'output_dir' not in nested['output'] or not nested['output']['output_dir']:
        # Use current_model_dir if it exists, otherwise fallback
        nested['output']['output_dir'] = flat.get('current_model_dir', 'models/current_model')

    # Map flat fields to nested structure
    mappings = [
        # Model paths
        ('model_path', ['model', 'model_path']),
        ('base_model', ['model', 'model_path']),  # Fallback
        ('current_model_dir', ['output', 'output_dir']),

        # Hyperparams
        ('batch_size', ['hyperparams', 'batch_size']),
        ('gradient_accumulation', ['hyperparams', 'gradient_accumulation']),
        ('learning_rate', ['hyperparams', 'learning_rate']),
        ('warmup_steps', ['hyperparams', 'warmup_steps']),
        ('epochs', ['hyperparams', 'num_epochs']),
        ('save_steps', ['hyperparams', 'save_steps']),
        ('eval_steps', ['hyperparams', 'eval_steps']),
        ('max_length', ['hyperparams', 'max_length']),

        # Monitoring
        ('num_eval_samples', ['monitoring', 'num_eval_samples']),
        ('eval_max_tokens', ['monitoring', 'max_eval_tokens']),
        ('eval_timeout_seconds', ['monitoring', 'remote_3090_timeout']),
    ]

    # Apply mappings
    for old_key, new_path in mappings:
        if old_key in flat and flat[old_key] is not None:
            # Navigate to nested location
            target = nested
            for key in new_path[:-1]:
                target = target[key]
            # Only set if not already set
            if new_path[-1] not in target or target[new_path[-1]] is None:
                target[new_path[-1]] = flat[old_key]

    # Handle locked config (special case)
    if 'locked' not in nested or not nested['locked']:
        # Build from flat fields
        nested['locked'] = {
            'base_model': flat.get('base_model') or flat.get('model_path', ''),
            'model_architecture': flat.get('model_architecture', 'Qwen3ForCausalLM'),
            'max_context_length': flat.get('max_length', 4096),
            'vocab_size': flat.get('vocab_size', 151936),
            'model_version': 'v1',
            'created_at': datetime.now().isoformat()
        }

    # Ensure locked.created_at exists
    if 'created_at' not in nested['locked']:
        nested['locked']['created_at'] = datetime.now().isoformat()

    # Preserve daemon-specific fields as top-level (outside TrainerConfig tree)
    daemon_fields = [
        'poll_interval',
        'snapshot_time',
        'self_correction',
        'auto_generate',
        'log_steps',
        'model_name',  # Daemon uses this for display
        'model_display_name',  # Daemon uses this for display
    ]

    for field in daemon_fields:
        if field in flat:
            nested[field] = flat[field]

    # Preserve comments
    for key in flat:
        if key.startswith('_comment'):
            nested[key] = flat[key]

    return nested


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate config.json to nested format")
    parser.add_argument('--config', default='config.json', help='Path to config.json')
    parser.add_argument('--backup', action='store_true', help='Create backup before migrating')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--force', action='store_true', help='Overwrite even if already nested')

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    # Load existing config
    print(f"📖 Reading config: {config_path}")
    with open(config_path, 'r') as f:
        old_config = json.load(f)

    # Check if already migrated
    is_flat = all(
        key not in old_config or not isinstance(old_config.get(key), dict)
        for key in ['hyperparams', 'model', 'output']
    )

    if not is_flat and not args.force:
        print("✓ Config already appears to be in nested format")
        print("  Use --force to re-migrate anyway")

        # Still validate it can be loaded
        try:
            config = TrainerConfig.from_dict(old_config)
            print(f"✓ Config validates successfully")
            print(f"  Model: {config.model.model_path}")
            print(f"  Batch size: {config.hyperparams.batch_size}")
            print(f"  Max length: {config.hyperparams.max_length}")
            return 0
        except Exception as e:
            print(f"⚠️  WARNING: Config validation failed: {e}")
            print("  Consider using --force to re-migrate")
            return 1

    # Migrate to nested format
    print("🔄 Migrating to nested format...")
    nested_config = migrate_flat_to_nested(old_config)

    # Validate it can be loaded as TrainerConfig
    try:
        config = TrainerConfig.from_dict(nested_config)
        print("✓ Migration successful, config validates")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1

    # Show what will be done
    print("\n📋 Migration Summary:")
    print(f"  Model: {config.model.model_path}")
    print(f"  Output: {config.output.output_dir}")
    print(f"  Batch size: {config.hyperparams.batch_size}")
    print(f"  Learning rate: {config.hyperparams.learning_rate}")
    print(f"  Max length: {config.hyperparams.max_length}")
    print(f"  Profile: {config.profile.name if config.profile else 'none'}")
    print(f"  Locked base: {config.locked.base_model}")

    if args.dry_run:
        print("\n🔍 DRY RUN - Would write:")
        print(json.dumps(nested_config, indent=2))
        return 0

    # Create backup if requested
    if args.backup:
        backup_path = config_path.with_suffix('.json.backup')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.parent / f"{config_path.stem}_{timestamp}.backup.json"

        print(f"\n💾 Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(old_config, f, indent=2)

    # Write migrated config
    print(f"\n✍️  Writing migrated config: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(nested_config, f, indent=2)

    print("\n✅ Migration complete!")
    print(f"\n📌 Next steps:")
    print(f"   1. Review the new config.json")
    print(f"   2. Test with: python3 trainer/config/loader.py")
    print(f"   3. When you run training, .config_lock.json will be created automatically")

    if args.backup:
        print(f"   4. Backup saved to: {backup_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
