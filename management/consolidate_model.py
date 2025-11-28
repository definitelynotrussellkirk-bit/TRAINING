#!/usr/bin/env python3
"""
Model Consolidation with Versioning & Backup System

SAFE consolidation that:
1. Creates verified backups BEFORE any changes
2. Creates a new version instead of overwriting
3. Preserves all training history with metadata
4. Enables rollback to any previous version

Never lose training progress again!

Usage:
    python3 consolidate_model.py --base-dir /path/to/TRAINING --description "What was trained"
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Import our new safety systems
from model_versioner import ModelVersioner
from backup_manager import BackupManager


def resolve_adapter_dir(base_dir: Path, config: dict, override: Path | None = None) -> Path:
    """
    Determine which current_model directory should be consolidated.
    Preference order:
      1. --current-dir CLI override
      2. config['current_model_dir'] (absolute or relative)
      3. <base_dir>/current_model
      4. <base_dir>/current_model_small
    Returns the first directory that exists and contains adapter_model.safetensors.
    """

    def normalize(path_value):
        if path_value is None:
            return None
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    candidates = []
    for candidate in [override, config.get('current_model_dir'), base_dir / 'current_model', base_dir / 'current_model_small']:
        path = normalize(candidate)
        if path and path not in candidates:
            candidates.append(path)

    for path in candidates:
        if path.exists() and (path / 'adapter_model.safetensors').exists():
            return path

    return normalize(override) or normalize(config.get('current_model_dir')) or (base_dir / 'current_model').resolve()


def consolidate_model(base_dir: Path, description: str, training_data: list = None, current_dir: Path | None = None):
    """
    Safely merge current adapter into base model with versioning.

    Args:
        base_dir: Training directory
        description: Human-readable description of this version
        training_data: List of training files used (optional)

    Returns:
        True if successful
    """

    config_file = base_dir / 'config.json'
    with open(config_file) as f:
        config = json.load(f)

    base_model_path = Path(config.get('base_model', config.get('model_path'))).resolve()
    current_model_path = resolve_adapter_dir(base_dir, config, current_dir)

    # Initialize versioning and backup systems
    versioner = ModelVersioner(str(base_dir))
    backup_mgr = BackupManager(str(base_dir))

    print("=" * 80)
    print("🔄 SAFE MODEL CONSOLIDATION WITH VERSIONING")
    print("=" * 80)
    print(f"Base model: {base_model_path}")
    print(f"Current adapter: {current_model_path}")
    print(f"Description: {description}")
    print()

    # ========================================================================
    # SAFETY CHECK 1: Verify we have an adapter
    # ========================================================================
    if not current_model_path.exists():
        print("❌ No adapter found at current_model/ - nothing to consolidate")
        return False

    adapter_file = current_model_path / 'adapter_model.safetensors'
    if not adapter_file.exists():
        print("❌ No adapter_model.safetensors found - nothing to consolidate")
        return False

    # ========================================================================
    # SAFETY CHECK 2: Create verified backup BEFORE any changes
    # ========================================================================
    print("🔒 SAFETY: Creating verified backup before consolidation...")
    backup_success, backup_path = backup_mgr.backup_before_consolidation(
        str(current_model_path)
    )

    if not backup_success:
        print("❌ ABORT: Backup failed - consolidation cancelled for safety")
        return False

    print(f"✅ Backup verified and saved to: {backup_path}")
    print()

    # ========================================================================
    # STEP 1: Load base model
    # ========================================================================
    print(f"📥 Loading base model from {base_model_path}...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return False

    # ========================================================================
    # STEP 2: Load tokenizer
    # ========================================================================
    print(f"📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False

    # ========================================================================
    # STEP 3: Load and merge adapter
    # ========================================================================
    print(f"🔗 Loading adapter from {current_model_path}...")
    try:
        model_with_adapter = PeftModel.from_pretrained(base_model, str(current_model_path))
    except Exception as e:
        print(f"❌ Failed to load adapter: {e}")
        return False

    print(f"⚙️  Merging adapter into base model...")
    try:
        merged_model = model_with_adapter.merge_and_unload()
    except Exception as e:
        print(f"❌ Failed to merge adapter: {e}")
        return False

    # ========================================================================
    # STEP 4: Get training metadata for version
    # ========================================================================
    training_status_file = base_dir / 'status' / 'training_status.json'
    if training_status_file.exists():
        with open(training_status_file) as f:
            status = json.load(f)
            total_steps = status.get('current_step', 0)
            final_loss = status.get('loss', None)
    else:
        total_steps = 0
        final_loss = None

    # ========================================================================
    # STEP 5: Create version (before we delete anything!)
    # ========================================================================
    print()
    print(f"📦 Creating version snapshot...")

    version_metadata = {
        'base_model': str(base_model_path),
        'total_steps': total_steps,
        'final_loss': final_loss,
    }

    try:
        version_id = versioner.create_version(
            adapter_path=str(current_model_path),
            description=description,
            training_data=training_data or [],
            metadata=version_metadata
        )
        print(f"✅ Created version: {version_id}")
    except Exception as e:
        print(f"❌ Failed to create version: {e}")
        print("⚠️  WARNING: Backup exists but version creation failed")
        return False

    # ========================================================================
    # STEP 6: Save merged model to consolidated location
    # ========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_base_path = base_dir / 'consolidated_models' / timestamp
    new_base_path.mkdir(parents=True, exist_ok=True)

    print()
    print(f"💾 Saving merged model to {new_base_path}...")
    try:
        merged_model.save_pretrained(str(new_base_path))
        tokenizer.save_pretrained(str(new_base_path))
    except Exception as e:
        print(f"❌ Failed to save merged model: {e}")
        return False

    # ========================================================================
    # STEP 7: Update config to point to new base
    # ========================================================================
    print(f"📝 Updating config.json to use new base model...")
    config['base_model'] = str(new_base_path)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # ========================================================================
    # STEP 8: SAFELY remove old adapter (we have backups + version!)
    # ========================================================================
    print(f"🗑️  Removing old adapter at {current_model_path}...")
    print(f"   (Safe to remove - backed up in version {version_id} and backup system)")
    shutil.rmtree(current_model_path)

    # ========================================================================
    # SUCCESS!
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ CONSOLIDATION COMPLETE!")
    print("=" * 80)
    print(f"📦 Version: {version_id}")
    print(f"📍 New base model: {new_base_path}")
    print(f"🔒 Backup: {backup_path}")
    print(f"📚 Version archive: models/versions/{version_id}_*")
    print()
    print("🆕 Next training will start with fresh adapter on merged base")
    print()
    print("To restore this version later:")
    print(f"   python3 model_versioner.py restore {version_id}")
    print()
    print("To list all versions:")
    print(f"   python3 model_versioner.py list")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Safely consolidate LoRA adapter into base model with versioning',
        epilog='Example: python3 consolidate_model.py --description "Math training 10k examples"'
    )
    parser.add_argument('--base-dir', type=Path, default=None, help='Base directory (default: auto-detect)')
    parser.add_argument('--current-dir', type=Path, help='Override current adapter directory (default: <base>/current_model)')
    parser.add_argument('--description', required=True, help='Description of this version (e.g., "Math training 10k")')
    parser.add_argument('--training-data', nargs='+', help='Training data files used (optional)')

    args = parser.parse_args()

    # Auto-detect base_dir if not provided
    if args.base_dir is None:
        try:
            from core.paths import get_base_dir
            args.base_dir = get_base_dir()
        except ImportError:
            args.base_dir = Path(__file__).parent.parent

    if not args.base_dir.exists():
        print(f"❌ Directory not found: {args.base_dir}")
        return 1

    try:
        success = consolidate_model(
            base_dir=args.base_dir,
            description=args.description,
            training_data=args.training_data,
            current_dir=args.current_dir
        )
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Consolidation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
