#!/usr/bin/env python3
"""
Consolidate (merge) the current LoRA adapter into the base model.

This "commits" all learned knowledge from the adapter into the base model,
then resets to a fresh adapter for future training.

Usage:
    python3 consolidate_model.py --base-dir /path/to/TRAINING
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def consolidate_model(base_dir: Path):
    """Merge current adapter into base model and reset for fresh training."""

    config_file = base_dir / 'config.json'
    with open(config_file) as f:
        config = json.load(f)

    base_model_path = Path(config['model_path'])
    current_model_path = base_dir / 'current_model'

    # Check if we have an adapter to consolidate
    if not current_model_path.exists():
        print("❌ No adapter found at current_model/ - nothing to consolidate")
        return False

    adapter_file = current_model_path / 'adapter_model.safetensors'
    if not adapter_file.exists():
        print("❌ No adapter_model.safetensors found - nothing to consolidate")
        return False

    print("=" * 80)
    print("🔄 STARTING MODEL CONSOLIDATION")
    print("=" * 80)
    print(f"Base model: {base_model_path}")
    print(f"Current adapter: {current_model_path}")
    print()

    # Create backup location with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = base_dir / 'consolidated_backups' / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Backing up current adapter to: {backup_dir}")
    shutil.copytree(current_model_path, backup_dir / 'adapter')

    # Load base model
    print(f"📥 Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )

    # Load tokenizer
    print(f"📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))

    # Load and merge adapter
    print(f"🔗 Loading adapter from {current_model_path}...")
    model_with_adapter = PeftModel.from_pretrained(base_model, str(current_model_path))

    print(f"⚙️  Merging adapter into base model...")
    merged_model = model_with_adapter.merge_and_unload()

    # Create new base model location
    new_base_path = base_dir / 'consolidated_models' / timestamp
    new_base_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving merged model to {new_base_path}...")
    merged_model.save_pretrained(str(new_base_path))
    tokenizer.save_pretrained(str(new_base_path))

    # Update config to point to new base
    print(f"📝 Updating config.json to use new base model...")
    config['base_model'] = str(new_base_path)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Remove current adapter (will start fresh on next training)
    print(f"🗑️  Removing old adapter at {current_model_path}...")
    shutil.rmtree(current_model_path)

    # Save consolidation metadata
    metadata = {
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat(),
        'old_base': str(base_model_path),
        'new_base': str(new_base_path),
        'adapter_backup': str(backup_dir / 'adapter')
    }

    metadata_file = backup_dir / 'consolidation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 80)
    print("✅ CONSOLIDATION COMPLETE!")
    print("=" * 80)
    print(f"📍 New base model: {new_base_path}")
    print(f"📦 Adapter backup: {backup_dir}")
    print(f"🆕 Next training will start with fresh adapter on merged base")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description='Consolidate LoRA adapter into base model')
    parser.add_argument('--base-dir', type=Path, required=True, help='Base directory (e.g., /path/to/TRAINING)')
    args = parser.parse_args()

    if not args.base_dir.exists():
        print(f"❌ Directory not found: {args.base_dir}")
        return 1

    try:
        success = consolidate_model(args.base_dir)
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Consolidation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
