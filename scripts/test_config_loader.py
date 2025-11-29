#!/usr/bin/env python3
"""Test harness for config loader.

Moved from trainer/config/loader.py to keep the module clean.
"""

from pathlib import Path

from trainer.config.loader import ConfigLoader


def main():
    """Test config loading."""
    print("Testing config loader...\n")

    # Example: Load from config.json
    try:
        config_path = Path("config.json")
        if config_path.exists():
            config_dict = ConfigLoader.from_json_file(config_path)
            print(f"✓ Loaded config from {config_path}")
            print(f"  Model: {config_dict.get('model_path', 'not set')}")
            print(f"  Batch size: {config_dict.get('batch_size', 'not set')}")
        else:
            print(f"✗ Config file not found: {config_path}")
    except Exception as e:
        print(f"✗ Error loading config: {e}")

    print("\nConfig loader ready!")


if __name__ == "__main__":
    main()
