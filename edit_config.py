#!/usr/bin/env python3
"""
Training Configuration Editor

Usage:
    python3 edit_config.py                    # Interactive editor
    python3 edit_config.py --set batch_size=2  # Set specific value
    python3 edit_config.py --show              # Show current config
"""

import json
import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"

# Config field descriptions
CONFIG_SCHEMA = {
    "model_name": {
        "type": str,
        "description": "Friendly name for your model",
        "example": "qwen3_vl_0.6b_thinking"
    },
    "model_path": {
        "type": str,
        "description": "Path to base model directory",
        "example": "/path/to/training/model"
    },
    "batch_size": {
        "type": int,
        "description": "Samples per training step (higher = more VRAM)",
        "example": "1-8",
        "tip": "Start with 1 for 0.6B models. Increase if you have VRAM."
    },
    "gradient_accumulation": {
        "type": int,
        "description": "Steps to accumulate before updating (effective_batch = batch_size × this)",
        "example": "4-32",
        "tip": "Higher = more stable training but slower updates"
    },
    "learning_rate": {
        "type": float,
        "description": "How fast the model learns",
        "example": "1e-4 to 5e-4",
        "tip": "Default 2e-4 is safe. Lower = slower but safer, higher = faster but risky"
    },
    "warmup_steps": {
        "type": int,
        "description": "Steps to gradually increase learning rate",
        "example": "50-200",
        "tip": "Prevents early training instability"
    },
    "lora_r": {
        "type": int,
        "description": "LoRA rank (adapter size)",
        "example": "8-128",
        "tip": "Higher = more parameters = better quality but more VRAM. 64 is good."
    },
    "lora_alpha": {
        "type": int,
        "description": "LoRA scaling factor",
        "example": "Usually lora_r / 2",
        "tip": "Default: half of lora_r"
    },
    "eval_steps": {
        "type": int,
        "description": "Run live inference every N steps",
        "example": "100-1000",
        "tip": "Shows progress during training"
    },
    "num_eval_samples": {
        "type": int,
        "description": "Number of examples to test during eval",
        "example": "3-10",
        "tip": "More = better sense of progress but slower"
    },
    "save_steps": {
        "type": int,
        "description": "Save checkpoint every N steps",
        "example": "500-5000",
        "tip": "More frequent = more disk usage"
    },
    "poll_interval": {
        "type": int,
        "description": "Daemon checks inbox every N seconds",
        "example": "10-60",
        "tip": "How often daemon looks for new training files"
    },
    "snapshot_time": {
        "type": str,
        "description": "Daily snapshot time (HH:MM format)",
        "example": "03:00",
        "tip": "Backs up model daily at this time"
    },
    "max_length": {
        "type": int,
        "description": "Maximum token length for training",
        "example": "1024-4096",
        "tip": "Higher = longer context but more VRAM. 2048 is good."
    }
}


def load_config():
    """Load config from file."""
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(config):
    """Save config to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Config saved to {CONFIG_FILE}")


def show_config():
    """Display current configuration."""
    config = load_config()

    print("\n" + "=" * 80)
    print("CURRENT TRAINING CONFIGURATION")
    print("=" * 80)

    for key, value in config.items():
        schema = CONFIG_SCHEMA.get(key, {})
        desc = schema.get("description", "No description")

        print(f"\n{key}: {value}")
        print(f"  → {desc}")

        if "tip" in schema:
            print(f"  💡 {schema['tip']}")

    print("\n" + "=" * 80)


def interactive_edit():
    """Interactive configuration editor."""
    config = load_config()

    print("\n" + "=" * 80)
    print("🔧 INTERACTIVE CONFIG EDITOR")
    print("=" * 80)
    print("\nType the setting name to edit, or 'done' to finish")
    print("Type 'show' to see current values, 'help' for tips\n")

    while True:
        cmd = input("Config> ").strip().lower()

        if cmd == "done":
            save_config(config)
            break

        if cmd == "show":
            show_config()
            continue

        if cmd == "help":
            print("\nAvailable settings:")
            for key in config.keys():
                schema = CONFIG_SCHEMA.get(key, {})
                desc = schema.get("description", "")
                print(f"  {key:25} - {desc}")
            continue

        if cmd in config:
            schema = CONFIG_SCHEMA.get(cmd, {})
            current = config[cmd]

            print(f"\nEditing: {cmd}")
            print(f"Current value: {current}")
            print(f"Description: {schema.get('description', '')}")

            if "example" in schema:
                print(f"Examples: {schema['example']}")
            if "tip" in schema:
                print(f"💡 {schema['tip']}")

            new_value = input(f"\nNew value (or Enter to keep current): ").strip()

            if new_value:
                # Type conversion
                if schema.get("type") == int:
                    try:
                        config[cmd] = int(new_value)
                        print(f"✓ Updated {cmd} = {config[cmd]}")
                    except ValueError:
                        print(f"❌ Must be an integer")
                elif schema.get("type") == float:
                    try:
                        config[cmd] = float(new_value)
                        print(f"✓ Updated {cmd} = {config[cmd]}")
                    except ValueError:
                        print(f"❌ Must be a number")
                else:
                    config[cmd] = new_value
                    print(f"✓ Updated {cmd} = {config[cmd]}")
        else:
            print(f"❌ Unknown setting: {cmd}")
            print("Type 'help' to see available settings")


def set_value(key_value):
    """Set a specific config value via command line."""
    if "=" not in key_value:
        print("❌ Format: --set key=value")
        return

    key, value = key_value.split("=", 1)
    key = key.strip()
    value = value.strip()

    config = load_config()

    if key not in config:
        print(f"❌ Unknown setting: {key}")
        print(f"Available: {', '.join(config.keys())}")
        return

    schema = CONFIG_SCHEMA.get(key, {})

    # Type conversion
    try:
        if schema.get("type") == int:
            value = int(value)
        elif schema.get("type") == float:
            value = float(value)
    except ValueError:
        print(f"❌ Invalid value type for {key}")
        return

    config[key] = value
    save_config(config)
    print(f"✓ Set {key} = {value}")


def main():
    if not CONFIG_FILE.exists():
        print(f"❌ Config file not found: {CONFIG_FILE}")
        return

    if len(sys.argv) == 1:
        # Interactive mode
        interactive_edit()
    elif "--show" in sys.argv:
        show_config()
    elif "--set" in sys.argv:
        idx = sys.argv.index("--set")
        if idx + 1 < len(sys.argv):
            set_value(sys.argv[idx + 1])
        else:
            print("❌ Missing value for --set")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
