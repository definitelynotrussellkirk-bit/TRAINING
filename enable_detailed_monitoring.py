#!/usr/bin/env python3
"""
Enable Detailed Monitoring - Integration Script

This script patches train.py to enable detailed monitoring with:
- Token-by-token comparison
- Full prompt visibility
- Real-time predictions

Run this once to enable monitoring, then start training normally.
"""

import re
from pathlib import Path


def patch_train_py():
    """Add detailed monitoring support to train.py"""

    train_file = Path(__file__).parent / "train.py"

    if not train_file.exists():
        print("❌ Error: train.py not found")
        return False

    print("Reading train.py...")
    with open(train_file, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'detail_collector' in content.lower():
        print("✅ train.py already has detailed monitoring enabled")
        return True

    # Find the imports section
    import_pattern = r'(from train import UltimateTrainer)'
    import_addition = '''from train import UltimateTrainer
try:
    from detail_collector import add_detail_collector_to_trainer
    DETAILED_MONITORING_AVAILABLE = True
except ImportError:
    DETAILED_MONITORING_AVAILABLE = False
    print("Note: Detailed monitoring not available (detail_collector.py not found)")
'''

    if re.search(import_pattern, content):
        content = re.sub(import_pattern, import_addition, content, count=1)
        print("✅ Added detail_collector import")
    else:
        # Try alternate import location
        import_alt = r'(import sys\nimport time)'
        if re.search(import_alt, content):
            content = re.sub(
                import_alt,
                r'\1\ntry:\n    from detail_collector import add_detail_collector_to_trainer\n    DETAILED_MONITORING_AVAILABLE = True\nexcept ImportError:\n    DETAILED_MONITORING_AVAILABLE = False',
                content,
                count=1
            )
            print("✅ Added detail_collector import (alternate location)")

    # Find where trainer is created and add collector
    # Look for pattern: trainer = Trainer(
    trainer_pattern = r'(trainer = Trainer\([^)]+\))'
    trainer_addition = r'''\1

        # Enable detailed monitoring if available
        if DETAILED_MONITORING_AVAILABLE and hasattr(self, 'val_dataset'):
            try:
                add_detail_collector_to_trainer(
                    trainer=trainer,
                    tokenizer=self.tokenizer,
                    eval_dataset=self.val_dataset,
                    update_frequency=50  # Update every 50 steps
                )
            except Exception as e:
                print(f"Warning: Could not enable detailed monitoring: {e}")
'''

    if re.search(trainer_pattern, content, re.DOTALL):
        content = re.sub(trainer_pattern, trainer_addition, content, flags=re.DOTALL)
        print("✅ Added detail collector initialization")
    else:
        print("⚠️  Could not find trainer initialization pattern")
        print("   You may need to add monitoring manually")

    # Write patched file
    backup_file = train_file.with_suffix('.py.backup')
    print(f"Creating backup: {backup_file}")
    with open(backup_file, 'w') as f:
        f.write(open(train_file).read())

    print(f"Writing patched train.py...")
    with open(train_file, 'w') as f:
        f.write(content)

    print("✅ train.py patched successfully")
    return True


def create_launch_script():
    """Create convenience script to launch monitoring dashboard"""

    script_path = Path(__file__).parent / "launch_detailed_monitor.sh"

    script_content = '''#!/bin/bash
# Launch Detailed Training Monitor

echo "🔬 Starting Detailed Training Monitor..."
echo ""
echo "This will show:"
echo "  - Current loss / eval loss"
echo "  - Complete prompt context"
echo "  - Golden vs predicted comparison"
echo "  - Token-by-token analysis"
echo ""
echo "Opening at: http://localhost:8081"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
python3 detailed_monitor.py
'''

    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)
    print(f"✅ Created launch script: {script_path}")


def main():
    print("=" * 80)
    print("🔧 ENABLE DETAILED TRAINING MONITORING")
    print("=" * 80)
    print()

    # Check required files
    required_files = [
        'train.py',
        'detailed_monitor.py',
        'detail_collector.py'
    ]

    missing = []
    for f in required_files:
        if not (Path(__file__).parent / f).exists():
            missing.append(f)

    if missing:
        print(f"❌ Error: Missing required files: {', '.join(missing)}")
        return

    print("✅ All required files present")
    print()

    # Patch train.py
    print("Step 1: Patching train.py...")
    if not patch_train_py():
        return

    print()

    # Create launch script
    print("Step 2: Creating launch script...")
    create_launch_script()

    print()
    print("=" * 80)
    print("✅ DETAILED MONITORING ENABLED")
    print("=" * 80)
    print()
    print("Next steps:")
    print()
    print("1. Start the monitoring dashboard:")
    print("   python3 detailed_monitor.py")
    print("   or")
    print("   ./launch_detailed_monitor.sh")
    print()
    print("2. Start training (in another terminal):")
    print("   python3 train.py --dataset inbox/leo_10k_with_system.jsonl \\")
    print("     --model model --output-dir adapters/test --epochs 1 --use-qlora")
    print()
    print("3. Open browser to: http://localhost:8081")
    print()
    print("The dashboard will update every 50 training steps with:")
    print("  ✓ Current loss / eval loss")
    print("  ✓ Complete prompt (system + user + assistant)")
    print("  ✓ Golden response (expected output)")
    print("  ✓ Model prediction")
    print("  ✓ Token-by-token comparison with accuracy")
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
