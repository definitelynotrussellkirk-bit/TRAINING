#!/usr/bin/env python3
"""
Config Validator - Prevents dangerous config changes during training

This prevents:
1. Changing base_model when checkpoints exist (would train on wrong version)
2. Invalid config values
3. Path references to non-existent files

Usage:
    python3 config_validator.py
    python3 config_validator.py --validate-only  # Don't fix, just check
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Use centralized path resolution instead of hard-coded paths
try:
    from core.paths import get_base_dir
    BASE_DIR = get_base_dir()
except ImportError:
    BASE_DIR = Path(__file__).parent.parent  # Fallback: parent of safety/

CONFIG_FILE = BASE_DIR / "config.json"
MODEL_DIR = BASE_DIR / "current_model"
CONFIG_LOCK_FILE = BASE_DIR / ".config_lock.json"

class ConfigValidator:
    def __init__(self):
        self.config = None
        self.locked_config = None
        self.issues = []
        self.warnings = []

    def load_config(self):
        """Load current config"""
        try:
            with open(CONFIG_FILE) as f:
                self.config = json.load(f)
            return True
        except Exception as e:
            self.issues.append(f"Cannot load config: {e}")
            return False

    def load_locked_config(self):
        """Load locked config (from when training started)"""
        if CONFIG_LOCK_FILE.exists():
            try:
                with open(CONFIG_LOCK_FILE) as f:
                    self.locked_config = json.load(f)
                return True
            except:
                pass
        return False

    def create_config_lock(self):
        """Create config lock file"""
        if not self.has_active_training():
            print("ℹ️  No active training - not creating lock")
            return

        lock_data = {
            "base_model": self.config.get('base_model'),
            "model_name": self.config.get('model_name'),
            "max_length": self.config.get('max_length'),
            "locked_at": datetime.now().isoformat(),
            "reason": "Prevent config changes during active training"
        }

        with open(CONFIG_LOCK_FILE, 'w') as f:
            json.dump(lock_data, f, indent=2)

        print(f"🔒 Config locked: {lock_data}")

    def has_active_training(self):
        """Check if training is active"""
        if not MODEL_DIR.exists():
            return False

        checkpoints = list(MODEL_DIR.glob("checkpoint-*"))
        return len(checkpoints) > 0

    def validate_no_critical_changes(self):
        """Validate no critical changes if training is active"""
        if not self.has_active_training():
            print("✅ No active training - config changes allowed")
            return True

        if not self.load_locked_config():
            # No lock file, create it now
            print("⚠️ No config lock found - creating one now")
            self.create_config_lock()
            return True

        # Check critical fields
        critical_fields = ['base_model', 'model_name', 'max_length']

        for field in critical_fields:
            current = self.config.get(field)
            locked = self.locked_config.get(field)

            if current != locked:
                self.issues.append(
                    f"CRITICAL: Cannot change '{field}' during active training!\n"
                    f"   Current: {current}\n"
                    f"   Locked:  {locked}\n"
                    f"   This would cause training to use wrong model version!"
                )

        return len(self.issues) == 0

    def validate_paths(self):
        """Validate file paths exist"""
        base_model = self.config.get('base_model')
        if base_model:
            if not Path(base_model).exists():
                self.issues.append(f"base_model path does not exist: {base_model}")

    def validate_values(self):
        """Validate config values are reasonable"""
        # Check batch_size
        batch_size = self.config.get('batch_size', 1)
        if batch_size < 1:
            self.issues.append(f"batch_size must be >= 1, got {batch_size}")
        elif batch_size > 32:
            self.warnings.append(f"batch_size very large ({batch_size}), may cause OOM")

        # Check learning_rate
        lr = self.config.get('learning_rate', 0.0002)
        if lr <= 0:
            self.issues.append(f"learning_rate must be > 0, got {lr}")
        elif lr > 0.01:
            self.warnings.append(f"learning_rate very high ({lr}), may cause instability")

        # Check max_length
        max_length = self.config.get('max_length', 2048)
        if max_length < 128:
            self.warnings.append(f"max_length very small ({max_length}), may truncate data")
        elif max_length > 8192:
            self.warnings.append(f"max_length very large ({max_length}), may cause OOM")

    def print_report(self):
        """Print validation report"""
        print("\n" + "="*80)
        print("🔍 CONFIG VALIDATION REPORT")
        print("="*80)

        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"\n{issue}")

            print("\n🚨 CANNOT PROCEED - Fix issues first!")
            print("\nTo fix:")
            print("   1. Revert config.json to locked values")
            print("   2. Or consolidate and restart training")
            print("   3. Or delete .config_lock.json (DANGEROUS - only if sure)")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        if not self.issues and not self.warnings:
            print("\n✅ ALL VALIDATIONS PASSED")
            print("   Config is safe to use")

        print("\n" + "="*80)

    def validate(self):
        """Run all validations"""
        if not self.load_config():
            self.print_report()
            return False

        self.validate_no_critical_changes()
        self.validate_paths()
        self.validate_values()

        self.print_report()

        return len(self.issues) == 0

def main():
    parser = argparse.ArgumentParser(description="Validate config.json")
    parser.add_argument('--create-lock', action='store_true',
                        help="Create config lock file")
    parser.add_argument('--remove-lock', action='store_true',
                        help="Remove config lock file (use with caution!)")
    args = parser.parse_args()

    validator = ConfigValidator()

    if args.create_lock:
        if validator.load_config():
            validator.create_config_lock()
        return

    if args.remove_lock:
        if CONFIG_LOCK_FILE.exists():
            CONFIG_LOCK_FILE.unlink()
            print("🔓 Config lock removed")
        else:
            print("ℹ️  No config lock to remove")
        return

    # Normal validation
    success = validator.validate()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
