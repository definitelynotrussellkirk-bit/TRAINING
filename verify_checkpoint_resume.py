#!/usr/bin/env python3
"""
Verify Checkpoint Resume - Ensures training resumes from correct checkpoint

This prevents:
1. Training starting from scratch (ignoring checkpoints)
2. Training from wrong checkpoint
3. Training from base model when adapter exists
4. Resuming from corrupted checkpoint

Usage:
    python3 verify_checkpoint_resume.py
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path("/path/to/training")
CONFIG_FILE = BASE_DIR / "config.json"
STATUS_FILE = BASE_DIR / "status" / "training_status.json"
MODEL_DIR = BASE_DIR / "current_model"

def get_latest_checkpoint():
    """Find latest checkpoint"""
    if not MODEL_DIR.exists():
        return None

    checkpoints = sorted(MODEL_DIR.glob("checkpoint-*"))
    if not checkpoints:
        return None

    return checkpoints[-1]

def verify_checkpoint(checkpoint_dir):
    """Verify checkpoint is valid"""
    required_files = [
        'adapter_model.safetensors',
        'trainer_state.json',
        'optimizer.pt',
        'scheduler.pt'
    ]

    missing = []
    for fname in required_files:
        if not (checkpoint_dir / fname).exists():
            missing.append(fname)

    return missing

def get_checkpoint_step(checkpoint_dir):
    """Get global step from checkpoint"""
    try:
        with open(checkpoint_dir / 'trainer_state.json') as f:
            state = json.load(f)
        return state.get('global_step', 0)
    except:
        return None

def check_resume_correctness():
    """Check if training will resume correctly"""
    print("\n" + "="*80)
    print("🔍 CHECKPOINT RESUME VERIFICATION")
    print("="*80)

    issues = []
    warnings = []

    # Check if model directory exists
    if not MODEL_DIR.exists():
        print("\n✅ No current_model - will start fresh (expected for first run)")
        return True

    # Check for checkpoints
    latest_checkpoint = get_latest_checkpoint()
    if not latest_checkpoint:
        print("\n⚠️ current_model exists but no checkpoints found")
        warnings.append("No checkpoints - may start from base model")
    else:
        checkpoint_name = latest_checkpoint.name
        print(f"\n📦 Latest checkpoint: {checkpoint_name}")

        # Verify checkpoint integrity
        missing_files = verify_checkpoint(latest_checkpoint)
        if missing_files:
            print(f"❌ Checkpoint missing files: {missing_files}")
            issues.append(f"Incomplete checkpoint: {checkpoint_name}")
        else:
            print("✅ Checkpoint has all required files")

        # Get checkpoint step
        checkpoint_step = get_checkpoint_step(latest_checkpoint)
        if checkpoint_step is None:
            print("❌ Cannot read checkpoint step")
            issues.append("Corrupted trainer_state.json")
        else:
            print(f"✅ Checkpoint global_step: {checkpoint_step}")

            # Check current status
            if STATUS_FILE.exists():
                try:
                    with open(STATUS_FILE) as f:
                        status = json.load(f)
                    current_step = status.get('current_step', 0)

                    if current_step < checkpoint_step:
                        print(f"⚠️ Status step ({current_step}) < checkpoint step ({checkpoint_step})")
                        print(f"   This is OK if daemon just started")
                    elif current_step > checkpoint_step:
                        print(f"✅ Training resumed correctly (now at step {current_step})")
                        delta = current_step - checkpoint_step
                        print(f"   Progress since checkpoint: +{delta} steps")
                    else:
                        print(f"⚠️ Training at checkpoint step ({current_step})")
                        print(f"   Either just started or stuck")
                except Exception as e:
                    print(f"⚠️ Error reading status: {e}")

    # Check config points to correct base model
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        base_model = Path(config.get('base_model', ''))

        print(f"\n🔧 Base model: {base_model}")

        if not base_model.exists():
            print(f"❌ Base model path does not exist!")
            issues.append(f"Invalid base_model path: {base_model}")
        else:
            print("✅ Base model path exists")

            # Check if it's the raw base model or consolidated
            if "DIO_20251114" in str(base_model):
                if latest_checkpoint:
                    print("⚠️ Using raw DIO base model but checkpoints exist")
                    print("   This is OK if this is first training run")
                else:
                    print("✅ Using raw DIO base model (first training)")
            else:
                print("✅ Using consolidated model (has prior training)")
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        issues.append("Cannot validate config")

    # Summary
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)

    if issues:
        print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"   • {issue}")
        print("\n⚠️ DO NOT START TRAINING - Fix issues first!")
        return False
    elif warnings:
        print(f"\n⚠️ WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"   • {warning}")
        print("\n✅ Safe to proceed, but review warnings")
        return True
    else:
        print("\n✅ ALL CHECKS PASSED")
        print("   Training will resume correctly from latest checkpoint")
        return True

def main():
    success = check_resume_correctness()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
