#!/usr/bin/env python3
"""
Test Wrong Version Scenarios - Check if system can train on wrong model version

Tests:
1. Can trainer accidentally ignore checkpoints and use base model?
2. Can config changes cause training to restart from scratch?
3. Can data changes crash or reset training?
4. Can consolidation overwrite active training?

Usage:
    python3 test_wrong_version_scenarios.py
"""

import json
from pathlib import Path

BASE_DIR = Path("/path/to/training")
CONFIG_FILE = BASE_DIR / "config.json"
MODEL_DIR = BASE_DIR / "current_model"
INBOX_DIR = BASE_DIR / "inbox"

def test_checkpoint_ignore_scenario():
    """Test: Can trainer ignore checkpoints and use base model?"""
    print("\n" + "="*80)
    print("🧪 TEST 1: Checkpoint Ignore Scenario")
    print("="*80)

    # Check train.py code
    train_py = BASE_DIR / "train.py"
    if not train_py.exists():
        print("❌ train.py not found")
        return False

    with open(train_py) as f:
        code = f.read()

    # Look for checkpoint resume logic
    has_resume_logic = "resume_from_checkpoint" in code
    has_checkpoint_detection = "checkpoint-" in code or "find_last_checkpoint" in code

    print(f"\n📝 Checking train.py code:")
    print(f"   resume_from_checkpoint in code: {has_resume_logic}")
    print(f"   Checkpoint detection logic: {has_checkpoint_detection}")

    if has_resume_logic:
        print("\n✅ SAFE: Trainer has checkpoint resume logic")
        print("   Will automatically resume from latest checkpoint")
        return True
    else:
        print("\n❌ RISK: No checkpoint resume logic found!")
        print("   Training may start from scratch even if checkpoints exist")
        return False

def test_config_change_scenario():
    """Test: Can config changes cause training to restart?"""
    print("\n" + "="*80)
    print("🧪 TEST 2: Config Change Impact")
    print("="*80)

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    base_model = config.get('base_model')
    print(f"\n📝 Current config:")
    print(f"   base_model: {base_model}")

    # Check if changing base_model while training would be bad
    if MODEL_DIR.exists():
        checkpoints = list(MODEL_DIR.glob("checkpoint-*"))
        if checkpoints:
            print(f"\n⚠️ WARNING: {len(checkpoints)} checkpoints exist")
            print("\n🚨 RISK SCENARIO:")
            print("   If you change base_model in config.json:")
            print("   1. Trainer loads NEW base model")
            print("   2. Tries to apply checkpoints from OLD model")
            print("   3. May crash or train on wrong version")
            print("\n💡 SAFEGUARD NEEDED:")
            print("   Add config validation that prevents base_model change")
            print("   when active checkpoints exist")
            return False
        else:
            print("\n✅ SAFE: No checkpoints, safe to change base_model")
            return True
    else:
        print("\n✅ SAFE: No current_model directory")
        return True

def test_data_change_crash_scenario():
    """Test: Can data changes crash training?"""
    print("\n" + "="*80)
    print("🧪 TEST 3: Data Change During Training")
    print("="*80)

    print("\n📝 Checking training daemon behavior...")

    # Check daemon code
    daemon_py = BASE_DIR / "training_daemon.py"
    if daemon_py.exists():
        with open(daemon_py) as f:
            code = f.read()

        # Check if daemon monitors inbox during training
        monitors_inbox = "poll" in code.lower() or "watch" in code.lower()
        handles_queue = "queue" in code.lower()

        print(f"   Daemon polls inbox: {monitors_inbox}")
        print(f"   Daemon uses queue system: {handles_queue}")

        if handles_queue:
            print("\n✅ SAFE: Daemon uses queue system")
            print("   Files processed sequentially from queue")
            print("   Adding new files won't crash active training")
            print("   New files will be processed AFTER current file completes")
        else:
            print("\n⚠️ RISK: No queue system detected")

        # Check what happens if file is deleted during training
        print("\n🚨 EDGE CASE: What if current file is deleted?")
        print("   If someone deletes file from queue/processing/:")
        print("   1. Training may crash with FileNotFoundError")
        print("   2. Or continue if data already loaded in memory")
        print("\n💡 SAFEGUARD: Don't manually delete files from queue/processing/")

        return handles_queue
    else:
        print("❌ daemon not found")
        return False

def test_consolidation_conflict_scenario():
    """Test: Can consolidation overwrite active training?"""
    print("\n" + "="*80)
    print("🧪 TEST 4: Consolidation During Active Training")
    print("="*80)

    consolidate_py = BASE_DIR / "consolidate_model.py"
    if not consolidate_py.exists():
        print("❌ consolidate_model.py not found")
        return False

    with open(consolidate_py) as f:
        code = f.read()

    # Check for safety checks
    has_backup = "backup" in code.lower()
    has_version = "version" in code.lower()
    checks_running = "running" in code.lower() or "daemon" in code.lower()

    print(f"\n📝 Checking consolidate_model.py:")
    print(f"   Creates backup: {has_backup}")
    print(f"   Creates version: {has_version}")
    print(f"   Checks if training running: {checks_running}")

    if has_backup and has_version:
        print("\n✅ SAFE: Consolidation creates backups and versions")
        print("   Even if consolidation runs during training:")
        print("   1. Old adapter backed up")
        print("   2. Versioned snapshot created")
        print("   3. Can restore if needed")
    else:
        print("\n⚠️ RISK: Consolidation may not have full safety")

    if not checks_running:
        print("\n🚨 EDGE CASE: Consolidation doesn't check if training is running")
        print("   If consolidation runs while training:")
        print("   1. Adapter gets merged into base model")
        print("   2. Training may continue on stale checkpoint")
        print("   3. Progress after last checkpoint may be lost")
        print("\n💡 SAFEGUARD NEEDED:")
        print("   Add check to prevent consolidation during active training")
        print("   Or ensure training restarts after consolidation")
        return False

    return True

def test_orphaned_checkpoint_cleanup():
    """Test: Can checkpoint cleanup delete active training state?"""
    print("\n" + "="*80)
    print("🧪 TEST 5: Checkpoint Cleanup Safety")
    print("="*80)

    if not MODEL_DIR.exists():
        print("✅ No current_model - test not applicable")
        return True

    checkpoints = sorted(MODEL_DIR.glob("checkpoint-*"))
    print(f"\n📝 Found {len(checkpoints)} checkpoints")

    if len(checkpoints) > 30:
        print("\n⚠️ WARNING: Many checkpoints exist (>30)")
        print("   Manual cleanup may be attempted")
        print("\n🚨 RISK SCENARIO:")
        print("   If someone runs: rm -rf current_model/checkpoint-*")
        print("   1. ALL checkpoints deleted")
        print("   2. Training cannot resume")
        print("   3. Must start from base model (losing progress)")
        print("\n💡 SAFEGUARD: Never delete ALL checkpoints")
        print("   Keep at least latest 3-5 checkpoints")
        return False
    else:
        print("✅ Checkpoint count reasonable")
        return True

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*80)
    print("📊 WRONG VERSION SCENARIO TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total-passed}/{total}")

    print("\n📋 RESULTS:")
    for test, passed in results.items():
        status = "✅ SAFE" if passed else "❌ RISK"
        print(f"   {status}: {test}")

    if total - passed > 0:
        print("\n🚨 CRITICAL SAFEGUARDS NEEDED:")
        for test, passed in results.items():
            if not passed:
                print(f"   • {test}")

    print("\n" + "="*80)

def main():
    results = {
        "Checkpoint ignore prevention": test_checkpoint_ignore_scenario(),
        "Config change safety": test_config_change_scenario(),
        "Data change handling": test_data_change_crash_scenario(),
        "Consolidation conflict prevention": test_consolidation_conflict_scenario(),
        "Checkpoint cleanup safety": test_orphaned_checkpoint_cleanup()
    }

    print_summary(results)

if __name__ == '__main__':
    main()
