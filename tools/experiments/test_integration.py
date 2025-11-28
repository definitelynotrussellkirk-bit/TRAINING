#!/usr/bin/env python3
"""
Integration test for stop emoji system
Verifies that UltimateTrainer methods exist and work correctly
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import train module
import train

def test_methods_exist():
    """Verify all required methods exist"""
    print("🔍 Checking UltimateTrainer methods...")

    required_methods = [
        'enforce_thinking_requirement',
        'enforce_stop_requirement',
        'sanitize_example',
    ]

    missing = []
    for method_name in required_methods:
        if not hasattr(train.UltimateTrainer, method_name):
            missing.append(method_name)
        else:
            print(f"   ✅ {method_name}")

    if missing:
        print(f"   ❌ Missing methods: {missing}")
        return False

    print("   ✅ All methods exist\n")
    return True

def test_enforce_functions():
    """Test the enforce functions work correctly"""
    print("🧪 Testing enforce functions...")

    # Create test messages
    test_messages = [
        {"role": "user", "content": "Test question?"},
        {"role": "assistant", "content": "Test answer."}
    ]

    # Create a mock trainer instance
    class MockArgs:
        max_length = 2048
        base_model = "test"

    trainer = train.UltimateTrainer(MockArgs())

    # Test thinking requirement
    print("   Testing enforce_thinking_requirement...")
    result = trainer.enforce_thinking_requirement(test_messages.copy())

    user_content = result[0]['content']
    assistant_content = result[1]['content']

    assert train.THINKING_INSTRUCTION in user_content, "Think instruction not added to user"
    assert assistant_content.startswith(train.THINKING_PREFIX), "Think prefix not added to assistant"
    print("   ✅ enforce_thinking_requirement works")

    # Test stop requirement
    print("   Testing enforce_stop_requirement...")
    result = trainer.enforce_stop_requirement(test_messages.copy())

    user_content = result[0]['content']
    assistant_content = result[1]['content']

    assert train.STOP_INSTRUCTION in user_content, "Stop instruction not added to user"
    assert assistant_content.endswith(train.STOP_SUFFIX), "Stop suffix not added to assistant"
    print("   ✅ enforce_stop_requirement works\n")

    return True

def test_combined_pipeline():
    """Test both functions together (as used in train.py)"""
    print("🔗 Testing combined pipeline...")

    test_messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]

    class MockArgs:
        max_length = 2048
        base_model = "test"

    trainer = train.UltimateTrainer(MockArgs())

    # Apply both functions (same order as train.py)
    result = trainer.enforce_thinking_requirement(test_messages.copy())
    result = trainer.enforce_stop_requirement(result)

    user_content = result[0]['content']
    assistant_content = result[1]['content']

    print("   User message:")
    print(f"      {user_content}")
    print()
    print("   Assistant message:")
    print(f"      {assistant_content}")
    print()

    # Verify both transformations applied
    checks = [
        (train.THINKING_INSTRUCTION in user_content, "User has think instruction"),
        (train.STOP_INSTRUCTION in user_content, "User has stop instruction"),
        (assistant_content.startswith(train.THINKING_PREFIX), "Assistant starts with think prefix"),
        (assistant_content.endswith(train.STOP_SUFFIX), "Assistant ends with stop suffix"),
    ]

    all_pass = True
    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {description}")
        if not passed:
            all_pass = False

    return all_pass

if __name__ == "__main__":
    print("="*80)
    print("STOP EMOJI SYSTEM - INTEGRATION TEST")
    print("="*80)
    print()

    tests = [
        ("Methods exist", test_methods_exist),
        ("Enforce functions", test_enforce_functions),
        ("Combined pipeline", test_combined_pipeline),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print()
    if all(r[1] for r in results):
        print("🎉 ALL TESTS PASSED!")
        print()
        print("✅ Stop emoji system is ready for production!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
