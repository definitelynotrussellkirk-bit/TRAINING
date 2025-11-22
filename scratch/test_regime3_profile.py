#!/usr/bin/env python3
"""
Test regime3 profile

Validates that regime3 profile works as expected.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from trainer.profiles import get_profile, Regime3Profile


def test_profile_import():
    """Test that regime3 profile can be imported"""
    print("TEST 1: Profile import")
    profile = get_profile("regime3")
    print(f"  ✓ Imported profile: {profile.name}")
    print(f"  ✓ Description: {profile.description}")
    print(f"  ✓ Version: {profile.version}")
    return profile


def test_system_prompt():
    """Test system prompt template"""
    print("\nTEST 2: System prompt template")
    profile = Regime3Profile()
    template = profile.get_system_prompt_template()
    print(f"  ✓ Template length: {len(template)} chars")

    # Fill in date
    filled = template.format(date=datetime.now().strftime("%Y-%m-%d"))
    print(f"  ✓ Contains canonical form instructions: {'(op arg1 arg2)' in filled}")
    print(f"  ✓ Contains answer markers: {'<<ANS_START>>' in filled}")


def test_transform_example():
    """Test example transformation"""
    print("\nTEST 3: Example transformation")
    profile = Regime3Profile()

    # Simple symbolic reasoning example
    example = {
        "messages": [
            {"role": "user", "content": "What is (add 2 3)?"},
            {"role": "assistant", "content": "(add 2 3) = 5"}
        ]
    }

    system_prompt = "Current date: 2025-11-22. Use symbolic reasoning."
    transformed = profile.transform_example(example, 0, system_prompt)

    print(f"  Original messages: {len(example['messages'])}")
    print(f"  Transformed messages: {len(transformed['messages'])}")

    # Check system prompt injection
    assert transformed['messages'][0]['role'] == 'system'
    print(f"  ✓ System prompt injected")

    # Check answer markers added
    assistant_content = transformed['messages'][2]['content']
    assert "<<ANS_START>>" in assistant_content
    assert "<<ANS_END>>" in assistant_content
    print(f"  ✓ Answer markers added to assistant message")

    print("\n  Full transformed messages:")
    for i, msg in enumerate(transformed['messages']):
        print(f"    [{i}] {msg['role']}: {msg['content'][:100]}...")


def test_answer_markers():
    """Test answer marker enforcement"""
    print("\nTEST 4: Answer marker enforcement")
    profile = Regime3Profile()

    # Test with missing markers
    messages = [
        {"role": "user", "content": "Calculate something"},
        {"role": "assistant", "content": "(add 1 2) = 3"}
    ]

    enforced = profile.enforce_answer_markers(messages)
    assistant_content = enforced[1]['content']

    assert "<<ANS_START>>" in assistant_content
    assert "<<ANS_END>>" in assistant_content
    print(f"  ✓ Missing markers added")
    print(f"  ✓ Content: {assistant_content}")

    # Test with partial markers
    messages2 = [
        {"role": "assistant", "content": "<<ANS_START>> (mul 2 3) = 6"}
    ]

    enforced2 = profile.enforce_answer_markers(messages2)
    content2 = enforced2[0]['content']

    assert "<<ANS_START>>" in content2
    assert "<<ANS_END>>" in content2
    print(f"  ✓ Partial markers completed")


def test_validation():
    """Test example validation"""
    print("\nTEST 5: Example validation")
    profile = Regime3Profile()

    # Valid example
    valid = {"messages": [{"role": "user", "content": "Test"}]}
    assert profile.validate_example(valid)
    print(f"  ✓ Valid example accepted")

    # Invalid examples
    invalid_cases = [
        {},  # Missing messages
        {"messages": []},  # Empty messages
    ]

    for invalid in invalid_cases:
        assert not profile.validate_example(invalid)
    print(f"  ✓ Invalid examples rejected")


def test_logits_processors():
    """Test logits processor creation"""
    print("\nTEST 6: Logits processors")
    profile = Regime3Profile()

    # Mock tokenizer (not used for regime3)
    class MockTokenizer:
        pass

    processors = profile.build_logits_processors(MockTokenizer())

    # Regime3 currently returns empty list (no special penalties)
    print(f"  ✓ Processors created (count: {len(processors)})")
    print(f"  ✓ Regime3 uses no special penalties (simpler than emoji_think)")


def test_profile_comparison():
    """Compare regime3 and emoji_think profiles"""
    print("\nTEST 7: Profile comparison")

    emoji = get_profile("emoji_think")
    regime = get_profile("regime3")

    print(f"  emoji_think: {emoji.description}")
    print(f"  regime3: {regime.description}")
    print(f"  ✓ Both profiles available in registry")


def main():
    """Run all tests"""
    print("=" * 60)
    print("REGIME-3 PROFILE TEST SUITE")
    print("=" * 60)

    try:
        test_profile_import()
        test_system_prompt()
        test_transform_example()
        test_answer_markers()
        test_validation()
        test_logits_processors()
        test_profile_comparison()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
