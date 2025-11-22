#!/usr/bin/env python3
"""
Test emoji profile extraction

Validates that the profile system works as expected.
"""

from datetime import datetime
from trainer.profiles import get_profile, EmojiThinkProfile


def test_profile_import():
    """Test that profile can be imported"""
    print("TEST 1: Profile import")
    profile = get_profile("emoji_think")
    print(f"  ✓ Imported profile: {profile.name}")
    print(f"  ✓ Description: {profile.description}")
    print(f"  ✓ Version: {profile.version}")
    return profile


def test_system_prompt():
    """Test system prompt template"""
    print("\nTEST 2: System prompt template")
    profile = EmojiThinkProfile()
    template = profile.get_system_prompt_template()
    print(f"  ✓ Template: {template}")

    # Fill in date
    filled = template.format(date=datetime.now().strftime("%Y-%m-%d"))
    print(f"  ✓ Filled: {filled}")


def test_transform_example():
    """Test example transformation"""
    print("\nTEST 3: Example transformation")
    profile = EmojiThinkProfile()

    # Simple example
    example = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
    }

    system_prompt = "Current date: 2025-11-22. Respond naturally and concisely."
    transformed = profile.transform_example(example, 0, system_prompt)

    print(f"  Original messages: {len(example['messages'])}")
    print(f"  Transformed messages: {len(transformed['messages'])}")

    # Check system prompt injection
    assert transformed['messages'][0]['role'] == 'system'
    print(f"  ✓ System prompt injected")

    # Check user message has thinking instruction
    user_content = transformed['messages'][1]['content']
    assert "think with" in user_content.lower()
    print(f"  ✓ Thinking instruction added to user message")

    # Check user message has stop instruction
    assert "When finished" in user_content
    print(f"  ✓ Stop instruction added to user message")

    # Check assistant message has thinking prefix
    assistant_content = transformed['messages'][2]['content']
    # Should start with one of the thinking emojis
    from trainer.profiles.emoji_think import THINKING_EMOJIS, STOP_EMOJI_POOL
    has_thinking_prefix = any(assistant_content.startswith(emoji) for emoji in THINKING_EMOJIS)
    assert has_thinking_prefix
    print(f"  ✓ Thinking emoji prefix added to assistant message")

    # Check assistant message has stop suffix
    has_stop_suffix = any(emoji in assistant_content[-10:] for emoji in STOP_EMOJI_POOL)
    assert has_stop_suffix
    print(f"  ✓ Stop emoji suffix added to assistant message")

    print("\n  Full transformed messages:")
    for i, msg in enumerate(transformed['messages']):
        print(f"    [{i}] {msg['role']}: {msg['content'][:100]}...")


def test_sanitization():
    """Test sanitization of <think> tags"""
    print("\nTEST 4: Sanitization")
    profile = EmojiThinkProfile()

    example = {
        "messages": [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "<think>Internal thought</think> Response"}
        ]
    }

    sanitized = profile.sanitize_example(example)
    assistant_content = sanitized['messages'][1]['content']

    assert "<think>" not in assistant_content
    assert "</think>" not in assistant_content
    assert "Internal thought Response" == assistant_content
    print(f"  ✓ <think> tags removed")
    print(f"  ✓ Content: {assistant_content}")


def test_validation():
    """Test example validation"""
    print("\nTEST 5: Example validation")
    profile = EmojiThinkProfile()

    # Valid example
    valid = {"messages": [{"role": "user", "content": "Hello"}]}
    assert profile.validate_example(valid)
    print(f"  ✓ Valid example accepted")

    # Invalid examples
    invalid_cases = [
        {},  # Missing messages
        {"messages": []},  # Empty messages
        {"messages": [{}]},  # Missing role/content
        {"messages": [{"role": "user"}]},  # Missing content
    ]

    for invalid in invalid_cases:
        assert not profile.validate_example(invalid)
    print(f"  ✓ Invalid examples rejected ({len(invalid_cases)} cases)")


def test_metadata():
    """Test metadata"""
    print("\nTEST 6: Profile metadata")
    profile = EmojiThinkProfile()
    metadata = profile.get_metadata()

    print(f"  Name: {metadata['name']}")
    print(f"  Description: {metadata['description']}")
    print(f"  Version: {metadata['version']}")

    assert metadata['name'] == 'emoji_think'
    print(f"  ✓ Metadata valid")


def main():
    """Run all tests"""
    print("=" * 60)
    print("EMOJI PROFILE TEST SUITE")
    print("=" * 60)

    try:
        test_profile_import()
        test_system_prompt()
        test_transform_example()
        test_sanitization()
        test_validation()
        test_metadata()

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
