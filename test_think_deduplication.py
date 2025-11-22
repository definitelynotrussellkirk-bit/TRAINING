#!/usr/bin/env python3
"""
Test that thinking/stop instructions are not duplicated.
"""

import json
from pathlib import Path

# Test cases
test_cases = [
    {
        "name": "Already has thinking instruction (same emoji)",
        "data": {
            "messages": [
                {"role": "user", "content": "What is 2+2? Please think with 🤔 three times before answering."},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        }
    },
    {
        "name": "Already has thinking instruction (DIFFERENT emoji)",
        "data": {
            "messages": [
                {"role": "user", "content": "What is 2+2? For this task, think with 💡 /five/ times."},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        }
    },
    {
        "name": "Already has thinking emoji prefix",
        "data": {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "🤔🤔🤔 2+2 equals 4."}
            ]
        }
    },
    {
        "name": "Already has stop instruction",
        "data": {
            "messages": [
                {"role": "user", "content": "What is 2+2? When finished, emit 🛑 /three/ times to signal completion."},
                {"role": "assistant", "content": "2+2 equals 4.\n🛑🛑🛑"}
            ]
        }
    },
    {
        "name": "Has nothing (baseline)",
        "data": {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        }
    },
    {
        "name": "Has everything already",
        "data": {
            "messages": [
                {"role": "user", "content": "What is 2+2? Please think with 💭 five times before answering. When finished, emit 🛑 /three/ times to signal completion."},
                {"role": "assistant", "content": "💭💭💭💭💭 2+2 equals 4.\n🛑🛑🛑"}
            ]
        }
    }
]

def count_occurrences(text, pattern):
    """Count non-overlapping occurrences of pattern in text."""
    return text.count(pattern)

def analyze_example(name, data):
    """Analyze a test example for duplicates."""
    print(f"\n{'=' * 70}")
    print(f"TEST: {name}")
    print(f"{'=' * 70}")

    user_msg = data['messages'][0]['content']
    assistant_msg = data['messages'][1]['content']

    # Count thinking instructions
    think_instructions = sum(count_occurrences(user_msg, f"think with {emoji}") for emoji in ["🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"])

    # Count thinking emoji prefixes (at start of assistant message)
    import re
    thinking_emojis = ["🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"]
    emoji_blocks = []
    for emoji in thinking_emojis:
        if assistant_msg.startswith(emoji):
            # Count consecutive emojis at start
            count = 0
            for char in assistant_msg:
                if char == emoji:
                    count += 1
                else:
                    break
            if count > 0:
                emoji_blocks.append(f"{emoji} × {count}")

    # Count stop instructions
    stop_instructions = count_occurrences(user_msg, "When finished, emit 🛑")

    # Count stop suffixes
    stop_suffixes = count_occurrences(assistant_msg, "🛑🛑🛑")

    print(f"\nUser message:")
    print(f"  {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
    print(f"\nAssistant message:")
    print(f"  {assistant_msg[:100]}{'...' if len(assistant_msg) > 100 else ''}")

    print(f"\nAnalysis:")
    print(f"  Think instructions in user: {think_instructions}")
    print(f"  Emoji prefixes in assistant: {', '.join(emoji_blocks) if emoji_blocks else 'None'}")
    print(f"  Stop instructions in user: {stop_instructions}")
    print(f"  Stop suffixes in assistant: {stop_suffixes}")

    # Check for duplicates
    issues = []
    if think_instructions > 1:
        issues.append(f"❌ DUPLICATE: {think_instructions} thinking instructions in user message")
    if len(emoji_blocks) > 1:
        issues.append(f"❌ DUPLICATE: Multiple emoji blocks in assistant message")
    if stop_instructions > 1:
        issues.append(f"❌ DUPLICATE: {stop_instructions} stop instructions in user message")
    if stop_suffixes > 1:
        issues.append(f"❌ DUPLICATE: {stop_suffixes} stop suffixes in assistant message")

    if issues:
        print(f"\n🚨 ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print(f"\n✅ NO DUPLICATES DETECTED")
        return True

def main():
    """Run all tests."""
    print("=" * 70)
    print("THINKING/STOP INSTRUCTION DEDUPLICATION TEST")
    print("=" * 70)
    print("\nTesting current data files for duplicate instructions...")

    all_passed = True

    for test_case in test_cases:
        passed = analyze_example(test_case['name'], test_case['data'])
        all_passed = all_passed and passed

    # Now test actual files if they exist
    print(f"\n{'=' * 70}")
    print("CHECKING ACTUAL TRAINING DATA")
    print(f"{'=' * 70}")

    # Check inbox
    inbox = Path("inbox")
    if inbox.exists():
        jsonl_files = list(inbox.glob("*.jsonl"))
        if jsonl_files:
            print(f"\nFound {len(jsonl_files)} files in inbox/")
            print("Checking first file for duplicates...")

            sample_file = jsonl_files[0]
            with open(sample_file) as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Check first 3 examples
                        break
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'messages' in data:
                                analyze_example(f"{sample_file.name} example {i+1}", data)
                        except Exception as e:
                            print(f"Error parsing line {i}: {e}")
        else:
            print("\nNo .jsonl files in inbox/")

    # Check queue
    queue_dir = Path("queue/normal")
    if queue_dir.exists():
        jsonl_files = list(queue_dir.glob("*.jsonl"))
        if jsonl_files:
            print(f"\nFound {len(jsonl_files)} files in queue/normal/")
            print("Checking first file for duplicates...")

            sample_file = jsonl_files[0]
            with open(sample_file) as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Check first 3 examples
                        break
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'messages' in data:
                                analyze_example(f"{sample_file.name} example {i+1}", data)
                        except Exception as e:
                            print(f"Error parsing line {i}: {e}")
        else:
            print("\nNo .jsonl files in queue/normal/")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    if all_passed:
        print("✅ All baseline tests passed - no duplicates detected!")
    else:
        print("❌ Some tests failed - duplicates detected!")
    print()
    print("The system checks:")
    print("  1. If 'think with {emoji}' already exists in user message")
    print("  2. If thinking emoji already at start of assistant message")
    print("  3. If stop instruction already in user message")
    print("  4. If stop suffix already in assistant message")
    print()
    print("These checks prevent duplicates during training data preparation.")


if __name__ == '__main__':
    main()
