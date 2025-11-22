#!/usr/bin/env python3
"""
Test script to verify stop emoji formatting
"""

import json
import sys
from pathlib import Path

# Import from train.py
sys.path.insert(0, str(Path(__file__).parent))

# Define constants (same as train.py)
THINKING_EMOJI = "🤔"
THINKING_INSTRUCTION = f"For this task, think with {THINKING_EMOJI} /four/ times."
THINKING_PREFIX = THINKING_EMOJI * 4 + "\n"

STOP_EMOJI = "🛑"
STOP_INSTRUCTION = f"When finished, emit {STOP_EMOJI} /three/ times to signal completion."
STOP_SUFFIX = "\n" + STOP_EMOJI * 3

def enforce_thinking_requirement(messages):
    """Apply thinking emojis"""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "user":
            if THINKING_INSTRUCTION not in content:
                content = content.rstrip() + "\n\n" + THINKING_INSTRUCTION
        elif role == "assistant":
            if not content.startswith(THINKING_PREFIX):
                content = THINKING_PREFIX + content.lstrip()
        msg["content"] = content
    return messages

def enforce_stop_requirement(messages):
    """Apply stop emojis"""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "user":
            if STOP_INSTRUCTION not in content:
                content = content.rstrip() + "\n\n" + STOP_INSTRUCTION
        elif role == "assistant":
            if not content.endswith(STOP_SUFFIX):
                content = content.rstrip() + STOP_SUFFIX
        msg["content"] = content
    return messages

def test_formatting(file_path):
    """Test formatting on a .jsonl file"""
    print(f"🧪 Testing stop emoji formatting on: {file_path}\n")
    print("="*80)

    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            example = json.loads(line)
            messages = example.get('messages', [])

            # Apply formatting (same order as train.py)
            messages = enforce_thinking_requirement(messages)
            messages = enforce_stop_requirement(messages)

            print(f"\n📝 Example {i}:")
            print("-"*80)

            for msg in messages:
                role = msg.get('role')
                content = msg.get('content', '')

                print(f"\n[{role.upper()}]")
                print(content)
                print()

            # Verify formatting
            user_msgs = [m for m in messages if m['role'] == 'user']
            assistant_msgs = [m for m in messages if m['role'] == 'assistant']

            checks = []

            # Check user messages have both instructions
            for msg in user_msgs:
                content = msg['content']
                has_think = THINKING_INSTRUCTION in content
                has_stop = STOP_INSTRUCTION in content
                checks.append(("User has think instruction", has_think))
                checks.append(("User has stop instruction", has_stop))

            # Check assistant messages have prefix and suffix
            for msg in assistant_msgs:
                content = msg['content']
                has_prefix = content.startswith(THINKING_PREFIX)
                has_suffix = content.endswith(STOP_SUFFIX)
                checks.append(("Assistant has think prefix", has_prefix))
                checks.append(("Assistant has stop suffix", has_suffix))

            print("✅ Validation Checks:")
            for check_name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"   {status} {check_name}")

            print("="*80)

if __name__ == "__main__":
    test_file = Path("test_stop_emoji.jsonl")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)

    test_formatting(test_file)
    print("\n✅ Formatting test complete!")
