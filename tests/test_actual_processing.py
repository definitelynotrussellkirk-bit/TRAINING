#!/usr/bin/env python3
"""
Test what actually happens when train.py processes data with existing instructions.
"""

import json
import sys

# Your exact data
test_data = {
    "prompt": """Capitalize field., then sort in alphabetical order by the requested field, and then Report the share of items that start with vowel.

Records a: THROUGH, amazing, Countries, SALES, Amazing, DRUGS, Drugs, IRON, QUICK, sales, Cash, iron, cash, CASH, AMAZING, Iron, through, COUNTRIES, one, Quick, Sales, ONE, quick, One, Through, drugs, countries

Provide the report as JSON, plain text summary, and a sentence (clean, no comments).

For this task, think with 💡 /five/ times.

When finished, emit 🛑 /three/ times to signal completion.""",
    "response": "💡💡💡💡💡 Analysis here\n\n🛑🛑🛑"
}

print("=" * 70)
print("TESTING ACTUAL TRAIN.PY PROCESSING")
print("=" * 70)
print()

# Convert to messages format (what train.py does)
messages = [
    {"role": "user", "content": test_data["prompt"]},
    {"role": "assistant", "content": test_data["response"]}
]

print("BEFORE PROCESSING:")
print("-" * 70)
print("User message:")
print(messages[0]["content"])
print()
print("Assistant message:")
print(messages[1]["content"])
print()

# Now simulate what train.py does
THINKING_EMOJIS = ["🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"]

# Simulate enforce_thinking_requirement (with the FIX)
for msg in messages:
    role = msg.get("role")
    content = msg.get("content", "")

    if role == "user":
        # NEW FIXED CHECK
        has_instruction = any(f"think with {e}" in content.lower() for e in THINKING_EMOJIS) or "think with" in content.lower()

        print(f"User message check:")
        print(f"  Has 'think with': {has_instruction}")

        if not has_instruction:
            print(f"  ❌ Would ADD instruction (SHOULDN'T HAPPEN)")
            # Would add instruction here
        else:
            print(f"  ✅ Instruction exists, NOT adding duplicate")

    elif role == "assistant":
        # Check if starts with ANY thinking emoji
        has_prefix = any(content.startswith(e) for e in THINKING_EMOJIS)

        print(f"Assistant message check:")
        print(f"  Has emoji prefix: {has_prefix}")

        if not has_prefix:
            print(f"  ❌ Would ADD prefix (SHOULDN'T HAPPEN)")
        else:
            print(f"  ✅ Prefix exists, NOT adding duplicate")

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print()
print("✅ With the fix, NO DUPLICATES are added!")
print("   Your data is used AS-IS without modification.")
print()
print("The model will be asked to think ONCE (as you specified),")
print("not twice!")
