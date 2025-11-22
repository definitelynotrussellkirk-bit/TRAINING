#!/usr/bin/env python3
"""
Test the exact example the user provided.
"""

# Simulate the user's data that already has thinking instructions
user_data = """Capitalize field., then sort in alphabetical order by the requested field, and then Report the share of items that start with vowel.

Records a: THROUGH, amazing, Countries, SALES, Amazing, DRUGS, Drugs, IRON, QUICK, sales, Cash, iron, cash, CASH, AMAZING, Iron, through, COUNTRIES, one, Quick, Sales, ONE, quick, One, Through, drugs, countries

Provide the report as JSON, plain text summary, and a sentence (clean, no comments).

For this task, think with 💡 /five/ times.

When finished, emit 🛑 /three/ times to signal completion."""

print("=" * 70)
print("TEST: User's Exact Example")
print("=" * 70)
print()
print("Original content:")
print(user_data)
print()

# Check what the system sees
import re

# Check for "think with" pattern (case insensitive)
has_think_with = "think with" in user_data.lower()
print(f"System detects 'think with': {has_think_with}")

# Count how many thinking instructions
think_count = len(re.findall(r'think with [🤔💭🧠💡🎯🔍🤨🧐⚡✨]', user_data, re.IGNORECASE))
print(f"Number of 'think with' instructions found: {think_count}")

if think_count == 0:
    print("❌ WOULD ADD ANOTHER INSTRUCTION (bug!)")
elif think_count == 1:
    print("✅ CORRECTLY DETECTS EXISTING INSTRUCTION (no duplicate)")
else:
    print(f"⚠️  ALREADY HAS {think_count} INSTRUCTIONS (duplicates present!)")

print()
print("=" * 70)

# Show what the check does
THINKING_EMOJIS = ["🤔", "💭", "🧠", "💡", "🎯", "🔍", "🤨", "🧐", "⚡", "✨"]

print("NEW FIXED CHECK:")
print("  Checks: any('think with {e}' in content.lower() for e in THINKING_EMOJIS)")
print("         OR 'think with' in content.lower()")
print()

has_instruction_new = any(f"think with {e}" in user_data.lower() for e in THINKING_EMOJIS) or "think with" in user_data.lower()
print(f"  Result: {has_instruction_new}")

if has_instruction_new:
    print("  ✅ Will NOT add duplicate!")
else:
    print("  ❌ Will add duplicate!")

print()
print("OLD BUGGY CHECK:")
print("  Checked only: 'think with {specific_emoji}' that it wanted to add")
print("  Problem: Different emoji = not detected = duplicate added")
print()
print("=" * 70)
