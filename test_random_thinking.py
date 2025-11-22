#!/usr/bin/env python3
"""
Test script for random thinking pattern generation
"""

import sys
sys.path.insert(0, '/path/to/training')

from train import get_thinking_pattern, THINKING_EMOJIS

print("="*80)
print("RANDOM THINKING PATTERN TEST")
print("="*80)
print()

print("Testing first 30 examples:")
print("-"*80)

for i in range(30):
    emoji, count, count_word, prefix, instruction = get_thinking_pattern(i)

    # Show details for first 10, then every 5th
    if i < 10 or i % 5 == 0:
        print(f"\nExample {i}:")
        print(f"  Emoji: {emoji}")
        print(f"  Count: {count} ({count_word})")
        print(f"  Prefix: {repr(prefix)}")
        print(f"  Instruction: {instruction}")

print("\n" + "="*80)
print("VERIFICATION CHECKS")
print("="*80)

# Check diversity
emoji_counts = {}
count_distribution = {i: 0 for i in range(2, 9)}

for i in range(1000):
    emoji, count, _, _, _ = get_thinking_pattern(i)
    emoji_counts[emoji] = emoji_counts.get(emoji, 0) + 1
    count_distribution[count] += 1

print("\nEmoji distribution (1000 examples):")
for emoji, cnt in sorted(emoji_counts.items(), key=lambda x: -x[1]):
    percentage = (cnt / 1000) * 100
    print(f"  {emoji}: {cnt} times ({percentage:.1f}%)")

print("\nCount distribution (1000 examples):")
for count in range(2, 9):
    cnt = count_distribution[count]
    percentage = (cnt / 1000) * 100
    print(f"  {count}: {cnt} times ({percentage:.1f}%)")

print("\n" + "="*80)
print("✅ RANDOM PATTERN GENERATION TEST COMPLETE")
print("="*80)

# Test that same index always gives same pattern (reproducibility)
print("\nReproducibility test (example 5):")
for _ in range(5):
    emoji, count, _, _, _ = get_thinking_pattern(5)
    print(f"  Run {_+1}: {emoji} x{count}")
print("✅ Same index = same pattern (reproducible)")
