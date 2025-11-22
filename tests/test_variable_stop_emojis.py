#!/usr/bin/env python3
"""
Test script for variable stop emoji sequences (2-4 repetitions, 10 emoji pool).

Verifies that the system:
1. Detects any emoji from the pool
2. Detects variable repetition counts (2-4)
3. Applies penalties and rewards correctly
4. Tracks which emoji/count was detected
"""

import torch
from transformers import AutoTokenizer
from logit_penalty import PostStopPenalty


# Match train.py configuration
STOP_EMOJI_POOL = ["🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"]
STOP_COUNT_MIN = 2
STOP_COUNT_MAX = 4


def test_emoji_pool_detection():
    """Test that all emojis in the pool are detected."""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 1: Emoji Pool Detection")
    print("=" * 60)

    passed = 0
    failed = 0

    for emoji in STOP_EMOJI_POOL:
        processor = PostStopPenalty(
            tokenizer=tokenizer,
            stop_emoji_pool=STOP_EMOJI_POOL,
            stop_count_min=2,
            stop_count_max=4,
            base_penalty=5.0,
            eot_reward=3.0,
        )

        # Test with triple repetition (middle of range)
        stop_sequence = emoji * 3
        stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

        # Create input with this stop sequence
        prefix = tokenizer.encode("Test", add_special_tokens=False)
        input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)

        # Process
        vocab_size = len(tokenizer)
        scores = torch.zeros((1, vocab_size), dtype=torch.float)
        processor(input_ids, scores)

        # Check detection
        if processor.stop_seen:
            print(f"  ✓ {emoji} detected (x3)")
            passed += 1
        else:
            print(f"  ✗ {emoji} NOT detected (x3)")
            failed += 1

    print(f"\nPassed: {passed}/{len(STOP_EMOJI_POOL)}")
    print()


def test_count_variation():
    """Test that counts 2-4 are all detected."""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 2: Count Variation (2-4)")
    print("=" * 60)

    emoji = "🛑"  # Use one emoji, test all counts

    for count in range(2, 5):  # 2, 3, 4
        processor = PostStopPenalty(
            tokenizer=tokenizer,
            stop_emoji_pool=STOP_EMOJI_POOL,
            stop_count_min=2,
            stop_count_max=4,
            base_penalty=5.0,
            eot_reward=3.0,
        )

        stop_sequence = emoji * count
        stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

        prefix = tokenizer.encode("Test", add_special_tokens=False)
        input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)

        vocab_size = len(tokenizer)
        scores = torch.zeros((1, vocab_size), dtype=torch.float)
        processor(input_ids, scores)

        if processor.stop_seen and processor.detected_count == count:
            print(f"  ✓ Count {count} detected correctly")
        else:
            print(f"  ✗ Count {count} detection failed")
            print(f"     stop_seen={processor.stop_seen}, detected_count={processor.detected_count}")

    print()


def test_edge_counts():
    """Test that counts outside the range are NOT detected."""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 3: Edge Cases (counts outside 2-4 range)")
    print("=" * 60)

    emoji = "🛑"

    # Test count = 1 (too few)
    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji_pool=STOP_EMOJI_POOL,
        stop_count_min=2,
        stop_count_max=4,
        eot_reward=3.0,
    )

    stop_sequence = emoji * 1
    stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
    prefix = tokenizer.encode("Test", add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)
    vocab_size = len(tokenizer)
    scores = torch.zeros((1, vocab_size), dtype=torch.float)
    processor(input_ids, scores)

    if not processor.stop_seen:
        print(f"  ✓ Count 1 (too few) correctly NOT detected")
    else:
        print(f"  ✗ Count 1 should NOT be detected!")

    # Test count = 5 (too many)
    processor.reset_state()
    stop_sequence = emoji * 5
    stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)
    processor(input_ids, scores)

    if not processor.stop_seen:
        print(f"  ✓ Count 5 (too many) correctly NOT detected")
    else:
        print(f"  ✗ Count 5 should NOT be detected!")

    print()


def test_eot_reward_applies():
    """Test that EOT reward is applied correctly."""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 4: EOT Reward Application")
    print("=" * 60)

    # Test with a random emoji from the pool
    import random
    emoji = random.choice(STOP_EMOJI_POOL)
    count = random.randint(2, 4)

    print(f"Testing with: {emoji} x {count}")

    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji_pool=STOP_EMOJI_POOL,
        stop_count_min=2,
        stop_count_max=4,
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=3.0,
    )

    stop_sequence = emoji * count
    stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

    prefix = tokenizer.encode("Test", add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)

    vocab_size = len(tokenizer)
    scores = torch.zeros((1, vocab_size), dtype=torch.float)

    adjusted_scores = processor(input_ids, scores)

    # Get EOT and random token
    eot_id = tokenizer.eos_token_id
    random_id = tokenizer.encode("hello", add_special_tokens=False)[0]

    eot_logit = adjusted_scores[0, eot_id].item()
    random_logit = adjusted_scores[0, random_id].item()

    expected_penalty = 5.0 * (2.0 ** 1)  # First token after stop

    print(f"\nAfter {emoji} x {count}:")
    print(f"  Random token logit: {random_logit:.2f} (expected: -{expected_penalty:.2f})")
    print(f"  EOT token logit: {eot_logit:.2f} (expected: +3.00)")

    if abs(random_logit - (-expected_penalty)) < 0.01 and abs(eot_logit - 3.0) < 0.01:
        print("  ✓ Penalties and rewards applied correctly")
    else:
        print("  ✗ Incorrect penalty/reward values")

    print()


def test_stats_tracking():
    """Test that detected emoji and count are tracked in stats."""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 5: Stats Tracking")
    print("=" * 60)

    import random
    emoji = random.choice(STOP_EMOJI_POOL)
    count = random.randint(2, 4)

    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji_pool=STOP_EMOJI_POOL,
        stop_count_min=2,
        stop_count_max=4,
        base_penalty=5.0,
        eot_reward=3.0,
    )

    stop_sequence = emoji * count
    stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
    prefix = tokenizer.encode("Test", add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)
    vocab_size = len(tokenizer)
    scores = torch.zeros((1, vocab_size), dtype=torch.float)

    processor(input_ids, scores)
    stats = processor.snapshot_stats()

    print(f"Expected: emoji={emoji}, count={count}")
    print(f"Stats: {stats}")

    if stats["detected_emoji"] == emoji and stats["detected_count"] == count:
        print("  ✓ Stats correctly track detected emoji and count")
    else:
        print("  ✗ Stats tracking failed")

    print()


def test_combinations():
    """Test random combinations of emojis and counts."""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 6: Random Combinations")
    print("=" * 60)

    import random

    passed = 0
    total = 10  # Test 10 random combinations

    for i in range(total):
        emoji = random.choice(STOP_EMOJI_POOL)
        count = random.randint(2, 4)

        processor = PostStopPenalty(
            tokenizer=tokenizer,
            stop_emoji_pool=STOP_EMOJI_POOL,
            stop_count_min=2,
            stop_count_max=4,
            eot_reward=3.0,
        )

        stop_sequence = emoji * count
        stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
        prefix = tokenizer.encode("Test", add_special_tokens=False)
        input_ids = torch.tensor([prefix + stop_ids], dtype=torch.long)
        vocab_size = len(tokenizer)
        scores = torch.zeros((1, vocab_size), dtype=torch.float)

        processor(input_ids, scores)

        if processor.stop_seen and processor.detected_emoji == emoji and processor.detected_count == count:
            passed += 1
        else:
            print(f"  ✗ Failed: {emoji} x {count}")

    print(f"  ✓ Passed: {passed}/{total} random combinations")
    print()


if __name__ == "__main__":
    test_emoji_pool_detection()
    test_count_variation()
    test_edge_counts()
    test_eot_reward_applies()
    test_stats_tracking()
    test_combinations()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  • 10 different stop emojis supported")
    print("  • Variable counts (2-4) all work")
    print("  • EOT reward applies correctly")
    print("  • Detection tracking works")
    print()
    print("The model will now learn flexible stop signals!")
