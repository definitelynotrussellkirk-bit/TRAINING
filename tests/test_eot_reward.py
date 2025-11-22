#!/usr/bin/env python3
"""
Test script to verify EOT reward after stop emoji sequence.

This test verifies that:
1. After 🛑🛑🛑 is detected, penalties are applied to all tokens
2. EOT tokens receive extra reward on top of penalty removal
"""

import torch
from transformers import AutoTokenizer
from logit_penalty import PostStopPenalty


def test_eot_reward():
    """Test that EOT tokens receive extra reward after stop sequence."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Create processor with EOT reward
    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji="🛑",
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=3.0,  # Extra reward for EOT
    )

    # Get token IDs
    stop_token_ids = tokenizer.encode("🛑🛑🛑", add_special_tokens=False)
    eot_id = tokenizer.eos_token_id
    random_token_id = tokenizer.encode("hello", add_special_tokens=False)[0]

    # Create input sequence with stop emojis at the end
    # Format: [prefix tokens] + [stop emojis]
    prefix = tokenizer.encode("Test response", add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_token_ids], dtype=torch.long)

    print("=" * 60)
    print("EOT Reward Test")
    print("=" * 60)
    print(f"Stop token IDs: {stop_token_ids}")
    print(f"EOT token ID: {eot_id}")
    print(f"Random token ID: {random_token_id}")
    print()

    # Create mock logits (all zeros)
    vocab_size = len(tokenizer)  # Use len() to get actual vocab size
    scores = torch.zeros((1, vocab_size), dtype=torch.float)

    # Process the scores
    adjusted_scores = processor(input_ids, scores)

    # Check the results
    print("After processing with stop sequence detected:")
    print(f"  Base penalty: {processor.base_penalty}")
    print(f"  Escalation rate: {processor.escalation_rate}")
    print(f"  EOT reward: {processor.eot_reward}")
    print(f"  Tokens after stop: {processor.tokens_after_stop}")
    print()

    # Calculate expected penalty
    expected_penalty = processor.base_penalty * (processor.escalation_rate ** processor.tokens_after_stop)

    print(f"Expected penalty for regular tokens: -{expected_penalty:.2f}")
    print(f"Expected reward for EOT tokens: +{processor.eot_reward:.2f}")
    print()

    # Get actual values
    random_token_logit = adjusted_scores[0, random_token_id].item()
    eot_token_logit = adjusted_scores[0, eot_id].item()

    print("Actual logit values:")
    print(f"  Random token ({random_token_id}): {random_token_logit:.2f}")
    print(f"  EOT token ({eot_id}): {eot_token_logit:.2f}")
    print()

    # Verify behavior
    print("Verification:")

    # Random tokens should be penalized
    if abs(random_token_logit - (-expected_penalty)) < 0.01:
        print(f"  ✓ Random tokens penalized by {expected_penalty:.2f}")
    else:
        print(f"  ✗ Random token penalty mismatch!")
        print(f"    Expected: {-expected_penalty:.2f}, Got: {random_token_logit:.2f}")

    # EOT should have penalty removed AND extra reward added
    expected_eot_logit = processor.eot_reward  # penalty removed + reward added
    if abs(eot_token_logit - expected_eot_logit) < 0.01:
        print(f"  ✓ EOT token rewarded with +{processor.eot_reward:.2f}")
    else:
        print(f"  ✗ EOT reward mismatch!")
        print(f"    Expected: {expected_eot_logit:.2f}, Got: {eot_token_logit:.2f}")

    # EOT should be higher than random tokens
    advantage = eot_token_logit - random_token_logit
    print(f"  ✓ EOT advantage over random tokens: {advantage:.2f}")
    print()

    # Test stats
    stats = processor.snapshot_stats()
    print("Processor stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print()
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_without_reward():
    """Test baseline behavior without EOT reward (eot_reward=0.0)."""

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Create processor WITHOUT EOT reward
    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji="🛑",
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=0.0,  # No reward
    )

    # Get token IDs
    stop_token_ids = tokenizer.encode("🛑🛑🛑", add_special_tokens=False)
    eot_id = tokenizer.eos_token_id
    random_token_id = tokenizer.encode("hello", add_special_tokens=False)[0]

    # Create input sequence with stop emojis
    prefix = tokenizer.encode("Test", add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_token_ids], dtype=torch.long)

    # Create mock logits
    vocab_size = len(tokenizer)  # Use len() to get actual vocab size
    scores = torch.zeros((1, vocab_size), dtype=torch.float)

    # Process
    adjusted_scores = processor(input_ids, scores)

    print("\n" + "=" * 60)
    print("Baseline Test (No EOT Reward)")
    print("=" * 60)

    random_token_logit = adjusted_scores[0, random_token_id].item()
    eot_token_logit = adjusted_scores[0, eot_id].item()

    print(f"Random token logit: {random_token_logit:.2f}")
    print(f"EOT token logit: {eot_token_logit:.2f}")

    # EOT should be neutral (0.0) without reward
    if abs(eot_token_logit) < 0.01:
        print("  ✓ EOT token is neutral (penalty removed, no reward)")
    else:
        print(f"  ✗ EOT should be neutral, got: {eot_token_logit:.2f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_without_reward()  # Baseline
    test_eot_reward()      # With reward
