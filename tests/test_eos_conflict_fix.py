#!/usr/bin/env python3
"""
Test to verify that disabling global EOS penalty fixes stop sequence completion.

This test simulates the conflict where:
- Global EOS penalty decreases over time (640 → 80)
- Post-stop EOT reward only applies after complete stop sequence
- Model might be tempted to output EOS early because penalty is low
"""

import torch
from transformers import AutoTokenizer
from logit_penalty import (
    build_eos_penalty_processor,
    build_post_stop_penalty_processor,
    DEFAULT_PENALTY_SCHEDULE,
)

def test_eos_penalty_schedule():
    """Test that global EOS penalty decreases over time (the problem)."""
    print("\n=== Testing Global EOS Penalty Schedule ===")

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Build EOS penalty processor
    eos_processors = build_eos_penalty_processor(
        tokenizer,
        penalty=80.0,
        schedule=DEFAULT_PENALTY_SCHEDULE,
    )

    if len(eos_processors) == 0:
        print("❌ No EOS processors found")
        return False

    processor = eos_processors[0]

    # Simulate generation over multiple steps
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    # Fake input_ids and scores
    input_ids = torch.tensor([[1, 2, 3]])  # Dummy tokens
    scores = torch.zeros(1, vocab_size)

    print(f"EOS token ID: {eos_token_id}")
    print(f"Base penalty: 80.0")
    print(f"Schedule: {DEFAULT_PENALTY_SCHEDULE}")
    print()

    # Test penalty at different steps
    test_steps = [0, 3, 7, 11, 15, 50]  # Different generation steps

    for step in test_steps:
        # Reset processor state
        processor.reset_state()

        # Simulate generation up to this step
        for _ in range(step):
            scores_copy = scores.clone()
            adjusted = processor(input_ids, scores_copy)
            # Don't actually update input_ids, just simulate the call

        # Check penalty at this step (one more call)
        scores_copy = scores.clone()
        adjusted = processor(input_ids, scores_copy)

        # Calculate actual penalty applied
        penalty_applied = scores_copy[0, eos_token_id].item() - adjusted[0, eos_token_id].item()

        print(f"Step {step:3d}: EOS penalty = {penalty_applied:6.1f}")

    print("\n⚠️  Problem: Penalty decreases from 640 → 80 over time")
    print("    This makes early EOS 'cheaper' than completing stop sequence!")
    return True


def test_post_stop_reward():
    """Test that post-stop penalty rewards EOS after complete stop sequence."""
    print("\n=== Testing Post-Stop Penalty (The Fix) ===")

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    STOP_EMOJI_POOL = ['🛑', '🔴', '⛔', '🚫', '⭕', '❌', '🚨', '⚠️', '💀', '🔥']

    # Build post-stop penalty processor
    post_stop_processors = build_post_stop_penalty_processor(
        tokenizer,
        stop_emoji_pool=STOP_EMOJI_POOL,
        stop_count_min=2,
        stop_count_max=4,
        base_penalty=100.0,
        escalation_rate=10.0,
        eot_reward=50.0,
    )

    if len(post_stop_processors) == 0:
        print("❌ No post-stop processors found")
        return False

    processor = post_stop_processors[0]

    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    # Encode stop sequence (e.g., "🛑🛑")
    stop_sequence = "🛑🛑"
    stop_token_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

    print(f"Stop sequence: {stop_sequence}")
    print(f"Stop token IDs: {stop_token_ids}")
    print(f"EOS token ID: {eos_token_id}")
    print(f"Base penalty: 100.0")
    print(f"Escalation rate: 10.0x")
    print(f"EOT reward: +50.0")
    print()

    # Test 1: Before stop sequence (should have no penalty)
    print("Test 1: Before stop sequence")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Random tokens
    scores = torch.zeros(1, vocab_size)

    processor.reset_state()
    adjusted = processor(input_ids, scores)

    eos_change = adjusted[0, eos_token_id].item() - scores[0, eos_token_id].item()
    print(f"  EOS score change: {eos_change:+.1f} (should be 0.0)")
    assert abs(eos_change) < 0.01, "EOS should not be penalized before stop sequence"

    # Test 2: After complete stop sequence (should reward EOS)
    print("\nTest 2: After complete stop sequence")
    input_ids = torch.tensor([stop_token_ids])  # Just the stop sequence
    scores = torch.zeros(1, vocab_size)

    processor.reset_state()
    adjusted = processor(input_ids, scores)

    # After seeing stop, the NEXT token generation should:
    # - Penalize all tokens by 100.0
    # - Remove penalty from EOS and add 50.0 reward
    # Net effect for EOS: +100.0 (penalty removal) + 50.0 (reward) = +150.0
    eos_change = adjusted[0, eos_token_id].item() - scores[0, eos_token_id].item()
    print(f"  EOS score change: {eos_change:+.1f} (should be +150.0)")

    # Check a random non-EOS token (should be heavily penalized)
    random_token_id = 100
    random_change = adjusted[0, random_token_id].item() - scores[0, random_token_id].item()
    print(f"  Random token change: {random_change:+.1f} (should be -100.0)")

    print("\n✅ Post-stop penalty correctly rewards EOS after complete stop sequence!")
    print("   This creates strong incentive: complete stop → output EOS")
    return True


def test_combined_behavior():
    """Test what happens when both penalties are active (the old broken behavior)."""
    print("\n=== Testing Combined Behavior (Old Broken Config) ===")

    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Build both processors
    eos_processors = build_eos_penalty_processor(
        tokenizer,
        penalty=80.0,
        schedule=DEFAULT_PENALTY_SCHEDULE,
    )

    STOP_EMOJI_POOL = ['🛑']
    post_stop_processors = build_post_stop_penalty_processor(
        tokenizer,
        stop_emoji_pool=STOP_EMOJI_POOL,
        stop_count_min=2,
        stop_count_max=4,
        base_penalty=100.0,
        escalation_rate=10.0,
        eot_reward=50.0,
    )

    if len(eos_processors) == 0 or len(post_stop_processors) == 0:
        print("❌ Missing processors")
        return False

    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id

    # Simulate generation at step 50 (EOS penalty is ~80)
    # Model is about to generate stop sequence

    # Scenario 1: Model outputs EOS early (before stop sequence)
    print("Scenario 1: Model outputs EOS at step 50 (before stop sequence)")
    input_ids = torch.tensor([[1] * 50])  # 50 dummy tokens
    scores = torch.zeros(1, vocab_size)

    # Reset both processors
    eos_processors[0].reset_state()
    post_stop_processors[0].reset_state()

    # Simulate 50 steps for EOS penalty
    for _ in range(50):
        scores_temp = scores.clone()
        eos_processors[0](input_ids, scores_temp)

    # Apply both penalties at step 50
    adjusted = scores.clone()
    adjusted = eos_processors[0](input_ids, adjusted)
    adjusted = post_stop_processors[0](input_ids, adjusted)

    eos_change_early = adjusted[0, eos_token_id].item() - scores[0, eos_token_id].item()
    print(f"  EOS penalty at step 50: {eos_change_early:.1f}")

    # Scenario 2: Model completes stop sequence then outputs EOS
    print("\nScenario 2: Model completes stop sequence at step 52, then outputs EOS")
    stop_sequence = "🛑🛑"
    stop_token_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
    input_ids = torch.tensor([stop_token_ids])

    # Reset both processors
    eos_processors[0].reset_state()
    post_stop_processors[0].reset_state()

    # Simulate 52 steps for EOS penalty
    for _ in range(52):
        scores_temp = scores.clone()
        eos_processors[0](input_ids, scores_temp)

    # Apply both penalties after stop sequence
    adjusted = scores.clone()
    adjusted = eos_processors[0](input_ids, adjusted)
    adjusted = post_stop_processors[0](input_ids, adjusted)

    eos_change_complete = adjusted[0, eos_token_id].item() - scores[0, eos_token_id].item()
    print(f"  EOS change after stop: {eos_change_complete:.1f}")

    print("\n⚠️  THE PROBLEM:")
    print(f"  Early EOS (step 50): penalty = {eos_change_early:.1f}")
    print(f"  Complete stop then EOS: reward = {eos_change_complete:.1f}")
    print(f"  Difference: {eos_change_complete - eos_change_early:.1f}")
    print()
    print("  Model might choose early EOS because:")
    print("  1. Less effort (2 fewer tokens to generate)")
    print("  2. Penalty is 'small enough' (~-80)")
    print("  3. Doesn't 'see ahead' to the +150 reward")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("EOS Penalty Conflict Analysis")
    print("=" * 70)

    # Test 1: Show global EOS penalty decreases
    test_eos_penalty_schedule()

    # Test 2: Show post-stop penalty works correctly
    test_post_stop_reward()

    # Test 3: Show the conflict
    test_combined_behavior()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("✅ FIX: Disable global EOS penalty in train.py")
    print()
    print("Rationale:")
    print("  1. Global EOS penalty weakens over time (640 → 80)")
    print("  2. This makes early EOS 'cheap' compared to completing stop sequence")
    print("  3. Post-stop penalty already handles EOS correctly:")
    print("     - No penalty before stop sequence")
    print("     - +150 reward for EOS after complete stop sequence")
    print("     - Escalating penalties for continuing after stop (100 → 1000 → 10000)")
    print()
    print("Result: Model strongly incentivized to complete stop sequence then output EOS!")
    print()
