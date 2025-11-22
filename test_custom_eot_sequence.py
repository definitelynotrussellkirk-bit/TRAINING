#!/usr/bin/env python3
"""
Test script to verify custom EOT sequence support.

This test verifies forward compatibility with custom EOT sequences.
"""

import torch
from transformers import AutoTokenizer
from logit_penalty import PostStopPenalty


def test_default_eot():
    """Test default behavior - uses tokenizer.eos_token_id"""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji="🛑",
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=3.0,
        eot_sequence=None,  # Default: use tokenizer EOS
    )

    print("=" * 60)
    print("Test 1: Default EOT (tokenizer.eos_token_id)")
    print("=" * 60)
    print(f"EOT IDs: {processor.eot_ids}")
    print(f"Expected: {{{tokenizer.eos_token_id}}}")

    if processor.eot_ids == {tokenizer.eos_token_id}:
        print("✓ Default EOT matches tokenizer.eos_token_id")
    else:
        print("✗ Default EOT mismatch!")
    print()


def test_custom_single_token():
    """Test custom EOT sequence - single token"""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Use a specific token as custom EOT
    custom_eot = "<|im_end|>"

    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji="🛑",
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=3.0,
        eot_sequence=custom_eot,  # Custom EOT token
    )

    expected_ids = set(tokenizer.encode(custom_eot, add_special_tokens=False))

    print("=" * 60)
    print(f"Test 2: Custom Single Token EOT ('{custom_eot}')")
    print("=" * 60)
    print(f"Custom EOT: {custom_eot}")
    print(f"Encoded IDs: {expected_ids}")
    print(f"Processor EOT IDs: {processor.eot_ids}")

    if processor.eot_ids == expected_ids:
        print(f"✓ Custom EOT correctly set to {expected_ids}")
    else:
        print(f"✗ Custom EOT mismatch!")
        print(f"  Expected: {expected_ids}")
        print(f"  Got: {processor.eot_ids}")
    print()


def test_custom_multi_token():
    """Test custom EOT sequence - multiple tokens (e.g., double EOT)"""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Use double EOT as custom sequence
    custom_eot = "<|im_end|><|im_end|>"

    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji="🛑",
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=3.0,
        eot_sequence=custom_eot,  # Double EOT
    )

    expected_ids = set(tokenizer.encode(custom_eot, add_special_tokens=False))

    print("=" * 60)
    print(f"Test 3: Custom Multi-Token EOT ('{custom_eot}')")
    print("=" * 60)
    print(f"Custom EOT: {custom_eot}")
    print(f"Encoded IDs (set): {expected_ids}")
    print(f"Processor EOT IDs: {processor.eot_ids}")

    # Note: Using a set means duplicates collapse
    # If "<|im_end|>" encodes to [123], then "<|im_end|><|im_end|>" encodes to [123, 123]
    # But as a set, it becomes {123}

    if processor.eot_ids == expected_ids:
        print(f"✓ Custom multi-token EOT correctly set")
        print(f"  (Note: Set automatically deduplicates token IDs)")
    else:
        print(f"✗ Custom EOT mismatch!")
    print()


def test_reward_applies_to_custom_eot():
    """Test that rewards actually apply to custom EOT tokens"""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    # Use a word as custom EOT for easier testing
    custom_eot = "STOP"

    processor = PostStopPenalty(
        tokenizer=tokenizer,
        stop_emoji="🛑",
        base_penalty=5.0,
        escalation_rate=2.0,
        eot_reward=3.0,
        eot_sequence=custom_eot,
    )

    # Get token IDs
    stop_token_ids = tokenizer.encode("🛑🛑🛑", add_special_tokens=False)
    custom_eot_ids = tokenizer.encode(custom_eot, add_special_tokens=False)
    random_token_id = tokenizer.encode("hello", add_special_tokens=False)[0]

    # Create input with stop sequence
    prefix = tokenizer.encode("Test", add_special_tokens=False)
    input_ids = torch.tensor([prefix + stop_token_ids], dtype=torch.long)

    # Create logits
    vocab_size = len(tokenizer)
    scores = torch.zeros((1, vocab_size), dtype=torch.float)

    # Process
    adjusted_scores = processor(input_ids, scores)

    print("=" * 60)
    print(f"Test 4: Reward Applies to Custom EOT ('{custom_eot}')")
    print("=" * 60)
    print(f"Custom EOT: {custom_eot}")
    print(f"Custom EOT token IDs: {custom_eot_ids}")
    print(f"Processor EOT IDs: {processor.eot_ids}")
    print()

    # Check each custom EOT token gets rewarded
    all_rewarded = True
    for eot_id in custom_eot_ids:
        eot_logit = adjusted_scores[0, eot_id].item()
        print(f"Token ID {eot_id}: logit = {eot_logit:.2f}")
        if abs(eot_logit - 3.0) < 0.01:
            print(f"  ✓ Rewarded correctly")
        else:
            print(f"  ✗ Expected +3.00, got {eot_logit:.2f}")
            all_rewarded = False

    # Check random token is penalized
    random_logit = adjusted_scores[0, random_token_id].item()
    print(f"\nRandom token {random_token_id}: logit = {random_logit:.2f}")
    if abs(random_logit - (-10.0)) < 0.01:
        print(f"  ✓ Penalized correctly")
    else:
        print(f"  ✗ Expected -10.00, got {random_logit:.2f}")
        all_rewarded = False

    if all_rewarded:
        print("\n✓ All custom EOT tokens rewarded correctly!")
    else:
        print("\n✗ Some tokens not rewarded correctly")
    print()


def test_future_compatibility():
    """Test various future EOT sequence formats"""
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    print("=" * 60)
    print("Test 5: Future Compatibility - Various EOT Formats")
    print("=" * 60)

    test_cases = [
        None,  # Default
        "<|im_end|>",  # Single special token
        "<|im_end|><|im_end|>",  # Double EOT
        "END",  # Custom word
        "🛑",  # Emoji (same as stop, just testing)
        "<|end|>",  # Hypothetical future token
    ]

    for eot_seq in test_cases:
        try:
            processor = PostStopPenalty(
                tokenizer=tokenizer,
                stop_emoji="🛑",
                eot_sequence=eot_seq,
            )

            display = f"'{eot_seq}'" if eot_seq else "None (default)"
            print(f"✓ {display}: {processor.eot_ids}")
        except Exception as e:
            print(f"✗ {eot_seq}: FAILED - {e}")

    print()


if __name__ == "__main__":
    test_default_eot()
    test_custom_single_token()
    test_custom_multi_token()
    test_reward_applies_to_custom_eot()
    test_future_compatibility()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  The system now supports:")
    print("    • Default EOT (tokenizer.eos_token_id)")
    print("    • Custom single token EOT")
    print("    • Custom multi-token sequences")
    print("    • Any string-based EOT sequence")
    print()
    print("  To use custom EOT in train.py:")
    print('    eot_sequence="<|end|>"  # Single token')
    print('    eot_sequence="<|end|><|end|>"  # Double emission')
    print('    eot_sequence="STOP"  # Custom word')
