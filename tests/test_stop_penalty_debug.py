#!/usr/bin/env python3
"""
Debug why stop penalties aren't working.
Tests the PostStopPenalty processor with the actual tokenizer.
"""

import torch
from transformers import AutoTokenizer
from logit_penalty import build_post_stop_penalty_processor

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("consolidated_models/20251119_152444")

# Build penalty processor (same as train.py)
STOP_EMOJI_POOL = ["🛑", "⛔", "🚫", "❌", "🔴", "⏹️", "🔚", "✋", "🚦", "🛡️"]
processors = build_post_stop_penalty_processor(
    tokenizer,
    stop_emoji_pool=STOP_EMOJI_POOL,
    stop_count_min=2,
    stop_count_max=4,
    base_penalty=10.0,
    escalation_rate=3.0,
    eot_reward=5.0,
)

processor = processors[0]  # Get the actual processor

# Test: Encode the stop sequence
test_sequence = "🛑🛑🛑"
tokens = tokenizer.encode(test_sequence, add_special_tokens=False)
print(f"Stop sequence: {test_sequence}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")
print()

# Check what the processor has stored
print("Processor's stored stop sequences:")
for (emoji, count), token_ids in processor.stop_sequences.items():
    print(f"  {emoji} x{count}: {token_ids} (length={len(token_ids)})")
print()

# Simulate generation
print("Simulating generation after stop emojis:")
print("=" * 60)

# Create fake input_ids that end with the stop sequence
fake_prefix = tokenizer.encode("Test response", add_special_tokens=False)
fake_input_ids = torch.tensor([fake_prefix + tokens])  # Batch size 1

print(f"Input IDs: {fake_input_ids.tolist()}")
print(f"Last {len(tokens)} tokens: {fake_input_ids[0, -len(tokens):].tolist()}")
print()

# Create fake scores (logits)
vocab_size = len(tokenizer)
fake_scores = torch.randn(1, vocab_size)  # Random logits

# Apply processor
print("Calling processor...")
adjusted_scores = processor(fake_input_ids, fake_scores)

print(f"Processor state:")
print(f"  stop_seen: {processor.stop_seen}")
print(f"  tokens_after_stop: {processor.tokens_after_stop}")
print(f"  detected_emoji: {processor.detected_emoji}")
print(f"  detected_count: {processor.detected_count}")
print()

if processor.stop_seen:
    print("✅ Stop sequence WAS detected!")

    # Check EOT token
    eos_id = tokenizer.eos_token_id
    print(f"EOS token ID: {eos_id}")
    print(f"EOT IDs in processor: {processor.eot_ids}")

    # Check penalty magnitude
    original_logit = fake_scores[0, eos_id].item()
    adjusted_logit = adjusted_scores[0, eos_id].item()
    print(f"EOS original logit: {original_logit:.2f}")
    print(f"EOS adjusted logit: {adjusted_logit:.2f}")
    print(f"Boost: {adjusted_logit - original_logit:.2f}")

    # Check penalty on "The" token
    the_token = tokenizer.encode("The", add_special_tokens=False)[0]
    orig_the = fake_scores[0, the_token].item()
    adj_the = adjusted_scores[0, the_token].item()
    print(f"'The' token ID: {the_token}")
    print(f"'The' original logit: {orig_the:.2f}")
    print(f"'The' adjusted logit: {adj_the:.2f}")
    print(f"Penalty: {adj_the - orig_the:.2f}")
else:
    print("❌ Stop sequence was NOT detected!")
    print("This is the problem - the penalty isn't triggering.")
