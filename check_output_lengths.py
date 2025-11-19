#!/usr/bin/env python3
"""
Check output (assistant response) lengths in training data.
Ensures model outputs aren't getting truncated.
"""

import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np

def check_output_lengths(data_file, config_file="config.json", sample_size=100):
    """Check assistant output lengths in training data."""

    # Load config
    with open(config_file) as f:
        config = json.load(f)

    max_length = config.get("max_length", 2048)
    base_model = config.get("base_model")

    print(f"📊 Checking OUTPUT lengths in: {data_file}")
    print(f"   Config max_length: {max_length}")
    print(f"   Base model: {base_model}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Sample examples
    examples = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            examples.append(json.loads(line))

    print(f"Sampled {len(examples)} examples\n")

    # Analyze OUTPUT lengths (assistant responses only)
    output_lengths = []
    full_lengths = []
    truncated_outputs = 0
    truncated_full = 0

    for ex in examples:
        messages = ex.get("messages", [])

        # Extract assistant response (last message should be assistant)
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break

        if assistant_msg:
            # Tokenize just the output
            output_tokens = tokenizer(assistant_msg, add_special_tokens=False)
            output_len = len(output_tokens['input_ids'])
            output_lengths.append(output_len)

            if output_len > max_length:
                truncated_outputs += 1

        # Also check full conversation length for context
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_tokens = tokenizer(full_text, add_special_tokens=True)
        full_len = len(full_tokens['input_ids'])
        full_lengths.append(full_len)

        if full_len > max_length:
            truncated_full += 1

    # Statistics
    output_lengths = np.array(output_lengths)
    full_lengths = np.array(full_lengths)

    print("=" * 60)
    print("ASSISTANT OUTPUT LENGTHS (responses only):")
    print("=" * 60)
    print(f"  Max output length:     {output_lengths.max():,} tokens")
    print(f"  Mean output length:    {output_lengths.mean():.1f} tokens")
    print(f"  Median output length:  {np.median(output_lengths):.0f} tokens")
    print(f"  95th percentile:       {np.percentile(output_lengths, 95):.0f} tokens")
    print(f"  99th percentile:       {np.percentile(output_lengths, 99):.0f} tokens")
    print()
    print(f"  Outputs > max_length:  {truncated_outputs}/{len(output_lengths)} ({truncated_outputs/len(output_lengths)*100:.1f}%)")
    print()

    print("=" * 60)
    print("FULL CONVERSATION LENGTHS (for context):")
    print("=" * 60)
    print(f"  Max full length:       {full_lengths.max():,} tokens")
    print(f"  Mean full length:      {full_lengths.mean():.1f} tokens")
    print(f"  Median full length:    {np.median(full_lengths):.0f} tokens")
    print(f"  95th percentile:       {np.percentile(full_lengths, 95):.0f} tokens")
    print(f"  99th percentile:       {np.percentile(full_lengths, 99):.0f} tokens")
    print()
    print(f"  Full > max_length:     {truncated_full}/{len(full_lengths)} ({truncated_full/len(full_lengths)*100:.1f}%)")
    print()

    # Recommendations
    print("=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    if truncated_outputs > 0:
        recommended_output_len = int(np.percentile(output_lengths, 99))
        print(f"⚠️  WARNING: {truncated_outputs} outputs exceed max_length!")
        print(f"   Some assistant responses are getting truncated.")
        print(f"   Recommended max_length for outputs: {recommended_output_len}")
        print()
    else:
        print("✅ All assistant outputs fit within max_length")
        print()

    if truncated_full / len(full_lengths) > 0.05:
        recommended_full_len = int(np.percentile(full_lengths, 99))
        print(f"⚠️  WARNING: {truncated_full/len(full_lengths)*100:.1f}% of full conversations exceed max_length!")
        print(f"   Recommended max_length for full context: {recommended_full_len}")
        print()
    else:
        print("✅ Full conversations fit well within max_length")
        print()

    # Show longest outputs
    if len(output_lengths) > 0:
        print("=" * 60)
        print("LONGEST ASSISTANT OUTPUTS (top 5):")
        print("=" * 60)
        sorted_indices = np.argsort(output_lengths)[::-1][:5]
        for i, idx in enumerate(sorted_indices, 1):
            msg = examples[idx].get("messages", [])
            assistant_msg = None
            for m in reversed(msg):
                if m.get("role") == "assistant":
                    assistant_msg = m.get("content", "")
                    break

            preview = assistant_msg[:100] if assistant_msg else "N/A"
            print(f"{i}. {output_lengths[idx]:,} tokens: {preview}...")
        print()

if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "queue/processing/syllo_training_contract_20k.jsonl"
    check_output_lengths(data_file)
