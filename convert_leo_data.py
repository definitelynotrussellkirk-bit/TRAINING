#!/usr/bin/env python3
"""
Convert LEO prompt/response format to messages format for training.

Usage:
    python3 convert_leo_data.py input.jsonl output.jsonl
"""

import json
import sys
from pathlib import Path


def convert_to_messages_format(example):
    """Convert LEO format to messages format."""
    prompt = example.get("prompt", "")
    response = example.get("response", "")

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    # Preserve metadata if present
    result = {"messages": messages}
    if "metadata" in example:
        result["metadata"] = example["metadata"]

    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_leo_data.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    print(f"Converting {input_file} → {output_file}")

    converted = 0
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                converted_example = convert_to_messages_format(example)
                f_out.write(json.dumps(converted_example) + '\n')
                converted += 1

                if converted % 1000 == 0:
                    print(f"  Converted {converted} examples...")

            except Exception as e:
                print(f"Warning: Failed to convert line {line_num}: {e}")
                continue

    print(f"\n✓ Converted {converted} examples")
    print(f"✓ Output: {output_file}")


if __name__ == "__main__":
    main()
