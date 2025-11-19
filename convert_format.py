#!/usr/bin/env python3
"""
Convert LEO compositional format to training system format
"""
import json
import sys

def convert_example(example):
    """Convert from {prompt, response} or {user_prompt, assistant_response} to {messages} format"""
    # Extract the prompt and response - try multiple field names
    prompt = example.get('prompt') or example.get('user_prompt', '')
    response = example.get('response') or example.get('assistant_response', '')

    # Create messages format
    converted = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
    }

    # Preserve metadata if it exists
    if 'metadata' in example:
        converted['metadata'] = example['metadata']

    return converted

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Converting {input_file} → {output_file}")

    converted_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            if line.strip():
                try:
                    example = json.loads(line)
                    converted = convert_example(example)
                    outfile.write(json.dumps(converted) + '\n')
                    converted_count += 1

                    if converted_count % 10000 == 0:
                        print(f"  Converted {converted_count:,} examples...")
                except Exception as e:
                    print(f"  Error on line {line_num}: {e}")
                    continue

    print(f"✅ Converted {converted_count:,} examples successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 convert_format.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    main()
