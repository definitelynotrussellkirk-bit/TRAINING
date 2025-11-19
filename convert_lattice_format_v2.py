#!/usr/bin/env python3
"""
Convert lattice training format to HuggingFace messages format.

This version handles the case where assistant_response contains a JSON string
with the full messages array.

Input format:
{
  "id": "...",
  "user_prompt": "...",
  "assistant_response": '{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}',
  ...
}

Output format:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

import json
import sys
from pathlib import Path


def convert_file(input_path: Path, output_path: Path):
    """Convert a single lattice format file to messages format."""
    converted_count = 0
    error_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse lattice format
                obj = json.loads(line)

                # Check if assistant_response contains a JSON string with messages
                assistant_response = obj.get('assistant_response', '')

                if not assistant_response:
                    print(f"  ⚠️  Line {line_num}: Missing assistant_response, skipping", file=sys.stderr)
                    error_count += 1
                    continue

                # Try to parse assistant_response as JSON
                try:
                    response_obj = json.loads(assistant_response)
                    if 'messages' in response_obj:
                        # Extract messages directly
                        converted = {"messages": response_obj['messages']}
                    else:
                        print(f"  ⚠️  Line {line_num}: assistant_response JSON has no 'messages' key, skipping", file=sys.stderr)
                        error_count += 1
                        continue
                except json.JSONDecodeError:
                    # assistant_response is not JSON, use old format
                    user_prompt = obj.get('user_prompt', '')
                    if not user_prompt:
                        print(f"  ⚠️  Line {line_num}: Missing user_prompt, skipping", file=sys.stderr)
                        error_count += 1
                        continue

                    converted = {
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_response}
                        ]
                    }

                # Write converted line
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"  ❌ Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                error_count += 1
            except Exception as e:
                print(f"  ❌ Line {line_num}: Unexpected error: {e}", file=sys.stderr)
                error_count += 1

    return converted_count, error_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_lattice_format_v2.py <input_file> [output_file]")
        print("  If output_file not specified, will use <input_file>_converted.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / f"{input_path.stem}_converted.jsonl"

    print(f"🔄 Converting: {input_path}")
    print(f"   Output to: {output_path}")

    converted_count, error_count = convert_file(input_path, output_path)

    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {converted_count} examples")
    if error_count > 0:
        print(f"   ⚠️  Errors: {error_count} examples")
    print(f"   Output: {output_path}")

    # Show first converted example
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                print(f"\n📝 First converted example:")
                example = json.loads(first_line)
                print(json.dumps(example, indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
