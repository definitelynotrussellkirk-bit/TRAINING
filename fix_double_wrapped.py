#!/usr/bin/env python3
"""
Fix double-wrapped messages in lattice training data.

The problem: assistant_response was a stringified messages array,
and the converter wrapped it again, creating double nesting:

BROKEN:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"messages\": [...]}"}  ← Nested!
  ]
}

FIXED:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "actual JSON response"}  ← Just the answer
  ]
}
"""

import json
import sys
from pathlib import Path


def unwrap_line(line: str) -> dict:
    """Unwrap a double-wrapped message line."""
    # Parse outer structure
    outer = json.loads(line.strip())

    # Get the messages array
    messages = outer.get('messages', [])
    if len(messages) != 2:
        raise ValueError(f"Expected 2 messages, got {len(messages)}")

    user_msg = messages[0]
    asst_msg = messages[1]

    # Check if assistant content is a stringified messages array
    asst_content = asst_msg['content']

    try:
        # Try to parse as JSON
        inner = json.loads(asst_content)

        # If it's a messages array, unwrap it
        if isinstance(inner, dict) and 'messages' in inner:
            inner_messages = inner['messages']
            if len(inner_messages) == 2:
                # Extract the actual assistant response from the inner structure
                actual_response = inner_messages[1]['content']

                # Return fixed structure
                return {
                    "messages": [
                        user_msg,
                        {"role": "assistant", "content": actual_response}
                    ]
                }
    except (json.JSONDecodeError, KeyError, IndexError):
        # If parsing fails, content is not nested - return as is
        pass

    # No unwrapping needed
    return outer


def fix_file(input_path: Path, output_path: Path):
    """Fix double-wrapping in a JSONL file."""
    fixed_count = 0
    unchanged_count = 0
    error_count = 0

    print(f"🔧 Fixing: {input_path.name}")
    print(f"   Output: {output_path.name}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                original = json.loads(line.strip())
                fixed = unwrap_line(line)

                # Check if anything changed
                if json.dumps(original, sort_keys=True) != json.dumps(fixed, sort_keys=True):
                    fixed_count += 1
                else:
                    unchanged_count += 1

                # Write fixed line
                outfile.write(json.dumps(fixed, ensure_ascii=False) + '\n')

                # Show progress
                if line_num % 1000 == 0:
                    print(f"   Processed: {line_num:,} lines ({fixed_count:,} fixed)", end='\r')

            except Exception as e:
                print(f"\n   ❌ Error on line {line_num}: {e}")
                error_count += 1

    print(f"\n   ✅ Complete!")
    print(f"      Fixed: {fixed_count:,} lines")
    print(f"      Unchanged: {unchanged_count:,} lines")
    if error_count > 0:
        print(f"      Errors: {error_count} lines")
    print()

    return fixed_count, unchanged_count, error_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fix_double_wrapped.py <input_file> [output_file]")
        print("  If output_file not specified, will use <input_file>.fixed.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.parent / f"{input_path.stem}_fixed.jsonl"

    # Fix the file
    fixed, unchanged, errors = fix_file(input_path, output_path)

    print("=" * 60)
    print(f"TOTAL: {fixed:,} lines fixed, {unchanged:,} unchanged")
    if errors > 0:
        print(f"       {errors} errors")
    print("=" * 60)
    print(f"\n📁 Fixed file: {output_path}")

    # Show first fixed example
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                print(f"\n📝 First fixed example (truncated):")
                example = json.loads(first_line)
                user_content = example['messages'][0]['content'][:200]
                asst_content = example['messages'][1]['content'][:300]
                print(f"   User: {user_content}...")
                print(f"   Assistant: {asst_content}...")


if __name__ == "__main__":
    main()
