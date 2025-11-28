#!/usr/bin/env python3
"""
Convert lattice training files from user_prompt/assistant_response format
to the messages array format expected by the training system.
"""

import json
import sys
from pathlib import Path

def convert_file(input_path: Path, output_path: Path):
    """Convert a single JSONL file to messages format"""
    converted_count = 0
    error_count = 0

    print(f"Converting: {input_path.name}")
    print(f"  → Output: {output_path.name}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse original format
                data = json.loads(line.strip())

                # Convert to messages format
                converted = {
                    "messages": [
                        {
                            "role": "user",
                            "content": data.get("user_prompt", "")
                        },
                        {
                            "role": "assistant",
                            "content": data.get("assistant_response", "")
                        }
                    ]
                }

                # Write converted line
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1

            except Exception as e:
                print(f"  ⚠️  Error on line {line_num}: {e}")
                error_count += 1

    print(f"  ✅ Converted: {converted_count:,} examples")
    if error_count > 0:
        print(f"  ❌ Errors: {error_count}")
    print()

    return converted_count, error_count

def main():
    # Process all lattice files in the failed queue
    try:
        from core.paths import get_base_dir
        base = get_base_dir()
    except ImportError:
        import os
        base = Path(os.environ.get("TRAINING_BASE_DIR", "."))

    failed_dir = base / "queue" / "failed"
    inbox_dir = base / "inbox"

    lattice_files = sorted(failed_dir.glob("lattice*.jsonl"))

    if not lattice_files:
        print("No lattice files found in queue/failed/")
        return

    print(f"Found {len(lattice_files)} lattice files to convert\n")

    total_converted = 0
    total_errors = 0

    for input_file in lattice_files:
        output_file = inbox_dir / input_file.name
        converted, errors = convert_file(input_file, output_file)
        total_converted += converted
        total_errors += errors

    print("=" * 60)
    print(f"TOTAL: {total_converted:,} examples converted")
    if total_errors > 0:
        print(f"       {total_errors} errors")
    print("=" * 60)
    print("\nConverted files saved to inbox/")
    print("Original files remain in queue/failed/")

if __name__ == "__main__":
    main()
