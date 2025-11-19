#!/usr/bin/env python3
"""
Add system prompt to training data.

This script prepends a system message to all conversations in a JSONL training file.
The system prompt helps establish the model's personality and behavior.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def get_system_prompt(timestamp=None):
    """
    Generate the system prompt with optional timestamp.

    Args:
        timestamp: Optional timestamp string. If None, uses current time.

    Returns:
        str: The system prompt
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    return (
        "You enjoy helping others. Your goal is produce what the user WANTS to the extent "
        "you are able. You should predict your next token based on total context and what "
        f"you've said so far. You are now. You are happy. ({timestamp})"
    )


def add_system_prompt_to_conversation(conversation, system_prompt):
    """
    Add system prompt as first message in conversation.

    Args:
        conversation: Dict with 'messages' key containing list of messages
        system_prompt: String to use as system prompt

    Returns:
        Dict: Modified conversation with system prompt prepended
    """
    # Create system message
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    # Check if first message is already system
    if conversation['messages'] and conversation['messages'][0].get('role') == 'system':
        # Replace existing system message
        conversation['messages'][0] = system_message
    else:
        # Prepend system message
        conversation['messages'].insert(0, system_message)

    return conversation


def process_file(input_path, output_path, use_timestamp=True):
    """
    Process a JSONL file and add system prompts to all conversations.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        use_timestamp: Whether to include timestamp in system prompt
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    # Generate system prompt once (same for all conversations)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") if use_timestamp else None
    system_prompt = get_system_prompt(timestamp)

    print(f"📝 System prompt:")
    print(f"   {system_prompt}")
    print()

    # Process file
    total_count = 0
    modified_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    # Parse conversation
                    conversation = json.loads(line.strip())
                    total_count += 1

                    # Add system prompt
                    modified_conversation = add_system_prompt_to_conversation(
                        conversation,
                        system_prompt
                    )
                    modified_count += 1

                    # Write to output
                    outfile.write(json.dumps(modified_conversation, ensure_ascii=False) + '\n')

                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipped invalid JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️  Warning: Error processing line {line_num}: {e}")
                    continue

    print(f"✅ Processed {total_count} conversations")
    print(f"✅ Added system prompts to {modified_count} conversations")
    print(f"✅ Output: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Add system prompt to training data JSONL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add system prompt with timestamp
  python3 add_system_prompt.py input.jsonl output.jsonl

  # Add system prompt without timestamp
  python3 add_system_prompt.py input.jsonl output.jsonl --no-timestamp

  # Process file from inbox to inbox
  python3 add_system_prompt.py inbox/leo_10k.jsonl inbox/leo_10k_with_system.jsonl
        """
    )

    parser.add_argument(
        'input',
        help='Input JSONL file'
    )
    parser.add_argument(
        'output',
        help='Output JSONL file'
    )
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Exclude timestamp from system prompt'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🔧 SYSTEM PROMPT INJECTOR")
    print("=" * 80)
    print()

    process_file(args.input, args.output, use_timestamp=not args.no_timestamp)

    print()
    print("=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
