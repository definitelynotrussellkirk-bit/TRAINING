#!/usr/bin/env python3
"""
Data Validator - Checks training data against config settings

Analyzes .jsonl files in inbox and validates:
- Sequence lengths vs max_length setting
- Token distribution statistics
- Recommendations for optimal config

Usage:
    python3 validate_data.py                    # Check inbox data
    python3 validate_data.py --auto-adjust      # Auto-update config if needed
    python3 validate_data.py --file data.jsonl  # Check specific file
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import sys

def load_config(base_dir: Path) -> Dict:
    """Load training configuration"""
    config_path = base_dir / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(base_dir: Path, config: Dict):
    """Save updated configuration"""
    config_path = base_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Updated config: {config_path}")

def load_tokenizer(model_path: str):
    """Load tokenizer from model"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def analyze_jsonl_file(file_path: Path, tokenizer, sample_limit: int = None) -> Dict:
    """Analyze a single .jsonl file with detailed output analysis"""
    full_lengths = []
    prompt_lengths = []
    output_lengths = []
    total_examples = 0
    errors = []

    print(f"📊 Analyzing: {file_path.name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Limit sampling for large files
                if sample_limit and line_num > sample_limit:
                    print(f"   (Sampled {sample_limit} examples)")
                    break

                try:
                    data = json.loads(line.strip())

                    # Build conversation text
                    if 'messages' in data:
                        messages = data['messages']

                        # Separate prompt (user messages) from output (assistant response)
                        prompt_text = ""
                        output_text = ""

                        for msg in messages:
                            role = msg.get('role', '')
                            content = msg.get('content', '')

                            if role == 'assistant':
                                # Last assistant message is the output
                                output_text = content
                            else:
                                # User messages are part of the prompt
                                prompt_text += f"{role}: {content}\n"

                        # Tokenize full conversation
                        full_text = prompt_text + f"assistant: {output_text}\n"
                        full_tokens = tokenizer.encode(full_text)
                        full_lengths.append(len(full_tokens))

                        # Tokenize output separately
                        if output_text:
                            output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
                            output_lengths.append(len(output_tokens))

                        # Tokenize prompt separately
                        if prompt_text:
                            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                            prompt_lengths.append(len(prompt_tokens))

                    elif 'text' in data:
                        # Legacy format - just measure total
                        text = data['text']
                        tokens = tokenizer.encode(text)
                        full_lengths.append(len(tokens))
                    else:
                        errors.append(f"Line {line_num}: Unknown format (no 'messages' or 'text')")
                        continue

                    total_examples += 1

                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    errors.append(f"Line {line_num}: {e}")

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

    if not full_lengths:
        print(f"⚠️  No valid examples found in {file_path.name}")
        return None

    # Compute statistics for full conversations
    full_lengths.sort()
    stats = {
        'file': file_path.name,
        'total_examples': total_examples,
        'min_length': min(full_lengths),
        'max_length': max(full_lengths),
        'mean_length': sum(full_lengths) / len(full_lengths),
        'median_length': full_lengths[len(full_lengths) // 2],
        'p95_length': full_lengths[int(len(full_lengths) * 0.95)],
        'p99_length': full_lengths[int(len(full_lengths) * 0.99)],
        'errors': errors
    }

    # Add output-specific statistics if available
    if output_lengths:
        output_lengths.sort()
        stats['output_stats'] = {
            'min': min(output_lengths),
            'max': max(output_lengths),
            'mean': sum(output_lengths) / len(output_lengths),
            'median': output_lengths[len(output_lengths) // 2],
            'p95': output_lengths[int(len(output_lengths) * 0.95)],
            'p99': output_lengths[int(len(output_lengths) * 0.99)],
        }

    # Add prompt-specific statistics if available
    if prompt_lengths:
        prompt_lengths.sort()
        stats['prompt_stats'] = {
            'min': min(prompt_lengths),
            'max': max(prompt_lengths),
            'mean': sum(prompt_lengths) / len(prompt_lengths),
            'median': prompt_lengths[len(prompt_lengths) // 2],
            'p95': prompt_lengths[int(len(prompt_lengths) * 0.95)],
            'p99': prompt_lengths[int(len(prompt_lengths) * 0.99)],
        }

    return stats

def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print(f"\n📈 Statistics for {stats['file']}:")
    print(f"   Total examples: {stats['total_examples']}")

    # Full conversation stats
    print(f"\n   📋 FULL CONVERSATION (prompt + output):")
    print(f"      Min length:      {stats['min_length']} tokens")
    print(f"      Max length:      {stats['max_length']} tokens")
    print(f"      Mean length:     {stats['mean_length']:.1f} tokens")
    print(f"      Median length:   {stats['median_length']} tokens")
    print(f"      95th percentile: {stats['p95_length']} tokens")
    print(f"      99th percentile: {stats['p99_length']} tokens")

    # Output-specific stats
    if 'output_stats' in stats:
        o = stats['output_stats']
        print(f"\n   🤖 ASSISTANT OUTPUTS (responses only):")
        print(f"      Min length:      {o['min']} tokens")
        print(f"      Max length:      {o['max']} tokens")
        print(f"      Mean length:     {o['mean']:.1f} tokens")
        print(f"      Median length:   {int(o['median'])} tokens")
        print(f"      95th percentile: {int(o['p95'])} tokens")
        print(f"      99th percentile: {int(o['p99'])} tokens")

    # Prompt-specific stats
    if 'prompt_stats' in stats:
        p = stats['prompt_stats']
        print(f"\n   💬 PROMPTS (user inputs):")
        print(f"      Min length:      {p['min']} tokens")
        print(f"      Max length:      {p['max']} tokens")
        print(f"      Mean length:     {p['mean']:.1f} tokens")
        print(f"      Median length:   {int(p['median'])} tokens")
        print(f"      95th percentile: {int(p['p95'])} tokens")
        print(f"      99th percentile: {int(p['p99'])} tokens")

    if stats['errors']:
        print(f"\n⚠️  Errors found: {len(stats['errors'])}")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(stats['errors']) > 5:
            print(f"   ... and {len(stats['errors']) - 5} more errors")

def validate_against_config(stats: Dict, config: Dict) -> Tuple[bool, List[str]]:
    """Validate data stats against config settings"""
    issues = []
    warnings = []

    max_length = config.get('max_length', 2048)

    # Check FULL CONVERSATION lengths
    if stats['max_length'] > max_length:
        issues.append(
            f"⚠️  TRUNCATION: Full conversations will be truncated "
            f"(max: {stats['max_length']} > max_length: {max_length})"
        )

    if stats['p95_length'] > max_length:
        issues.append(
            f"⚠️  CRITICAL: 95% of full conversations exceed max_length "
            f"({stats['p95_length']} > {max_length})"
        )

    # Check OUTPUT lengths specifically (NEW!)
    if 'output_stats' in stats:
        out = stats['output_stats']

        if out['max'] > max_length:
            issues.append(
                f"🚨 CRITICAL: Assistant outputs exceed max_length! "
                f"(max output: {out['max']} > {max_length})"
            )
            issues.append(
                f"   This means responses are getting TRUNCATED during training!"
            )

        if out['p95'] > max_length * 0.8:
            warnings.append(
                f"⚠️  WARNING: Large assistant outputs detected "
                f"(95th percentile: {int(out['p95'])} tokens, {int(out['p95']/max_length*100)}% of max_length)"
            )

        if out['p99'] > max_length * 0.9:
            warnings.append(
                f"⚠️  Some outputs very close to limit "
                f"(99th percentile: {int(out['p99'])} tokens, {int(out['p99']/max_length*100)}% of max_length)"
            )

    # Check if max_length is way too high (wasting memory)
    if max_length > stats['p99_length'] * 1.5:
        warnings.append(
            f"💡 Consider reducing max_length to ~{int(stats['p99_length'] * 1.2)} "
            f"(currently {max_length}, but 99th percentile is only {stats['p99_length']})"
        )

    # Recommend optimal max_length
    if max_length < stats['p95_length']:
        recommended = int(stats['p99_length'] * 1.1)  # Add 10% buffer
        issues.append(
            f"🔧 RECOMMENDED: Set max_length to {recommended} "
            f"(covers 99% of examples with buffer)"
        )

    return len(issues) == 0, issues + warnings

def recommend_config(stats_list: List[Dict], current_config: Dict) -> Dict:
    """Recommend optimal config settings based on data"""
    recommendations = {}

    # Find the maximum p99 length across all files
    max_p99 = max(s['p99_length'] for s in stats_list)
    max_max = max(s['max_length'] for s in stats_list)

    # Recommend max_length with 10% buffer over p99
    recommended_max_length = int(max_p99 * 1.1)

    # Round up to nearest power of 2 or multiple of 256 for efficiency
    for candidate in [256, 512, 1024, 2048, 4096, 8192]:
        if candidate >= recommended_max_length:
            recommended_max_length = candidate
            break

    recommendations['max_length'] = recommended_max_length

    # Warn if absolute max is much higher
    if max_max > recommended_max_length:
        recommendations['warning'] = (
            f"⚠️  {max_max - recommended_max_length} tokens will be truncated from longest example. "
            f"Consider increasing max_length to {max_max} if this example is important."
        )

    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Validate training data against config")
    parser.add_argument('--base-dir', type=str, default='/path/to/training',
                       help='Base training directory')
    parser.add_argument('--file', type=str, help='Specific file to validate')
    parser.add_argument('--auto-adjust', action='store_true',
                       help='Automatically adjust config if needed')
    parser.add_argument('--quiet', action='store_true',
                       help='Only show warnings and errors')

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    # Load config
    config = load_config(base_dir)
    if not args.quiet:
        print(f"📋 Current config:")
        print(f"   Model: {config.get('model_name', 'unknown')}")
        print(f"   Max length: {config.get('max_length', 2048)} tokens")

    # Load tokenizer
    if not args.quiet:
        print(f"\n🔧 Loading tokenizer...")
    tokenizer = load_tokenizer(config['base_model'])

    # Find files to analyze
    if args.file:
        files = [Path(args.file)]
    else:
        inbox = base_dir / 'inbox'
        files = list(inbox.glob('*.jsonl'))

    if not files:
        print("⚠️  No .jsonl files found to validate")
        return 0

    # Analyze all files (sample 100 examples for speed)
    all_stats = []
    for file_path in files:
        stats = analyze_jsonl_file(file_path, tokenizer, sample_limit=100)
        if stats:
            all_stats.append(stats)
            if not args.quiet:
                print_statistics(stats)

    if not all_stats:
        print("❌ No valid data found")
        return 1

    # Validate against config
    print("\n" + "="*60)
    print("🔍 VALIDATION RESULTS")
    print("="*60)

    all_valid = True
    all_messages = []

    for stats in all_stats:
        valid, messages = validate_against_config(stats, config)
        all_valid = all_valid and valid
        all_messages.extend(messages)

    if all_messages:
        for msg in all_messages:
            print(msg)
    else:
        print("✅ All data fits within current config settings!")

    # Recommendations
    if len(all_stats) > 0:
        recommendations = recommend_config(all_stats, config)

        current_max = config.get('max_length', 2048)
        recommended_max = recommendations['max_length']

        if current_max != recommended_max:
            print(f"\n💡 RECOMMENDATION:")
            print(f"   Current max_length: {current_max}")
            print(f"   Recommended max_length: {recommended_max}")
            print(f"   Reason: Covers 99% of examples with 10% buffer")

            if 'warning' in recommendations:
                print(f"\n   {recommendations['warning']}")

            if args.auto_adjust:
                config['max_length'] = recommended_max
                save_config(base_dir, config)
                print(f"\n✅ Config updated automatically!")
            else:
                print(f"\n   Run with --auto-adjust to update config automatically")

    print()
    return 0 if all_valid else 1

if __name__ == '__main__':
    sys.exit(main())
