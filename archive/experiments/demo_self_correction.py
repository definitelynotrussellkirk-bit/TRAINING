#!/usr/bin/env python3
"""
Demo of self-correction training system.

Shows the complete flow with example data.
"""

import json
from self_correction_trainer import SelfCorrectionPipeline


def demo():
    """Run a demo of the self-correction system."""

    print("=" * 70)
    print("SELF-CORRECTION TRAINING DEMO")
    print("=" * 70)

    # Sample Q&A pairs
    test_cases = [
        {
            "name": "Math problem (too brief)",
            "prompt": "What is 15 × 12?",
            "golden": "15 × 12 = 180. This is calculated by multiplying 15 by 12, which gives us 180.",
            "initial": "180"
        },
        {
            "name": "Explanation (wrong structure)",
            "prompt": "List three primary colors.",
            "golden": "The three primary colors are:\n1. Red\n2. Blue\n3. Yellow",
            "initial": "The three primary colors are red, blue, and yellow."
        },
        {
            "name": "Wrong answer entirely",
            "prompt": "What is the capital of France?",
            "golden": "The capital of France is Paris, located in the north-central part of the country.",
            "initial": "The capital of France is Lyon, which is a major city in the country."
        },
        {
            "name": "Correct answer (no correction needed)",
            "prompt": "What is 2+2?",
            "golden": "2+2 equals 4.",
            "initial": "2+2 equals 4."
        }
    ]

    pipeline = SelfCorrectionPipeline()

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {i}: {case['name']}")
        print(f"{'=' * 70}\n")

        print(f"📝 PROMPT:\n{case['prompt']}\n")
        print(f"✅ GOLDEN ANSWER:\n{case['golden']}\n")
        print(f"🤖 MODEL'S INITIAL ANSWER:\n{case['initial']}\n")

        # Generate training examples
        examples = pipeline.generate_from_qa_pair(
            prompt=case['prompt'],
            golden_answer=case['golden'],
            initial_answer=case['initial']
        )

        print(f"📊 GENERATED {len(examples)} TRAINING EXAMPLE(S)\n")

        for j, ex in enumerate(examples, 1):
            print(f"\n--- Example {j}: {ex.get('type', 'standard')} ---\n")

            if 'error_codes' in ex and ex['error_codes']:
                print(f"🔍 ERROR CODES DETECTED:")
                for code in ex['error_codes']:
                    print(f"   • {code}")
                print()

            print(f"PROMPT:\n{ex['prompt']}\n")
            print(f"RESPONSE:\n{ex['response']}\n")

        print(f"\n{'=' * 70}\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThis demo showed:")
    print("  • Error code generation (programmatic feedback)")
    print("  • Self-evaluation training examples")
    print("  • Correction training examples")
    print("  • Handling of already-correct answers")
    print("\nTo use on real data:")
    print("  1. python3 generate_initial_answers.py --input qa.jsonl --output answers.jsonl")
    print("  2. python3 self_correction_trainer.py --input qa.jsonl --output train.jsonl --initial-answers answers.jsonl")
    print("  3. cp train.jsonl inbox/")
    print("\nSee SELF_CORRECTION_GUIDE.md for full documentation.")
    print()


if __name__ == '__main__':
    demo()
