#!/usr/bin/env python3
"""
Edge case testing for output cleaner.

IMPORTANT: This is EVALUATION DISPLAY ONLY!
- Training uses LOSS (already working correctly)
- This is just for human-readable "accuracy" metrics
- Does NOT affect training in any way
"""

from output_cleaner import clean_model_output, smart_compare
import json


def test_edge_cases():
    """Test edge cases that might break the cleaner."""

    edge_cases = [
        {
            'name': 'Empty output',
            'golden': 'ANSWER',
            'model': '',
            'should_match': False
        },
        {
            'name': 'Only whitespace',
            'golden': 'ANSWER',
            'model': '   \n\n\t   ',
            'should_match': False
        },
        {
            'name': 'Checkmark IS the answer',
            'golden': '✅',
            'model': '✅',
            'should_match': True
        },
        {
            'name': 'Answer contains "Answer:"',
            'golden': 'The Answer: 42',
            'model': 'The Answer: 42',
            'should_match': True
        },
        {
            'name': 'Multiple JSON objects',
            'golden': '{"a": 1}',
            'model': 'Here is one: {"x": 0}\n\nActual answer: {"a": 1}',
            'should_match': False  # Extracts first JSON
        },
        {
            'name': 'Nested JSON with newlines',
            'golden': '{"solutions": [{"nested": {"deep": "value"}}]}',
            'model': '✅\n\n```json\n{"solutions": [{"nested": {"deep": "value"}}]}\n```',
            'should_match': True
        },
        {
            'name': 'Emoticon tree with checkmarks',
            'golden': '😀\n├─ 😊\n│  └─ 😁\n└─ 😎',
            'model': '✅\n\n😀\n├─ 😊\n│  └─ 😁\n└─ 😎',
            'should_match': True
        },
        {
            'name': 'List with bullet points',
            'golden': '• Item 1\n• Item 2\n• Item 3',
            'model': 'Answer:\n• Item 1\n• Item 2\n• Item 3',
            'should_match': True
        },
        {
            'name': 'Code block without language',
            'golden': 'code here',
            'model': '```\ncode here\n```',
            'should_match': True
        },
        {
            'name': 'Answer with intentional prefix',
            'golden': 'Answer: This is the actual answer',
            'model': 'Answer: This is the actual answer',
            'should_match': True
        },
        {
            'name': 'Syllables only (SYLLO wrong output)',
            'golden': '{"solutions": [{"ans_num": 1, "syllables": ["a", "b"]}]}',
            'model': 'a\nb\nc\nd\ne',
            'should_match': False  # Correctly should not match
        },
        {
            'name': 'JSON with trailing garbage',
            'golden': '{"answer": "test"}',
            'model': '{"answer": "test"}\n\nSome extra text here.',
            'should_match': True  # Extract mode should get the JSON
        },
        {
            'name': 'Explanation only (no answer)',
            'golden': 'ANSWER',
            'model': 'To solve this, I need to analyze the problem carefully.',
            'should_match': False  # Correctly should not match
        },
        {
            'name': 'Unicode characters preserved',
            'golden': '→ → →\n↓ ↓ ↓',
            'model': '✅\n\n→ → →\n↓ ↓ ↓',
            'should_match': True
        },
        {
            'name': 'Very long JSON (truncation test)',
            'golden': '{"data": "' + 'x' * 10000 + '"}',
            'model': '```json\n{"data": "' + 'x' * 10000 + '"}\n```',
            'should_match': True
        },
        {
            'name': 'Actual real example from evolution',
            'golden': '{"solutions": [{"ans_num": 1, "syllables": ["op", "er", "at", "ing"], "answer": "OPERATING"}]}',
            'model': '✅\n\nAnswer: ✅\n\n{"solutions": [{"ans_num": 1, "syllables": ["op", "er", "at", "ing"], "answer": "OPERATING"}]}',
            'should_match': True
        },
        {
            'name': 'Multiple prefixes',
            'golden': 'RESULT',
            'model': '✅ Answer: ✅ Output: ✅\n\nRESULT',
            'should_match': True
        },
        {
            'name': 'Whitespace-sensitive format',
            'golden': 'A    B    C',
            'model': '✅\n\nA    B    C',
            'should_match': True
        }
    ]

    print("=" * 80)
    print("EDGE CASE TESTING - Output Cleaner")
    print("=" * 80)
    print("\n⚠️  NOTE: This is for EVALUATION DISPLAY ONLY!")
    print("   Training uses LOSS (already working correctly)")
    print("   This just makes accuracy metrics human-readable\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, test in enumerate(edge_cases, 1):
        golden = test['golden']
        model = test['model']
        expected = test['should_match']
        name = test['name']

        # Test both modes
        result_clean = smart_compare(golden, model, mode='clean')
        result_extract = smart_compare(golden, model, mode='extract')

        # Consider it a match if either mode matches
        got_match = result_clean or result_extract

        success = got_match == expected

        if success:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"

        print(f"\n{status} Test {i}: {name}")
        print(f"   Expected match: {expected}")
        print(f"   Got match: {got_match} (clean={result_clean}, extract={result_extract})")

        if not success:
            print(f"   Golden ({len(golden)} chars): {golden[:100]}")
            print(f"   Model ({len(model)} chars): {model[:100]}")
            cleaned = clean_model_output(model)
            print(f"   Cleaned ({len(cleaned)} chars): {cleaned[:100]}")

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(edge_cases)} tests")
    print("=" * 80)

    if failed == 0:
        print("✅ All edge cases handled correctly!")
    else:
        print(f"⚠️  {failed} edge case(s) need attention")

    return failed == 0


def test_with_real_evolution_data():
    """Test with actual examples from evolution snapshots."""

    print("\n" + "=" * 80)
    print("REAL DATA TESTING - Evolution Snapshot Examples")
    print("=" * 80)

    # Read actual evolution snapshot
    try:
        with open('data/evolution_snapshots/syllo_training_contract_20k/step_002000.json') as f:
            data = json.load(f)

        examples = data.get('examples', [])[:10]  # Test first 10

        matches_exact = 0
        matches_clean = 0
        matches_extract = 0

        for i, ex in enumerate(examples, 1):
            golden = ex.get('expected_output', '')
            model = ex.get('model_output', '')

            exact = golden.strip() == model.strip()
            clean = smart_compare(golden, model, mode='clean')
            extract = smart_compare(golden, model, mode='extract')

            if exact:
                matches_exact += 1
            if clean:
                matches_clean += 1
            if extract:
                matches_extract += 1

            print(f"\nExample {i}:")
            print(f"  Exact match: {exact}")
            print(f"  Clean match: {clean}")
            print(f"  Extract match: {extract}")

            if not exact and (clean or extract):
                print(f"  ✅ Cleaner would improve accuracy!")
                print(f"     Golden: {golden[:80]}...")
                print(f"     Model:  {model[:80]}...")

        total = len(examples)
        print("\n" + "=" * 80)
        print(f"Real Data Results (n={total}):")
        print(f"  Exact matches:   {matches_exact}/{total} ({matches_exact/total*100:.1f}%)")
        print(f"  Clean matches:   {matches_clean}/{total} ({matches_clean/total*100:.1f}%)")
        print(f"  Extract matches: {matches_extract}/{total} ({matches_extract/total*100:.1f}%)")
        print(f"  Improvement: +{max(matches_clean, matches_extract) - matches_exact} matches")
        print("=" * 80)

    except FileNotFoundError:
        print("⚠️  Evolution snapshot not found, skipping real data test")
    except Exception as e:
        print(f"⚠️  Error reading evolution data: {e}")


if __name__ == '__main__':
    # Run edge case tests
    edge_success = test_edge_cases()

    # Run real data tests
    test_with_real_evolution_data()

    # Summary
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\n📋 Remember:")
    print("   • This cleaner is for DISPLAY/EVALUATION only")
    print("   • Training still uses raw outputs + LOSS (correct)")
    print("   • Just makes accuracy metrics more readable")
    print("   • Does NOT affect training in any way")
    print("\n" + "=" * 80)
