#!/usr/bin/env python3
"""
Output Cleaner - Format-Agnostic Response Cleaning

Strips common prefixes/wrappers that models add while being "helpful":
- Checkmarks (✅)
- "Answer:" labels
- Markdown code blocks (```json ... ```)
- Explanatory prefixes
- Extra newlines

Works with ANY output format: JSON, lists, trees, plain text, etc.
"""

import re


def clean_model_output(text: str, aggressive: bool = False) -> str:
    """
    Clean model output to match golden format.

    Args:
        text: Raw model output
        aggressive: If True, more aggressive cleaning (may remove wanted content)

    Returns:
        Cleaned text
    """
    if not text:
        return text

    original = text

    # Step 1: Remove leading checkmarks and common symbols
    text = re.sub(r'^[\s✅❌🔍📝⚠️🎯]+', '', text)

    # Step 2: Remove common prefix patterns (case insensitive)
    # "Answer:", "Output:", "Solution:", "Result:", etc.
    text = re.sub(r'^(?:Answer|Output|Solution|Result|Response|JSON output|Final answer)\s*:\s*', '', text, flags=re.IGNORECASE)

    # Step 3: Remove markdown code blocks
    # ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json|python|text)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)

    # Step 4: Remove explanatory prefixes
    # "To complete this puzzle..." -> skip to actual content
    # Look for first line that looks like actual content
    if aggressive:
        lines = text.split('\n')
        # Skip lines that look like explanations
        skip_patterns = [
            r'^To\s+',
            r'^In order to\s+',
            r'^First\s*,?\s+',
            r'^Let me\s+',
            r'^I will\s+',
            r'^Here\s+(?:is|are)\s+',
        ]

        content_start = 0
        for i, line in enumerate(lines):
            # Check if line matches any skip pattern
            is_explanation = any(re.match(pat, line.strip(), re.IGNORECASE) for pat in skip_patterns)
            if not is_explanation and line.strip():
                content_start = i
                break

        if content_start > 0:
            text = '\n'.join(lines[content_start:])

    # Step 5: Remove trailing checkmarks and symbols
    text = re.sub(r'[\s✅❌🔍📝⚠️🎯]+$', '', text)

    # Step 6: Clean up excessive whitespace
    # But preserve intentional structure (like emoticon trees)
    text = text.strip()

    # Step 7: Normalize line endings (but don't collapse all newlines)
    # Remove more than 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def extract_json_if_present(text: str) -> str:
    """
    If text contains JSON, extract it. Otherwise return cleaned text.

    Args:
        text: Text that might contain JSON

    Returns:
        Extracted JSON or cleaned text
    """
    # Try to find JSON object in the text
    # Look for {...} pattern
    json_match = re.search(r'\{[^}]*(?:\{[^}]*\}[^}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Try to find JSON array in the text
    # Look for [...] pattern
    array_match = re.search(r'\[[^\]]*(?:\[[^\]]*\][^\]]*)*\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)

    # No JSON found, return cleaned text
    return clean_model_output(text, aggressive=True)


def smart_compare(golden: str, model_output: str, mode: str = 'clean') -> bool:
    """
    Compare golden answer to model output with smart cleaning.

    Args:
        golden: Expected output
        model_output: Model's actual output
        mode: Comparison mode
            - 'exact': Exact string match (no cleaning)
            - 'clean': Clean both outputs before comparing
            - 'extract': Try to extract content (JSON, lists, etc.)

    Returns:
        True if outputs match
    """
    if mode == 'exact':
        return golden.strip() == model_output.strip()

    elif mode == 'clean':
        golden_clean = clean_model_output(golden)
        model_clean = clean_model_output(model_output)
        return golden_clean == model_clean

    elif mode == 'extract':
        # Try JSON extraction first
        golden_content = extract_json_if_present(golden)
        model_content = extract_json_if_present(model_output)

        # If both extracted, compare
        if '{' in golden_content or '[' in golden_content:
            return golden_content == model_content

        # Otherwise fall back to clean comparison
        return clean_model_output(golden) == clean_model_output(model_output)

    return False


if __name__ == '__main__':
    # Test cases
    test_cases = [
        # JSON with prefix
        {
            'golden': '{"solutions": [{"ans_num": 1, "answer": "TEST"}]}',
            'model': '✅\n\nAnswer: ✅\n\n{"solutions": [{"ans_num": 1, "answer": "TEST"}]}',
            'should_match': True
        },
        # Markdown wrapped
        {
            'golden': '{"solutions": [{"ans_num": 1}]}',
            'model': '✅\n\nJSON output:\n```json\n{"solutions": [{"ans_num": 1}]}\n```',
            'should_match': True
        },
        # Simple list
        {
            'golden': 'apple\nbanana\ncherry',
            'model': '✅\n\nAnswer:\napple\nbanana\ncherry',
            'should_match': True
        },
        # Emoticon tree (should preserve structure)
        {
            'golden': '😀\n├─ 😊\n└─ 😁',
            'model': '✅\n\n😀\n├─ 😊\n└─ 😁',
            'should_match': True
        },
        # Explanation before answer
        {
            'golden': 'ANSWER',
            'model': 'To solve this puzzle, I will analyze the clues.\n\nANSWER',
            'should_match': True  # With aggressive mode
        }
    ]

    print("Testing output cleaner...")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        golden = test['golden']
        model = test['model']
        expected = test['should_match']

        # Test with clean mode
        result_clean = smart_compare(golden, model, mode='clean')

        # Test with extract mode
        result_extract = smart_compare(golden, model, mode='extract')

        # Show cleaned versions
        model_clean = clean_model_output(model)

        print(f"\nTest {i}:")
        print(f"Golden: {golden[:80]}")
        print(f"Model:  {model[:80]}")
        print(f"Cleaned: {model_clean[:80]}")
        print(f"Match (clean): {result_clean} | Match (extract): {result_extract}")

        status = "✅" if (result_clean or result_extract) == expected else "❌"
        print(f"{status} Expected: {expected}, Got: clean={result_clean}, extract={result_extract}")

    print("\n" + "=" * 80)
    print("Testing complete!")
