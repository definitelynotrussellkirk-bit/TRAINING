#!/usr/bin/env python3
"""
Quick test script to load and test the trained model locally
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys

# Test cases from the 3090 predictions
TEST_CASES = [
    {
        "name": "Test 1 - ACCEPTED puzzle",
        "system": "You are a helpful AI assistant.",
        "prompt": """SYLLO Puzzle syllo_train_00007
You must recover every hidden word by assigning syllable tiles to definitions.
Difficulty: Easy (no red herrings or duplicate syllables).
Rules:
- 5 target words (always between 4 and 8).
- Each word lists its syllable count via blank slots.
- Use every syllable tile exactly once.
- Return your answers as JSON with keys `solutions` and `inventory_check`.

Word slots:
1. ___ ___ ___ — Judged to be in conformity with approved usage.
2. ___ ___ — (semantics) the meaning of a word that is superordinate
3. ___ ___ ___ ___ — Commonness by virtue of not being unusual.
4. ___ ___ ___ — To place stress on, or to make more important.
5. ___ ___ ___ — To expect, to consider (to be) probable.

Syllable bank (shuffled):
per | em | nor | ha | pec | su | mal | ex | pha | ness | ted | size | si | ac | cep

Output contract:
- Return a single JSON object.
- Top-level keys: `solutions` (array) and `inventory_check` (object).
- Each `solutions` entry contains `ans_num` (1-indexed clue number), the ordered `syllables` you used, and the final UPPERCASE `answer`.
- `inventory_check` must include `total_tiles`, a `usage` map of tile→count, the `used` counts per tile, and a short `status` string.
Do not include literal JSON examples or commentary outside the payload.
- Format: respond with `solutions` array of `{ans_num, syllables, answer}` objects plus an `inventory_check` object.

For this task, think with 🧠 /once/.

When finished, emit ✋ /once/ to signal completion.""",
        "expected_answer": "ACCEPTED"
    },
    {
        "name": "Test 2 - Multiple words",
        "system": "You are a helpful AI assistant.",
        "prompt": """SYLLO Puzzle syllo_train_00123
You must recover every hidden word by assigning syllable tiles to definitions.
Difficulty: Easy (no red herrings or duplicate syllables).
Rules:
- 5 target words (always between 4 and 8).
- Each word lists its syllable count via blank slots.
- Use every syllable tile exactly once.
- Return your answers as JSON with keys `solutions` and `inventory_check`.

Word slots:
1. ___ ___ — a concept or idea not associated with any specific instance
2. ___ ___ — the position on a scale in relation to others in a sport
3. ___ ___ — Frequently; many times.
4. ___ ___ ___ ___ — the ninth month of the year
5. ___ ___ ___ — In its most common sense, refers to the present condition of things, or the course of affairs.

Syllable bank (shuffled):
ber | cur | of | ten | rent | lev | el | cept | sep | tem | con

Output contract:
- Return a single JSON object.
- Top-level keys: `solutions` (array) and `inventory_check` (object).
- Each `solutions` entry contains `ans_num` (1-indexed clue number), the ordered `syllables` you used, and the final UPPERCASE `answer`.
- `inventory_check` must include `total_tiles`, a `usage` map of tile→count, the `used` counts per tile, and a short `status` string.
Do not include literal JSON examples or commentary outside the payload.
- Format: respond with `solutions` array of `{ans_num, syllables, answer}` objects plus an `inventory_check` object.

For this task, think with 🧠 /once/.

When finished, emit ✋ /once/ to signal completion.""",
        "expected_answers": ["CONCEPT", "LEVEL", "OFTEN", "SEPTEMBER", "CURRENT"]
    }
]

def load_model(checkpoint_path):
    """Load model and tokenizer from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("✓ Model loaded")
    return model, tokenizer

def generate_response(model, tokenizer, system_prompt, user_prompt, max_tokens=2048):
    """Generate response from model"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "current_model/checkpoint-156000"

    print(f"\n{'='*80}")
    print(f"TESTING LOCAL MODEL: {checkpoint}")
    print(f"{'='*80}\n")

    model, tokenizer = load_model(checkpoint)

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'─'*80}")

        response = generate_response(model, tokenizer, test['system'], test['prompt'])

        print(f"\nGenerated response:")
        print(response[:1000])  # First 1000 chars
        if len(response) > 1000:
            print(f"\n... (truncated, total {len(response)} chars)")

        # Check if expected answer(s) appear in response
        if 'expected_answer' in test:
            found = test['expected_answer'] in response
            print(f"\n✓ Contains '{test['expected_answer']}': {found}")
        elif 'expected_answers' in test:
            found = sum(1 for ans in test['expected_answers'] if ans in response)
            print(f"\n✓ Contains {found}/{len(test['expected_answers'])} expected words")
            for ans in test['expected_answers']:
                status = "✓" if ans in response else "✗"
                print(f"  {status} {ans}")

        # Try to parse as JSON
        try:
            # Extract JSON from response (might have emoji prefix)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                print(f"\n✓ Valid JSON structure")
                if 'solutions' in data:
                    print(f"  Solutions: {len(data.get('solutions', []))} entries")
                if 'inventory_check' in data:
                    print(f"  Inventory check: Present")
        except json.JSONDecodeError as e:
            print(f"\n✗ JSON parsing failed: {e}")

        print()

if __name__ == "__main__":
    main()
