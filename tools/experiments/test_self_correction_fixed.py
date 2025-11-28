#!/usr/bin/env python3
"""Test the fixed self-correction system."""

from self_correction_trainer import SelfCorrectionPipeline
import json

def test_correct_answer():
    """Test when model gets it right."""
    pipeline = SelfCorrectionPipeline()

    prompt = "What is 2+2?"
    golden = "4"
    initial = "4"

    examples = pipeline.generate_from_qa_pair(prompt, golden, initial)

    print("=== CORRECT ANSWER TEST ===")
    print(f"Examples generated: {len(examples)}")
    assert len(examples) == 1, "Should generate 1 example for correct answer"

    ex = examples[0]
    print(f"\nType: {ex['type']}")
    print(f"Correct: {ex['correct']}")
    print(f"\nPrompt:\n{ex['prompt']}")
    print(f"\nResponse:\n{ex['response']}")

    assert ex['type'] == 'self_evaluation'
    assert ex['correct'] == True
    assert ex['response'] == "This was correct."
    assert "Your previous answer:" in ex['prompt']
    assert "Was this correct?" in ex['prompt']

    print("\n✅ Correct answer test PASSED")


def test_wrong_answer():
    """Test when model gets it wrong."""
    pipeline = SelfCorrectionPipeline()

    prompt = "What is the capital of France?"
    golden = "Paris"
    initial = "London"

    examples = pipeline.generate_from_qa_pair(prompt, golden, initial)

    print("\n=== WRONG ANSWER TEST ===")
    print(f"Examples generated: {len(examples)}")
    assert len(examples) == 2, "Should generate 2 examples for wrong answer"

    # Example 1: Self-evaluation
    ex1 = examples[0]
    print(f"\nExample 1 - Type: {ex1['type']}")
    print(f"Prompt:\n{ex1['prompt']}")
    print(f"\nResponse:\n{ex1['response']}")

    assert ex1['type'] == 'self_evaluation'
    assert ex1['response'] == "This was not correct."
    assert "Your previous answer:" in ex1['prompt']
    assert "Was this correct?" in ex1['prompt']
    assert initial in ex1['prompt']  # Should show the wrong answer

    # Example 2: Correction
    ex2 = examples[1]
    print(f"\nExample 2 - Type: {ex2['type']}")
    print(f"Prompt:\n{ex2['prompt']}")
    print(f"\nResponse:\n{ex2['response']}")

    assert ex2['type'] == 'correction'
    assert ex2['response'] == golden  # Should be the correct answer
    assert "This was not correct." in ex2['prompt']
    assert "Error indicators:" in ex2['prompt']
    assert initial in ex2['prompt']  # Should show the wrong answer

    print("\n✅ Wrong answer test PASSED")


def test_training_format():
    """Test that output is in correct training format."""
    pipeline = SelfCorrectionPipeline()

    prompt = "What is 5+3?"
    golden = "8"
    initial = "7"

    examples = pipeline.generate_from_qa_pair(prompt, golden, initial)

    print("\n=== TRAINING FORMAT TEST ===")
    for i, ex in enumerate(examples):
        training_ex = {
            "prompt": ex['prompt'],
            "response": ex['response']
        }
        print(f"\nTraining Example {i+1}:")
        print(json.dumps(training_ex, indent=2))

        # Verify no wrong answers are being taught
        if ex['type'] == 'self_evaluation':
            # Response should only be "This was correct." or "This was not correct."
            assert ex['response'] in ["This was correct.", "This was not correct."]
            print("✅ Self-evaluation response is binary")
        elif ex['type'] == 'correction':
            # Response should be the golden answer
            assert ex['response'] == golden
            print("✅ Correction response is golden answer")

    print("\n✅ Training format test PASSED")


if __name__ == '__main__':
    test_correct_answer()
    test_wrong_answer()
    test_training_format()

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
    print("\nSummary:")
    print("- Correct answers → 1 self-evaluation example")
    print("- Wrong answers → 2 examples (self-eval + correction)")
    print("- NO wrong answers taught as target responses")
    print("- Old answers shown as CONTEXT only")
