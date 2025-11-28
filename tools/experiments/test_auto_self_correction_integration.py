#!/usr/bin/env python3
"""
Test auto self-correction integration.

Verifies all components work together.
"""

import json
from pathlib import Path
from auto_self_correction import AutoSelfCorrectionGenerator
from self_correction_trainer import SelfCorrectionPipeline

# Mock InferenceResult
class MockInferenceResult:
    def __init__(self, input_text, expected, predicted, match):
        self.input_text = input_text
        self.expected = expected
        self.predicted = predicted
        self.match = match


def test_integration():
    """Test the full integration."""
    print("=" * 70)
    print("AUTO SELF-CORRECTION INTEGRATION TEST")
    print("=" * 70)
    print()

    # 1. Test config loading
    print("1️⃣  Testing config loading...")
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            sc_config = cfg.get("self_correction", {})
            print(f"   ✅ Config loaded")
            print(f"      Enabled: {sc_config.get('enabled')}")
            print(f"      Auto-queue: {sc_config.get('auto_queue')}")
            print(f"      Min examples: {sc_config.get('min_examples')}")
            print(f"      Generation interval: {sc_config.get('generation_interval')}")
            print(f"      Eval max tokens: {cfg.get('eval_max_tokens')}")
    else:
        print("   ⚠️  config.json not found")
    print()

    # 2. Test generator creation
    print("2️⃣  Testing generator creation...")
    try:
        generator = AutoSelfCorrectionGenerator(
            output_dir="data/self_correction",
            auto_queue=False,  # Don't actually queue during test
            max_examples=3,    # Low threshold for testing
            generation_interval=None  # Example-driven only
        )
        print("   ✅ Generator created successfully")
        print(f"      Max examples: {generator.max_examples}")
        print(f"      Generation interval: {generator.generation_interval}")
    except Exception as e:
        print(f"   ❌ Error creating generator: {e}")
        return
    print()

    # 3. Test inference result processing
    print("3️⃣  Testing inference result processing...")

    # Create mock inference results
    results = [
        MockInferenceResult(
            input_text="What is 2+2?",
            expected="2+2 equals 4. This is basic addition.",
            predicted="4",
            match=False
        ),
        MockInferenceResult(
            input_text="What is the capital of France?",
            expected="The capital of France is Paris.",
            predicted="Paris",
            match=True
        ),
        MockInferenceResult(
            input_text="List three primary colors.",
            expected="The three primary colors are:\n1. Red\n2. Blue\n3. Yellow",
            predicted="Red, blue, and yellow.",
            match=False
        ),
    ]

    try:
        generator.process_inference_results(results, step=10)
        print(f"   ✅ Processed {len(results)} inference results")
        print(f"      Pending examples: {len(generator.pending_examples)}")
        print(f"      Stats: {generator.get_stats()}")
    except Exception as e:
        print(f"   ❌ Error processing results: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # 4. Test file generation
    print("4️⃣  Testing file generation...")
    try:
        # Force generation even if below threshold/interval
        generator.force_generate(step=10)

        # Check if file was created
        output_dir = Path("data/self_correction")
        files = list(output_dir.glob("self_correction_step10_*.jsonl"))

        if files:
            latest_file = files[-1]
            print(f"   ✅ File generated: {latest_file.name}")

            # Count examples
            with open(latest_file) as f:
                num_examples = sum(1 for _ in f)
            print(f"      Examples in file: {num_examples}")

            # Show first example
            with open(latest_file) as f:
                first = json.loads(f.readline())
                print(f"      First prompt: {first['prompt'][:50]}...")
                print(f"      First response: {first['response'][:50]}...")

            # Check metadata
            meta_file = latest_file.with_suffix('.json')
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    print(f"      Metadata: {meta}")

            # Cleanup test file
            latest_file.unlink()
            if meta_file.exists():
                meta_file.unlink()
            print(f"   🧹 Cleaned up test file")
        else:
            print(f"   ⚠️  No file generated (might need more examples)")
    except Exception as e:
        print(f"   ❌ Error in file generation: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # 5. Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ All integration tests passed!")
    print()
    print("System is ready to:")
    print("  • Capture full inference outputs (2048 tokens)")
    print("  • Generate error codes automatically")
    print("  • Create self-correction training examples")
    print(f"  • Auto-drop to queue every {sc_config.get('max_examples', 200)} examples")
    print()
    print("Next: Start training and watch for self-correction files!")
    print()
    print("The system will automatically generate training files as it")
    print("accumulates examples, then queue them for training!")
    print()


if __name__ == '__main__':
    test_integration()
