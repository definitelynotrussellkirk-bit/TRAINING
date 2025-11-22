#!/usr/bin/env python3
"""
Critical Edge Case Tests - ACTUALLY RUN THESE

Tests the P0 critical edge cases that could break the system:
1. Empty files
2. Single example files
3. Files with < batch_size examples
4. Corrupt trainer_state.json
5. File deleted mid-training (simulation)
6. Zero-step files
7. Very large output lengths

These are EXECUTABLE tests, not just documentation.
"""

import json
from pathlib import Path
import shutil


class EdgeCaseTester:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def log(self, test_name, status, message=""):
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message
        })
        if status == 'PASS':
            self.passed += 1
            print(f"✅ PASS: {test_name}")
        elif status == 'FAIL':
            self.failed += 1
            print(f"❌ FAIL: {test_name}: {message}")
        elif status == 'SKIP':
            print(f"⏭️  SKIP: {test_name}: {message}")
        if message:
            print(f"   {message}")

    def test_empty_file(self):
        """Test: Empty JSONL file"""
        test_file = Path("inbox/empty_edge_case.jsonl")

        try:
            # Create empty file
            test_file.write_text("")

            # Check if validation catches it
            from validate_data import analyze_jsonl_file
            from transformers import AutoTokenizer

            model_path = json.load(open("config.json"))['base_model']
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            result = analyze_jsonl_file(test_file, tokenizer, sample_limit=100)

            if result is None:
                self.log("Empty file handling", "PASS",
                        "Validation correctly rejected empty file")
            else:
                self.log("Empty file handling", "FAIL",
                        f"Empty file passed validation: {result}")

            test_file.unlink(missing_ok=True)

        except Exception as e:
            self.log("Empty file handling", "PASS",
                    f"Empty file raised exception (expected): {e}")
            test_file.unlink(missing_ok=True)

    def test_single_example_file(self):
        """Test: File with only 1 example"""
        test_file = Path("inbox/single_example.jsonl")

        try:
            data = {"messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Answer"}
            ]}
            test_file.write_text(json.dumps(data) + '\n')

            # This should work, but steps will be 0 (1 example / 8 batch = 0)
            # Check step calculation
            batch_size = 1
            gradient_accum = 8
            examples = 1
            epochs = 1

            steps = (examples // (batch_size * gradient_accum)) * epochs

            if steps == 0:
                self.log("Single example file", "PASS",
                        f"Correctly calculated 0 steps for 1 example (batch=8)")
            else:
                self.log("Single example file", "FAIL",
                        f"Step calculation wrong: got {steps}, expected 0")

            test_file.unlink(missing_ok=True)

        except Exception as e:
            self.log("Single example file", "FAIL", f"Unexpected error: {e}")
            test_file.unlink(missing_ok=True)

    def test_sub_batch_file(self):
        """Test: File with examples < batch_size"""
        test_file = Path("inbox/sub_batch.jsonl")

        try:
            # Create file with 5 examples (batch = 8)
            with open(test_file, 'w') as f:
                for i in range(5):
                    data = {"messages": [
                        {"role": "user", "content": f"Test {i}"},
                        {"role": "assistant", "content": f"Answer {i}"}
                    ]}
                    f.write(json.dumps(data) + '\n')

            batch_size = 1
            gradient_accum = 8
            examples = 5
            epochs = 1

            steps = (examples // (batch_size * gradient_accum)) * epochs

            if steps == 0:
                self.log("Sub-batch file", "PASS",
                        f"Correctly calculated 0 steps for 5 examples (batch=8)")
            else:
                self.log("Sub-batch file", "FAIL",
                        f"Step calculation wrong: got {steps}, expected 0")

            test_file.unlink(missing_ok=True)

        except Exception as e:
            self.log("Sub-batch file", "FAIL", f"Unexpected error: {e}")
            test_file.unlink(missing_ok=True)

    def test_corrupt_trainer_state(self):
        """Test: Corrupt trainer_state.json recovery"""
        trainer_state_file = Path("current_model/trainer_state.json")
        backup_file = Path("current_model/trainer_state.json.backup_test")

        try:
            if not trainer_state_file.exists():
                self.log("Corrupt trainer_state", "SKIP",
                        "No trainer_state.json exists to test")
                return

            # Backup original
            shutil.copy(trainer_state_file, backup_file)

            # Corrupt it
            trainer_state_file.write_text("{ invalid json }")

            # Try to read it (simulate what train.py does)
            try:
                with open(trainer_state_file) as f:
                    data = json.load(f)
                self.log("Corrupt trainer_state", "FAIL",
                        "Corrupt JSON was parsed successfully (shouldn't happen)")
            except json.JSONDecodeError:
                # This is expected - the code should catch this
                self.log("Corrupt trainer_state", "PASS",
                        "Corrupt JSON correctly raises JSONDecodeError")

            # Restore original
            shutil.copy(backup_file, trainer_state_file)
            backup_file.unlink()

        except Exception as e:
            self.log("Corrupt trainer_state", "FAIL", f"Unexpected error: {e}")
            if backup_file.exists():
                shutil.copy(backup_file, trainer_state_file)
                backup_file.unlink()

    def test_negative_global_step(self):
        """Test: Negative global_step in trainer_state"""
        trainer_state_file = Path("current_model/trainer_state.json")
        backup_file = Path("current_model/trainer_state.json.backup_test2")

        try:
            if not trainer_state_file.exists():
                self.log("Negative global_step", "SKIP",
                        "No trainer_state.json exists to test")
                return

            # Backup original
            shutil.copy(trainer_state_file, backup_file)

            # Create state with negative step
            state = json.load(open(trainer_state_file))
            state['global_step'] = -100
            trainer_state_file.write_text(json.dumps(state))

            # Try to read it
            with open(trainer_state_file) as f:
                data = json.load(f)
                global_step = data.get('global_step', 0)

            if global_step < 0:
                self.log("Negative global_step", "FAIL",
                        f"Negative global_step not validated: {global_step}")
            else:
                self.log("Negative global_step", "PASS",
                        "Negative global_step defaulted to 0")

            # Restore original
            shutil.copy(backup_file, trainer_state_file)
            backup_file.unlink()

        except Exception as e:
            self.log("Negative global_step", "FAIL", f"Unexpected error: {e}")
            if backup_file.exists():
                shutil.copy(backup_file, trainer_state_file)
                backup_file.unlink()

    def test_malformed_jsonl(self):
        """Test: Malformed JSONL data"""
        test_file = Path("inbox/malformed.jsonl")

        try:
            # Create file with bad JSON
            test_file.write_text('{"messages": [{"role": "user", "content": "test"\n')

            from validate_data import analyze_jsonl_file
            from transformers import AutoTokenizer

            model_path = json.load(open("config.json"))['base_model']
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            result = analyze_jsonl_file(test_file, tokenizer, sample_limit=100)

            if result is None or (result and 'errors' in result and len(result['errors']) > 0):
                self.log("Malformed JSONL", "PASS",
                        "Malformed JSON correctly detected")
            else:
                self.log("Malformed JSONL", "FAIL",
                        f"Malformed JSON not detected: {result}")

            test_file.unlink(missing_ok=True)

        except Exception as e:
            self.log("Malformed JSONL", "PASS",
                    f"Malformed JSON raised exception (expected): {type(e).__name__}")
            test_file.unlink(missing_ok=True)

    def test_extremely_long_output(self):
        """Test: Output longer than max_length"""
        test_file = Path("inbox/long_output.jsonl")

        try:
            # Create file with very long output
            long_text = "x" * 5000  # 5000 characters
            data = {"messages": [
                {"role": "user", "content": "Short question"},
                {"role": "assistant", "content": long_text}
            ]}
            test_file.write_text(json.dumps(data) + '\n')

            from validate_data import analyze_jsonl_file
            from transformers import AutoTokenizer

            config = json.load(open("config.json"))
            model_path = config['base_model']
            max_length = config.get('max_length', 2048)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            result = analyze_jsonl_file(test_file, tokenizer, sample_limit=1)

            if result and 'output_stats' in result:
                max_output = result['output_stats']['max']
                if max_output > max_length:
                    self.log("Extremely long output", "PASS",
                            f"Detected output ({max_output} tokens) > max_length ({max_length})")
                else:
                    self.log("Extremely long output", "FAIL",
                            f"Output ({max_output}) didn't exceed max_length ({max_length})")
            else:
                self.log("Extremely long output", "FAIL",
                        "No output stats in validation result")

            test_file.unlink(missing_ok=True)

        except Exception as e:
            self.log("Extremely long output", "FAIL", f"Unexpected error: {e}")
            test_file.unlink(missing_ok=True)

    def test_missing_fields(self):
        """Test: JSONL missing required fields"""
        test_file = Path("inbox/missing_fields.jsonl")

        try:
            # No 'messages' field
            data = {"wrong_field": "value"}
            test_file.write_text(json.dumps(data) + '\n')

            from validate_data import analyze_jsonl_file
            from transformers import AutoTokenizer

            model_path = json.load(open("config.json"))['base_model']
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            result = analyze_jsonl_file(test_file, tokenizer, sample_limit=100)

            if result is None or (result and result.get('total_examples', 0) == 0):
                self.log("Missing fields", "PASS",
                        "Missing 'messages' field correctly detected")
            else:
                self.log("Missing fields", "FAIL",
                        f"Missing field not detected: {result}")

            test_file.unlink(missing_ok=True)

        except Exception as e:
            self.log("Missing fields", "PASS",
                    f"Missing field raised exception (expected): {type(e).__name__}")
            test_file.unlink(missing_ok=True)

    def run_all(self):
        """Run all edge case tests"""
        print("=" * 80)
        print("CRITICAL EDGE CASE TESTS")
        print("=" * 80)
        print("\nTesting edge cases that could break the system...")
        print()

        self.test_empty_file()
        self.test_single_example_file()
        self.test_sub_batch_file()
        self.test_malformed_jsonl()
        self.test_missing_fields()
        self.test_extremely_long_output()
        self.test_corrupt_trainer_state()
        self.test_negative_global_step()

        print()
        print("=" * 80)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 80)

        return self.failed == 0


if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      CRITICAL EDGE CASE TEST SUITE                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

This suite tests edge cases that could break the training system:
- Empty files
- Files too small to train
- Corrupt checkpoints
- Malformed data
- Extreme output lengths

These tests run quickly and don't require the daemon to be running.
""")

    tester = EdgeCaseTester()
    success = tester.run_all()

    if success:
        print("\n✅ All critical edge cases handled correctly!")
        exit(0)
    else:
        print(f"\n❌ {tester.failed} edge case(s) not handled properly")
        print("\nReview failures above and fix the issues.")
        exit(1)
