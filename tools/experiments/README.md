# Experiments & Local Tests

These scripts require:
- GPU with CUDA
- Specific model paths on the dev machine (e.g., `/path/to/training/models/`)
- Manual execution

They are **NOT** part of the automated test suite (`pytest tests/`).

## Contents

| Script | Purpose |
|--------|---------|
| `test_model.py` | Interactive model testing with prompts |
| `test_specific.py` | Test specific model outputs |
| `test_callback_minimal.py` | Debug training callbacks |
| `test_continuous_training.py` | Test continuous training scenarios |
| `test_critical_edge_cases.py` | Edge case testing |
| `test_edge_cases.py` | More edge cases |
| `test_minimal_training.py` | Minimal training test |
| `test_random_thinking.py` | Random thinking tests |
| `test_wrong_version_scenarios.py` | Version mismatch testing |
| `test_custom_eot_sequence.py` | EOT sequence testing |
| `test_eos_conflict_fix.py` | EOS conflict testing |
| `test_eot_reward.py` | EOT reward testing |
| `test_stop_penalty_debug.py` | Stop penalty debugging |
| `test_variable_stop_emojis.py` | Variable stop emoji testing |
| `test_auto_self_correction_integration.py` | Self-correction integration |
| `test_dataset_hash.py` | Dataset hashing tests |
| `test_integration.py` | Integration tests |
| `test_output_cleaner.py` | Output cleaner tests |
| `test_self_correction_fixed.py` | Fixed self-correction tests |
| `test_formatting.py` | Formatting tests |

## Running Manually

```bash
# Activate environment
source ~/venv/bin/activate  # or your venv path

# Run interactive model tester
python tools/experiments/test_model.py --base-model /path/to/model

# Run specific test
python tools/experiments/test_specific.py

# Debug stop penalties
python tools/experiments/test_stop_penalty_debug.py
```

## Requirements

- PyTorch with CUDA
- Transformers
- PEFT
- bitsandbytes (for quantization)
- Local model files at expected paths

## Moving Back to tests/

If you want to make any of these CI-safe:
1. Remove hardcoded paths
2. Add `@pytest.mark.gpu` if GPU is required
3. Use fixtures from `tests/conftest.py`
4. Move back to `tests/` directory
