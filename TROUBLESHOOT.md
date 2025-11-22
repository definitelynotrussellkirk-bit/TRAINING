# Troubleshooting Log

Log critical issues encountered during training plus their fixes so future operators can unblock quickly.

## 2025-11-20 – Penalty/Heatmap Regression
- **Symptom:** `TrainingStatus.__init__()` raised `TypeError: unexpected keyword argument 'penalty_heatmap'`, causing every dataset to fail immediately.
- **Cause:** Live monitor callback still forwarded `penalty_heatmap` data even though the dataclass did not define the field; extra analytics were also unnecessary for recovery.
- **Fix:** Added `penalty_heatmap` to `TrainingStatus` and gated all penalty/heatmap bookkeeping behind `ENABLE_PENALTY_MONITORING` (default off). Once patched, training resumes cleanly from checkpoint `116300`.

## 2025-11-20 – Corrupted Checkpoints (`116360_broken`)
- **Symptom:** Continuous crashes with `ValueError: invalid literal for int() with base 10: '116360_broken'` and missing `trainer_state.json`.
- **Cause:** A partial checkpoint directory without `trainer_state.json` was still being scanned.
- **Fix:** Renamed the bad checkpoint for inspection, restored `current_model_small` from `checkpoint-116300`, and taught both checkpoint scanners to skip non-numeric directories or ones missing state files. Training now resumes from the latest healthy checkpoint automatically.

## 2025-11-20 – EOS Penalty Crash
- **Symptom:** During model load `build_eos_penalty_processor` raised `TypeError: 'int' object is not iterable`.
- **Cause:** Tokenizer’s EOS fields sometimes returned scalars or nested tensors that the penalty builder couldn’t flatten.
- **Fix:** Added `_coerce_token_ids/_coerce_token_sequence` helpers in `logit_penalty.py` so any EOS specification (int, tensor, iterable) is normalized before building the logits processor.
