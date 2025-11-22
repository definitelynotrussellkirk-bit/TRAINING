# Training Data Contract

This repository expects **every** JSONL sample to follow the same schema and behavioural guarantees so the trainer, monitors, and analytics can operate without guessing. Use this checklist whenever you add new generators or import third‑party data.

## 1. File Format

- **Encoding:** UTF‑8 JSON Lines (`.jsonl`). Each line is a standalone JSON object.
- **Top‑level keys:**
  - `messages`: list of chat turns (see below).
  - `metadata`: object containing origin details (required, even if empty).
- **No BOM / trailing commas**; files should be streamable by `python -m json.tool`.

## 2. Message Structure

- `messages` is ordered oldest → newest.
- **Roles**: `system`, `user`, `assistant` (lowercase). Additional roles are currently ignored.
- **Content**: plain strings; no binary blobs.
- At least one `user` + one `assistant` message. A `system` message is optional (the trainer injects its own enforced system prompt before training).
- **Assistant outputs MUST:**
  - Begin with the four‑emoji prefix (`🤔🤔🤔🤔\n`). The trainer enforces this but data should comply so analytics remain accurate.
  - Exclude `<think>`/`</think>` tags or other metacognition blocks. Sanitise upstream.

## 3. Metadata Requirements

`metadata` powers queue automation and analytics; keep the following keys current so new dashboards stay meaningful.

| Key | Required? | Description |
| --- | --- | --- |
| `skill` | yes | High‑level generator (e.g., `no_think_tags`, `syllo`). |
| `scenario` | recommended | Fine‑grained template (e.g., `math_mc`, `sequence_predict`). |
| `source` | recommended | Data pipeline or dataset tag (`syllo_autogen`, `nothink_batch`). |
| `difficulty` / `length_hint` | optional | Provide if the generator controls difficulty or token length. |
| `generator_version` | optional | Semantic or git hash so regressions can be traced. |
| `notes` | optional | Freeform text; keep short (<200 chars). |

You may include additional fields; they will be preserved in the inference logs.

## 4. Behavioural Guarantees

1. **Deterministic replay:** given the `metadata` (especially `seed`), the generator should be able to reproduce the sample for debugging.
2. **Contract compliance:** instructions in the `user` turn must not contradict the enforced system prompt (which always states “You no longer use `<think>`…”).
3. **Response-only training:** the trainer masks everything except the assistant output, so *all supervision signal must be contained in the assistant message*. Avoid hiding answers elsewhere.
4. **Asset safety:** samples must not reference files outside the repo; if a prompt needs a table or dataset, embed it inline.

## 5. Validation Checklist

Before dropping a batch into `inbox/`:

1. Run `python3 validate_data.py --file your_file.jsonl`.
2. Spot-check a few samples:
   - Confirm emoji prefix + no `<think>`.
   - Ensure `metadata.skill` matches the generator.
   - Verify answers are within the expected JSON schema for that scenario (e.g., SYLLO’s `{solutions, inventory_check}`).
3. Compress/rename large folders using the `inbox/<dataset_name>/training_samples.jsonl` convention so the daemon can ingest automatically.

## 6. Extending the Contract

If you add new metadata fields or scenario types:

1. Document them here (append a table entry).
2. Ensure generators populate the field for **every** sample.
3. Update the trainer/analytics only after the data format is stable; this keeps UI changes predictable.

Keeping this contract tight ensures the training daemon can scale unattended, analytics remain trustworthy, and downstream automation tools (queue watchers, penalty dashboards, etc.) can rely on the data without extra heuristics. Feel free to ping the `Training Daemon Manager` before introducing breaking changes.***
