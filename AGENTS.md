# Agents Guide

Orientation for LLM/code agents working in this TRAINING repo. Two-GPU split: **4090 = training/orchestration**, **3090 = inference**.

## Snapshot (2025-11-25)
- Base model: `models/Qwen3-0.6B/`; active checkpoints live in `current_model/` (~1.6GB, do not delete).
- Daemon is running (`control/state.json` status `training`), auto-generates data when the queue is empty.
- Validation sets reorganized under `data/validation/{benchmarks,binary,primitives}/` plus `val_{easy,medium,hard}_200.jsonl`; old flat sets archived to `data/validation_archive_20251124/`.
- Dashboards updated: master dashboard adds a **Transfer Learning** card; `monitoring/ui/syllo_dashboard.html` for skill-specific views. Unified API now serves `skill_metrics` + `training_analytics`.

## Directory Map (what matters)
- `core/` - Training stack (`train.py`, `training_daemon.py`, queue/controller, validation, logit penalties). Shared prompts live in `core/prompts.py` (single source of truth).
- `trainer/` - Refactored engine/config/profiles (`emoji_think`, `regime3`), monitoring callbacks.
- `monitoring/api/` - Unified API + plugins (status, curriculum, GPUs, adversarial, checkpoint_sync, regression, model_comparison, confidence, testing, self_correction, **skill_metrics**, **training_analytics**). Server lives in `monitoring/api/server.py`.
- `monitoring/analytics/` - Layer drift, parameter stability, per-file impact daemons producing `status/layer_drift.json`, `status/parameter_stability.json`, `status/data_file_impact.jsonl`.
- `monitoring/ui/` - Master dashboard + transfer card (`master_dashboard.html`), syllo dashboard, shared CSS/JS.
- `data/validation/` - Fixed eval sets (transfer benchmarks, primitives, binary) plus val_easy/medium/hard_200. Baseline results land in `status/baselines/` (consumed by skill_metrics).
- `current_model/` - Active training checkpoint dir; `models/Qwen3-0.6B/` is the base. Runtime state: `inbox/`, `queue/`, `logs/`, `status/`, `control/`.
- `monitoring/run_transfer_baseline*.py` - Refresh baseline files; `test_results/` holds recent data quality runs.
- `scratch/`, `archive/` - Plans and backups; leave as-is.

## Ops Quick Commands
- Start stack: `scripts/start_all.sh`
- Queue data: `python3 core/training_queue.py add <file.jsonl> --priority high`
- Check daemon: `python3 core/training_controller.py status`; logs at `logs/daemon_$(date +%Y%m%d).log`
- Manual train: `python3 core/train.py --dataset <path> --output-dir current_model`
- Dashboards: `http://localhost:8080/live_monitor_ui_v2.html`, `http://localhost:8080/master_dashboard.html`, `http://localhost:8080/syllo_dashboard.html`
- Unified API (aggregated metrics): served by `monitoring/api/server.py` -> `/api/unified`, `/api/health`
- Refresh transfer baselines: `python3 monitoring/run_transfer_baseline.py --mode both`

## Data & Validation Rules
- Format: OpenAI-style chat JSONL. Spec gate in daemon via `core/validation/spec.py` (schema ids: `chat_sft_v1`, `syllo_v1`, `completion_v1`). Content gate via `core/validation/validator.py` (QUICK/STANDARD/DEEP).
- DatasetPreparer enforces base prompt from `core/prompts.py` (`Today is {date}. You are happy. You enjoy helping others.`). Profiles may extend but must preserve the base prompt.
- Validation sets: use `data/validation/val_{easy,medium,hard}_200.jsonl` for quick checks; transfer/skill baselines pull from `data/validation/{benchmarks,binary,primitives}/`.
- Auto-generation: daemon requests data via DataManager using local skill APIs (localhost:8080 syllo, localhost:8090 binary) with curriculum-based difficulty.

## Monitoring & Analytics
- Aggregator plugins (default): training_status, curriculum, gpu_4090, gpu_3090, adversarial, checkpoint_sync, regression, model_comparison, confidence, testing, self_correction, **skill_metrics** (local+remote baselines), **training_analytics** (layer drift/parameter stability/data-file impact via SSH to 3090).
- Analytics daemons on 3090 write `status/layer_drift.json`, `status/parameter_stability.json`, `status/data_file_impact.jsonl`; plugin surfaces summaries to dashboards.
- Transfer Learning card: compares base vs trained accuracy for trained skills (syllable/binary), primitives, and benchmarks using `status/baselines/baseline_*.json`.

## Development & Testing
- Install: `pip install -e .` (lightweight) or `pip install -e ".[training]"` for GPU deps.
- Style: PEP8 + full type hints/docstrings per `MODULE_CONTRACTS.md`. Keep prompt constants centralized in `core/prompts.py`.
- Fast tests: `python3 -m pytest tests/test_inference_auth.py tests/test_retention_manager.py -q` (CPU). Markers `slow`, `gpu`, `integration` available; full suite expects GPU + populated `current_model/`.
- Pause before config edits: `python3 core/training_controller.py pause`; avoid touching locked fields (`model_name`, `base_model`, `max_length`) without approval.

## Safety / Runtime Guardrails
- Do **not** delete or overwrite `current_model/`, `models/Qwen3-0.6B/`, `status/`, or `status/baselines/`.
- Keep runtime artifacts (`inbox/`, `queue/`, `logs/`, `status/`, `control/`, `test_results/`) out of git.
- Primary inference runs on 3090 at `http://192.168.x.x:8765` (auth expected).
- Local skill APIs (8080/8090) are allowed for data generation; do not deploy general user-facing inference on training box.
