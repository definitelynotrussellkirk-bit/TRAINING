# Repository Guidelines

## Project Structure & Module Organization
- Training core: `train.py`, `training_daemon.py`, `training_controller.py`, `training_queue.py`, `atomic_ops.py`. LoRA/adapters live under `current_model/` and `current_model_small/`; consolidated versions in `snapshots/` and `consolidated_models/`. **Current merged base:** `/path/to/training/consolidated_models/20251119_152444`.
- Data flow: drop JSONL into `inbox/`; daemon moves items through `queue/` and tracks progress in `status/` (live JSON + logs). Long-form logs in `logs/` and `training_output*.log`.
- Monitoring/UI: `launch_live_monitor.py`, `live_monitor_ui_v2.html` with JS/CSS in `js/` and `css/`; supporting APIs `memory_stats_api.py`, `gpu_stats_api.py`.
- Docs: root guides and `docs/`; safety playbooks in `SAFEGUARDS_SUMMARY.md`, `CRASH_PREVENTION_GUIDE.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
- Start end-to-end stack: `./start_all.sh` (daemon + monitor + metrics APIs).
- Manual training (example): `python3 train.py --dataset inbox/sample.jsonl --model current_model --output-dir adapters/sample_run --epochs 1 --use-qlora`.
- Control loop: `python3 training_controller.py pause|resume|stop|skip|status`; queue ops via `python3 training_queue.py add/list/status`.
- Launch daemon safely via `bin/launch_training_daemon.sh` (guards against multiple instances).
- Monitor: open `http://localhost:8080/live_monitor_ui_v2.html`; logs at `logs/daemon_$(date +%Y%m%d).log` and `status/training_status.json`.

## Coding Style & Naming Conventions
- Python-first; 4-space indent, PEP 8/257, type hints, `pathlib` for paths, `logging` over `print`.
- Config stays JSON; avoid altering locked keys (`max_length`, `model_name`, `base_model`, batch sizing) without explicit approval.
- Treat `current_model*/` and `queue/processing/` as protected; use atomic helpers in `atomic_ops.py`.

## Testing Guidelines
- Fast checks only when needed: `python3 comprehensive_health_check.py`, `python3 config_validator.py`, `python3 verify_checkpoint_resume.py`. Prefer small dry-runs with a tiny JSONL to validate pipeline instead of heavy test suites.
- If adding logic, keep tests light (`pytest test_output_cleaner.py test_dataset_hash.py`) and avoid GPU-heavy runs unless requested.

## Commit & Pull Request Guidelines
- Commits: short imperative lines (e.g., `fix: dedup recents`, `chore: bump monitor refresh`). Keep scope tight.
- PRs/changes: describe intent, risks, commands run, and any UI impact (screenshots/URLs). Note config changes and linked issues/tasks.

## Safety & Operations (Read This)
- **Training Daemon Manager (per user directive, 2025-11-20):** This agent is responsible for launching/monitoring `training_daemon.py`, keeping the queue clean when crashes occur, and documenting any daemon intervention. Always use `bin/launch_training_daemon.sh` for restarts, verify `status/training_status.json` returns to `"training"`, and move any stuck files out of `queue/processing/` before relaunching.
- Never delete `current_model/` or alter locked config keys without owner approval. Always create a backup/description before consolidation (`python3 consolidate_model.py --description "<what was trained>"`).
- Always keep `/path/to/training/consolidated_models/20251119_152444` intact—it is the only approved base, and every adapter/checkpoint must reference it.
- Before/after sessions: `python3 state_tracker.py --check`, `python3 comprehensive_health_check.py`, `python3 verify_checkpoint_resume.py`.
- Graceful control first (controller commands); emergency stop via `touch .stop`. Avoid `kill -9` unless recovering from a hang.
- Keep base dir consistent (`/path/to/training`) to avoid queue/monitor drift; use `toggle_preset.sh` only with confirmation of the active run.

## Operational Snapshot (2025-11-20 @ ≈03:30)
- **Daemon state:** running via `bin/launch_training_daemon.sh` (PID recorded in `.daemon.pid` when active). Watch `training_output.log` and `status/training_status.json`.
- **Active batch:** `syllo_autogen_20251120_032136_count100000.jsonl` (`status/training_status.json` currently `{"status":"training","current_step":~27.8k,"loss":~0.0024}`).
- **Config highlights:** `batch_size=40`, `gradient_accumulation=2`, `max_length=4096`, `model_display_name="MERGED-2025-11-19 • Qwen3 0.6B"`, auto-generate enabled (100k SYLLO puzzles, host `127.0.0.1:8091`).
- **Queue (normal priority):** `training_samples.jsonl`, `training_samples_20k.jsonl`, `syllo_production_v1_20251120_023651.jsonl`, `syllo_batch_20251120_023636.jsonl`, `syllo_mega_batch_20251120_024801.jsonl`. `queue/failed/` currently empty; re-add future failures manually.
- **Housekeeping:** Removed stale JSONL files that previously sat in `queue/failed/`; only last-hour retries remain queued. When new failures appear, triage (retry vs delete) before the next daemon restart.
- **Neural-change instrumentation:** The old “Pattern Coverage” and “LoRA adaptation” UI cards were removed (they never received data). To inspect LoRA layer activity, hook `lora_monitor.py` into the trainer or run it manually around the PEFT model; wire its summary into `training_status` when ready so we can expose real neuron/layer change data instead of dead panels.
- **Dedicated monitor:** `layer_activity_monitor.html` polls `status/training_status.json` and visualizes the new `layer_activity_summary`. Open it in the same live-monitor server (`http://localhost:8080/layer_activity_monitor.html`) to watch top-changing layers in real time.
