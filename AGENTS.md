# Repository Guidelines

## Project Structure & Module Organization
- Training core: `train.py`, `training_daemon.py`, `training_controller.py`, `training_queue.py`, `atomic_ops.py`. LoRA/adapters live under `current_model/` and `current_model_small/`; consolidated versions in `snapshots/` and `consolidated_models/`.
- Data flow: drop JSONL into `inbox/`; daemon moves items through `queue/` and tracks progress in `status/` (live JSON + logs). Long-form logs in `logs/` and `training_output*.log`.
- Monitoring/UI: `launch_live_monitor.py`, `live_monitor_ui_v2.html` with JS/CSS in `js/` and `css/`; supporting APIs `memory_stats_api.py`, `gpu_stats_api.py`.
- Docs: root guides and `docs/`; safety playbooks in `SAFEGUARDS_SUMMARY.md`, `CRASH_PREVENTION_GUIDE.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
- Start end-to-end stack: `./start_all.sh` (daemon + monitor + metrics APIs).
- Manual training (example): `python3 train.py --dataset inbox/sample.jsonl --model current_model --output-dir adapters/sample_run --epochs 1 --use-qlora`.
- Control loop: `python3 training_controller.py pause|resume|stop|skip|status`; queue ops via `python3 training_queue.py add/list/status`.
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
- Never delete `current_model/` or alter locked config keys without owner approval. Always create a backup/description before consolidation (`python3 consolidate_model.py --description "<what was trained>"`).
- Before/after sessions: `python3 state_tracker.py --check`, `python3 comprehensive_health_check.py`, `python3 verify_checkpoint_resume.py`.
- Graceful control first (controller commands); emergency stop via `touch .stop`. Avoid `kill -9` unless recovering from a hang.
- Keep base dir consistent (`/path/to/training`) to avoid queue/monitor drift; use `toggle_preset.sh` only with confirmation of the active run.
