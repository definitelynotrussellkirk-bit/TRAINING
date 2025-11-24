# Repository Guidelines

## Project Structure & Module Organization
- `core/`: Training orchestrator (`train.py`, daemon/controller/queue utilities, validator, status handlers).
- `management/`: Checkpoint/version lifecycle (`consolidate_model.py`, `backup_manager.py`, `retention_manager.py`, disk automation).
- `monitoring/servers/`: Dashboards/APIs on ports 8080-8082.
- `scripts/`: Ops helpers (`start_all.sh`, health checks, checkpoint sync, monitor launchers).
- `tools/`, `data_manager/`: Data validation, config editing, and remote evaluation.
- Runtime dirs `inbox/`, `queue/`, `models/current_model/`, `logs/`, `status/` are generated artifacts; keep them out of git. Tests live in `tests/`.

## Build, Test, and Development Commands
- Python 3.10+ with CUDA drivers. Recommended install: `pip install torch transformers datasets peft accelerate bitsandbytes jq`.
- Start the full stack (daemon, disk manager, monitors): `scripts/start_all.sh`.
- Queue data: `python3 core/training_queue.py add inbox/sample.jsonl --priority high`; inspect: `python3 core/training_queue.py status`.
- Check status/logs: `python3 core/training_controller.py status` and `tail -f logs/daemon_$(date +%Y%m%d).log`.
- Manual run for experiments: `python3 core/train.py --dataset inbox/example.jsonl --output-dir models/current_model`.

## Coding Style & Naming Conventions
- Follow PEP8: 4-space indent; snake_case for functions/modules, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- Prefer type hints and docstrings on public functions; keep GPU work behind explicit checks and reuse existing helpers (`DatasetValidator`, `TrainingStatusWriter`, `RetentionManager`) instead of ad-hoc utilities.
- Align CLI flags/log phrasing with `core/train.py` and controller scripts; emit user-facing output through existing status/log channels rather than raw prints.

## Testing Guidelines
- Framework: pytest. Target fast suites first: `python3 -m pytest tests/test_retention_manager.py -v`. Full run (`python3 -m pytest tests -q`) expects GPU and populated `models/current_model/`.
- Name tests `test_<feature>.py` with functions starting `test_...`; use fixtures/temp dirs and avoid writing under `models/` or `queue/`.
- Prefer assertions on metrics/log content over console prints; refresh fixtures when retention/queue behaviors change.

## Commit & Pull Request Guidelines
- Commit messages use the observed convention (`fix: ...`, `perf: ...`, `chore: ...`) with concise, imperative summaries.
- PRs should outline the change, risks, and verification (commands run, logs/metrics). Link issues/tasks and call out training downtime or data impacts.
- Include screenshots or CLI snippets when modifying monitoring UIs or status output; for ops changes, add quick rollback notes.

## Security & Configuration Tips
- Config lives in `config.json` and `.config_lock.json`; avoid hardcoding tokens/paths and keep backups out of git.
- Do not commit checkpoints, logs, or inbox/queue data; they are runtime artifacts. Clean temp files after tests.
- When editing config during a run, pause via `python3 core/training_controller.py pause`, edit, then resume.
