# Agents Guide

Orientation for LLM/code agents working in this TRAINING repo. Two-GPU split stays: **4090 = training/orchestration**, **3090 = inference**.

## Snapshot (2025-11-27)
- Active hero: DIO (Qwen3-0.6B) campaign-001 (Binary+SY) ~184k steps; `control/state.json` shows idle (recovered) with active pointer in `control/active_campaign.json` → `campaigns/active/`.
- Daemons run under The Weaver (`.pids/`): training_daemon, Tavern UI (8888), VaultKeeper (8767), data flow/autogen; start via `scripts/start_all.sh`, stop via `scripts/stop_all.sh`.
- Host registry: `config/hosts.json` v3 (4090 trainer 192.168.x.x, 3090 inference 192.168.x.x, NAS storage) with zone warden on 8760; use `core/hosts.py` instead of hardcoded IPs.
- Models/checkpoints: base `models/Qwen3-0.6B/`; active checkpoints in `current_model/` with `.ledger.json` sidecars and index at `status/checkpoint_ledger.json` (do not delete). TITAN (Qwen3-4B) hero available via `configs/heroes/titan-qwen3-4b.yaml` + `scripts/train_4b_full.py` (paged 8-bit Adam + Liger).
- Data lineage + baselines: generator/validator versions tracked in `status/data_lineage.json` (requires `.meta.json` sidecars); transfer/skill baselines live in `status/baselines/` for dashboards and `skill_metrics`.

## Directory Map (what matters)
- `core/` – Training stack (`train.py`, `training_daemon.py`, training_queue/controller, validation/spec + masking_validators, prompts), config_builder (hero→campaign merge), checkpoint_ledger, hosts.
- `guild/` – Skills/progression (sparring, task_master, quests), hero registry (`guild/heroes/*` + `configs/heroes/*.yaml`), campaign CLI; `campaigns/` holds per-hero runs (campaign.json, checkpoints/status/logs).
- `data_manager/` – Curriculum-based autogen via local skill APIs (syllo 8080, binary 8090), quality checks, queueing; state in `data_manager/curriculum_state.json`.
- `tavern/` – Game UI (port 8888: quests/oracle/settings/VRAM calc/scheduler/guild hall); `weaver/` orchestrates daemons (status/restart/shutdown).
- `monitoring/api` – Unified API server (8081) + plugins (training_status, curriculum, GPU, adversarial/regression/confidence/testing/self_correction, skill_metrics, training_analytics, storage, scheduler, inference_model, retention, system_health); `monitoring/ui/` dashboards (master_dashboard with Transfer Learning/Data Lineage cards, syllo dashboard, live_monitor_ui_v2); `monitoring/analytics/` drift/impact trackers.
- `vault/` – VaultKeeper asset registry/ledger API (8767), zone_warden (8760) for service health, storage manager + NAS config (`config/storage.json`).
- `trainer/` – HF trainer configs, optimizer factory (muon, paged AdamW 8-bit, param_groups), profiles (`regime3`, `emoji_think`).
- `data/validation/` – Benchmarks, binary, primitives, syllo_10level sets; baselines consumed by skill_metrics/transfer cards.
- `scripts/` – start_all/stop_all (Weaver), `train_4b_full.py` (4B full FT), DeepSpeed configs in `configs/ds_zero{2,3}_offload.json`.

## Ops Quick Commands
- Start/stop stack: `./scripts/start_all.sh` (launches Weaver + services), `./scripts/stop_all.sh`; check tapestry `python3 weaver/weaver.py --status`, restart service `python3 weaver/weaver.py --restart tavern`, clean shutdown `python3 weaver/weaver.py --shutdown`.
- Training control: `python3 core/training_controller.py pause|resume|status|stop|skip`; queue ops `python3 core/training_queue.py status|list|add <file.jsonl> [--priority high]`.
- Campaigns/heroes: `python3 -m guild.campaigns.cli active|list [hero]`, `python3 -m guild.campaigns.cli switch <hero> <campaign>`, `python3 -m guild.campaigns.cli new --hero titan-qwen3-4b --name "First 4B run"`; hero definitions in `configs/heroes/*.yaml`.
- Data autogen: `python3 data_manager/manager.py status` or `generate --force --count 1000` (needs singleSKILL APIs: syllo 8080, binary 8090).
- Task Master (3090 scheduler): `python3 guild/task_master.py --status|--once|--daemon|--run <task>`; registry in `guild/task_registry.py`.
- Transfer baselines / monitoring API: `python3 monitoring/run_transfer_baseline.py --mode both`; serve dashboards `python3 monitoring/api/server.py` (port 8081) if not already via Weaver.

## Data & Validation Rules
- Format: OpenAI-style chat JSONL; schema enforced by `core/validation/spec.py` (ids: chat_sft_v1, syllo_v1, completion_v1) plus `core/validation/validator.py` (QUICK/STANDARD/DEEP). Masking validators in `core/masking_validators.py` fail fast if packing masks instructions.
- Base prompt lives in `core/prompts.py` (`Today is {date}. You are happy. You enjoy helping others.`); DatasetPreparer enforces it for training runs.
- Autogen uses curriculum in `data_manager/curriculum_state.json` (active skill/level). Generated files named `train_<skill>_level<level>_<count>_<timestamp>.jsonl`.
- Lineage: generators/validators carry VERSION constants and must write `.meta.json` sidecars; aggregated stats at `status/data_lineage.json` and `/api/lineage`.
- Validation sets reorganized under `data/validation/{benchmarks,binary,primitives,syllo_10level}/`; baselines consumed from `status/baselines/`.
- Remote eval/inference stays on 3090 (`inference`: 192.168.x.x:8765 with auth); local skill APIs (8080/8090) are allowed for data generation only.

## Monitoring & Analytics
- Unified API (8081) aggregates plugins above; clear cache via `/api/cache/clear`. Health at `/api/health`, unified payload `/api/unified`, queue `/api/queue`, curriculum `/api/curriculum-state`, lineage `/api/lineage`.
- Dashboards: `http://localhost:8080/master_dashboard.html` (Transfer Learning, Data Lineage, storage, analytics), `http://localhost:8080/syllo_dashboard.html`, `http://localhost:8080/live_monitor_ui_v2.html`; Tavern UI at `http://localhost:8888` (quests, oracle with strict step selection, settings with VRAM calculator and scheduler card, Task Master card).
- VaultKeeper API (8767) exposes ledger/training/asset routes; zone warden (8760) reports service health; Task Master UI data exposed via Tavern Guild Hall card.
- Training analytics consumes `status/data_file_impact.jsonl` (and remote layer_drift/parameter_stability when present); storage plugin reads `config/storage.json` + `status/storage_status.json`.
- Baseline/transfer metrics come from `status/baselines/baseline_*.json` refreshed by `monitoring/run_transfer_baseline*.py`.

## Development & Testing
- Install: `pip install -e .` (lightweight) or `pip install -e ".[training]"` for GPU deps (liger kernel, paged optimizers). Prompt constants stay centralized in `core/prompts.py`.
- Quick tests (CPU): `python3 -m pytest tests/test_inference_auth.py tests/test_retention_manager.py -q`; markers `slow`, `gpu`, `integration` available; full suite expects `current_model/` populated.
- Follow module contracts (`MODULE_CONTRACTS.md`, `TRAINER_CONTRACT.md`); prefer config_builder + hero/campaign config over ad-hoc `config.json` edits.

## Safety / Runtime Guardrails
- Do **not** delete or overwrite `current_model/`, `models/Qwen3-0.6B/`, `status/`, `status/baselines/`, `campaigns/`, or ledger sidecars. Avoid touching `control/` pointers unless switching campaigns via CLI.
- Pause training (`core/training_controller.py pause`) before editing config; locked fields `model_name`, `base_model`, `max_length` in `config.json` remain off-limits without approval.
- Let The Weaver manage daemons; avoid manual kills—use `weaver/weaver.py --restart <svc>` or `stop_all.sh` if needed.
- Host registry is source of truth for service URLs/ports (`config/hosts.json`); avoid hardcoded IPs. Remote inference stays on 3090 only.
- Keep runtime artifacts (`inbox/`, `queue/`, `logs/`, `status/`, `control/`, `test_results/`, `outputs/`) out of git; storage/NAS paths configured in `config/storage.json` should not be pruned.
