# Master Monitoring System - Data Source Inventory

**Generated:** 2025-11-23 23:24
**Status:** Phase 1, Task 1.1 Complete

## Executive Summary

- **2 GPU Machines:** RTX 4090 (training), RTX 3090 (intelligence)
- **10 Status Files:** 4 local (4090), 6 remote (3090)
- **15 Active Processes:** 8 on 4090, 7 on 3090
- **27 Skill Domains:** Located in `/path/to/skills/`
- **18 HTML Monitors:** Various dashboards and test pages
- **1 Active API Server:** Port 8080 (working)
- **8 Queue Directories:** Processing training data

---

## 1. GPU INFRASTRUCTURE

### Machine 1: RTX 4090 (192.168.x.x) - Training Machine
- **GPU:** NVIDIA GeForce RTX 4090
- **VRAM:** 24564 MiB (24GB)
- **Role:** Primary training + automation
- **Location:** `/path/to/training/`

### Machine 2: RTX 3090 (192.168.x.x) - Intelligence Machine
- **GPU:** NVIDIA GeForce RTX 3090
- **VRAM:** 24576 MiB (24GB)
- **Role:** Monitoring, validation, intelligence systems
- **Location:** `~/TRAINING/`

---

## 2. STATUS FILES (JSON Data Sources)

### Local Status Files (4090 - /path/to/training/status/)

| File | Size | Last Modified | Description |
|------|------|---------------|-------------|
| `training_status.json` | 42KB | 2025-11-23 23:23 | Live training metrics (step, loss, throughput) |
| `latest_preview.json` | 39KB | 2025-11-23 16:34 | Model output previews |
| `curriculum_state.json` | 767B | 2025-11-23 16:07 | Curriculum state (deprecated?) |
| `3090_watchdog_status.json` | 613B | 2025-11-23 23:22 | 3090 health monitoring |

### Remote Status Files (3090 - ~/TRAINING/status/)

| File | Size | Last Modified | Description |
|------|------|---------------|-------------|
| `curriculum_optimization.json` | 25KB | 2025-11-23 23:21 | Curriculum strategy + difficulty metrics |
| `model_comparisons.json` | 6.3KB | 2025-11-23 23:22 | Checkpoint rankings |
| `regression_monitoring.json` | 4.8KB | 2025-11-23 23:22 | Regression detection |
| `confidence_calibration.json` | 1.8KB | 2025-11-23 23:22 | Confidence bins |
| `adversarial_mining.json` | 1.6KB | 2025-11-23 22:29 | Adversarial examples found |
| `checkpoint_sync.json` | 436B | 2025-11-23 23:19 | Checkpoint sync status |

---

## 3. ACTIVE MONITORING PROCESSES

### Local Processes (4090)

| Process | Script | Role |
|---------|--------|------|
| Live Monitor Server | `monitoring/servers/launch_live_monitor.py` | API server (port 8080) |
| 3090 Watchdog | `monitoring/3090_watchdog.py` | Monitor 3090 health |
| 3090 Health Dashboard | `monitoring/3090_health_dashboard.py` | Health metrics |
| 3090 Memory Guardian | `monitoring/3090_memory_guardian.py` | Memory management |
| Automated Testing | `monitoring/automated_testing_daemon.py` | Run validation tests |
| Checkpoint Auto-Deploy | `monitoring/checkpoint_auto_deployment.py` | Deploy best checkpoints |
| Data Gen Automation | `monitoring/data_generation_automation.py` | Auto-generate training data |
| Self-Correction Loop | `monitoring/self_correction_loop.py` | Error correction |

### Remote Processes (3090)

| Process | Script | Role |
|---------|--------|------|
| Curriculum Optimizer | `monitoring/curriculum_optimizer.py` | Optimize difficulty mix |
| Checkpoint Sync | `monitoring/checkpoint_sync_daemon.py` | Sync checkpoints 3090→4090 |
| Regression Monitor | `monitoring/continuous_regression_monitor.py` | Detect bad checkpoints |
| Model Comparison | `monitoring/model_comparison_engine.py` | Rank checkpoints |
| Confidence Calibrator | `monitoring/confidence_calibrator.py` | Calibrate predictions |
| Automated Testing | `monitoring/automated_testing_daemon.py` | Validation suite |
| Self-Correction Loop | `monitoring/self_correction_loop.py` | Error analysis |

**Note:** Adversarial Miner is NOT currently running on 3090

---

## 4. SKILL DOMAINS

**Location:** `/path/to/skills/`
**Total Count:** 27 skills

### Discovered Skills:

1. skill_abstract_algebra
2. skill_anagram_extraction
3. skill_basic_math
4. skill_binary
5. skill_chess (main)
6. skill_chess_attacks
7. skill_chess_best_move
8. skill_chess_check
9. skill_chess_legal_moves
10. skill_chess_make_move
11. skill_chess_material
12. skill_chess_special_moves
13. skill_chess_tactics
14. skill_chess_threats
15. skill_count
16. skill_crypto
17. skill_customer_objection_handling
18. skill_format_translator
19. skill_incident_postmortem
20. skill_information_extraction
21-27: (Additional skills - run `ls /path/to/skills/` for complete list)

**APIs Detected:**
- Syllo API: http://192.168.x.x:8080 (likely)
- Binary API: Port 8090 (mentioned in plans)

**Action Required:** Scan each skill folder for `skill_manifest.json` or API info

---

## 5. EXISTING MONITORS (HTML/JS Dashboards)

**Location:** `/path/to/training/monitoring/ui/`

### Production Monitors:
- `control_room_v2.html` - Main control room (VERIFIED WORKING)
- `live_monitor_ui_v2.html` - Live training monitor
- `systems_dashboard.html` - Systems dashboard (API broken)

### Legacy/Test Monitors:
- `live_monitor_ui_v1.html`
- `live_monitor_ui.html`
- `live_monitor_ui_backup.html`
- `live_monitor_ui_modular.html`
- `index.html`
- `getting_started.html`
- `evolution_viewer.html`
- `layer_activity_monitor.html`

### Test/Debug Pages:
- `debug_api.html`
- `test_api.html`
- `test_display.html`
- `test.html`
- `test_imports.html`
- `test_render.html`
- `test_status_bar.html`

---

## 6. API ENDPOINTS

### Current API Server (Port 8080)

**Base URL:** `http://localhost:8080`
**Status:** ✅ WORKING

**Discovered Endpoints:**
- `/api/status/live` - ✅ WORKING (returns training status)
- `/api/status/system` - Status unknown
- `/api/status/preview` - Status unknown
- `/api/status/evals` - Status unknown
- `/api/config` - Status unknown
- `/api/inbox_files` - Status unknown
- `/api/systems_status` - ❌ BROKEN (was added, then removed)

**Test Results:**
```
✓ API Working
Status: training
Step: 109804/153553
```

**Action Required:** Test all endpoints systematically (Task 1.2)

---

## 7. TRAINING QUEUE

**Location:** `/path/to/training/queue/`

### Queue Status:

| Directory | File Count | Purpose |
|-----------|------------|---------|
| `corrections/` | 0 | Error corrections |
| `failed/` | 0 | Failed training files |
| `high/` | 0 | High priority |
| `normal/` | 1 | Normal priority |
| `low/` | 4 | Low priority (auto-generated) |
| `processing/` | 1 | Currently training |
| `recently_completed/` | 2 | Recently finished |
| `rejected/` | 0 | Rejected files |
| `unvalidated/` | 0 | Needs validation |

### Files in Queue:
- **Processing:** `syllo_mixed_20251123_count100000.jsonl`
- **Normal:** `syllo_curriculum_e60h10_20251123_160653_count1000.jsonl`
- **Low:** 4 auto-generated files (50K examples each)

---

## 8. DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│  RTX 4090 (192.168.x.x) - TRAINING MACHINE               │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Training     │───▶│ Status Files │───▶│ API Server   │ │
│  │ Process      │    │ (4 files)    │    │ Port 8080    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                                         │         │
│         │                                         │         │
│         ▼                                         ▼         │
│  ┌──────────────┐                          ┌──────────────┐│
│  │ Queue System │                          │ HTML Monitors││
│  │ (8 folders)  │                          │ (18 pages)   ││
│  └──────────────┘                          └──────────────┘│
└─────────────────────────────────────────────────────────────┘
                             │
                             │ SSH
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  RTX 3090 (192.168.x.x) - INTELLIGENCE MACHINE          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ INTELLIGENCE SYSTEMS (7 processes)                   │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │ • Curriculum Optimizer (curriculum_optimization.json)│  │
│  │ • Checkpoint Sync (checkpoint_sync.json)             │  │
│  │ • Regression Monitor (regression_monitoring.json)    │  │
│  │ • Model Comparison (model_comparisons.json)          │  │
│  │ • Confidence Calibrator (confidence_calibration.json)│  │
│  │ • Automated Testing (writes to status)               │  │
│  │ • Self-Correction Loop (writes corrections)          │  │
│  └──────────────────────────────────────────────────────┘  │
│                             │                                │
│                             ▼                                │
│                  ┌──────────────────┐                        │
│                  │ Status Files     │                        │
│                  │ (6 JSON files)   │                        │
│                  └──────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ SSH Access
                             ▼
                   (Read by 4090 API Server)
```

---

## 9. EXTERNAL DATA SOURCES (Future)

### Skill APIs (Need Discovery):
- Syllo API (likely on 3090)
- Binary Arithmetic API (mentioned in plans)
- Other skill APIs (TBD)

### Potential Additions:
- Synology NAS devices (storage stats)
- Additional compute machines
- Weights & Biases integration
- GitHub stats
- Discord webhooks

---

## 10. CRITICAL GAPS IDENTIFIED

### Missing/Broken:
1. ❌ Adversarial Miner NOT running on 3090
2. ❌ `/api/systems_status` endpoint removed (was causing errors)
3. ⚠️ No unified API for all data sources
4. ⚠️ No skill manifest standard
5. ⚠️ Multiple redundant monitors

### Action Items:
1. Restart adversarial miner
2. Test all API endpoints (Task 1.2)
3. Document JSON schemas (Task 1.3)
4. Proceed to Phase 2 (plugin system)

---

## 11. REFRESH INTERVALS

### Real-time (< 5 seconds):
- Training status (`training_status.json`)
- GPU stats (nvidia-smi)

### Frequent (10-60 seconds):
- Preview outputs (`latest_preview.json`)
- Queue status

### Periodic (5-10 minutes):
- Curriculum optimizer (300s)
- Checkpoint sync (300s)
- Regression monitor (300s)
- Self-correction loop (300s)

### Slow (10-30 minutes):
- Model comparison (600s)
- Confidence calibrator (600s)
- Automated testing (600s)
- Checkpoint auto-deploy (600s)

---

## 12. NEXT STEPS

✅ **Phase 1, Task 1.1:** COMPLETE - Inventory created
⏩ **Phase 1, Task 1.2:** Test and verify all endpoints
⏩ **Phase 1, Task 1.3:** Document data schemas

**Ready for Phase 2:** Plugin system architecture

---

## APPENDIX: Quick Reference Commands

```bash
# List local status files
ls -lh status/*.json

# List remote status files
ssh 192.168.x.x "ls -lh ~/TRAINING/status/*.json"

# Test main API
curl -s http://localhost:8080/api/status/live | python3 -m json.tool

# Check local monitoring processes
ps aux | grep python3 | grep monitoring | grep -v grep

# Check remote monitoring processes
ssh 192.168.x.x "ps aux | grep python3 | grep monitoring | grep -v grep"

# Count skill folders
ls -d /path/to/skills/skill_* | wc -l

# Check queue status
ls queue/*/

# GPU stats (local)
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

# GPU stats (remote)
ssh 192.168.x.x "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv"
```
