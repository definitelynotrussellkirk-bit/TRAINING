# REALM OF TRAINING - Game Design Document

**Last Updated:** 2025-12-06
**Philosophy:** This repo is the method, not the results (see META section)

---

## THE GAME VISION

**This is an RPG Idler game about training an AI hero.**

A **Campaign** is a hero's journey to maximum potential - one attempt to push a model as far as it can go. The goal: discover the level cap.

| RPG Concept | Training Reality |
|-------------|------------------|
| **Hero** | A model (DIO=0.6B, GOU=4B, OJAS=8B) |
| **Campaign** | One playthrough/attempt to reach max potential |
| **Level Cap** | The theoretical limit of what this model can learn |
| **Skill Levels** | Curriculum progression (L1→L50 per skill) |

### Key URLs

| Location | URL | Purpose |
|----------|-----|---------|
| **Tavern** | http://localhost:8888 | Main game UI |
| **Quests** | http://localhost:8888/quests | Quest board |
| **Oracle** | http://localhost:8888/oracle | Chat with checkpoint |
| **Vault** | http://localhost:8888/vault | Browse checkpoints |
| **Settings** | http://localhost:8888/settings | Config, VRAM calc |
| VaultKeeper API | http://localhost:8767 | Asset registry |
| RealmState API | http://localhost:8866 | Real-time state (SSE) |

### Quick Start

```bash
./scripts/bootstrap_dev.sh      # First time only
python3 -m training doctor      # Check environment
python3 -m training start-all   # Start everything
python3 -m training stop-all    # Stop everything
```

---

## MEMORY OPTIMIZATION (THE KNOBS)

Quick reference for 24GB VRAM:

| Model Size | Method | Optimizer | batch_size | max_length |
|------------|--------|-----------|------------|------------|
| 8B | QLoRA | `paged_adamw_32bit` | 1-2 | 1024 |
| 4B | Full | `adamw_torch_fused` | 2 | 2048 |
| 0.6B | Full | `adamw_torch_fused` | 4 | 4096 |

Set in hero YAML: `configs/heroes/<hero>.yaml`
```yaml
memory_profile: "24gb_qlora"  # THE KNOB
training_defaults:
  optimizer_type: "paged_adamw_32bit"
  batch_size: 2
  gradient_accumulation: 4
```

**See ARCHITECTURE.md for full optimizer/PEFT details.**

---

## META: THIS REPO IS THE METHOD

> **The main point is to have fun learning.**

This project is the **method** for doing research, not finished results. Clone the repo and you get the lab:
- How campaigns are structured
- How skills and curricula are defined
- How evaluation and "maxed out" behavior are tracked
- How the human is pulled into forward momentum loops

**Humans are neural nets too.** If the system works, players will eventually try to build skills that produce new skills - a meta-skill that designs curricula for new domains. That's the HITL AGI twist.

**A hero is maxed when** the experience required to gain a new skill level causes too much regression in previously mastered skills.

---

## THE REALM (Architecture Overview)

| Game Location | Technical System | Port |
|--------------|------------------|------|
| **Tavern** | Game UI (`tavern/`) | 8888 |
| **Vault** | VaultKeeper API (`vault/`) | 8767 |
| **Arena** | Hero Loop (`arena/hero_loop.py`) | - |
| **Oracle** | Inference server (3090) | 8765 |
| **RealmState** | Real-time state (`realm/`) | 8866 |

| RPG Term | Technical Equivalent |
|----------|---------------------|
| Quest | Training data file |
| Battle | Training run |
| Champion | Best checkpoint |
| Strain | Loss - floor (difficulty) |
| Effort | Cumulative strain |

**See ARCHITECTURE.md for full system details.**
**See LORE.md for complete RPG vocabulary.**

---

## CONCEPTS (Summary)

### Training Schools
Six schools define HOW the Hero learns:
- **Scribe** (SFT) - Imitate correct examples
- **Mirror** (Sparring) - Self-correction
- **Oracle** (Fortune Teller) - Weight by uncertainty
- Future: Judge (DPO), Champion (RLHF), Whisper (Distill)

### Skills & Primitives
- Skills: SY (syllacrostic), BIN (binary arithmetic)
- Primitives: Atomic cognitive operations (seq_, logic_, mem_, fmt_, attn_, xfm_)
- Skills are composed of primitives

### Strain/Effort Zones
| Zone | Meaning | Action |
|------|---------|--------|
| Recovery | Under-challenged | Level up |
| Productive | Optimal | Continue |
| Stretch | Challenging | Continue if improving |
| Overload | Too hard | Back off |

**See LORE.md for full Training Schools, Primitives, Temple system details.**

---

## GAME STATUS

**Phases 1-5: Complete** - Full game UI, notifications, achievements, strain visualization
**Phase 6: Planned** - Quest modules & primitives (not yet implemented)

**See CHANGELOG.md for full history.**

---

## CRITICAL RULES

### Documentation Policy
**8 Canonical Docs** - Only write to:
- `README.md`, `LORE.md`, `QUICKSTART.md`, `ARCHITECTURE.md`
- `TROUBLESHOOTING.md`, `REMOTE_INFERENCE.md`, `DEVELOPMENT.md`, `CHANGELOG.md`

### Safety Policies
1. **NEVER delete `models/current_model/`** without permission
2. **ALWAYS backup** before risky operations
3. **ASK FIRST** before modifying `config.json` critical params (`max_length`, `model_name`, `base_model`)
4. **Remote Inference** - See `REMOTE_INFERENCE.md`. Training machine does NOT run inference.

### Communication Style
- State facts, not recommendations (unless asked)
- No evaluative language ("excellent", "better")
- Present options without ranking

---

## KEY DIRECTORIES

```
$TRAINING_BASE_DIR/
├── tavern/          # Game UI (8888)
├── vault/           # VaultKeeper API (8767)
├── guild/           # Skills, quests, progression
├── arena/           # Training execution (hero_loop.py)
├── core/            # Training system
├── trainer/         # TrainerEngine, optimizers
├── realm/           # RealmState service (8866)
├── configs/         # Hero YAMLs, skills, services
├── campaigns/       # Per-hero campaign data
├── status/          # Runtime status JSON
└── scripts/         # start_all.sh, bootstrap_dev.sh
```

---

## COMMON TASKS

```bash
# Health check
python3 -m training doctor

# Control training
python3 core/training_controller.py status|pause|resume|stop

# Queue status
python3 core/training_queue.py status

# Service management
python3 core/service_registry.py status
python3 core/service_registry.py realm-up

# Ledger cleanup
python3 -c "from core.checkpoint_ledger import get_ledger; get_ledger().cleanup_stale_entries()"
```

**See ARCHITECTURE.md for full API reference and code examples.**

---

## NOTES FOR CLAUDE

1. Run health check: `python3 -m training doctor`
2. **ASK USER** before making changes
3. **ASK USER** before creating new documentation
4. Trust code as ground truth, not old docs
5. Use `core.paths.get_base_dir()` for paths, never hardcode
6. **When you need details:** Read ARCHITECTURE.md (systems), LORE.md (vocabulary), REMOTE_INFERENCE.md (inference)
