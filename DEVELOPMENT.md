# Development Guide

Working on the codebase, testing, and contributing.

## Setup

```bash
# Bootstrap (first time)
./scripts/bootstrap_dev.sh

# Check environment
python3 -m training doctor

# Install for development
pip install -e .

# Install with GPU deps
pip install -e ".[training]"
```

## Testing

```bash
# Quick tests (CPU)
python3 -m pytest tests/test_inference_auth.py tests/test_retention_manager.py -q

# Full suite (expects current_model/ populated)
python3 -m pytest tests/

# Markers available: slow, gpu, integration
python3 -m pytest -m "not slow" tests/
```

## Code Style

- Follow existing patterns in each module
- Centralized prompts in `core/prompts.py`
- Use `config_builder` + hero/campaign config over ad-hoc `config.json` edits
- Use `core.paths.get_base_dir()` for paths, never hardcode

## Key Contracts

- `MODULE_CONTRACTS.md` - Module interface contracts
- `TRAINER_CONTRACT.md` - Trainer interface

---

# Sharing Checklist

## Current Status: FEEDBACK MODE (Private Sharing OK)

You can share this repo privately for feedback. The following files are gitignored
and won't be pushed:

| File | Contains | Status |
|------|----------|--------|
| `.env` | Network IPs, ports | Gitignored |
| `config/hosts.json` | LAN IPs, SSH usernames | Gitignored |
| `config/devices.json` | Device names, IPs | Gitignored |
| `config/storage_zones.json` | Local paths | Gitignored |
| `config/storage_registry.json` | Asset paths | Gitignored |
| `config/workers.json` | Worker configs | Gitignored |
| `config/secrets.json` | Secrets | Gitignored |
| `vault/*.db` | SQLite databases | Gitignored |

## Before Going PUBLIC

Run this checklist:

### 1. Audit for Personal Data

```bash
# Check for your username in tracked files
git ls-files | xargs grep -l "user" 2>/dev/null

# Check for your home directory
git ls-files | xargs grep -l "/home/user" 2>/dev/null

# Check for private IPs
git ls-files | xargs grep -l "192.168" 2>/dev/null

# Run the hardcode checker
python3 scripts/check_hardcodes.py
```

### 2. Remove/Scrub These Files (if still tracked)

```bash
# These should already be gitignored, but verify:
git status config/hosts.json config/devices.json config/storage_zones.json

# If they show as tracked, remove them:
git rm --cached config/hosts.json config/devices.json \
    config/storage_zones.json config/storage_registry.json \
    config/workers.json
```

### 3. Check config.json

The main `config.json` file may contain:
- Model paths (probably fine - relative paths)
- Remote eval host IPs (scrub these!)
- SSH usernames

Either:
- Gitignore it: Add `config.json` to `.gitignore`
- Or scrub it: Replace IPs with `localhost`, remove usernames

### 4. Check Documentation Files

Some .md files have example paths. These are usually fine (they're examples),
but search for your actual username:

```bash
grep -r "user" *.md scratch/*.md
```

### 5. Check Archive Directory

Old config files in `archive/` may have your settings:

```bash
grep -r "192.168" archive/
grep -r "/home/user" archive/
```

Consider: `git rm -r archive/configs/` if it has personal data.

### 6. Final Verification

```bash
# This should return 0 files for public release:
git ls-files | xargs grep -l "/home/user" 2>/dev/null | wc -l

# This should return 0 files (or only example IPs in docs):
git ls-files | xargs grep -l "192.168" 2>/dev/null | wc -l
```

## Quick Public-Ready Command

When you're ready to go fully public:

```bash
# Nuclear option: remove all potentially sensitive tracked files
git rm --cached -f \
    config/hosts.json \
    config/devices.json \
    config/storage_zones.json \
    config/storage_registry.json \
    config/workers.json \
    config.json \
    archive/configs/*.json \
    data/prediction_history/*.json \
    2>/dev/null

git commit -m "chore: Remove personal config files before public release"
```

## What's Safe to Share

These files are designed to be shared:
- All code (`.py` files) - hardcodes have been fixed
- `.env.example` - template with placeholder values
- `config/*.example.json` - templates for setup
- `config.defaults.json` - default settings with relative paths
- Documentation (README, QUICKSTART, etc.)
- Skill configs (`configs/skills/*.yaml`) - use env vars now
