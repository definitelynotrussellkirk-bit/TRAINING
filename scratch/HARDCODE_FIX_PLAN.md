# Hardcode Elimination Plan

**Created:** 2025-11-28
**Problem:** Centralization infrastructure exists but has ~10% adoption
**Scope:** 342 base dir refs, 321 IP refs across 160+ files

---

## Root Cause Analysis

### Why infrastructure isn't being used:

1. **Discovery problem** - Developers don't know `core.paths` and `core.hosts` exist
2. **Import friction** - Need to add imports; copy-paste a literal is faster
3. **No enforcement** - Nothing prevents new hardcodes from being added
4. **Incomplete coverage** - Some needed helpers don't exist
5. **Fallback defaults** - Infrastructure itself has hardcoded fallbacks, which propagate

### Current Infrastructure (underutilized)

```
core/paths.py          - get_base_dir(), get_status_dir(), etc. (16 files use it)
core/hosts.py          - get_service_url(), get_host(), etc. (4 files use it)
config/hosts.json      - Network config (single source of truth)
```

---

## Comprehensive Fix Strategy

### Phase 1: Harden the Infrastructure

**Goal:** Make the centralization layer complete, fail-fast, and environment-aware.

#### 1.1 Remove Hardcoded Fallbacks from Infrastructure

**Problem:** `core/paths.py` line 85 has `/path/to/training` as fallback.
`core/hosts.py` has `DEFAULTS` dict with hardcoded IPs.

**Fix:**
```python
# core/paths.py - Remove line 85's literal fallback
# Instead of fallback to /home/user, fail with clear error:
raise RuntimeError(
    "Cannot detect base dir. Set TRAINING_BASE_DIR or run from repo root."
)

# core/hosts.py - Remove DEFAULTS dict
# If hosts.json missing, fail with clear error instead of using defaults
```

#### 1.2 Add Missing Path Helpers

Add to `core/paths.py`:
```python
def get_config_dir() -> Path:
    """Get config directory (base/config)."""
    return get_base_dir() / "config"

def get_hosts_config_path() -> Path:
    """Get hosts.json path."""
    return get_config_dir() / "hosts.json"

def get_skills_config_dir() -> Path:
    """Get skills config directory (base/configs/skills)."""
    return get_base_dir() / "configs" / "skills"
```

#### 1.3 Add Missing Host/Service Helpers

Add to `core/hosts.py`:
```python
def get_ssh_command(host_id: str, command: str) -> str:
    """Build SSH command for a host."""
    host = get_host(host_id)
    return f"ssh {host.ssh_user}@{host.host} '{command}'"

def get_remote_path(host_id: str, relative_path: str) -> str:
    """Get full path on a remote host."""
    host = get_host(host_id)
    return f"{host.models_dir}/{relative_path}"
```

#### 1.4 Environment Variable Overrides

Ensure all key values can be overridden via env vars:
```bash
TRAINING_BASE_DIR=/custom/path      # Override base directory
TRAINING_TRAINER_HOST=10.0.0.1      # Override 4090 IP
TRAINING_INFERENCE_HOST=10.0.0.2    # Override 3090 IP
TRAINING_SSH_USER=otheruser         # Override SSH user
```

---

### Phase 2: Migration (Largest Effort)

**Goal:** Replace all hardcoded values with infrastructure calls.

#### 2.1 Priority Order

1. **Core systems** (core/, trainer/, guild/) - Fix first, highest impact
2. **Services** (tavern/, vault/, weaver/) - User-facing
3. **Monitoring** (monitoring/) - 50+ files, bulk of the work
4. **Scripts** (scripts/) - Shell scripts need different approach
5. **Tests** (tests/) - Should use fixtures
6. **Documentation** - Reference config not literals

#### 2.2 Migration Patterns

**Pattern A: Python - Direct replacement**
```python
# Before
BASE_DIR = Path("/path/to/training")

# After
from core.paths import get_base_dir
BASE_DIR = get_base_dir()
```

**Pattern B: Python - Service URL**
```python
# Before
api_url = "http://192.168.x.x:8765"

# After
from core.hosts import get_service_url
api_url = get_service_url("inference")
```

**Pattern C: Python - SSH command**
```python
# Before
ssh_cmd = f"ssh user@xxx.xxx.88.149 '{cmd}'"

# After
from core.hosts import get_host
host = get_host("3090")
ssh_cmd = f"ssh {host.ssh_user}@{host.host} '{cmd}'"
```

**Pattern D: Shell scripts**
```bash
# Before
BASE_DIR="/path/to/training"

# After (use env var with auto-detect fallback)
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Or source a common script
source "$(dirname "$0")/common.sh"
```

**Pattern E: HTML templates (Jinja2)**
```html
<!-- Before -->
<option value="192.168.x.x">RTX 3090</option>

<!-- After - Populate from server -->
{% for host in inference_hosts %}
<option value="{{ host.host }}">{{ host.name }}</option>
{% endfor %}
```

#### 2.3 File-by-File Migration List

Generate with:
```bash
# All Python files with hardcoded base dir
grep -r "/path/to/training" --include="*.py" -l | sort

# All Python files with hardcoded IPs
grep -rE "192\.168\.(88|30)\.[0-9]+" --include="*.py" -l | sort
```

**Estimated effort per category:**
- Core (10 files): 1-2 hours
- Guild (8 files): 1 hour
- Vault (12 files): 1-2 hours
- Monitoring (50+ files): 4-6 hours
- Scripts (15 files): 1-2 hours
- Tests (5 files): 30 min
- Total: ~10-15 hours of mechanical changes

---

### Phase 3: Enforcement

**Goal:** Prevent new hardcodes from being added.

#### 3.1 Pre-commit Hook

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: no-hardcoded-paths
        name: Check for hardcoded paths
        entry: scripts/hooks/check_hardcodes.sh
        language: script
        files: \.(py|sh|html)$
```

Create `scripts/hooks/check_hardcodes.sh`:
```bash
#!/bin/bash
# Fail if new hardcoded values are introduced

PATTERNS=(
    "/path/to/training"
    "192.168.x.x"
    "192.168.x.x"
    "192.168.x.x"
)

for file in "$@"; do
    for pattern in "${PATTERNS[@]}"; do
        if grep -q "$pattern" "$file"; then
            echo "ERROR: Hardcoded value '$pattern' found in $file"
            echo "Use core.paths or core.hosts instead"
            exit 1
        fi
    done
done
```

#### 3.2 CI Check (if CI exists)

Add to CI pipeline:
```yaml
- name: Check for hardcoded values
  run: |
    count=$(grep -rE "(192\.168\.(88|30)\.[0-9]+|/home/user)" --include="*.py" | wc -l)
    if [ "$count" -gt 0 ]; then
      echo "Found $count hardcoded values"
      exit 1
    fi
```

#### 3.3 Linting Rule

Add custom rule to any Python linter:
```python
# For pylint: create checker plugin
# For ruff: use ban-specific pattern
```

---

### Phase 4: Testing & Validation

#### 4.1 Portability Tests

```python
# tests/test_portability.py
def test_no_hardcoded_base_dir():
    """Ensure no hardcoded base directory paths in Python files."""
    import subprocess
    result = subprocess.run(
        ["grep", "-r", "/path/to/training", "--include=*.py", "."],
        capture_output=True
    )
    # Should find 0 matches (or only in test files/scratch)
    assert result.returncode == 1, "Found hardcoded base dir"

def test_no_hardcoded_ips():
    """Ensure no hardcoded IP addresses."""
    # Similar check for IPs
```

#### 4.2 Environment Override Tests

```python
def test_base_dir_env_override():
    """Verify TRAINING_BASE_DIR env var is respected."""
    import os
    os.environ["TRAINING_BASE_DIR"] = "/tmp/test"
    from core.paths import get_base_dir
    get_base_dir.cache_clear()
    assert str(get_base_dir()) == "/tmp/test"
```

---

## Implementation Sequence

### Week 1: Infrastructure Hardening
1. [ ] Remove fallback defaults from `core/paths.py`
2. [ ] Remove DEFAULTS from `core/hosts.py`
3. [ ] Add missing path helpers
4. [ ] Add missing host helpers
5. [ ] Add environment variable override support
6. [ ] Write portability tests

### Week 2: Core Migration
1. [ ] Migrate `core/*.py` (10 files)
2. [ ] Migrate `guild/*.py` (8 files)
3. [ ] Migrate `vault/*.py` (12 files)
4. [ ] Migrate `trainer/*.py` (5 files)

### Week 3: Services & Monitoring
1. [ ] Migrate `tavern/*.py` and templates
2. [ ] Migrate `weaver/*.py`
3. [ ] Migrate `monitoring/*.py` (50+ files - bulk of work)
4. [ ] Migrate `sentinels/*.py`

### Week 4: Scripts, Tests, Docs
1. [ ] Create `scripts/common.sh` for shell scripts
2. [ ] Migrate all shell scripts
3. [ ] Update tests to use fixtures
4. [ ] Update documentation to reference config
5. [ ] Add pre-commit hook
6. [ ] Final grep audit to confirm zero violations

---

## Quick Wins (Can Do Now)

These changes have high impact and low risk:

1. **Add environment override for base dir** - Already partially exists, just ensure it works

2. **Fix the 4 highest-traffic files**:
   - `core/train.py` - Training script, runs constantly
   - `core/eval_runner.py` - Eval system
   - `vault/server.py` - VaultKeeper API
   - `tavern/server.py` - Main UI

3. **Add `/api/config` endpoint to Tavern** - Return hosts from hosts.json for dynamic UI population

4. **Update Jinja templates** - Pass hosts to templates from server, remove hardcoded options

---

## Metrics to Track

| Metric | Before | Target |
|--------|--------|--------|
| Files with hardcoded base dir | 160 | 0 (production code) |
| Files with hardcoded IPs | 106 | 0 (production code) |
| Files using core.paths | 16 | 100+ |
| Files using core.hosts | 4 | 50+ |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing functionality | Gradual rollout, test each change |
| Missing a hardcode | Grep audit after each phase |
| Import cycles | `core/paths.py` and `core/hosts.py` should have zero local imports |
| Performance (repeated config loads) | Already uses `@lru_cache` |

---

## Decision: Keep vs Remove Fallbacks?

**Option A: Fail Fast (Recommended)**
- Remove all fallback defaults
- Code fails immediately if config is wrong
- Forces proper setup but prevents silent misconfiguration

**Option B: Graceful Degradation**
- Keep fallbacks but log warnings
- Works on the 4090 without setup
- Risk: hides configuration problems

**Recommendation: Option A** - Better to fail clearly than work incorrectly.
