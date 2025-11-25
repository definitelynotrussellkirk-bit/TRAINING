# Refactor Master Plan

**Created:** 2025-11-24
**Status:** Active
**Priority Order:** Security → Tests → Consistency → Architecture

---

## Overview

This plan addresses feedback identifying sharp edges in the codebase. Issues are broken into discrete tasks that can be completed independently.

## Task Index

| ID | Title | Priority | Effort | Dependencies |
|----|-------|----------|--------|--------------|
| TASK001 | Lock down inference API | CRITICAL | 2h | None |
| TASK002 | Clean up tests | HIGH | 3h | None |
| TASK003 | Wire retention system | HIGH | 2h | None |
| TASK004 | Refactor UltimateTrainer | MEDIUM | 6h | None |
| TASK005 | Refactor TrainingDaemon | MEDIUM | 6h | TASK003 |
| TASK006 | Remove hardcoded paths | MEDIUM | 3h | None |
| TASK007 | Package structure | LOW | 4h | TASK004, TASK005 |
| TASK008 | Consolidate data validation | LOW | 2h | TASK004, TASK005 |
| TASK009 | Background heavy operations | LOW | 4h | TASK005 |

## Execution Order

### Phase 1: Security & CI (Do First)
1. **TASK001** - Lock down inference API (security risk)
2. **TASK002** - Clean up tests (enable CI)

### Phase 2: Consistency
3. **TASK003** - Wire retention system (resolve old vs new)
4. **TASK006** - Remove hardcoded paths

### Phase 3: Architecture
5. **TASK004** - Refactor UltimateTrainer
6. **TASK005** - Refactor TrainingDaemon
7. **TASK008** - Consolidate data validation

### Phase 4: Polish
8. **TASK007** - Package structure
9. **TASK009** - Background heavy operations

## Success Criteria

- [ ] `pytest tests/` passes on fresh machine without GPU
- [ ] Inference API requires authentication
- [ ] No hardcoded `/home/user/` paths in code
- [ ] Single retention system wired to daemon
- [ ] `train.py` < 500 lines
- [ ] `training_daemon.py` < 400 lines

## Progress Tracking

| Task | Started | Completed | Notes |
|------|---------|-----------|-------|
| TASK001 | 2025-11-24 | 2025-11-24 | Auth middleware, endpoint protection, client updates |
| TASK002 | 2025-11-24 | 2025-11-24 | pytest.ini, conftest.py, 20+ tests moved to experiments |
| TASK003 | | | |
| TASK004 | | | |
| TASK005 | | | |
| TASK006 | | | |
| TASK007 | | | |
| TASK008 | | | |
| TASK009 | | | |
