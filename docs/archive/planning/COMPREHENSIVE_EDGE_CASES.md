# Comprehensive Edge Case Analysis - Training System

**Created:** 2025-11-16
**Purpose:** Identify ALL edge cases that could break the system

---

## 🎯 Categories of Edge Cases

### 1. FILE & DATA EDGE CASES

#### Empty/Tiny Files
- [ ] **Empty file (0 examples)** - What happens?
- [ ] **File with 1 example** - Can it train?
- [ ] **File with < batch_size examples** - Steps calculation?
- [ ] **File that results in 0 steps** - Does it exit gracefully?

**Risks:**
- Division by zero in step calculation
- Training exits immediately
- Queue marks as "completed" when nothing trained

**Tests needed:**
```bash
# Create empty file
echo "" > inbox/empty.jsonl
# Expected: Daemon should skip or error gracefully

# Create 1-example file
echo '{"messages": [...]}' > inbox/single.jsonl
# Expected: Should train or warn about insufficient data
```

#### Corrupt Data
- [ ] **Malformed JSON** - Does validation catch it?
- [ ] **Missing required fields** - Does tokenizer fail gracefully?
- [ ] **Invalid UTF-8** - Does parser handle it?
- [ ] **Extremely long lines** - Memory issues?
- [ ] **Binary data in JSONL** - Does it crash?

**Risks:**
- Training crashes mid-batch
- Partial file processed, rest lost
- Tokenizer OOM

**Tests needed:**
```bash
# Malformed JSON
echo '{"messages": [{"role": "user", "content": "test"' > inbox/bad.jsonl
# Expected: Validation should fail, file should not train

# Invalid UTF-8
echo -e '\xff\xfe Invalid UTF-8' > inbox/bad_utf8.jsonl
# Expected: Should error gracefully, not crash
```

#### File System Edge Cases
- [ ] **File deleted while training** - Does training crash?
- [ ] **File modified during training** - Corrupt data?
- [ ] **Symbolic link to file** - Does it follow?
- [ ] **File in subdirectory** - Already handled?
- [ ] **File permissions changed** - Can't read mid-training?

**Risks:**
- FileNotFoundError mid-training
- Data inconsistency
- Training hangs

---

### 2. CHECKPOINT & STEP EDGE CASES

#### Step Boundaries
- [ ] **Checkpoint exactly at max_steps** - Does it retrain?
- [ ] **Checkpoint > max_steps** - ✅ JUST FIXED THIS
- [ ] **Checkpoint at step 0** - Fresh start?
- [ ] **No checkpoints exist** - First training?
- [ ] **Checkpoint-N exists but N != global_step** - Mismatch?

**Risks:**
- Training skipped when it shouldn't be
- Steps miscounted
- Resume from wrong checkpoint

**Tests needed:**
```python
# Simulate checkpoint at exact max_steps
# Expected: Should train new file, not skip

# Simulate checkpoint past max_steps
# Expected: Should still train (test the fix!)
```

#### Checkpoint Corruption
- [ ] **trainer_state.json corrupt** - Can it recover?
- [ ] **trainer_state.json missing** - Falls back to 0?
- [ ] **global_step = null/undefined** - Handled?
- [ ] **global_step negative** - Validation?
- [ ] **Multiple checkpoint dirs with same step** - Which one wins?

**Risks:**
- Training starts from wrong step
- Progress lost
- Duplicate step numbers

---

### 3. QUEUE & MULTI-FILE EDGE CASES

#### Sequential Files
- [ ] **3+ files in sequence** - Steps keep accumulating?
- [ ] **10+ files in sequence** - No overflow?
- [ ] **Same file added twice** - Trains twice or dedupe?
- [ ] **File A, B, then A again** - Steps continue from last?

**Risks:**
- Step accumulation breaks after N files
- Integer overflow (unlikely but possible)
- Duplicate training

**Tests needed:**
```bash
# Train 5 files in sequence
# Verify: step_after_file5 = step_after_file4 + steps_file5
# NOT: step_after_file5 = steps_file5
```

#### Priority & Ordering
- [ ] **Same file in multiple queues** - Which priority wins?
- [ ] **High priority file added mid-training** - Interrupts?
- [ ] **Files complete out of order** - Queue state consistent?
- [ ] **All files fail** - System recovers?

---

### 4. TRAINING PROCESS EDGE CASES

#### Resource Limits
- [ ] **OOM during training** - Partial progress saved?
- [ ] **Disk full during checkpoint save** - Corrupt checkpoint?
- [ ] **GPU crashes mid-batch** - Recovery?
- [ ] **CPU out of memory** - Graceful degradation?

**Risks:**
- Partial checkpoint saved (corrupt)
- Training progress lost
- System hangs

#### Numeric Issues
- [ ] **Loss becomes NaN** - Training stops?
- [ ] **Loss becomes infinity** - Handled?
- [ ] **Gradient overflow** - Clipping works?
- [ ] **Learning rate = 0** - No progress but doesn't crash?
- [ ] **All gradients = 0** - Dead neurons?

**Risks:**
- Training continues with NaN (corrupts model)
- Loss metrics broken
- Silent failure

**Tests needed:**
```python
# Inject NaN loss
# Expected: Training should stop with clear error
# NOT: Continue training and save NaN-corrupted model
```

---

### 5. CONTINUOUS TRAINING ACROSS RESTARTS

#### Long-Term Accumulation
- [ ] **Training for days/weeks** - Step counter overflow?
- [ ] **100+ files trained** - Global step still accumulates?
- [ ] **Daemon restart between files** - Continues correctly?
- [ ] **System reboot mid-training** - Recovery?

**Risks:**
- Integer overflow (if using 32-bit ints)
- State desync after restart
- Lost queue position

**Tests needed:**
```python
# Simulate training to step 1,000,000
# Verify: No overflow, metrics still work
```

#### Config Changes
- [ ] **max_length changed between files** - Uses new or old?
- [ ] **batch_size changed** - Step calculation breaks?
- [ ] **Learning rate changed** - Applied immediately?
- [ ] **Model path changed** - Loads wrong model?

**Risks:**
- Training uses inconsistent config
- Incompatible checkpoint loaded
- Metrics become meaningless

---

### 6. VALIDATION & EVOLUTION EDGE CASES

#### Validation Set Issues
- [ ] **Validation set empty** - Skip validation?
- [ ] **Validation set > training set** - Handled?
- [ ] **Validation examples truncated** - Metrics wrong?
- [ ] **Validation file corrupt** - Graceful fallback?

**Risks:**
- Validation loss always 0 or NaN
- Metrics misleading
- Training crashes during validation

#### Evolution Tracking
- [ ] **Evolution directory full** - Stops saving?
- [ ] **Evolution snapshot too large** - OOM?
- [ ] **Can't write snapshot** - Training continues?
- [ ] **1000+ snapshots** - Performance degradation?

---

### 7. DAEMON & SYSTEM EDGE CASES

#### Daemon Lifecycle
- [ ] **Multiple daemons running** - Race conditions?
- [ ] **Daemon crashes mid-training** - File stuck in "processing"?
- [ ] **Daemon running for weeks** - Memory leaks?
- [ ] **Daemon restarted while saving checkpoint** - Corrupt checkpoint?

**Risks:**
- Concurrent writes to same file
- Files lost in limbo state
- Memory grows unbounded
- Data corruption

**Tests needed:**
```bash
# Start two daemons
# Expected: Second daemon should detect first and exit
# OR: Lock file prevents concurrent access
```

#### Time & Clock Edge Cases
- [ ] **System clock jumps backward** - Timestamp issues?
- [ ] **System clock jumps forward** - Retry logic breaks?
- [ ] **Timezone change** - Log timestamps wrong?
- [ ] **Training crosses midnight** - Daily snapshot triggers mid-file?

**Risks:**
- Files deleted prematurely
- Snapshots created at wrong time
- Retry delays become negative

---

### 8. OUTPUT & METRIC EDGE CASES

#### Accuracy Calculation
- [ ] **0 evaluations** - Division by zero?
- [ ] **All examples match** - 100% accuracy (valid)?
- [ ] **No examples match** - 0% accuracy (concerning but valid)?
- [ ] **Accuracy > 100%** - Bug in counting?

#### Output Length Tracking
- [ ] **Output length = 0** - Empty response?
- [ ] **Output length > max_length** - Truncation detected?
- [ ] **max_golden_output_length keeps growing** - Memory leak?
- [ ] **Output length negative** - Bug in tokenizer?

**Risks:**
- Metrics incorrect
- Memory grows unbounded
- Misleading accuracy reports

---

### 9. INTEGRATION EDGE CASES

#### Multi-System Interactions
- [ ] **Training + consolidation at same time** - Race condition?
- [ ] **Training + backup at same time** - Lock contention?
- [ ] **Training + queue manipulation** - State desync?
- [ ] **Training + model deletion** - Crash?

**Risks:**
- Data corruption
- Deadlocks
- Lost progress

---

### 10. EDGE CASES I JUST FIXED (BUT NEED TO VERIFY)

#### The max_steps Fix
- [ ] **File 1 → 2488 steps, File 2 → should reach 4976**
  - With fix: ✅ Should work
  - Need to verify: Actually test it!

- [ ] **File 1 → 100 steps, File 2 → 50 steps, File 3 → 75 steps**
  - Expected: 100 → 150 → 225
  - Need to verify: Does it work for 3+ files?

- [ ] **Checkpoint at 1000, new file with 10 steps**
  - Expected: Trains to 1010
  - Old behavior: Would skip (1000 >= 10)
  - New behavior: Should train (max_steps = 1010)

---

## 🧪 PRIORITY TEST MATRIX

### P0 - CRITICAL (Must test before production use)
1. ✅ Multi-file continuous training (2+ files)
2. ⏳ Checkpoint > max_steps (verify fix works)
3. ⏳ Empty/0-step files don't break queue
4. ⏳ Corrupt checkpoint recovery
5. ⏳ OOM during training (partial checkpoint)

### P1 - HIGH (Should test soon)
6. ⏳ 3+ files in sequence
7. ⏳ Daemon crash recovery
8. ⏳ NaN loss handling
9. ⏳ Validation set errors
10. ⏳ Duplicate files in queue

### P2 - MEDIUM (Test when possible)
11. ⏳ Very large files (millions of examples)
12. ⏳ Config changes between files
13. ⏳ Multiple daemon instances
14. ⏳ File deleted mid-training
15. ⏳ Long-term accumulation (100+ files)

### P3 - LOW (Nice to have)
16. ⏳ Integer overflow (step > 2^31)
17. ⏳ Clock skew issues
18. ⏳ Symbolic links
19. ⏳ UTF-8 edge cases
20. ⏳ Evolution snapshot disk full

---

## 📋 TEST IMPLEMENTATION PLAN

### Immediate (Today)
```bash
# 1. Verify the max_steps fix
python3 test_continuous_training.py

# 2. Test empty file
echo "" > inbox/empty_test.jsonl
# Monitor: Should skip or error gracefully

# 3. Test 3-file sequence
# Create 3 small files, verify steps accumulate
```

### This Week
```bash
# 4. Corrupt checkpoint test
# Manually corrupt trainer_state.json
# Verify: Falls back gracefully

# 5. NaN loss test
# Inject NaN into training
# Verify: Training stops with clear error
```

---

## 🎯 SYSTEMATIC TESTING APPROACH

### For Each Edge Case:
1. **Identify** - What's the edge case?
2. **Predict** - What should happen?
3. **Test** - Does it actually happen?
4. **Fix** - If broken, fix it
5. **Verify** - Re-test to confirm
6. **Document** - Add to test suite

### Test Categories:
- **Unit Tests** - Individual function behavior
- **Integration Tests** - Multi-component interactions
- **System Tests** - End-to-end workflows
- **Chaos Tests** - Inject failures, verify recovery

---

## ✅ NEXT STEPS

1. **Run existing test:** `python3 test_continuous_training.py`
2. **Create additional tests** for P0 edge cases
3. **Document results** in test log
4. **Fix any failures** found
5. **Add to CI/CD** (if applicable)

---

**Remember:** The user SPECIFICALLY asked for edge case testing.
I failed once. Won't happen again.

**Motto:** "If you can imagine it breaking, test that it doesn't break."
