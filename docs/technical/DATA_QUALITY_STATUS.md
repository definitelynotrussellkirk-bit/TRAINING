# Training Data Quality Status

**Current Dataset:** `inbox/leo_10k_fixed.jsonl`
**Status:** ⚠️ QUALITY ISSUES IDENTIFIED - DO NOT USE FOR PRODUCTION TRAINING
**Last Updated:** 2025-11-07

---

## 🚨 Current Status: ON HOLD

**DO NOT USE** the current training data (`leo_10k_fixed.jsonl`) for production training runs.

**Reason:** Multiple data quality bugs identified that will degrade model performance.

---

## 📊 Known Issues

### Issue #1: Bare Count Outputs ✅ IDENTIFIED
- **Count:** 596 examples (5.96%)
- **Severity:** HIGH
- **Problem:** Outputs bare numbers instead of structured data
- **Example:** Assistant responds with just `"0"` instead of `{"count": 0}`

### Issues #2-6: ⏳ IN PROGRESS
- User reports at least 6 total bugs
- Analysis continuing to identify remaining issues

---

## ✅ What's Working

1. **System prompt auto-injection** - Fixed in trainer (2025-11-07)
2. **Data format conversion** - Fixed in trainer (2025-11-03)
3. **Memory optimization** - QLoRA config working (2025-11-03)

---

## 🛠️ Action Plan

### Immediate (Today)
1. ✅ Document known issues
2. ⏳ Continue analysis to find all bugs
3. ⏳ Create comprehensive bug list

### Short-term (This Week)
1. Fix all identified bugs in LEO pipeline
2. Regenerate clean 10k sample dataset
3. Run validation analysis
4. Generate clean 50k+ dataset

### Before Training
- ✅ Verify all issues fixed
- ✅ Run quality validation
- ✅ Spot-check random samples
- ✅ Confirm no bare numbers/malformed outputs

---

## 📋 Checklist Before Training

**Data Quality:**
- [ ] All 6+ bugs identified and documented
- [ ] LEO pipeline fixes implemented
- [ ] Clean data generated
- [ ] Validation analysis shows 100% well-formed responses
- [ ] No bare number outputs
- [ ] All responses properly formatted (JSON/tables/text)
- [ ] System prompts ready (auto-injected by trainer)

**Trainer Ready:**
- [x] System prompt auto-injection working
- [x] Data format conversion working
- [x] QLoRA memory optimization working
- [x] Live monitoring functional

---

## 🔬 How to Check Data Quality

```bash
cd /path/to/training

# Quick check for short responses (potential bare numbers)
python3 << 'EOF'
import json
short = 0
with open('inbox/leo_10k_fixed.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if len(str(data['messages'][1]['content'])) < 5:
            short += 1
print(f"Short responses: {short} / 10000 ({short/100:.1f}%)")
EOF
```

**Expected result for clean data:** 0 short responses

**Current result:** 596 short responses ❌

---

## 📞 Contact

**Issue Tracking:** `/home/user/leo_composition_system/docs/TRAINING_DATA_BUGS.md`
**Fix Location:** LEO composition system generators/
**Status Updates:** This file

---

**Bottom Line:** Wait for clean data before production training. Current data has quality issues that will hurt model performance.
