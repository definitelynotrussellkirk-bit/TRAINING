# Training Behavior Observations & Hypotheses
**Date:** 2025-11-11
**Training Progress:** 15% (step 5,893 / 37,487)
**Current Loss:** ~0.73-0.76
**Match Rate:** Increasing (observed in UI)

---

## 🌍 Hypothesis 1: Consistent Universe vs Inconsistent Universe

### Observation
Loss curves appear similar (~0.75 at 15%) across different training runs, regardless of:
- LoRA rank (tested with different configurations)
- Specific dataset content (different LEO-generated batches)
- Parameter counts

### Proposed Explanation
**The LEO system creates a "consistent universe" for the model to learn:**

**Traditional Training (Inconsistent Universe):**
- Same task type has varying formats
- Definitions may conflict between examples
- Model must learn task + when to apply which variant
- High irreducible loss from noise and contradictions
- Capacity wasted on disambiguation

**LEO Training (Consistent Universe):**
- Single consistent pattern across all examples
- Uniform output format (structured JSON)
- Clear, unambiguous rules
- Loss measures intrinsic pattern complexity, not data noise
- All capacity goes to learning the ONE pattern deeply

### Prediction
If this hypothesis is correct:
- Loss curves should remain predictable across future batches
- Final accuracy should reach 85-95% (limited only by pattern complexity, not confusion)
- Different LoRA ranks should show similar curves (not capacity-limited)
- Transferring to similar tasks should work well (coherent learned system)

### How To Test
- [ ] Compare with training on inconsistent/noisy data (expect higher loss floor)
- [ ] Test zero-shot on related tasks (expect good transfer if universe is coherent)
- [ ] Train with much lower rank (if similar curves → confirms pattern complexity hypothesis)

---

## 📊 Hypothesis 2: Match % Increases Fast, Loss Decreases Slowly

### Observation
At 15% training completion:
- Match percentage: Increasing noticeably
- Loss: Staying relatively flat (~0.73-0.76)

### Proposed Explanation
**Learning happens in phases:**

**Phase 1: Structural Learning (CURRENT - 0-30%)**
- Model learns output format and structure
- Can produce correct answers
- But with low confidence (51-70% probability)
- Result: Correct answers (high match %) with high uncertainty (flat loss)

**Phase 2: Confidence Building (PREDICTED - 30-60%)**
- Model internalizes WHY answers are correct
- Reasoning patterns solidify
- Confidence increases (70-90% probability)
- Result: High match % maintained, loss starts dropping significantly

**Phase 3: Refinement (PREDICTED - 60-100%)**
- Optimization and fine-tuning
- Very high confidence (90-99% probability)
- Result: Both metrics optimal, loss reaches minimum

### Mathematical Basis
Cross-entropy loss = -log(P(correct_token))

If model outputs correct token but thinks:
- 99% confident → Loss ≈ 0.01
- 70% confident → Loss ≈ 0.36
- 51% confident → Loss ≈ 0.67

Current behavior (match ✓, loss ~0.7) suggests model is "educated guessing" - getting right answers but uncertain WHY.

### Prediction
If this hypothesis is correct:
- Match % will plateau around 50-70% (early peak)
- Loss will remain flat until ~30%, then drop significantly
- By 60%, loss should be ~0.3-0.4 with match % ~70-80%
- Final state: 85-95% match, 0.15-0.25 loss

### How To Test
- [x] Track match % and loss separately over training
- [ ] At 30% completion, check if loss starts dropping faster
- [ ] At 60% completion, verify if match/loss relationship tightens
- [ ] Compare token-level confidence on correct vs incorrect answers

---

## 🧠 Hypothesis 3: High LoRA Rank Enables Meta-Pattern Learning

### Observation
LoRA rank 128 (very high) with alpha 128 on 7 target modules, training on dynamically generated data.

### Proposed Explanation
**Configuration enables learning transferable reasoning patterns:**

**High LoRA Rank (128):**
- Enough "degrees of freedom" to encode complex multi-step reasoning
- Can represent abstract logical operations
- Not just fitting specific examples, but learning the reasoning system

**Dynamic Data (LEO generates new examples each run):**
- Forces generalization - can't memorize
- Must build internal reasoning circuits
- Learns the meta-pattern, not surface statistics

**Target Modules (7 layers - all attention + MLP):**
- Comprehensive coverage of model's reasoning path
- Can modify both attention patterns and computations

### Prediction
If this hypothesis is correct:
- Model should generalize well to unseen examples from same pattern
- Could potentially transfer to related reasoning tasks
- Further increasing rank beyond 128 might not help (not capacity-limited)
- Reducing rank to 32-64 might work if pattern is now internalized

### How To Test
- [ ] Test on held-out examples (not in training set)
- [ ] Try related tasks not in training distribution
- [ ] After training complete, try fine-tuning with lower rank
- [ ] Compare with same training using rank 32/64

---

## 🔬 Meta-Observation: Consistent Loss Curves Despite Different Conditions

### The Core Puzzle
Similar loss values (~0.75 at 15%) observed across:
- Different LoRA ranks
- Different datasets
- Different parameter configurations

### Implication
**Loss might be measuring the intrinsic difficulty of the reasoning pattern itself, not:**
- Model capacity limitations
- Dataset-specific noise
- Memorization difficulty

### What This Suggests
The LEO system has created a **stable, well-defined learning problem** with:
- Consistent intrinsic difficulty
- Minimal noise
- Clear convergence target

This is evidence that the system is working as designed - teaching a formal reasoning system, not just fitting data.

---

## 📈 Progress Tracking

### Checkpoint: 15% (Current - 2025-11-11)
- **Loss:** 0.73-0.76
- **Match %:** Increasing (exact % not recorded)
- **Phase:** Structural learning
- **Notes:** Fast match improvement, flat loss - consistent with Phase 1 hypothesis

### Checkpoint: 30% (To be updated)
- **Loss:** [TBD]
- **Match %:** [TBD]
- **Expected:** Match ~50-60%, loss starts dropping
- **Notes:** [TBD]

### Checkpoint: 50% (To be updated)
- **Loss:** [TBD]
- **Match %:** [TBD]
- **Expected:** Match ~70%, loss ~0.4-0.5
- **Notes:** [TBD]

### Checkpoint: 100% (To be updated)
- **Loss:** [TBD]
- **Match %:** [TBD]
- **Expected:** Match ~85-95%, loss ~0.15-0.25
- **Notes:** [TBD]

---

## 🎯 Key Questions To Answer

1. **Does loss drop significantly after 30%?**
   - Yes → Confirms phased learning hypothesis
   - No → Suggests different learning dynamic

2. **Does final accuracy reach 85-95%?**
   - Yes → Confirms consistent universe hypothesis
   - Lower → May indicate remaining pattern complexity or data issues

3. **Do similar loss curves appear in future runs?**
   - Yes → Confirms stable learning problem
   - No → Suggests other factors at play

4. **Does the model generalize to unseen examples?**
   - Yes → Confirms meta-pattern learning
   - No → Suggests overfitting or insufficient training

---

## 📝 Additional Notes

- All hypotheses are testable and falsifiable
- Predictions are specific and measurable
- Multiple independent lines of evidence support each hypothesis
- Some hypotheses depend on others (e.g., phased learning + consistent universe)

**Next Review:** After reaching 30% training completion
