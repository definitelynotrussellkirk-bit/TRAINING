# OBSERVATIONS Directory

## Purpose
This directory contains documented observations, hypotheses, and reasoning patterns about the training system's behavior. The goal is to:

1. **Track hypotheses over time** - Record what we think is happening
2. **Test predictions** - Come back later to see if we were right
3. **Build intuition** - Understand what works and why
4. **Identify patterns** - Find consistent behaviors across runs
5. **Learn from mistakes** - Document what didn't work as expected

## File Naming Convention
`YYYY-MM-DD_<topic>_<type>.md`

Examples:
- `2025-11-11_training_behavior_hypotheses.md`
- `2025-11-15_loss_curve_analysis.md`
- `2025-11-20_generalization_test_results.md`

## Types of Documents

### Hypotheses
- Proposed explanations for observed behavior
- Specific predictions to test
- Timeline for validation
- Update with results as training progresses

### Analyses
- Deep dives into specific phenomena
- Data analysis and visualizations
- Comparisons across runs

### Results
- Test outcomes
- Validation of previous predictions
- Lessons learned

### Anomalies
- Unexpected behavior
- Things that don't match predictions
- Open questions

## How To Use

### When Creating a Hypothesis:
1. State the observation clearly
2. Propose a testable explanation
3. Make specific, measurable predictions
4. Define how to test it
5. Set checkpoints to review

### When Updating:
1. Reference the original hypothesis file
2. Add checkpoint data at specified intervals
3. Compare actual vs predicted outcomes
4. Update conclusions based on evidence
5. Keep original predictions intact (learn from differences)

### When Concluding:
1. Summarize what was correct
2. Identify what was wrong and why
3. Extract general principles learned
4. Suggest new hypotheses based on findings

## Current Active Hypotheses

- **2025-11-11_training_behavior_hypotheses.md**
  - Consistent Universe vs Inconsistent Universe
  - Match % increases fast, Loss decreases slowly (phased learning)
  - High LoRA rank enables meta-pattern learning
  - **Next checkpoint:** 30% training completion

## Tips

- Be specific with predictions (numbers, timings, behaviors)
- Don't edit predictions after making them (track actual vs expected)
- It's OK to be wrong - that's how we learn!
- Cross-reference related observations
- Update regularly at checkpoints

## Related Documentation

- `/TRAINING/CLAUDE.md` - System documentation
- `/TRAINING/CONTINUOUS_TRAINING_GUIDE.md` - Technical details
- `/TRAINING/config.json` - Current configuration
- `/TRAINING/status/training_status.json` - Real-time metrics
