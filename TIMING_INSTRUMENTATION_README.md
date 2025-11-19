# Timing Instrumentation Added - 2025-11-16

## What Was Added

Detailed timing measurements to understand where time is spent during training:

### Metrics Tracked

1. **Total Step Time** - Overall time from start to end of on_step_end callback
2. **Inference Time** - Time spent in model.generate() (only on eval steps)
3. **Validation Loss Time** - Time spent computing loss on validation set (only on eval steps)
4. **Example Loss Time** - Time spent computing loss on current training example
5. **Overhead** - Time spent on everything else (display, status updates, etc.)

### Output Format

**Every 10 steps (Fast Log):**
```
⏱️  Fast log time: 0.123s (loss calc: 0.045s, overhead: 0.078s)
```

**Every 200 steps (Full Eval):**
```
⏱️  TIMING BREAKDOWN (Step 2,800):
   Total:          8.234s
   Inference:      6.123s (74.4%)
   Val Loss:       1.456s (17.7%)
   Example Loss:   0.345s (4.2%)
   Overhead:       0.310s (3.7%)
```

## How to Read the Output

### Fast Log Steps (Every 10)
- **Fast log time:** Total callback overhead
- **loss calc:** Time to compute loss on training example
- **overhead:** Display, status updates, etc.
- **Expected:** ~0.1-0.3 seconds total

### Full Eval Steps (Every 200)
- **Total:** Complete time for the eval step
- **Inference:** model.generate() - MOST EXPENSIVE (typically 60-80%)
- **Val Loss:** Computing loss on 20 validation examples (typically 10-20%)
- **Example Loss:** Computing loss on current training example (typically 2-5%)
- **Overhead:** Everything else (typically 1-5%)
- **Expected:** ~5-10 seconds total

## Performance Analysis

Use this data to:

1. **Identify bottlenecks** - Which operation takes longest?
2. **Measure eval cost** - How much does model inference slow down training?
3. **Optimize** - If overhead is high (>10%), there's an issue
4. **Validate two-tier system** - Confirm fast logs are truly fast

## Expected Patterns

**Healthy system:**
- Fast logs: 0.1-0.3s (mostly overhead)
- Full evals: 5-10s (mostly inference + val loss)
- Overhead: <10% on full evals
- Inference: 60-80% on full evals
- Val loss: 10-20% on full evals

**Red flags:**
- Fast logs > 1s: Something slow running when it shouldn't
- Overhead > 20%: Inefficient status updates or display logic
- Inference > 90%: Model generation is hanging (check timeout)
- Val loss > 50%: Validation set too large or inefficient

## Implementation Details

**Code locations:**
- Timing start: train.py line 596 (`step_start_time = time.time()`)
- Inference timing: train.py line 686 (around model.generate())
- Val loss timing: train.py line 820 (around compute_validation_loss())
- Example loss timing: train.py line 736 (around loss calculation)
- Timing display: train.py line ~845 (after all operations)

**Files modified:**
- train.py - Added timing instrumentation to LiveMonitorCallback

## Usage

Just run training normally - timing output will appear automatically:
```bash
# Start daemon
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &

# Watch timing output
tail -f training_output.log | grep "⏱️"
```

## Cost Analysis

With this instrumentation, you can now answer:
- **Q:** How much does showing model outputs cost?
  - **A:** Look at "Inference" percentage on eval steps

- **Q:** How much does validation loss cost?
  - **A:** Look at "Val Loss" percentage on eval steps

- **Q:** What's the true cost of our two-tier logging?
  - **A:** Compare fast log time (0.1-0.3s) vs full eval time (5-10s)

- **Q:** Is our overhead acceptable?
  - **A:** Look at "Overhead" percentage - should be <10%

## Next Steps

After collecting timing data, you can:
1. Adjust `log_steps` based on overhead cost
2. Adjust `eval_steps` based on inference cost
3. Optimize high-overhead operations
4. Tune validation set size if val loss is too expensive

---

**Status:** Timing instrumentation added and syntax-checked.
**Waiting for:** Next eval step (e.g., step 2900, 3000) to see full timing breakdown.
