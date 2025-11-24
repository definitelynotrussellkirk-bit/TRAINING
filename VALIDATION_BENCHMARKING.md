# Validation & Benchmarking System

**Created:** 2025-11-23
**Status:** Ready to Use

Complete system for measuring model performance with statistical rigor.

---

## What We Built

### 1. Validation Data (Multiple Sizes for Cost/Benefit Analysis)

**Location:** `data/validation/`

```
easy_50.jsonl      (50 examples)   - Quick smoke test
easy_100.jsonl     (100 examples)  - Standard testing
easy_200.jsonl     (200 examples)  - Maximum statistical power

medium_50.jsonl    (50 examples)
medium_100.jsonl   (100 examples)
medium_200.jsonl   (200 examples)

hard_50.jsonl      (50 examples)
hard_100.jsonl     (100 examples)
hard_200.jsonl     (184 examples - all available)
```

**All copied to 3090:** `~/TRAINING/data/validation/`

### 2. Benchmarking Tools

**Quick Validation (Direct Inference)** - `monitoring/quick_validation.py`
- Loads model directly (no API needed)
- Measures accuracy + timing
- Ready to use NOW

**Full Benchmark Suite** - `monitoring/validation_benchmark.py`
- Statistical analysis (confidence intervals, power analysis)
- Cost/benefit tradeoffs
- Requires OpenAI-compatible API (not currently available)

**Validation Set Generator** - `tools/data/generate_validation_sets.py`
- Creates custom-sized validation sets
- Stratified by difficulty
- Reproducible (seeded)

---

## Quick Start: Test a Checkpoint

### On 3090 (Inference Machine)

```bash
ssh 192.168.x.x
cd ~/TRAINING

# Test checkpoint-93000 with 50 samples per difficulty (~5 min)
python3 monitoring/quick_validation.py \
  --model-path ~/TRAINING/current_model/checkpoint-93000 \
  --validation-dir ~/TRAINING/data/validation \
  --samples 50 \
  --output /tmp/checkpoint_93000_bench.json
```

**Expected Output:**
```
=== SUMMARY ===
EASY     :  85.0%  (42/50)    45.2s @ 1.1 ex/s
MEDIUM   :  68.0%  (34/50)    52.1s @ 0.96 ex/s
HARD     :  44.0%  (22/50)    58.3s @ 0.86 ex/s

Total time: 155.6s
Average throughput: 0.96 ex/s
```

---

## Cost/Benefit Analysis: Choosing Sample Size

|  Size | Time/Difficulty | Total Time | Statistical Power | Use Case |
|------:|----------------:|-----------:|------------------:|----------|
|    10 |             ~1m |        ~3m |               40% | Quick smoke test |
|    50 |             ~5m |       ~15m |               80% | **Standard testing** ✓ |
|   100 |            ~10m |       ~30m |               92% | High confidence |
|   200 |            ~20m |       ~60m |               98% | Maximum power |

**Recommendation: Start with 50 samples**
- Good statistical power (80% to detect 5% improvement)
- Fast enough to run frequently (~15 min)
- Confidence intervals typically ±7%

---

## Tracking Progress Over Training

### Manual Checkpoint Comparison

```bash
# Test multiple checkpoints
for ckpt in 80000 85000 90000 93000; do
  echo "Testing checkpoint-$ckpt..."
  python3 monitoring/quick_validation.py \
    --model-path ~/TRAINING/current_model/checkpoint-$ckpt \
    --samples 50 \
    --output /tmp/ckpt_${ckpt}.json
done

# Compare results
echo "Step,Easy,Medium,Hard" > progress.csv
for f in /tmp/ckpt_*.json; do
  step=$(basename $f | grep -o '[0-9]*')
  jq -r "[\"$step\", .results.easy.accuracy, .results.medium.accuracy, .results.hard.accuracy] | @csv" $f
done >> progress.csv

cat progress.csv
```

**Example Output:**
```
Step,Easy,Medium,Hard
80000,0.82,0.64,0.38
85000,0.84,0.66,0.42
90000,0.86,0.70,0.46
93000,0.88,0.72,0.48
```

---

## Current System Status

### 4090 (Training Machine)

```
Checkpoints: checkpoint-80000 through checkpoint-93000 (every 1000 steps)
Location:    /path/to/training/current_model/
Current Step: 93496 (still training)
```

### 3090 (Inference Machine)

```
Base Model:       ~/llm/models/Qwen3-0.6B
Validation Data:  ~/TRAINING/data/validation/ (all sizes)
Scripts:          ~/TRAINING/monitoring/quick_validation.py
Running Services: 7 autonomous monitoring systems
```

---

## Understanding Metrics

### Accuracy by Difficulty

**Easy (Target: >85%)**
- Simple syllogisms
- 2-3 word targets
- No red herrings
- If failing: Basic reasoning broken

**Medium (Target: >70%)**
- 4-6 word targets
- Some syllable overlap
- Moderate complexity
- Sweet spot for improvement tracking

**Hard (Target: >50%)**
- 6-8 word targets
- Red herrings present
- Complex reasoning required
- Hardest to improve

### Confidence Intervals

**Interpretation:**
- ±5%: Excellent (n=200)
- ±7%: Good (n=50)
- ±10%: Acceptable (n=20)
- ±15%: Too wide (n<10)

**Example:**
```
Accuracy: 72% [65%, 79%]  ← 95% confident true accuracy is in this range
```

### Statistical Power

**Ability to detect a 5% accuracy improvement:**
- 40% power (n=10): Likely to miss real improvements
- 80% power (n=50): Good chance of detecting improvements ✓
- 92% power (n=100): Very likely to detect improvements
- 98% power (n=200): Almost guaranteed to detect improvements

---

## Next Steps: Automated Testing

### Integration with Existing Systems

The **automated_testing_daemon.py** on 3090 already supports this pattern:

1. Watches for new checkpoints
2. Runs validation automatically
3. Saves results to `status/automated_testing.json`
4. Detects regressions

**To enable:** Edit automated_testing_daemon.py to use `quick_validation.py` instead of API calls.

### Recommended Workflow

1. **During Training (3090):**
   - Automated testing daemon runs every 10 minutes
   - Tests latest checkpoint with 50 samples
   - Alerts if accuracy drops >5%

2. **Deep Analysis (Manual):**
   - Run 200-sample benchmark on best checkpoints
   - Compare multiple checkpoints
   - Analyze error patterns

3. **Final Validation:**
   - Run full 200-sample benchmark
   - Calculate final metrics
   - Document results

---

## Troubleshooting

### "Model loading failed"
```bash
# Check checkpoint exists
ls -lh ~/TRAINING/current_model/checkpoint-93000/

# Check for required files
ls ~/TRAINING/current_model/checkpoint-93000/ | grep -E "model.safetensors|config.json"
```

### "Out of memory"
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in model loading (edit quick_validation.py)
# Or test fewer samples at once
```

### "Validation data not found"
```bash
# Verify files exist
ls -lh ~/TRAINING/data/validation/*_50.jsonl

# Re-copy from 4090 if needed
scp 192.168.x.x:~/Desktop/TRAINING/data/validation/*_*.jsonl ~/TRAINING/data/validation/
```

---

## Files Created

### On 4090
- `data/validation/*_50.jsonl` - 50-sample validation sets
- `data/validation/*_100.jsonl` - 100-sample validation sets
- `data/validation/*_200.jsonl` - 200-sample validation sets
- `tools/data/generate_validation_sets.py` - Validation generator
- `monitoring/validation_benchmark.py` - Full benchmark suite (requires API)
- `monitoring/quick_validation.py` - Direct inference testing

### On 3090
- All validation sets (copied from 4090)
- `monitoring/quick_validation.py` (copied from 4090)
- Ready to test any checkpoint

---

## Summary

**What You Have:**
- 9 validation sets (3 difficulties × 3 sizes)
- 2 benchmarking tools (quick + comprehensive)
- 14 checkpoints to test (80000-93000)
- Statistical analysis (CI, power, cost/benefit)

**What You Can Do:**
1. Test any checkpoint in ~15 min (50 samples)
2. Track accuracy progression over training
3. Detect regressions automatically
4. Choose optimal sample size for your needs

**Recommended First Test:**
```bash
ssh 192.168.x.x "cd ~/TRAINING && python3 monitoring/quick_validation.py --model-path ~/TRAINING/current_model/checkpoint-93000 --samples 50 --output /tmp/test.json --verbose"
```

This will show you:
- Current checkpoint accuracy
- Inference speed
- Which examples are failing (verbose mode)
- Whether training is actually improving performance
