#!/bin/bash
# Cron job for discrimination training data generation
# Runs every hour in RATIO mode
# - Scans queue for SYLLO data
# - Generates discrimination at 20% ratio (target: 10k per 50k SYLLO)
# - 2000 examples/run @ ~20/min = ~1.5 hours max runtime
# - 24,000/day throughput keeps up with ~100k SYLLO/day

cd /path/to/training
export INFERENCE_ADMIN_KEY=admin123

# Log file
LOG=/path/to/training/logs/discrimination_generator.log

echo "$(date): Starting discrimination generator (ratio mode)" >> $LOG

# Run in RATIO mode - automatically calculates deficit and generates
# ~27 examples/min @ 3.5s each with trained model
python3 monitoring/discrimination_generator.py \
    --ratio \
    --max-per-run 1500 \
    --difficulty auto \
    --priority high \
    >> $LOG 2>&1

echo "$(date): Completed" >> $LOG
