#!/bin/bash
#
# Test Self-Correction Loop with sample data
# Creates test examples and runs through the pipeline
#

set -e

TRAINING_DIR="/path/to/training"
cd "$TRAINING_DIR"

echo "================================================================================"
echo "🧪 TESTING SELF-CORRECTION LOOP"
echo "================================================================================"
echo ""

# Create test data
echo "1️⃣  Creating test data..."
cat > /tmp/test_corrections.jsonl << 'TESTEOF'
{"input": "All cats are mammals. Fluffy is a cat. Therefore, Fluffy is a mammal.", "output": "valid"}
{"input": "All cats are mammals. Fluffy is not a cat. Therefore, Fluffy is not a mammal.", "output": "invalid"}
{"input": "If it rains, the ground is wet. The ground is wet. Therefore, it rained.", "output": "invalid"}
{"input": "All dogs bark. Rex is a dog. Therefore, Rex barks.", "output": "valid"}
{"input": "Some birds can fly. Penguins are birds. Therefore, penguins can fly.", "output": "invalid"}
TESTEOF

echo "   ✅ Created 5 test examples"
echo ""

# Run test
echo "2️⃣  Running self-correction pipeline..."
python3 monitoring/self_correction_loop.py \
    --file /tmp/test_corrections.jsonl \
    2>&1 | head -100

echo ""
echo "================================================================================"
echo "✅ TEST COMPLETE"
echo "================================================================================"
echo ""
echo "Check outputs:"
echo "   - Corrections: queue/corrections/"
echo "   - Error patterns: logs/error_patterns/"
echo "   - Validated: queue/normal/"
echo ""
echo "================================================================================"
