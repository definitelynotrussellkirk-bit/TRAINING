#!/bin/bash
# Quick test of the self-correction pipeline

set -e

echo "==================================="
echo "Self-Correction Pipeline Test"
echo "==================================="
echo

# Create test dataset
cat > /tmp/test_qa.jsonl <<EOF
{"prompt": "What is 5 × 7?", "response": "5 × 7 = 35. This is calculated by multiplying 5 by 7."}
{"prompt": "Name the four cardinal directions.", "response": "The four cardinal directions are:\n1. North\n2. South\n3. East\n4. West"}
{"prompt": "What is the boiling point of water?", "response": "The boiling point of water is 100°C (212°F) at standard atmospheric pressure."}
EOF

echo "✓ Created test dataset (3 Q&A pairs)"
echo

# Option 1: Demo mode (no model required)
echo "Running demo mode (no model inference)..."
echo
python3 demo_self_correction.py | head -100
echo
echo "... (output truncated)"
echo

# Option 2: Test error codes
echo "==================================="
echo "Testing Error Code Generation"
echo "==================================="
echo
python3 self_correction_trainer.py --test-error-codes

echo
echo "==================================="
echo "Test Complete!"
echo "==================================="
echo
echo "To use on real data:"
echo "  1. Generate initial answers:"
echo "     python3 generate_initial_answers.py \\"
echo "       --input your_qa.jsonl \\"
echo "       --output initial_answers.jsonl \\"
echo "       --model current_model/"
echo
echo "  2. Generate training data:"
echo "     python3 self_correction_trainer.py \\"
echo "       --input your_qa.jsonl \\"
echo "       --output training_data.jsonl \\"
echo "       --initial-answers initial_answers.jsonl"
echo
echo "  3. Train:"
echo "     cp training_data.jsonl inbox/"
echo

# Cleanup
rm -f /tmp/test_qa.jsonl

echo "See SELF_CORRECTION_GUIDE.md for full documentation."
