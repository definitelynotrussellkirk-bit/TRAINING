#!/bin/bash
# Launch Detailed Training Monitor

echo "🔬 Starting Detailed Training Monitor..."
echo ""
echo "This will show:"
echo "  - Current loss / eval loss"
echo "  - Complete prompt context"
echo "  - Golden vs predicted comparison"
echo "  - Token-by-token analysis"
echo ""
echo "Opening at: http://localhost:8081"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
python3 detailed_monitor.py
