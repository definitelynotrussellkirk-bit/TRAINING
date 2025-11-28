#!/bin/bash
# Launch Training Control Center Web UI

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$BASE_DIR"

# Python interpreter - use env var or default to system python
PYTHON="${WEB_UI_PYTHON:-python3}"

echo "=================================="
echo "Training Control Center"
echo "=================================="
echo ""
echo "Starting web interface..."
echo "Access at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================="
echo ""

$PYTHON training_web_ui.py
