#!/bin/bash
# Launch Training Control Center Web UI

cd "$(dirname "$0")"

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

# Use ultimate_trainer venv which has gradio installed
/home/user/ultimate_trainer/web_ui_venv/bin/python3 training_web_ui.py
