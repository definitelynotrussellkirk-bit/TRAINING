#!/bin/bash
# Restart Web UI to pick up new changes

echo "Stopping existing web UI..."
pkill -f "training_web_ui.py"
sleep 2

echo "Starting web UI..."
cd /path/to/training
nohup /home/user/ultimate_trainer/web_ui_venv/bin/python3 training_web_ui.py > /tmp/web_ui.log 2>&1 &

sleep 3

if pgrep -f "training_web_ui.py" > /dev/null; then
    echo "✓ Web UI restarted successfully"
    echo "  Access at: http://localhost:7860"
    echo "  Logs in: /tmp/web_ui.log"
else
    echo "✗ Failed to start web UI"
    echo "Check /tmp/web_ui.log for errors"
fi
