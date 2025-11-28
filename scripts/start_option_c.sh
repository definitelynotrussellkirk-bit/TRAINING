#!/bin/bash
# Start all Option C services on 4090 (training machine)

set -e

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$BASE_DIR"

echo "=================================================="
echo "Starting Option C Services (4090)"
echo "=================================================="
echo ""

# Create directories
mkdir -p logs .pids

echo "1. Checking training daemon..."
if ps aux | grep "[t]raining_daemon.py" > /dev/null; then
    echo "   ✅ Training daemon already running"
else
    echo "   Starting training daemon..."
    nohup python3 core/training_daemon.py --base-dir "$BASE_DIR" \
        > logs/training_output.log 2>&1 &
    echo $! > .pids/training_daemon.pid
    echo "   ✅ Started (PID: $(cat .pids/training_daemon.pid))"
fi

echo ""
echo "2. Starting model comparison engine..."
if ps aux | grep "[m]odel_comparison_engine.py" > /dev/null; then
    echo "   ⚠️  Already running, stopping first..."
    pkill -f model_comparison_engine.py
    sleep 2
fi

nohup python3 monitoring/model_comparison_engine.py \
    --base-dir "$BASE_DIR" \
    --interval 600 \
    > logs/model_comparison.log 2>&1 &
echo $! > .pids/model_comparison.pid
echo "   ✅ Started (PID: $(cat .pids/model_comparison.pid))"

echo ""
echo "3. Starting deployment orchestrator..."
if ps aux | grep "[d]eployment_orchestrator.py" > /dev/null; then
    echo "   ⚠️  Already running, stopping first..."
    pkill -f deployment_orchestrator.py
    sleep 2
fi

nohup python3 monitoring/deployment_orchestrator.py \
    --base-dir "$BASE_DIR" \
    --interval 600 \
    > logs/deployment_orchestrator.log 2>&1 &
echo $! > .pids/deployment_orchestrator.pid
echo "   ✅ Started (PID: $(cat .pids/deployment_orchestrator.pid))"

echo ""
echo "4. Checking disk manager..."
if ps aux | grep "[a]uto_disk_manager.py" > /dev/null; then
    echo "   ✅ Disk manager already running"
else
    echo "   Starting disk manager..."
    nohup python3 management/auto_disk_manager.py \
        > logs/disk_manager.log 2>&1 &
    echo "   ✅ Started"
fi

echo ""
echo "=================================================="
echo "✅ All services started"
echo "=================================================="
echo ""

# Show status
echo "Running processes:"
ps aux | grep -E 'training_daemon|model_comparison|deployment_orchestrator|disk_manager' | grep python | grep -v grep

echo ""
echo "Logs:"
echo "  tail -f logs/training_output.log"
echo "  tail -f logs/model_comparison.log"
echo "  tail -f logs/deployment_orchestrator.log"

echo ""
echo "Status files:"
echo "  cat status/training_status.json | jq ."
echo "  cat status/model_comparisons.json | jq ."
echo "  cat status/deployment_status.json | jq ."

echo ""
echo "3090 Status:"
INFERENCE_HOST="${INFERENCE_HOST:-$(python3 -c 'from core.hosts import get_host; print(get_host("3090").host)' 2>/dev/null || echo "192.168.x.x")}"
echo "  curl http://$INFERENCE_HOST:8765/health | jq ."
echo "  curl http://$INFERENCE_HOST:8765/models/info | jq ."
