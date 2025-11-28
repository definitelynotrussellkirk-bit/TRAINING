#!/bin/bash
#
# Quick 3090 Status Check
# Shows current state of remote GPU server and robustness components
#

set -e

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Get remote host from config or env var
REMOTE_HOST="${INFERENCE_HOST:-$(python3 -c 'from core.hosts import get_host; print(get_host("3090").host)' 2>/dev/null || echo "inference.local")}"
REMOTE_USER="${INFERENCE_SSH_USER:-$(python3 -c 'from core.hosts import get_host; print(get_host("3090").ssh_user)' 2>/dev/null || echo "$USER")}"
API_PORT="${INFERENCE_PORT:-8765}"

echo "================================================================================"
echo "🖥️  3090 REMOTE GPU SERVER STATUS"
echo "================================================================================"
echo ""
echo "Remote Host: $REMOTE_HOST:$API_PORT"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check SSH connectivity
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 NETWORK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ssh -o ConnectTimeout=5 "$REMOTE_USER@$REMOTE_HOST" "echo 'SSH OK'" &> /dev/null; then
    echo "✅ SSH Connection: OK"
else
    echo "❌ SSH Connection: FAILED"
    exit 1
fi
echo ""

# Check API health
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 API SERVER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if curl -s --max-time 5 "http://$REMOTE_HOST:$API_PORT/health" > /dev/null 2>&1; then
    echo "✅ API Health: OK"

    # Get response time
    RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}\n' "http://$REMOTE_HOST:$API_PORT/health")
    echo "   Response Time: ${RESPONSE_TIME}s"
else
    echo "❌ API Health: FAILED (not responding)"
fi

# Check process
PROCESS_COUNT=$(ssh "$REMOTE_USER@$REMOTE_HOST" "ps aux | grep 'python3 main.py' | grep -v grep | wc -l")
if [ "$PROCESS_COUNT" -eq 1 ]; then
    echo "✅ Process Status: Running (1 process)"
    PID=$(ssh "$REMOTE_USER@$REMOTE_HOST" "pgrep -f 'python3 main.py'")
    echo "   PID: $PID"
elif [ "$PROCESS_COUNT" -eq 0 ]; then
    echo "❌ Process Status: NOT RUNNING"
else
    echo "⚠️  Process Status: Multiple processes ($PROCESS_COUNT)"
fi
echo ""

# Check GPU
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎮 GPU STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
GPU_INFO=$(ssh "$REMOTE_USER@$REMOTE_HOST" "nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")

if [ -n "$GPU_INFO" ]; then
    IFS=', ' read -r GPU_NAME GPU_TEMP VRAM_USED VRAM_TOTAL GPU_UTIL <<< "$GPU_INFO"

    echo "GPU: $GPU_NAME"

    # Temperature
    if (( $(echo "$GPU_TEMP < 80" | bc -l) )); then
        echo "✅ Temperature: ${GPU_TEMP}°C"
    elif (( $(echo "$GPU_TEMP < 85" | bc -l) )); then
        echo "⚠️  Temperature: ${GPU_TEMP}°C (elevated)"
    else
        echo "🔥 Temperature: ${GPU_TEMP}°C (HIGH!)"
    fi

    # VRAM
    VRAM_PCT=$(echo "scale=1; $VRAM_USED / $VRAM_TOTAL * 100" | bc)
    if (( $(echo "$VRAM_PCT < 80" | bc -l) )); then
        echo "✅ VRAM: ${VRAM_USED}MB / ${VRAM_TOTAL}MB (${VRAM_PCT}%)"
    elif (( $(echo "$VRAM_PCT < 90" | bc -l) )); then
        echo "⚠️  VRAM: ${VRAM_USED}MB / ${VRAM_TOTAL}MB (${VRAM_PCT}%)"
    else
        echo "🚨 VRAM: ${VRAM_USED}MB / ${VRAM_TOTAL}MB (${VRAM_PCT}%) - CRITICAL!"
    fi

    # Utilization
    echo "   GPU Utilization: ${GPU_UTIL}%"
else
    echo "❌ GPU: nvidia-smi failed"
fi
echo ""

# Check robustness components
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛡️  ROBUSTNESS COMPONENTS (Local)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Watchdog
if pgrep -f "3090_watchdog.py" > /dev/null; then
    WATCHDOG_PID=$(pgrep -f "3090_watchdog.py")
    echo "✅ Watchdog: Running (PID: $WATCHDOG_PID)"
else
    echo "❌ Watchdog: Not running"
fi

# Health Dashboard
if pgrep -f "3090_health_dashboard.py" > /dev/null; then
    HEALTH_PID=$(pgrep -f "3090_health_dashboard.py")
    echo "✅ Health Dashboard: Running (PID: $HEALTH_PID)"
else
    echo "❌ Health Dashboard: Not running"
fi

# Memory Guardian
if pgrep -f "3090_memory_guardian.py" > /dev/null; then
    MEMORY_PID=$(pgrep -f "3090_memory_guardian.py")
    echo "✅ Memory Guardian: Running (PID: $MEMORY_PID)"
else
    echo "❌ Memory Guardian: Not running"
fi

# Self Healer
if pgrep -f "3090_self_healer.py" > /dev/null; then
    HEALER_PID=$(pgrep -f "3090_self_healer.py")
    echo "✅ Self Healer: Running (PID: $HEALER_PID)"
else
    echo "⚠️  Self Healer: Not running"
fi

# Testing Daemon
if pgrep -f "automated_testing_daemon.py" > /dev/null; then
    TESTING_PID=$(pgrep -f "automated_testing_daemon.py")
    echo "✅ Testing Daemon: Running (PID: $TESTING_PID)"
else
    echo "⚠️  Testing Daemon: Not running"
fi

echo ""

# System uptime
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💻 SYSTEM INFO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
UPTIME=$(ssh "$REMOTE_USER@$REMOTE_HOST" "uptime -p")
echo "Uptime: $UPTIME"

LOAD=$(ssh "$REMOTE_USER@$REMOTE_HOST" "uptime | awk -F'load average:' '{print \$2}'")
echo "Load Average:$LOAD"

echo ""
echo "================================================================================"
echo "📊 QUICK ACTIONS"
echo "================================================================================"
echo ""
echo "Start all robustness components:"
echo "  $TRAINING_DIR/scripts/start_3090_robustness.sh"
echo ""
echo "View logs:"
echo "  tail -f logs/3090_watchdog/watchdog.log"
echo "  tail -f logs/3090_health/dashboard.log"
echo "  tail -f logs/memory_guardian/guardian.log"
echo ""
echo "Manual recovery:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST 'pkill -f main.py && cd ~/llm && source venv/bin/activate && nohup python3 main.py > logs/api_server.log 2>&1 &'"
echo ""
echo "================================================================================"
