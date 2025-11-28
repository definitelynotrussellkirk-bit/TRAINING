#!/bin/bash
#
# Start 3090 Robustness System
# Launches all monitoring and self-healing components
#

set -e

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$TRAINING_DIR"

# Create necessary directories
mkdir -p logs/3090_watchdog
mkdir -p logs/3090_health
mkdir -p logs/memory_guardian
mkdir -p status

echo "================================================================================"
echo "🚀 STARTING 3090 ROBUSTNESS SYSTEM"
echo "================================================================================"
echo ""

# Check if already running
if pgrep -f "3090_watchdog.py" > /dev/null; then
    echo "⚠️  Warning: 3090 watchdog already running"
    echo "   Kill with: pkill -f 3090_watchdog.py"
    echo ""
fi

if pgrep -f "3090_health_dashboard.py" > /dev/null; then
    echo "⚠️  Warning: 3090 health dashboard already running"
    echo "   Kill with: pkill -f 3090_health_dashboard.py"
    echo ""
fi

if pgrep -f "3090_memory_guardian.py" > /dev/null; then
    echo "⚠️  Warning: Memory guardian already running"
    echo "   Kill with: pkill -f 3090_memory_guardian.py"
    echo ""
fi

# 1. Start Process Watchdog
echo "1️⃣  Starting Process Watchdog..."
nohup python3 monitoring/3090_watchdog.py \
    > logs/3090_watchdog/watchdog.log 2>&1 &
WATCHDOG_PID=$!
echo "   ✅ Watchdog started (PID: $WATCHDOG_PID)"
echo "   Log: logs/3090_watchdog/watchdog.log"
echo ""

# Wait a moment
sleep 2

# 2. Start Health Dashboard
echo "2️⃣  Starting Health Dashboard..."
nohup python3 monitoring/3090_health_dashboard.py \
    > logs/3090_health/dashboard.log 2>&1 &
HEALTH_PID=$!
echo "   ✅ Health dashboard started (PID: $HEALTH_PID)"
echo "   Log: logs/3090_health/dashboard.log"
echo ""

# Wait a moment
sleep 2

# 3. Start Memory Guardian
echo "3️⃣  Starting Memory Guardian..."
nohup python3 monitoring/3090_memory_guardian.py \
    > logs/memory_guardian/guardian.log 2>&1 &
MEMORY_PID=$!
echo "   ✅ Memory guardian started (PID: $MEMORY_PID)"
echo "   Log: logs/memory_guardian/guardian.log"
echo ""

# 4. Start Automated Testing Daemon (if exists)
if [ -f "monitoring/automated_testing_daemon.py" ]; then
    echo "4️⃣  Starting Automated Testing Daemon..."
    nohup python3 monitoring/automated_testing_daemon.py \
        --base-dir "$TRAINING_DIR" \
        --gpu-target 0.25 \
        > logs/testing_daemon.log 2>&1 &
    TESTING_PID=$!
    echo "   ✅ Testing daemon started (PID: $TESTING_PID)"
    echo "   Log: logs/testing_daemon.log"
    echo ""
fi

echo "================================================================================"
echo "✅ ROBUSTNESS SYSTEM STARTED"
echo "================================================================================"
echo ""
echo "📊 Components Running:"
echo "   - Process Watchdog (auto-restart on crash)"
echo "   - Health Dashboard (metrics & analytics)"
echo "   - Memory Guardian (leak detection & cleanup)"
if [ -f "monitoring/automated_testing_daemon.py" ]; then
    echo "   - Automated Testing (continuous validation)"
fi
echo ""
echo "📁 Log Files:"
echo "   - Watchdog: logs/3090_watchdog/watchdog.log"
echo "   - Health: logs/3090_health/dashboard.log"
echo "   - Memory: logs/memory_guardian/guardian.log"
if [ -f "monitoring/automated_testing_daemon.py" ]; then
    echo "   - Testing: logs/testing_daemon.log"
fi
echo ""
echo "📈 Status Files:"
echo "   - Watchdog: status/3090_watchdog_status.json"
echo "   - Health: logs/3090_health/health_<date>.json"
echo "   - Memory: logs/memory_guardian/memory_report_<date>.json"
echo ""
echo "🛑 To stop all components:"
echo "   pkill -f '3090_watchdog.py|3090_health_dashboard.py|3090_memory_guardian.py|automated_testing_daemon.py'"
echo ""
echo "📊 To view logs:"
echo "   tail -f logs/3090_watchdog/watchdog.log"
echo "   tail -f logs/3090_health/dashboard.log"
echo "   tail -f logs/memory_guardian/guardian.log"
echo ""
echo "================================================================================"
