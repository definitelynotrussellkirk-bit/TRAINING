#!/bin/bash
# Ultimate Trainer Health Check Script
# Checks all systems and reports status

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         ULTIMATE TRAINER HEALTH CHECK                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0
WARNINGS=0

# Function to check if a process is running
check_process() {
    local name=$1
    local pattern=$2

    if ps aux | grep -E "$pattern" | grep python3 | grep -v grep > /dev/null; then
        echo "✅ $name is running"
        return 0
    else
        echo "❌ $name is NOT running"
        ((ERRORS++))
        return 1
    fi
}

# Function to check HTTP endpoint
check_endpoint() {
    local name=$1
    local url=$2

    if curl -s "$url" > /dev/null 2>&1; then
        echo "✅ $name responding"
        return 0
    else
        echo "❌ $name not responding"
        ((ERRORS++))
        return 1
    fi
}

echo "📋 Checking Processes..."
echo "────────────────────────────────────────────────────────────"

# Check training daemon
check_process "Training daemon" "training_daemon.py"

# Check monitors
check_process "Live monitor (port 8080)" "launch_live_monitor.py"
check_process "Memory API (port 8081)" "memory_stats_api.py"

# Enhanced monitor is optional
if check_process "Enhanced monitor (port 8082)" "enhanced_monitor.py"; then
    : # Already printed success
else
    ((ERRORS--))  # Don't count this as an error
    ((WARNINGS++))  # Count as warning instead
    echo "   ⚠️  Enhanced monitor optional, not critical"
fi

echo ""
echo "🌐 Checking API Endpoints..."
echo "────────────────────────────────────────────────────────────"

# Check endpoints
check_endpoint "Status JSON" "http://localhost:8080/status/training_status.json"
check_endpoint "GPU stats" "http://localhost:8080/api/gpu_stats"
check_endpoint "Memory stats (port 8081)" "http://localhost:8081/api/memory_stats"
check_endpoint "Config API" "http://localhost:8080/api/config"

echo ""
echo "📊 Training Status..."
echo "────────────────────────────────────────────────────────────"

if [ -f "status/training_status.json" ]; then
    STEP=$(cat status/training_status.json | jq -r '.current_step // 0')
    TOTAL=$(cat status/training_status.json | jq -r '.total_steps // 0')
    STATUS=$(cat status/training_status.json | jq -r '.status // "unknown"')
    EVALS=$(cat status/training_status.json | jq -r '.total_evals // 0')

    echo "Status: $STATUS"
    echo "Step: $STEP / $TOTAL"
    echo "Evaluations: $EVALS"

    if [ "$EVALS" -eq "0" ]; then
        echo "⚠️  No evaluations yet (wait for step 10, 20, 30...)"
        ((WARNINGS++))
    fi
else
    echo "❌ status/training_status.json not found"
    ((ERRORS++))
fi

echo ""
echo "💾 Data Sources..."
echo "────────────────────────────────────────────────────────────"

if [ -d "inbox" ]; then
    INBOX_COUNT=$(find inbox -maxdepth 1 -name "*.jsonl" -type f | wc -l)
    echo "Inbox files: $INBOX_COUNT .jsonl files"

    if [ "$INBOX_COUNT" -eq "0" ] && [ "$STATUS" != "training" ]; then
        echo "⚠️  No training data in inbox"
        ((WARNINGS++))
    fi
else
    echo "❌ inbox/ directory not found"
    ((ERRORS++))
fi

echo ""
echo "🎯 GPU Status..."
echo "────────────────────────────────────────────────────────────"

if command -v nvidia-smi &> /dev/null; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

    echo "GPU Utilization: ${GPU_UTIL}%"
    echo "GPU Memory: ${GPU_MEM} MB"
    echo "GPU Temperature: ${GPU_TEMP}°C"

    if [ "$GPU_UTIL" -eq "0" ] && [ "$STATUS" == "training" ]; then
        echo "⚠️  Training status is 'training' but GPU utilization is 0%"
        ((WARNINGS++))
    fi
else
    echo "⚠️  nvidia-smi not available"
    ((WARNINGS++))
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Summary
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ ALL SYSTEMS OPERATIONAL"
    echo ""
    echo "Monitor URLs:"
    echo "  • Live Monitor: http://localhost:8080/live_monitor_ui.html"
    echo "  • Test Page: http://localhost:8080/test_display.html"
    echo "  • Enhanced Monitor: http://localhost:8082"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  SYSTEM OPERATIONAL WITH $WARNINGS WARNING(S)"
    echo ""
    echo "Monitor URLs:"
    echo "  • Live Monitor: http://localhost:8080/live_monitor_ui.html"
    echo "  • Test Page: http://localhost:8080/test_display.html"
    exit 0
else
    echo "❌ SYSTEM HAS $ERRORS ERROR(S) AND $WARNINGS WARNING(S)"
    echo ""
    echo "🔧 Quick Fix Commands:"
    echo ""

    # Suggest fixes based on what's broken
    if ! ps aux | grep "training_daemon.py" | grep -v grep > /dev/null; then
        echo "Start training daemon:"
        echo "  ./scripts/start_all.sh"
        echo ""
    fi

    if ! ps aux | grep "launch_live_monitor.py" | grep -v grep > /dev/null; then
        echo "Start live monitor:"
        echo "  nohup python3 launch_live_monitor.py > /dev/null 2>&1 &"
        echo ""
    fi

    if ! ps aux | grep "memory_stats_api.py" | grep -v grep > /dev/null; then
        echo "Start memory API:"
        echo "  nohup python3 memory_stats_api.py > /dev/null 2>&1 &"
        echo ""
    fi

    exit 1
fi
