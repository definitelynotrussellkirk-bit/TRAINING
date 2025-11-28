#!/bin/bash
# Quick check for duplicate training processes

echo "==================================="
echo "DUPLICATE PROCESS CHECKER"
echo "==================================="
echo

# Check training daemons
echo "Training daemon processes:"
DAEMON_COUNT=$(ps aux | grep -E "training_daemon\.py" | grep -v grep | wc -l)
ps aux | grep -E "training_daemon\.py" | grep -v grep

if [ "$DAEMON_COUNT" -gt 1 ]; then
    echo
    echo "⚠️  WARNING: $DAEMON_COUNT training daemons running!"
    echo "Run: pkill -f training_daemon.py"
    echo "Then: ./scripts/start_all.sh"
elif [ "$DAEMON_COUNT" -eq 1 ]; then
    echo
    echo "✅ OK: Only 1 training daemon running"
else
    echo
    echo "❌ No training daemon running!"
fi

echo
echo "-----------------------------------"
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "  %d MB / %d MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'

echo
echo "-----------------------------------"
echo "Training processes using GPU:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read line; do
    PID=$(echo $line | cut -d',' -f1)
    MEM=$(echo $line | cut -d',' -f2)
    CMD=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown")
    echo "  PID $PID ($CMD): $MEM"
done

echo
echo "==================================="
