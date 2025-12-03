#!/bin/bash
# Memory Watchdog - Kill processes if RAM exceeds limit
# Usage: ./mem_watchdog.sh [limit_gb] [target_pattern]
# Example: ./mem_watchdog.sh 50 python

LIMIT_GB=${1:-50}
TARGET=${2:-python}
LIMIT_KB=$((LIMIT_GB * 1024 * 1024))

echo "Memory watchdog started: kill $TARGET processes if RAM > ${LIMIT_GB}GB"

while true; do
    TOTAL_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    USED_KB=$((TOTAL_KB - AVAIL_KB))
    USED_GB=$((USED_KB / 1024 / 1024))

    if [ $USED_KB -gt $LIMIT_KB ]; then
        echo "$(date): RAM at ${USED_GB}GB exceeds ${LIMIT_GB}GB!"
        # Find biggest matching process
        PID=$(ps aux --sort=-%mem | grep -E "$TARGET" | grep -v grep | head -1 | awk '{print $2}')
        if [ -n "$PID" ] && [ "$PID" != "$$" ]; then
            PNAME=$(ps -p $PID -o comm= 2>/dev/null)
            echo "Killing $PNAME (PID $PID)"
            kill -9 $PID 2>/dev/null
        fi
    fi
    sleep 2
done
