#!/bin/bash
# Memory monitor - warns if training process uses too much RAM

THRESHOLD=40  # Alert if using more than 40GB

while true; do
    # Get memory usage of python training processes
    MEM_USAGE=$(ps aux | grep "python3.*train" | grep -v grep | awk '{sum+=$6} END {print sum/1024/1024}')

    if [ ! -z "$MEM_USAGE" ]; then
        MEM_GB=$(echo "$MEM_USAGE" | awk '{printf "%.1f", $1}')

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training memory: ${MEM_GB} GB"

        # Alert if over threshold
        if (( $(echo "$MEM_GB > $THRESHOLD" | bc -l) )); then
            echo "⚠️  WARNING: Memory usage is high (${MEM_GB} GB > ${THRESHOLD} GB threshold)"
            echo "⚠️  Possible memory leak - consider restarting training"

            # Log to file
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] HIGH MEMORY: ${MEM_GB} GB" >> memory_alerts.log
        fi
    fi

    sleep 60  # Check every minute
done
