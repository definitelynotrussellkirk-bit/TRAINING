#!/bin/bash
# Automated Maintenance Script
# Run this periodically to keep system clean

cd "$(dirname "$0")"

echo "🔧 System Maintenance"
echo "================================"
echo "Date: $(date)"
echo ""

# 1. Clean old checkpoints (graduated retention: last 20 + every 1000th)
echo "1. Checkpoint Cleanup"
echo "--------------------"
CHECKPOINTS=$(ls -1 current_model/ | grep -c "checkpoint-" 2>/dev/null || echo 0)
if [ $CHECKPOINTS -gt 30 ]; then
    echo "⚠️  Found $CHECKPOINTS checkpoints"
    echo "Policy: Keep last 20 + every 1000th older checkpoint"
    echo "Run: ./cleanup_checkpoints.sh"
else
    echo "✓ Checkpoints: $CHECKPOINTS (OK)"
fi
echo ""

# 2. Archive old logs (> 7 days)
echo "2. Log Rotation"
echo "---------------"
OLD_LOGS=$(find logs/ -name "daemon_*.log" -mtime +7 2>/dev/null | wc -l)
if [ $OLD_LOGS -gt 0 ]; then
    echo "Found $OLD_LOGS logs > 7 days old"
    echo "Archiving..."
    mkdir -p logs/archive
    find logs/ -name "daemon_*.log" -mtime +7 -exec mv {} logs/archive/ \;
    echo "✓ Archived $OLD_LOGS old logs"
else
    echo "✓ No old logs to archive"
fi
echo ""

# 3. Check disk space
echo "3. Disk Space"
echo "-------------"
echo "current_model/: $(du -sh current_model/ 2>/dev/null | cut -f1)"
echo "logs/: $(du -sh logs/ 2>/dev/null | cut -f1)"
echo "snapshots/: $(du -sh snapshots/ 2>/dev/null | cut -f1)"
echo ""

# 4. Clean temporary files
echo "4. Temporary Files"
echo "------------------"
TEMP_FILES=$(find . -maxdepth 1 -type f \( -name "*.tmp" -o -name "*.cache" -o -name "nohup.out" \) 2>/dev/null)
if [ -n "$TEMP_FILES" ]; then
    echo "Found temporary files:"
    echo "$TEMP_FILES"
    echo "$TEMP_FILES" | xargs rm -f
    echo "✓ Cleaned"
else
    echo "✓ No temporary files"
fi
echo ""

# 5. Check running services
echo "5. Services Status"
echo "------------------"
ps aux | grep -E "(training_daemon|launch_live_monitor|memory_stats_api|enhanced_monitor)" | grep -v grep | awk '{print "✓", $11, "(PID", $2")"}' || echo "⚠️  No services running"
echo ""

# 6. Check memory usage
echo "6. Memory Status"
echo "----------------"
if command -v free &> /dev/null; then
    RAM_PERCENT=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
    RAM_GB=$(free -h | grep Mem | awk '{print $3 "/" $2}')
    echo "RAM: $RAM_GB ($RAM_PERCENT%)"
    if [ $RAM_PERCENT -gt 85 ]; then
        echo "⚠️  High memory usage!"
    else
        echo "✓ Memory OK"
    fi
else
    echo "⚠️  'free' command not available"
fi
echo ""

echo "================================"
echo "Maintenance complete!"
echo ""
echo "Recommendations:"
if [ $CHECKPOINTS -gt 30 ]; then
    echo "  - Run ./cleanup_checkpoints.sh to free disk space"
fi
echo "  - Check logs/archive/ periodically and delete very old archives"
echo "  - Run this script weekly: ./maintenance.sh"
