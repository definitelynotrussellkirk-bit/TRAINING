#!/bin/bash
# Start Unified Monitoring API Server
# Phase 3: Standalone API on port 8081

cd /path/to/training

echo "Starting Unified Monitoring API Server on port 8081..."
nohup python3 monitoring/api/server.py > logs/unified_api.log 2>&1 &
PID=$!

sleep 2

# Check if server started
if ps -p $PID > /dev/null; then
    echo "✓ Server started successfully (PID: $PID)"
    echo "  Logs: logs/unified_api.log"
    echo "  API: http://localhost:8081"
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:8081 | jq"
    echo "  curl http://localhost:8081/api/unified | jq"
    echo "  curl http://localhost:8081/api/health | jq"
else
    echo "✗ Server failed to start"
    echo "Check logs/unified_api.log for details"
    exit 1
fi
