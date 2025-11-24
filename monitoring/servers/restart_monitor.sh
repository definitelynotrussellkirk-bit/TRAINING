#!/bin/bash
# Restart monitoring server

cd /path/to/training

# Kill any existing monitor servers
pkill -9 -f launch_live_monitor.py
sleep 3

# Wait for port to be free
for i in {1..10}; do
    if ! lsof -i:8080 >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Start new server
nohup python3 monitoring/servers/launch_live_monitor.py > /tmp/monitor_server.log 2>&1 &

sleep 2

# Check if it started
if curl -s http://localhost:8080/api/status/live > /dev/null 2>&1; then
    echo "✓ Monitor server started successfully on port 8080"
    echo "  http://localhost:8080/monitoring/ui/control_room_v2.html"
else
    echo "✗ Monitor server failed to start"
    echo "  Check /tmp/monitor_server.log for errors"
    tail -20 /tmp/monitor_server.log
fi
