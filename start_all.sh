#!/bin/bash
# Start all Ultimate Trainer services

echo "Starting Ultimate Trainer services..."
echo ""

cd /path/to/training

# Start training daemon
echo "▶️  Starting training daemon..."
nohup python3 training_daemon.py --base-dir /path/to/training > training_output.log 2>&1 &
sleep 1

# Start monitors
echo "▶️  Starting live monitor (port 8080)..."
nohup python3 launch_live_monitor.py > /dev/null 2>&1 &
sleep 1

echo "▶️  Starting memory API (port 8081)..."
nohup python3 memory_stats_api.py > /dev/null 2>&1 &
sleep 1

echo "▶️  Starting enhanced monitor (port 8082)..."
nohup python3 enhanced_monitor.py > /dev/null 2>&1 &
sleep 2

echo ""
echo "✅ All services started!"
echo ""
echo "Running health check..."
echo ""

./check_health.sh
