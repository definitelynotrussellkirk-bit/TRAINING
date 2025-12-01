#!/bin/bash
# Start Fleet Agent - Run on each node to enable remote monitoring/management
#
# Usage:
#   ./scripts/start_fleet_agent.sh           # Auto-detect host ID
#   ./scripts/start_fleet_agent.sh --host-id 3090  # Explicit host ID
#   ./scripts/start_fleet_agent.sh stop      # Stop agent
#   ./scripts/start_fleet_agent.sh status    # Check status

set -e

# Detect script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
PID_FILE="$BASE_DIR/.pids/fleet_agent.pid"
LOG_FILE="$BASE_DIR/logs/fleet_agent.log"
PORT=8769

# Ensure directories exist
mkdir -p "$BASE_DIR/.pids" "$BASE_DIR/logs"

# Parse arguments
ACTION="start"
HOST_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        stop)
            ACTION="stop"
            shift
            ;;
        status)
            ACTION="status"
            shift
            ;;
        restart)
            ACTION="restart"
            shift
            ;;
        --host-id)
            HOST_ID="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [start|stop|status|restart] [--host-id ID] [--port PORT]"
            exit 1
            ;;
    esac
done

stop_agent() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping Fleet Agent (PID: $PID)..."
            kill "$PID"
            sleep 1
            if kill -0 "$PID" 2>/dev/null; then
                echo "Force killing..."
                kill -9 "$PID"
            fi
            rm -f "$PID_FILE"
            echo "Stopped"
        else
            echo "Agent not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        # Check if running anyway
        if pgrep -f "fleet.agent" > /dev/null; then
            echo "Killing orphan agent processes..."
            pkill -f "fleet.agent"
        else
            echo "Agent not running"
        fi
    fi
}

start_agent() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Agent already running (PID: $PID)"
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    echo "Starting Fleet Agent..."

    # Build command
    CMD="python3 -m fleet.agent --port $PORT"
    if [ -n "$HOST_ID" ]; then
        CMD="$CMD --host-id $HOST_ID"
    fi

    # Start in background
    cd "$BASE_DIR"
    nohup $CMD >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"

    # Wait and verify
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Fleet Agent started (PID: $PID)"
        echo "Log: $LOG_FILE"
        echo "API: http://localhost:$PORT/api/status"
    else
        echo "Failed to start agent. Check log: $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi
}

show_status() {
    echo "Fleet Agent Status"
    echo "=================="

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Status: Running (PID: $PID)"

            # Try to get health
            HEALTH=$(curl -s "http://localhost:$PORT/health" 2>/dev/null || echo '{"error": "unreachable"}')
            echo "Health: $HEALTH"

            # Show uptime
            if [ -f "/proc/$PID/stat" ]; then
                START_TIME=$(stat -c %Y "/proc/$PID")
                NOW=$(date +%s)
                UPTIME=$((NOW - START_TIME))
                echo "Uptime: ${UPTIME}s"
            fi
        else
            echo "Status: Not running (stale PID file)"
        fi
    else
        if pgrep -f "fleet.agent" > /dev/null; then
            echo "Status: Running (orphan process)"
            pgrep -f "fleet.agent"
        else
            echo "Status: Not running"
        fi
    fi

    echo ""
    echo "Log tail:"
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE"
    else
        echo "(no log file)"
    fi
}

case $ACTION in
    start)
        start_agent
        ;;
    stop)
        stop_agent
        ;;
    restart)
        stop_agent
        sleep 1
        start_agent
        ;;
    status)
        show_status
        ;;
    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac
