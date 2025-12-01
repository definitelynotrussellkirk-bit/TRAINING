#!/bin/bash
# Start Fleet Controller - Central management daemon
#
# The controller runs on the control plane (4090) and:
# - Polls all node agents for health
# - Triggers retention when thresholds exceeded
# - Stores health history in SQLite
#
# Usage:
#   ./scripts/start_fleet_controller.sh           # Start controller
#   ./scripts/start_fleet_controller.sh stop      # Stop controller
#   ./scripts/start_fleet_controller.sh status    # Show status

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

PID_FILE="$BASE_DIR/.pids/fleet_controller.pid"
LOG_FILE="$BASE_DIR/logs/fleet_controller.log"

mkdir -p "$BASE_DIR/.pids" "$BASE_DIR/logs"

ACTION="${1:-start}"

stop_controller() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping Fleet Controller (PID: $PID)..."
            kill "$PID"
            sleep 1
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
            fi
            rm -f "$PID_FILE"
            echo "Stopped"
        else
            echo "Controller not running (stale PID)"
            rm -f "$PID_FILE"
        fi
    else
        echo "Controller not running"
    fi
}

start_controller() {
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Controller already running (PID: $PID)"
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    echo "Starting Fleet Controller..."

    cd "$BASE_DIR"
    nohup python3 -m fleet.controller --daemon --interval 60 >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"

    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Fleet Controller started (PID: $PID)"
        echo "Log: $LOG_FILE"
    else
        echo "Failed to start. Check log: $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi
}

show_status() {
    echo "Fleet Controller Status"
    echo "======================="

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Status: Running (PID: $PID)"

            # Show fleet status
            echo ""
            python3 -m fleet.controller --status 2>/dev/null || echo "(status unavailable)"
        else
            echo "Status: Not running (stale PID)"
        fi
    else
        echo "Status: Not running"
    fi

    echo ""
    echo "Recent log:"
    if [ -f "$LOG_FILE" ]; then
        tail -10 "$LOG_FILE"
    else
        echo "(no log)"
    fi
}

case "$ACTION" in
    start)
        start_controller
        ;;
    stop)
        stop_controller
        ;;
    restart)
        stop_controller
        sleep 1
        start_controller
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status]"
        exit 1
        ;;
esac
