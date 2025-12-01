#!/bin/bash
# Deploy Fleet Agent to a remote node
#
# This script copies the self-contained agent to a remote node and sets it up.
# The agent will run standalone without needing the full TRAINING codebase.
#
# Usage:
#   ./scripts/deploy_fleet_agent.sh 3090          # Deploy to 3090
#   ./scripts/deploy_fleet_agent.sh 3090 --start  # Deploy and start
#   ./scripts/deploy_fleet_agent.sh 3090 --test   # Deploy and test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Node configurations
# CUSTOMIZE THESE for your setup - read from config/hosts.json or set manually
# Example values shown below - replace with your actual hosts

# Load from environment or use defaults (customize these!)
INFERENCE_HOST="${INFERENCE_HOST:-inference.local}"
TRAINER_HOST="${TRAINER_HOST:-trainer.local}"
DEPLOY_USER="${DEPLOY_USER:-$USER}"

declare -A HOSTS=(
    ["3090"]="$INFERENCE_HOST"
    ["4090"]="$TRAINER_HOST"
)

declare -A USERS=(
    ["3090"]="$DEPLOY_USER"
    ["4090"]="$DEPLOY_USER"
)

declare -A REMOTE_DIRS=(
    ["3090"]="$HOME/llm"
    ["4090"]="$HOME/Desktop/TRAINING"
)

declare -A CHECKPOINT_DIRS=(
    ["3090"]="$HOME/llm/models"
    ["4090"]="$HOME/Desktop/TRAINING/models/current_model"
)

declare -A MAX_CHECKPOINTS=(
    ["3090"]="3"
    ["4090"]="5"
)

declare -A MAX_GB=(
    ["3090"]="150"
    ["4090"]="400"
)

# Parse arguments
HOST_ID="$1"
ACTION="${2:-deploy}"

if [ -z "$HOST_ID" ]; then
    echo "Usage: $0 <host_id> [--start|--stop|--test|--status]"
    echo ""
    echo "Available hosts:"
    for h in "${!HOSTS[@]}"; do
        echo "  $h (${HOSTS[$h]})"
    done
    exit 1
fi

if [ -z "${HOSTS[$HOST_ID]}" ]; then
    echo "Error: Unknown host '$HOST_ID'"
    exit 1
fi

HOST="${HOSTS[$HOST_ID]}"
USER="${USERS[$HOST_ID]}"
REMOTE_DIR="${REMOTE_DIRS[$HOST_ID]}"
CKPT_DIR="${CHECKPOINT_DIRS[$HOST_ID]}"
MAX_CKPT="${MAX_CHECKPOINTS[$HOST_ID]}"
MAX_GB_VAL="${MAX_GB[$HOST_ID]}"

TARGET="$USER@$HOST"

echo "=== Fleet Agent Deployment ==="
echo "Host: $HOST_ID ($HOST)"
echo "User: $USER"
echo "Remote dir: $REMOTE_DIR"
echo ""

deploy_agent() {
    echo "Deploying agent to $TARGET..."

    # Create remote directory structure
    ssh "$TARGET" "mkdir -p $REMOTE_DIR/fleet $REMOTE_DIR/logs $REMOTE_DIR/.pids"

    # Copy the agent (self-contained)
    scp "$BASE_DIR/fleet/agent.py" "$TARGET:$REMOTE_DIR/fleet/agent.py"

    # Create local config
    echo "Creating config..."
    ssh "$TARGET" "cat > $REMOTE_DIR/fleet_agent.json << 'EOF'
{
  \"host_id\": \"$HOST_ID\",
  \"device_id\": \"${HOST_ID}_device\",
  \"hostname\": \"$HOST\",
  \"checkpoints_dir\": \"$CKPT_DIR\",
  \"models_dir\": \"$CKPT_DIR\",
  \"retention\": {
    \"max_checkpoints\": $MAX_CKPT,
    \"max_gb\": $MAX_GB_VAL,
    \"keep_strategy\": \"recently_used\",
    \"is_vault\": false,
    \"cleanup_threshold_pct\": 90.0
  }
}
EOF"

    # Create start script
    echo "Creating start script..."
    ssh "$TARGET" "cat > $REMOTE_DIR/start_agent.sh << 'SCRIPT'
#!/bin/bash
cd $REMOTE_DIR
PID_FILE=\".pids/fleet_agent.pid\"
LOG_FILE=\"logs/fleet_agent.log\"

# Stop existing
if [ -f \"\$PID_FILE\" ]; then
    kill \$(cat \"\$PID_FILE\") 2>/dev/null
    rm -f \"\$PID_FILE\"
fi

# Start new
nohup python3 fleet/agent.py --config fleet_agent.json >> \"\$LOG_FILE\" 2>&1 &
echo \$! > \"\$PID_FILE\"
echo \"Started fleet agent with PID: \$(cat \$PID_FILE)\"
SCRIPT
chmod +x $REMOTE_DIR/start_agent.sh"

    # Create stop script
    ssh "$TARGET" "cat > $REMOTE_DIR/stop_agent.sh << 'SCRIPT'
#!/bin/bash
cd $REMOTE_DIR
PID_FILE=\".pids/fleet_agent.pid\"

if [ -f \"\$PID_FILE\" ]; then
    kill \$(cat \"\$PID_FILE\") 2>/dev/null
    rm -f \"\$PID_FILE\"
    echo \"Stopped fleet agent\"
else
    echo \"Agent not running\"
fi
SCRIPT
chmod +x $REMOTE_DIR/stop_agent.sh"

    echo "Deployment complete!"
}

start_agent() {
    echo "Starting agent on $TARGET..."
    ssh "$TARGET" "cd $REMOTE_DIR && ./start_agent.sh"
    sleep 2

    # Verify it's running
    echo ""
    echo "Checking agent status..."
    curl -s "http://$HOST:8769/health" 2>/dev/null || echo "Agent not responding (may still be starting)"
}

stop_agent() {
    echo "Stopping agent on $TARGET..."
    ssh "$TARGET" "cd $REMOTE_DIR && ./stop_agent.sh"
}

test_agent() {
    echo "Testing agent on $HOST:8769..."
    echo ""

    echo "=== Health Check ==="
    curl -s "http://$HOST:8769/health" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"

    echo ""
    echo "=== Full Status ==="
    curl -s "http://$HOST:8769/api/status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Host: {d['host_id']} ({d['device_id']})\")
print(f\"Status: {d['status']}\")
print(f\"Uptime: {d['uptime_seconds']}s\")
print(f\"CPU: {d['cpu_pct']}%\")
print(f\"Memory: {d['memory']['used_pct']}%\")
if d['storage']:
    s = d['storage'][0]
    print(f\"Storage: {s['used_pct']}% ({s['checkpoint_count']} checkpoints)\")
if d['gpus']:
    g = d['gpus'][0]
    print(f\"GPU: {g['name']} - {g['vram_used_pct']}% VRAM, {g['utilization_pct']}% util\")
if d['alerts']:
    print(f\"Alerts: {d['alerts']}\")
"

    echo ""
    echo "=== Checkpoints ==="
    curl -s "http://$HOST:8769/api/checkpoints" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Count: {d['count']}\")
for c in d['checkpoints'][-3:]:
    print(f\"  {c['name']}: {c['size_gb']} GB\")
"
}

show_status() {
    echo "Agent status on $TARGET..."
    ssh "$TARGET" "
        cd $REMOTE_DIR
        if [ -f .pids/fleet_agent.pid ]; then
            PID=\$(cat .pids/fleet_agent.pid)
            if kill -0 \$PID 2>/dev/null; then
                echo \"Running (PID: \$PID)\"
                curl -s http://localhost:8769/health
            else
                echo \"Not running (stale PID)\"
            fi
        else
            echo \"Not running\"
        fi
    "
}

# Execute action
case "$ACTION" in
    --start)
        deploy_agent
        start_agent
        ;;
    --stop)
        stop_agent
        ;;
    --test)
        test_agent
        ;;
    --status)
        show_status
        ;;
    *)
        deploy_agent
        ;;
esac
