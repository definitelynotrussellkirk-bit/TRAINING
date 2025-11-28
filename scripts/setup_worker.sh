#!/bin/bash
#
# Setup Worker - Deploy a worker to a remote machine
#
# Usage:
#   ./scripts/setup_worker.sh <target_host> <worker_type> [port]
#
# Examples:
#   ./scripts/setup_worker.sh macmini-eval-1.local eval 8900
#   ./scripts/setup_worker.sh 192.168.x.x data_forge 8901
#
# Prerequisites:
#   - SSH access to target host
#   - Python 3.8+ on target host
#   - Network connectivity to inference server (for eval workers)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
WORKER_PORT="${3:-8900}"
REMOTE_BASE="/tmp/training_worker"

# Parse arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <target_host> <worker_type> [port]"
    echo ""
    echo "  target_host: SSH target (e.g., user@hostname or hostname)"
    echo "  worker_type: eval or data_forge"
    echo "  port: Worker port (default: 8900)"
    echo ""
    echo "Examples:"
    echo "  $0 macmini-eval-1.local eval 8900"
    echo "  $0 user@xxx.xxx.88.200 data_forge 8901"
    exit 1
fi

TARGET_HOST="$1"
WORKER_TYPE="$2"

# Validate worker type
if [ "$WORKER_TYPE" != "eval" ] && [ "$WORKER_TYPE" != "data_forge" ]; then
    echo -e "${RED}Error: Invalid worker type '$WORKER_TYPE'${NC}"
    echo "Valid types: eval, data_forge"
    exit 1
fi

echo -e "${GREEN}=== Worker Setup ===${NC}"
echo "Target: $TARGET_HOST"
echo "Type: $WORKER_TYPE"
echo "Port: $WORKER_PORT"
echo ""

# Step 1: Check SSH connectivity
echo -e "${YELLOW}[1/5] Checking SSH connectivity...${NC}"
if ! ssh -o ConnectTimeout=5 "$TARGET_HOST" "echo OK" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to $TARGET_HOST${NC}"
    echo "Make sure SSH is configured and the host is reachable."
    exit 1
fi
echo "  ✓ SSH connection OK"

# Step 2: Check Python
echo -e "${YELLOW}[2/5] Checking Python on target...${NC}"
PYTHON_VERSION=$(ssh "$TARGET_HOST" "python3 --version 2>/dev/null || echo 'NOT FOUND'")
if [[ "$PYTHON_VERSION" == "NOT FOUND" ]]; then
    echo -e "${RED}Error: Python 3 not found on target${NC}"
    exit 1
fi
echo "  ✓ $PYTHON_VERSION"

# Step 3: Create remote directory and copy files
echo -e "${YELLOW}[3/5] Copying worker files...${NC}"
ssh "$TARGET_HOST" "mkdir -p $REMOTE_BASE"

# Copy workers module
rsync -avz --exclude='__pycache__' \
    "$BASE_DIR/workers/" \
    "$TARGET_HOST:$REMOTE_BASE/workers/"

# Copy required guild modules (for eval worker)
if [ "$WORKER_TYPE" == "eval" ]; then
    rsync -avz --exclude='__pycache__' \
        "$BASE_DIR/guild/skills/" \
        "$TARGET_HOST:$REMOTE_BASE/guild/skills/" 2>/dev/null || true
    rsync -avz --exclude='__pycache__' \
        "$BASE_DIR/guild/passives/" \
        "$TARGET_HOST:$REMOTE_BASE/guild/passives/" 2>/dev/null || true
fi

# Copy config files
ssh "$TARGET_HOST" "mkdir -p $REMOTE_BASE/config"
scp "$BASE_DIR/config/devices.json" "$TARGET_HOST:$REMOTE_BASE/config/" 2>/dev/null || true
scp "$BASE_DIR/config/hosts.json" "$TARGET_HOST:$REMOTE_BASE/config/" 2>/dev/null || true

echo "  ✓ Files copied"

# Step 4: Install dependencies
echo -e "${YELLOW}[4/5] Installing dependencies...${NC}"
ssh "$TARGET_HOST" "cd $REMOTE_BASE && pip3 install --user requests" || true
echo "  ✓ Dependencies installed"

# Step 5: Create start script
echo -e "${YELLOW}[5/5] Creating start script...${NC}"

# Get inference URL for eval workers
INFERENCE_URL="http://192.168.x.x:8765"  # Default

# Create wrapper script
cat << EOF | ssh "$TARGET_HOST" "cat > $REMOTE_BASE/start_worker.sh"
#!/bin/bash
# Auto-generated worker start script

cd $REMOTE_BASE
export PYTHONPATH="$REMOTE_BASE:\$PYTHONPATH"
export TRAINING_DEVICE_ID="${TARGET_HOST%%.*}"
export INFERENCE_URL="$INFERENCE_URL"

echo "Starting $WORKER_TYPE worker on port $WORKER_PORT..."
python3 -m workers.${WORKER_TYPE}_worker --port $WORKER_PORT
EOF

ssh "$TARGET_HOST" "chmod +x $REMOTE_BASE/start_worker.sh"
echo "  ✓ Start script created"

# Done
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "To start the worker on $TARGET_HOST:"
echo "  ssh $TARGET_HOST '$REMOTE_BASE/start_worker.sh'"
echo ""
echo "Or run in background:"
echo "  ssh $TARGET_HOST 'nohup $REMOTE_BASE/start_worker.sh > /tmp/worker.log 2>&1 &'"
echo ""
echo "To test the worker:"
echo "  curl http://${TARGET_HOST%%@*}:$WORKER_PORT/health"
