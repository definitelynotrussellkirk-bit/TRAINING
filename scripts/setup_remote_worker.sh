#!/bin/bash
# Setup a remote worker node
# Usage: ./setup_remote_worker.sh <host> <device_id>
# Example: ./setup_remote_worker.sh root@192.168.x.x r730xd

set -e

HOST="${1:-root@192.168.x.x}"
DEVICE_ID="${2:-r730xd}"
TRAINER_IP="${TRAINER_IP:-192.168.x.x}"
VAULT_PORT="${VAULT_PORT:-8767}"

echo "=== Setting up remote worker: $DEVICE_ID on $HOST ==="
echo "Trainer: http://$TRAINER_IP:$VAULT_PORT"

# Get the base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "1. Syncing codebase to remote..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'models/' \
    --exclude 'data/' \
    --exclude 'queue/' \
    --exclude 'inbox/' \
    --exclude 'vault/*.db' \
    --exclude 'status/' \
    --exclude '*.log' \
    "$BASE_DIR/" "$HOST:/opt/TRAINING/"

echo ""
echo "2. Installing dependencies on remote..."
ssh "$HOST" "cd /opt/TRAINING && pip3 install requests --quiet 2>/dev/null || pip install requests --quiet"

echo ""
echo "3. Testing connection to VaultKeeper..."
ssh "$HOST" "curl -s http://$TRAINER_IP:$VAULT_PORT/health | head -c 100"

echo ""
echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the worker on $HOST, run:"
echo ""
echo "  ssh $HOST 'cd /opt/TRAINING && python3 -m workers.claiming_worker --device $DEVICE_ID --server http://$TRAINER_IP:$VAULT_PORT'"
echo ""
echo "Or for background with nohup:"
echo ""
echo "  ssh $HOST 'cd /opt/TRAINING && nohup python3 -m workers.claiming_worker --device $DEVICE_ID --server http://$TRAINER_IP:$VAULT_PORT > /tmp/worker_$DEVICE_ID.log 2>&1 &'"
echo ""
