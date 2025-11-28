#!/usr/bin/env bash
# =============================================================================
# REALM OF TRAINING - Bootstrap Script for New Developers
# =============================================================================
# This script sets up a fresh clone for local development.
#
# Usage:
#   ./scripts/bootstrap_dev.sh
#
# After running:
#   python3 -m training doctor      # Check everything is configured
#   python3 -m training start-all   # Start services
#   open http://localhost:8888      # View Tavern UI
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory (works even if called from elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}REALM OF TRAINING - Bootstrap${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Setting up development environment in:"
echo "  $ROOT_DIR"
echo ""

# -----------------------------------------------------------------------------
# 1. Create .env from .env.example if it doesn't exist
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/5] Environment configuration${NC}"

if [ -f "$ROOT_DIR/.env" ]; then
    echo -e "  ${GREEN}✓${NC} .env already exists"
else
    if [ -f "$ROOT_DIR/.env.example" ]; then
        cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
        echo -e "  ${GREEN}✓${NC} Created .env from .env.example"
        echo -e "  ${YELLOW}!${NC} Edit .env to customize for your setup"
    else
        # Create minimal .env
        cat > "$ROOT_DIR/.env" <<'EOF'
# REALM OF TRAINING - Minimal Environment
TRAINING_ENV=dev
TRAINING_DEVICE_ID=local_gpu_1
TRAINER_HOST=localhost
INFERENCE_HOST=localhost
TAVERN_PORT=8888
VAULTKEEPER_PORT=8767
EOF
        echo -e "  ${GREEN}✓${NC} Created minimal .env"
    fi
fi

# -----------------------------------------------------------------------------
# 2. Create config/devices.json if missing
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/5] Device configuration${NC}"

DEVICES_JSON="$ROOT_DIR/config/devices.json"
if [ -f "$DEVICES_JSON" ]; then
    echo -e "  ${GREEN}✓${NC} config/devices.json exists"
else
    mkdir -p "$ROOT_DIR/config"
    cat > "$DEVICES_JSON" <<'EOF'
{
  "_comment": "Device registry for single-machine development",
  "devices": [
    {
      "device_id": "local_gpu_1",
      "roles": ["trainer", "eval_worker", "analytics"],
      "host": "localhost",
      "gpu_type": "auto-detect",
      "notes": "Single-machine dev GPU"
    }
  ]
}
EOF
    echo -e "  ${GREEN}✓${NC} Created config/devices.json with local_gpu_1"
fi

# -----------------------------------------------------------------------------
# 3. Create config/hosts.json from example if missing
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/5] Host configuration${NC}"

HOSTS_JSON="$ROOT_DIR/config/hosts.json"
HOSTS_EXAMPLE="$ROOT_DIR/config/hosts.example.json"
if [ -f "$HOSTS_JSON" ]; then
    echo -e "  ${GREEN}✓${NC} config/hosts.json exists"
elif [ -f "$HOSTS_EXAMPLE" ]; then
    # Create a localhost-only version for dev
    cat > "$HOSTS_JSON" <<EOF
{
  "version": "3.0",
  "warden_port": 8760,
  "hosts": {
    "local": {
      "name": "Local Development",
      "host": "localhost",
      "role": "trainer",
      "base_dir": "$ROOT_DIR",
      "services": {
        "tavern": {"port": 8888, "health": "/health", "name": "Tavern UI", "critical": true},
        "vault_keeper": {"port": 8767, "health": "/health", "name": "VaultKeeper", "critical": true}
      },
      "models_dir": "$ROOT_DIR/models",
      "checkpoints_dir": "$ROOT_DIR/models/current_model",
      "capabilities": ["training", "vault", "ledger", "tavern", "inference", "eval"]
    }
  },
  "local_host": "local",
  "default_trainer": "local",
  "default_inference": "local"
}
EOF
    echo -e "  ${GREEN}✓${NC} Created config/hosts.json for localhost"
else
    echo -e "  ${YELLOW}!${NC} config/hosts.example.json not found, skipping"
fi

# -----------------------------------------------------------------------------
# 4. Create required directories
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/5] Creating directories${NC}"

DIRS=(
    "models"
    "models/current_model"
    "data"
    "data/validation"
    "logs"
    "status"
    "queue/high"
    "queue/normal"
    "queue/low"
    "queue/processing"
    "queue/failed"
    "queue/recently_completed"
    "inbox"
    "vault/hot"
    "vault/warm"
    ".pids"
    "control"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$ROOT_DIR/$dir" ]; then
        mkdir -p "$ROOT_DIR/$dir"
        echo -e "  ${GREEN}✓${NC} Created $dir/"
    fi
done
echo -e "  ${GREEN}✓${NC} All directories present"

# -----------------------------------------------------------------------------
# 5. Initialize databases (if needed)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/5] Database initialization${NC}"

# Check if vault/catalog.db exists
if [ -f "$ROOT_DIR/vault/catalog.db" ]; then
    echo -e "  ${GREEN}✓${NC} vault/catalog.db exists"
else
    echo -e "  ${YELLOW}!${NC} vault/catalog.db will be created on first VaultKeeper start"
fi

# Check if vault/jobs.db exists
if [ -f "$ROOT_DIR/vault/jobs.db" ]; then
    echo -e "  ${GREEN}✓${NC} vault/jobs.db exists"
else
    echo -e "  ${YELLOW}!${NC} vault/jobs.db will be created on first VaultKeeper start"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Bootstrap complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Run the doctor to verify everything:"
echo -e "     ${YELLOW}python3 -m training doctor${NC}"
echo ""
echo "  2. Start the services:"
echo -e "     ${YELLOW}python3 -m training start-all${NC}"
echo ""
echo "  3. Open the Tavern UI:"
echo -e "     ${YELLOW}open http://localhost:8888${NC}"
echo ""
echo "For multi-machine setups, edit:"
echo "  - .env (network hosts)"
echo "  - config/hosts.json (service endpoints)"
echo "  - config/devices.json (GPU capabilities)"
echo ""
