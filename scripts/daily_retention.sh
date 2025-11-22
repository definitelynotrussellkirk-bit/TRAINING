#!/bin/bash
#
# Daily Retention Maintenance
#
# This script runs daily maintenance for the retention management system:
# 1. Creates daily snapshot (if not already created)
# 2. Enforces retention policy (36h + 150GB limits)
# 3. Logs all operations
#
# Usage:
#   ./daily_retention.sh [--dry-run]
#
# Cron example (runs daily at 3 AM):
#   0 3 * * * /path/to/training/scripts/daily_retention.sh >> /path/to/training/logs/retention.log 2>&1

set -e  # Exit on error

# Configuration
BASE_DIR="/path/to/training"
OUTPUT_DIR="${BASE_DIR}/models/current_model"
BASE_MODEL="${BASE_DIR}/models/Qwen3-0.6B"
LOG_DIR="${BASE_DIR}/logs"

# Parse arguments
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY RUN MODE]"
fi

# Create log directory if needed
mkdir -p "${LOG_DIR}"

# Log header
echo "=================================================="
echo "Daily Retention Maintenance - $(date)"
echo "=================================================="

# Change to base directory
cd "${BASE_DIR}"

# Run retention manager
echo ""
echo "Running retention manager..."
python3 management/retention_manager.py \
    --output-dir "${OUTPUT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --snapshot \
    --enforce \
    ${DRY_RUN}

echo ""
echo "Status after maintenance:"
python3 management/retention_manager.py \
    --output-dir "${OUTPUT_DIR}" \
    --status

echo ""
echo "=================================================="
echo "Completed at $(date)"
echo "=================================================="
