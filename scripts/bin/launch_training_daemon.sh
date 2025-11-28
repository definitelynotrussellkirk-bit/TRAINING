#!/usr/bin/env bash
# Safe launcher to ensure only one training daemon runs at a time.
set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DAEMON_CMD="python3 -u training_daemon.py --base-dir $BASE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if pgrep -f "python3 -u training_daemon.py" >/dev/null; then
  echo "❌ training_daemon.py is already running. Stop it first before launching a new one."
  exit 1
fi

cd "$BASE_DIR"
echo "🚀 Starting training daemon (guarded launch)..."
nohup $DAEMON_CMD > training_output.log 2>&1 &
echo "✅ training_daemon.py started with PID $!."
