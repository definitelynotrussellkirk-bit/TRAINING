#!/usr/bin/env bash
# Toggle config.json between 0.6B presets and set current model dir.
set -euo pipefail

# Auto-detect base directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${TRAINING_BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

usage() {
  echo "Usage: $0 {0.6b|0.6b-short|0.6b-med|0.6b-long}"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

case "$1" in
  0.6b)
    SRC="$BASE_DIR/config_qwen3_06b.json"
    ;;
  0.6b-short)
    SRC="$BASE_DIR/config_qwen3_06b_short.json"
    ;;
  0.6b-med)
    SRC="$BASE_DIR/config_qwen3_06b_med.json"
    ;;
  0.6b-long)
    SRC="$BASE_DIR/config_qwen3_06b_long.json"
    ;;
  *)
    usage
    ;;
esac

if [[ ! -f "$SRC" ]]; then
  echo "Preset config not found: $SRC"
  exit 1
fi

cp "$SRC" "$BASE_DIR/config.json"
echo "Switched config.json to preset: $1"
echo "Restart daemon to apply: python3 training_daemon.py --base-dir $BASE_DIR"
