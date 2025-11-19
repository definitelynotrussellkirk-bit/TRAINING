#!/usr/bin/env bash
# Toggle config.json between presets (8B or 0.6B) and set current model dir.
set -euo pipefail
BASE_DIR="/path/to/training"

usage() {
  echo "Usage: $0 {8b|0.6b|0.6b-short|0.6b-med|0.6b-long}"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

case "$1" in
  8b)
    SRC="$BASE_DIR/config_qwen3_8b.json"
    ;;
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
