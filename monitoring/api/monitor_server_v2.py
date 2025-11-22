#!/usr/bin/env python3
"""
Monitoring API Server V2

Clean, focused API endpoints for the new UI.

Endpoints:
- GET /api/status/live          - Core training state (1-2s poll)
- GET /api/status/preview       - Latest preview + stats (2-5s poll)
- GET /api/status/evals         - Eval metrics (10-30s poll)
- GET /api/status/system        - System resources (5-10s poll)
- GET /api/preview/history      - Paginated preview history
- GET /api/throughput/samples   - Throughput vs VRAM data

Legacy endpoints (backward compatibility):
- GET /status/training_status.json

Usage:
    python3 monitor_server_v2.py --port 8080 --base-dir /training
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from view_models import (
    LiveStatusView,
    PreviewStatusView,
    EvalStatusView,
    SystemStatusView,
    create_live_status_from_training_status,
)
from preview_history import get_preview_logger


app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Global config
BASE_DIR = Path("/path/to/training")
STATUS_FILE = BASE_DIR / "status" / "training_status.json"


def load_training_status() -> dict:
    """Load current training status from disk"""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading training status: {e}")
        return {}


@app.route('/api/status/live')
def api_status_live():
    """
    GET /api/status/live

    Core training state + hardware.
    Polled every 1-2 seconds.
    """
    status = load_training_status()
    live_view = create_live_status_from_training_status(status)
    return jsonify(live_view.to_dict())


@app.route('/api/status/preview')
def api_status_preview():
    """
    GET /api/status/preview

    Latest preview result + aggregated stats.
    Polled every 2-5 seconds.
    """
    status = load_training_status()
    preview_logger = get_preview_logger(BASE_DIR)

    # Get latest preview
    recent_previews = preview_logger.get_recent_previews(limit=1)
    latest_preview = recent_previews[0] if recent_previews else None

    # Get aggregate stats
    stats_20 = preview_logger.get_preview_stats(limit=20)
    stats_50 = preview_logger.get_preview_stats(limit=50)
    stats_100 = preview_logger.get_preview_stats(limit=100)

    # Build EM trend (last 20 previews)
    last_20 = preview_logger.get_recent_previews(limit=20)
    em_trend = [1.0 if p.get("exact_match") else 0.0 for p in reversed(last_20)]

    # Get pending preview jobs (from status)
    pending_jobs = status.get("preview_jobs_pending", 0)
    last_job_latency = status.get("last_preview_latency_ms", 0)

    view = PreviewStatusView(
        latest_preview=latest_preview,
        preview_em_last_20=stats_20.get("exact_match_rate", 0.0),
        preview_em_last_50=stats_50.get("exact_match_rate", 0.0),
        preview_em_last_100=stats_100.get("exact_match_rate", 0.0),
        domain_stats=stats_100.get("by_source", {}),  # Using source as proxy for domain
        regime_stats=stats_100.get("by_regime", {}),
        pending_jobs=pending_jobs,
        last_job_latency_ms=last_job_latency,
        em_trend=em_trend,
        pattern_heatmap=status.get("pattern_heatmap"),  # If precomputed
    )

    return jsonify(view.to_dict())


@app.route('/api/status/evals')
def api_status_evals():
    """
    GET /api/status/evals

    Evaluation metrics from fixed eval set.
    Polled every 10-30 seconds.
    """
    status = load_training_status()

    # Extract eval metrics
    fixed_eval = status.get("fixed_eval", {})
    micro_eval = status.get("micro_eval", {})

    # Determine eval trend
    eval_trend = "unknown"
    if "fixed_eval_history" in status and len(status["fixed_eval_history"]) >= 2:
        recent = status["fixed_eval_history"][-2:]
        if recent[1].get("em", 0) > recent[0].get("em", 0):
            eval_trend = "improving"
        elif recent[1].get("em", 0) < recent[0].get("em", 0):
            eval_trend = "declining"
        else:
            eval_trend = "stable"

    # Val/train gap
    val_loss = status.get("validation_loss")
    train_loss = status.get("loss")
    val_train_gap = None
    gap_status = "good"

    if val_loss is not None and train_loss is not None:
        val_train_gap = val_loss - train_loss
        if val_train_gap > 0.5:
            gap_status = "overfitting"
        elif val_train_gap > 0.3:
            gap_status = "warning"

    # Snapshots (from retention system if available)
    snapshots = status.get("recent_snapshots", [])

    view = EvalStatusView(
        fixed_eval_em=fixed_eval.get("em"),
        fixed_eval_ce=fixed_eval.get("ce"),
        fixed_eval_ece=fixed_eval.get("ece"),
        fixed_eval_trend=eval_trend,
        fixed_eval_step=fixed_eval.get("step"),
        fixed_eval_timestamp=fixed_eval.get("timestamp"),
        domain_evals=fixed_eval.get("by_domain", {}),
        micro_eval_loss=micro_eval.get("loss"),
        micro_eval_step=micro_eval.get("step"),
        val_loss=val_loss,
        train_loss=train_loss,
        val_train_gap=val_train_gap,
        gap_status=gap_status,
        snapshots=snapshots,
    )

    return jsonify(view.to_dict())


@app.route('/api/status/system')
def api_status_system():
    """
    GET /api/status/system

    System resources + job queues.
    Polled every 5-10 seconds.
    """
    status = load_training_status()

    # Extract system stats
    system_stats = status.get("system_stats", {})
    gpu_stats = status.get("gpu_stats", {})

    view = SystemStatusView(
        system_4090={
            "cpu_pct": system_stats.get("cpu_pct", 0),
            "ram_gb": system_stats.get("ram_used_gb", 0),
            "ram_total_gb": system_stats.get("ram_total_gb", 0),
            "disk_used_gb": system_stats.get("disk_used_gb", 0),
            "disk_total_gb": system_stats.get("disk_total_gb", 0),
            "disk_pct": system_stats.get("disk_pct", 0),
            "swap_used_gb": system_stats.get("swap_used_gb", 0),
            "swap_total_gb": system_stats.get("swap_total_gb", 0),
        },
        system_3090={
            "online": status.get("remote_3090_online", False),
            "cpu_pct": 0,  # TODO: fetch from 3090
            "ram_gb": 0.0,
            "ram_total_gb": 0.0,
            "disk_used_gb": 0.0,
            "disk_total_gb": 0.0,
        },
        queues={
            "preview_jobs_pending": status.get("preview_jobs_pending", 0),
            "eval_jobs_pending": status.get("eval_jobs_pending", 0),
            "data_gen_jobs_pending": status.get("data_gen_jobs_pending", 0),
            "training_queue_high": status.get("queue_high", 0),
            "training_queue_normal": status.get("queue_normal", 0),
            "training_queue_low": status.get("queue_low", 0),
        }
    )

    return jsonify(view.to_dict())


@app.route('/api/preview/history')
def api_preview_history():
    """
    GET /api/preview/history?limit=100&offset=0&regime=emoji_think&exact_match=false

    Paginated preview history with filtering.
    """
    preview_logger = get_preview_logger(BASE_DIR)

    # Parse query parameters
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    regime = request.args.get('regime')
    source_pool = request.args.get('source_pool')

    # Parse exact_match (can be 'true', 'false', or None)
    exact_match = None
    if 'exact_match' in request.args:
        exact_match = request.args.get('exact_match').lower() == 'true'

    # Parse since timestamp
    since = None
    if 'since' in request.args:
        try:
            since = datetime.fromisoformat(request.args.get('since'))
        except:
            pass

    # Get previews
    previews = preview_logger.get_recent_previews(
        limit=limit,
        offset=offset,
        since=since,
        regime=regime,
        source_pool=source_pool,
        exact_match=exact_match
    )

    return jsonify({
        "previews": previews,
        "limit": limit,
        "offset": offset,
        "count": len(previews),
    })


@app.route('/api/throughput/samples')
def api_throughput_samples():
    """
    GET /api/throughput/samples?limit=100

    Throughput vs VRAM correlation data.
    """
    status = load_training_status()
    limit = int(request.args.get('limit', 100))

    # Get VRAM samples from status
    samples = status.get("throughput_vram_samples", [])

    # Return last N samples
    return jsonify({
        "samples": samples[-limit:] if samples else [],
        "count": len(samples[-limit:]) if samples else 0,
    })


# Legacy endpoints (backward compatibility)

@app.route('/status/training_status.json')
def legacy_training_status():
    """
    GET /status/training_status.json

    Legacy endpoint for old UI. Returns full training_status.json.
    """
    status = load_training_status()
    return jsonify(status)


# Static file serving (for UI)

@app.route('/')
def index():
    """Serve the landing page"""
    return send_from_directory(BASE_DIR / "monitoring" / "ui", "index.html")


@app.route('/ui/<path:path>')
def serve_ui(path):
    """Serve UI static files"""
    return send_from_directory(BASE_DIR / "monitoring" / "ui", path)


@app.route('/live_monitor_ui_v2.html')
def legacy_monitor():
    """Serve legacy monitor UI"""
    return send_from_directory(BASE_DIR / "monitoring" / "ui", "live_monitor_ui_v2.html")


@app.route('/js/<path:path>')
def serve_js(path):
    """Serve JavaScript files"""
    return send_from_directory(BASE_DIR / "monitoring" / "js", path)


@app.route('/css/<path:path>')
def serve_css(path):
    """Serve CSS files"""
    return send_from_directory(BASE_DIR / "monitoring" / "css", path)


# Health check

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    })


def main():
    parser = argparse.ArgumentParser(description="Monitoring API Server V2")
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--base-dir', default='/path/to/training',
                       help='Base directory for training system')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    global BASE_DIR, STATUS_FILE
    BASE_DIR = Path(args.base_dir)
    STATUS_FILE = BASE_DIR / "status" / "training_status.json"

    print(f"Starting Monitoring API Server V2")
    print(f"  Base dir: {BASE_DIR}")
    print(f"  Status file: {STATUS_FILE}")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"\nEndpoints:")
    print(f"  http://{args.host}:{args.port}/api/status/live")
    print(f"  http://{args.host}:{args.port}/api/status/preview")
    print(f"  http://{args.host}:{args.port}/api/status/evals")
    print(f"  http://{args.host}:{args.port}/api/status/system")
    print(f"  http://{args.host}:{args.port}/api/preview/history")
    print(f"\nUI:")
    print(f"  http://{args.host}:{args.port}/")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
