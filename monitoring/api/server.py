#!/usr/bin/env python3
"""
Unified Monitoring API Server
Phase 3, Task 3.1: Standalone API server on port 8081

Provides unified access to all monitoring data sources through a
single REST API endpoint.
"""

from flask import Flask, jsonify, request
import logging
import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from monitoring.api.aggregator import DataAggregator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Enable CORS manually (without flask-cors dependency)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Create aggregator
aggregator = DataAggregator()

logger.info("Unified Monitoring API Server initialized")
logger.info(f"Registered {len(aggregator.registry.plugins)} plugins")


@app.route('/')
def index():
    """API documentation root"""
    return jsonify({
        'name': 'Unified Monitoring API',
        'version': '1.0.0',
        'description': 'Unified access to all training system monitoring data',
        'endpoints': {
            '/api/unified': 'Get all monitoring data from all sources',
            '/api/health': 'Get health status of all plugins',
            '/api/sources': 'List available data sources',
            '/api/cache/clear': 'Clear all plugin caches (POST)',
            '/api/queue': 'Get training queue status and pipeline info',
        },
        'machines': {
            '4090': 'Training machine (local)',
            '3090': 'Intelligence machine (192.168.x.x)'
        }
    })


def clean_nan(obj):
    """Recursively replace NaN and Infinity with None in nested dicts/lists"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


@app.route('/api/unified')
def api_unified():
    """
    Get unified data from all plugins.

    Returns consolidated data from:
    - Training status (4090)
    - GPU statistics (4090 + 3090)
    - Curriculum optimization (3090)
    - Plus high-level summary
    """
    try:
        data = aggregator.get_unified_data()
        # Clean NaN values before returning
        data = clean_nan(data)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in /api/unified: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'timestamp': None
        }), 500


@app.route('/api/health')
def api_health():
    """
    Get health status of all plugins.

    Returns health check for each data source including:
    - Plugin name and status
    - Cache age
    - Error information
    """
    try:
        health = aggregator.get_health()
        return jsonify(health)
    except Exception as e:
        logger.error(f"Error in /api/health: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'overall_status': 'error'
        }), 500


@app.route('/api/sources')
def api_sources():
    """
    List available data sources.

    Returns metadata for each registered plugin.
    """
    try:
        plugins = aggregator.registry.get_all()

        sources = {}
        for name, plugin in plugins.items():
            metadata = plugin.get_metadata()
            sources[name] = {
                'name': name,
                'description': metadata.get('description', ''),
                'machine': metadata.get('machine', 'unknown'),
                'location': metadata.get('location', 'unknown'),
                'refresh_interval': metadata.get('refresh_interval', 0),
                'critical': metadata.get('critical', False),
                'data_type': metadata.get('data_type', 'unknown')
            }

        return jsonify({
            'count': len(sources),
            'sources': sources
        })
    except Exception as e:
        logger.error(f"Error in /api/sources: {e}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/cache/clear', methods=['POST'])
def api_cache_clear():
    """
    Clear all plugin caches.

    Forces fresh data fetch on next request.
    Method: POST
    """
    try:
        aggregator.clear_caches()
        return jsonify({
            'success': True,
            'message': 'All caches cleared'
        })
    except Exception as e:
        logger.error(f"Error in /api/cache/clear: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/queue')
def api_queue():
    """
    Get training queue status.

    Returns queue depth, pending examples, and recent batch info.
    Used by SYLLO dashboard for pipeline status.
    """
    try:
        base_dir = Path('/path/to/training')
        queue_dir = base_dir / 'queue'

        # Count files in each priority queue
        files = []
        total_examples = 0

        for priority in ('high', 'normal', 'low'):
            priority_dir = queue_dir / priority
            if priority_dir.exists():
                for f in priority_dir.glob('*.jsonl'):
                    # Count lines (examples) in file
                    try:
                        with open(f, 'r') as fh:
                            line_count = sum(1 for _ in fh)
                        files.append({
                            'name': f.name,
                            'priority': priority,
                            'examples': line_count,
                            'size_mb': round(f.stat().st_size / (1024 * 1024), 2),
                            'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                        })
                        total_examples += line_count
                    except Exception:
                        files.append({
                            'name': f.name,
                            'priority': priority,
                            'examples': 0,
                            'error': 'Could not read file'
                        })

        # Get recently completed files
        recent = []
        recently_completed = queue_dir / 'recently_completed'
        if recently_completed.exists():
            for f in sorted(recently_completed.glob('*.jsonl'),
                           key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                recent.append({
                    'name': f.name,
                    'completed': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })

        # Check inbox
        inbox_dir = base_dir / 'inbox'
        inbox_files = list(inbox_dir.glob('*.jsonl')) if inbox_dir.exists() else []

        return jsonify({
            'total_files': len(files),
            'total_examples': total_examples,
            'files': sorted(files, key=lambda x: (
                {'high': 0, 'normal': 1, 'low': 2}.get(x.get('priority', 'normal'), 1),
                x.get('modified', '')
            )),
            'recent': recent,
            'inbox_count': len(inbox_files),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in /api/queue: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'total_files': 0,
            'total_examples': 0
        }), 500


@app.route('/api/curriculum-state')
def api_curriculum_state():
    """
    Get curriculum state directly from curriculum_state.json.

    Returns current level, active skill, and progression history.
    """
    try:
        state_file = Path('/path/to/training/data_manager/curriculum_state.json')
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            return jsonify(state)
        else:
            return jsonify({
                'error': 'Curriculum state file not found',
                'skills': {
                    'syllo': {'current_level': 1, 'accuracy_history': []},
                    'binary': {'current_level': 1, 'accuracy_history': []}
                },
                'active_skill': 'syllo'
            })
    except Exception as e:
        logger.error(f"Error in /api/curriculum-state: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/',
            '/api/unified',
            '/api/health',
            '/api/sources',
            '/api/cache/clear',
            '/api/queue',
            '/api/curriculum-state'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


def main():
    """Start the API server"""
    port = 8081
    host = '0.0.0.0'

    logger.info("=" * 60)
    logger.info("Starting Unified Monitoring API Server")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Plugins: {len(aggregator.registry.plugins)}")
    logger.info("=" * 60)

    # List registered plugins
    for name, plugin in aggregator.registry.plugins.items():
        metadata = plugin.get_metadata()
        logger.info(f"  - {name}: {metadata.get('description', 'No description')}")

    logger.info("=" * 60)
    logger.info(f"API available at: http://{host}:{port}")
    logger.info(f"Try: curl http://localhost:{port}/api/unified | jq")
    logger.info("=" * 60)

    # Start server
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
