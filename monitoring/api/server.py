#!/usr/bin/env python3
"""
Unified Monitoring API Server
Phase 3, Task 3.1: Standalone API server on port 8081

Provides unified access to all monitoring data sources through a
single REST API endpoint.
"""

from flask import Flask, jsonify, request, send_from_directory
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

# Static file directories
MONITORING_DIR = Path(__file__).parent.parent
UI_DIR = MONITORING_DIR / 'ui'
CSS_DIR = MONITORING_DIR / 'css'
JS_DIR = MONITORING_DIR / 'js'


# Static file routes for dashboard UI
@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files"""
    return send_from_directory(CSS_DIR, filename)


@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files"""
    return send_from_directory(JS_DIR, filename)


@app.route('/ui/<path:filename>')
def serve_ui(filename):
    """Serve UI HTML files"""
    return send_from_directory(UI_DIR, filename)


@app.route('/<path:filename>.html')
def serve_html(filename):
    """Serve HTML files from root path"""
    return send_from_directory(UI_DIR, f'{filename}.html')


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
            '/api/curriculum-state': 'Get curriculum progression state',
            '/api/lineage': 'Get data lineage stats (per-generator/validator rejection rates)',
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

        # Get min_queue_depth from config
        config_path = base_dir / 'config.json'
        min_queue_depth = 2  # default
        try:
            with open(config_path) as f:
                config = json.load(f)
                min_queue_depth = config.get('auto_generate', {}).get('min_queue_depth', 2)
        except Exception:
            pass

        # Quest Board health status
        queue_healthy = len(files) >= min_queue_depth

        return jsonify({
            'total_files': len(files),
            'total_examples': total_examples,
            'min_queue_depth': min_queue_depth,
            'queue_healthy': queue_healthy,
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


@app.route('/api/queue/preview/<path:filename>')
def api_queue_preview(filename):
    """
    Get random examples from a queue file for preview.

    Args:
        filename: Name of the JSONL file in the queue

    Query params:
        count: Number of examples to return (default 5, max 20)

    Returns random examples with prompt and golden answer.
    """
    import random

    try:
        base_dir = Path('/path/to/training')
        count = min(int(request.args.get('count', 5)), 20)

        # Find the file in queue directories
        file_path = None
        for priority in ('high', 'normal', 'low', 'processing'):
            candidate = base_dir / 'queue' / priority / filename
            if candidate.exists():
                file_path = candidate
                break

        # Also check inbox
        if not file_path:
            candidate = base_dir / 'inbox' / filename
            if candidate.exists():
                file_path = candidate

        if not file_path:
            return jsonify({
                'error': f'File not found: {filename}',
                'filename': filename
            }), 404

        # Read all examples
        examples = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not examples:
            return jsonify({
                'error': 'No valid examples in file',
                'filename': filename,
                'total_examples': 0
            }), 400

        # Select random examples
        selected = random.sample(examples, min(count, len(examples)))

        # Extract prompt and response for display
        previews = []
        for ex in selected:
            # Handle different formats (messages array or prompt/response)
            if 'messages' in ex:
                messages = ex['messages']
                # Find user message (prompt) and assistant message (response)
                prompt = None
                response = None
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'user' and not prompt:
                        prompt = content
                    elif role == 'assistant':
                        response = content
                previews.append({
                    'prompt': prompt or '(no prompt)',
                    'response': response or '(no response)',
                    'format': 'messages'
                })
            elif 'prompt' in ex:
                previews.append({
                    'prompt': ex.get('prompt', '(no prompt)'),
                    'response': ex.get('response', ex.get('completion', '(no response)')),
                    'format': 'prompt_response'
                })
            else:
                # Unknown format - show raw
                previews.append({
                    'prompt': str(ex)[:500],
                    'response': '(unknown format)',
                    'format': 'raw'
                })

        return jsonify({
            'filename': filename,
            'total_examples': len(examples),
            'preview_count': len(previews),
            'previews': previews
        })

    except Exception as e:
        logger.error(f"Error in /api/queue/preview: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'filename': filename
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


@app.route('/api/syllo-generator')
def api_syllo_generator():
    """
    Get SYLLO L1 generator daemon status.

    Returns last generation info and daemon status.
    """
    try:
        import subprocess
        status_file = Path('/path/to/training/status/syllo_l1_generator.json')
        pid_file = Path('/path/to/training/.pids/syllo_l1_generator.pid')

        result = {
            'daemon_running': False,
            'pid': None,
            'last_generation': None
        }

        # Check if daemon is running
        if pid_file.exists():
            pid = pid_file.read_text().strip()
            try:
                # Check if process exists
                subprocess.run(['kill', '-0', pid], check=True, capture_output=True)
                result['daemon_running'] = True
                result['pid'] = int(pid)
            except subprocess.CalledProcessError:
                pass

        # Load last generation info
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
            result['last_generation'] = status.get('last_generation')
            result['generator_id'] = status.get('generator_id')
            result['generator_version'] = status.get('generator_version')

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /api/syllo-generator: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/lineage')
def api_lineage():
    """
    Get data lineage statistics.

    Returns per-generator and per-validator validation stats to answer:
    - Which generator produces the most rejections?
    - Which validator is most aggressive?
    - What are the top error reasons?

    Response:
    {
        "total_validations": 1234,
        "last_updated": "2025-11-26T...",
        "generators": {
            "discrimination@1.0.0": {"total": 100, "passed": 95, "failed": 5, ...},
            ...
        },
        "validators": {
            "data_validator@1.0.0": {"total": 1000, "passed": 950, ...},
            ...
        },
        "summary": {
            "worst_generator": {"id": "...", "fail_rate": 5.0},
            "worst_validator": {"id": "...", "fail_rate": 2.0},
            "overall_fail_rate": 3.5
        }
    }
    """
    try:
        lineage_file = Path('/path/to/training/status/data_lineage.json')
        if lineage_file.exists():
            with open(lineage_file) as f:
                lineage = json.load(f)
            return jsonify(lineage)
        else:
            # Return empty structure if no data yet
            return jsonify({
                'total_validations': 0,
                'last_updated': None,
                'generators': {},
                'validators': {},
                'summary': {
                    'total_generators': 0,
                    'total_validators': 0,
                    'worst_generator': None,
                    'worst_validator': None,
                    'overall_fail_rate': 0.0
                }
            })
    except Exception as e:
        logger.error(f"Error in /api/lineage: {e}", exc_info=True)
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
            '/api/curriculum-state',
            '/api/lineage'
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
