#!/usr/bin/env python3
"""
Launch Live Monitor UI Server

Serves the live monitor HTML and status JSON files.
Run this and open http://localhost:8080 in your browser.
"""

import http.server
import socketserver
from pathlib import Path
import subprocess
import json
import threading
import time
from urllib import request, error as urllib_error

PORT = 8080
DIRECTORY = Path(__file__).parent.parent.parent  # Serve from project root

# Global cache for 3090 data
GPU_3090_CACHE = {
    'online': False,
    'temp_c': 0,
    'vram_used_gb': 0.0,
    'vram_total_gb': 24.0,
    'vram_pct': 0,
    'power_profile': None,
    'last_update': None,
    'utilization_gpu': 0,
    'power_draw_w': 0,
    'power_limit_w': 350
}

def poll_3090_gpu():
    """Background thread to poll 3090 API every 10 seconds"""
    REMOTE_API = "http://192.168.x.x:8765"

    print("  ✓ 3090 GPU polling thread started")

    while True:
        try:
            # Health check
            health_req = request.Request(f"{REMOTE_API}/health", method='GET')
            with request.urlopen(health_req, timeout=5) as resp:
                health = json.loads(resp.read().decode())

            # GPU stats
            gpu_req = request.Request(f"{REMOTE_API}/gpu", method='GET')
            with request.urlopen(gpu_req, timeout=5) as resp:
                gpu = json.loads(resp.read().decode())

            # Power profile
            power_req = request.Request(f"{REMOTE_API}/settings/power_profile", method='GET')
            with request.urlopen(power_req, timeout=5) as resp:
                power = json.loads(resp.read().decode())

            # Update cache
            GPU_3090_CACHE.update({
                'online': health.get('status') == 'ok',
                'temp_c': gpu.get('temperature_gpu', 0),
                'vram_used_gb': gpu.get('memory_used_mb', 0) / 1024,
                'vram_total_gb': gpu.get('memory_total_mb', 24576) / 1024,
                'vram_pct': int((gpu.get('memory_used_mb', 0) / gpu.get('memory_total_mb', 24576)) * 100) if gpu.get('memory_total_mb', 24576) > 0 else 0,
                'power_profile': power.get('current', 'unknown'),
                'utilization_gpu': gpu.get('utilization_gpu', 0),
                'power_draw_w': int(gpu.get('power_draw_w', 0)),
                'power_limit_w': int(gpu.get('power_limit_w', 350)),
                'last_update': time.time()
            })

        except Exception as e:
            # Mark offline on error
            GPU_3090_CACHE['online'] = False
            # Don't spam errors, just update once per minute when offline
            if int(time.time()) % 60 == 0:
                print(f"  ⚠ 3090 poll error: {e}")

        time.sleep(10)  # Poll every 10 seconds

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        # Add CORS headers to allow browser access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def guess_type(self, path):
        """Override to ensure HTML files are served with UTF-8 charset"""
        content_type = super().guess_type(path)
        if content_type == 'text/html':
            return 'text/html; charset=utf-8'
        return content_type

    def do_GET(self):
        # Serve live status endpoint
        if self.path.startswith('/api/status/live'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                status_file = DIRECTORY / 'status' / 'training_status.json'

                # Get GPU stats
                gpu_result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)

                gpu_stats = {'temp_c': 0, 'util_pct': 0, 'vram_used_gb': 0, 'vram_total_gb': 24, 'vram_pct': 0, 'power_w': 0, 'power_limit_w': 450}
                if gpu_result.returncode == 0:
                    parts = [p.strip() for p in gpu_result.stdout.split(',')]
                    gpu_stats = {
                        'temp_c': int(float(parts[0])),
                        'util_pct': int(float(parts[1])),
                        'vram_used_gb': float(parts[2]) / 1024,
                        'vram_total_gb': float(parts[3]) / 1024,
                        'vram_pct': int((float(parts[2]) / float(parts[3])) * 100),
                        'power_w': int(float(parts[4])),
                        'power_limit_w': int(float(parts[5]))
                    }

                # Get RAM stats
                import psutil
                mem = psutil.virtual_memory()
                ram_stats = {
                    'used_gb': mem.used / (1024**3),
                    'total_gb': mem.total / (1024**3)
                }

                # Load training status
                status_data = {}
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        status_data = json.load(f)

                response = {
                    'status': status_data.get('status', 'idle'),
                    'current_step': status_data.get('current_step', 0),
                    'total_steps': status_data.get('total_steps', 0),
                    'epoch': status_data.get('epoch', 0.0),
                    'loss': status_data.get('loss', 0.0),
                    'streaming_ce': status_data.get('streaming_ce'),
                    'loss_variance': status_data.get('loss_variance', 0.0),
                    'val_train_gap': status_data.get('val_train_gap'),
                    'val_loss': status_data.get('val_loss'),
                    'train_loss': status_data.get('train_loss'),
                    'loss_trend': status_data.get('loss_trend', 'stable'),
                    'throughput_trend': status_data.get('throughput_trend', 'stable'),
                    'tokens_per_sec': status_data.get('tokens_per_sec', 0),
                    'tokens_per_sec_avg': status_data.get('tokens_per_sec_avg', 0),
                    'tokens_per_sec_baseline': status_data.get('tokens_per_sec_baseline', 0),
                    'current_model_name': status_data.get('model_name', '-'),
                    'current_checkpoint_id': status_data.get('checkpoint_id', '-'),
                    'current_file': status_data.get('current_file', '-'),
                    'batch_step': status_data.get('batch_idx', 0),
                    'batch_total_steps': status_data.get('total_batches', 0),
                    'batch_queue_size': status_data.get('queue_size', 0),
                    'eta_current_file': status_data.get('eta_current_file'),
                    'eta_overall': status_data.get('eta_overall'),
                    'gpu_4090': gpu_stats,
                    'gpu_3090': GPU_3090_CACHE.copy(),
                    'ram': ram_stats
                }

                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve preview status endpoint
        if self.path.startswith('/api/status/preview'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                # Load latest preview from file
                latest_preview_file = DIRECTORY / 'status' / 'latest_preview.json'
                latest_preview = None
                if latest_preview_file.exists():
                    with open(latest_preview_file, 'r') as f:
                        latest_preview = json.load(f)

                # Load preview history for trends
                preview_history_dir = DIRECTORY / 'data' / 'preview_history'
                em_trend = []
                if preview_history_dir.exists():
                    history_files = sorted(preview_history_dir.glob('preview_step_*.json'))[-20:]
                    for file in history_files:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            em_trend.append({
                                'step': data['training_step'],
                                'em_rate': data['metrics']['exact_match_rate'],
                                'timestamp': data['timestamp']
                            })

                # Calculate rolling averages
                all_em = [p['em_rate'] for p in em_trend]
                em_last_20 = sum(all_em[-20:]) / len(all_em[-20:]) if len(all_em) >= 1 else 0.0
                em_last_50 = sum(all_em[-50:]) / len(all_em[-50:]) if len(all_em) >= 1 else 0.0

                response = {
                    'latest_preview': latest_preview,
                    'preview_em_last_20': em_last_20,
                    'preview_em_last_50': em_last_50,
                    'em_trend': em_trend
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve evals status endpoint
        if self.path.startswith('/api/status/evals'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                response = {
                    'fixed_eval_em': None,
                    'fixed_eval_ce': None,
                    'fixed_eval_ece': None,
                    'snapshots': []
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve queue health endpoint
        if self.path.startswith('/api/status/queue_health'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                import sys
                sys.path.insert(0, str(DIRECTORY / 'monitoring'))
                from queue_health import QueueHealthMonitor

                monitor = QueueHealthMonitor(base_dir=DIRECTORY)
                health = monitor.analyze_queue_health()
                self.wfile.write(json.dumps(health).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve system status endpoint
        if self.path.startswith('/api/status/system'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                import psutil
                disk = psutil.disk_usage('/')

                response = {
                    'system_4090': {
                        'disk_used_gb': disk.used / (1024**3),
                        'disk_total_gb': disk.total / (1024**3)
                    }
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve config endpoint
        if self.path.startswith('/api/config'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                config_file = DIRECTORY / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        # Extract key fields for display
                        self.wfile.write(json.dumps({
                            'lora_r': config.get('lora_r', 128),
                            'lora_alpha': config.get('lora_alpha', 128),
                            'model_name': config.get('model_name', 'Qwen 2.5 7B Instruct')
                        }).encode())
                else:
                    self.wfile.write(json.dumps({"error": "config not found"}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve inbox files endpoint
        if self.path.startswith('/api/inbox_files'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                inbox_path = DIRECTORY / 'inbox'
                import glob
                import os

                # Get all .jsonl files in inbox root (not subdirectories)
                jsonl_files = glob.glob(str(inbox_path / '*.jsonl'))

                files = []
                for f in jsonl_files:
                    stat = os.stat(f)
                    files.append({
                        'name': os.path.basename(f),
                        'size_mb': stat.st_size / (1024 * 1024),
                        'modified': stat.st_mtime
                    })

                # Sort by modification time (newest first)
                files.sort(key=lambda x: x['modified'], reverse=True)

                self.wfile.write(json.dumps({
                    'count': len(files),
                    'files': files
                }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e), "count": 0, "files": []}).encode())
            return

        # Serve flagged examples endpoint
        if self.path.startswith('/api/flagged_examples'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                flagged_file = DIRECTORY / 'flagged_examples' / 'flagged_examples.json'
                if flagged_file.exists():
                    with open(flagged_file, 'r') as f:
                        flagged_data = json.load(f)

                        # Calculate statistics
                        total = len(flagged_data)
                        by_reason = {}
                        for ex in flagged_data:
                            reason = ex.get('flag_reason', 'unknown')
                            by_reason[reason] = by_reason.get(reason, 0) + 1

                        avg_loss = sum(ex.get('loss', 0) for ex in flagged_data) / total if total > 0 else 0
                        matches = sum(1 for ex in flagged_data if ex.get('matches', False))
                        match_rate = matches / total if total > 0 else 0

                        self.wfile.write(json.dumps({
                            'total': total,
                            'examples': flagged_data,
                            'statistics': {
                                'by_reason': by_reason,
                                'avg_loss': avg_loss,
                                'match_rate': match_rate
                            }
                        }).encode())
                else:
                    self.wfile.write(json.dumps({
                        'total': 0,
                        'examples': [],
                        'statistics': {'by_reason': {}, 'avg_loss': 0, 'match_rate': 0}
                    }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({
                    "error": str(e),
                    'total': 0,
                    'examples': [],
                    'statistics': {}
                }).encode())
            return

        # Serve random samples from inbox files
        if self.path.startswith('/api/inbox_samples'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                inbox_path = DIRECTORY / 'inbox'
                import glob
                import os
                import random

                # Get all .jsonl files in inbox
                jsonl_files = glob.glob(str(inbox_path / '*.jsonl'))

                samples = []
                for filepath in jsonl_files:
                    filename = os.path.basename(filepath)

                    # Read file and get random sample
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if lines:
                                # Pick random line
                                sample_line = random.choice(lines)
                                sample_data = json.loads(sample_line)

                                # Extract prompt and answer from messages format
                                messages = sample_data.get('messages', [])
                                prompt = None
                                answer = None

                                for msg in messages:
                                    if msg.get('role') == 'user':
                                        prompt = msg.get('content', '')
                                    elif msg.get('role') == 'assistant':
                                        answer = msg.get('content', '')

                                if prompt and answer:
                                    samples.append({
                                        'filename': filename,
                                        'prompt': prompt,
                                        'answer': answer,
                                        'total_lines': len(lines)
                                    })
                    except Exception as e:
                        # Skip files that can't be read
                        pass

                self.wfile.write(json.dumps({
                    'samples': samples
                }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({
                    'error': str(e),
                    'samples': []
                }).encode())
            return

        # Serve evolution datasets list endpoint
        if self.path.startswith('/api/evolution/datasets'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                evolution_dir = DIRECTORY / 'data' / 'evolution_snapshots'
                datasets = []

                if evolution_dir.exists():
                    for dataset_dir in evolution_dir.iterdir():
                        if dataset_dir.is_dir():
                            snapshots = list(dataset_dir.glob('step_*.json'))
                            if snapshots:
                                # Get first and last snapshot for summary
                                snapshot_steps = sorted([int(s.stem.split('_')[1]) for s in snapshots])
                                datasets.append({
                                    'name': dataset_dir.name,
                                    'snapshot_count': len(snapshots),
                                    'first_step': snapshot_steps[0],
                                    'last_step': snapshot_steps[-1],
                                    'path': str(dataset_dir.relative_to(DIRECTORY))
                                })

                datasets.sort(key=lambda x: x['last_step'], reverse=True)
                self.wfile.write(json.dumps({'datasets': datasets}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e), 'datasets': []}).encode())
            return

        # Serve evolution dataset snapshots endpoint
        if self.path.startswith('/api/evolution/') and '/snapshots' in self.path:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                # Extract dataset name from path like /api/evolution/DATASET/snapshots
                parts = self.path.split('/')
                dataset_name = parts[3] if len(parts) > 3 else None

                if not dataset_name:
                    self.wfile.write(json.dumps({'error': 'Dataset name required'}).encode())
                    return

                dataset_dir = DIRECTORY / 'data' / 'evolution_snapshots' / dataset_name

                if not dataset_dir.exists():
                    self.wfile.write(json.dumps({'error': 'Dataset not found', 'snapshots': []}).encode())
                    return

                # Load all snapshots
                snapshots = []
                for snapshot_file in sorted(dataset_dir.glob('step_*.json')):
                    try:
                        with open(snapshot_file, 'r') as f:
                            data = json.load(f)
                            snapshots.append({
                                'step': data['training_step'],
                                'timestamp': data['timestamp'],
                                'summary': data['summary'],
                                'filename': snapshot_file.name
                            })
                    except Exception as e:
                        print(f"Error reading {snapshot_file}: {e}")

                self.wfile.write(json.dumps({'snapshots': snapshots}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e), 'snapshots': []}).encode())
            return

        # Serve specific evolution snapshot endpoint
        if self.path.startswith('/api/evolution/') and '/snapshot/' in self.path:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                # Extract dataset and step from path like /api/evolution/DATASET/snapshot/100
                parts = self.path.split('/')
                dataset_name = parts[3] if len(parts) > 3 else None
                step = parts[5] if len(parts) > 5 else None

                if not dataset_name or not step:
                    self.wfile.write(json.dumps({'error': 'Dataset and step required'}).encode())
                    return

                snapshot_file = DIRECTORY / 'data' / 'evolution_snapshots' / dataset_name / f'step_{int(step):06d}.json'

                if snapshot_file.exists():
                    with open(snapshot_file, 'r') as f:
                        self.wfile.write(f.read().encode())
                else:
                    self.wfile.write(json.dumps({'error': 'Snapshot not found'}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({'error': str(e)}).encode())
            return

        # Serve GPU stats endpoint
        if self.path.startswith('/api/gpu_stats'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                result = subprocess.run([
                    'nvidia-smi',
                    '--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,name',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    stats = {
                        'temperature': float(parts[0].strip()),
                        'gpu_utilization': float(parts[1].strip()),
                        'memory_utilization': float(parts[2].strip()),
                        'memory_used_mb': float(parts[3].strip()),
                        'memory_total_mb': float(parts[4].strip()),
                        'power_draw_w': float(parts[5].strip()),
                        'power_limit_w': float(parts[6].strip()),
                        'gpu_name': parts[7].strip()
                    }
                    self.wfile.write(json.dumps(stats).encode())
                else:
                    self.wfile.write(json.dumps({"error": "nvidia-smi failed"}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Serve predictions endpoint (fetch from 3090)
        if self.path.startswith('/api/predictions'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                import subprocess
                # Fetch predictions from 3090 where daemon is running
                result = subprocess.run(
                    ['ssh', '192.168.x.x', 'cat /home/user/TRAINING/status/latest_predictions.json'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0 and result.stdout:
                    predictions_data = json.loads(result.stdout)
                    self.wfile.write(json.dumps(predictions_data).encode())
                else:
                    # Fallback to local file
                    predictions_file = DIRECTORY / 'status' / 'latest_predictions.json'
                    if predictions_file.exists():
                        with open(predictions_file, 'r') as f:
                            predictions_data = json.load(f)
                        self.wfile.write(json.dumps(predictions_data).encode())
                    else:
                        self.wfile.write(json.dumps({
                            "error": "No predictions available",
                            "checkpoint": None,
                            "predictions": [],
                            "stats": {
                                "total": 0,
                                "accuracy_auto": 0,
                                "accuracy_human": None
                            }
                        }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Default handling for other paths
        super().do_GET()

    def do_POST(self):
        """Handle POST requests for grading predictions"""

        # Handle prediction grading
        if self.path.startswith('/api/predictions/grade'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                # Read POST data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))

                prediction_id = data.get('prediction_id')
                grade = data.get('grade')  # "correct", "incorrect", or "unsure"
                notes = data.get('notes', '')

                if not prediction_id or not grade:
                    self.wfile.write(json.dumps({
                        "success": False,
                        "error": "prediction_id and grade required"
                    }).encode())
                    return

                # Import and use the prediction engine to save grading
                import sys
                sys.path.insert(0, str(DIRECTORY / 'monitoring'))
                from prediction_viewer_engine import PredictionViewerEngine

                engine = PredictionViewerEngine(base_dir=DIRECTORY)
                engine.save_grading(prediction_id, grade, notes)

                self.wfile.write(json.dumps({
                    "success": True,
                    "message": f"Grading saved for {prediction_id}"
                }).encode())

            except Exception as e:
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": str(e)
                }).encode())
            return

        # Default 404 for other POST paths
        self.send_response(404)
        self.end_headers()

def main():
    socketserver.TCPServer.allow_reuse_address = True

    # Start 3090 polling thread before starting server
    poller_thread = threading.Thread(target=poll_3090_gpu, daemon=True)
    poller_thread.start()

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║         TRAINING LIVE MONITOR STARTED                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Open in your browser:                                   ║
║  http://localhost:{PORT}/monitoring/ui/control_room_v2.html ║
║                                                          ║
║  This page will auto-refresh every 2 seconds             ║
║  showing current training progress.                      ║
║                                                          ║
║  RTX 3090 remote monitoring: ENABLED                     ║
║  Polling interval: 10 seconds                            ║
║                                                          ║
║  Press Ctrl+C to stop the server                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")

if __name__ == "__main__":
    main()
