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

PORT = 8080
DIRECTORY = Path(__file__).parent

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

        # Default handling for other paths
        super().do_GET()

def main():
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"""
╔══════════════════════════════════════════════════════════╗
║         TRAINING LIVE MONITOR STARTED                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Open in your browser:                                   ║
║  http://localhost:{PORT}/live_monitor_ui.html               ║
║                                                          ║
║  This page will auto-refresh every 2 seconds             ║
║  showing current training progress.                      ║
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
