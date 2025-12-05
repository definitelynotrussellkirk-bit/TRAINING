#!/usr/bin/env python3
"""
Simple API server for system memory stats
Runs on port 8081
"""
import json
import psutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

class MemoryStatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/memory_stats'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Get system memory
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Get Python training process memory (if running)
            training_mem_mb = 0
            training_pid = None
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
                    try:
                        # Get cmdline safely
                        cmdline = proc.cmdline()
                        cmdline_str = ' '.join(cmdline).lower()

                        # Look for hero_loop or train.py
                        if 'python' in proc.info['name'].lower() and ('hero_loop' in cmdline_str or 'train.py' in cmdline_str):
                            training_mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
                            training_pid = proc.info['pid']
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                pass

            stats = {
                'total_gb': mem.total / (1024**3),
                'used_gb': mem.used / (1024**3),
                'available_gb': mem.available / (1024**3),
                'percent': mem.percent,
                'ram_percent': mem.percent,  # For UI compatibility
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent,
                'training_process': {
                    'pid': training_pid,
                    'memory_mb': round(training_mem_mb, 2),
                    'memory_gb': round(training_mem_mb / 1024, 2)
                } if training_pid else None,
                'training_process_mb': training_mem_mb,
                'training_process_gb': training_mem_mb / 1024,
                'oom_risk': 'high' if mem.percent > 85 else 'medium' if mem.percent > 70 else 'low'
            }

            self.wfile.write(json.dumps(stats).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Silent

if __name__ == '__main__':
    # Use ThreadingHTTPServer to handle concurrent requests without blocking
    server = ThreadingHTTPServer(('0.0.0.0', 8081), MemoryStatsHandler)
    print("Memory stats API running on http://0.0.0.0:8081/api/memory_stats")
    server.serve_forever()
