#!/usr/bin/env python3
"""
GPU Stats API - Serves GPU information as JSON
"""
import subprocess
import json
import sys

def get_gpu_stats():
    """Get GPU stats from nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,name',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(',')
        if len(parts) >= 8:
            return {
                'temperature': float(parts[0].strip()),
                'gpu_utilization': float(parts[1].strip()),
                'memory_utilization': float(parts[2].strip()),
                'memory_used_mb': float(parts[3].strip()),
                'memory_total_mb': float(parts[4].strip()),
                'power_draw_w': float(parts[5].strip()),
                'power_limit_w': float(parts[6].strip()),
                'gpu_name': parts[7].strip()
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}", file=sys.stderr)

    return None

if __name__ == '__main__':
    stats = get_gpu_stats()
    if stats:
        print(json.dumps(stats, indent=2))
    else:
        print(json.dumps({"error": "Failed to get GPU stats"}))
