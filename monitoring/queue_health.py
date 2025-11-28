#!/usr/bin/env python3
"""
Queue Health Monitor

Analyzes training queue health and provides traffic light indicators.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta


class QueueHealthMonitor:
    """
    Monitor queue health with traffic light indicators

    Health factors:
    - File freshness (age)
    - File validity (parseable JSON)
    - Processing time (stuck files)
    - Queue balance
    """

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.queue_dir = self.base_dir / "queue"

    def analyze_queue_health(self) -> Dict[str, Any]:
        """
        Analyze overall queue health

        Returns dict with:
        - status: 'healthy', 'warning', 'critical'
        - indicators: List of health indicators
        - queues: Per-queue stats
        """
        queues = {
            'high': self._analyze_queue('high'),
            'normal': self._analyze_queue('normal'),
            'low': self._analyze_queue('low'),
            'processing': self._analyze_queue('processing'),
            'failed': self._analyze_queue('failed')
        }

        # Determine overall health
        status = 'healthy'
        issues = []

        # Check for stuck processing files
        if queues['processing']['file_count'] > 0:
            for file_info in queues['processing']['files']:
                age_hours = file_info['age_hours']
                if age_hours > 24:
                    status = 'critical'
                    issues.append(f"File stuck in processing for {age_hours:.1f}h: {file_info['name']}")
                elif age_hours > 12:
                    if status == 'healthy':
                        status = 'warning'
                    issues.append(f"File in processing for {age_hours:.1f}h: {file_info['name']}")

        # Check for failed files
        if queues['failed']['file_count'] > 0:
            status = 'critical' if status != 'critical' else status
            issues.append(f"{queues['failed']['file_count']} file(s) in failed queue")

        # Check for empty queues (warning, not critical)
        total_pending = (queues['high']['file_count'] +
                        queues['normal']['file_count'] +
                        queues['low']['file_count'])
        if total_pending == 0 and queues['processing']['file_count'] == 0:
            if status == 'healthy':
                status = 'warning'
            issues.append("All queues empty")

        # Check for large queue backlog
        if total_pending > 10:
            if status == 'healthy':
                status = 'warning'
            issues.append(f"Large backlog: {total_pending} files pending")

        return {
            'status': status,
            'issues': issues,
            'timestamp': datetime.now().isoformat(),
            'queues': queues,
            'summary': {
                'total_pending': total_pending,
                'processing': queues['processing']['file_count'],
                'failed': queues['failed']['file_count']
            }
        }

    def _analyze_queue(self, queue_name: str) -> Dict[str, Any]:
        """Analyze a single queue directory"""
        queue_path = self.queue_dir / queue_name
        if not queue_path.exists():
            return {
                'file_count': 0,
                'total_size_mb': 0.0,
                'files': [],
                'health': 'unknown'
            }

        files = list(queue_path.glob('*.jsonl'))
        file_infos = []

        for file_path in files:
            stat = file_path.stat()
            age_seconds = time.time() - stat.st_mtime
            age_hours = age_seconds / 3600

            # Check file validity
            is_valid, sample_count = self._validate_file(file_path)

            file_infos.append({
                'name': file_path.name,
                'size_mb': stat.st_size / (1024 * 1024),
                'age_hours': age_hours,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'is_valid': is_valid,
                'sample_count': sample_count
            })

        total_size = sum(f['size_mb'] for f in file_infos)

        # Determine queue health
        health = 'healthy'
        if queue_name == 'failed' and len(files) > 0:
            health = 'critical'
        elif queue_name == 'processing' and file_infos:
            max_age = max(f['age_hours'] for f in file_infos)
            if max_age > 24:
                health = 'critical'
            elif max_age > 12:
                health = 'warning'

        return {
            'file_count': len(files),
            'total_size_mb': total_size,
            'files': sorted(file_infos, key=lambda x: x['age_hours'], reverse=True),
            'health': health
        }

    def _validate_file(self, file_path: Path, max_samples: int = 10) -> tuple[bool, int]:
        """
        Validate JSONL file

        Returns (is_valid, sample_count)
        """
        try:
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    json.loads(line.strip())
                    count += 1

            # Try to count total lines (fast estimate)
            try:
                with open(file_path, 'rb') as f:
                    total_count = sum(1 for _ in f)
                return (True, total_count)
            except:
                return (True, count)

        except Exception:
            return (False, 0)

    def get_queue_health_json(self) -> str:
        """Get queue health as JSON string"""
        health = self.analyze_queue_health()
        return json.dumps(health, indent=2)


def main():
    """CLI for testing queue health monitor"""
    import argparse

    parser = argparse.ArgumentParser(description='Monitor queue health')
    parser.add_argument('--base-dir', default=None, help='Base directory (default: auto-detected)')
    parser.add_argument('--watch', action='store_true', help='Watch mode (updates every 10s)')

    args = parser.parse_args()

    if args.base_dir is None:
        try:
            from core.paths import get_base_dir
            args.base_dir = str(get_base_dir())
        except (ImportError, Exception):
            from core.paths import get_base_dir; args.base_dir = str(get_base_dir())

    monitor = QueueHealthMonitor(base_dir=args.base_dir)

    if args.watch:
        print("Queue Health Monitor (Ctrl+C to exit)\n")
        while True:
            health = monitor.analyze_queue_health()

            # Clear screen
            print("\033[2J\033[H")

            # Print header
            print("=" * 60)
            print(f"Queue Health: {health['status'].upper()}")
            print(f"Updated: {health['timestamp']}")
            print("=" * 60)

            # Print summary
            print(f"\nSummary:")
            print(f"  Pending:    {health['summary']['total_pending']} files")
            print(f"  Processing: {health['summary']['processing']} files")
            print(f"  Failed:     {health['summary']['failed']} files")

            # Print issues
            if health['issues']:
                print(f"\nIssues:")
                for issue in health['issues']:
                    print(f"  ⚠ {issue}")

            # Print queue details
            print(f"\nQueue Details:")
            for queue_name, queue_data in health['queues'].items():
                status_icon = {
                    'healthy': '●',
                    'warning': '●',
                    'critical': '●',
                    'unknown': '○'
                }.get(queue_data['health'], '○')

                print(f"  {status_icon} {queue_name:12s}: {queue_data['file_count']:2d} files, {queue_data['total_size_mb']:6.1f} MB")

                # Show file details for problematic queues
                if queue_name in ['processing', 'failed'] and queue_data['files']:
                    for file_info in queue_data['files']:
                        print(f"      - {file_info['name'][:50]:50s} ({file_info['age_hours']:5.1f}h, {file_info['size_mb']:6.1f} MB)")

            time.sleep(10)
    else:
        health = monitor.analyze_queue_health()
        print(json.dumps(health, indent=2))


if __name__ == '__main__':
    main()
