#!/usr/bin/env python3
"""
Master Monitoring System - Endpoint Testing Suite
Phase 1, Task 1.2: Test and verify all endpoints

Tests all data sources discovered in Task 1.1
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import requests

from core.paths import get_base_dir, get_external_tool_path

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class EndpointTester:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        self.base_dir = get_base_dir()

    def test(self, name, func):
        """Run a test and record result"""
        try:
            result = func()
            if result['status'] == 'pass':
                self.results['passed'].append((name, result))
                print(f"{GREEN}✓{RESET} {name}")
                if result.get('data'):
                    print(f"  → {result['data']}")
            elif result['status'] == 'warn':
                self.results['warnings'].append((name, result))
                print(f"{YELLOW}⚠{RESET} {name}")
                print(f"  → {result['message']}")
            else:
                self.results['failed'].append((name, result))
                print(f"{RED}✗{RESET} {name}")
                print(f"  → {result['error']}")
        except Exception as e:
            self.results['failed'].append((name, {'error': str(e)}))
            print(f"{RED}✗{RESET} {name}")
            print(f"  → Exception: {str(e)}")

    def print_summary(self):
        """Print test summary"""
        total = len(self.results['passed']) + len(self.results['failed']) + len(self.results['warnings'])
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}TEST SUMMARY{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"{GREEN}✓ Passed:{RESET} {len(self.results['passed'])}/{total}")
        print(f"{RED}✗ Failed:{RESET} {len(self.results['failed'])}/{total}")
        print(f"{YELLOW}⚠ Warnings:{RESET} {len(self.results['warnings'])}/{total}")

        if self.results['failed']:
            print(f"\n{RED}Failed Tests:{RESET}")
            for name, result in self.results['failed']:
                print(f"  - {name}: {result.get('error', 'Unknown error')}")

    def save_report(self):
        """Save test report to JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': len(self.results['passed']) + len(self.results['failed']) + len(self.results['warnings']),
                'passed': len(self.results['passed']),
                'failed': len(self.results['failed']),
                'warnings': len(self.results['warnings'])
            },
            'results': self.results
        }

        report_file = self.base_dir / 'monitoring' / 'tests' / 'endpoint_test_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {report_file}")


def main():
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}MASTER MONITORING SYSTEM - ENDPOINT TESTS{RESET}")
    print(f"{BLUE}Phase 1, Task 1.2: Verify All Data Sources{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    tester = EndpointTester()

    # ==================== LOCAL STATUS FILES (4090) ====================
    print(f"\n{BLUE}[1] LOCAL STATUS FILES (4090){RESET}")

    base = tester.base_dir

    tester.test("training_status.json", lambda: test_json_file(
        base / "status/training_status.json",
        required_fields=['status', 'current_step', 'total_steps', 'loss']
    ))

    tester.test("latest_preview.json", lambda: test_json_file(
        base / "status/latest_preview.json",
        required_fields=['step']
    ))

    tester.test("curriculum_state.json", lambda: test_json_file(
        base / "status/curriculum_state.json"
    ))

    tester.test("3090_watchdog_status.json", lambda: test_json_file(
        base / "status/3090_watchdog_status.json"
    ))

    # ==================== REMOTE STATUS FILES (3090) ====================
    print(f"\n{BLUE}[2] REMOTE STATUS FILES (3090){RESET}")

    tester.test("curriculum_optimization.json (3090)", lambda: test_remote_json(
        'inference.local',
        '~/TRAINING/status/curriculum_optimization.json',
        required_fields=['evaluations']
    ))

    tester.test("model_comparisons.json (3090)", lambda: test_remote_json(
        'inference.local',
        '~/TRAINING/status/model_comparisons.json'
    ))

    tester.test("regression_monitoring.json (3090)", lambda: test_remote_json(
        'inference.local',
        '~/TRAINING/status/regression_monitoring.json'
    ))

    tester.test("confidence_calibration.json (3090)", lambda: test_remote_json(
        'inference.local',
        '~/TRAINING/status/confidence_calibration.json'
    ))

    tester.test("adversarial_mining.json (3090)", lambda: test_remote_json(
        'inference.local',
        '~/TRAINING/status/adversarial_mining.json'
    ))

    tester.test("checkpoint_sync.json (3090)", lambda: test_remote_json(
        'inference.local',
        '~/TRAINING/status/checkpoint_sync.json'
    ))

    # ==================== API ENDPOINTS ====================
    print(f"\n{BLUE}[3] API ENDPOINTS (Port 8080){RESET}")

    tester.test("/api/status/live", lambda: test_api_endpoint(
        'http://localhost:8080/api/status/live',
        required_fields=['status', 'current_step']
    ))

    tester.test("/api/status/system", lambda: test_api_endpoint(
        'http://localhost:8080/api/status/system'
    ))

    tester.test("/api/status/preview", lambda: test_api_endpoint(
        'http://localhost:8080/api/status/preview'
    ))

    tester.test("/api/status/evals", lambda: test_api_endpoint(
        'http://localhost:8080/api/status/evals'
    ))

    tester.test("/api/config", lambda: test_api_endpoint(
        'http://localhost:8080/api/config'
    ))

    # ==================== GPU STATS ====================
    print(f"\n{BLUE}[4] GPU STATS{RESET}")

    tester.test("nvidia-smi (4090)", lambda: test_nvidia_smi('local'))
    tester.test("nvidia-smi (3090)", lambda: test_nvidia_smi('inference.local'))

    # ==================== MONITORING PROCESSES ====================
    print(f"\n{BLUE}[5] MONITORING PROCESSES{RESET}")

    tester.test("Local monitoring processes (4090)", lambda: test_local_processes())
    tester.test("Remote monitoring processes (3090)", lambda: test_remote_processes())

    # ==================== QUEUE SYSTEM ====================
    print(f"\n{BLUE}[6] QUEUE SYSTEM{RESET}")

    tester.test("Queue directories accessible", lambda: test_queue_access())

    # ==================== SKILL FOLDERS ====================
    print(f"\n{BLUE}[7] SKILL FOLDERS{RESET}")

    tester.test("Skill folder access", lambda: test_skill_folders())

    # ==================== SUMMARY ====================
    tester.print_summary()
    tester.save_report()

    # Exit code
    sys.exit(0 if len(tester.results['failed']) == 0 else 1)


# ==================== TEST FUNCTIONS ====================

def test_json_file(filepath, required_fields=None):
    """Test if JSON file exists and is valid"""
    if not filepath.exists():
        return {'status': 'fail', 'error': f'File not found: {filepath}'}

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if required_fields:
            missing = [f for f in required_fields if f not in data]
            if missing:
                return {'status': 'warn', 'message': f'Missing fields: {missing}'}

        size = filepath.stat().st_size
        return {'status': 'pass', 'data': f'{len(data)} keys, {size} bytes'}
    except json.JSONDecodeError as e:
        return {'status': 'fail', 'error': f'Invalid JSON: {e}'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_remote_json(host, filepath, required_fields=None):
    """Test remote JSON file via SSH"""
    try:
        result = subprocess.run(
            ['ssh', host, f'cat {filepath}'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {'status': 'fail', 'error': f'SSH failed: {result.stderr[:100]}'}

        data = json.loads(result.stdout)

        if required_fields:
            missing = [f for f in required_fields if f not in data]
            if missing:
                return {'status': 'warn', 'message': f'Missing fields: {missing}'}

        return {'status': 'pass', 'data': f'{len(data)} keys, {len(result.stdout)} bytes'}
    except json.JSONDecodeError:
        return {'status': 'fail', 'error': 'Invalid JSON'}
    except subprocess.TimeoutExpired:
        return {'status': 'fail', 'error': 'SSH timeout'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_api_endpoint(url, required_fields=None):
    """Test HTTP API endpoint"""
    try:
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return {'status': 'fail', 'error': f'HTTP {response.status_code}'}

        data = response.json()

        if 'error' in data:
            return {'status': 'warn', 'message': f'API error: {data["error"]}'}

        if required_fields:
            missing = [f for f in required_fields if f not in data]
            if missing:
                return {'status': 'warn', 'message': f'Missing fields: {missing}'}

        return {'status': 'pass', 'data': f'{len(data)} keys'}
    except requests.exceptions.ConnectionError:
        return {'status': 'fail', 'error': 'Connection refused'}
    except requests.exceptions.Timeout:
        return {'status': 'fail', 'error': 'Timeout'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_nvidia_smi(host):
    """Test nvidia-smi access"""
    try:
        if host == 'local':
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
        else:
            result = subprocess.run(
                ['ssh', host, 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

        if result.returncode != 0:
            return {'status': 'fail', 'error': result.stderr[:100]}

        gpu_info = result.stdout.strip()
        return {'status': 'pass', 'data': gpu_info}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_local_processes():
    """Test local monitoring processes"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )

        monitoring_procs = [line for line in result.stdout.split('\n')
                          if 'python3' in line and 'monitoring' in line and 'grep' not in line]

        count = len(monitoring_procs)
        return {'status': 'pass', 'data': f'{count} processes running'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_remote_processes():
    """Test remote monitoring processes"""
    try:
        result = subprocess.run(
            ['ssh', 'inference.local', 'ps aux | grep python3 | grep monitoring | grep -v grep | wc -l'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {'status': 'fail', 'error': 'SSH failed'}

        count = int(result.stdout.strip())
        return {'status': 'pass', 'data': f'{count} processes running'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_queue_access():
    """Test queue directory access"""
    try:
        from core.paths import get_queue_dir
        queue_dir = get_queue_dir()
        if not queue_dir.exists():
            return {'status': 'fail', 'error': 'Queue directory not found'}

        subdirs = [d.name for d in queue_dir.iterdir() if d.is_dir()]
        file_count = sum(1 for d in queue_dir.iterdir() if d.is_dir()
                        for f in d.iterdir() if f.is_file())

        return {'status': 'pass', 'data': f'{len(subdirs)} dirs, {file_count} files'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


def test_skill_folders():
    """Test skill folder access"""
    try:
        skill_dir = get_external_tool_path("singleSKILL")
        if not skill_dir.exists():
            return {'status': 'fail', 'error': 'Skill directory not found'}

        skills = [d.name for d in skill_dir.iterdir()
                 if d.is_dir() and d.name.startswith('skill_')]

        return {'status': 'pass', 'data': f'{len(skills)} skills found'}
    except RuntimeError as e:
        # External tool not configured - skip gracefully
        return {'status': 'warn', 'message': f'singleSKILL not configured: {e}'}
    except Exception as e:
        return {'status': 'fail', 'error': str(e)}


if __name__ == '__main__':
    main()
