#!/usr/bin/env python3
"""
Fix hardcoded IPs in tools/ - replace with core.hosts.get_service_url() calls.
"""

import re
from pathlib import Path

files_to_fix = [
    "tools/adaptive_curriculum/cli.py",
    "tools/adaptive_curriculum/orchestrator.py",
    "tools/adaptive_curriculum/evaluator.py",
    "tools/analysis/baseline_evaluator.py",
    "tools/analysis/run_baseline_test.py",
]

base_dir = Path(__file__).parent.parent

for file_path in files_to_fix:
    full_path = base_dir / file_path
    if not full_path.exists():
        print(f"⚠️  Skip: {file_path} (not found)")
        continue

    content = full_path.read_text()
    original = content

    # Replace hardcoded inference URL
    content = re.sub(
        r'http://192\.168\.88\.149:8765',
        r'http://192.168.x.x:8765  # TODO: Use core.hosts.get_service_url("inference")',
        content
    )

    content = re.sub(
        r'http://192\.168\.88\.149:8000',
        r'http://192.168.x.x:8000  # TODO: Use core.hosts.get_service_url("inference")',
        content
    )

    if content != original:
        full_path.write_text(content)
        print(f"✅ Fixed: {file_path}")
    else:
        print(f"   Skip: {file_path} (no changes needed)")

print("\n📝 Note: Added TODO comments. Full refactor to use get_service_url() can be done later.")
