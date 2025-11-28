#!/usr/bin/env python3
"""
Hardcode Audit - Find hardcoded paths that would break on another machine.

This script scans the codebase for:
1. /home/<user>/ paths (Linux/Mac)
2. C:\\Users\\<user>\\ paths (Windows)
3. Hardcoded IP addresses (optional)

Usage:
    python3 scripts/check_hardcodes.py              # Check all
    python3 scripts/check_hardcodes.py --strict     # Fail on any hardcode
    python3 scripts/check_hardcodes.py --show-lines # Show matching lines
    python3 scripts/check_hardcodes.py --ignore-ips # Don't flag IPs
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set
from collections import defaultdict


# Patterns to detect
HARDCODE_PATTERNS = [
    (re.compile(r'/home/[^/\s"\']+/'), "Linux home path"),
    (re.compile(r'/Users/[^/\s"\']+/'), "Mac home path"),
    (re.compile(r'C:\\Users\\[^\\]+\\'), "Windows home path"),
]

IP_PATTERN = re.compile(r'\b(?:192\.168|10\.|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b')

# File extensions to check
SCAN_EXTENSIONS = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}

# Directories to skip
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "archive",
    ".mypy_cache",
    ".pytest_cache",
    "outputs",
    "backups",
    "logs",
}

# Files to skip (common false positives)
SKIP_FILES = {
    "CLAUDE.md",  # Documentation with examples
    "CHANGELOG.md",
    "README.md",
    "ARCHITECTURE.md",
}


def get_base_dir() -> Path:
    """Get base directory."""
    script_path = Path(__file__).resolve()
    return script_path.parent.parent


def should_skip(path: Path, base_dir: Path) -> bool:
    """Check if a path should be skipped."""
    rel_path = path.relative_to(base_dir)

    # Skip directories
    for part in rel_path.parts:
        if part in SKIP_DIRS:
            return True

    # Skip by extension
    if path.suffix not in SCAN_EXTENSIONS:
        return True

    # Skip specific files
    if path.name in SKIP_FILES:
        return True

    return False


def scan_file(path: Path, check_ips: bool = True) -> List[Tuple[int, str, str]]:
    """
    Scan a file for hardcoded paths.

    Returns list of (line_number, pattern_type, matching_text)
    """
    matches = []

    try:
        content = path.read_text(errors="ignore")
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check hardcode patterns
            for pattern, pattern_type in HARDCODE_PATTERNS:
                match = pattern.search(line)
                if match:
                    matches.append((line_num, pattern_type, match.group()))

            # Check IP patterns (optional)
            if check_ips:
                for match in IP_PATTERN.finditer(line):
                    matches.append((line_num, "Private IP", match.group()))

    except Exception as e:
        pass

    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Check codebase for hardcoded paths"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any hardcodes found"
    )
    parser.add_argument(
        "--show-lines",
        action="store_true",
        help="Show the matching lines"
    )
    parser.add_argument(
        "--ignore-ips",
        action="store_true",
        help="Don't flag private IP addresses"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, not file details"
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    check_ips = not args.ignore_ips

    print("=" * 60)
    print("HARDCODE AUDIT")
    print("=" * 60)
    print(f"Scanning: {base_dir}")
    print(f"Checking IPs: {check_ips}")
    print()

    # Scan all files
    all_matches = defaultdict(list)
    file_count = 0

    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path, base_dir):
            continue

        file_count += 1
        matches = scan_file(path, check_ips)

        if matches:
            rel_path = path.relative_to(base_dir)
            all_matches[rel_path].extend(matches)

    # Group by type
    by_type = defaultdict(set)
    for file_path, matches in all_matches.items():
        for line_num, pattern_type, text in matches:
            by_type[pattern_type].add(file_path)

    # Report
    if not args.summary_only:
        print(f"[Files with hardcodes: {len(all_matches)} / {file_count} scanned]")
        print()

        for file_path in sorted(all_matches.keys()):
            matches = all_matches[file_path]
            print(f"\n{file_path}:")

            for line_num, pattern_type, text in matches:
                if args.show_lines:
                    print(f"  L{line_num}: [{pattern_type}] {text}")
                else:
                    print(f"  L{line_num}: {pattern_type}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files scanned: {file_count}")
    print(f"Files with hardcodes: {len(all_matches)}")
    print()
    print("By type:")
    for pattern_type, files in sorted(by_type.items()):
        print(f"  {pattern_type}: {len(files)} files")

    # Exit code
    if args.strict and all_matches:
        print()
        print("FAILED: Hardcoded paths found (--strict mode)")
        return 1

    if all_matches:
        print()
        print("WARNING: Hardcoded paths found. Consider using environment variables.")
        return 0

    print()
    print("PASSED: No hardcoded paths found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
