#!/usr/bin/env python3
"""
Atomic Operations Utility
Provides atomic file operations to prevent corruption from partial writes
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict


def write_json_atomic(data: Dict[str, Any], path: Path) -> None:
    """
    Write JSON file atomically to prevent corruption from partial writes

    Uses write-to-temp-then-rename pattern which is atomic on POSIX systems.
    """
    path = Path(path)

    # Write to temporary file in same directory (same filesystem required for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
        suffix=".json"
    )

    try:
        # Write JSON to temp file
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2)

        # Atomic rename (only succeeds if write completed)
        os.rename(temp_path, path)

    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise


def write_text_atomic(content: str, path: Path) -> None:
    """
    Write text file atomically
    """
    path = Path(path)

    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
        suffix=".txt"
    )

    try:
        with os.fdopen(temp_fd, 'w') as f:
            f.write(content)

        os.rename(temp_path, path)

    except Exception:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise


def copy_directory_atomic(src: Path, dest: Path) -> None:
    """
    Copy directory atomically using temp-then-rename pattern

    Useful for checkpoint/snapshot operations
    """
    src = Path(src)
    dest = Path(dest)

    # Create temp directory in same parent (same filesystem)
    temp_dir = dest.parent / f".{dest.name}.tmp.{os.getpid()}"

    try:
        # Copy to temp directory
        shutil.copytree(src, temp_dir)

        # Atomic rename
        os.rename(temp_dir, dest)

    except Exception:
        # Clean up temp directory on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def safe_file_operation(func):
    """
    Decorator to wrap file operations with TOCTOU protection

    Instead of:
        if file.exists():
            with open(file) as f:
                ...

    Use:
        @safe_file_operation
        def read_file(path):
            with open(path) as f:
                return f.read()
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError:
            # File disappeared between check and use - handle gracefully
            return None
        except PermissionError:
            # Permission denied
            return None
    return wrapper
