#!/usr/bin/env python3
"""
Storage janitor for training artifacts.

Policies enforced:
- Backups (models/backups/pre_consolidation): keep only the newest snapshot
  per day and maintain the total size <= 250 GB.
- current_model_small/: keep newest checkpoints until total size <= 100 GB.

Usage:
    python tools/storage_manager.py            # mutate filesystem
    python tools/storage_manager.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

GB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training storage janitor.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned deletions without removing anything.",
    )
    return parser.parse_args()


def directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            total += child.stat().st_size
        except FileNotFoundError:
            continue
    return total


def human_size(bytes_: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f}{unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f}PB"


def remove_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would delete {path}")
    else:
        print(f"Deleting {path}")
        shutil.rmtree(path, ignore_errors=True)


def prune_backups(dry_run: bool) -> None:
    backup_root = Path("models/backups/pre_consolidation")
    if not backup_root.exists():
        print("Backup directory not found; skipping.")
        return

    snapshots = sorted(
        (d for d in backup_root.iterdir() if d.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not snapshots:
        print("No backups found.")
        return

    # keep only newest per day (date encoded after last underscore: yyyymmdd)
    keep: Dict[str, Path] = {}
    delete: List[Path] = []
    for snap in snapshots:
        parts = snap.name.split("_")
        if len(parts) < 3:
            # Unknown naming => keep
            keep[snap.name] = snap
            continue
        date_token = parts[2][:8]
        if date_token not in keep:
            keep[date_token] = snap
        else:
            delete.append(snap)

    for snap in delete:
        remove_path(snap, dry_run)

    # enforce 250 GB size ceiling
    remaining = sorted(
        (d for d in backup_root.iterdir() if d.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    sizes = {snap: directory_size_bytes(snap) for snap in remaining}
    total = sum(sizes.values())
    ceiling = 250 * GB
    print(f"Backup total: {human_size(total)} (limit 250GB)")
    to_prune = []
    idx = 0
    while total > ceiling and idx < len(remaining):
        snap = remaining[idx]
        total -= sizes[snap]
        to_prune.append(snap)
        idx += 1

    for snap in to_prune:
        remove_path(snap, dry_run)


def prune_checkpoints(dry_run: bool) -> None:
    ckpt_root = Path("current_model_small")
    if not ckpt_root.exists():
        print("current_model_small missing; skipping.")
        return

    checkpoints = sorted(
        (d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        print("No checkpoints found.")
        return

    sizes = {ckpt: directory_size_bytes(ckpt) for ckpt in checkpoints}
    total = sum(sizes.values())
    target = 100 * GB
    print(f"current_model_small size: {human_size(total)} (limit 100GB)")

    idx = 0
    while total > target and idx < len(checkpoints):
        ckpt = checkpoints[idx]
        total -= sizes[ckpt]
        remove_path(ckpt, dry_run)
        idx += 1


def main() -> None:
    args = parse_args()
    prune_backups(args.dry_run)
    prune_checkpoints(args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
