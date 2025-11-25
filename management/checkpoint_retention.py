"""Checkpoint retention utilities.

DEPRECATED: This module is deprecated. Use retention_service.RetentionService instead.
The new system uses retention_manager.RetentionManager with:
- 36-hour minimum age rule
- 150GB total limit
- Protection for latest, best, today, yesterday

Legacy behavior (kept for backward compatibility):
Enforces a two-tier limit:
- Recent: keep newest checkpoints up to a size budget (default 100 GB).
- Historic: from the remaining, keep at most one per day until a second
  size budget is hit (default 150 GB).

Anything outside those budgets is deleted.
"""

from __future__ import annotations

import shutil
import warnings
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


KB = 1024
MB = 1024 * KB
GB = 1024 * MB


@dataclass(eq=True, frozen=True)
class Checkpoint:
    path: Path
    size_bytes: int
    mtime: float

    @property
    def day(self) -> str:
        """Date string YYYY-MM-DD derived from mtime."""
        return time.strftime("%Y-%m-%d", time.localtime(self.mtime))


def _safe_du_bytes(path: Path) -> int:
    """Use du -sb for accurate on-disk size; fall back to Python walk."""
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0]
        return int(out)
    except Exception:
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    continue
        return total


def find_checkpoints(dirs: Iterable[Path]) -> List[Checkpoint]:
    """Locate checkpoint-* directories under the provided roots."""
    found: List[Checkpoint] = []
    for root in dirs:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for candidate in root.rglob("checkpoint-*"):
            if not candidate.is_dir():
                continue
            try:
                size = _safe_du_bytes(candidate)
                mtime = candidate.stat().st_mtime
                found.append(Checkpoint(candidate, size, mtime))
            except OSError:
                continue
    return found


def plan_retention(
    checkpoints: Sequence[Checkpoint],
    recent_limit_gb: int = 100,
    historic_limit_gb: int = 150,
    must_keep: Optional[Sequence[Checkpoint]] = None,
) -> dict:
    """Return a plan with keep/delete lists respecting tiered size caps."""
    recent_limit = recent_limit_gb * GB
    historic_limit = historic_limit_gb * GB

    sorted_cps = sorted(checkpoints, key=lambda c: c.mtime, reverse=True)

    keep_recent: List[Checkpoint] = []
    recent_bytes = 0

    for cp in sorted_cps:
        if recent_bytes + cp.size_bytes <= recent_limit or not keep_recent:
            keep_recent.append(cp)
            recent_bytes += cp.size_bytes
        else:
            break

    remaining = [cp for cp in sorted_cps if cp not in keep_recent]

    keep_historic: List[Checkpoint] = []
    historic_bytes = 0
    seen_days = set()

    for cp in remaining:
        if cp.day in seen_days:
            continue
        if historic_bytes + cp.size_bytes > historic_limit and keep_historic:
            continue
        keep_historic.append(cp)
        historic_bytes += cp.size_bytes
        seen_days.add(cp.day)
        if historic_bytes >= historic_limit:
            break

    keep_set = set(keep_recent + keep_historic)

    # Ensure required checkpoints are retained (e.g., latest per root)
    if must_keep:
        for cp in must_keep:
            if cp not in keep_set:
                keep_historic.append(cp)
                historic_bytes += cp.size_bytes
                keep_set.add(cp)

    delete_list = [cp for cp in sorted_cps if cp not in keep_set]

    return {
        "recent": keep_recent,
        "historic": keep_historic,
        "delete": delete_list,
        "recent_bytes": recent_bytes,
        "historic_bytes": historic_bytes,
        "total_bytes": recent_bytes + historic_bytes,
    }


def _fmt_gb(num_bytes: float) -> str:
    return f"{num_bytes / GB:.1f} GB"


def enforce_retention(
    dirs: Iterable[Path],
    recent_limit_gb: int = 100,
    historic_limit_gb: int = 150,
    logger=None,
    dry_run: bool = False,
) -> dict:
    """Compute and apply a retention plan. Deletes old checkpoints when needed.

    DEPRECATED: Use retention_service.RetentionService instead.
    """
    warnings.warn(
        "checkpoint_retention.enforce_retention is deprecated. "
        "Use retention_service.RetentionService instead.",
        DeprecationWarning,
        stacklevel=2
    )
    checkpoints = find_checkpoints(dirs)

    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    must_keep = []
    for root in dirs:
        root_path = Path(root)
        cands = [cp for cp in checkpoints if _is_relative_to(cp.path, root_path)]
        if cands:
            latest = max(cands, key=lambda cp: cp.mtime)
            must_keep.append(latest)

    plan = plan_retention(
        checkpoints,
        recent_limit_gb,
        historic_limit_gb,
        must_keep=must_keep,
    )

    if logger:
        logger.info(
            f"Checkpoint retention: keeping {_fmt_gb(plan['recent_bytes'])} recent, "
            f"{_fmt_gb(plan['historic_bytes'])} historic; "
            f"deleting {len(plan['delete'])} checkpoints totalling "
            f"{_fmt_gb(sum(cp.size_bytes for cp in plan['delete']))}"
        )

    if not dry_run:
        for cp in plan["delete"]:
            try:
                shutil.rmtree(cp.path)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to delete {cp.path}: {e}")
                else:
                    print(f"Failed to delete {cp.path}: {e}")

    return plan


if __name__ == "__main__":
    roots = [
        Path("current_model"),
        Path("current_model_small"),
        Path("snapshots"),
    ]
    summary = enforce_retention(roots, dry_run=False)
    print(
        f"Recent: {len(summary['recent'])}, "
        f"Historic: {len(summary['historic'])}, "
        f"Deleted: {len(summary['delete'])}"
    )
