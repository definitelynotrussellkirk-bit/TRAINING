#!/usr/bin/env python3
"""
Preview History Logger

Logs all preview results to JSONL for:
- Historical analysis
- Pattern tracking
- Failure investigation
- UI pagination

File: logs/preview_history.jsonl
Format: One JSON object per line (JSONL)
Rotation: Daily (keeps last 30 days)
"""

import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import asdict

from view_models import PreviewHistoryEntry


class PreviewHistoryLogger:
    """
    Append-only logger for preview results
    """

    def __init__(self, base_dir: Path = Path("/path/to/training")):
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs" / "preview"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Current log file (rotates daily)
        self.current_log_file = self._get_log_file_for_date(datetime.now())

    def _get_log_file_for_date(self, date: datetime) -> Path:
        """Get log file path for a given date"""
        date_str = date.strftime("%Y-%m-%d")
        return self.logs_dir / f"preview_history_{date_str}.jsonl"

    def log_preview(self, entry: PreviewHistoryEntry):
        """
        Log a preview result

        Thread-safe: uses append mode
        """
        # Rotate if date changed
        today_file = self._get_log_file_for_date(datetime.now())
        if today_file != self.current_log_file:
            self.current_log_file = today_file

        # Append to current log
        with open(self.current_log_file, 'a') as f:
            json.dump(asdict(entry), f)
            f.write('\n')

    def log_preview_dict(self, preview_dict: Dict):
        """Log from a dictionary (convenience method)"""
        entry = PreviewHistoryEntry(**preview_dict)
        self.log_preview(entry)

    def get_recent_previews(
        self,
        limit: int = 100,
        offset: int = 0,
        since: Optional[datetime] = None,
        regime: Optional[str] = None,
        source_pool: Optional[str] = None,
        exact_match: Optional[bool] = None
    ) -> List[Dict]:
        """
        Get recent preview results with pagination and filtering

        Args:
            limit: Maximum number of results
            offset: Skip first N results
            since: Only return previews after this timestamp
            regime: Filter by regime (emoji_think, regime3, etc.)
            source_pool: Filter by source (fixed_eval, train, failures)
            exact_match: Filter by match status (True/False/None=all)

        Returns:
            List of preview dicts (newest first)
        """
        previews = []

        # Determine which log files to read
        log_files = self._get_recent_log_files(since)

        # Read in reverse chronological order
        for log_file in reversed(log_files):
            if len(previews) >= limit + offset:
                break

            try:
                for entry in self._read_log_file_reversed(log_file):
                    # Apply filters
                    if since and entry.get("ts", "") < since.isoformat():
                        continue

                    if regime and entry.get("regime") != regime:
                        continue

                    if source_pool and entry.get("source_pool") != source_pool:
                        continue

                    if exact_match is not None and entry.get("exact_match") != exact_match:
                        continue

                    previews.append(entry)

                    if len(previews) >= limit + offset:
                        break
            except Exception as e:
                print(f"Warning: Failed to read {log_file}: {e}")
                continue

        # Apply offset and limit
        return previews[offset:offset + limit]

    def _get_recent_log_files(self, since: Optional[datetime] = None) -> List[Path]:
        """Get list of log files to read (most recent 30 days)"""
        if since is None:
            since = datetime.now() - timedelta(days=30)

        log_files = []
        current_date = datetime.now()

        # Go back up to 30 days or until 'since' date
        for days_back in range(31):
            date = current_date - timedelta(days=days_back)
            if date < since:
                break

            log_file = self._get_log_file_for_date(date)
            if log_file.exists():
                log_files.append(log_file)

        return log_files

    def _read_log_file_reversed(self, log_file: Path) -> Iterator[Dict]:
        """
        Read JSONL file in reverse order (newest first)

        For large files, this is memory-efficient
        """
        # For simplicity, read entire file and reverse
        # TODO: For very large files, use a more sophisticated approach
        lines = []

        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Parse and yield in reverse
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def get_preview_stats(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Get aggregated preview statistics

        Args:
            since: Only count previews after this timestamp
            limit: Only use last N previews

        Returns:
            Statistics dict with EM rates, regime breakdown, etc.
        """
        previews = self.get_recent_previews(
            limit=limit or 1000,
            since=since
        )

        if not previews:
            return {
                "total": 0,
                "exact_match_rate": 0.0,
                "by_regime": {},
                "by_source": {},
                "by_failure_mode": {},
            }

        # Overall stats
        total = len(previews)
        exact_matches = sum(1 for p in previews if p.get("exact_match"))

        # By regime
        by_regime = {}
        for p in previews:
            regime = p.get("regime", "unknown")
            if regime not in by_regime:
                by_regime[regime] = {"total": 0, "em": 0}
            by_regime[regime]["total"] += 1
            if p.get("exact_match"):
                by_regime[regime]["em"] += 1

        # Calculate rates
        for regime in by_regime:
            by_regime[regime]["em_rate"] = by_regime[regime]["em"] / by_regime[regime]["total"]

        # By source pool
        by_source = {}
        for p in previews:
            source = p.get("source_pool", "unknown")
            if source not in by_source:
                by_source[source] = {"total": 0, "em": 0}
            by_source[source]["total"] += 1
            if p.get("exact_match"):
                by_source[source]["em"] += 1

        for source in by_source:
            by_source[source]["em_rate"] = by_source[source]["em"] / by_source[source]["total"]

        # By failure mode
        by_failure = {}
        for p in previews:
            if not p.get("exact_match"):
                failure = p.get("failure_mode", "unknown")
                by_failure[failure] = by_failure.get(failure, 0) + 1

        return {
            "total": total,
            "exact_match_rate": exact_matches / total,
            "by_regime": by_regime,
            "by_source": by_source,
            "by_failure_mode": by_failure,
        }

    def rotate_old_logs(self, keep_days: int = 30):
        """
        Compress and archive logs older than keep_days
        """
        cutoff_date = datetime.now() - timedelta(days=keep_days)

        for log_file in self.logs_dir.glob("preview_history_*.jsonl"):
            # Extract date from filename
            try:
                date_str = log_file.stem.replace("preview_history_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    # Compress
                    gz_file = log_file.with_suffix('.jsonl.gz')
                    if not gz_file.exists():
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(gz_file, 'wb') as f_out:
                                f_out.writelines(f_in)
                        print(f"Compressed {log_file} -> {gz_file}")

                    # Delete original
                    log_file.unlink()
                    print(f"Deleted old log: {log_file}")

            except Exception as e:
                print(f"Warning: Failed to process {log_file}: {e}")


# Singleton instance
_logger_instance = None


def get_preview_logger(base_dir: Path = Path("/path/to/training")) -> PreviewHistoryLogger:
    """Get or create singleton preview logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PreviewHistoryLogger(base_dir)
    return _logger_instance


if __name__ == '__main__':
    # Test preview history logger
    logger = get_preview_logger()

    # Log some test previews
    for i in range(5):
        entry = PreviewHistoryEntry(
            ts=datetime.now().isoformat(),
            step=1000 + i * 100,
            checkpoint_id=f"checkpoint-{1000 + i * 100}",
            example_id=f"test_{i}",
            dataset_id="test_dataset",
            regime="emoji_think",
            source_pool="fixed_eval",
            prompt="What is 2+2?",
            golden="4",
            model_answer="4" if i % 2 == 0 else "5",
            exact_match=(i % 2 == 0),
            normalized_match=(i % 2 == 0),
            failure_mode=None if i % 2 == 0 else "wrong_answer",
            ce=0.1 if i % 2 == 0 else 0.8,
            prompt_tokens=10,
            golden_tokens=2,
            model_tokens=2,
            generation_time_ms=100,
        )
        logger.log_preview(entry)

    print("Logged 5 test previews")

    # Get recent previews
    recent = logger.get_recent_previews(limit=3)
    print(f"\nRecent {len(recent)} previews:")
    for p in recent:
        print(f"  Step {p['step']}: EM={p['exact_match']}")

    # Get stats
    stats = logger.get_preview_stats(limit=100)
    print(f"\nStats:")
    print(f"  Total: {stats['total']}")
    print(f"  EM rate: {stats['exact_match_rate']:.1%}")
    print(f"  By regime: {stats['by_regime']}")
