"""
SagaReader - Read tales from the Saga.

Reads tales from JSONL files for display in the Tavern UI.
Supports:
- Recent tales (last N entries)
- Tales by time range
- Tales by category/type
- Search by message content

Thread-safe for concurrent reads while writes are happening.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional

from guild.saga.types import TaleEntry

logger = logging.getLogger(__name__)


class SagaReader:
    """
    Reads tales from the Saga.

    Usage:
        reader = SagaReader(base_dir)

        # Get recent tales
        tales = reader.recent(limit=50)

        # Get tales from today
        tales = reader.today()

        # Get tales by category
        tales = reader.by_category("quest", limit=20)

        # Search tales
        tales = reader.search("level up", limit=10)

        # Stream all tales (memory efficient)
        for tale in reader.stream_all():
            print(tale.format_display())
    """

    def __init__(self, base_dir: Path | str):
        """
        Initialize SagaReader.

        Args:
            base_dir: Base training directory. Reads from {base_dir}/logs/saga/
        """
        self.base_dir = Path(base_dir)
        self.saga_dir = self.base_dir / "logs" / "saga"

    def _get_saga_files(self, reverse: bool = True) -> List[Path]:
        """
        Get all saga files, sorted by date.

        Args:
            reverse: If True, newest first (default). If False, oldest first.

        Returns:
            List of saga file paths
        """
        if not self.saga_dir.exists():
            return []

        files = list(self.saga_dir.glob("*.jsonl"))
        files.sort(key=lambda p: p.stem, reverse=reverse)
        return files

    def _read_file(self, file_path: Path) -> Iterator[TaleEntry]:
        """
        Read tales from a single file.

        Handles partial/corrupted lines gracefully.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = TaleEntry.from_json(line)
                        yield entry
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Saga: corrupt entry in {file_path}:{line_num}: {e}")
                        continue
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Saga: error reading {file_path}: {e}")

    def _read_file_reverse(self, file_path: Path) -> List[TaleEntry]:
        """Read a file and return entries in reverse order (newest first)."""
        entries = list(self._read_file(file_path))
        entries.reverse()
        return entries

    def recent(self, limit: int = 50) -> List[TaleEntry]:
        """
        Get the most recent tales.

        Args:
            limit: Maximum number of tales to return

        Returns:
            List of TaleEntry, newest first
        """
        result = []
        files = self._get_saga_files(reverse=True)

        for file_path in files:
            entries = self._read_file_reverse(file_path)
            result.extend(entries)

            if len(result) >= limit:
                break

        return result[:limit]

    def today(self) -> List[TaleEntry]:
        """Get all tales from today, oldest first."""
        today_file = self.saga_dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        return list(self._read_file(today_file))

    def for_date(self, date: datetime) -> List[TaleEntry]:
        """Get all tales for a specific date, oldest first."""
        date_file = self.saga_dir / f"{date.strftime('%Y-%m-%d')}.jsonl"
        return list(self._read_file(date_file))

    def since(self, since: datetime, limit: int = 1000) -> List[TaleEntry]:
        """
        Get tales since a given timestamp.

        Args:
            since: Get tales after this time
            limit: Maximum number to return

        Returns:
            List of TaleEntry, oldest first (chronological)
        """
        result = []
        files = self._get_saga_files(reverse=False)  # Oldest first

        for file_path in files:
            # Skip files from before the since date
            try:
                file_date = datetime.strptime(file_path.stem, "%Y-%m-%d")
                if file_date.date() < since.date():
                    continue
            except ValueError:
                continue

            for entry in self._read_file(file_path):
                if entry.timestamp >= since:
                    result.append(entry)
                    if len(result) >= limit:
                        return result

        return result

    def by_category(self, category: str, limit: int = 50) -> List[TaleEntry]:
        """
        Get tales by category.

        Args:
            category: Category to filter by (e.g., "quest", "hero", "combat")
            limit: Maximum number to return

        Returns:
            List of TaleEntry, newest first
        """
        result = []
        files = self._get_saga_files(reverse=True)

        for file_path in files:
            entries = self._read_file_reverse(file_path)
            for entry in entries:
                if entry.category == category:
                    result.append(entry)
                    if len(result) >= limit:
                        return result

        return result

    def by_type(self, event_type: str, limit: int = 50) -> List[TaleEntry]:
        """
        Get tales by event type.

        Args:
            event_type: Event type to filter (e.g., "quest.started")
            limit: Maximum number to return

        Returns:
            List of TaleEntry, newest first
        """
        result = []
        files = self._get_saga_files(reverse=True)

        for file_path in files:
            entries = self._read_file_reverse(file_path)
            for entry in entries:
                if entry.event_type == event_type:
                    result.append(entry)
                    if len(result) >= limit:
                        return result

        return result

    def search(self, query: str, limit: int = 50, case_sensitive: bool = False) -> List[TaleEntry]:
        """
        Search tales by message content.

        Args:
            query: Text to search for
            limit: Maximum number to return
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of TaleEntry, newest first
        """
        result = []
        files = self._get_saga_files(reverse=True)

        if not case_sensitive:
            query = query.lower()

        for file_path in files:
            entries = self._read_file_reverse(file_path)
            for entry in entries:
                message = entry.message if case_sensitive else entry.message.lower()
                if query in message:
                    result.append(entry)
                    if len(result) >= limit:
                        return result

        return result

    def stream_all(self, reverse: bool = False) -> Iterator[TaleEntry]:
        """
        Stream all tales (memory efficient).

        Args:
            reverse: If True, newest first. If False, oldest first (default).

        Yields:
            TaleEntry objects one at a time
        """
        files = self._get_saga_files(reverse=reverse)

        for file_path in files:
            if reverse:
                # Need to read whole file to reverse
                entries = self._read_file_reverse(file_path)
                for entry in entries:
                    yield entry
            else:
                for entry in self._read_file(file_path):
                    yield entry

    def count(self) -> int:
        """Count total tales in all files."""
        total = 0
        for file_path in self._get_saga_files():
            try:
                with open(file_path, "r") as f:
                    total += sum(1 for line in f if line.strip())
            except Exception:
                pass
        return total

    def stats(self) -> dict:
        """
        Get statistics about the Saga.

        Returns:
            Dict with counts by category, date range, etc.
        """
        files = self._get_saga_files(reverse=False)

        if not files:
            return {
                "total_tales": 0,
                "total_files": 0,
                "date_range": None,
                "categories": {},
            }

        total = 0
        categories = {}
        oldest = None
        newest = None

        for file_path in files:
            for entry in self._read_file(file_path):
                total += 1
                categories[entry.category] = categories.get(entry.category, 0) + 1

                if oldest is None or entry.timestamp < oldest:
                    oldest = entry.timestamp
                if newest is None or entry.timestamp > newest:
                    newest = entry.timestamp

        return {
            "total_tales": total,
            "total_files": len(files),
            "date_range": {
                "oldest": oldest.isoformat() if oldest else None,
                "newest": newest.isoformat() if newest else None,
            },
            "categories": categories,
        }


# Module-level singleton
_default_reader: Optional[SagaReader] = None


def init_reader(base_dir: Path | str) -> SagaReader:
    """Initialize the default Saga reader."""
    global _default_reader
    _default_reader = SagaReader(base_dir)
    return _default_reader


def get_reader() -> Optional[SagaReader]:
    """Get the default Saga reader (None if not initialized)."""
    return _default_reader


def recent(limit: int = 50) -> List[TaleEntry]:
    """Get recent tales from default reader."""
    if _default_reader:
        return _default_reader.recent(limit)
    return []
