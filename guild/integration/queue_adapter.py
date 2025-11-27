"""
Queue adapter for training queue integration.

Bridges guild runs with the existing training queue system.

Features:
- Submit training files to queue
- Map quest priorities to queue priorities
- Monitor queue status
- Track submission history
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from guild.integration.adapters import (
    BaseAdapter,
    AdapterConfig,
    AdapterResult,
)

logger = logging.getLogger(__name__)


# Priority mapping
PRIORITY_MAP = {
    # Guild priorities → Queue priorities
    "critical": "high",
    "high": "high",
    "normal": "normal",
    "medium": "normal",
    "low": "low",
    "idle": "low",
}


@dataclass
class QueueStatus:
    """Status of the training queue."""
    high_count: int = 0
    normal_count: int = 0
    low_count: int = 0
    processing_count: int = 0
    failed_count: int = 0
    recently_completed_count: int = 0

    @property
    def total_queued(self) -> int:
        return self.high_count + self.normal_count + self.low_count

    @property
    def is_empty(self) -> bool:
        return self.total_queued == 0

    @property
    def is_busy(self) -> bool:
        return self.processing_count > 0

    def to_dict(self) -> dict:
        return {
            "queued": {
                "high": self.high_count,
                "normal": self.normal_count,
                "low": self.low_count,
                "total": self.total_queued,
            },
            "processing": self.processing_count,
            "failed": self.failed_count,
            "recently_completed": self.recently_completed_count,
            "is_empty": self.is_empty,
            "is_busy": self.is_busy,
        }


@dataclass
class SubmissionResult:
    """Result of submitting a file to the queue."""
    file_path: Path
    queue_path: Path
    priority: str
    submitted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "file_path": str(self.file_path),
            "queue_path": str(self.queue_path),
            "priority": self.priority,
            "submitted_at": self.submitted_at.isoformat(),
        }


class QueueAdapter(BaseAdapter):
    """
    Adapter for training queue integration.

    Features:
    - Submit files to inbox or directly to priority queues
    - Monitor queue status
    - Track submission history
    - Map guild priorities to queue priorities
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self._submissions: List[SubmissionResult] = []

    @property
    def name(self) -> str:
        return "queue"

    @property
    def inbox_dir(self) -> Path:
        return self.config.inbox_dir

    @property
    def queue_dir(self) -> Path:
        return self.config.queue_dir

    def _get_queue_path(self, priority: str) -> Path:
        """Get path to a priority queue directory."""
        mapped_priority = PRIORITY_MAP.get(priority.lower(), "normal")
        return self.queue_dir / mapped_priority

    def health_check(self) -> bool:
        """Check if queue directories exist and are writable."""
        try:
            # Check inbox
            if not self.inbox_dir.exists():
                self.inbox_dir.mkdir(parents=True, exist_ok=True)

            # Check queue directories
            for priority in ["high", "normal", "low", "processing", "failed"]:
                queue_path = self.queue_dir / priority
                if not queue_path.exists():
                    queue_path.mkdir(parents=True, exist_ok=True)

            return True

        except Exception as e:
            logger.error(f"Queue adapter health check failed: {e}")
            return False

    def get_status(self) -> AdapterResult[QueueStatus]:
        """Get current queue status."""
        try:
            status = QueueStatus()

            # Count files in each queue
            for priority, attr in [
                ("high", "high_count"),
                ("normal", "normal_count"),
                ("low", "low_count"),
                ("processing", "processing_count"),
                ("failed", "failed_count"),
                ("recently_completed", "recently_completed_count"),
            ]:
                queue_path = self.queue_dir / priority
                if queue_path.exists():
                    count = len(list(queue_path.glob("*.jsonl")))
                    setattr(status, attr, count)

            return AdapterResult.ok(status)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def submit_to_inbox(self, file_path: Path) -> AdapterResult[SubmissionResult]:
        """
        Submit a file to the inbox for processing.

        The training daemon will move it to the appropriate queue.

        Args:
            file_path: Path to the JSONL file

        Returns:
            AdapterResult with SubmissionResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return AdapterResult.fail(f"File not found: {file_path}")

        if not file_path.suffix == ".jsonl":
            return AdapterResult.fail(f"File must be .jsonl: {file_path}")

        try:
            # Ensure inbox exists
            self.inbox_dir.mkdir(parents=True, exist_ok=True)

            # Copy to inbox
            dest_path = self.inbox_dir / file_path.name

            # Handle name collision
            if dest_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                dest_path = self.inbox_dir / new_name

            shutil.copy2(file_path, dest_path)

            result = SubmissionResult(
                file_path=file_path,
                queue_path=dest_path,
                priority="normal",  # Inbox goes to default priority
            )
            self._submissions.append(result)

            logger.info(f"Submitted {file_path.name} to inbox")
            return AdapterResult.ok(result)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def submit_to_queue(
        self,
        file_path: Path,
        priority: str = "normal",
    ) -> AdapterResult[SubmissionResult]:
        """
        Submit a file directly to a priority queue.

        Args:
            file_path: Path to the JSONL file
            priority: Priority level (high/normal/low)

        Returns:
            AdapterResult with SubmissionResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return AdapterResult.fail(f"File not found: {file_path}")

        if not file_path.suffix == ".jsonl":
            return AdapterResult.fail(f"File must be .jsonl: {file_path}")

        try:
            # Map priority
            mapped_priority = PRIORITY_MAP.get(priority.lower(), "normal")
            queue_path = self._get_queue_path(priority)
            queue_path.mkdir(parents=True, exist_ok=True)

            # Copy to queue
            dest_path = queue_path / file_path.name

            # Handle name collision
            if dest_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                dest_path = queue_path / new_name

            shutil.copy2(file_path, dest_path)

            result = SubmissionResult(
                file_path=file_path,
                queue_path=dest_path,
                priority=mapped_priority,
            )
            self._submissions.append(result)

            logger.info(f"Submitted {file_path.name} to {mapped_priority} queue")
            return AdapterResult.ok(result)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def submit_content(
        self,
        content: str,
        filename: str,
        priority: str = "normal",
    ) -> AdapterResult[SubmissionResult]:
        """
        Submit JSONL content directly (without existing file).

        Args:
            content: JSONL content string
            filename: Desired filename
            priority: Priority level

        Returns:
            AdapterResult with SubmissionResult
        """
        try:
            # Map priority and get queue path
            mapped_priority = PRIORITY_MAP.get(priority.lower(), "normal")
            queue_path = self._get_queue_path(priority)
            queue_path.mkdir(parents=True, exist_ok=True)

            # Ensure .jsonl extension
            if not filename.endswith(".jsonl"):
                filename = f"{filename}.jsonl"

            # Write directly to queue
            dest_path = queue_path / filename

            # Handle name collision
            if dest_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base = filename.rsplit(".", 1)[0]
                dest_path = queue_path / f"{base}_{timestamp}.jsonl"

            dest_path.write_text(content)

            result = SubmissionResult(
                file_path=dest_path,
                queue_path=dest_path,
                priority=mapped_priority,
            )
            self._submissions.append(result)

            logger.info(f"Wrote {dest_path.name} to {mapped_priority} queue")
            return AdapterResult.ok(result)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def list_queue(self, priority: Optional[str] = None) -> AdapterResult[List[Path]]:
        """
        List files in queue(s).

        Args:
            priority: Specific priority to list, or None for all

        Returns:
            AdapterResult with list of file paths
        """
        try:
            files: List[Path] = []

            if priority:
                priorities = [PRIORITY_MAP.get(priority.lower(), priority)]
            else:
                priorities = ["high", "normal", "low"]

            for p in priorities:
                queue_path = self.queue_dir / p
                if queue_path.exists():
                    files.extend(sorted(queue_path.glob("*.jsonl")))

            return AdapterResult.ok(files)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def get_processing_file(self) -> AdapterResult[Optional[Path]]:
        """Get the currently processing file (if any)."""
        try:
            processing_dir = self.queue_dir / "processing"
            if processing_dir.exists():
                files = list(processing_dir.glob("*.jsonl"))
                if files:
                    return AdapterResult.ok(files[0])
            return AdapterResult.ok(None)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def get_failed_files(self) -> AdapterResult[List[Path]]:
        """Get list of failed files."""
        try:
            failed_dir = self.queue_dir / "failed"
            if failed_dir.exists():
                files = list(sorted(failed_dir.glob("*.jsonl")))
                return AdapterResult.ok(files)
            return AdapterResult.ok([])

        except Exception as e:
            return AdapterResult.fail(str(e))

    def retry_failed(self, filename: str, priority: str = "normal") -> AdapterResult[SubmissionResult]:
        """
        Move a failed file back to a queue for retry.

        Args:
            filename: Name of the failed file
            priority: Priority for retry

        Returns:
            AdapterResult with SubmissionResult
        """
        try:
            failed_dir = self.queue_dir / "failed"
            failed_path = failed_dir / filename

            if not failed_path.exists():
                return AdapterResult.fail(f"Failed file not found: {filename}")

            # Move to queue
            mapped_priority = PRIORITY_MAP.get(priority.lower(), "normal")
            queue_path = self._get_queue_path(priority)
            queue_path.mkdir(parents=True, exist_ok=True)

            dest_path = queue_path / filename
            shutil.move(str(failed_path), str(dest_path))

            result = SubmissionResult(
                file_path=failed_path,
                queue_path=dest_path,
                priority=mapped_priority,
            )

            logger.info(f"Moved {filename} from failed to {mapped_priority} queue")
            return AdapterResult.ok(result)

        except Exception as e:
            return AdapterResult.fail(str(e))

    def clear_failed(self) -> AdapterResult[int]:
        """
        Clear all failed files.

        Returns:
            AdapterResult with count of cleared files
        """
        try:
            failed_dir = self.queue_dir / "failed"
            if not failed_dir.exists():
                return AdapterResult.ok(0)

            files = list(failed_dir.glob("*.jsonl"))
            for f in files:
                f.unlink()

            logger.info(f"Cleared {len(files)} failed files")
            return AdapterResult.ok(len(files))

        except Exception as e:
            return AdapterResult.fail(str(e))

    def get_submission_history(self, limit: int = 100) -> List[SubmissionResult]:
        """Get recent submission history."""
        return self._submissions[-limit:]

    def clear_submission_history(self) -> None:
        """Clear submission history."""
        self._submissions.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive queue summary."""
        status_result = self.get_status()
        if not status_result.success:
            return {"error": status_result.error}

        status = status_result.data

        return {
            "status": status.to_dict(),
            "recent_submissions": len(self._submissions),
            "directories": {
                "inbox": str(self.inbox_dir),
                "queue": str(self.queue_dir),
            }
        }


# Global adapter instance
_queue_adapter: Optional[QueueAdapter] = None


def init_queue_adapter(config: Optional[AdapterConfig] = None) -> QueueAdapter:
    """Initialize the global queue adapter."""
    global _queue_adapter
    _queue_adapter = QueueAdapter(config)
    return _queue_adapter


def get_queue_adapter() -> QueueAdapter:
    """Get the global queue adapter."""
    global _queue_adapter
    if _queue_adapter is None:
        _queue_adapter = QueueAdapter()
    return _queue_adapter


def reset_queue_adapter() -> None:
    """Reset the global queue adapter (for testing)."""
    global _queue_adapter
    _queue_adapter = None
