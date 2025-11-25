# TASK005: Refactor TrainingDaemon

**Priority:** MEDIUM
**Effort:** 6 hours
**Dependencies:** TASK003 (retention wiring)
**Files:** `core/training_daemon.py`

---

## Problem

`TrainingDaemon` is another god object handling:

1. Config loading/validation
2. Directory bootstrapping
3. PID locking
4. Snapshot creation & verification
5. Retention calls
6. Consolidation
7. Auto data generation
8. Signal handling
9. File queue processing

This makes it hard to test, modify, and understand.

## Solution

Extract responsibilities into focused services that the daemon orchestrates.

## Target Architecture

```
core/
├── training_daemon.py         # Slim orchestrator (~200 lines)
├── daemon/
│   ├── __init__.py
│   ├── config.py             # DaemonConfig dataclass
│   ├── pid_manager.py        # PID file locking
│   ├── file_watcher.py       # Inbox monitoring
│   ├── queue_processor.py    # Queue management
│   ├── snapshot_service.py   # Snapshot creation/verification
│   └── data_generator.py     # Auto data generation
```

## Implementation Steps

### Step 1: Extract PIDManager

```python
# core/daemon/pid_manager.py
from pathlib import Path
import os
import sys

class PIDManager:
    """Manages PID file for single-instance enforcement."""

    def __init__(self, pid_file: Path):
        self.pid_file = pid_file

    def acquire(self) -> bool:
        """Acquire PID lock. Returns False if another instance running."""
        if self.pid_file.exists():
            old_pid = int(self.pid_file.read_text().strip())
            if self._is_running(old_pid):
                return False
            # Stale PID file
            self.pid_file.unlink()

        self.pid_file.write_text(str(os.getpid()))
        return True

    def release(self):
        """Release PID lock."""
        if self.pid_file.exists():
            self.pid_file.unlink()

    def _is_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Another daemon instance is running")
        return self

    def __exit__(self, *args):
        self.release()
```

### Step 2: Extract FileWatcher

```python
# core/daemon/file_watcher.py
from pathlib import Path
from typing import List, Callable
import time

class FileWatcher:
    """Watches directory for new files."""

    def __init__(self, watch_dir: Path, pattern: str = "*.jsonl"):
        self.watch_dir = watch_dir
        self.pattern = pattern
        self._seen = set()

    def get_new_files(self) -> List[Path]:
        """Return files that haven't been seen yet."""
        current = set(self.watch_dir.glob(self.pattern))
        new_files = current - self._seen
        self._seen = current
        return sorted(new_files, key=lambda p: p.stat().st_mtime)

    def mark_processed(self, path: Path):
        """Mark a file as processed."""
        self._seen.add(path)
```

### Step 3: Extract SnapshotService

```python
# core/daemon/snapshot_service.py
from pathlib import Path
from datetime import datetime
import shutil
import hashlib

class SnapshotService:
    """Handles checkpoint snapshots and verification."""

    def __init__(self, checkpoints_dir: Path, snapshots_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.snapshots_dir = snapshots_dir
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, checkpoint_name: str) -> Path:
        """Create verified snapshot of checkpoint."""
        src = self.checkpoints_dir / checkpoint_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self.snapshots_dir / f"{checkpoint_name}_{timestamp}"

        shutil.copytree(src, dst)

        if not self._verify(src, dst):
            shutil.rmtree(dst)
            raise ValueError(f"Snapshot verification failed for {checkpoint_name}")

        return dst

    def _verify(self, src: Path, dst: Path) -> bool:
        """Verify snapshot matches source."""
        for src_file in src.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(src)
                dst_file = dst / rel
                if not dst_file.exists():
                    return False
                if self._hash_file(src_file) != self._hash_file(dst_file):
                    return False
        return True

    def _hash_file(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()
```

### Step 4: Extract DataGenerator

```python
# core/daemon/data_generator.py
from pathlib import Path
from typing import Optional
import subprocess

class DataGenerator:
    """Handles automatic training data generation."""

    def __init__(self, output_dir: Path, min_queue_size: int = 2):
        self.output_dir = output_dir
        self.min_queue_size = min_queue_size

    def should_generate(self, current_queue_size: int) -> bool:
        """Check if we need more data."""
        return current_queue_size < self.min_queue_size

    def generate(self, count: int = 1000) -> Optional[Path]:
        """Generate training data file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"auto_gen_{timestamp}.jsonl"

        result = subprocess.run([
            "python", "tools/data/generate_syllo_batch.py",
            "--count", str(count),
            "--output", str(output_file)
        ], capture_output=True, timeout=300)

        if result.returncode == 0 and output_file.exists():
            return output_file
        return None
```

### Step 5: Slim TrainingDaemon

```python
# core/training_daemon.py
"""Training Daemon - orchestrates continuous training."""

from pathlib import Path
import signal
import time
from daemon.pid_manager import PIDManager
from daemon.file_watcher import FileWatcher
from daemon.snapshot_service import SnapshotService
from daemon.data_generator import DataGenerator
from management.retention_service import RetentionService
from training_queue import TrainingQueue

class TrainingDaemon:
    """Orchestrates continuous training workflow."""

    def __init__(self, base_dir: Path, config: dict):
        self.base_dir = base_dir
        self.config = config
        self.running = True

        # Initialize services
        self.pid_manager = PIDManager(base_dir / ".pids" / "daemon.pid")
        self.file_watcher = FileWatcher(base_dir / "inbox")
        self.queue = TrainingQueue(base_dir / "queue")
        self.retention = RetentionService(base_dir / "models/current_model")
        self.snapshots = SnapshotService(
            base_dir / "models/current_model",
            base_dir / "backups/snapshots"
        )
        self.data_gen = DataGenerator(base_dir / "inbox")

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def run(self):
        """Main daemon loop."""
        with self.pid_manager:
            while self.running:
                self._process_cycle()
                time.sleep(self.config.get("poll_interval", 30))

    def _process_cycle(self):
        """Single processing cycle."""
        # Check for new files
        for new_file in self.file_watcher.get_new_files():
            self.queue.add(new_file)

        # Auto-generate if queue low
        if self.data_gen.should_generate(self.queue.size()):
            self.data_gen.generate()

        # Process queue
        next_file = self.queue.get_next()
        if next_file:
            self._train_file(next_file)

        # Maintenance
        self.retention.enforce()

    def _train_file(self, file_path: Path):
        """Train on a single file."""
        # ... training logic ...
        pass

    def _handle_signal(self, signum, frame):
        self.running = False
```

## Checkpoints

- [x] Create `core/daemon/` package directory
- [x] Extract `PIDManager` class (with tests)
- [x] Extract `FileWatcher` class (with tests)
- [x] Extract `InboxFlattener` class (added as part of FileWatcher)
- [x] Wire PIDManager into daemon (acquire_lock/release_lock delegate)
- [x] Wire InboxFlattener into daemon (flatten_inbox delegates)
- [ ] Extract `SnapshotService` class (future work)
- [ ] Extract `DataGenerator` class (future work)
- [ ] Slim `training_daemon.py` to ~200 lines (future work)
- [ ] Add integration tests

## Verification

```bash
# Daemon should work as before
python core/training_daemon.py --base-dir /path/to/training

# Services should be importable
python -c "from core.daemon.pid_manager import PIDManager; print('OK')"
python -c "from core.daemon.snapshot_service import SnapshotService; print('OK')"
```

## Migration Strategy

1. Create new services alongside existing code
2. One service at a time: extract, test, wire in
3. Keep daemon functional during migration
4. Each extraction = separate commit
