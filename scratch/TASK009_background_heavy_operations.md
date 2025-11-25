# TASK009: Background Heavy Operations

**Priority:** LOW
**Effort:** 4 hours
**Dependencies:** TASK005
**Files:** `core/training_daemon.py`, new background worker

---

## Problem

The training daemon runs heavy operations synchronously in its main loop:

1. **Snapshot creation** - copies checkpoint files, verifies integrity
2. **Consolidation** - shells out to `consolidate_model.py` with 30-min timeout
3. **Data validation** - loads tokenizer, samples files

If disk is slow or consolidation hangs, the daemon is blocked and can't:
- Process new files
- Respond to control signals promptly
- Update status

## Solution

Offload heavy maintenance tasks to background workers or scheduled jobs.

## Implementation Options

### Option A: Threading with Queue

```python
# core/daemon/background_worker.py
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Task:
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None

class BackgroundWorker:
    """Runs heavy tasks in background thread."""

    def __init__(self, max_concurrent: int = 1):
        self.task_queue = queue.Queue()
        self.max_concurrent = max_concurrent
        self.workers = []
        self._stop = threading.Event()

    def start(self):
        for i in range(self.max_concurrent):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def submit(self, task: Task):
        self.task_queue.put(task)

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                try:
                    task.func(*task.args, **(task.kwargs or {}))
                except Exception as e:
                    logger.error(f"Background task {task.name} failed: {e}")
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self._stop.set()
        for worker in self.workers:
            worker.join(timeout=5.0)
```

### Option B: Subprocess Pool

```python
# core/daemon/task_pool.py
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Callable

class TaskPool:
    """Runs heavy tasks in separate processes."""

    def __init__(self, max_workers: int = 2):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.pending = {}

    def submit(self, name: str, func: Callable, *args, timeout: int = 1800):
        """Submit task, returns future."""
        future = self.executor.submit(func, *args)
        self.pending[name] = (future, timeout)
        return future

    def check_completed(self):
        """Check and cleanup completed tasks."""
        completed = []
        for name, (future, timeout) in list(self.pending.items()):
            if future.done():
                try:
                    result = future.result()
                    logger.info(f"Task {name} completed: {result}")
                except Exception as e:
                    logger.error(f"Task {name} failed: {e}")
                completed.append(name)

        for name in completed:
            del self.pending[name]

    def shutdown(self):
        self.executor.shutdown(wait=False)
```

### Option C: Cron/Systemd Timers (Recommended for maintenance)

For periodic tasks like retention and snapshots:

```bash
# /etc/cron.d/llm-training
# Run retention every 6 hours
0 */6 * * * user /path/to/training/scripts/run_retention.sh

# Daily snapshot at 3am
0 3 * * * user /path/to/training/scripts/daily_snapshot.sh

# Weekly consolidation on Sunday 4am
0 4 * * 0 user /path/to/training/scripts/weekly_consolidation.sh
```

## Recommended Approach

**Hybrid:** Use cron for scheduled maintenance, threading for on-demand tasks.

### Step 1: Move scheduled tasks to cron

```bash
# scripts/run_retention.sh
#!/bin/bash
cd /path/to/training
source venv/bin/activate
python -c "
from management.retention_service import RetentionService
from pathlib import Path
svc = RetentionService(Path('models/current_model'))
result = svc.enforce()
print(f'Retention: deleted {len(result[\"deleted\"])}, freed {result[\"freed_gb\"]:.1f}GB')
" >> logs/retention.log 2>&1
```

### Step 2: Add background worker to daemon

```python
# core/training_daemon.py
from daemon.background_worker import BackgroundWorker, Task

class TrainingDaemon:
    def __init__(self, ...):
        self.background = BackgroundWorker(max_concurrent=1)

    def run(self):
        self.background.start()
        try:
            while self.running:
                self._process_cycle()
                time.sleep(self.config.get("poll_interval", 30))
        finally:
            self.background.stop()

    def _process_cycle(self):
        # On-demand snapshot (non-blocking)
        if self._should_snapshot():
            self.background.submit(Task(
                name="snapshot",
                func=self.snapshots.create_snapshot,
                args=(self._get_latest_checkpoint(),)
            ))

        # Retention now handled by cron, but can trigger manually
        if self._should_run_retention():
            self.background.submit(Task(
                name="retention",
                func=self.retention.enforce
            ))
```

### Step 3: Add status tracking

```python
# core/daemon/background_worker.py
@dataclass
class TaskStatus:
    name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None

class BackgroundWorker:
    def __init__(self, ...):
        self.task_history: List[TaskStatus] = []

    def get_status(self) -> dict:
        """Return status for monitoring."""
        return {
            "queue_size": self.task_queue.qsize(),
            "running": [t for t in self.task_history if t.completed_at is None],
            "recent": self.task_history[-10:]
        }
```

### Step 4: Expose in status API

```python
# In daemon status writer
def write_status(self):
    status = {
        "training": {...},
        "background_tasks": self.background.get_status()
    }
```

## Checkpoints

- [ ] Create `core/daemon/background_worker.py`
- [ ] Create maintenance scripts in `scripts/`
- [ ] Set up cron jobs for retention/snapshots
- [ ] Update daemon to use background worker
- [ ] Add task status tracking
- [ ] Update status writer to include background tasks
- [ ] Test: heavy task doesn't block daemon loop
- [ ] Test: cron jobs run successfully

## Verification

```bash
# Check daemon responsiveness during heavy task
# In terminal 1:
python core/training_daemon.py

# In terminal 2:
# Trigger a heavy operation
touch control/.trigger_snapshot

# Check daemon still responds
python core/training_controller.py status
# Should return quickly even if snapshot running

# Check background task status
cat status/training_status.json | jq '.background_tasks'
```

## Configuration

```json
{
  "background": {
    "max_workers": 1,
    "snapshot_interval_hours": 6,
    "use_cron_for_retention": true
  }
}
```

## Rollback

If background execution causes issues:
1. Set `"use_cron_for_retention": false` in config
2. Disable cron jobs
3. Tasks will run synchronously as before
