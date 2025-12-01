# VERBOSE MODE - Task Lifecycle Tracking

Track evaluation tasks through their complete lifecycle with detailed timestamps and durations.

## Quick Start

```bash
# Enable verbose mode
export VERBOSE=1

# Start eval runner (verbose logging is automatic)
python3 core/eval_runner.py --daemon

# Watch verbose logs in real-time
python3 scripts/verbose_monitor.py --watch

# View recent activity
python3 scripts/verbose_monitor.py --tail 20

# Show statistics
python3 scripts/verbose_monitor.py --stats
```

## What Gets Tracked

For each evaluation task, verbose mode tracks:

1. **Task Created** - When the eval is queued
2. **Task Started** - When eval worker picks it up from queue
3. **Task Progress** - Checkpoints during execution (model loading, etc.)
4. **Task Finished** - Completion with success/failure status

### Timestamps Captured

| Timestamp | Description | Calculation |
|-----------|-------------|-------------|
| `created_at` | When task was queued | Immediate |
| `started_at` | When execution began | When worker claims task |
| `finished_at` | When execution completed | When result is recorded |
| `time_in_queue` | Queue wait time | `started_at - created_at` |
| `execution_time` | Runtime | `finished_at - started_at` |
| `total_time` | End-to-end | `finished_at - created_at` |

## Configuration

### Environment Variables

```bash
# Enable verbose mode (required)
export VERBOSE=1

# Set custom log file location (optional)
export VERBOSE_FILE=/path/to/custom/verbose.log
```

Default log location: `status/verbose.log`

## Monitoring Tools

### Watch Live Logs

```bash
python3 scripts/verbose_monitor.py --watch
```

Output:
```
[2025-11-30T02:12:16.280453] 📥 QUEUED  eval/eval-183726-binary-1
[2025-11-30T02:12:16.559641] ▶️  STARTED eval/eval-183726-binary-1
[2025-11-30T02:12:16.859842] ⚙️  PROGRESS eval/eval-183726-binary-1: Checkpoint loaded
[2025-11-30T02:12:17.360031] ✅ SUCCESS  eval/eval-183726-binary-1 (1.1s)
```

### View Statistics

```bash
# All tasks
python3 scripts/verbose_monitor.py --stats

# Eval tasks only
python3 scripts/verbose_monitor.py --stats --type eval
```

Output:
```
======================================================================
Task Statistics - eval
======================================================================
Total tasks:        42
Queued:             2
In progress:        3
Completed:          37
  Successful:       35
  Failed:           2

Avg queue time:     2.3s
Avg execution time: 15.4s
Avg total time:     17.7s
======================================================================
```

### View Specific Task

```bash
python3 scripts/verbose_monitor.py --task eval-183726-binary-1
```

Output:
```
======================================================================
Task Lifecycle: eval-183726-binary-1
======================================================================
Type:          eval
Metadata:      {"skill": "binary", "level": 1, "checkpoint": 183726}

Created:       2025-11-30T02:12:16.280453
Started:       2025-11-30T02:12:16.559641
Queue time:    279ms
Finished:      2025-11-30T02:12:17.360031
Execution:     800ms
Total time:    1.1s
Status:        SUCCESS
Result:        {"accuracy": 0.8, "correct": 4, "total": 5}
======================================================================
```

### Tail Recent Activity

```bash
python3 scripts/verbose_monitor.py --tail 10
```

## Log Format

Each log entry is a JSON line with the following structure:

```json
{
  "timestamp": "2025-11-30T02:12:16.280453",
  "unix_timestamp": 1732936336.280453,
  "event": "task_queued",
  "task_type": "eval",
  "task_id": "eval-183726-binary-1",
  "created_at": "2025-11-30T02:12:16.280453",
  "metadata": {
    "checkpoint_step": 183726,
    "skill": "binary",
    "level": 1,
    "eval_type": "quick",
    "priority": 10
  }
}
```

### Event Types

| Event | When | Fields |
|-------|------|--------|
| `task_queued` | Task added to queue | `created_at`, `metadata` |
| `task_started` | Execution begins | `started_at`, `time_in_queue_seconds` |
| `task_progress` | Progress update | `message`, custom data |
| `task_finished` | Task complete | `finished_at`, `success`, `result`, `error`, durations |

## Integration with Eval System

Verbose logging is automatically integrated into:

### Evaluation Queue

- **queue_evaluation()** - Logs when eval is queued
- **pop_evaluation()** - Logs when eval is dequeued for execution

### Eval Runner

- **run_skill_evaluation()** - Logs:
  - Task started
  - Checkpoint loaded (progress)
  - Task finished (with accuracy result)

### Example Flow

```python
from core.evaluation_ledger import queue_evaluation
from core.eval_runner import EvalRunner

# 1. Queue an evaluation (logged if VERBOSE=1)
queue_evaluation(
    checkpoint_step=183726,
    skill="binary",
    level=1,
    eval_type="quick"
)
# Logs: task_queued

# 2. Runner picks it up
runner = EvalRunner()
runner.process_skill_queue(limit=1)
# Logs: task_started, task_progress, task_finished
```

## Use Cases

### Performance Analysis

Track eval system performance:

```bash
# Show stats to see average queue and execution times
python3 scripts/verbose_monitor.py --stats --type eval

# Identify slow evaluations
python3 scripts/verbose_monitor.py --tail 100 | grep "execution_time_seconds"
```

### Debugging

Find failed evaluations:

```bash
# Watch for failures in real-time
python3 scripts/verbose_monitor.py --watch | grep "FAILED"

# View recent failures
python3 scripts/verbose_monitor.py --tail 50 | grep "❌"
```

### Monitoring Production

Track system health:

```bash
# Monitor active tasks
python3 scripts/verbose_monitor.py --stats

# Watch eval queue
watch -n 5 "python3 scripts/verbose_monitor.py --stats --type eval"
```

## Programmatic Access

Use VerboseLogger directly in your code:

```python
from core.verbose_logger import VerboseLogger, is_verbose_mode

if is_verbose_mode():
    task_id = "my-custom-task-123"

    # Log task creation
    VerboseLogger.task_queued("custom", task_id, {
        "operation": "model_sync",
        "checkpoint": 183726
    })

    # Log start
    VerboseLogger.task_started("custom", task_id)

    # Log progress
    VerboseLogger.task_progress("custom", task_id, "Syncing to remote server", {
        "bytes_transferred": 1024000
    })

    # Log completion
    VerboseLogger.task_finished("custom", task_id, success=True, result={
        "sync_duration": 15.3,
        "size_mb": 500
    })

    # Query task lifecycle
    lifecycle = VerboseLogger.get_task_lifecycle(task_id)
    print(f"Total time: {lifecycle.total_time}s")
```

## Maintenance

### Cleanup Old Tasks

```bash
# Clean up completed tasks older than 1 hour
python3 scripts/verbose_monitor.py --cleanup
```

### Log Rotation

The verbose log is append-only. Rotate it periodically:

```bash
# Manual rotation
mv status/verbose.log status/verbose.log.$(date +%Y%m%d)

# Or use logrotate
cat > /etc/logrotate.d/training-verbose <<EOF
/path/to/training/status/verbose.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

## Troubleshooting

### Verbose Mode Not Working

Check environment:
```bash
echo $VERBOSE
# Should show: 1
```

Re-export if needed:
```bash
export VERBOSE=1
```

### No Log File

Verbose logs only appear when verbose mode is enabled AND tasks are executing.

Create test logs:
```bash
VERBOSE=1 python3 core/verbose_logger.py
```

### Too Much Logging

Disable verbose mode when not needed:
```bash
unset VERBOSE
# Or
export VERBOSE=0
```

## Advanced: Custom Task Types

Track your own task types:

```python
from core.verbose_logger import VerboseLogger

# Track model sync
VerboseLogger.task_queued("model_sync", "sync-183726", {
    "source": "training_machine",
    "dest": "inference_server",
    "size_mb": 500
})

VerboseLogger.task_started("model_sync", "sync-183726")
VerboseLogger.task_finished("model_sync", "sync-183726", success=True)

# View stats for this task type
python3 scripts/verbose_monitor.py --stats --type model_sync
```

## See Also

- [core/verbose_logger.py](../core/verbose_logger.py) - Implementation
- [scripts/verbose_monitor.py](../scripts/verbose_monitor.py) - CLI monitoring tool
- [core/eval_runner.py](../core/eval_runner.py) - Eval system integration
