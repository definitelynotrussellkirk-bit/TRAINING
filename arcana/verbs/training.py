"""
Training Verbs - Control training operations.

Verbs:
    (train :quest q1 :steps 500)
    (train-file :path "data/train.jsonl" :steps 100)
    (pause)
    (resume)
    (stop)
    (checkpoint :name "my-save")
    (queue-status)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .core import _parse_kwargs


def verb_train(engine, *args):
    """Start training on a quest.

    (train :quest q1 :steps 500)

    Adds the quest's dataset to the training queue.
    """
    kwargs = _parse_kwargs(args)
    quest_id = kwargs.get('quest')
    steps = kwargs.get('steps', 100)

    if not quest_id:
        raise ValueError("train requires :quest")

    quest = engine.world.get('quest', quest_id)
    if not quest:
        raise ValueError(f"Unknown quest: {quest_id}")

    dataset = quest.get('dataset')
    if not dataset:
        raise ValueError(f"Quest {quest_id} has no dataset")

    # Add to queue
    return _add_to_queue(engine, dataset, steps, kwargs.get('priority', 'normal'))


def verb_train_file(engine, *args):
    """Train on a specific file.

    (train-file :path "data/train.jsonl" :steps 100 :priority high)
    """
    kwargs = _parse_kwargs(args)
    path = kwargs.get('path')
    steps = kwargs.get('steps', 100)
    priority = kwargs.get('priority', 'normal')

    if not path:
        raise ValueError("train-file requires :path")

    return _add_to_queue(engine, path, steps, priority)


def _add_to_queue(engine, dataset_path: str, steps: int, priority: str) -> Dict[str, Any]:
    """Add a training file to the queue."""
    base_dir = engine.world.base_dir

    # Resolve dataset path
    dataset = Path(dataset_path)
    if not dataset.is_absolute():
        dataset = base_dir / dataset

    if not dataset.exists():
        raise ValueError(f"Dataset not found: {dataset}")

    # Determine queue directory
    priority_map = {'high': 'high', 'normal': 'normal', 'low': 'low'}
    queue_name = priority_map.get(priority, 'normal')
    queue_dir = base_dir / 'queue' / queue_name

    queue_dir.mkdir(parents=True, exist_ok=True)

    # Copy or symlink to queue
    dest = queue_dir / dataset.name
    if not dest.exists():
        shutil.copy2(dataset, dest)

    return {
        'action': 'queued',
        'dataset': str(dataset),
        'queue': queue_name,
        'steps': steps
    }


def verb_pause(engine):
    """Pause training.

    (pause)
    """
    return _send_control_command(engine, 'pause')


def verb_resume(engine):
    """Resume training.

    (resume)
    """
    return _send_control_command(engine, 'resume')


def verb_stop(engine):
    """Stop training.

    (stop)
    """
    return _send_control_command(engine, 'stop')


def _send_control_command(engine, command: str) -> Dict[str, Any]:
    """Send a control command to the training daemon."""
    base_dir = engine.world.base_dir
    control_file = base_dir / 'control' / 'command.json'

    control_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = {
        'command': command,
        'timestamp': datetime.now().isoformat()
    }

    with open(control_file, 'w') as f:
        json.dump(cmd, f)

    return {'command': command, 'status': 'sent'}


def verb_checkpoint(engine, *args):
    """Request a checkpoint save.

    (checkpoint :name "my-checkpoint")
    """
    kwargs = _parse_kwargs(args)
    name = kwargs.get('name', f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M')}")

    return _send_control_command(engine, f'checkpoint:{name}')


def verb_queue_status(engine):
    """Get training queue status.

    (queue-status)
    """
    base_dir = engine.world.base_dir
    queue_base = base_dir / 'queue'

    status = {'queues': {}}

    for priority in ['high', 'normal', 'low']:
        queue_dir = queue_base / priority
        if queue_dir.exists():
            files = list(queue_dir.glob('*.jsonl'))
            status['queues'][priority] = {
                'count': len(files),
                'files': [f.name for f in files[:5]]  # First 5
            }

    return status


def verb_clear_queue(engine, *args):
    """Clear the training queue.

    (clear-queue)
    (clear-queue :priority high)
    """
    kwargs = _parse_kwargs(args) if args else {}
    priority = kwargs.get('priority')

    base_dir = engine.world.base_dir
    queue_base = base_dir / 'queue'

    cleared = 0
    priorities = [priority] if priority else ['high', 'normal', 'low']

    for p in priorities:
        queue_dir = queue_base / p
        if queue_dir.exists():
            for f in queue_dir.glob('*.jsonl'):
                f.unlink()
                cleared += 1

    return {'cleared': cleared}


def verb_training_status(engine):
    """Get current training status.

    (training-status)
    """
    base_dir = engine.world.base_dir
    status_file = base_dir / 'status' / 'training_status.json'

    if not status_file.exists():
        return {'status': 'unknown'}

    try:
        actual = status_file.resolve()
        with open(actual) as f:
            return json.load(f)
    except Exception as e:
        return {'status': 'error', 'error': str(e)}
