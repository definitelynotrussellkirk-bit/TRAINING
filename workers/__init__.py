"""
Workers - Distributed job execution agents.

Workers are HTTP servers that accept job submissions and execute them.
Each worker type handles specific job types.

Usage:
    # Start a worker on a remote machine
    python3 -m workers.eval_worker --port 8900

    # Or programmatically
    from workers import EvalWorker

    worker = EvalWorker(device_id="macmini_eval_1")
    worker.run(port=8900)

Available Workers:
    - BaseWorker: Abstract base class for workers
    - EvalWorker: Handles EVAL and SPARRING jobs
    - DataForgeWorker: Handles DATA_GEN and DATA_FILTER jobs
"""

from workers.base_worker import BaseWorker, WorkerConfig
from workers.eval_worker import EvalWorker
from workers.data_forge_worker import DataForgeWorker

__all__ = [
    "BaseWorker",
    "WorkerConfig",
    "EvalWorker",
    "DataForgeWorker",
]
