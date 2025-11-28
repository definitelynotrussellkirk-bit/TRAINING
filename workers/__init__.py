"""
Workers - Distributed job execution agents.

Two worker models:
1. Push model (EvalWorker, DataForgeWorker): HTTP servers that receive jobs
2. Pull model (ClaimingWorker): Polls central job store and claims jobs

Usage:
    # Push model - start HTTP server
    python3 -m workers.eval_worker --port 8900

    # Pull model - claims jobs from VaultKeeper
    python3 -m workers.claiming_worker \
        --device macmini_eval_1 \
        --server http://trainer.local:8767

Available Workers:
    - BaseWorker: Abstract base for push-model workers
    - EvalWorker: Handles EVAL and SPARRING jobs (push)
    - DataForgeWorker: Handles DATA_GEN and DATA_FILTER jobs (push)
    - ClaimingWorker: Claims jobs from central store (pull)
"""

from workers.base_worker import BaseWorker, WorkerConfig
from workers.eval_worker import EvalWorker
from workers.data_forge_worker import DataForgeWorker
from workers.claiming_worker import ClaimingWorker, ClaimingWorkerConfig

__all__ = [
    "BaseWorker",
    "WorkerConfig",
    "EvalWorker",
    "DataForgeWorker",
    "ClaimingWorker",
    "ClaimingWorkerConfig",
]
