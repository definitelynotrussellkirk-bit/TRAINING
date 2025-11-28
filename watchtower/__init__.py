"""
Watchtower - The high observation post overlooking all training operations.

The Watchtower stands at the highest point of the training citadel,
providing a commanding view of all operations:

    Scrying Pool    - Real-time battle observation
    Oracle Client   - Communication with inference
    Heralds         - Alert notifications
    Task Sentries   - GPU task coordination

RPG Mapping:
    Live Monitor        → Scrying Pool
    Inference Client    → Oracle
    Alerts              → Heralds
    Task Scheduler      → Task Sentries
    Dashboards          → War Room
    API Server          → Crystal Network

Quick Start:
    from watchtower import ScryingPool, OracleClient

    # Observe current training
    pool = ScryingPool(base_dir)
    vision = pool.gaze()
    print(f"Battle: {vision.battle_state}, Damage: {vision.damage_taken}")

    # Get a prophecy from the Oracle
    oracle = OracleClient()
    response = oracle.seek_prophecy("What is 2 + 2?")
    print(f"Prophecy: {response.prophecy}")

This module wraps monitoring/ with RPG-themed naming while maintaining
backward compatibility.
"""

__version__ = "0.1.0"

# Types
from watchtower.types import (
    # Watcher states
    WatcherState,
    AlertLevel,
    # Oracle
    OracleResponse,
    # Scrying Pool
    ScryingVision,
    # Heralds
    HeraldMessage,
    # Task Sentry
    TaskPriority,
    TaskStatus,
    SentryTask,
)

# Scrying Pool (real-time observation)
from watchtower.scrying_pool import (
    ScryingPool,
    get_scrying_pool,
)

# Oracle Client (inference)
from watchtower.oracle_client import (
    OracleClient,
    get_oracle_client,
    # Backward compat
    PredictionClient,
)


__all__ = [
    # Types - Watcher
    "WatcherState",
    "AlertLevel",
    # Types - Oracle
    "OracleResponse",
    # Types - Scrying Pool
    "ScryingVision",
    # Types - Herald
    "HeraldMessage",
    # Types - Task Sentry
    "TaskPriority",
    "TaskStatus",
    "SentryTask",
    # Scrying Pool
    "ScryingPool",
    "get_scrying_pool",
    # Oracle
    "OracleClient",
    "get_oracle_client",
    "PredictionClient",  # Backward compat
]


# =============================================================================
# RPG TERMINOLOGY GUIDE
# =============================================================================

"""
WATCHTOWER GLOSSARY
===================

The Watchtower uses RPG terminology to make monitoring concepts more intuitive:

OBSERVATION TERMS
-----------------
Scrying Pool    = Live training monitor (real-time status display)
Vision          = Training status snapshot
Gaze            = Read current status
Disturbance     = Anomaly or alert condition

RANKING TERMS
-------------
Champion Board  = Model comparison rankings
Champion        = Best checkpoint (highest score)
Contender       = Checkpoint being evaluated
Tournament      = Comparison run (evaluate multiple checkpoints)
Combat Score    = Composite performance metric
Crown           = Select as best checkpoint

ORACLE TERMS
------------
Oracle          = Inference server (3090)
Prophecy        = Model output (generated text)
Seek Prophecy   = Request inference
Oracle Wisdom   = Model info (name, path, VRAM)
Oracle Pulse    = Health check
Rebirth         = Reload model with new checkpoint
Crystal Tower   = Inference server location
Seeker Key      = Read API key
High Priest Key = Admin API key

HERALD TERMS
------------
Herald          = Alert/notification carrier
Announcement    = Alert message
Whisper         = Debug level
Notice          = Info level
Warning         = Warning level
Alarm           = Error level
Crisis          = Critical level

SENTRY TERMS
------------
Task Sentry     = GPU task scheduler
Dispatch        = Submit task for execution
Patrol          = Monitor GPU utilization
Critical/High/Normal/Low/Idle = Task priorities

LOCATION TERMS
--------------
Watchtower      = monitoring/ (observation systems)
War Room        = Dashboards (strategic overview)
Crystal Network = API layer (communication)
"""
