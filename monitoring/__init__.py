"""
Monitoring modules for training observation.

Contains:
- deployment_orchestrator.py: Auto-deployment
- prediction_client.py: Inference client
- servers/: API servers for monitoring

RPG-themed wrappers available in watchtower/:
- ScryingPool: Real-time training observation
- OracleClient: Inference client

Usage:
    from monitoring.prediction_client import PredictionClient

    # RPG-themed
    from watchtower import ScryingPool, OracleClient
"""

# Re-export RPG-themed classes for convenience
try:
    from watchtower import (
        # Scrying Pool
        ScryingPool,
        get_scrying_pool,
        ScryingVision,
        # Oracle
        OracleClient,
        get_oracle_client,
        OracleResponse,
        # Types
        WatcherState,
        AlertLevel,
        HeraldMessage,
        TaskPriority,
        TaskStatus,
        SentryTask,
    )
except ImportError:
    # watchtower not yet available
    pass
