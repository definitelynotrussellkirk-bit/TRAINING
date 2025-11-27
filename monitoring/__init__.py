"""
Monitoring modules for training observation.

Contains:
- model_comparison_engine.py: Compare checkpoints
- deployment_orchestrator.py: Auto-deployment
- prediction_client.py: Inference client
- servers/: API servers for monitoring

NEW: RPG-themed wrappers available in watchtower/:
- ScryingPool: Real-time training observation
- ChampionBoard: Model checkpoint rankings
- OracleClient: Inference client

Usage:
    # Traditional
    from monitoring.model_comparison_engine import ModelComparisonEngine
    from monitoring.prediction_client import PredictionClient

    # RPG-themed (new)
    from watchtower import ScryingPool, ChampionBoard, OracleClient
"""

# Re-export RPG-themed classes for convenience
try:
    from watchtower import (
        # Scrying Pool
        ScryingPool,
        get_scrying_pool,
        ScryingVision,
        # Champion Board
        ChampionBoard,
        get_champion_board,
        ChampionRank,
        ChampionBoardStatus,
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
