"""
Champion Board - Rankings of model checkpoints by combat prowess.

The Champion Board is a grand display in the Watchtower showing the
rankings of all hero checkpoints. Champions are ranked by their
performance in validation battles.

RPG Flavor:
    After each training session, heroes compete in validation tournaments.
    The Champion Board displays the current rankings. The reigning champion
    is automatically deployed to serve the realm.

Metrics Mapping:
    validation_loss   → Damage Resilience (1 - loss, higher is better)
    validation_acc    → Hit Accuracy
    tokens/sec        → Response Speed
    composite_score   → Combat Score (weighted combination)

This module wraps monitoring/model_comparison_engine.py with RPG-themed naming.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from watchtower.types import ChampionRank, ChampionBoardStatus

# Import the underlying engine
from monitoring.model_comparison_engine import ModelComparisonEngine as _ModelComparisonEngine


class ChampionBoard(_ModelComparisonEngine):
    """
    The Champion Board - ranks model checkpoints by performance.

    RPG wrapper around ModelComparisonEngine with themed method names.

    Usage:
        board = ChampionBoard(base_dir)

        # Run a tournament (compare checkpoints)
        rankings = board.run_tournament()

        # Get current champion
        champion = board.get_current_champion()

        # Get board status
        status = board.get_board_status()
    """

    def __init__(
        self,
        base_dir: str = None,
        checkpoint_dir: str = None,
        tournament_interval: int = 600,  # 10 minutes
        trial_samples: int = 100,
        min_contenders: int = 3,
    ):
        """
        Initialize the Champion Board.

        Args:
            base_dir: Base training directory
            checkpoint_dir: Directory containing checkpoint contenders
            tournament_interval: Seconds between tournaments
            trial_samples: Samples to test per contender
            min_contenders: Minimum contenders needed for tournament
        """
        super().__init__(
            base_dir=base_dir,
            checkpoint_dir=checkpoint_dir,
            comparison_interval=tournament_interval,
            test_samples=trial_samples,
            min_checkpoints_for_comparison=min_contenders,
        )

    # =========================================================================
    # TOURNAMENT METHODS
    # =========================================================================

    def run_tournament(self) -> List[ChampionRank]:
        """
        Run a tournament to rank all contenders.

        Compares recent checkpoints and produces rankings.

        Returns:
            List of ChampionRank, sorted by rank (1 = champion)
        """
        # Call underlying comparison
        results = self.compare_recent_checkpoints()

        # Convert to ChampionRank objects
        rankings = []
        for i, result in enumerate(results, 1):
            rank = ChampionRank(
                checkpoint_name=result.get("checkpoint", "unknown"),
                checkpoint_path=result.get("path", ""),
                combat_score=result.get("composite_score", 0.0),
                damage_resilience=1.0 - result.get("validation_loss", 1.0),
                hit_accuracy=result.get("validation_accuracy", 0.0),
                response_speed=result.get("tokens_per_second", 0.0),
                validation_loss=result.get("validation_loss", 0.0),
                validation_accuracy=result.get("validation_accuracy", 0.0),
                tokens_per_second=result.get("tokens_per_second", 0.0),
                rank=i,
                evaluated_at=datetime.now(),
            )
            rankings.append(rank)

        return rankings

    def get_current_champion(self) -> Optional[ChampionRank]:
        """
        Get the current reigning champion.

        Returns:
            ChampionRank of the best checkpoint, or None
        """
        best = self.get_best_checkpoint()
        if not best:
            return None

        return ChampionRank(
            checkpoint_name=best.get("checkpoint", "unknown"),
            checkpoint_path=best.get("path", ""),
            combat_score=best.get("composite_score", 0.0),
            damage_resilience=1.0 - best.get("validation_loss", 1.0),
            hit_accuracy=best.get("validation_accuracy", 0.0),
            response_speed=best.get("tokens_per_second", 0.0),
            validation_loss=best.get("validation_loss", 0.0),
            validation_accuracy=best.get("validation_accuracy", 0.0),
            tokens_per_second=best.get("tokens_per_second", 0.0),
            rank=1,
            evaluated_at=datetime.now(),
        )

    def get_board_status(self) -> ChampionBoardStatus:
        """
        Get current Champion Board status.

        Returns:
            ChampionBoardStatus with rankings and tournament info
        """
        # Get all rankings
        rankings = self.run_tournament() if self._should_run_tournament() else []

        champion = rankings[0] if rankings else None
        contenders = rankings[1:6] if len(rankings) > 1 else []  # Top 5 contenders

        return ChampionBoardStatus(
            total_champions=len(rankings),
            current_champion=champion,
            recent_contenders=contenders,
            last_tournament=datetime.now() if rankings else None,
        )

    def _should_run_tournament(self) -> bool:
        """Check if we should run a tournament now."""
        # Check if enough checkpoints exist
        checkpoints = self.get_available_checkpoints()
        return len(checkpoints) >= self.min_checkpoints_for_comparison

    def get_available_checkpoints(self) -> List[str]:
        """Get list of available checkpoint contenders."""
        return self.list_checkpoints()

    def list_contenders(self, limit: int = 10) -> List[ChampionRank]:
        """
        List recent contenders on the board.

        Args:
            limit: Maximum contenders to return

        Returns:
            List of ChampionRank
        """
        rankings = self.run_tournament()
        return rankings[:limit]

    # =========================================================================
    # DEPLOYMENT INTEGRATION
    # =========================================================================

    def crown_champion(self, checkpoint_name: str) -> bool:
        """
        Crown a specific checkpoint as champion (force selection).

        Args:
            checkpoint_name: Name of checkpoint to crown

        Returns:
            True if successful
        """
        # Mark as manually selected best
        return self.set_best_checkpoint(checkpoint_name)

    def get_champion_for_deployment(self) -> Optional[Dict[str, Any]]:
        """
        Get champion info formatted for deployment.

        Returns:
            Dict with checkpoint info for deployment orchestrator
        """
        champion = self.get_current_champion()
        if not champion:
            return None

        return {
            "checkpoint_name": champion.checkpoint_name,
            "checkpoint_path": champion.checkpoint_path,
            "combat_score": champion.combat_score,
            "validation_loss": champion.validation_loss,
            "validation_accuracy": champion.validation_accuracy,
            "selected_at": datetime.now().isoformat(),
        }


# Convenience function
def get_champion_board(base_dir: str = None) -> ChampionBoard:
    """Get a ChampionBoard instance for the given base directory."""
    return ChampionBoard(base_dir=base_dir)


# Re-export original for backward compatibility
ModelComparisonEngine = _ModelComparisonEngine
