"""
TrainerEngine wrapper for hero campaigns.

This is a Layer 3 interface (see ARCHITECTURE.md "Training Flow Architecture")
that wraps TrainerEngine (Layer 1) for use in the hero loop system.

Used for all hero types (DIO, Titan, etc.) that use the standard training loop.
"""

import time
from pathlib import Path
from typing import Any, Dict

from arena.trainers.base import BaseTrainer, TrainingResult as ArenaTrainingResult

# TrainerEngine is Layer 1 - THE training executor
from trainer.core.engine import TrainerEngine, TrainingResult, MonitorContext
from trainer.config import ConfigLoader
from trainer.monitoring.status_writer import TrainingStatusWriter


class UltimateTrainerWrapper(BaseTrainer):
    """
    Wrapper around TrainerEngine for hero campaigns.

    This is a thin Layer 3 wrapper that:
    1. Builds TrainerConfig from hero config + campaign config
    2. Delegates to TrainerEngine.run_job()
    3. Converts TrainingResult to arena-compatible format

    Used for all hero types. The name "UltimateTrainer" is kept for
    backward compatibility with existing campaign configs.
    """

    def __init__(self, hero_config: Dict[str, Any], campaign_path: Path):
        """Initialize with hero and campaign configuration."""
        super().__init__(hero_config, campaign_path)

        # Create status writer for this campaign
        status_file = campaign_path / "status" / "training_status.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)

        model_name = hero_config.get("model", {}).get("display_name", "hero")
        self.status_writer = TrainingStatusWriter(
            str(status_file),
            max_output_tokens=2048,
            context_window=hero_config.get("training_defaults", {}).get("max_length", 2048),
            model_name=model_name
        )

    @property
    def name(self) -> str:
        return "engine"  # Using TrainerEngine now

    def train(self, data_path: Path) -> ArenaTrainingResult:
        """
        Run training using TrainerEngine.

        Args:
            data_path: Path to JSONL training data

        Returns:
            ArenaTrainingResult with success status and metrics
        """
        start_time = time.time()

        try:
            from core.paths import get_base_dir

            base_dir = get_base_dir()

            # Get model path (latest checkpoint or base model)
            model_path = self.get_model_path()

            # Build training config
            training_defaults = self.hero_config.get("training_defaults", {})

            # Look for campaign-specific config.json first, fall back to base
            campaign_config = self.campaign_path / "config.json"
            base_config = base_dir / "config.json"

            config_path = str(campaign_config) if campaign_config.exists() else str(base_config)

            # Create TrainerConfig from config file + overrides
            config, config_dict = ConfigLoader.from_file_and_defaults_with_raw(
                dataset_path=str(data_path),
                base_config=config_path,
                validate_lock=False,  # Campaigns may have different lock requirements
                **{
                    "model.model_path": str(model_path),
                    "output.output_dir": str(self.checkpoints_dir),
                    # Apply hero-specific training defaults
                    "hyperparams.batch_size": training_defaults.get("batch_size", 1),
                    "hyperparams.gradient_accumulation": training_defaults.get("gradient_accumulation", 16),
                    "hyperparams.learning_rate": training_defaults.get("learning_rate", 4e-4),
                    "hyperparams.max_length": training_defaults.get("max_length", 2048),
                    "hyperparams.save_steps": training_defaults.get("save_steps", 10000),
                    "hyperparams.warmup_steps": training_defaults.get("warmup_steps", 100),
                }
            )

            # Create MonitorContext (minimal for campaign training)
            monitors = MonitorContext(
                live_monitor=None,
                controller=None,  # Campaigns don't use daemon controller
                current_file=data_path.name,
                status_writer=self.status_writer,
            )

            # Create engine and run training
            engine = TrainerEngine(self.status_writer, verbose=True)
            result = engine.run_job(
                config=config,
                config_dict=config_dict,
                monitors=monitors,
            )

            duration = time.time() - start_time

            if result.success:
                return ArenaTrainingResult(
                    success=True,
                    steps_completed=result.global_step,
                    final_loss=result.final_loss,
                    peak_vram_gb=result.summary.get("peak_vram_gb"),
                    duration_seconds=duration,
                    checkpoint_path=Path(result.last_checkpoint_path) if result.last_checkpoint_path else None,
                    metrics=result.summary,
                )
            else:
                return ArenaTrainingResult(
                    success=False,
                    duration_seconds=duration,
                    error_message=result.error_message,
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ArenaTrainingResult(
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e),
            )
