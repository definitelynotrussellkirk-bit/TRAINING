"""UltimateTrainer wrapper for the hero loop (DIO)."""

import time
from pathlib import Path
from typing import Any, Dict

from arena.trainers.base import BaseTrainer, TrainingResult


class UltimateTrainerWrapper(BaseTrainer):
    """
    Wrapper around the existing UltimateTrainer (core/train.py).
    
    Used for smaller models like DIO (Qwen3-0.6B) that don't need
    special memory optimizations like GaLore.
    """
    
    @property
    def name(self) -> str:
        return "ultimate"
    
    def train(self, data_path: Path) -> TrainingResult:
        """Run training using UltimateTrainer."""
        start_time = time.time()
        
        try:
            from core.train import UltimateTrainer
            from core.paths import get_base_dir
            
            base_dir = get_base_dir()
            
            # Build config for UltimateTrainer
            training_defaults = self.hero_config.get("training_defaults", {})
            model_path = self.get_model_path()
            
            # UltimateTrainer expects a config dict
            config = {
                "model_path": str(model_path),
                "output_dir": str(self.checkpoints_dir),
                "data_path": str(data_path),
                "batch_size": training_defaults.get("batch_size", 1),
                "gradient_accumulation": training_defaults.get("gradient_accumulation", 16),
                "learning_rate": training_defaults.get("learning_rate", 4e-4),
                "max_length": training_defaults.get("max_length", 2048),
                "epochs": training_defaults.get("epochs", 1),
                "save_steps": training_defaults.get("save_steps", 10000),
                "warmup_steps": training_defaults.get("warmup_steps", 100),
                "gradient_checkpointing": training_defaults.get("gradient_checkpointing", True),
            }
            
            trainer = UltimateTrainer(config)
            result = trainer.train()
            
            duration = time.time() - start_time
            
            if result.get("success", False):
                return TrainingResult(
                    success=True,
                    steps_completed=result.get("steps", 0),
                    final_loss=result.get("final_loss"),
                    peak_vram_gb=result.get("peak_vram_gb"),
                    duration_seconds=duration,
                    checkpoint_path=self.checkpoints_dir,
                    metrics=result.get("metrics", {}),
                )
            else:
                return TrainingResult(
                    success=False,
                    duration_seconds=duration,
                    error_message=result.get("error", "Unknown error"),
                )
                
        except Exception as e:
            return TrainingResult(
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e),
            )
