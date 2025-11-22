#!/usr/bin/env python3
"""
Training Engine

Clean API-style trainer that orchestrates all training operations.

Usage:
    from trainer.core import TrainerEngine
    from trainer.config import TrainerConfig
    from trainer.monitoring import TrainingStatusWriter

    config = TrainerConfig(...)
    status_writer = TrainingStatusWriter("status/training_status.json")
    engine = TrainerEngine(status_writer)
    result = engine.run_job(config)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trainer.config.schema import TrainerConfig
from trainer.monitoring.status_writer import TrainingStatusWriter


@dataclass
class TrainingResult:
    """Result of a training job"""
    success: bool
    global_step: int
    runtime_sec: float
    last_checkpoint_path: Optional[str]
    final_loss: float
    summary: Dict[str, Any]
    error_message: Optional[str] = None


class TrainerEngine:
    """
    Core training engine - stable API surface.

    This is the ONLY public API for training. All training goes through
    engine.run_job(config).

    Architecture:
    - Load profile (emoji_think, regime3, etc.)
    - Load model & tokenizer
    - Load & transform dataset using profile
    - Setup HF Trainer with monitoring callbacks
    - Execute training
    - Save final checkpoint
    - Return result
    """

    def __init__(self, status_writer: TrainingStatusWriter):
        """
        Initialize TrainerEngine.

        Args:
            status_writer: Status writer for UI updates
        """
        self.status_writer = status_writer

    def run_job(self, config: TrainerConfig) -> TrainingResult:
        """
        Execute a training job.

        This is the ONLY public method. All training goes through here.

        Args:
            config: Complete training configuration

        Returns:
            TrainingResult with success status and metrics

        Architecture Flow:
            1. Validate config
            2. Load profile (from config.profile.name)
            3. Load model & tokenizer
            4. Load dataset & apply profile transformations
            5. Setup HF Trainer with monitoring callbacks
            6. Execute training
            7. Save final checkpoint
            8. Return result
        """
        print("\n" + "=" * 80)
        print("🚀 TRAINER ENGINE - API-DRIVEN TRAINING")
        print("=" * 80)
        print(f"Profile: {config.profile.name}")
        print(f"Model: {config.model.model_path}")
        print(f"Dataset: {config.data.dataset_path}")
        print(f"Output: {config.output.output_dir}")
        print("=" * 80 + "\n")

        # NOTE: Full implementation would go here
        # For now, this is a proof-of-concept showing the clean API

        # The actual implementation would:
        # 1. profile = get_profile(config.profile.name)
        # 2. model, tokenizer = load_model(config.model)
        # 3. dataset = load_dataset(config.data)
        # 4. dataset = [profile.transform_example(ex, i, prompt) for i, ex in enumerate(dataset)]
        # 5. trainer = create_trainer(config, model, dataset, callbacks)
        # 6. trainer.train()
        # 7. trainer.save_model()
        # 8. return TrainingResult(...)

        print("⚠️  TrainerEngine.run_job() is a proof-of-concept API")
        print("   Full implementation would orchestrate:")
        print("   1. Profile loading")
        print("   2. Model loading")
        print("   3. Dataset transformation")
        print("   4. HF Trainer setup")
        print("   5. Training execution")
        print("   6. Checkpoint saving")
        print("\n   For production training, use core/train.py (existing system)")
        print("   This engine demonstrates the clean API architecture.\n")

        # Return mock result
        return TrainingResult(
            success=False,
            global_step=0,
            runtime_sec=0.0,
            last_checkpoint_path=None,
            final_loss=0.0,
            summary={"status": "proof_of_concept"},
            error_message="TrainerEngine.run_job() is a proof-of-concept. Use core/train.py for actual training."
        )


__all__ = ["TrainerEngine", "TrainingResult"]
