#!/usr/bin/env python3
"""
Core Training Engine - Fully Implemented!

Clean API-driven training orchestration.

Usage:
    from trainer.core import TrainerEngine, TrainingResult
    from trainer.config import create_default_config
    from trainer.monitoring import TrainingStatusWriter

    config = create_default_config(...)
    status_writer = TrainingStatusWriter("status.json")
    engine = TrainerEngine(status_writer)
    result = engine.run_job(config)

    if result.success:
        print(f"Training complete! Final loss: {result.final_loss}")
"""

from trainer.core.engine import TrainerEngine, TrainingResult

__all__ = ["TrainerEngine", "TrainingResult"]
