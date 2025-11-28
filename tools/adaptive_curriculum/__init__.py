"""Adaptive curriculum learning system for LLM training.

Automatically adjusts training data difficulty to keep model performance
around 80% accuracy through:
- Generator stats tracking (rolling window accuracy)
- Difficulty controller (adaptive adjustment)
- Evaluation harness (periodic testing)
- Generator registry (multiple data sources)
"""

__version__ = "1.0.0"
