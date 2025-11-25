"""
Data validation package.

Provides unified validation for training data.

Components:
    - DataValidator: Unified validator with configurable depth
    - ValidationLevel: QUICK, STANDARD, DEEP
    - ValidationResult: Result with errors, warnings, stats
"""

from .validator import DataValidator, ValidationLevel, ValidationResult

__all__ = ["DataValidator", "ValidationLevel", "ValidationResult"]
