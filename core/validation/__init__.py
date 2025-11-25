"""
Data validation package.

Provides two-layer validation for training data:

Layer 1 - Spec Validation (outer gate):
    - SpecValidator: Ensures job declares a known schema
    - DatasetSpec: Defines valid dataset schemas
    - DATASET_SPECS: Registry of known schemas

Layer 2 - Content Validation (inner checks):
    - DataValidator: Unified validator with configurable depth
    - ValidationLevel: QUICK, STANDARD, DEEP
    - ValidationResult: Result with errors, warnings, stats

Flow:
    Job → SpecValidator (schema gate) → DataValidator (content) → Training
"""

from .validator import DataValidator, ValidationLevel, ValidationResult
from .spec import (
    DatasetSpec,
    SpecValidator,
    SpecValidationError,
    DATASET_SPECS,
    DEFAULT_SPEC_ID,
    validate_job_config,
    extract_dataset_metadata,
)

__all__ = [
    # Content validation
    "DataValidator",
    "ValidationLevel",
    "ValidationResult",
    # Spec validation
    "DatasetSpec",
    "SpecValidator",
    "SpecValidationError",
    "DATASET_SPECS",
    "DEFAULT_SPEC_ID",
    "validate_job_config",
    "extract_dataset_metadata",
]
