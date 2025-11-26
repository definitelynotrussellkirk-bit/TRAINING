#!/usr/bin/env python3
"""
Dataset Spec Validation - Deny-by-default schema gate.

This module implements a two-layer validation strategy:
1. SpecValidator (this module): Checks job config + dataset metadata match a known spec
2. DataValidator (validator.py): Checks content quality (QUICK/STANDARD/DEEP)

The key principle: NO TRAINING without a declared, known schema.

Flow:
    Job arrives → SpecValidator (metadata/schema) → DataValidator (content)
                → if both pass → enqueue for training

Usage:
    from core.validation.spec import SpecValidator, DATASET_SPECS

    validator = SpecValidator(DATASET_SPECS)

    # Validate a job
    try:
        spec = validator.validate_job(job_config, dataset_metadata)
        # spec.build_trainer_components(job_config) → trainer kwargs
    except SpecValidationError as e:
        # Reject job with clear reason
        print(f"Rejected: {e}")
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Any

logger = logging.getLogger(__name__)

# =============================================================================
# DATA LINEAGE - Validator identification for tracking
# =============================================================================
# Bump SPEC_VALIDATOR_VERSION when validation logic changes significantly
VALIDATOR_NAME = "spec_validator"
VALIDATOR_VERSION = "1.0.0"


class SpecValidationError(Exception):
    """Raised when spec validation fails. Job should not proceed."""
    pass


@dataclass(frozen=True)
class DatasetSpec:
    """
    Defines a valid dataset schema that the trainer understands.

    Each spec declares:
    - What must exist in job config
    - What must exist in dataset metadata/rows
    - How to build trainer components for this data type

    Without a matching spec, training is refused.
    """
    id: str                         # e.g. "chat_sft_v1"
    kind: Literal["chat_sft", "reward", "dpo", "tool_use", "completion"]
    description: str

    # What must exist in the job config
    required_job_keys: List[str] = field(default_factory=list)

    # What must exist in dataset metadata (first row or sidecar JSON)
    required_metadata_keys: List[str] = field(default_factory=list)

    # Required columns/fields in each data row
    required_columns: List[str] = field(default_factory=list)

    # Optional columns (documented but not enforced)
    optional_columns: List[str] = field(default_factory=list)

    # Hook to build trainer components from this spec
    # Returns dict of kwargs for trainer construction
    build_trainer_components: Optional[Callable[[Dict], Dict]] = None

    def get_trainer_kwargs(self, job_config: Dict) -> Dict:
        """Get trainer configuration for this spec."""
        if self.build_trainer_components:
            return self.build_trainer_components(job_config)
        return {}


# =============================================================================
# TRAINER COMPONENT BUILDERS
# =============================================================================

def build_chat_sft_components(job_config: Dict) -> Dict:
    """Build trainer components for standard chat SFT."""
    return {
        "loss_type": "language_modeling",
        "collator_type": "chat",
        "max_length": job_config.get("max_length", 4096),
        "response_template": None,  # Use chat template
    }


def build_completion_components(job_config: Dict) -> Dict:
    """Build trainer components for completion/text data."""
    return {
        "loss_type": "language_modeling",
        "collator_type": "completion",
        "max_length": job_config.get("max_length", 4096),
    }


# =============================================================================
# SPEC REGISTRY - Known dataset schemas
# =============================================================================

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "chat_sft_v1": DatasetSpec(
        id="chat_sft_v1",
        kind="chat_sft",
        description="Standard chat SFT: messages[] array with {role, content} objects",
        required_job_keys=["dataset_path", "base_model"],
        required_metadata_keys=[],  # Relaxed for now
        required_columns=["messages"],
        optional_columns=["system_prompt", "tags", "difficulty"],
        build_trainer_components=build_chat_sft_components,
    ),

    "syllo_v1": DatasetSpec(
        id="syllo_v1",
        kind="chat_sft",
        description="Syllogism reasoning data: messages[] with symbolic reasoning",
        required_job_keys=["dataset_path", "base_model"],
        required_metadata_keys=[],
        required_columns=["messages"],
        optional_columns=["difficulty", "category", "variant"],
        build_trainer_components=build_chat_sft_components,
    ),

    "completion_v1": DatasetSpec(
        id="completion_v1",
        kind="completion",
        description="Raw text completion: text field only",
        required_job_keys=["dataset_path", "base_model"],
        required_metadata_keys=[],
        required_columns=["text"],
        optional_columns=[],
        build_trainer_components=build_completion_components,
    ),
}

# Default spec for backward compatibility (can be disabled)
DEFAULT_SPEC_ID = "chat_sft_v1"


# =============================================================================
# SPEC VALIDATOR
# =============================================================================

class SpecValidator:
    """
    Validates that jobs declare a known schema before training.

    This is the "outer gate" - if spec validation fails, no training happens.
    Content validation (DataValidator) only runs after spec is accepted.
    """

    def __init__(
        self,
        registry: Dict[str, DatasetSpec] = None,
        allow_default: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize spec validator.

        Args:
            registry: Dict of schema_id → DatasetSpec. Defaults to DATASET_SPECS.
            allow_default: If True, missing schema_id uses DEFAULT_SPEC_ID.
                          If False, missing schema_id is an error.
            strict_mode: If True, also check required_metadata_keys.
        """
        self._registry = registry or DATASET_SPECS
        self._allow_default = allow_default
        self._strict_mode = strict_mode

    def validate_job(
        self,
        job_config: Dict,
        metadata: Optional[Dict] = None
    ) -> DatasetSpec:
        """
        Validate a job against the spec registry.

        Args:
            job_config: Job configuration dict (from config.json or CLI)
            metadata: Optional dataset metadata (first row or sidecar file)

        Returns:
            DatasetSpec if validation passes

        Raises:
            SpecValidationError if validation fails
        """
        metadata = metadata or {}

        # 1) Get schema_id (from job config or metadata)
        schema_id = job_config.get("schema_id") or metadata.get("schema_id")

        if not schema_id:
            if self._allow_default:
                schema_id = DEFAULT_SPEC_ID
                logger.info(f"No schema_id specified, using default: {schema_id}")
            else:
                raise SpecValidationError(
                    "Job is missing 'schema_id'. "
                    "Add schema_id to job config or set allow_default=True."
                )

        # 2) Schema must be known
        spec = self._registry.get(schema_id)
        if not spec:
            known = list(self._registry.keys())
            raise SpecValidationError(
                f"Unknown schema_id='{schema_id}'. "
                f"Known schemas: {known}. "
                f"Register a new spec or fix the schema_id."
            )

        # 3) Check required job config keys
        missing_job = [k for k in spec.required_job_keys if k not in job_config]
        if missing_job:
            raise SpecValidationError(
                f"Job config missing required keys for {schema_id}: {missing_job}"
            )

        # 4) Check required metadata keys (if strict mode)
        if self._strict_mode and spec.required_metadata_keys:
            missing_meta = [k for k in spec.required_metadata_keys if k not in metadata]
            if missing_meta:
                raise SpecValidationError(
                    f"Dataset metadata missing required keys for {schema_id}: {missing_meta}"
                )

        logger.info(f"Spec validation passed: {schema_id} ({spec.kind})")
        return spec

    def validate_row(self, row: Dict, spec: DatasetSpec) -> List[str]:
        """
        Validate a single data row against spec requirements.

        Args:
            row: A single example from the dataset
            spec: The DatasetSpec to validate against

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        for col in spec.required_columns:
            if col not in row:
                errors.append(f"Missing required column: {col}")
            elif col == "messages":
                # Special validation for messages column
                messages = row.get("messages", [])
                if not isinstance(messages, list):
                    errors.append("'messages' must be a list")
                elif len(messages) == 0:
                    errors.append("'messages' cannot be empty")
                else:
                    # Check message structure
                    for i, msg in enumerate(messages):
                        if not isinstance(msg, dict):
                            errors.append(f"messages[{i}] must be a dict")
                        elif "role" not in msg:
                            errors.append(f"messages[{i}] missing 'role'")
                        elif "content" not in msg:
                            errors.append(f"messages[{i}] missing 'content'")

        return errors

    def get_spec(self, schema_id: str) -> Optional[DatasetSpec]:
        """Get a spec by ID, or None if not found."""
        return self._registry.get(schema_id)

    def list_specs(self) -> List[str]:
        """List all registered spec IDs."""
        return list(self._registry.keys())

    def get_lineage_info(self) -> Dict[str, str]:
        """Return validator lineage info for tracking."""
        return {
            "validator_name": VALIDATOR_NAME,
            "validator_version": VALIDATOR_VERSION,
        }


def extract_dataset_metadata(dataset_path: Path) -> Dict:
    """
    Extract metadata from a dataset file.

    Tries:
    1. Sidecar file: {dataset_path}.meta.json
    2. First row of JSONL if it has a "metadata" or "schema_id" key

    Returns:
        Dict of metadata (may be empty)
    """
    metadata = {}
    dataset_path = Path(dataset_path)

    # Try sidecar metadata file
    meta_path = dataset_path.with_suffix(dataset_path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
                logger.debug(f"Loaded metadata from {meta_path}")
                return metadata
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_path}: {e}")

    # Try first row of JSONL
    if dataset_path.suffix == ".jsonl" and dataset_path.exists():
        try:
            with open(dataset_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_row = json.loads(first_line)
                    # Check if first row is metadata-only
                    if "metadata" in first_row:
                        metadata = first_row["metadata"]
                    elif "schema_id" in first_row and "messages" not in first_row:
                        # First row looks like metadata, not data
                        metadata = first_row
        except Exception as e:
            logger.debug(f"Could not extract metadata from first row: {e}")

    return metadata


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_job_config(job_config: Dict, allow_default: bool = True) -> DatasetSpec:
    """
    Quick validation of job config against default registry.

    Args:
        job_config: Job configuration dict
        allow_default: If True, missing schema_id uses default

    Returns:
        DatasetSpec if valid

    Raises:
        SpecValidationError if invalid
    """
    validator = SpecValidator(allow_default=allow_default)
    return validator.validate_job(job_config)


def get_spec_for_schema(schema_id: str) -> Optional[DatasetSpec]:
    """Get spec by ID from default registry."""
    return DATASET_SPECS.get(schema_id)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Registered specs:")
    for spec_id, spec in DATASET_SPECS.items():
        print(f"  {spec_id}: {spec.description}")

    print("\nTesting validation...")

    # Test 1: Valid job config
    job = {"dataset_path": "data/train.jsonl", "base_model": "Qwen3-0.6B"}
    try:
        spec = validate_job_config(job)
        print(f"✅ Valid: {spec.id}")
    except SpecValidationError as e:
        print(f"❌ Invalid: {e}")

    # Test 2: Unknown schema
    job2 = {"schema_id": "unknown_v99", "dataset_path": "x", "base_model": "y"}
    try:
        spec = validate_job_config(job2)
        print(f"✅ Valid: {spec.id}")
    except SpecValidationError as e:
        print(f"✅ Correctly rejected: {e}")

    # Test 3: Strict mode without schema_id
    validator = SpecValidator(allow_default=False)
    job3 = {"dataset_path": "x", "base_model": "y"}
    try:
        spec = validator.validate_job(job3)
        print(f"❌ Should have rejected")
    except SpecValidationError as e:
        print(f"✅ Correctly rejected (strict): {e}")

    print("\nSpec validation module ready!")
