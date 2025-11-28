#!/usr/bin/env python3
"""
Dataset Contracts - YAML-defined schemas for training data.

Each dataset has a contract that defines:
- Schema (field names and types)
- Required/optional fields
- Validation rules (invariants)
- Max invalid fraction allowed
- Linked skill for leakage detection

Contracts are loaded from configs/datasets/*.yaml

Usage:
    from forge.contracts import get_contract, list_contracts

    contract = get_contract("binary_training_v1")
    if contract:
        print(contract.schema)
        print(contract.skill_id)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

logger = logging.getLogger(__name__)

# Version tracking
CONTRACTS_VERSION = "1.0.0"


@dataclass
class DatasetContract:
    """
    Machine-readable specification for a dataset.

    Loaded from YAML files in configs/datasets/
    """
    # Identity
    id: str
    version: str
    description: str

    # Format
    kind: str = "jsonl"  # "jsonl" | "parquet"

    # Schema contract
    schema: Dict[str, str] = field(default_factory=dict)  # field -> type
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)

    # Validation rules
    invariants: List[str] = field(default_factory=list)  # Human-readable
    max_invalid_fraction: float = 0.01  # 1% default

    # Safety
    safety_profile: str = "internal"  # "internal" | "public" | "restricted"

    # Skill linkage (for leakage detection)
    skill_id: Optional[str] = None
    eval_leakage_guard: Optional[str] = None  # Eval bank name

    # Tags for organization
    task_tags: List[str] = field(default_factory=list)

    # Source tracking
    source_file: Optional[str] = None

    def validate_example(self, example: Dict[str, Any]) -> List[str]:
        """
        Validate a single example against this contract.

        Returns list of error messages (empty if valid).
        """
        errors = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in example:
                errors.append(f"Missing required field: {field_name}")
            elif example[field_name] is None:
                errors.append(f"Required field is null: {field_name}")

        # Check types
        type_validators = {
            "str": lambda x: isinstance(x, str),
            "int": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "bool": lambda x: isinstance(x, bool),
            "list": lambda x: isinstance(x, list),
            "list[str]": lambda x: isinstance(x, list) and all(isinstance(i, str) for i in x),
            "dict": lambda x: isinstance(x, dict),
            "any": lambda x: True,
        }

        for field_name, expected_type in self.schema.items():
            if field_name in example and example[field_name] is not None:
                validator = type_validators.get(expected_type, lambda x: True)
                if not validator(example[field_name]):
                    errors.append(
                        f"Type mismatch for '{field_name}': expected {expected_type}, "
                        f"got {type(example[field_name]).__name__}"
                    )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "kind": self.kind,
            "schema": self.schema,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "invariants": self.invariants,
            "max_invalid_fraction": self.max_invalid_fraction,
            "safety_profile": self.safety_profile,
            "skill_id": self.skill_id,
            "eval_leakage_guard": self.eval_leakage_guard,
            "task_tags": self.task_tags,
        }


class ContractRegistry:
    """
    Registry of all dataset contracts.

    Loads contracts from YAML files in configs/datasets/
    """

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            try:
                from core.paths import get_base_dir
                config_dir = get_base_dir() / "configs" / "datasets"
            except ImportError:
                config_dir = Path("configs/datasets")

        self.config_dir = Path(config_dir)
        self._contracts: Dict[str, DatasetContract] = {}
        self._loaded = False

    def _load_contracts(self):
        """Load all contracts from YAML files."""
        if self._loaded:
            return

        self._contracts = {}

        if not self.config_dir.exists():
            logger.warning(f"Contract directory not found: {self.config_dir}")
            self._loaded = True
            return

        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)

                if not data:
                    continue

                # Handle both single contract and list of contracts
                if isinstance(data, list):
                    contracts_data = data
                else:
                    contracts_data = [data]

                for contract_data in contracts_data:
                    if "id" not in contract_data:
                        logger.warning(f"Contract in {yaml_file} missing 'id' field")
                        continue

                    contract = DatasetContract(
                        id=contract_data["id"],
                        version=contract_data.get("version", "1"),
                        description=contract_data.get("description", ""),
                        kind=contract_data.get("kind", "jsonl"),
                        schema=contract_data.get("schema", {}),
                        required_fields=contract_data.get("required_fields", []),
                        optional_fields=contract_data.get("optional_fields", []),
                        invariants=contract_data.get("invariants", []),
                        max_invalid_fraction=contract_data.get("max_invalid_fraction", 0.01),
                        safety_profile=contract_data.get("safety_profile", "internal"),
                        skill_id=contract_data.get("skill_id"),
                        eval_leakage_guard=contract_data.get("eval_leakage_guard"),
                        task_tags=contract_data.get("task_tags", []),
                        source_file=str(yaml_file),
                    )

                    self._contracts[contract.id] = contract
                    logger.debug(f"Loaded contract: {contract.id} from {yaml_file.name}")

            except Exception as e:
                logger.error(f"Failed to load contract from {yaml_file}: {e}")

        logger.info(f"Loaded {len(self._contracts)} dataset contracts")
        self._loaded = True

    def get(self, contract_id: str) -> Optional[DatasetContract]:
        """Get contract by ID."""
        self._load_contracts()
        return self._contracts.get(contract_id)

    def list_all(self) -> List[DatasetContract]:
        """List all contracts."""
        self._load_contracts()
        return list(self._contracts.values())

    def get_by_skill(self, skill_id: str) -> List[DatasetContract]:
        """Get all contracts linked to a skill."""
        self._load_contracts()
        return [c for c in self._contracts.values() if c.skill_id == skill_id]

    def reload(self):
        """Force reload contracts from disk."""
        self._loaded = False
        self._load_contracts()


# Global registry instance
_registry: Optional[ContractRegistry] = None


def get_registry() -> ContractRegistry:
    """Get the global contract registry."""
    global _registry
    if _registry is None:
        _registry = ContractRegistry()
    return _registry


def get_contract(contract_id: str) -> Optional[DatasetContract]:
    """Get a contract by ID."""
    return get_registry().get(contract_id)


def list_contracts() -> List[DatasetContract]:
    """List all contracts."""
    return get_registry().list_all()


def get_contracts_for_skill(skill_id: str) -> List[DatasetContract]:
    """Get all contracts linked to a skill."""
    return get_registry().get_by_skill(skill_id)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Dataset Contracts")
    print("=" * 60)

    contracts = list_contracts()

    if not contracts:
        print("No contracts found.")
        print(f"Add YAML files to: configs/datasets/")
        sys.exit(0)

    for contract in contracts:
        print(f"\n{contract.id} (v{contract.version})")
        print(f"  Description: {contract.description[:60]}...")
        print(f"  Skill: {contract.skill_id or 'none'}")
        print(f"  Required fields: {contract.required_fields}")
        print(f"  Max invalid: {contract.max_invalid_fraction:.1%}")
