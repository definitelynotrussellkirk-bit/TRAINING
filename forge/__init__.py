"""
Data Forge - Validation, gating, and lifecycle management for training data.

The Forge system provides:
- Dataset contracts (YAML-defined schemas)
- Shard state tracking (validation status per file)
- Validation pipeline (schema, content, leakage)
- Training gating (only use validated data)
- Profiling (distributional statistics)

Usage:
    # Validate a file
    from forge import validate_file
    result = validate_file(Path("data.jsonl"), skill_id="bin")

    # Get validated shards for training
    from forge.gating import get_training_shards
    shards = get_training_shards("binary_training_v1")

    # Register and track shards
    from forge.state import get_forge_state
    state = get_forge_state()
    state.register_shard("binary_training_v1", "shard_001.jsonl", "/path")

    # Load dataset contracts
    from forge.contracts import list_contracts
    for contract in list_contracts():
        print(contract.id, contract.skill_id)
"""

# Core validation
from forge.validator import (
    validate_file,
    validate_for_queue,
    ValidationResult,
    ForgeValidator,
)

# Eval leakage detection
from forge.leakage import EvalBankManager

# Dataset contracts
from forge.contracts import (
    DatasetContract,
    get_contract,
    list_contracts,
    get_contracts_for_skill,
    get_registry as get_contract_registry,
)

# Shard state
from forge.state import (
    ShardStatus,
    ShardInfo,
    DatasetState,
    ForgeState,
    get_forge_state,
)

# Training gating
from forge.gating import (
    get_training_shards,
    get_all_training_shards,
    can_train_on,
    validate_before_training,
    ValidationRequired,
)

# Profiling
from forge.profiler import (
    profile_shard,
    ShardProfile,
)

# Queue integration
from forge.queue_integration import (
    validated_add_to_queue,
    validated_process_inbox,
    get_forge_queue_status,
)

__all__ = [
    # Validation
    "validate_file",
    "validate_for_queue",
    "ValidationResult",
    "ForgeValidator",
    # Leakage
    "EvalBankManager",
    # Contracts
    "DatasetContract",
    "get_contract",
    "list_contracts",
    "get_contracts_for_skill",
    "get_contract_registry",
    # State
    "ShardStatus",
    "ShardInfo",
    "DatasetState",
    "ForgeState",
    "get_forge_state",
    # Gating
    "get_training_shards",
    "get_all_training_shards",
    "can_train_on",
    "validate_before_training",
    "ValidationRequired",
    # Profiling
    "profile_shard",
    "ShardProfile",
    # Queue
    "validated_add_to_queue",
    "validated_process_inbox",
    "get_forge_queue_status",
]

FORGE_VERSION = "2.0.0"
