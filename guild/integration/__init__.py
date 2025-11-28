"""
Integration adapters for existing systems.

The integration module bridges guild framework with existing infrastructure:
- TrainingDataAdapter: Convert quests to JSONL training data
- InferenceAdapter: Execute quests against the model (3090 server)
- CurriculumAdapter: Sync skill levels with curriculum system
- QueueAdapter: Submit training files to queue

Example:
    from guild.integration import (
        AdapterConfig,
        get_training_adapter,
        get_inference_adapter,
        get_curriculum_adapter,
        get_queue_adapter,
    )

    # Configure adapters (using hosts.json for defaults)
    from core.paths import get_base_dir
    from core.hosts import get_host
    inference = get_host("3090")
    config = AdapterConfig(
        base_dir=get_base_dir(),
        inference_host=inference.host if inference else "localhost",
        inference_port=8765,
    )

    # Create training data
    training = get_training_adapter()
    example = training.create_example_from_quest(...)
    training.buffer_example(example)
    result = training.write_batch()

    # Execute quest against model
    inference = get_inference_adapter()
    result = inference.execute_quest(quest)
    model_answer = result.data.model_answer

    # Track progress
    curriculum = get_curriculum_adapter()
    curriculum.record_accuracy("syllo", accuracy=0.85, step=1000)
    curriculum.progress_if_ready("syllo")

    # Submit to queue
    queue = get_queue_adapter()
    queue.submit_to_queue(training_file, priority="high")
"""

# Base adapter infrastructure
from guild.integration.adapters import (
    AdapterStatus,
    AdapterConfig,
    AdapterResult,
    BaseAdapter,
    CompositeAdapter,
    register_adapter,
    get_adapter,
    list_adapters,
    reset_adapters,
)

# Training data adapter
from guild.integration.training_adapter import (
    TrainingExample,
    StanceFormatter,
    TrainingDataAdapter,
    init_training_adapter,
    get_training_adapter,
    reset_training_adapter,
)

# Inference adapter
from guild.integration.inference_adapter import (
    InferenceRequest,
    InferenceResponse,
    QuestExecution,
    InferenceAdapter,
    init_inference_adapter,
    get_inference_adapter,
    reset_inference_adapter,
)

# Curriculum adapter
from guild.integration.curriculum_adapter import (
    CurriculumSkillState,
    CurriculumState,
    CurriculumAdapter,
    init_curriculum_adapter,
    get_curriculum_adapter,
    reset_curriculum_adapter,
    SKILL_CONFIGS,
)

# Queue adapter
from guild.integration.queue_adapter import (
    QueueStatus,
    SubmissionResult,
    QueueAdapter,
    init_queue_adapter,
    get_queue_adapter,
    reset_queue_adapter,
    PRIORITY_MAP,
)

__all__ = [
    # Base
    "AdapterStatus",
    "AdapterConfig",
    "AdapterResult",
    "BaseAdapter",
    "CompositeAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "reset_adapters",
    # Training
    "TrainingExample",
    "StanceFormatter",
    "TrainingDataAdapter",
    "init_training_adapter",
    "get_training_adapter",
    "reset_training_adapter",
    # Inference
    "InferenceRequest",
    "InferenceResponse",
    "QuestExecution",
    "InferenceAdapter",
    "init_inference_adapter",
    "get_inference_adapter",
    "reset_inference_adapter",
    # Curriculum
    "CurriculumSkillState",
    "CurriculumState",
    "CurriculumAdapter",
    "init_curriculum_adapter",
    "get_curriculum_adapter",
    "reset_curriculum_adapter",
    "SKILL_CONFIGS",
    # Queue
    "QueueStatus",
    "SubmissionResult",
    "QueueAdapter",
    "init_queue_adapter",
    "get_queue_adapter",
    "reset_queue_adapter",
    "PRIORITY_MAP",
]
