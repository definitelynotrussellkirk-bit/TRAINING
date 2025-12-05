"""
Training Schools Data Formats
=============================

Defines the data formats required for each Training School:

- School of the Scribe (SFT): Standard messages format
- School of the Mirror (Sparring): Self-correction format
- School of the Judge (DPO): Preference pair format
- School of the Champion (RLHF): Reward-labeled format
- School of the Whisper (Distillation): Teacher-student format

Each format includes:
1. Schema definition (dataclass)
2. Validation function
3. Conversion utilities
4. Example data

Usage:
    from guild.training_schools.data_formats import (
        SFTSample,
        DPOPair,
        RLHFSample,
        validate_format,
        convert_to_format,
    )

    # Create a DPO preference pair
    pair = DPOPair(
        prompt="What is 2+2?",
        chosen="4",
        rejected="5",
        skill_id="math",
        level=1,
    )

    # Validate format
    assert validate_format(pair)

    # Convert SFT sample to DPO pair
    dpo_pair = convert_to_format(sft_sample, target="dpo")
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal, Union
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SCHOOL OF THE SCRIBE (SFT) - Standard Supervised Fine-Tuning
# =============================================================================

@dataclass
class SFTSample:
    """
    Standard SFT training sample.

    This is the basic format for supervised fine-tuning.
    Maps directly to HuggingFace's messages format.

    Example:
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ],
            "metadata": {"skill_id": "math", "level": 1}
        }
    """
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields for tracking
    skill_id: Optional[str] = None
    level: Optional[int] = None
    source: Optional[str] = None  # "generated", "human", "synthetic"

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "messages": self.messages,
            "metadata": {
                **self.metadata,
                "skill_id": self.skill_id,
                "level": self.level,
                "source": self.source,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SFTSample":
        """Create from dict."""
        meta = data.get("metadata", {})
        return cls(
            messages=data["messages"],
            metadata=meta,
            skill_id=meta.get("skill_id"),
            level=meta.get("level"),
            source=meta.get("source"),
        )

    @property
    def prompt(self) -> str:
        """Extract the user prompt."""
        for msg in self.messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @property
    def response(self) -> str:
        """Extract the assistant response."""
        for msg in self.messages:
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""


# =============================================================================
# SCHOOL OF THE MIRROR (Sparring) - Self-Correction Training
# =============================================================================

@dataclass
class SparringSample:
    """
    Sparring (self-correction) training sample.

    Three-stage format:
    1. identify_incorrect: Model identifies a wrong answer
    2. correction: Model provides the correct answer
    3. confirm_correct: Model confirms a correct answer

    Example:
        {
            "stage": "correction",
            "messages": [
                {"role": "user", "content": "Previous answer was wrong: 5. Correct it."},
                {"role": "assistant", "content": "The correct answer is 4."}
            ],
            "original_prompt": "What is 2+2?",
            "incorrect_answer": "5",
            "correct_answer": "4",
            "metadata": {"skill_id": "math", "level": 1}
        }
    """
    stage: Literal["identify_incorrect", "correction", "confirm_correct"]
    messages: list[dict[str, str]]
    original_prompt: str
    correct_answer: str
    incorrect_answer: Optional[str] = None  # Not needed for confirm_correct
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking fields
    skill_id: Optional[str] = None
    level: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "messages": self.messages,
            "original_prompt": self.original_prompt,
            "correct_answer": self.correct_answer,
            "incorrect_answer": self.incorrect_answer,
            "metadata": {
                **self.metadata,
                "skill_id": self.skill_id,
                "level": self.level,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SparringSample":
        meta = data.get("metadata", {})
        return cls(
            stage=data["stage"],
            messages=data["messages"],
            original_prompt=data["original_prompt"],
            correct_answer=data["correct_answer"],
            incorrect_answer=data.get("incorrect_answer"),
            metadata=meta,
            skill_id=meta.get("skill_id"),
            level=meta.get("level"),
        )


# =============================================================================
# SCHOOL OF THE JUDGE (DPO) - Direct Preference Optimization
# =============================================================================

@dataclass
class DPOPair:
    """
    DPO preference pair for Direct Preference Optimization.

    Contains a prompt with a chosen (preferred) and rejected response.
    The model learns to prefer chosen over rejected.

    Example:
        {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
            "system": "You are a helpful math assistant.",
            "metadata": {"skill_id": "math", "level": 1, "margin": 0.9}
        }

    The margin field indicates how much better chosen is than rejected:
    - margin > 0.8: Strong preference (clear winner)
    - margin 0.5-0.8: Moderate preference
    - margin < 0.5: Weak preference (both similar)
    """
    prompt: str
    chosen: str
    rejected: str
    system: Optional[str] = None
    margin: float = 1.0  # How much better is chosen? (0.0-1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking fields
    skill_id: Optional[str] = None
    level: Optional[int] = None
    chosen_source: Optional[str] = None  # "golden", "model", "human"
    rejected_source: Optional[str] = None  # "model", "synthetic"

    def to_dict(self) -> dict:
        """Convert to standard DPO format for training libraries."""
        result = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": {
                **self.metadata,
                "skill_id": self.skill_id,
                "level": self.level,
                "margin": self.margin,
                "chosen_source": self.chosen_source,
                "rejected_source": self.rejected_source,
            },
        }
        if self.system:
            result["system"] = self.system
        return result

    def to_messages_format(self) -> dict:
        """
        Convert to messages format for libraries that expect it.

        Returns:
            {
                "prompt": [{"role": "user", "content": "..."}],
                "chosen": [{"role": "assistant", "content": "..."}],
                "rejected": [{"role": "assistant", "content": "..."}]
            }
        """
        prompt_msgs = []
        if self.system:
            prompt_msgs.append({"role": "system", "content": self.system})
        prompt_msgs.append({"role": "user", "content": self.prompt})

        return {
            "prompt": prompt_msgs,
            "chosen": [{"role": "assistant", "content": self.chosen}],
            "rejected": [{"role": "assistant", "content": self.rejected}],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DPOPair":
        meta = data.get("metadata", {})
        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            system=data.get("system"),
            margin=meta.get("margin", 1.0),
            metadata=meta,
            skill_id=meta.get("skill_id"),
            level=meta.get("level"),
            chosen_source=meta.get("chosen_source"),
            rejected_source=meta.get("rejected_source"),
        )

    @classmethod
    def from_sft_and_model(
        cls,
        sft_sample: SFTSample,
        model_answer: str,
        model_is_correct: bool,
    ) -> "DPOPair":
        """
        Create DPO pair from SFT sample and model's answer.

        If model is wrong, golden answer is chosen and model answer is rejected.
        If model is right but less elegant, still use golden as chosen.
        """
        golden_answer = sft_sample.response

        if model_is_correct:
            # Model got it right - could still use for margin training
            # but typically skip these
            return cls(
                prompt=sft_sample.prompt,
                chosen=golden_answer,
                rejected=model_answer,
                margin=0.1,  # Low margin - both acceptable
                skill_id=sft_sample.skill_id,
                level=sft_sample.level,
                chosen_source="golden",
                rejected_source="model",
            )
        else:
            # Model got it wrong - strong preference signal
            return cls(
                prompt=sft_sample.prompt,
                chosen=golden_answer,
                rejected=model_answer,
                margin=1.0,  # High margin - clear winner
                skill_id=sft_sample.skill_id,
                level=sft_sample.level,
                chosen_source="golden",
                rejected_source="model",
            )


# =============================================================================
# SCHOOL OF THE CHAMPION (RLHF) - Reinforcement Learning from Human Feedback
# =============================================================================

@dataclass
class RLHFSample:
    """
    RLHF training sample with reward signal.

    Contains a prompt, response, and reward score.
    The reward model learns to predict rewards, then policy is optimized.

    Example:
        {
            "prompt": "What is 2+2?",
            "response": "4",
            "reward": 1.0,
            "reward_source": "golden_match",
            "metadata": {"skill_id": "math", "level": 1}
        }

    Reward scales:
    - reward = 1.0: Correct/preferred response
    - reward = 0.5: Partially correct
    - reward = 0.0: Incorrect/rejected response
    - reward = -1.0: Harmful/very wrong (optional negative rewards)
    """
    prompt: str
    response: str
    reward: float  # -1.0 to 1.0 typically
    system: Optional[str] = None
    reward_source: Optional[str] = None  # "human", "golden_match", "model"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking fields
    skill_id: Optional[str] = None
    level: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "metadata": {
                **self.metadata,
                "skill_id": self.skill_id,
                "level": self.level,
                "reward_source": self.reward_source,
            },
        }
        if self.system:
            result["system"] = self.system
        return result

    def to_messages_format(self) -> dict:
        """Convert to messages format with reward."""
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.prompt})
        msgs.append({"role": "assistant", "content": self.response})

        return {
            "messages": msgs,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RLHFSample":
        meta = data.get("metadata", {})
        return cls(
            prompt=data["prompt"],
            response=data["response"],
            reward=data["reward"],
            system=data.get("system"),
            reward_source=meta.get("reward_source"),
            metadata=meta,
            skill_id=meta.get("skill_id"),
            level=meta.get("level"),
        )

    @classmethod
    def from_sft_with_eval(
        cls,
        sft_sample: SFTSample,
        model_answer: str,
        is_correct: bool,
    ) -> "RLHFSample":
        """Create RLHF sample from SFT sample with correctness evaluation."""
        return cls(
            prompt=sft_sample.prompt,
            response=model_answer,
            reward=1.0 if is_correct else 0.0,
            reward_source="golden_match",
            skill_id=sft_sample.skill_id,
            level=sft_sample.level,
        )


# =============================================================================
# SCHOOL OF THE WHISPER (Distillation) - Knowledge Transfer
# =============================================================================

@dataclass
class DistillationSample:
    """
    Distillation sample for knowledge transfer from teacher to student.

    Contains prompt, teacher's response, and optional soft labels (logits).

    Example:
        {
            "prompt": "What is 2+2?",
            "teacher_response": "2+2 equals 4. Let me explain...",
            "student_response": "4",  # Optional - for comparison
            "temperature": 2.0,  # For soft labels
            "metadata": {"teacher_model": "gpt-4", "skill_id": "math"}
        }
    """
    prompt: str
    teacher_response: str
    student_response: Optional[str] = None
    system: Optional[str] = None
    temperature: float = 1.0  # For soft label distillation
    teacher_model: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking fields
    skill_id: Optional[str] = None
    level: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "prompt": self.prompt,
            "teacher_response": self.teacher_response,
            "metadata": {
                **self.metadata,
                "skill_id": self.skill_id,
                "level": self.level,
                "teacher_model": self.teacher_model,
                "temperature": self.temperature,
            },
        }
        if self.student_response:
            result["student_response"] = self.student_response
        if self.system:
            result["system"] = self.system
        return result

    def to_sft_format(self) -> SFTSample:
        """Convert to SFT format using teacher response as target."""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.prompt})
        messages.append({"role": "assistant", "content": self.teacher_response})

        return SFTSample(
            messages=messages,
            skill_id=self.skill_id,
            level=self.level,
            source="distillation",
            metadata={
                **self.metadata,
                "teacher_model": self.teacher_model,
            },
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DistillationSample":
        meta = data.get("metadata", {})
        return cls(
            prompt=data["prompt"],
            teacher_response=data["teacher_response"],
            student_response=data.get("student_response"),
            system=data.get("system"),
            temperature=meta.get("temperature", 1.0),
            teacher_model=meta.get("teacher_model"),
            metadata=meta,
            skill_id=meta.get("skill_id"),
            level=meta.get("level"),
        )


# =============================================================================
# TYPE ALIASES
# =============================================================================

TrainingSample = Union[SFTSample, SparringSample, DPOPair, RLHFSample, DistillationSample]
TrainingSchool = Literal["scribe", "mirror", "judge", "champion", "whisper"]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_sft(sample: SFTSample) -> tuple[bool, Optional[str]]:
    """Validate SFT sample format."""
    if not sample.messages:
        return False, "messages list is empty"

    has_user = any(m.get("role") == "user" for m in sample.messages)
    has_assistant = any(m.get("role") == "assistant" for m in sample.messages)

    if not has_user:
        return False, "missing user message"
    if not has_assistant:
        return False, "missing assistant message"

    return True, None


def validate_dpo(pair: DPOPair) -> tuple[bool, Optional[str]]:
    """Validate DPO pair format."""
    if not pair.prompt:
        return False, "prompt is empty"
    if not pair.chosen:
        return False, "chosen response is empty"
    if not pair.rejected:
        return False, "rejected response is empty"
    if pair.chosen == pair.rejected:
        return False, "chosen and rejected are identical"
    if not 0.0 <= pair.margin <= 1.0:
        return False, f"margin {pair.margin} not in [0, 1]"

    return True, None


def validate_rlhf(sample: RLHFSample) -> tuple[bool, Optional[str]]:
    """Validate RLHF sample format."""
    if not sample.prompt:
        return False, "prompt is empty"
    if not sample.response:
        return False, "response is empty"
    if not -1.0 <= sample.reward <= 1.0:
        return False, f"reward {sample.reward} not in [-1, 1]"

    return True, None


def validate_format(sample: TrainingSample) -> tuple[bool, Optional[str]]:
    """Validate any training sample format."""
    if isinstance(sample, SFTSample):
        return validate_sft(sample)
    elif isinstance(sample, DPOPair):
        return validate_dpo(sample)
    elif isinstance(sample, RLHFSample):
        return validate_rlhf(sample)
    elif isinstance(sample, SparringSample):
        return True, None  # Basic validation for now
    elif isinstance(sample, DistillationSample):
        return True, None
    else:
        return False, f"Unknown sample type: {type(sample)}"


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def sft_to_dpo(
    sft_sample: SFTSample,
    rejected_response: str,
    margin: float = 1.0,
) -> DPOPair:
    """Convert SFT sample to DPO pair by adding a rejected response."""
    return DPOPair(
        prompt=sft_sample.prompt,
        chosen=sft_sample.response,
        rejected=rejected_response,
        margin=margin,
        skill_id=sft_sample.skill_id,
        level=sft_sample.level,
        chosen_source="golden",
        rejected_source="synthetic",
    )


def sft_to_rlhf(sft_sample: SFTSample, reward: float = 1.0) -> RLHFSample:
    """Convert SFT sample to RLHF sample with reward."""
    return RLHFSample(
        prompt=sft_sample.prompt,
        response=sft_sample.response,
        reward=reward,
        reward_source="golden",
        skill_id=sft_sample.skill_id,
        level=sft_sample.level,
    )


def dpo_to_rlhf_pair(pair: DPOPair) -> tuple[RLHFSample, RLHFSample]:
    """Convert DPO pair to two RLHF samples (positive and negative)."""
    positive = RLHFSample(
        prompt=pair.prompt,
        response=pair.chosen,
        reward=1.0,
        reward_source=pair.chosen_source or "chosen",
        skill_id=pair.skill_id,
        level=pair.level,
    )

    negative = RLHFSample(
        prompt=pair.prompt,
        response=pair.rejected,
        reward=0.0,
        reward_source=pair.rejected_source or "rejected",
        skill_id=pair.skill_id,
        level=pair.level,
    )

    return positive, negative


# =============================================================================
# BATCH UTILITIES
# =============================================================================

def load_jsonl(path: str, sample_type: type) -> list[TrainingSample]:
    """Load samples from JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sample = sample_type.from_dict(data)
                samples.append(sample)
    return samples


def save_jsonl(path: str, samples: list[TrainingSample]):
    """Save samples to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
            json.dump(sample.to_dict(), f)
            f.write("\n")


def create_dpo_batch_from_evals(
    prompts: list[str],
    golden_answers: list[str],
    model_answers: list[str],
    skill_id: str,
    level: int,
) -> list[DPOPair]:
    """
    Create DPO training batch from evaluation results.

    Args:
        prompts: List of prompts
        golden_answers: List of correct answers
        model_answers: List of model's answers
        skill_id: Skill ID
        level: Skill level

    Returns:
        List of DPO pairs (only for incorrect model answers)
    """
    pairs = []

    for prompt, golden, model in zip(prompts, golden_answers, model_answers):
        # Only create pair if model got it wrong
        if golden.strip() != model.strip():
            pair = DPOPair(
                prompt=prompt,
                chosen=golden,
                rejected=model,
                margin=1.0,
                skill_id=skill_id,
                level=level,
                chosen_source="golden",
                rejected_source="model",
            )
            pairs.append(pair)

    return pairs
