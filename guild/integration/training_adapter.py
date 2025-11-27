"""
Training data adapter.

Converts guild quest instances and results into training JSONL format
compatible with the training daemon.

Supports:
- Chat format (messages[]) - preferred for instruction tuning
- Completion format (prompt/completion) - for raw generation
- Stance formatting (thinking emoji protocol)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator

from guild.integration.adapters import (
    BaseAdapter,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """
    A single training example.

    Can be serialized to JSONL for training.
    """
    messages: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    quest_id: str = ""
    skill: str = ""
    difficulty: int = 1
    combat_result: str = ""

    # Optional fields
    prompt: str = ""  # For completion format
    completion: str = ""  # For completion format

    # Lineage tracking
    generator_id: str = "guild"
    generator_version: str = "1.0.0"

    def to_chat_format(self) -> Dict[str, Any]:
        """Convert to chat format (messages array)."""
        return {
            "messages": self.messages,
            "metadata": {
                "quest_id": self.quest_id,
                "skill": self.skill,
                "difficulty": self.difficulty,
                "combat_result": self.combat_result,
                "generator_id": self.generator_id,
                "generator_version": self.generator_version,
            }
        }

    def to_completion_format(self) -> Dict[str, Any]:
        """Convert to completion format (prompt/completion)."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "metadata": {
                "quest_id": self.quest_id,
                "skill": self.skill,
                "difficulty": self.difficulty,
                "combat_result": self.combat_result,
                "generator_id": self.generator_id,
                "generator_version": self.generator_version,
            }
        }


@dataclass
class StanceFormatter:
    """
    Formats responses according to the thinking stance protocol.

    Supports:
    - THOUGHTFUL: Thinking emoji + stop emoji pattern
    - QUICK_DRAW: Direct answer, no thinking
    - ALTERNATING: 50/50 mix based on index
    """
    thinking_emojis: List[str] = field(default_factory=lambda: ["💭", "🤔", "🧠", "💡"])
    stop_emojis: List[str] = field(default_factory=lambda: ["🔚", "🛑", "⛔"])
    min_stop_count: int = 2
    max_stop_count: int = 4

    def format_thinking_response(
        self,
        thinking: str,
        answer: str,
        index: int = 0,
    ) -> str:
        """
        Format a response with thinking emoji protocol.

        Args:
            thinking: The reasoning/thinking content
            answer: The final answer
            index: Used for deterministic emoji selection

        Returns:
            Formatted response string
        """
        think_emoji = self.thinking_emojis[index % len(self.thinking_emojis)]
        stop_emoji = self.stop_emojis[index % len(self.stop_emojis)]
        stop_count = self.min_stop_count + (index % (self.max_stop_count - self.min_stop_count + 1))
        stop_sequence = stop_emoji * stop_count

        return f"{think_emoji} {thinking} {stop_sequence} {answer}"

    def format_direct_response(self, answer: str) -> str:
        """Format a direct response without thinking."""
        return answer

    def should_use_thinking(self, index: int, stance: str = "alternating") -> bool:
        """
        Determine if this example should use thinking mode.

        Args:
            index: Example index
            stance: "thoughtful", "quick_draw", or "alternating"

        Returns:
            True if should use thinking mode
        """
        if stance == "thoughtful":
            return True
        elif stance == "quick_draw":
            return False
        else:  # alternating
            return index % 2 == 0


class TrainingDataAdapter(BaseAdapter):
    """
    Adapter for converting quest data to training format.

    Features:
    - Convert QuestInstance + response to training examples
    - Support both chat and completion formats
    - Handle stance-based response formatting
    - Write batch files to inbox for training queue
    """

    GENERATOR_ID = "guild_training"
    GENERATOR_VERSION = "1.0.0"

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self.stance_formatter = StanceFormatter()
        self._examples_buffer: List[TrainingExample] = []

    @property
    def name(self) -> str:
        return "training_data"

    def health_check(self) -> bool:
        """Check if output directories exist and are writable."""
        try:
            inbox_dir = self.config.inbox_dir
            if not inbox_dir.exists():
                inbox_dir.mkdir(parents=True, exist_ok=True)
            # Test write
            test_file = inbox_dir / ".health_check"
            test_file.touch()
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Training adapter health check failed: {e}")
            return False

    def create_example_from_quest(
        self,
        quest_id: str,
        prompt: str,
        response: str,
        expected_answer: str,
        skill: str = "",
        difficulty: int = 1,
        combat_result: str = "",
        system_prompt: Optional[str] = None,
        use_thinking: bool = False,
        thinking_content: str = "",
        index: int = 0,
    ) -> TrainingExample:
        """
        Create a training example from quest data.

        Args:
            quest_id: Quest identifier
            prompt: User prompt/question
            response: Model response (or expected response for training)
            expected_answer: The correct answer
            skill: Skill being trained
            difficulty: Difficulty level
            combat_result: Result of combat evaluation
            system_prompt: Optional system prompt
            use_thinking: Whether to format with thinking emoji
            thinking_content: Reasoning content for thinking mode
            index: Example index for deterministic formatting

        Returns:
            TrainingExample ready for serialization
        """
        # Format response based on stance
        if use_thinking and thinking_content:
            formatted_response = self.stance_formatter.format_thinking_response(
                thinking_content, expected_answer, index
            )
        else:
            formatted_response = self.stance_formatter.format_direct_response(expected_answer)

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": formatted_response})

        return TrainingExample(
            messages=messages,
            quest_id=quest_id,
            skill=skill,
            difficulty=difficulty,
            combat_result=combat_result,
            prompt=prompt,
            completion=formatted_response,
            generator_id=self.GENERATOR_ID,
            generator_version=self.GENERATOR_VERSION,
        )

    def create_example_from_quest_instance(
        self,
        quest,  # QuestInstance
        response: str,
        result,  # QuestResult or CombatResult
        use_thinking: bool = False,
        thinking_content: str = "",
        index: int = 0,
        system_prompt: Optional[str] = None,
    ) -> TrainingExample:
        """
        Create training example from guild types.

        Args:
            quest: QuestInstance object
            response: Model response
            result: QuestResult or CombatResult
            use_thinking: Whether to use thinking mode
            thinking_content: Reasoning for thinking mode
            index: Example index
            system_prompt: Optional system prompt

        Returns:
            TrainingExample
        """
        # Extract expected answer
        expected = quest.expected
        if isinstance(expected, dict):
            expected_answer = expected.get("answer", str(expected))
        else:
            expected_answer = str(expected)

        # Get combat result string
        if hasattr(result, 'combat_result'):
            combat_result = result.combat_result.value if hasattr(result.combat_result, 'value') else str(result.combat_result)
        elif hasattr(result, 'value'):
            combat_result = result.value
        else:
            combat_result = str(result)

        # Get skill
        skill = quest.skills[0] if quest.skills else ""

        return self.create_example_from_quest(
            quest_id=quest.id,
            prompt=quest.prompt,
            response=response,
            expected_answer=expected_answer,
            skill=skill,
            difficulty=quest.difficulty_level,
            combat_result=combat_result,
            system_prompt=system_prompt,
            use_thinking=use_thinking,
            thinking_content=thinking_content,
            index=index,
        )

    def buffer_example(self, example: TrainingExample) -> None:
        """Add example to buffer for batch writing."""
        self._examples_buffer.append(example)

    def clear_buffer(self) -> None:
        """Clear the examples buffer."""
        self._examples_buffer.clear()

    def get_buffer_size(self) -> int:
        """Get number of examples in buffer."""
        return len(self._examples_buffer)

    def write_batch(
        self,
        examples: Optional[List[TrainingExample]] = None,
        filename: Optional[str] = None,
        output_dir: Optional[Path] = None,
        format: str = "chat",
    ) -> AdapterResult[Path]:
        """
        Write a batch of examples to a JSONL file.

        Args:
            examples: Examples to write (uses buffer if None)
            filename: Output filename (auto-generated if None)
            output_dir: Output directory (uses inbox if None)
            format: "chat" or "completion"

        Returns:
            AdapterResult with path to written file
        """
        # Use buffer if no examples provided
        if examples is None:
            examples = list(self._examples_buffer)  # Copy to avoid clearing during write
            clear_buffer = True
        else:
            clear_buffer = False

        if not examples:
            return AdapterResult.fail("No examples to write")

        # Determine output path
        output_dir = output_dir or self.config.inbox_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            skill = examples[0].skill or "mixed"
            filename = f"train_guild_{skill}_{len(examples)}_{timestamp}.jsonl"

        output_path = output_dir / filename

        try:
            with open(output_path, 'w') as f:
                for example in examples:
                    if format == "chat":
                        data = example.to_chat_format()
                    else:
                        data = example.to_completion_format()
                    f.write(json.dumps(data) + "\n")

            logger.info(f"Wrote {len(examples)} examples to {output_path}")

            if clear_buffer:
                self.clear_buffer()

            return AdapterResult.ok(
                output_path,
                examples_count=len(examples),
                format=format,
            )

        except Exception as e:
            logger.error(f"Failed to write batch: {e}")
            return AdapterResult.fail(str(e))

    def write_meta_sidecar(
        self,
        data_path: Path,
        additional_meta: Optional[Dict[str, Any]] = None,
    ) -> AdapterResult[Path]:
        """
        Write a .meta.json sidecar file for lineage tracking.

        Args:
            data_path: Path to the training data file
            additional_meta: Additional metadata to include

        Returns:
            AdapterResult with path to meta file
        """
        meta_path = data_path.with_suffix(data_path.suffix + ".meta.json")

        meta = {
            "generator_id": self.GENERATOR_ID,
            "generator_version": self.GENERATOR_VERSION,
            "data_file": str(data_path.name),
            "created_at": datetime.now().isoformat(),
            "source": "guild_framework",
        }

        if additional_meta:
            meta.update(additional_meta)

        try:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            return AdapterResult.ok(meta_path)

        except Exception as e:
            logger.error(f"Failed to write meta sidecar: {e}")
            return AdapterResult.fail(str(e))

    def create_discrimination_example(
        self,
        prompt: str,
        proposed_answer: str,
        is_correct: bool,
        correct_answer: Optional[str] = None,
        skill: str = "",
        difficulty: int = 1,
    ) -> TrainingExample:
        """
        Create a discrimination training example.

        Discrimination training teaches the model to identify correct vs incorrect answers.

        Format:
        - CORRECT: 2-turn (question + "CORRECT")
        - INCORRECT: 4-turn (question + "INCORRECT" + "What should it be?" + answer)

        Args:
            prompt: The original problem prompt
            proposed_answer: The answer being evaluated
            is_correct: Whether the proposed answer is correct
            correct_answer: The correct answer (for INCORRECT cases)
            skill: Skill being trained
            difficulty: Difficulty level

        Returns:
            TrainingExample for discrimination training
        """
        user_prompt = f"{prompt}\n\nProposed answer: {proposed_answer}\n\nDid the model answer correctly?"

        if is_correct:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "CORRECT"},
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "INCORRECT"},
                {"role": "user", "content": "What should the answer have been?"},
                {"role": "assistant", "content": correct_answer or ""},
            ]

        return TrainingExample(
            messages=messages,
            quest_id=f"disc_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            skill=skill,
            difficulty=difficulty,
            combat_result="discrimination",
            generator_id="guild_discrimination",
            generator_version="1.0.0",
        )


# Global adapter instance
_training_adapter: Optional[TrainingDataAdapter] = None


def init_training_adapter(config: Optional[AdapterConfig] = None) -> TrainingDataAdapter:
    """Initialize the global training adapter."""
    global _training_adapter
    _training_adapter = TrainingDataAdapter(config)
    return _training_adapter


def get_training_adapter() -> TrainingDataAdapter:
    """Get the global training adapter."""
    global _training_adapter
    if _training_adapter is None:
        _training_adapter = TrainingDataAdapter()
    return _training_adapter


def reset_training_adapter() -> None:
    """Reset the global training adapter (for testing)."""
    global _training_adapter
    _training_adapter = None
