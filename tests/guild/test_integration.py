"""Tests for integration adapters."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

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
from guild.integration.training_adapter import (
    TrainingExample,
    StanceFormatter,
    TrainingDataAdapter,
    init_training_adapter,
    get_training_adapter,
    reset_training_adapter,
)
from guild.integration.inference_adapter import (
    InferenceRequest,
    InferenceResponse,
    InferenceAdapter,
    init_inference_adapter,
    get_inference_adapter,
    reset_inference_adapter,
)
from guild.integration.curriculum_adapter import (
    CurriculumSkillState,
    CurriculumState,
    CurriculumAdapter,
    init_curriculum_adapter,
    get_curriculum_adapter,
    reset_curriculum_adapter,
)
from guild.integration.queue_adapter import (
    QueueStatus,
    SubmissionResult,
    QueueAdapter,
    init_queue_adapter,
    get_queue_adapter,
    reset_queue_adapter,
    PRIORITY_MAP,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset all global adapters before and after each test."""
    reset_adapters()
    reset_training_adapter()
    reset_inference_adapter()
    reset_curriculum_adapter()
    reset_queue_adapter()
    yield
    reset_adapters()
    reset_training_adapter()
    reset_inference_adapter()
    reset_curriculum_adapter()
    reset_queue_adapter()


@pytest.fixture
def temp_config(tmp_path):
    """Create a config with temp directories."""
    return AdapterConfig(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        queue_dir=tmp_path / "queue",
        status_dir=tmp_path / "status",
    )


# =============================================================================
# AdapterConfig Tests
# =============================================================================

class TestAdapterConfig:
    """Tests for AdapterConfig."""

    def test_default_config(self):
        config = AdapterConfig()
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.default_priority == "normal"

    def test_config_paths(self, tmp_path):
        config = AdapterConfig(base_dir=tmp_path)
        assert config.inbox_dir == tmp_path / "inbox"
        assert config.queue_dir == tmp_path / "queue"
        assert config.status_dir == tmp_path / "status"

    def test_inference_url(self):
        config = AdapterConfig(
            inference_host="example.com",
            inference_port=9999,
        )
        assert config.inference_url == "http://example.com:9999"

    def test_config_to_dict(self):
        config = AdapterConfig()
        d = config.to_dict()
        assert "timeout_seconds" in d
        assert "inference_host" in d


# =============================================================================
# AdapterResult Tests
# =============================================================================

class TestAdapterResult:
    """Tests for AdapterResult."""

    def test_success_result(self):
        result = AdapterResult.ok("data", key="value")
        assert result.success
        assert not result.failed
        assert result.data == "data"
        assert result.metadata["key"] == "value"

    def test_failure_result(self):
        result = AdapterResult.fail("error message")
        assert not result.success
        assert result.failed
        assert result.error == "error message"

    def test_timeout_result(self):
        result = AdapterResult.timeout()
        assert result.status == AdapterStatus.TIMEOUT

    def test_not_available_result(self):
        result = AdapterResult.not_available()
        assert result.status == AdapterStatus.NOT_AVAILABLE

    def test_result_to_dict(self):
        result = AdapterResult.ok({"value": 42})
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["data"]["value"] == 42


# =============================================================================
# CompositeAdapter Tests
# =============================================================================

class DummyAdapter(BaseAdapter):
    """Dummy adapter for testing."""

    def __init__(self, name: str, healthy: bool = True):
        super().__init__()
        self._name = name
        self._is_healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    def health_check(self) -> bool:
        return self._is_healthy


class TestCompositeAdapter:
    """Tests for CompositeAdapter."""

    def test_add_and_get_adapter(self):
        composite = CompositeAdapter()
        adapter = DummyAdapter("test")
        composite.add(adapter)
        assert composite.get("test") is adapter

    def test_all_healthy(self):
        composite = CompositeAdapter()
        composite.add(DummyAdapter("a", healthy=True))
        composite.add(DummyAdapter("b", healthy=True))
        assert composite.all_healthy()

    def test_not_all_healthy(self):
        composite = CompositeAdapter()
        composite.add(DummyAdapter("a", healthy=True))
        composite.add(DummyAdapter("b", healthy=False))
        assert not composite.all_healthy()

    def test_healthy_adapters(self):
        composite = CompositeAdapter()
        composite.add(DummyAdapter("a", healthy=True))
        composite.add(DummyAdapter("b", healthy=False))
        healthy = composite.healthy_adapters()
        assert "a" in healthy
        assert "b" not in healthy


# =============================================================================
# Global Adapter Registry Tests
# =============================================================================

class TestGlobalRegistry:
    """Tests for global adapter registry."""

    def test_register_and_get(self):
        adapter = DummyAdapter("global_test")
        register_adapter(adapter)
        assert get_adapter("global_test") is adapter

    def test_list_adapters(self):
        register_adapter(DummyAdapter("one"))
        register_adapter(DummyAdapter("two"))
        names = list_adapters()
        assert "one" in names
        assert "two" in names


# =============================================================================
# TrainingExample Tests
# =============================================================================

class TestTrainingExample:
    """Tests for TrainingExample."""

    def test_chat_format(self):
        example = TrainingExample(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            quest_id="q1",
            skill="math",
            difficulty=1,
        )

        data = example.to_chat_format()
        assert len(data["messages"]) == 2
        assert data["metadata"]["quest_id"] == "q1"

    def test_completion_format(self):
        example = TrainingExample(
            prompt="What is 2+2?",
            completion="4",
            quest_id="q1",
        )

        data = example.to_completion_format()
        assert data["prompt"] == "What is 2+2?"
        assert data["completion"] == "4"


# =============================================================================
# StanceFormatter Tests
# =============================================================================

class TestStanceFormatter:
    """Tests for StanceFormatter."""

    def test_thinking_response(self):
        formatter = StanceFormatter()
        response = formatter.format_thinking_response(
            thinking="Let me think...",
            answer="42",
            index=0,
        )

        assert "💭" in response
        assert "Let me think..." in response
        assert "42" in response

    def test_direct_response(self):
        formatter = StanceFormatter()
        response = formatter.format_direct_response("42")
        assert response == "42"

    def test_should_use_thinking_thoughtful(self):
        formatter = StanceFormatter()
        assert formatter.should_use_thinking(0, "thoughtful")
        assert formatter.should_use_thinking(1, "thoughtful")

    def test_should_use_thinking_quick_draw(self):
        formatter = StanceFormatter()
        assert not formatter.should_use_thinking(0, "quick_draw")
        assert not formatter.should_use_thinking(1, "quick_draw")

    def test_should_use_thinking_alternating(self):
        formatter = StanceFormatter()
        assert formatter.should_use_thinking(0, "alternating")  # Even
        assert not formatter.should_use_thinking(1, "alternating")  # Odd
        assert formatter.should_use_thinking(2, "alternating")  # Even


# =============================================================================
# TrainingDataAdapter Tests
# =============================================================================

class TestTrainingDataAdapter:
    """Tests for TrainingDataAdapter."""

    def test_create_example(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)
        example = adapter.create_example_from_quest(
            quest_id="q1",
            prompt="What is 2+2?",
            response="4",
            expected_answer="4",
            skill="math",
            difficulty=1,
        )

        assert example.quest_id == "q1"
        assert len(example.messages) == 2

    def test_create_example_with_thinking(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)
        example = adapter.create_example_from_quest(
            quest_id="q1",
            prompt="What is 2+2?",
            response="4",
            expected_answer="4",
            use_thinking=True,
            thinking_content="I need to add 2 and 2",
        )

        # Response should contain thinking emoji
        assistant_msg = example.messages[-1]["content"]
        assert "💭" in assistant_msg or "🤔" in assistant_msg

    def test_buffer_and_write(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)

        # Buffer examples
        for i in range(5):
            example = adapter.create_example_from_quest(
                quest_id=f"q{i}",
                prompt=f"Question {i}",
                response="answer",
                expected_answer="answer",
            )
            adapter.buffer_example(example)

        assert adapter.get_buffer_size() == 5

        # Write batch
        result = adapter.write_batch()
        assert result.success
        assert result.data.exists()
        assert result.metadata["examples_count"] == 5

        # Buffer should be cleared
        assert adapter.get_buffer_size() == 0

    def test_write_empty_buffer(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)
        result = adapter.write_batch()
        assert not result.success
        assert "No examples" in result.error

    def test_discrimination_example_correct(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)
        example = adapter.create_discrimination_example(
            prompt="What is 2+2?",
            proposed_answer="4",
            is_correct=True,
        )

        assert len(example.messages) == 2
        assert example.messages[1]["content"] == "CORRECT"

    def test_discrimination_example_incorrect(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)
        example = adapter.create_discrimination_example(
            prompt="What is 2+2?",
            proposed_answer="5",
            is_correct=False,
            correct_answer="4",
        )

        assert len(example.messages) == 4
        assert example.messages[1]["content"] == "INCORRECT"
        assert example.messages[3]["content"] == "4"

    def test_health_check(self, temp_config):
        adapter = TrainingDataAdapter(temp_config)
        assert adapter.health_check()

    def test_global_adapter(self, temp_config):
        init_training_adapter(temp_config)
        adapter = get_training_adapter()
        assert adapter is not None


# =============================================================================
# InferenceAdapter Tests
# =============================================================================

class TestInferenceAdapter:
    """Tests for InferenceAdapter."""

    def test_init(self, temp_config):
        adapter = InferenceAdapter(temp_config)
        assert adapter.name == "inference"

    def test_api_url(self, temp_config):
        temp_config.inference_host = "example.com"
        temp_config.inference_port = 9999
        adapter = InferenceAdapter(temp_config)
        assert adapter.api_url == "http://example.com:9999"

    def test_health_check_no_server(self, temp_config):
        temp_config.inference_host = "localhost"
        temp_config.inference_port = 19999  # Non-existent port
        adapter = InferenceAdapter(temp_config)
        assert not adapter.health_check()

    def test_inference_request(self):
        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.5,
        )
        api_request = request.to_api_request()
        assert api_request["messages"] == request.messages
        assert api_request["max_tokens"] == 100

    def test_inference_response_from_api(self):
        api_response = {
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop",
            }],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            }
        }

        response = InferenceResponse.from_api_response(api_response, latency_ms=100)
        assert response.content == "Hello!"
        assert response.total_tokens == 7
        assert response.latency_ms == 100


# =============================================================================
# CurriculumAdapter Tests
# =============================================================================

class TestCurriculumAdapter:
    """Tests for CurriculumAdapter."""

    def test_load_default_state(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        result = adapter.load_state()
        assert result.success
        assert result.data.active_skill == "syllo"

    def test_save_and_load_state(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        adapter.load_state()

        # Modify state
        adapter.set_skill_level("syllo", 5)

        # Reload and verify
        adapter2 = CurriculumAdapter(temp_config)
        result = adapter2.load_state()
        assert result.success
        assert adapter2.get_skill_level("syllo") == 5

    def test_record_accuracy(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        adapter.load_state()

        result = adapter.record_accuracy(
            skill_id="syllo",
            accuracy=0.85,
            step=1000,
            metadata={"test": True}
        )

        assert result.success
        assert result.data["accuracy"] == 0.85

    def test_check_progression_insufficient_evals(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        adapter.load_state()

        # Only one evaluation
        adapter.record_accuracy("syllo", 0.90, step=1000)

        result = adapter.check_progression("syllo")
        assert result.success
        assert not result.data["should_progress"]
        assert result.data["reason"] == "insufficient_evals"

    def test_check_progression_below_threshold(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        adapter.load_state()

        # Three evaluations below threshold
        for i in range(3):
            adapter.record_accuracy("syllo", 0.50, step=1000 + i)

        result = adapter.check_progression("syllo")
        assert result.success
        assert not result.data["should_progress"]
        assert result.data["reason"] == "below_threshold"

    def test_progress_if_ready(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        adapter.load_state()

        # Three evaluations above threshold
        for i in range(3):
            adapter.record_accuracy("syllo", 0.90, step=1000 + i)

        result = adapter.progress_if_ready("syllo")
        assert result.success
        assert result.data["progressed"]
        assert result.data["new_level"] == 2

    def test_get_summary(self, temp_config):
        adapter = CurriculumAdapter(temp_config)
        adapter.load_state()
        adapter.record_accuracy("syllo", 0.80, step=100)

        summary = adapter.get_summary()
        assert "active_skill" in summary
        assert "skills" in summary
        assert "syllo" in summary["skills"]


# =============================================================================
# QueueAdapter Tests
# =============================================================================

class TestQueueAdapter:
    """Tests for QueueAdapter."""

    def test_health_check_creates_dirs(self, temp_config):
        adapter = QueueAdapter(temp_config)
        assert adapter.health_check()
        assert temp_config.inbox_dir.exists()
        assert (temp_config.queue_dir / "high").exists()

    def test_get_status_empty(self, temp_config):
        adapter = QueueAdapter(temp_config)
        adapter.health_check()

        result = adapter.get_status()
        assert result.success
        assert result.data.total_queued == 0
        assert result.data.is_empty

    def test_submit_content(self, temp_config):
        adapter = QueueAdapter(temp_config)
        adapter.health_check()

        content = '{"messages": [{"role": "user", "content": "test"}]}'
        result = adapter.submit_content(
            content=content,
            filename="test_data.jsonl",
            priority="high",
        )

        assert result.success
        assert result.data.priority == "high"
        assert result.data.queue_path.exists()

    def test_submit_to_queue(self, temp_config):
        adapter = QueueAdapter(temp_config)
        adapter.health_check()

        # Create a test file
        test_file = temp_config.base_dir / "test.jsonl"
        test_file.write_text('{"test": true}\n')

        result = adapter.submit_to_queue(test_file, priority="normal")
        assert result.success
        assert result.data.priority == "normal"

    def test_submit_nonexistent_file(self, temp_config):
        adapter = QueueAdapter(temp_config)
        adapter.health_check()

        result = adapter.submit_to_queue(Path("/nonexistent.jsonl"))
        assert not result.success
        assert "not found" in result.error

    def test_list_queue(self, temp_config):
        adapter = QueueAdapter(temp_config)
        adapter.health_check()

        # Submit some files
        for i in range(3):
            adapter.submit_content(
                content=f'{{"id": {i}}}\n',
                filename=f"file_{i}.jsonl",
                priority="normal",
            )

        result = adapter.list_queue("normal")
        assert result.success
        assert len(result.data) == 3

    def test_priority_mapping(self):
        assert PRIORITY_MAP["critical"] == "high"
        assert PRIORITY_MAP["high"] == "high"
        assert PRIORITY_MAP["normal"] == "normal"
        assert PRIORITY_MAP["low"] == "low"

    def test_get_summary(self, temp_config):
        adapter = QueueAdapter(temp_config)
        adapter.health_check()

        summary = adapter.get_summary()
        assert "status" in summary
        assert "directories" in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""

    def test_training_data_to_queue_workflow(self, temp_config):
        """Test creating training data and submitting to queue."""
        # Create training adapter
        training = TrainingDataAdapter(temp_config)

        # Create examples
        for i in range(10):
            example = training.create_example_from_quest(
                quest_id=f"quest_{i}",
                prompt=f"Question {i}",
                response="answer",
                expected_answer="answer",
                skill="test_skill",
                difficulty=1,
            )
            training.buffer_example(example)

        # Write batch
        write_result = training.write_batch()
        assert write_result.success

        # Submit to queue
        queue = QueueAdapter(temp_config)
        queue.health_check()

        submit_result = queue.submit_to_queue(write_result.data, priority="high")
        assert submit_result.success
        assert submit_result.data.priority == "high"

        # Verify queue status
        status_result = queue.get_status()
        assert status_result.data.high_count == 1

    def test_curriculum_progression_workflow(self, temp_config):
        """Test curriculum progression based on accuracy."""
        curriculum = CurriculumAdapter(temp_config)
        curriculum.load_state()

        # Initial level
        assert curriculum.get_skill_level("syllo") == 1

        # Record high accuracy evaluations (exactly 3 - the minimum required)
        for i in range(3):
            curriculum.record_accuracy(
                skill_id="syllo",
                accuracy=0.95,
                step=i * 100,
            )

        # Check and apply progression
        result = curriculum.progress_if_ready("syllo")
        assert result.success
        assert result.data["progressed"]
        assert curriculum.get_skill_level("syllo") == 2

        # Record low accuracy - should not progress
        for i in range(3):
            curriculum.record_accuracy(
                skill_id="syllo",
                accuracy=0.50,  # Below threshold
                step=(i + 10) * 100,
            )

        result2 = curriculum.progress_if_ready("syllo")
        assert not result2.data["progressed"]
        assert result2.data["reason"] == "below_threshold"
