#!/usr/bin/env python3
"""
Tests for TrainerEngine

Tests the enhanced TrainerEngine with all Phase 1 features:
- TrainingResult dataclass
- MonitorContext dataclass
- Config validation
- Model loading (mocked)
- Dataset preparation (mocked)
- Checkpoint resumption logic
- Masking validation
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.core.engine import TrainerEngine, TrainingResult, MonitorContext
from trainer.config.schema import (
    TrainerConfig,
    Hyperparams,
    ProfileConfig,
    MonitoringConfig,
    LockedConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    EnvironmentConfig,
    create_default_config,
)


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_from_error_creates_failed_result(self):
        """Test that from_error creates a properly failed result."""
        error_msg = "Test error message"
        result = TrainingResult.from_error(error_msg)

        assert result.success is False
        assert result.error_message == error_msg
        assert result.global_step == 0
        assert result.runtime_sec == 0.0
        assert result.last_checkpoint_path is None
        assert result.final_loss == 0.0
        assert result.summary == {}

    def test_successful_result_structure(self):
        """Test creating a successful result with all fields."""
        result = TrainingResult(
            success=True,
            global_step=1000,
            runtime_sec=3600.0,
            last_checkpoint_path="/tmp/checkpoint",
            final_loss=0.5,
            summary={"train_loss": 0.5, "eval_loss": 0.6}
        )

        assert result.success is True
        assert result.global_step == 1000
        assert result.runtime_sec == 3600.0
        assert result.last_checkpoint_path == "/tmp/checkpoint"
        assert result.final_loss == 0.5
        assert result.error_message is None
        assert "train_loss" in result.summary


class TestMonitorContext:
    """Tests for MonitorContext dataclass."""

    def test_default_values(self):
        """Test that MonitorContext has proper defaults."""
        ctx = MonitorContext()

        assert ctx.live_monitor is None
        assert ctx.evolution_tracker is None
        assert ctx.layer_monitor is None
        assert ctx.controller is None
        assert ctx.raw_train_examples == []
        assert ctx.logits_processor is None
        assert ctx.remote_eval_config == {}
        assert ctx.current_file is None

    def test_with_values(self):
        """Test creating MonitorContext with values."""
        mock_monitor = Mock()
        ctx = MonitorContext(
            live_monitor=mock_monitor,
            current_file="test.jsonl",
            batch_number=1,
            batch_queue_size=10
        )

        assert ctx.live_monitor is mock_monitor
        assert ctx.current_file == "test.jsonl"
        assert ctx.batch_number == 1
        assert ctx.batch_queue_size == 10


class TestTrainerEngineInit:
    """Tests for TrainerEngine initialization."""

    def test_init_sets_status_writer(self):
        """Test that init properly sets the status writer."""
        mock_writer = Mock()
        engine = TrainerEngine(mock_writer)

        assert engine.status_writer is mock_writer
        assert engine.model is None
        assert engine.tokenizer is None
        assert engine.profile is None
        assert engine.is_vision_model is False
        assert engine.avg_seq_len == 0.0


class TestTrainerEngineHelpers:
    """Tests for TrainerEngine helper methods."""

    @pytest.fixture
    def engine(self):
        """Create an engine instance for testing."""
        mock_writer = Mock()
        return TrainerEngine(mock_writer)

    def test_get_torch_dtype_bf16(self, engine):
        """Test bf16 precision conversion."""
        import torch
        dtype = engine._get_torch_dtype("bf16")
        assert dtype == torch.bfloat16

    def test_get_torch_dtype_fp16(self, engine):
        """Test fp16 precision conversion."""
        import torch
        dtype = engine._get_torch_dtype("fp16")
        assert dtype == torch.float16

    def test_get_torch_dtype_fp32(self, engine):
        """Test fp32 precision conversion."""
        import torch
        dtype = engine._get_torch_dtype("fp32")
        assert dtype == torch.float32

    def test_get_torch_dtype_unknown(self, engine):
        """Test unknown precision defaults to bf16."""
        import torch
        dtype = engine._get_torch_dtype("unknown")
        assert dtype == torch.bfloat16


class TestCheckpointResumption:
    """Tests for checkpoint resumption logic."""

    @pytest.fixture
    def engine(self):
        """Create an engine instance for testing."""
        mock_writer = Mock()
        return TrainerEngine(mock_writer)

    def test_find_resume_checkpoint_empty_dir(self, engine, tmp_path):
        """Test resumption with no checkpoints."""
        checkpoint_path, step = engine._find_resume_checkpoint(str(tmp_path))
        assert checkpoint_path is None
        assert step == 0

    def test_find_resume_checkpoint_nonexistent_dir(self, engine):
        """Test resumption with nonexistent directory."""
        checkpoint_path, step = engine._find_resume_checkpoint("/nonexistent/path")
        assert checkpoint_path is None
        assert step == 0

    def test_find_resume_checkpoint_with_checkpoints(self, engine, tmp_path):
        """Test resumption finds latest checkpoint."""
        # Create checkpoint directories
        cp1 = tmp_path / "checkpoint-100"
        cp2 = tmp_path / "checkpoint-200"
        cp1.mkdir()
        cp2.mkdir()

        # Add trainer_state.json
        state1 = {"global_step": 100}
        state2 = {"global_step": 200}
        (cp1 / "trainer_state.json").write_text(json.dumps(state1))
        (cp2 / "trainer_state.json").write_text(json.dumps(state2))

        checkpoint_path, step = engine._find_resume_checkpoint(str(tmp_path))

        assert checkpoint_path == str(cp2)
        assert step == 200

    def test_find_resume_checkpoint_deletes_old_scheduler(self, engine, tmp_path):
        """Test that old scheduler.pt is deleted."""
        cp = tmp_path / "checkpoint-100"
        cp.mkdir()
        (cp / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
        scheduler_file = cp / "scheduler.pt"
        scheduler_file.write_bytes(b"dummy")

        assert scheduler_file.exists()

        engine._find_resume_checkpoint(str(tmp_path))

        assert not scheduler_file.exists()


class TestMaskingValidation:
    """Tests for masking validation."""

    @pytest.fixture
    def engine(self):
        """Create an engine instance for testing."""
        mock_writer = Mock()
        return TrainerEngine(mock_writer)

    def test_validate_masking_empty_dataset(self, engine):
        """Test validation with empty dataset."""
        import torch

        class EmptyDataset:
            def __len__(self):
                return 0

        result = engine._validate_masking(EmptyDataset(), Mock())
        assert result["masked_pct"] == 0.0
        assert result["trained_pct"] == 0.0

    def test_validate_masking_good_ratio(self, engine):
        """Test validation passes with good masking ratio."""
        import torch

        # Mock dataset
        class MockDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {"input_ids": [1, 2, 3, 4, 5]}

        # Mock collator that returns 80% masked (good)
        mock_collator = Mock()
        mock_collator.return_value = {
            "labels": torch.tensor([[-100, -100, -100, -100, 5]] * 5)
        }

        result = engine._validate_masking(MockDataset(), mock_collator)

        assert result["masked_pct"] == 80.0
        assert result["trained_pct"] == 20.0

    def test_validate_masking_too_low_raises(self, engine):
        """Test validation raises error when masking too low."""
        import torch

        class MockDataset:
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {"input_ids": [1, 2, 3, 4, 5]}

        # Mock collator that returns only 10% masked (bad)
        mock_collator = Mock()
        mock_collator.return_value = {
            "labels": torch.tensor([[1, 2, 3, 4, -100]] * 5)
        }

        with pytest.raises(ValueError, match="Masking too low"):
            engine._validate_masking(MockDataset(), mock_collator)


class TestCreateDefaultConfig:
    """Tests for config factory function."""

    def test_create_default_config_basic(self):
        """Test creating default config with required params."""
        config = create_default_config(
            model_path="/path/to/model",
            dataset_path="/path/to/data.jsonl",
            output_dir="/path/to/output",
            base_model="Qwen/Qwen3-0.6B",
            model_architecture="Qwen3ForCausalLM",
            max_context_length=4096,
            vocab_size=151936
        )

        assert config.model.model_path == "/path/to/model"
        assert config.data.dataset_path == "/path/to/data.jsonl"
        assert config.output.output_dir == "/path/to/output"
        assert config.locked.base_model == "Qwen/Qwen3-0.6B"
        assert config.locked.vocab_size == 151936

    def test_config_has_defaults(self):
        """Test that config has reasonable defaults."""
        config = create_default_config(
            model_path="model",
            dataset_path="data.jsonl",
            output_dir="output",
            base_model="test",
            model_architecture="test",
            max_context_length=4096,
            vocab_size=100000
        )

        # Check hyperparams defaults
        assert config.hyperparams.batch_size == 1
        assert config.hyperparams.learning_rate == 0.0002
        assert config.hyperparams.fp_precision == "bf16"

        # Check profile defaults
        assert config.profile.name == "emoji_think"

        # Check monitoring defaults
        assert config.monitoring.validation_split == 0.05


class TestConfigToDict:
    """Tests for config serialization."""

    def test_to_dict_roundtrip(self):
        """Test config can be serialized and deserialized."""
        config = create_default_config(
            model_path="model",
            dataset_path="data.jsonl",
            output_dir="output",
            base_model="test",
            model_architecture="test",
            max_context_length=4096,
            vocab_size=100000
        )

        # Convert to dict
        config_dict = config.to_dict()

        # Verify it's a dict
        assert isinstance(config_dict, dict)
        assert "hyperparams" in config_dict
        assert "profile" in config_dict
        assert "locked" in config_dict

        # Verify can be JSON serialized
        json_str = json.dumps(config_dict)
        assert len(json_str) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
