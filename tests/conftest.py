"""
Shared pytest fixtures for CI-safe testing.

All fixtures use temporary directories - no hardcoded paths.
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Provides a temporary directory for tests.

    Automatically cleaned up after test completes.
    """
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_base_dir(temp_dir: Path) -> Path:
    """
    Creates a mock TRAINING directory structure.

    Returns path to the temp base dir with standard subdirs created.
    """
    # Create standard subdirectories
    (temp_dir / "queue" / "high").mkdir(parents=True)
    (temp_dir / "queue" / "normal").mkdir(parents=True)
    (temp_dir / "queue" / "low").mkdir(parents=True)
    (temp_dir / "queue" / "processing").mkdir(parents=True)
    (temp_dir / "queue" / "failed").mkdir(parents=True)
    (temp_dir / "status").mkdir(parents=True)
    (temp_dir / "logs").mkdir(parents=True)
    (temp_dir / "data").mkdir(parents=True)
    (temp_dir / "models").mkdir(parents=True)
    (temp_dir / "current_model").mkdir(parents=True)

    return temp_dir


@pytest.fixture
def mock_checkpoint(temp_dir: Path) -> Path:
    """
    Creates a fake checkpoint directory.

    Returns path to the checkpoint.
    """
    ckpt = temp_dir / "checkpoint-1000"
    ckpt.mkdir(parents=True)

    # Create minimal checkpoint files
    (ckpt / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "hidden_size": 512,
        "num_hidden_layers": 2
    }))

    (ckpt / "trainer_state.json").write_text(json.dumps({
        "global_step": 1000,
        "log_history": [
            {"step": 1000, "loss": 0.45, "eval_loss": 0.52}
        ]
    }))

    # Create dummy model file (small)
    (ckpt / "model.safetensors").write_bytes(b"fake model weights" * 100)

    return ckpt


@pytest.fixture
def mock_config(temp_base_dir: Path) -> Path:
    """
    Creates a mock config.json in the base dir.

    Returns path to the config file.
    """
    config = {
        "model_name": "test_model",
        "model_path": str(temp_base_dir / "models" / "test_model"),
        "batch_size": 4,
        "learning_rate": 0.0001,
        "eval_steps": 50,
        "save_steps": 100,
        "max_length": 512,
        "poll_interval": 10
    }

    config_path = temp_base_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    return config_path


@pytest.fixture
def mock_training_data(temp_dir: Path) -> Path:
    """
    Creates a mock training data file.

    Returns path to the jsonl file.
    """
    data_path = temp_dir / "training_data.jsonl"

    # Create sample training examples
    examples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]
        }
    ]

    with open(data_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    return data_path


@pytest.fixture
def mock_status_file(temp_base_dir: Path) -> Path:
    """
    Creates a mock training_status.json.

    Returns path to the status file.
    """
    status = {
        "current_step": 1000,
        "total_steps": 5000,
        "loss": 0.45,
        "validation_loss": 0.52,
        "epoch": 1,
        "samples_processed": 4000,
        "last_checkpoint": "checkpoint-1000",
        "started_at": datetime.now().isoformat(),
        "status": "training"
    }

    status_path = temp_base_dir / "status" / "training_status.json"
    status_path.write_text(json.dumps(status, indent=2))

    return status_path


# Skip markers for conditional test execution
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "local: marks tests as local-only (hardcoded paths)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
