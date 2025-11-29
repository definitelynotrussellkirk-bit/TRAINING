"""Base trainer interface for all hero trainers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class TrainingResult:
    """Result from a training run."""
    success: bool
    steps_completed: int = 0
    final_loss: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    duration_seconds: float = 0.0
    checkpoint_path: Optional[Path] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseTrainer(ABC):
    """
    Abstract base class for hero trainers.
    
    All trainers must implement:
    - train(data_path) -> TrainingResult
    - name property
    
    The hero loop uses this interface to train any hero type.
    """
    
    def __init__(self, hero_config: Dict[str, Any], campaign_path: Path):
        """
        Initialize trainer.
        
        Args:
            hero_config: Hero profile from configs/heroes/*.yaml
            campaign_path: Path to campaign directory (e.g., campaigns/titan-qwen3-4b/campaign-001)
        """
        self.hero_config = hero_config
        self.campaign_path = campaign_path
        self.checkpoints_dir = campaign_path / "checkpoints"
        self.data_dir = campaign_path / "data"
        
        # Ensure directories exist
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Trainer name (e.g., 'flo', 'ultimate')."""
        pass
    
    @abstractmethod
    def train(self, data_path: Path) -> TrainingResult:
        """
        Run training on the given data file.
        
        Args:
            data_path: Path to JSONL training data
            
        Returns:
            TrainingResult with success status and metrics
        """
        pass
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint in the campaign."""
        checkpoints = list(self.checkpoints_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    def get_model_path(self) -> Path:
        """Get the model path to use for training."""
        # Check for latest checkpoint first
        latest = self.get_latest_checkpoint()
        if latest:
            return latest
        
        # Fall back to base model from hero config
        from core.paths import get_base_dir
        base_dir = get_base_dir()
        model_path = self.hero_config.get("model", {}).get("hf_name", "")
        return base_dir / model_path
