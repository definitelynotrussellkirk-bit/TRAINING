"""
Watchtower Types - Core types for the observation and monitoring system.

The Watchtower stands high above the training grounds, providing
a commanding view of all operations:

    WatcherState    - Current status of a watcher
    AlertLevel      - Severity of notifications
    ChampionRank    - Model checkpoint ranking
    OracleResponse  - Prophecy (inference) result
    ScryingVision   - Real-time training view

RPG Flavor:
    From the Watchtower, sentries observe the training grounds below.
    The Scrying Pool shows real-time combat. The Champion Board ranks
    the heroes. The Oracle speaks prophecies. Heralds carry alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# WATCHER STATES
# =============================================================================

class WatcherState(Enum):
    """State of a watcher daemon."""
    DORMANT = "dormant"        # Not started
    WATCHING = "watching"      # Actively observing
    ALERTING = "alerting"      # Triggered alert
    RESTING = "resting"        # Paused
    FALLEN = "fallen"          # Crashed


class AlertLevel(Enum):
    """Severity of herald announcements."""
    WHISPER = "whisper"        # Debug info
    NOTICE = "notice"          # Informational
    WARNING = "warning"        # Needs attention
    ALARM = "alarm"            # Urgent
    CRISIS = "crisis"          # Critical emergency


# =============================================================================
# CHAMPION BOARD (Model Rankings)
# =============================================================================

@dataclass
class ChampionRank:
    """
    Ranking of a model checkpoint on the Champion Board.

    The Champion Board tracks all heroes who have trained in the Arena,
    ranking them by their combat prowess.
    """
    checkpoint_name: str           # e.g., "checkpoint-175000"
    checkpoint_path: str           # Full path

    # Combat scores (normalized 0-1)
    combat_score: float = 0.0      # Overall composite score
    damage_resilience: float = 0.0 # 1 - validation_loss (higher = better)
    hit_accuracy: float = 0.0      # Validation accuracy
    response_speed: float = 0.0    # Tokens per second

    # Raw metrics
    validation_loss: float = 0.0
    validation_accuracy: float = 0.0
    tokens_per_second: float = 0.0

    # Ranking
    rank: int = 0                  # 1 = current champion
    previous_rank: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": self.checkpoint_path,
            "combat_score": self.combat_score,
            "damage_resilience": self.damage_resilience,
            "hit_accuracy": self.hit_accuracy,
            "response_speed": self.response_speed,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "tokens_per_second": self.tokens_per_second,
            "rank": self.rank,
            "previous_rank": self.previous_rank,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


@dataclass
class ChampionBoardStatus:
    """Current state of the Champion Board."""
    total_champions: int = 0
    current_champion: Optional[ChampionRank] = None
    recent_contenders: List[ChampionRank] = field(default_factory=list)
    last_tournament: Optional[datetime] = None
    next_tournament: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_champions": self.total_champions,
            "current_champion": self.current_champion.to_dict() if self.current_champion else None,
            "recent_contenders": [c.to_dict() for c in self.recent_contenders],
            "last_tournament": self.last_tournament.isoformat() if self.last_tournament else None,
            "next_tournament": self.next_tournament.isoformat() if self.next_tournament else None,
        }


# =============================================================================
# ORACLE (Inference)
# =============================================================================

@dataclass
class OracleResponse:
    """
    Response from the Oracle (inference result).

    The Oracle speaks prophecies - given a question, it reveals the answer.
    """
    prophecy: str                  # The generated text

    # Performance
    tokens_generated: int = 0
    time_taken_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Model info
    oracle_name: str = ""          # Model name
    oracle_version: str = ""       # Checkpoint

    # Request details
    prompt_tokens: int = 0
    total_tokens: int = 0

    # Status
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prophecy": self.prophecy,
            "tokens_generated": self.tokens_generated,
            "time_taken_ms": self.time_taken_ms,
            "tokens_per_second": self.tokens_per_second,
            "oracle_name": self.oracle_name,
            "oracle_version": self.oracle_version,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "error": self.error,
        }


# =============================================================================
# SCRYING POOL (Real-time Observation)
# =============================================================================

@dataclass
class ScryingVision:
    """
    A vision from the Scrying Pool showing current training state.

    The Scrying Pool is a magical basin that shows real-time events
    in the Arena below.
    """
    # Current battle
    battle_state: str = "idle"     # idle, fighting, paused, victory, etc.
    quest_file: Optional[str] = None

    # Combat progress
    current_round: int = 0
    total_rounds: int = 0
    campaign: int = 0              # Epoch
    total_campaigns: int = 1

    # Combat metrics
    damage_taken: float = 0.0      # Loss
    hit_rate: Optional[float] = None  # Accuracy
    rounds_per_second: float = 0.0

    # Time estimates
    time_remaining: Optional[str] = None
    eta: Optional[str] = None

    # Hero state
    hero_vram_mb: float = 0.0
    hero_checkpoint: Optional[str] = None

    # Observation metadata
    observed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_rounds == 0:
            return 0.0
        return (self.current_round / self.total_rounds) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battle_state": self.battle_state,
            "quest_file": self.quest_file,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "progress_percent": self.progress_percent,
            "campaign": self.campaign,
            "total_campaigns": self.total_campaigns,
            "damage_taken": self.damage_taken,
            "hit_rate": self.hit_rate,
            "rounds_per_second": self.rounds_per_second,
            "time_remaining": self.time_remaining,
            "eta": self.eta,
            "hero_vram_mb": self.hero_vram_mb,
            "hero_checkpoint": self.hero_checkpoint,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
        }


# =============================================================================
# HERALD (Alerts)
# =============================================================================

@dataclass
class HeraldMessage:
    """
    A message carried by a Herald.

    Heralds are messengers who announce important events throughout
    the training kingdom.
    """
    level: AlertLevel
    title: str
    message: str
    source: str = "watchtower"     # Which system sent it

    # Context
    details: Dict[str, Any] = field(default_factory=dict)

    # Timing
    announced_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "details": self.details,
            "announced_at": self.announced_at.isoformat() if self.announced_at else None,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


# =============================================================================
# TASK SENTRY (GPU Task Scheduling)
# =============================================================================

class TaskPriority(Enum):
    """Priority levels for Task Sentry queue."""
    CRITICAL = 0    # Must run immediately
    HIGH = 1        # Important, run soon
    NORMAL = 2      # Standard priority
    LOW = 3         # Can wait
    IDLE = 4        # Only when nothing else


class TaskStatus(Enum):
    """Status of a scheduled task."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SentryTask:
    """
    A task managed by the Task Sentry.

    The Task Sentry coordinates all GPU-bound operations, ensuring
    efficient use of the realm's magical resources (GPU).
    """
    task_id: str
    task_type: str                 # curriculum_eval, self_correction, etc.
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING

    # Execution
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority.name,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }
