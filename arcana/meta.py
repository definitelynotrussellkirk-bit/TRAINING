"""
Arcana Meta-Awareness Module.

Provides rich context about:
- Skill progress (level, accuracy, trends)
- Eval history (recent results, patterns)
- Training dynamics (loss trend, throughput)
- Plan history (what worked, what didn't)

This enables the LLM to make informed decisions based on historical patterns.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .world import World


@dataclass
class SkillProgress:
    """Progress tracking for a single skill."""
    id: str
    current_level: int
    training_level: int
    mastered_level: int
    max_level: int
    accuracy_history: List[Dict[str, Any]]

    @property
    def recent_accuracy(self) -> Optional[float]:
        """Most recent accuracy measurement."""
        if self.accuracy_history:
            return self.accuracy_history[-1].get('accuracy')
        return None

    @property
    def best_accuracy(self) -> Optional[float]:
        """Best accuracy ever achieved at current level."""
        if not self.accuracy_history:
            return None
        current_level_evals = [
            h['accuracy'] for h in self.accuracy_history
            if h.get('training_level') == self.training_level
        ]
        return max(current_level_evals) if current_level_evals else None

    @property
    def trend(self) -> str:
        """Compute accuracy trend from recent history."""
        if len(self.accuracy_history) < 2:
            return 'unknown'

        # Get last 5 at current level
        recent = [
            h['accuracy'] for h in self.accuracy_history[-5:]
            if h.get('training_level') == self.training_level
        ]

        if len(recent) < 2:
            return 'unknown'

        # Simple trend: compare first half to second half
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / mid if mid > 0 else 0
        second_half = sum(recent[mid:]) / (len(recent) - mid)

        diff = second_half - first_half
        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        return 'stable'

    @property
    def evals_at_level(self) -> int:
        """Number of evals at current training level."""
        return sum(
            1 for h in self.accuracy_history
            if h.get('training_level') == self.training_level
        )

    def to_sexpr(self) -> str:
        """Serialize to S-expression."""
        parts = [
            f":id {self.id}",
            f":level {self.training_level}",
            f":max-level {self.max_level}",
            f":mastered {self.mastered_level}",
        ]

        if self.recent_accuracy is not None:
            parts.append(f":accuracy {self.recent_accuracy:.2f}")

        parts.append(f":trend :{self.trend}")

        if self.best_accuracy is not None:
            parts.append(f":best {self.best_accuracy:.2f}")

        parts.append(f":evals {self.evals_at_level}")

        return f"(skill-progress {' '.join(parts)})"


@dataclass
class EvalResult:
    """A single evaluation result."""
    skill: str
    level: int
    accuracy: float
    timestamp: datetime
    step: int
    problems: int = 0
    correct: int = 0

    def to_sexpr(self) -> str:
        """Serialize to S-expression."""
        time_str = self.timestamp.strftime("%H:%M")
        return f"(eval-result :skill {self.skill} :level {self.level} :accuracy {self.accuracy:.2f} :time \"{time_str}\")"


@dataclass
class PlanOutcome:
    """Outcome of a previous plan."""
    plan_id: str
    goal: str
    forms_count: int
    executed_at: datetime
    outcome: str  # 'success', 'partial', 'failed'
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]

    @property
    def accuracy_delta(self) -> Optional[float]:
        """Change in accuracy from this plan."""
        before = self.metrics_before.get('accuracy')
        after = self.metrics_after.get('accuracy')
        if before is not None and after is not None:
            return after - before
        return None

    def to_sexpr(self) -> str:
        """Serialize to S-expression."""
        parts = [
            f":goal \"{self.goal}\"",
            f":forms {self.forms_count}",
            f":outcome :{self.outcome}",
        ]
        if self.accuracy_delta is not None:
            parts.append(f":accuracy-delta {self.accuracy_delta:+.2f}")
        return f"(last-plan {' '.join(parts)})"


class MetaContext:
    """
    Builds rich meta-aware context for the LLM planner.

    Loads data from:
    - curriculum_state.json (skill progress)
    - eval_results_history.json (recent evals)
    - training_status.json (current training)
    - plan_history.json (previous plans)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            from core.paths import get_base_dir
            self.base_dir = get_base_dir()
        self.skill_progress: Dict[str, SkillProgress] = {}
        self.eval_history: List[EvalResult] = []
        self.plan_history: List[PlanOutcome] = []
        self.training_status: Dict[str, Any] = {}

    def load(self):
        """Load all context from filesystem."""
        self._load_curriculum_state()
        self._load_eval_history()
        self._load_training_status()
        self._load_plan_history()

    def _load_curriculum_state(self):
        """Load skill progress from curriculum_state.json."""
        path = self.base_dir / 'data_manager' / 'curriculum_state.json'
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            # Load skill configs for max_level
            skill_configs = self._load_skill_configs()

            for skill_id, skill_data in data.get('skills', {}).items():
                max_level = skill_configs.get(skill_id, {}).get('max_level', 50)

                self.skill_progress[skill_id] = SkillProgress(
                    id=skill_id,
                    current_level=skill_data.get('current_level', 1),
                    training_level=skill_data.get('training_level', 1),
                    mastered_level=skill_data.get('mastered_level', 0),
                    max_level=max_level,
                    accuracy_history=skill_data.get('accuracy_history', [])
                )
        except Exception as e:
            print(f"Warning: Could not load curriculum state: {e}")

    def _load_skill_configs(self) -> Dict[str, Dict]:
        """Load skill configs from YAML files."""
        import yaml
        configs = {}
        skills_dir = self.base_dir / 'configs' / 'skills'

        if skills_dir.exists():
            for path in skills_dir.glob('*.yaml'):
                try:
                    with open(path) as f:
                        data = yaml.safe_load(f)
                    if data and 'id' in data:
                        configs[data['id']] = data
                except:
                    pass

        return configs

    def _load_eval_history(self):
        """Load recent eval results."""
        path = self.base_dir / 'status' / 'eval_results_history.json'

        # Resolve symlink
        if path.is_symlink():
            path = path.resolve()

        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            # Parse eval results
            for entry in data.get('results', [])[-20:]:  # Last 20
                try:
                    ts = datetime.fromisoformat(entry.get('timestamp', ''))
                except:
                    ts = datetime.now()

                self.eval_history.append(EvalResult(
                    skill=entry.get('skill', 'unknown'),
                    level=entry.get('level', 1),
                    accuracy=entry.get('accuracy', 0.0),
                    timestamp=ts,
                    step=entry.get('step', 0),
                    problems=entry.get('problems', 0),
                    correct=entry.get('correct', 0)
                ))
        except Exception as e:
            print(f"Warning: Could not load eval history: {e}")

    def _load_training_status(self):
        """Load current training status."""
        path = self.base_dir / 'status' / 'training_status.json'

        if path.is_symlink():
            path = path.resolve()

        if not path.exists():
            return

        try:
            with open(path) as f:
                self.training_status = json.load(f)
        except:
            pass

    def _load_plan_history(self):
        """Load previous plan outcomes."""
        path = self.base_dir / 'status' / 'arcana_plan_history.json'

        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for entry in data.get('plans', [])[-5:]:  # Last 5
                try:
                    ts = datetime.fromisoformat(entry.get('executed_at', ''))
                except:
                    ts = datetime.now()

                self.plan_history.append(PlanOutcome(
                    plan_id=entry.get('id', ''),
                    goal=entry.get('goal', ''),
                    forms_count=entry.get('forms_count', 0),
                    executed_at=ts,
                    outcome=entry.get('outcome', 'unknown'),
                    metrics_before=entry.get('metrics_before', {}),
                    metrics_after=entry.get('metrics_after', {})
                ))
        except:
            pass

    def save_plan_outcome(self, outcome: PlanOutcome):
        """Save a plan outcome to history."""
        path = self.base_dir / 'status' / 'arcana_plan_history.json'

        # Load existing
        data = {'plans': []}
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
            except:
                pass

        # Add new outcome
        data['plans'].append({
            'id': outcome.plan_id,
            'goal': outcome.goal,
            'forms_count': outcome.forms_count,
            'executed_at': outcome.executed_at.isoformat(),
            'outcome': outcome.outcome,
            'metrics_before': outcome.metrics_before,
            'metrics_after': outcome.metrics_after,
        })

        # Keep last 50
        data['plans'] = data['plans'][-50:]

        # Save
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        self.plan_history.append(outcome)

    def get_loss_trend(self) -> str:
        """Compute loss trend from training status."""
        trend = self.training_status.get('loss_trend')
        if trend:
            return trend

        # Fallback: compute from variance
        variance = self.training_status.get('loss_variance', 0)
        if variance > 0.001:
            return 'unstable'
        return 'stable'

    def serialize_rich(self) -> str:
        """Serialize full meta-aware context to S-expressions."""
        lines = ["; === META-AWARE WORLD STATE ==="]

        # Hero & Campaign
        hero_name = self.training_status.get('model_name', 'unknown')
        step = self.training_status.get('current_step', 0)
        total = self.training_status.get('total_steps', 0)
        lines.append("")
        lines.append("; Training Status")
        lines.append(f"(training :step {step} :total {total} :status :{self.training_status.get('status', 'unknown')})")

        # Loss and throughput
        loss = self.training_status.get('loss')
        if loss is not None:
            lines.append(f"(metric :loss {loss:.4f} :trend :{self.get_loss_trend()})")

        throughput = self.training_status.get('tokens_per_sec')
        if throughput:
            lines.append(f"(metric :throughput {throughput:.0f})")

        # Skill Progress
        if self.skill_progress:
            lines.append("")
            lines.append("; Skill Progress (level, accuracy, trend)")
            for skill in self.skill_progress.values():
                lines.append(skill.to_sexpr())

        # Recent Evals (last 5 per skill)
        if self.eval_history:
            lines.append("")
            lines.append("; Recent Evaluations")
            # Group by skill, show last 3 per skill
            by_skill: Dict[str, List[EvalResult]] = {}
            for ev in self.eval_history:
                by_skill.setdefault(ev.skill, []).append(ev)

            for skill_id, evals in by_skill.items():
                for ev in evals[-3:]:
                    lines.append(ev.to_sexpr())

        # Queue status
        queue_size = self.training_status.get('batch_queue_size', 0)
        lines.append("")
        lines.append("; Queue Status")
        lines.append(f"(queue :depth {queue_size})")

        # Last plan outcome (if any)
        if self.plan_history:
            lines.append("")
            lines.append("; Previous Plan Outcome")
            lines.append(self.plan_history[-1].to_sexpr())

        # Recommendations based on analysis
        lines.append("")
        lines.append("; Analysis Hints")
        hints = self._generate_hints()
        for hint in hints:
            lines.append(f"; {hint}")

        return "\n".join(lines)

    def _generate_hints(self) -> List[str]:
        """Generate analysis hints for the LLM."""
        hints = []

        for skill_id, progress in self.skill_progress.items():
            # Accuracy declining
            if progress.trend == 'declining':
                hints.append(f"{skill_id}: accuracy declining, consider easier content or more training")

            # Ready to level up
            if progress.recent_accuracy and progress.recent_accuracy >= 0.8:
                if progress.evals_at_level >= 3:
                    hints.append(f"{skill_id}: accuracy >= 80% over 3+ evals, ready to level up")

            # Struggling
            if progress.recent_accuracy and progress.recent_accuracy < 0.3:
                if progress.evals_at_level >= 2:
                    hints.append(f"{skill_id}: accuracy < 30%, consider level down")

        # Loss trend
        loss_trend = self.get_loss_trend()
        if loss_trend == 'unstable':
            hints.append("Loss variance high - training may be unstable")

        return hints


def get_meta_context(base_dir: Optional[Path] = None) -> MetaContext:
    """Create and load a MetaContext."""
    ctx = MetaContext(base_dir)
    ctx.load()
    return ctx


# --- Integration with Planner ---

def serialize_world_meta(world: World, base_dir: Optional[Path] = None) -> str:
    """
    Serialize world with full meta-awareness.

    This is the enhanced version of serialize_world_compact that includes:
    - Skill progress with trends
    - Recent eval history
    - Plan history
    - Analysis hints
    """
    ctx = get_meta_context(base_dir or world.base_dir)
    return ctx.serialize_rich()


if __name__ == '__main__':
    # Demo
    ctx = get_meta_context()
    print(ctx.serialize_rich())
