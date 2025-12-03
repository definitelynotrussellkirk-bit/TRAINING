"""
World State for the Arcana DSL.

The World holds all entities (heroes, campaigns, quests, skills) and
provides query/mutation methods. It can sync with the actual filesystem
(loading from YAML/JSON configs) or be purely in-memory.
"""

import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Entity:
    """A single entity in the world."""
    kind: str           # 'hero', 'campaign', 'quest', 'skill'
    id: str             # Unique identifier
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a property from entity data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """Set a property in entity data."""
        self.data[key] = value
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'kind': self.kind,
            'id': self.id,
            **self.data
        }


@dataclass
class ActionLog:
    """Record of an action taken in the world."""
    action: str
    args: Dict[str, Any]
    result: Any
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None


class World:
    """
    The game world - holds all entities and metrics.

    Entity types:
        - hero: A model being trained (DIO, GOU, etc.)
        - campaign: A training campaign/playthrough
        - quest: A training job (dataset + config)
        - skill: A learnable skill (SY, BIN, etc.)
        - checkpoint: A saved model state

    Metrics are key-value pairs updated during training.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.entities: Dict[str, Dict[str, Entity]] = {}
        self.metrics: Dict[str, float] = {}
        self.log: List[ActionLog] = []
        self._observers: List[callable] = []

    # --- Entity Management ---

    def add(self, kind: str, id: str, data: Dict[str, Any]) -> Entity:
        """Add or update an entity."""
        if kind not in self.entities:
            self.entities[kind] = {}

        entity = Entity(kind=kind, id=id, data=data)
        self.entities[kind][id] = entity
        self._notify('entity_added', entity)
        return entity

    def get(self, kind: str, id: str) -> Optional[Entity]:
        """Get an entity by kind and id."""
        return self.entities.get(kind, {}).get(id)

    def get_or_fail(self, kind: str, id: str) -> Entity:
        """Get an entity, raising if not found."""
        entity = self.get(kind, id)
        if entity is None:
            raise KeyError(f"No {kind} with id '{id}'")
        return entity

    def list(self, kind: str) -> List[Entity]:
        """List all entities of a kind."""
        return list(self.entities.get(kind, {}).values())

    def remove(self, kind: str, id: str) -> bool:
        """Remove an entity."""
        if kind in self.entities and id in self.entities[kind]:
            del self.entities[kind][id]
            return True
        return False

    def query(self, kind: str, **filters) -> List[Entity]:
        """Query entities with filters."""
        results = []
        for entity in self.list(kind):
            match = True
            for key, value in filters.items():
                if entity.get(key) != value:
                    match = False
                    break
            if match:
                results.append(entity)
        return results

    # --- Metrics ---

    def set_metric(self, name: str, value: float):
        """Set a metric value."""
        self.metrics[name] = value
        self._notify('metric_updated', name, value)

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a metric value."""
        return self.metrics.get(name, default)

    # --- Action Logging ---

    def record_action(self, action: str, args: Dict[str, Any],
                      result: Any, success: bool = True, error: Optional[str] = None):
        """Record an action in the log."""
        entry = ActionLog(
            action=action,
            args=args,
            result=result,
            success=success,
            error=error
        )
        self.log.append(entry)
        self._notify('action_logged', entry)
        return entry

    def get_log(self, limit: int = 100) -> List[ActionLog]:
        """Get recent log entries."""
        return self.log[-limit:]

    # --- Observers ---

    def observe(self, callback: callable):
        """Add an observer for world changes."""
        self._observers.append(callback)

    def _notify(self, event: str, *args):
        """Notify observers of an event."""
        for observer in self._observers:
            try:
                observer(event, *args)
            except Exception:
                pass  # Don't let observer errors break the world

    # --- Sync with Filesystem ---

    def load_heroes(self) -> int:
        """Load heroes from configs/heroes/*.yaml"""
        count = 0
        heroes_dir = self.base_dir / 'configs' / 'heroes'
        if not heroes_dir.exists():
            return 0

        for path in heroes_dir.glob('*.yaml'):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and 'id' in data:
                    self.add('hero', data['id'], data)
                    count += 1
            except Exception as e:
                print(f"Warning: Could not load hero from {path}: {e}")

        return count

    def load_skills(self) -> int:
        """Load skills from configs/skills/*.yaml"""
        count = 0
        skills_dir = self.base_dir / 'configs' / 'skills'
        if not skills_dir.exists():
            return 0

        for path in skills_dir.glob('*.yaml'):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and 'id' in data:
                    self.add('skill', data['id'], data)
                    count += 1
            except Exception as e:
                print(f"Warning: Could not load skill from {path}: {e}")

        return count

    def load_active_campaign(self) -> Optional[Entity]:
        """Load the active campaign from control/active_campaign.json"""
        active_file = self.base_dir / 'control' / 'active_campaign.json'
        if not active_file.exists():
            return None

        try:
            with open(active_file) as f:
                active = json.load(f)

            hero_id = active.get('hero_id')
            campaign_id = active.get('campaign_id')
            if not hero_id or not campaign_id:
                return None

            # Load campaign.json
            campaign_dir = self.base_dir / 'campaigns' / hero_id / campaign_id
            campaign_file = campaign_dir / 'campaign.json'
            if campaign_file.exists():
                with open(campaign_file) as f:
                    data = json.load(f)
                data['hero_id'] = hero_id
                data['active'] = True
                return self.add('campaign', campaign_id, data)

        except Exception as e:
            print(f"Warning: Could not load active campaign: {e}")

        return None

    def load_training_status(self) -> Dict[str, Any]:
        """Load current training status."""
        status_file = self.base_dir / 'status' / 'training_status.json'
        if not status_file.exists():
            return {}

        try:
            # Resolve symlink
            actual = status_file.resolve()
            with open(actual) as f:
                status = json.load(f)

            # Update metrics from status
            if 'loss' in status:
                self.set_metric('train_loss', status['loss'])
            if 'current_step' in status:
                self.set_metric('step', status['current_step'])
            if 'accuracy_percent' in status:
                self.set_metric('accuracy', status['accuracy_percent'])
            if 'tokens_per_sec' in status:
                self.set_metric('throughput', status['tokens_per_sec'])

            return status

        except Exception as e:
            print(f"Warning: Could not load training status: {e}")
            return {}

    def sync_from_filesystem(self) -> Dict[str, int]:
        """Load all entities from filesystem."""
        counts = {
            'heroes': self.load_heroes(),
            'skills': self.load_skills(),
        }

        campaign = self.load_active_campaign()
        counts['campaigns'] = 1 if campaign else 0

        self.load_training_status()

        return counts

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize world to dictionary."""
        return {
            'entities': {
                kind: {id: e.to_dict() for id, e in entities.items()}
                for kind, entities in self.entities.items()
            },
            'metrics': self.metrics.copy(),
        }

    def to_sexpr(self) -> str:
        """Serialize world to S-expressions."""
        from .parser import to_sexpr

        lines = ['; World state', '']

        for kind in ['hero', 'campaign', 'skill', 'quest']:
            entities = self.list(kind)
            if entities:
                lines.append(f'; {kind.title()}s')
                for entity in entities:
                    form = [kind]
                    form.append(':id')
                    form.append(entity.id)
                    for k, v in entity.data.items():
                        if k != 'id':
                            form.append(f':{k}')
                            form.append(v)
                    lines.append(to_sexpr(form))
                lines.append('')

        if self.metrics:
            lines.append('; Metrics')
            for name, value in self.metrics.items():
                lines.append(f'(metric :{name} {value})')

        return '\n'.join(lines)


def get_world(base_dir: Optional[Path] = None) -> World:
    """Get a World instance, optionally synced from filesystem."""
    world = World(base_dir)
    return world


if __name__ == '__main__':
    # Quick test
    from core.paths import get_base_dir

    base = get_base_dir()
    world = World(base)
    counts = world.sync_from_filesystem()
    print(f"Loaded: {counts}")
    print()
    print(world.to_sexpr())
