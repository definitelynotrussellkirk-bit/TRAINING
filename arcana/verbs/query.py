"""
Query Verbs - Read world state and metrics.

Verbs:
    (metric :name loss)
    (status)
    (world-state)
    (log :limit 10)
"""

import json
from pathlib import Path
from typing import Any, Dict

from .core import _parse_kwargs


def verb_metric(engine, *args):
    """Get a metric value.

    (metric :name loss)
    (metric :name step)
    (metric :campaign c-001 :name accuracy)
    """
    kwargs = _parse_kwargs(args)
    name = kwargs.get('name')

    if not name:
        raise ValueError("metric requires :name")

    # Handle keyword format (:loss -> loss)
    if isinstance(name, str) and name.startswith(':'):
        name = name[1:]

    return engine.world.get_metric(name, 0.0)


def verb_status(engine):
    """Get overall system status.

    (status)
    """
    base_dir = engine.world.base_dir

    # Load training status
    status_file = base_dir / 'status' / 'training_status.json'
    training = {}
    if status_file.exists():
        try:
            actual = status_file.resolve()
            with open(actual) as f:
                training = json.load(f)
        except:
            pass

    # Count entities
    entities = {}
    for kind in ['hero', 'campaign', 'quest', 'skill']:
        entities[kind] = len(engine.world.list(kind))

    return {
        'training': {
            'status': training.get('status', 'unknown'),
            'step': training.get('current_step', 0),
            'total_steps': training.get('total_steps', 0),
            'loss': training.get('loss'),
            'model': training.get('model_name'),
        },
        'entities': entities,
        'metrics': dict(engine.world.metrics),
    }


def verb_world_state(engine):
    """Get full world state as S-expressions.

    (world-state)
    """
    return engine.world.to_sexpr()


def verb_log(engine, *args):
    """Get recent action log.

    (log)
    (log :limit 20)
    """
    kwargs = _parse_kwargs(args) if args else {}
    limit = kwargs.get('limit', 10)

    entries = engine.world.get_log(limit)
    return [
        {
            'action': e.action,
            'args': e.args,
            'result': str(e.result)[:100],  # Truncate
            'success': e.success,
            'timestamp': e.timestamp.isoformat()
        }
        for e in entries
    ]


def verb_get_hero(engine, hero_id):
    """Get a hero by id.

    (get-hero dio)
    """
    hero_id = engine.eval(hero_id)
    entity = engine.world.get('hero', hero_id)
    if entity:
        return entity.to_dict()
    return None


def verb_get_campaign(engine, campaign_id=None):
    """Get a campaign, or the active campaign if no id given.

    (get-campaign)
    (get-campaign c-001)
    """
    if campaign_id:
        campaign_id = engine.eval(campaign_id)
        entity = engine.world.get('campaign', campaign_id)
        if entity:
            return entity.to_dict()
        return None

    # Get active campaign
    campaigns = engine.world.query('campaign', active=True)
    if campaigns:
        return campaigns[0].to_dict()
    return None


def verb_get_skill(engine, skill_id):
    """Get a skill by id.

    (get-skill sy)
    """
    skill_id = engine.eval(skill_id)
    entity = engine.world.get('skill', skill_id)
    if entity:
        return entity.to_dict()
    return None


def verb_all_metrics(engine):
    """Get all metrics.

    (all-metrics)
    """
    return dict(engine.world.metrics)


def verb_refresh_metrics(engine):
    """Refresh metrics from filesystem.

    (refresh-metrics)
    """
    status = engine.world.load_training_status()
    return {
        'refreshed': True,
        'metrics': dict(engine.world.metrics)
    }


def verb_sync_world(engine):
    """Sync world state from filesystem.

    (sync-world)
    """
    counts = engine.world.sync_from_filesystem()
    return {
        'synced': True,
        'counts': counts
    }
