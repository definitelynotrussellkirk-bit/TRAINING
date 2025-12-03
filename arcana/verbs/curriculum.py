"""
Curriculum Verbs - Skill level management and evaluation control.

Verbs:
    (level-up :skill sy)           ; Advance skill to next level
    (level-down :skill sy)         ; Regress skill to previous level
    (set-level :skill sy :level 3) ; Set skill to specific level
    (run-eval :skill sy :samples 20)  ; Trigger evaluation
    (generate-data :skill bin :level 2 :count 1000)  ; Generate training data
    (skill-status :skill sy)       ; Get detailed skill status
    (compare-skills)               ; Compare all skills
    (suggest-action)               ; Get AI suggestion for next action
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import _parse_kwargs


def _get_curriculum_state(base_dir: Path) -> Dict[str, Any]:
    """Load curriculum state."""
    path = base_dir / 'data_manager' / 'curriculum_state.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {'skills': {}}


def _save_curriculum_state(base_dir: Path, state: Dict[str, Any]):
    """Save curriculum state."""
    path = base_dir / 'data_manager' / 'curriculum_state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def verb_level_up(engine, *args):
    """Advance skill to next level.

    (level-up :skill sy)
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.get('skill')

    if not skill_id:
        raise ValueError("level-up requires :skill")

    # Handle keyword format
    if isinstance(skill_id, str) and skill_id.startswith(':'):
        skill_id = skill_id[1:]

    base_dir = engine.world.base_dir
    state = _get_curriculum_state(base_dir)

    if skill_id not in state.get('skills', {}):
        state.setdefault('skills', {})[skill_id] = {
            'current_level': 1,
            'training_level': 1,
            'mastered_level': 0,
            'accuracy_history': []
        }

    skill = state['skills'][skill_id]
    old_level = skill.get('training_level', 1)
    new_level = old_level + 1

    # Check max level from skill config
    skill_entity = engine.world.get('skill', skill_id)
    max_level = 50
    if skill_entity:
        max_level = skill_entity.get('max_level', 50)

    if new_level > max_level:
        return {'error': f'{skill_id} already at max level {max_level}'}

    skill['training_level'] = new_level
    skill['current_level'] = new_level
    state['last_updated'] = datetime.now().isoformat()

    _save_curriculum_state(base_dir, state)

    return {
        'skill': skill_id,
        'action': 'level_up',
        'old_level': old_level,
        'new_level': new_level
    }


def verb_level_down(engine, *args):
    """Regress skill to previous level.

    (level-down :skill sy)
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.get('skill')

    if not skill_id:
        raise ValueError("level-down requires :skill")

    if isinstance(skill_id, str) and skill_id.startswith(':'):
        skill_id = skill_id[1:]

    base_dir = engine.world.base_dir
    state = _get_curriculum_state(base_dir)

    if skill_id not in state.get('skills', {}):
        return {'error': f'{skill_id} not found'}

    skill = state['skills'][skill_id]
    old_level = skill.get('training_level', 1)
    new_level = max(1, old_level - 1)

    if new_level == old_level:
        return {'error': f'{skill_id} already at level 1'}

    skill['training_level'] = new_level
    skill['current_level'] = new_level
    state['last_updated'] = datetime.now().isoformat()

    _save_curriculum_state(base_dir, state)

    return {
        'skill': skill_id,
        'action': 'level_down',
        'old_level': old_level,
        'new_level': new_level
    }


def verb_set_level(engine, *args):
    """Set skill to specific level.

    (set-level :skill sy :level 3)
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.get('skill')
    level = kwargs.get('level')

    if not skill_id or level is None:
        raise ValueError("set-level requires :skill and :level")

    if isinstance(skill_id, str) and skill_id.startswith(':'):
        skill_id = skill_id[1:]

    base_dir = engine.world.base_dir
    state = _get_curriculum_state(base_dir)

    state.setdefault('skills', {}).setdefault(skill_id, {
        'current_level': 1,
        'training_level': 1,
        'mastered_level': 0,
        'accuracy_history': []
    })

    skill = state['skills'][skill_id]
    old_level = skill.get('training_level', 1)

    skill['training_level'] = int(level)
    skill['current_level'] = int(level)
    state['last_updated'] = datetime.now().isoformat()

    _save_curriculum_state(base_dir, state)

    return {
        'skill': skill_id,
        'action': 'set_level',
        'old_level': old_level,
        'new_level': int(level)
    }


def verb_run_eval(engine, *args):
    """Trigger evaluation for a skill.

    (run-eval :skill sy)
    (run-eval :skill sy :samples 20)
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.get('skill')
    samples = kwargs.get('samples', 10)

    if not skill_id:
        raise ValueError("run-eval requires :skill")

    if isinstance(skill_id, str) and skill_id.startswith(':'):
        skill_id = skill_id[1:]

    base_dir = engine.world.base_dir

    # Add eval request to queue
    eval_queue_path = base_dir / 'status' / 'eval_queue.json'

    queue = {'pending': []}
    if eval_queue_path.exists():
        try:
            with open(eval_queue_path) as f:
                queue = json.load(f)
        except:
            pass

    request = {
        'skill': skill_id,
        'samples': int(samples),
        'requested_at': datetime.now().isoformat(),
        'source': 'arcana'
    }

    queue.setdefault('pending', []).append(request)

    with open(eval_queue_path, 'w') as f:
        json.dump(queue, f, indent=2)

    return {
        'action': 'eval_queued',
        'skill': skill_id,
        'samples': samples
    }


def verb_generate_data(engine, *args):
    """Generate training data for a skill.

    (generate-data :skill bin :level 2 :count 1000)
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.get('skill')
    level = kwargs.get('level', 1)
    count = kwargs.get('count', 500)

    if not skill_id:
        raise ValueError("generate-data requires :skill")

    if isinstance(skill_id, str) and skill_id.startswith(':'):
        skill_id = skill_id[1:]

    # Get skill API info
    skill_entity = engine.world.get('skill', skill_id)
    if not skill_entity:
        return {'error': f'Unknown skill: {skill_id}'}

    api_info = skill_entity.get('api', {})
    port = api_info.get('port')

    if not port:
        return {'error': f'No API port configured for {skill_id}'}

    # Queue a data generation request
    base_dir = engine.world.base_dir
    gen_queue_path = base_dir / 'status' / 'data_gen_queue.json'

    queue = {'pending': []}
    if gen_queue_path.exists():
        try:
            with open(gen_queue_path) as f:
                queue = json.load(f)
        except:
            pass

    request = {
        'skill': skill_id,
        'level': int(level),
        'count': int(count),
        'port': port,
        'requested_at': datetime.now().isoformat(),
        'source': 'arcana'
    }

    queue.setdefault('pending', []).append(request)

    with open(gen_queue_path, 'w') as f:
        json.dump(queue, f, indent=2)

    return {
        'action': 'data_gen_queued',
        'skill': skill_id,
        'level': level,
        'count': count
    }


def verb_skill_status(engine, *args):
    """Get detailed skill status.

    (skill-status :skill sy)
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.get('skill')

    if not skill_id:
        raise ValueError("skill-status requires :skill")

    if isinstance(skill_id, str) and skill_id.startswith(':'):
        skill_id = skill_id[1:]

    base_dir = engine.world.base_dir
    state = _get_curriculum_state(base_dir)

    skill_state = state.get('skills', {}).get(skill_id, {})
    skill_entity = engine.world.get('skill', skill_id)

    # Compute stats
    history = skill_state.get('accuracy_history', [])
    recent = history[-5:] if history else []

    return {
        'skill': skill_id,
        'training_level': skill_state.get('training_level', 1),
        'mastered_level': skill_state.get('mastered_level', 0),
        'max_level': skill_entity.get('max_level', 50) if skill_entity else 50,
        'recent_evals': len(recent),
        'recent_accuracy': [h.get('accuracy', 0) for h in recent],
        'best_accuracy': max((h.get('accuracy', 0) for h in history), default=0),
    }


def verb_compare_skills(engine):
    """Compare all skills and identify which needs attention.

    (compare-skills)
    """
    base_dir = engine.world.base_dir
    state = _get_curriculum_state(base_dir)

    results = []
    for skill_id, skill_state in state.get('skills', {}).items():
        history = skill_state.get('accuracy_history', [])
        recent_acc = history[-1].get('accuracy', 0) if history else None

        skill_entity = engine.world.get('skill', skill_id)
        max_level = skill_entity.get('max_level', 50) if skill_entity else 50

        results.append({
            'skill': skill_id,
            'level': skill_state.get('training_level', 1),
            'max_level': max_level,
            'progress_pct': (skill_state.get('training_level', 1) / max_level) * 100,
            'recent_accuracy': recent_acc,
            'evals': len(history),
        })

    # Sort by which needs most attention (lowest accuracy first)
    results.sort(key=lambda x: x.get('recent_accuracy') or 0)

    return {
        'skills': results,
        'recommendation': results[0]['skill'] if results else None
    }


def verb_suggest_action(engine):
    """Suggest next action based on current state.

    (suggest-action)
    """
    from ..meta import get_meta_context

    ctx = get_meta_context(engine.world.base_dir)

    suggestions = []

    for skill_id, progress in ctx.skill_progress.items():
        # Ready to level up
        if progress.recent_accuracy and progress.recent_accuracy >= 0.8:
            if progress.evals_at_level >= 3:
                suggestions.append({
                    'action': 'level-up',
                    'skill': skill_id,
                    'reason': f'accuracy {progress.recent_accuracy:.0%} over {progress.evals_at_level} evals'
                })

        # Struggling, level down
        elif progress.recent_accuracy and progress.recent_accuracy < 0.3:
            if progress.evals_at_level >= 2 and progress.training_level > 1:
                suggestions.append({
                    'action': 'level-down',
                    'skill': skill_id,
                    'reason': f'accuracy only {progress.recent_accuracy:.0%}'
                })

        # Needs eval
        elif progress.evals_at_level == 0:
            suggestions.append({
                'action': 'run-eval',
                'skill': skill_id,
                'reason': 'no evals at current level'
            })

        # Declining, more training
        elif progress.trend == 'declining':
            suggestions.append({
                'action': 'train',
                'skill': skill_id,
                'reason': 'accuracy declining'
            })

    return {
        'suggestions': suggestions[:3],  # Top 3
        'analysis': 'Based on skill progress and eval history'
    }
