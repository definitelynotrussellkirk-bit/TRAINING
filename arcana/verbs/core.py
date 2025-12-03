"""
Core Verbs - Entity definition and management.

Verbs:
    (hero :id dio :name "DIO" :model "qwen3-0.6b" ...)
    (campaign :id c-001 :hero dio :step 0 ...)
    (quest :id q1 :dataset "train.jsonl" :steps 500 ...)
    (skill :id sy :name "Syllacrostic" :level 1 ...)
    (get-entity kind id)
    (update-entity kind id :key value ...)
"""

from typing import Any, Dict


def _parse_kwargs(args) -> Dict[str, Any]:
    """Parse keyword arguments from (:key value :key2 value2 ...)"""
    result = {}
    it = iter(args)
    for item in it:
        if isinstance(item, str) and item.startswith(':'):
            key = item[1:]  # Remove colon
            try:
                value = next(it)
                result[key] = value
            except StopIteration:
                raise ValueError(f"Missing value for keyword {item}")
        else:
            raise ValueError(f"Expected keyword, got {item}")
    return result


def verb_hero(engine, *args):
    """Define or update a hero.

    (hero :id dio :name "DIO" :model "qwen3-0.6b" :level 7)
    """
    kwargs = _parse_kwargs(args)
    hero_id = kwargs.pop('id', None)
    if not hero_id:
        raise ValueError("hero requires :id")

    entity = engine.world.add('hero', hero_id, kwargs)
    return entity.to_dict()


def verb_campaign(engine, *args):
    """Define or update a campaign.

    (campaign :id c-001 :hero dio :step 0 :objective "mastery")
    """
    kwargs = _parse_kwargs(args)
    campaign_id = kwargs.pop('id', None)
    if not campaign_id:
        raise ValueError("campaign requires :id")

    entity = engine.world.add('campaign', campaign_id, kwargs)
    return entity.to_dict()


def verb_quest(engine, *args):
    """Define a training quest.

    (quest :id q1 :dataset "train.jsonl" :steps 500 :priority :normal)
    """
    kwargs = _parse_kwargs(args)
    quest_id = kwargs.pop('id', None)
    if not quest_id:
        raise ValueError("quest requires :id")

    # Set defaults
    kwargs.setdefault('priority', 'normal')
    kwargs.setdefault('steps', 100)
    kwargs.setdefault('status', 'pending')

    entity = engine.world.add('quest', quest_id, kwargs)
    return entity.to_dict()


def verb_skill(engine, *args):
    """Define or reference a skill.

    (skill :id sy :name "Syllacrostic" :level 1 :category "reasoning")
    """
    kwargs = _parse_kwargs(args)
    skill_id = kwargs.pop('id', None)
    if not skill_id:
        raise ValueError("skill requires :id")

    entity = engine.world.add('skill', skill_id, kwargs)
    return entity.to_dict()


def verb_get_entity(engine, kind, entity_id):
    """Get an entity by kind and id.

    (get-entity hero dio)
    """
    kind = engine.eval(kind)
    entity_id = engine.eval(entity_id)

    entity = engine.world.get(kind, entity_id)
    if entity:
        return entity.to_dict()
    return None


def verb_update_entity(engine, kind, entity_id, *args):
    """Update an entity's properties.

    (update-entity hero dio :level 8 :xp 5000)
    """
    kind = engine.eval(kind)
    entity_id = engine.eval(entity_id)
    kwargs = _parse_kwargs(args)

    entity = engine.world.get(kind, entity_id)
    if not entity:
        raise ValueError(f"No {kind} with id '{entity_id}'")

    for key, value in kwargs.items():
        entity.set(key, value)

    return entity.to_dict()


def verb_delete_entity(engine, kind, entity_id):
    """Delete an entity.

    (delete-entity quest q1)
    """
    kind = engine.eval(kind)
    entity_id = engine.eval(entity_id)

    return engine.world.remove(kind, entity_id)


def verb_list_entities(engine, kind):
    """List all entities of a kind.

    (list-entities hero)
    """
    kind = engine.eval(kind)
    entities = engine.world.list(kind)
    return [e.to_dict() for e in entities]


def verb_query_entities(engine, kind, *args):
    """Query entities with filters.

    (query-entities quest :status pending :priority high)
    """
    kind = engine.eval(kind)
    filters = _parse_kwargs(args)

    entities = engine.world.query(kind, **filters)
    return [e.to_dict() for e in entities]
