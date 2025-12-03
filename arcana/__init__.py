"""
Arcana - LISP-style DSL for the Realm of Training.

The Arcana layer provides:
    - S-expression parser for defining world state
    - Entity management (heroes, campaigns, quests, skills)
    - Action verbs for training control
    - Query verbs for reading metrics and status
    - LLM-emittable grammar for AI-driven planning

Usage:
    # CLI
    python -m arcana scripts/world.lisp
    python -m arcana --repl

    # Python API
    from arcana import create_engine

    engine = create_engine()
    engine.run('(hero :id dio :name "DIO" :level 7)')
    engine.run('(status)')
"""

from .parser import parse, parse_file, to_sexpr, pprint
from .world import World, Entity, get_world
from .engine import Engine, create_engine, EvalError
from .planner import Planner, Plan, serialize_world_compact
from .meta import MetaContext, get_meta_context, serialize_world_meta

__all__ = [
    'parse', 'parse_file', 'to_sexpr', 'pprint',
    'World', 'Entity', 'get_world',
    'Engine', 'create_engine', 'EvalError',
    'Planner', 'Plan', 'serialize_world_compact',
    'MetaContext', 'get_meta_context', 'serialize_world_meta',
]

__version__ = '0.1.0'
