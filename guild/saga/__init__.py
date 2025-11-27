"""
Saga - The narrative log of the realm.

The Saga is a persistent, append-only log of tales (events) that happened
in the realm. Unlike the Herald (which broadcasts ephemeral events), the
Saga persists to disk and can be displayed in the Tavern UI.

Storage:
    Tales are stored in JSONL files, one per day:
        logs/saga/2025-11-27.jsonl
        logs/saga/2025-11-26.jsonl

Quick Start:
    from guild.saga import SagaWriter, SagaReader

    # Write tales
    saga = SagaWriter(base_dir)
    saga.tell("quest.started", "DIO begins quest: binary_L5.jsonl")
    saga.level_up("DIO", 42)

    # Read tales
    reader = SagaReader(base_dir)
    recent = reader.recent(limit=50)
    for tale in recent:
        print(tale.format_display())

Connect to Herald:
    from guild.saga.bridge import connect_herald_to_saga

    # Wire up Herald -> Saga (all events auto-recorded)
    saga, bridge = connect_herald_to_saga(base_dir)

    # Now any Herald.emit() will be recorded in the Saga
    from guild.herald import emit, EventType
    emit(EventType.LEVEL_UP, {"hero": "DIO", "level": 42})

Module Initialization:
    from guild.saga import init_saga, init_reader, tell, recent

    init_saga(base_dir)
    init_reader(base_dir)

    tell("quest.started", "DIO begins quest")
    tales = recent(limit=50)
"""

# Types
from guild.saga.types import (
    TaleCategory,
    TaleEntry,
    TALE_ICONS,
    get_icon,
    get_category,
)

# Writer
from guild.saga.writer import (
    SagaWriter,
    init_saga,
    get_saga,
    tell,
)

# Reader
from guild.saga.reader import (
    SagaReader,
    init_reader,
    get_reader,
    recent,
)

# Bridge (Herald -> Saga)
from guild.saga.bridge import (
    HeraldBridge,
    MESSAGE_TEMPLATES,
    init_bridge,
    get_bridge,
    connect_herald_to_saga,
)

__all__ = [
    # Types
    "TaleCategory",
    "TaleEntry",
    "TALE_ICONS",
    "get_icon",
    "get_category",
    # Writer
    "SagaWriter",
    "init_saga",
    "get_saga",
    "tell",
    # Reader
    "SagaReader",
    "init_reader",
    "get_reader",
    "recent",
    # Bridge
    "HeraldBridge",
    "MESSAGE_TEMPLATES",
    "init_bridge",
    "get_bridge",
    "connect_herald_to_saga",
]
