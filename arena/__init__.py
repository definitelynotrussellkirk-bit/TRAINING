"""
Arena - The Training Grounds where the hero battles quest challenges.

The Arena is the central location where training happens:

    Quest Board     - Where quests wait for the hero
    Battle Log      - Records combat progress
    Battle Control  - Commands the hero during battle
    Gatekeepers     - Inspect quest scrolls before entry

RPG Mapping:
    Training           → Battle/Combat
    Training Step      → Combat Round
    Epoch              → Campaign
    Loss               → Damage Taken
    Accuracy           → Hit Rate
    Checkpoint         → Hero Snapshot
    Queue              → Quest Board
    Validation         → Gatekeepers

Quick Start:
    from arena import QuestBoard, BattleControl, BattleLog

    # Set up the arena
    board = QuestBoard(base_dir)
    control = BattleControl(base_dir)
    log = BattleLog(base_dir)

    # Check quests available
    status = board.get_board_status()
    print(f"Quests waiting: {status['total_pending']}")

    # Take a quest
    quest = board.take_next_quest()

    # Start battle
    log.begin_battle(quest.name, total_rounds=1000)

    # During battle, check for commands
    if control.should_rally():
        log.pause_battle()
        control.wait_for_charge()
        log.resume_battle()

    # End battle
    log.end_battle(victory=True)

This module wraps core/ with RPG-themed naming while maintaining
backward compatibility.
"""

__version__ = "0.1.0"

# Types
from arena.types import (
    BattleState,
    CombatResult,
    BattleStatus,
    QuestBoardStatus,
    HeroSnapshot,
)

# Quest Board (training queue)
from arena.quest_board import (
    QuestBoard,
    get_quest_board,
    # Backward compat
    TrainingQueue,
)

# Battle Log (training status)
from arena.battle_log import (
    BattleLog,
    get_battle_log,
)

# Battle Control (training controller)
from arena.battle_control import (
    BattleControl,
    get_battle_control,
    # Backward compat
    TrainingController,
)

# Gatekeepers (validation)
from arena.gatekeepers import (
    ScrollInspector,
    ScrollSpec,
    SCROLL_SPECS,
    ContentWarden,
    InspectionDepth,
    InspectionReport,
)


__all__ = [
    # Types
    "BattleState",
    "CombatResult",
    "BattleStatus",
    "QuestBoardStatus",
    "HeroSnapshot",
    # Quest Board
    "QuestBoard",
    "get_quest_board",
    "TrainingQueue",  # Backward compat
    # Battle Log
    "BattleLog",
    "get_battle_log",
    # Battle Control
    "BattleControl",
    "get_battle_control",
    "TrainingController",  # Backward compat
    # Gatekeepers
    "ScrollInspector",
    "ScrollSpec",
    "SCROLL_SPECS",
    "ContentWarden",
    "InspectionDepth",
    "InspectionReport",
]


# =============================================================================
# RPG TERMINOLOGY GUIDE
# =============================================================================

"""
ARENA GLOSSARY
==============

The Arena uses RPG terminology to make training concepts more intuitive:

BATTLE TERMS
------------
Battle          = Training run (processing one JSONL file)
Combat Round    = Training step
Campaign        = Epoch (full pass through data)
Damage Taken    = Loss (lower is better - hero is hurt less)
Hit Rate        = Accuracy
Hero Snapshot   = Checkpoint
Victory         = Training completed successfully
Retreat         = Training stopped early (graceful)
Defeat          = Training crashed

QUEST TERMS
-----------
Quest           = Training data file (JSONL)
Quest Scroll    = Individual training example
Quest Board     = Training queue
Post Quest      = Add file to queue
Take Quest      = Get next file for training
Abandon Quest   = Skip file, move to failed

COMMAND TERMS
-------------
Rally           = Pause (regroup after current round)
Charge          = Resume (continue battle)
Retreat         = Stop (end battle gracefully)
Abandon         = Skip (give up on current quest)

INSPECTION TERMS
----------------
Gatekeeper      = Validator
Scroll Inspector = Schema validator (format check)
Content Warden  = Data validator (content check)
Inspection      = Validation
QUICK/STANDARD/DEEP = Validation levels

STATE TERMS
-----------
idle            = Not fighting, waiting at Quest Board
fighting        = Training in progress
rallied         = Paused, waiting for orders
retreating      = Finishing current round, will exit
abandoning      = Skipping current quest
victory         = Training completed
defeat          = Training crashed
"""
