"""
Guild Trainer - A generic framework for LLM training with RPG-style progression.

The Guild framework provides:
- Skills: Trainable capabilities with metrics and progression
- Quests: Task instances with templates and evaluation
- Dispatch: Quest coordination from Trainers to Quest Board
- Progression: XP, levels, and status effects
- Facilities: Hardware abstraction for multi-machine setups
- Runs: Unified training/eval/audit execution
- Incidents: Structured error tracking
- Combat: Result calculation (CRIT/HIT/MISS)
- Consistency: World model validation
- Herald: Event bus for broadcasting events across the realm
- Saga: Persistent narrative log (scrolling chat log for Tavern UI)

Quick Start:
    import guild

    # Initialize systems
    guild.init_resolver("/path/to/configs/facilities")
    guild.init_skill_registry("/path/to/configs")
    guild.init_quest_registry("/path/to/configs")
    guild.init_run_state_manager("/path/to/status")

    # Create and execute a run
    config = guild.RunConfig(id="train", type=guild.RunType.TRAINING)
    run = guild.create_run(config)
    guild.start_run(run.run_id)

Event System:
    # Subscribe to events
    guild.subscribe(guild.EventType.LEVEL_UP, my_callback)

    # Emit events (auto-recorded to Saga)
    guild.emit(guild.EventType.LEVEL_UP, {"hero": "DIO", "level": 42})

    # Read narrative log
    saga, bridge = guild.connect_herald_to_saga("/path/to/training")
    tales = guild.recent(limit=50)
"""

__version__ = "0.1.0"

# Facility resolution (core functionality)
from guild.facilities.resolver import (
    init_resolver,
    resolve,
    get_facility,
    set_current_facility,
    get_resolver,
    reset_resolver,
)

# Skill management
from guild.skills import (
    # Types
    SkillCategory,
    SkillConfig,
    SkillState,
    # Registry
    init_registry as init_skill_registry,
    get_registry as get_skill_registry,
    reset_registry as reset_skill_registry,
    get_skill,
    list_skills,
    skills_by_category,
    skills_by_tag,
    # State management
    init_state_manager as init_skill_state_manager,
    get_state_manager as get_skill_state_manager,
    reset_state_manager as reset_skill_state_manager,
)

# Quest management
from guild.quests import (
    # Types
    QuestDifficulty,
    CombatResult,
    QuestTemplate,
    QuestInstance,
    QuestResult,
    # Registry
    init_quest_registry,
    get_quest_registry,
    reset_quest_registry,
    get_quest,
    list_quests,
    quests_by_skill,
    # Forge
    get_forge as get_quest_forge,
    reset_forge as reset_quest_forge,
    create_quest,
    create_quest_by_id,
    # Evaluator
    get_judge as get_quest_judge,
    reset_judge as reset_quest_judge,
    evaluate_quest,
)

# Progression system
from guild.progression import (
    # Types
    EffectType,
    StatusEffect,
    HeroIdentity,
    HeroState,
    # XP
    LevelConfig,
    XPCalculator,
    get_xp_calculator,
    reset_xp_calculator,
    calculate_xp,
    xp_for_level,
    level_from_xp,
    # Effects
    get_effect_manager,
    reset_effect_manager,
    apply_effect,
    update_effects,
    get_xp_multiplier,
    # Hero
    HeroManager,
    init_hero_manager,
    get_hero_manager,
    reset_hero_manager,
    get_hero,
    record_result,
    get_hero_status,
)

# Run management
from guild.runs import (
    # Types
    RunType,
    RunConfig,
    RunState,
    # Registry
    init_run_registry,
    get_run_registry,
    reset_run_registry,
    get_run_config,
    list_run_configs,
    run_configs_by_type,
    # State management
    RunStateManager,
    init_run_state_manager,
    get_run_state_manager,
    reset_run_state_manager,
    create_run,
    get_run,
    start_run,
    pause_run,
    resume_run,
    complete_run,
    get_current_run,
    # Executor
    RunCallback,
    RunCallbackAdapter,
    StepResult,
    RunHandler,
    RunExecutor,
    init_run_executor,
    get_run_executor,
    reset_run_executor,
)

# Incident management
from guild.incidents import (
    # Types
    IncidentCategory,
    IncidentStatus,
    Incident,
    IncidentRule,
    # Registry
    init_incident_rule_registry,
    get_incident_rule_registry,
    reset_incident_rule_registry,
    get_incident_rule,
    list_incident_rules,
    incident_rules_by_category,
    # Tracker
    IncidentTracker,
    init_incident_tracker,
    get_incident_tracker,
    reset_incident_tracker,
    create_incident,
    get_incident,
    resolve_incident,
    list_open_incidents,
    get_incident_stats,
    # Detector
    DetectionContext,
    IncidentDetector,
    init_incident_detector,
    get_incident_detector,
    reset_incident_detector,
)

# Combat system
from guild.combat import (
    # Types
    CombatStance,
    CombatConfig,
    StanceConfig,
    # Evaluator
    MatchQuality,
    EvaluationResult,
    CombatEvaluator,
    init_combat_evaluator,
    get_combat_evaluator,
    reset_combat_evaluator,
    evaluate_combat,
    # Calculator
    XPBreakdown,
    CombatCalculator,
    init_combat_calculator,
    get_combat_calculator,
    reset_combat_calculator,
    calculate_combat_xp,
    # Stance
    StanceSelection,
    StanceManager,
    init_stance_manager,
    get_stance_manager,
    reset_stance_manager,
    select_stance,
)

# Consistency checking
from guild.consistency import (
    # Rule types
    RuleCategory,
    RuleSeverity,
    RuleViolation,
    ConsistencyRule,
    RuleBuilder,
    # Checker
    CheckResult,
    ConsistencyChecker,
    init_consistency_checker,
    get_consistency_checker,
    reset_consistency_checker,
    check_entity,
    check_all,
    # Built-in rules
    get_hero_rules,
    get_quest_rules,
    get_run_rules,
    get_incident_rules,
    get_skill_rules,
    get_all_rules,
    get_rules_by_entity,
)

# Dispatch system (quest coordination)
from guild.dispatch import (
    # Types
    DispatchDecision,
    QuestVerdict,
    DispatchStatus,
    DispatchResult,
    QualityReport,
    ProgressionStatus,
    # Quality Gate
    QuestQualityGate,
    # Progression Advisor
    ProgressionAdvisor,
    SKILL_CURRICULA,
    # Quest Dispatcher
    QuestDispatcher,
)

# Herald (event bus)
from guild.herald import (
    # Types
    Event,
    EventType,
    # Herald class
    Herald,
    # Convenience functions
    get_herald,
    emit,
    subscribe,
    unsubscribe,
)

# Saga (narrative log)
from guild.saga import (
    # Types
    TaleCategory,
    TaleEntry,
    TALE_ICONS,
    # Writer
    SagaWriter,
    init_saga,
    get_saga,
    tell,
    # Reader
    SagaReader,
    init_reader,
    get_reader,
    recent,
    # Bridge
    HeraldBridge,
    connect_herald_to_saga,
)

__all__ = [
    # Facilities
    "init_resolver",
    "resolve",
    "get_facility",
    "set_current_facility",
    "get_resolver",
    "reset_resolver",
    # Skills - types
    "SkillCategory",
    "SkillConfig",
    "SkillState",
    # Skills - registry
    "init_skill_registry",
    "get_skill_registry",
    "reset_skill_registry",
    "get_skill",
    "list_skills",
    "skills_by_category",
    "skills_by_tag",
    # Skills - state
    "init_skill_state_manager",
    "get_skill_state_manager",
    "reset_skill_state_manager",
    # Quests - types
    "QuestDifficulty",
    "CombatResult",
    "QuestTemplate",
    "QuestInstance",
    "QuestResult",
    # Quests - registry
    "init_quest_registry",
    "get_quest_registry",
    "reset_quest_registry",
    "get_quest",
    "list_quests",
    "quests_by_skill",
    # Quests - forge
    "get_quest_forge",
    "reset_quest_forge",
    "create_quest",
    "create_quest_by_id",
    # Quests - evaluator
    "get_quest_judge",
    "reset_quest_judge",
    "evaluate_quest",
    # Progression - types
    "EffectType",
    "StatusEffect",
    "HeroIdentity",
    "HeroState",
    # Progression - XP
    "LevelConfig",
    "XPCalculator",
    "get_xp_calculator",
    "reset_xp_calculator",
    "calculate_xp",
    "xp_for_level",
    "level_from_xp",
    # Progression - effects
    "get_effect_manager",
    "reset_effect_manager",
    "apply_effect",
    "update_effects",
    "get_xp_multiplier",
    # Progression - hero
    "HeroManager",
    "init_hero_manager",
    "get_hero_manager",
    "reset_hero_manager",
    "get_hero",
    "record_result",
    "get_hero_status",
    # Runs - types
    "RunType",
    "RunConfig",
    "RunState",
    # Runs - registry
    "init_run_registry",
    "get_run_registry",
    "reset_run_registry",
    "get_run_config",
    "list_run_configs",
    "run_configs_by_type",
    # Runs - state management
    "RunStateManager",
    "init_run_state_manager",
    "get_run_state_manager",
    "reset_run_state_manager",
    "create_run",
    "get_run",
    "start_run",
    "pause_run",
    "resume_run",
    "complete_run",
    "get_current_run",
    # Runs - executor
    "RunCallback",
    "RunCallbackAdapter",
    "StepResult",
    "RunHandler",
    "RunExecutor",
    "init_run_executor",
    "get_run_executor",
    "reset_run_executor",
    # Incidents - types
    "IncidentCategory",
    "IncidentStatus",
    "Incident",
    "IncidentRule",
    # Incidents - registry
    "init_incident_rule_registry",
    "get_incident_rule_registry",
    "reset_incident_rule_registry",
    "get_incident_rule",
    "list_incident_rules",
    "incident_rules_by_category",
    # Incidents - tracker
    "IncidentTracker",
    "init_incident_tracker",
    "get_incident_tracker",
    "reset_incident_tracker",
    "create_incident",
    "get_incident",
    "resolve_incident",
    "list_open_incidents",
    "get_incident_stats",
    # Incidents - detector
    "DetectionContext",
    "IncidentDetector",
    "init_incident_detector",
    "get_incident_detector",
    "reset_incident_detector",
    # Combat - types
    "CombatStance",
    "CombatConfig",
    "StanceConfig",
    # Combat - evaluator
    "MatchQuality",
    "EvaluationResult",
    "CombatEvaluator",
    "init_combat_evaluator",
    "get_combat_evaluator",
    "reset_combat_evaluator",
    "evaluate_combat",
    # Combat - calculator
    "XPBreakdown",
    "CombatCalculator",
    "init_combat_calculator",
    "get_combat_calculator",
    "reset_combat_calculator",
    "calculate_combat_xp",
    # Combat - stance
    "StanceSelection",
    "StanceManager",
    "init_stance_manager",
    "get_stance_manager",
    "reset_stance_manager",
    "select_stance",
    # Consistency - rule types
    "RuleCategory",
    "RuleSeverity",
    "RuleViolation",
    "ConsistencyRule",
    "RuleBuilder",
    # Consistency - checker
    "CheckResult",
    "ConsistencyChecker",
    "init_consistency_checker",
    "get_consistency_checker",
    "reset_consistency_checker",
    "check_entity",
    "check_all",
    # Consistency - built-in rules
    "get_hero_rules",
    "get_quest_rules",
    "get_run_rules",
    "get_incident_rules",
    "get_skill_rules",
    "get_all_rules",
    "get_rules_by_entity",
    # Dispatch - types
    "DispatchDecision",
    "QuestVerdict",
    "DispatchStatus",
    "DispatchResult",
    "QualityReport",
    "ProgressionStatus",
    # Dispatch - components
    "QuestQualityGate",
    "ProgressionAdvisor",
    "SKILL_CURRICULA",
    "QuestDispatcher",
    # Herald - types
    "Event",
    "EventType",
    # Herald - class and functions
    "Herald",
    "get_herald",
    "emit",
    "subscribe",
    "unsubscribe",
    # Saga - types
    "TaleCategory",
    "TaleEntry",
    "TALE_ICONS",
    # Saga - writer
    "SagaWriter",
    "init_saga",
    "get_saga",
    "tell",
    # Saga - reader
    "SagaReader",
    "init_reader",
    "get_reader",
    "recent",
    # Saga - bridge
    "HeraldBridge",
    "connect_herald_to_saga",
]
