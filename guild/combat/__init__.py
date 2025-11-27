"""
Combat system - evaluation and XP calculation.

The combat module provides:
- CombatEvaluator: Evaluate model responses to determine CombatResult
- CombatCalculator: Calculate XP from combat results
- StanceManager: Manage protocol modes (thinking vs direct)

Combat Results:
- CRITICAL_HIT: Perfect answer (exact match)
- HIT: Correct answer (normalized match)
- GLANCING: Partial/close answer
- MISS: Wrong answer
- CRITICAL_MISS: Invalid/malformed response

Example:
    from guild.combat import (
        evaluate_combat,
        calculate_combat_xp,
        select_stance,
        CombatStance,
    )

    # Select stance for this combat
    stance = select_stance(index=0)

    # After model response...
    eval_result = evaluate_combat(quest, response)

    # Calculate XP
    xp = calculate_combat_xp(
        combat_result=eval_result.combat_result,
        difficulty=quest.difficulty.value,
    )
"""

# Types
from guild.combat.types import (
    CombatStance,
    CombatConfig,
    StanceConfig,
)

# Re-export CombatResult for convenience
from guild.quests.types import CombatResult

# Evaluator
from guild.combat.evaluator import (
    MatchQuality,
    EvaluationResult,
    normalize_answer,
    extract_answer,
    BaseEvaluator,
    ExactMatchEvaluator,
    MultipleChoiceEvaluator,
    NumericEvaluator,
    CustomEvaluator,
    EVALUATOR_REGISTRY,
    get_evaluator,
    register_evaluator,
    CombatEvaluator,
    init_combat_evaluator,
    get_combat_evaluator,
    reset_combat_evaluator,
    evaluate_combat,
)

# Calculator
from guild.combat.calculator import (
    XPBreakdown,
    CombatCalculator,
    CombatReporter,
    init_combat_calculator,
    get_combat_calculator,
    reset_combat_calculator,
    calculate_combat_xp,
)

# Stance
from guild.combat.stance import (
    StanceSelection,
    StanceManager,
    ResponseFormatter,
    init_stance_manager,
    get_stance_manager,
    reset_stance_manager,
    select_stance,
)

__all__ = [
    # Types
    "CombatStance",
    "CombatConfig",
    "StanceConfig",
    "CombatResult",
    # Evaluator
    "MatchQuality",
    "EvaluationResult",
    "normalize_answer",
    "extract_answer",
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "MultipleChoiceEvaluator",
    "NumericEvaluator",
    "CustomEvaluator",
    "EVALUATOR_REGISTRY",
    "get_evaluator",
    "register_evaluator",
    "CombatEvaluator",
    "init_combat_evaluator",
    "get_combat_evaluator",
    "reset_combat_evaluator",
    "evaluate_combat",
    # Calculator
    "XPBreakdown",
    "CombatCalculator",
    "CombatReporter",
    "init_combat_calculator",
    "get_combat_calculator",
    "reset_combat_calculator",
    "calculate_combat_xp",
    # Stance
    "StanceSelection",
    "StanceManager",
    "ResponseFormatter",
    "init_stance_manager",
    "get_stance_manager",
    "reset_stance_manager",
    "select_stance",
]
