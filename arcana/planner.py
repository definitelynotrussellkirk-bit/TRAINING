"""
Arcana Planner - LLM-driven planning for training orchestration.

The planner:
1. Serializes world state to compact S-expressions
2. Prompts an LLM with goals and available verbs
3. Parses and validates the emitted plan
4. Executes (or dry-runs) the plan

Usage:
    from arcana.planner import Planner

    planner = Planner()
    plan = planner.propose("improve accuracy on binary skill")
    print(plan.forms)  # The LISP forms
    plan.execute()     # Run them
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .engine import Engine, EvalError, create_engine
from .parser import parse, to_sexpr, ParseError
from .world import World


# --- Compact World Serialization ---

def serialize_world_compact(world: World) -> str:
    """Serialize world state to compact S-expressions for LLM context."""
    lines = ["; === CURRENT WORLD STATE ==="]

    # Active hero
    campaigns = world.query('campaign', active=True)
    if campaigns:
        campaign = campaigns[0]
        hero_id = campaign.get('hero') or campaign.get('hero_id')
        hero = world.get('hero', hero_id) if hero_id else None

        lines.append("")
        lines.append("; Active Campaign")
        lines.append(f"(campaign :id {campaign.id} :hero {hero_id} :step {campaign.get('current_step', 0)})")

        if hero:
            # Extract model info - handle both dict and string formats
            model_info = hero.get('model', 'unknown')
            if isinstance(model_info, dict):
                model_str = f"{model_info.get('family', 'unknown')}-{model_info.get('size_b', '?')}B"
            else:
                model_str = str(model_info)
            lines.append(f"(hero :id {hero.id} :name \"{hero.get('name', hero.id)}\" :model \"{model_str}\")")

    # Current metrics
    if world.metrics:
        lines.append("")
        lines.append("; Current Metrics")
        for name, value in world.metrics.items():
            if isinstance(value, float):
                lines.append(f"(metric :{name} {value:.4f})")
            else:
                lines.append(f"(metric :{name} {value})")

    # Available skills
    skills = world.list('skill')
    if skills:
        lines.append("")
        lines.append("; Available Skills")
        for skill in skills:
            # Only include real skills, not templates
            if skill.id in ('skill_id', 'template'):
                continue
            max_level = skill.get('max_level', 10)
            lines.append(f"(skill :id {skill.id} :max-level {max_level})")

    # Pending quests
    quests = world.query('quest', status='pending')
    if quests:
        lines.append("")
        lines.append("; Pending Quests")
        for quest in quests[:5]:  # Limit to 5
            lines.append(f"(quest :id {quest.id} :skill {quest.get('skill')} :level {quest.get('level', 1)} :steps {quest.get('steps', 100)})")

    return "\n".join(lines)


# --- Prompt Templates ---

SYSTEM_PROMPT = '''You are a training planner for an AI model training system.
You emit ONLY valid Arcana DSL code. No prose, no explanations, just S-expressions.

## Grammar

Forms are S-expressions: (verb :key value :key2 value2)
Keywords start with colon: :id, :steps, :level
Strings use quotes: "like this"
Numbers are bare: 100, 0.5

## Available Verbs

Training Control:
  (train :quest QUEST_ID :steps N)     ; Queue training on a quest
  (train-file :path "..." :steps N)    ; Train on specific file
  (pause)                              ; Pause training
  (resume)                             ; Resume training
  (checkpoint :name "...")             ; Save checkpoint

Curriculum Management:
  (level-up :skill SKILL_ID)           ; Advance skill to next level
  (level-down :skill SKILL_ID)         ; Regress skill to previous level
  (set-level :skill SKILL_ID :level N) ; Set skill to specific level
  (run-eval :skill SKILL_ID :samples N) ; Trigger evaluation
  (generate-data :skill SKILL_ID :level N :count N) ; Generate training data

Queries (read-only, use for conditions):
  (metric :name METRIC)                ; Get metric value (loss, accuracy, step, throughput)
  (skill-status :skill SKILL_ID)       ; Get detailed skill status
  (compare-skills)                     ; Compare all skills
  (suggest-action)                     ; Get AI suggestion
  (status)                             ; Get full status
  (queue-status)                       ; Check training queue

Control Flow:
  (if COND THEN ELSE)                  ; Conditional
  (when COND BODY...)                  ; One-armed conditional
  (do FORM1 FORM2...)                  ; Sequence
  (> A B), (< A B), (= A B)            ; Comparisons

## Understanding Skill Progress

The world state shows skill-progress forms with:
  :level       - Current training level
  :max-level   - Maximum level for this skill
  :mastered    - Highest level with 80%+ accuracy
  :accuracy    - Most recent accuracy measurement
  :trend       - :improving, :stable, :declining, or :unknown
  :evals       - Number of evals at current level

## Decision Guidelines

- :trend :declining + :accuracy < 0.5 → Consider (level-down)
- :trend :improving + :accuracy >= 0.8 + :evals >= 3 → Consider (level-up)
- :evals 0 → Run (run-eval) before making level decisions
- Queue depth low → Generate more data or add quests

## Rules

1. Emit 1-5 forms maximum
2. Only reference skills that exist in world state
3. Check skill progress before level changes
4. Run evals before making curriculum decisions
5. NO prose, NO comments in output - just the forms
'''

GOAL_PROMPTS = {
    'improve_accuracy': '''Goal: Improve model accuracy.
Current accuracy is shown in metrics. If below target, train more.
If accuracy is stagnant, try a different skill level or quest.''',

    'reduce_loss': '''Goal: Reduce training loss.
Current loss is shown in metrics. If high, continue training.
If loss is very low (<0.01), consider harder content.''',

    'explore_skill': '''Goal: Explore a specific skill.
Train on the skill's quests at current level.
If accuracy is high (>0.8), suggest leveling up.''',

    'maintain': '''Goal: Maintain current training.
Keep the queue fed. Monitor for anomalies.
If queue is empty, add pending quests.''',

    'custom': '''Goal: {goal}
Analyze the world state and propose appropriate actions.''',
}


@dataclass
class Plan:
    """A proposed plan from the LLM."""
    raw_output: str
    forms: List[Any]
    goal: str
    world_state: str
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if plan parsed successfully."""
        return len(self.forms) > 0 and len(self.errors) == 0

    def to_sexpr(self) -> str:
        """Convert forms back to S-expression string."""
        return "\n".join(to_sexpr(f) for f in self.forms)

    def execute(self, engine: Engine, dry_run: bool = False) -> List[Any]:
        """Execute the plan forms."""
        if dry_run:
            print(f"[DRY RUN] Would execute {len(self.forms)} forms:")
            for form in self.forms:
                print(f"  {to_sexpr(form)}")
            return []

        results = []
        for form in self.forms:
            try:
                result = engine.eval(form)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                self.errors.append(f"Error executing {to_sexpr(form)}: {e}")
                results.append(None)

        self.executed = True
        return results


class Planner:
    """
    LLM-driven planner for training orchestration.

    Usage:
        planner = Planner()
        plan = planner.propose("improve accuracy")
        if plan.is_valid():
            plan.execute(planner.engine)
    """

    def __init__(self, engine: Optional[Engine] = None, llm: Optional[Any] = None):
        self.engine = engine or create_engine()
        self.llm = llm  # LLM interface (set via set_llm or passed in)
        self.history: List[Plan] = []

    def set_llm(self, llm):
        """Set the LLM interface."""
        self.llm = llm

    def get_world_state(self, meta_aware: bool = True) -> str:
        """Get world state for LLM context.

        Args:
            meta_aware: If True, use rich meta-aware serialization with
                        skill progress, eval history, and trends.
                        If False, use compact serialization.
        """
        if meta_aware:
            from .meta import serialize_world_meta
            return serialize_world_meta(self.engine.world, self.engine.world.base_dir)
        else:
            # Refresh metrics first
            self.engine.world.load_training_status()
            return serialize_world_compact(self.engine.world)

    def build_prompt(self, goal: str) -> Tuple[str, str]:
        """Build system and user prompts for a goal."""
        # Get goal-specific prompt
        if goal in GOAL_PROMPTS:
            goal_prompt = GOAL_PROMPTS[goal]
        else:
            goal_prompt = GOAL_PROMPTS['custom'].format(goal=goal)

        world_state = self.get_world_state()

        user_prompt = f"""{world_state}

{goal_prompt}

Emit your plan (1-5 S-expression forms):"""

        return SYSTEM_PROMPT, user_prompt

    def parse_response(self, response: str, goal: str, world_state: str) -> Plan:
        """Parse LLM response into a Plan."""
        # Extract S-expressions from response
        # LLM might include backticks or other formatting
        clean = response.strip()

        # Remove markdown code blocks if present
        clean = re.sub(r'```\w*\n?', '', clean)
        clean = clean.strip()

        # Try to parse
        try:
            forms = parse(clean)
        except ParseError as e:
            return Plan(
                raw_output=response,
                forms=[],
                goal=goal,
                world_state=world_state,
                errors=[f"Parse error: {e}"]
            )

        # Validate forms
        errors = self._validate_forms(forms)

        return Plan(
            raw_output=response,
            forms=forms,
            goal=goal,
            world_state=world_state,
            errors=errors
        )

    def _validate_forms(self, forms: List[Any]) -> List[str]:
        """Validate that forms are safe to execute."""
        errors = []

        # Check form count
        if len(forms) > 10:
            errors.append(f"Too many forms ({len(forms)}), max is 10")

        # Check each form
        for form in forms:
            if not isinstance(form, list) or not form:
                continue

            verb = form[0]

            # Check for known verbs
            known_verbs = {
                'train', 'train-file', 'pause', 'resume', 'checkpoint',
                'metric', 'status', 'queue-status',
                'if', 'when', 'unless', 'do', 'let',
                '>', '<', '=', '>=', '<=', '!=',
                '+', '-', '*', '/',
                'and', 'or', 'not',
                'print', 'log',
                'hero', 'campaign', 'quest', 'skill',
                'get-entity', 'list-entities', 'query-entities',
            }

            if verb not in known_verbs:
                errors.append(f"Unknown verb: {verb}")

        return errors

    def propose(self, goal: str) -> Plan:
        """Propose a plan for a goal using the LLM."""
        if not self.llm:
            raise ValueError("No LLM configured. Use set_llm() first.")

        system_prompt, user_prompt = self.build_prompt(goal)
        world_state = self.get_world_state()

        # Call LLM
        response = self.llm.complete(system_prompt, user_prompt)

        # Parse response
        plan = self.parse_response(response, goal, world_state)
        self.history.append(plan)

        return plan

    def propose_mock(self, goal: str, mock_response: str) -> Plan:
        """Propose using a mock response (for testing without LLM)."""
        world_state = self.get_world_state()
        plan = self.parse_response(mock_response, goal, world_state)
        self.history.append(plan)
        return plan

    def interactive(self, goal: str) -> Plan:
        """Interactive mode - show prompt, get user input, parse."""
        system_prompt, user_prompt = self.build_prompt(goal)

        print("=" * 60)
        print("SYSTEM PROMPT:")
        print("=" * 60)
        print(system_prompt)
        print()
        print("=" * 60)
        print("USER PROMPT:")
        print("=" * 60)
        print(user_prompt)
        print()
        print("=" * 60)
        print("Enter your plan (Ctrl+D when done):")
        print("=" * 60)

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        response = "\n".join(lines)
        return self.parse_response(response, goal, self.get_world_state())


# --- CLI Extension ---

def add_planner_commands(engine: Engine):
    """Add planner-related verbs to the engine."""

    def verb_propose(eng, goal):
        """(propose "goal") - Show what an LLM prompt would look like."""
        goal = eng.eval(goal)
        planner = Planner(engine=eng)
        system_prompt, user_prompt = planner.build_prompt(goal)

        return {
            'system_prompt': system_prompt[:500] + '...',
            'user_prompt': user_prompt,
        }

    def verb_world_state(eng):
        """(world-state-compact) - Get compact world state for LLM."""
        return serialize_world_compact(eng.world)

    engine.register('propose', verb_propose)
    engine.register('world-state-compact', verb_world_state)


if __name__ == '__main__':
    # Demo: show what a planning prompt looks like
    from core.paths import get_base_dir

    engine = create_engine(get_base_dir())
    planner = Planner(engine=engine)

    print("=== World State ===")
    print(planner.get_world_state())
    print()

    print("=== Prompt for 'improve accuracy' ===")
    system, user = planner.build_prompt('improve_accuracy')
    print("SYSTEM:", system[:200], "...")
    print()
    print("USER:", user)
