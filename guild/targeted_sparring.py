#!/usr/bin/env python3
"""
Targeted Sparring - Focus on weak primitives using Skill Engine data.

Instead of randomly generating problems, this system:
1. Queries the Skill Engine for primitive accuracy
2. Identifies weak primitives (below threshold)
3. Generates problems specifically targeting those primitives
4. Produces focused training data for the weakest areas

Usage:
    python3 guild/targeted_sparring.py --skill bin --threshold 0.8 --count 100
    python3 guild/targeted_sparring.py --skill sy --count 50 --focus binary_add_with_carry
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WeakPrimitive:
    """A primitive that needs more practice."""
    name: str
    accuracy: float
    skill_id: str
    priority: float  # Lower accuracy = higher priority


@dataclass
class TargetedProblem:
    """A problem targeting a specific primitive."""
    prompt: str
    expected: str
    primitive_id: str
    skill_id: str


@dataclass
class TargetedSession:
    """Results from a targeted sparring session."""
    skill_id: str
    timestamp: str
    target_primitives: List[str]
    problems_generated: int = 0
    problems_by_primitive: Dict[str, int] = field(default_factory=dict)


def get_weak_primitives(
    skill_id: str = None,
    threshold: float = 0.8,
) -> List[WeakPrimitive]:
    """
    Query the Skill Engine for weak primitives.

    Args:
        skill_id: Filter to specific skill (None = all skills)
        threshold: Consider primitives below this accuracy as weak

    Returns:
        List of WeakPrimitive sorted by accuracy (weakest first)
    """
    try:
        from guild.skills import get_engine
        engine = get_engine()
    except ImportError as e:
        logger.error(f"Skill Engine not available: {e}")
        return []

    weak = []

    # Get skill states
    if skill_id:
        states = {skill_id: engine.get_state(skill_id)}
    else:
        states = engine.all_states()

    for sid, state in states.items():
        primitive_acc = state.primitive_accuracy or {}

        for prim_name, acc in primitive_acc.items():
            if acc < threshold:
                weak.append(WeakPrimitive(
                    name=prim_name,
                    accuracy=acc,
                    skill_id=sid,
                    priority=1.0 - acc,  # Lower accuracy = higher priority
                ))

    # Sort by priority (weakest first)
    weak.sort(key=lambda p: p.priority, reverse=True)
    return weak


def generate_targeted_problems(
    skill_id: str,
    target_primitives: List[str] = None,
    count: int = 100,
    threshold: float = 0.8,
) -> Tuple[List[TargetedProblem], TargetedSession]:
    """
    Generate problems targeting weak primitives.

    Args:
        skill_id: Skill to generate problems for
        target_primitives: Specific primitives to target (None = auto-detect weak)
        count: Total problems to generate
        threshold: Weakness threshold (if auto-detecting)

    Returns:
        (problems, session)
    """
    try:
        from guild.skills import get_engine
        from guild.passives import get_passive
        engine = get_engine()
    except ImportError as e:
        logger.error(f"Required modules not available: {e}")
        return [], None

    # Get skill and config
    try:
        skill = engine.get(skill_id)
        config = skill.config
    except Exception as e:
        logger.error(f"Could not load skill {skill_id}: {e}")
        return [], None

    # Determine target primitives
    if target_primitives is None:
        weak = get_weak_primitives(skill_id, threshold)
        target_primitives = [p.name for p in weak]

        if not target_primitives:
            logger.info(f"No weak primitives found for {skill_id} (threshold={threshold})")
            # Fall back to all primitives
            target_primitives = [p.name for p in skill.primitives]

    logger.info(f"Targeting {len(target_primitives)} primitives: {target_primitives[:5]}...")

    # Get passive for problem generation
    passive_id = config.passive_id
    if not passive_id:
        logger.warning(f"Skill {skill_id} has no passive_id configured")
        return [], None

    passive = get_passive(passive_id)
    if not passive:
        logger.error(f"Passive {passive_id} not found")
        return [], None

    # Generate problems with focus on target primitives
    problems = []
    problems_by_primitive = {p: 0 for p in target_primitives}

    # Generate more problems than needed, then filter
    raw_problems = passive.generate_problems(count * 3)

    for prob in raw_problems:
        prim_id = prob.get("primitive_id")

        if prim_id in target_primitives:
            problems.append(TargetedProblem(
                prompt=prob["prompt"],
                expected=prob["expected"],
                primitive_id=prim_id,
                skill_id=skill_id,
            ))
            problems_by_primitive[prim_id] = problems_by_primitive.get(prim_id, 0) + 1

            if len(problems) >= count:
                break

    # If we still need more, include any remaining
    if len(problems) < count:
        for prob in raw_problems:
            if len(problems) >= count:
                break

            prim_id = prob.get("primitive_id")
            if prim_id not in [p.primitive_id for p in problems]:
                problems.append(TargetedProblem(
                    prompt=prob["prompt"],
                    expected=prob["expected"],
                    primitive_id=prim_id,
                    skill_id=skill_id,
                ))

    session = TargetedSession(
        skill_id=skill_id,
        timestamp=datetime.now().isoformat(),
        target_primitives=target_primitives,
        problems_generated=len(problems),
        problems_by_primitive=problems_by_primitive,
    )

    return problems, session


def report_weak_primitives(threshold: float = 0.8) -> Dict:
    """
    Generate a report of weak primitives across all skills.

    Returns dict suitable for JSON output.
    """
    weak = get_weak_primitives(threshold=threshold)

    report = {
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "total_weak": len(weak),
        "by_skill": {},
        "weakest": [],
    }

    # Group by skill
    for p in weak:
        if p.skill_id not in report["by_skill"]:
            report["by_skill"][p.skill_id] = []
        report["by_skill"][p.skill_id].append({
            "name": p.name,
            "accuracy": round(p.accuracy, 3),
            "gap": round(threshold - p.accuracy, 3),
        })

    # Top 10 weakest overall
    report["weakest"] = [
        {
            "skill": p.skill_id,
            "primitive": p.name,
            "accuracy": round(p.accuracy, 3),
        }
        for p in weak[:10]
    ]

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Targeted Sparring - Focus on weak primitives"
    )
    parser.add_argument("--skill", help="Skill to target (bin, sy)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Weakness threshold (default: 0.8)")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of problems to generate")
    parser.add_argument("--focus", nargs="+",
                        help="Specific primitives to target")
    parser.add_argument("--report", action="store_true",
                        help="Generate weak primitives report")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")

    args = parser.parse_args()

    if args.report:
        # Generate report
        report = report_weak_primitives(threshold=args.threshold)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"\n=== Weak Primitives Report (threshold={args.threshold}) ===")
            print(f"Total weak: {report['total_weak']}")

            for skill_id, prims in report["by_skill"].items():
                print(f"\n{skill_id.upper()}:")
                for p in prims:
                    bar = "█" * int(p["accuracy"] * 10) + "░" * (10 - int(p["accuracy"] * 10))
                    print(f"  {p['name']:<30} {bar} {p['accuracy']:.0%} (gap: {p['gap']:.0%})")

            if report["weakest"]:
                print(f"\nTop 10 Weakest:")
                for i, p in enumerate(report["weakest"], 1):
                    print(f"  {i}. {p['skill']}/{p['primitive']}: {p['accuracy']:.0%}")

        return

    if not args.skill:
        parser.error("--skill required unless using --report")

    # Generate targeted problems
    problems, session = generate_targeted_problems(
        skill_id=args.skill,
        target_primitives=args.focus,
        count=args.count,
        threshold=args.threshold,
    )

    if args.json:
        output = {
            "session": {
                "skill_id": session.skill_id,
                "timestamp": session.timestamp,
                "target_primitives": session.target_primitives,
                "problems_generated": session.problems_generated,
                "problems_by_primitive": session.problems_by_primitive,
            },
            "problems": [
                {
                    "prompt": p.prompt,
                    "expected": p.expected,
                    "primitive_id": p.primitive_id,
                }
                for p in problems
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n=== Targeted Session: {args.skill.upper()} ===")
        print(f"Target primitives: {len(session.target_primitives)}")
        print(f"Problems generated: {session.problems_generated}")
        print(f"\nDistribution:")
        for prim, count in sorted(session.problems_by_primitive.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {prim}: {count}")

        print(f"\nSample problems:")
        for p in problems[:3]:
            print(f"  [{p.primitive_id}] {p.prompt[:60]}...")
            print(f"    Expected: {p.expected}")


if __name__ == "__main__":
    main()
