"""
Ritual of Deep Divination - Comprehensive system diagnostic.

This ritual runs ALL other rituals for a complete system check.
It aggregates results and provides an overall health assessment.
"""

from datetime import datetime
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("deep", "Ritual of Deep Divination", "Comprehensive system diagnostic (runs all rituals)")
def run() -> List[RitualCheckResult]:
    """Execute all rituals and aggregate results."""
    results = []

    # Import and run each ritual module
    rituals_to_run = [
        ("quick", "temple.rituals.quick"),
        ("api", "temple.rituals.api"),
        ("forge", "temple.rituals.forge"),
        ("weaver", "temple.rituals.weaver"),
        ("champion", "temple.rituals.champion"),
        ("oracle", "temple.rituals.oracle"),
        ("guild", "temple.rituals.guild"),
        ("scribe", "temple.rituals.scribe"),
    ]

    for ritual_id, module_path in rituals_to_run:
        start = datetime.utcnow()
        try:
            # Import and run the ritual
            module = __import__(module_path, fromlist=['run'])
            checks = module.run()

            # Add a summary check for this ritual
            fail_count = sum(1 for c in checks if c.status == "fail")
            warn_count = sum(1 for c in checks if c.status == "warn")
            ok_count = sum(1 for c in checks if c.status == "ok")

            if fail_count > 0:
                sub_status = "fail"
            elif warn_count > 0:
                sub_status = "warn"
            else:
                sub_status = "ok"

            # Add all individual checks
            results.extend(checks)

        except Exception as e:
            # If a ritual fails to load/run, add an error check
            results.append(RitualCheckResult(
                id=f"ritual_{ritual_id}_error",
                name=f"Ritual {ritual_id.title()} Error",
                description=f"Failed to execute {ritual_id} ritual",
                status="fail",
                category="ritual",
                details={"error": str(e)},
                started_at=start,
                finished_at=datetime.utcnow(),
            ))

    return results
