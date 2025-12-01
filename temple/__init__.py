"""
Temple of Diagnostics - Dynamic integration testing for the Realm.

The Temple is where a Cleric runs diagnostic rituals to verify
that the Realm's services and systems are functioning correctly.

Unlike `doctor` which does static environment checks, Temple
performs live integration tests against running services.

Usage:
    from temple import list_rituals, run_ritual

    # List available rituals
    rituals = list_rituals()
    # {'quick': {'id': 'quick', 'name': 'Ritual of Quick', ...}}

    # Run a ritual
    result = run_ritual('quick')
    print(f"Status: {result.status}")  # ok, warn, fail, skip
    for check in result.checks:
        print(f"  {check.status}: {check.name}")
"""

from .cleric import list_rituals, run_ritual, register_ritual
from .schemas import RitualResult, RitualCheckResult, ResultStatus

__all__ = [
    "list_rituals",
    "run_ritual",
    "register_ritual",
    "RitualResult",
    "RitualCheckResult",
    "ResultStatus",
]
