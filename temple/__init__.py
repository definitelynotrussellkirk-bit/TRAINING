"""
Temple of Diagnostics - Dynamic integration testing for the Realm.
===================================================================

The Temple is where a Cleric runs diagnostic rituals to verify
that the Realm's services and systems are functioning correctly.

Unlike `doctor` which does static environment checks, Temple
performs live integration tests against running services.

NEW: The Temple now includes a comprehensive Training Diagnostics system:
- NaN Detective: Root cause analysis for NaN/Inf losses
- Gradient Health: Per-layer gradient flow tracking
- Memory Prophet: OOM prediction and leak detection
- LR Autopsy: Learning rate health analysis
- Data Sentinel: Batch quality checking
- Recovery Suggestions: Smart recovery from failures

Usage - Rituals:
    from temple import list_rituals, run_ritual

    # List available rituals
    rituals = list_rituals()
    # {'quick': {...}, 'vitals': {...}, ...}

    # Run a ritual
    result = run_ritual('quick')
    print(f"Status: {result.status}")  # ok, warn, fail, skip

    # Run the new Vitals ritual for training health
    result = run_ritual('vitals')

Usage - Training Diagnostics:
    from temple.diagnostics import TrainingDiagnostics

    diagnostics = TrainingDiagnostics()

    # In training loop
    report = diagnostics.on_step(
        step=step,
        loss=loss,
        model=model,
        batch=batch,
        lr=lr,
    )

    if report.has_critical:
        print(f"CRITICAL: {report.critical_issues}")
        print(report.to_rpg_report())

Usage - HuggingFace Trainer:
    from temple.hooks import TempleDiagnosticsCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[TempleDiagnosticsCallback()],
    )
"""

from .cleric import list_rituals, run_ritual, register_ritual
from .schemas import RitualResult, RitualCheckResult, ResultStatus, Blessing

__all__ = [
    # Rituals
    "list_rituals",
    "run_ritual",
    "register_ritual",
    "RitualResult",
    "RitualCheckResult",
    "ResultStatus",
    "Blessing",
]
