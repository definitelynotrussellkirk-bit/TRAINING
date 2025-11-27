"""
Gatekeepers - Guards who inspect quest scrolls before battle.

The Arena has two gatekeepers at the entrance:

1. ScrollInspector (SpecValidator)
   - Checks quest scroll FORMAT is correct
   - Denies unknown scroll types
   - "Is this a valid quest scroll?"

2. ContentWarden (DataValidator)
   - Checks quest scroll CONTENT is appropriate
   - Three inspection depths: QUICK, STANDARD, DEEP
   - "Is this quest suitable for the hero?"

RPG Flavor:
    Before a quest reaches the Arena floor, it must pass the Gatekeepers.
    The ScrollInspector examines the scroll's seal and format. The Content
    Warden reads the contents to ensure they won't harm the hero.

Usage:
    from arena.gatekeepers import ScrollInspector, ContentWarden

    # Quick format check
    inspector = ScrollInspector()
    if not inspector.validate(quest_data):
        raise ValueError("Invalid scroll format")

    # Deep content check
    warden = ContentWarden()
    report = warden.inspect(quest_data, depth="deep")
    if not report.passed:
        reject_quest(quest_data)
"""

# Re-export from core/validation with RPG aliases
from core.validation.spec import (
    SpecValidator as ScrollInspector,
    DatasetSpec as ScrollSpec,
    DATASET_SPECS as SCROLL_SPECS,
)

from core.validation.validator import (
    DataValidator as ContentWarden,
    ValidationLevel as InspectionDepth,
    ValidationResult as InspectionReport,
)


__all__ = [
    # Scroll Inspector (format validation)
    "ScrollInspector",
    "ScrollSpec",
    "SCROLL_SPECS",
    # Content Warden (content validation)
    "ContentWarden",
    "InspectionDepth",
    "InspectionReport",
]
