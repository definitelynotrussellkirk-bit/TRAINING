"""
Ritual of the Guild - Skills and curriculum diagnostics.

This ritual checks the health of skill configurations:
- Skill YAML configs are valid
- Validation data exists for each skill
- Skill generators are functional
- Curriculum state is consistent
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("guild", "Ritual of the Guild", "Skills and curriculum diagnostics")
def run() -> List[RitualCheckResult]:
    """Execute all guild ritual checks."""
    results = []
    results.append(_check_skill_configs())
    results.append(_check_validation_data())
    results.append(_check_curriculum_state())
    results.append(_check_skill_generators())
    return results


def _check_skill_configs() -> RitualCheckResult:
    """Check that skill YAML configs are valid."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir
        import yaml

        skills_dir = get_base_dir() / "configs" / "skills"

        if not skills_dir.exists():
            return RitualCheckResult(
                id="skill_configs",
                name="Skill Configurations",
                description="Verify all skill YAML configs are valid",
                status="fail",
                category="skills",
                details={"error": "configs/skills directory not found"},
                remediation="Create skill configs in configs/skills/",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        yaml_files = list(skills_dir.glob("*.yaml")) + list(skills_dir.glob("*.yml"))

        if not yaml_files:
            return RitualCheckResult(
                id="skill_configs",
                name="Skill Configurations",
                description="Verify all skill YAML configs are valid",
                status="warn",
                category="skills",
                details={"error": "No skill YAML files found"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        valid = []
        invalid = []

        for yf in yaml_files:
            try:
                with open(yf) as f:
                    data = yaml.safe_load(f)
                skill_id = data.get("id", yf.stem)
                skill_name = data.get("name", skill_id)
                levels = data.get("levels", [])
                valid.append({
                    "file": yf.name,
                    "id": skill_id,
                    "name": skill_name,
                    "level_count": len(levels),
                })
            except Exception as e:
                invalid.append({
                    "file": yf.name,
                    "error": str(e),
                })

        status = "ok" if not invalid else ("warn" if valid else "fail")

        return RitualCheckResult(
            id="skill_configs",
            name="Skill Configurations",
            description="Verify all skill YAML configs are valid",
            status=status,
            category="skills",
            details={
                "valid_count": len(valid),
                "invalid_count": len(invalid),
                "skills": valid,
                "errors": invalid if invalid else None,
            },
            remediation="Fix invalid YAML syntax in skill configs" if invalid else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="skill_configs",
            name="Skill Configurations",
            description="Verify all skill YAML configs are valid",
            status="fail",
            category="skills",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_validation_data() -> RitualCheckResult:
    """Check that validation data exists for skills."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir

        val_dir = get_base_dir() / "data" / "validation"

        if not val_dir.exists():
            return RitualCheckResult(
                id="validation_data",
                name="Validation Data",
                description="Verify validation datasets exist for skills",
                status="warn",
                category="skills",
                details={"error": "data/validation directory not found"},
                remediation="Generate validation data: python3 scripts/generate_validation_files.py",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Check for skill subdirectories
        skill_dirs = {}
        for subdir in val_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*.json")) + list(subdir.glob("*.jsonl"))
                skill_dirs[subdir.name] = len(files)

        if not skill_dirs:
            return RitualCheckResult(
                id="validation_data",
                name="Validation Data",
                description="Verify validation datasets exist for skills",
                status="warn",
                category="skills",
                details={"error": "No skill validation directories found"},
                remediation="Generate validation data for each skill",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Check for skills with no data
        empty_skills = [s for s, count in skill_dirs.items() if count == 0]

        return RitualCheckResult(
            id="validation_data",
            name="Validation Data",
            description="Verify validation datasets exist for skills",
            status="ok" if not empty_skills else "warn",
            category="skills",
            details={
                "skill_data": skill_dirs,
                "empty_skills": empty_skills if empty_skills else None,
            },
            remediation="Generate validation data for: " + ", ".join(empty_skills) if empty_skills else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="validation_data",
            name="Validation Data",
            description="Verify validation datasets exist for skills",
            status="fail",
            category="skills",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_curriculum_state() -> RitualCheckResult:
    """Check curriculum state consistency."""
    start = datetime.utcnow()
    try:
        from core.paths import get_base_dir

        state_file = get_base_dir() / "data_manager" / "curriculum_state.json"

        if not state_file.exists():
            return RitualCheckResult(
                id="curriculum_state",
                name="Curriculum State",
                description="Verify curriculum state is consistent",
                status="warn",
                category="skills",
                details={"error": "curriculum_state.json not found"},
                remediation="State will be created on first training run",
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        with open(state_file) as f:
            state = json.load(f)

        skill_levels = state.get("skill_levels", {})
        current_skill = state.get("current_skill")

        return RitualCheckResult(
            id="curriculum_state",
            name="Curriculum State",
            description="Verify curriculum state is consistent",
            status="ok",
            category="skills",
            details={
                "current_skill": current_skill,
                "skill_levels": skill_levels,
                "state_file": str(state_file),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except json.JSONDecodeError as e:
        return RitualCheckResult(
            id="curriculum_state",
            name="Curriculum State",
            description="Verify curriculum state is consistent",
            status="fail",
            category="skills",
            details={"error": f"Invalid JSON: {e}"},
            remediation="Fix or delete curriculum_state.json",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="curriculum_state",
            name="Curriculum State",
            description="Verify curriculum state is consistent",
            status="fail",
            category="skills",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_skill_generators() -> RitualCheckResult:
    """Check that skill generators are functional."""
    start = datetime.utcnow()
    try:
        # Try to import the skill engine
        from guild.skills import get_engine

        engine = get_engine()
        skills = engine.list_skills()

        if not skills:
            return RitualCheckResult(
                id="skill_generators",
                name="Skill Generators",
                description="Verify skill generators are functional",
                status="warn",
                category="skills",
                details={"error": "No skills registered in engine"},
                started_at=start,
                finished_at=datetime.utcnow(),
            )

        # Test each skill's generator
        results = {}
        for skill_id in skills:
            try:
                batch = engine.generate_eval_batch(skill_id, level=1, count=1)
                results[skill_id] = {
                    "ok": True,
                    "sample_count": len(batch) if batch else 0,
                }
            except Exception as e:
                results[skill_id] = {
                    "ok": False,
                    "error": str(e)[:100],
                }

        failed = [s for s, r in results.items() if not r.get("ok")]

        return RitualCheckResult(
            id="skill_generators",
            name="Skill Generators",
            description="Verify skill generators are functional",
            status="ok" if not failed else "warn",
            category="skills",
            details={
                "skill_count": len(skills),
                "results": results,
                "failed": failed if failed else None,
            },
            remediation="Fix generators for: " + ", ".join(failed) if failed else None,
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except ImportError as e:
        return RitualCheckResult(
            id="skill_generators",
            name="Skill Generators",
            description="Verify skill generators are functional",
            status="warn",
            category="skills",
            details={"error": f"Could not import skill engine: {e}"},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="skill_generators",
            name="Skill Generators",
            description="Verify skill generators are functional",
            status="fail",
            category="skills",
            details={"error": str(e)},
            started_at=start,
            finished_at=datetime.utcnow(),
        )
