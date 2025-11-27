"""
Skill verification - check that YAML configs match running APIs.

Usage:
    from guild.skills.verify import verify_skill, verify_all_skills

    # Verify single skill
    result = verify_skill("sy")
    if not result.ok:
        print(f"Issues: {result.issues}")

    # Verify all skills
    results = verify_all_skills()
    for r in results:
        print(f"{r.skill_id}: {'OK' if r.ok else 'ISSUES'}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

from guild.skills.loader import load_skill_config, discover_skills
from guild.skills.contract import SkillClient

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of a skill verification check."""
    OK = "ok"
    VERSION_MISMATCH = "version_mismatch"
    API_UNREACHABLE = "api_unreachable"
    NO_API_CONFIGURED = "no_api_configured"
    LEVEL_MISMATCH = "level_mismatch"
    CONFIG_ERROR = "config_error"


@dataclass
class VerificationIssue:
    """A single verification issue."""
    status: VerificationStatus
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class SkillVerification:
    """Result of verifying a skill config against its API."""
    skill_id: str
    ok: bool
    config_version: str
    api_version: Optional[str] = None
    api_reachable: bool = False
    issues: list[VerificationIssue] = field(default_factory=list)

    # Additional info from API
    api_max_level: Optional[int] = None
    config_max_level: Optional[int] = None

    def __str__(self) -> str:
        if self.ok:
            return f"✓ {self.skill_id} v{self.config_version}"
        issues_str = ", ".join(i.message for i in self.issues)
        return f"✗ {self.skill_id}: {issues_str}"


def verify_skill(skill_id: str, warn_on_mismatch: bool = True) -> SkillVerification:
    """
    Verify a skill's YAML config matches its running API.

    Checks:
    - API is reachable
    - Version matches (YAML version == API /info version)
    - Max level matches

    Args:
        skill_id: Skill ID to verify
        warn_on_mismatch: If True, log warnings for mismatches

    Returns:
        SkillVerification with results
    """
    issues: list[VerificationIssue] = []
    api_version = None
    api_max_level = None
    api_reachable = False

    # Load config
    try:
        config = load_skill_config(skill_id)
    except Exception as e:
        return SkillVerification(
            skill_id=skill_id,
            ok=False,
            config_version="unknown",
            issues=[VerificationIssue(
                status=VerificationStatus.CONFIG_ERROR,
                message=f"Failed to load config: {e}"
            )]
        )

    # Check if API is configured
    if not config.api_url:
        issues.append(VerificationIssue(
            status=VerificationStatus.NO_API_CONFIGURED,
            message="No API URL configured in YAML"
        ))
        return SkillVerification(
            skill_id=skill_id,
            ok=False,
            config_version=config.version,
            config_max_level=config.max_level,
            issues=issues,
        )

    # Try to reach API
    client = SkillClient(skill_id, config.api_url)

    if not client.health():
        issues.append(VerificationIssue(
            status=VerificationStatus.API_UNREACHABLE,
            message=f"API not reachable at {config.api_url}"
        ))
        return SkillVerification(
            skill_id=skill_id,
            ok=False,
            config_version=config.version,
            config_max_level=config.max_level,
            api_reachable=False,
            issues=issues,
        )

    api_reachable = True

    # Get API info
    try:
        info = client.info()
        api_version = info.version
        api_max_level = info.max_level
    except Exception as e:
        issues.append(VerificationIssue(
            status=VerificationStatus.API_UNREACHABLE,
            message=f"Failed to get API info: {e}"
        ))
        return SkillVerification(
            skill_id=skill_id,
            ok=False,
            config_version=config.version,
            config_max_level=config.max_level,
            api_reachable=True,
            issues=issues,
        )

    # Check version match
    if api_version and api_version != config.version:
        issue = VerificationIssue(
            status=VerificationStatus.VERSION_MISMATCH,
            message=f"Version mismatch: YAML={config.version}, API={api_version}",
            expected=config.version,
            actual=api_version,
        )
        issues.append(issue)
        if warn_on_mismatch:
            logger.warning(
                f"Skill '{skill_id}' version mismatch: "
                f"YAML config has {config.version}, API reports {api_version}. "
                f"Update configs/skills/{skill_id}.yaml or the API."
            )

    # Check max level match
    if api_max_level and api_max_level != config.max_level:
        issue = VerificationIssue(
            status=VerificationStatus.LEVEL_MISMATCH,
            message=f"Max level mismatch: YAML={config.max_level}, API={api_max_level}",
            expected=str(config.max_level),
            actual=str(api_max_level),
        )
        issues.append(issue)
        if warn_on_mismatch:
            logger.warning(
                f"Skill '{skill_id}' max_level mismatch: "
                f"YAML config has {config.max_level}, API reports {api_max_level}."
            )

    return SkillVerification(
        skill_id=skill_id,
        ok=len(issues) == 0,
        config_version=config.version,
        api_version=api_version,
        api_reachable=api_reachable,
        config_max_level=config.max_level,
        api_max_level=api_max_level,
        issues=issues,
    )


def verify_all_skills(
    warn_on_mismatch: bool = True,
    skip_unreachable: bool = False,
) -> list[SkillVerification]:
    """
    Verify all discovered skill configs against their APIs.

    Args:
        warn_on_mismatch: Log warnings for mismatches
        skip_unreachable: If True, don't fail on unreachable APIs

    Returns:
        List of SkillVerification results
    """
    results = []

    for skill_id in discover_skills():
        result = verify_skill(skill_id, warn_on_mismatch=warn_on_mismatch)

        # Optionally mark unreachable as OK
        if skip_unreachable and not result.api_reachable:
            result = SkillVerification(
                skill_id=result.skill_id,
                ok=True,
                config_version=result.config_version,
                api_reachable=False,
                issues=[VerificationIssue(
                    status=VerificationStatus.API_UNREACHABLE,
                    message="API not running (skipped)"
                )],
            )

        results.append(result)

    return results


def print_verification_report(results: list[SkillVerification]) -> None:
    """Print a formatted verification report."""
    print("\n" + "=" * 60)
    print("SKILL VERIFICATION REPORT")
    print("=" * 60)

    ok_count = sum(1 for r in results if r.ok)
    total = len(results)

    for result in results:
        if result.ok:
            status = "✓ OK"
            if not result.api_reachable:
                status = "○ SKIP (API not running)"
        else:
            status = "✗ ISSUES"

        print(f"\n{result.skill_id}:")
        print(f"  Status: {status}")
        print(f"  Config version: {result.config_version}")

        if result.api_reachable:
            print(f"  API version: {result.api_version or 'unknown'}")
            print(f"  API max_level: {result.api_max_level}")

        if result.issues:
            print(f"  Issues:")
            for issue in result.issues:
                print(f"    - {issue.message}")

    print("\n" + "-" * 60)
    print(f"Summary: {ok_count}/{total} skills verified OK")
    print("=" * 60 + "\n")


# CLI support
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        # Verify specific skill
        skill_id = sys.argv[1]
        result = verify_skill(skill_id)
        print(result)
        sys.exit(0 if result.ok else 1)
    else:
        # Verify all
        results = verify_all_skills(skip_unreachable=True)
        print_verification_report(results)

        # Exit with error if any issues (except unreachable)
        real_issues = [r for r in results if not r.ok and r.api_reachable]
        sys.exit(1 if real_issues else 0)
