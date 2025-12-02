"""
Test that all passives adhere to the PassiveModule contract.

This test ensures all passive implementations:
1. Accept the required generate_problems(count, seed, level) signature
2. Return problems with required fields (prompt, expected, primitive_id)
3. Have required class attributes (id, name, category, description, version)
4. Have check_answer method that works correctly

Run with: pytest tests/test_passive_contract.py -v
Or standalone: python3 tests/test_passive_contract.py
"""

import sys
from pathlib import Path

# Add project root to path for standalone execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from typing import Dict, List, Any

from guild.passives import list_passives, get_passive, get_all_passives
from guild.passives.base import PassiveModule


class TestPassiveContract:
    """Test all passives comply with the contract."""

    @pytest.fixture
    def all_passive_ids(self) -> List[str]:
        """Get all registered passive IDs."""
        return list_passives()

    def test_all_passives_discovered(self, all_passive_ids):
        """Verify passives are discovered."""
        assert len(all_passive_ids) > 0, "No passives discovered"
        print(f"\nFound {len(all_passive_ids)} passives: {all_passive_ids}")

    @pytest.mark.parametrize("passive_id", list_passives())
    def test_required_class_attributes(self, passive_id: str):
        """Each passive must have required class attributes."""
        passive = get_passive(passive_id)
        assert passive is not None, f"Could not get passive {passive_id}"

        # Required attributes
        assert hasattr(passive, 'id') and passive.id is not None, \
            f"{passive_id}: missing 'id' attribute"
        assert hasattr(passive, 'name') and passive.name is not None, \
            f"{passive_id}: missing 'name' attribute"
        assert hasattr(passive, 'category') and passive.category is not None, \
            f"{passive_id}: missing 'category' attribute"
        assert hasattr(passive, 'description') and passive.description is not None, \
            f"{passive_id}: missing 'description' attribute"
        assert hasattr(passive, 'version') and passive.version is not None, \
            f"{passive_id}: missing 'version' attribute"

        # Version format
        parts = passive.version.split('.')
        assert len(parts) == 3, \
            f"{passive_id}: version must be semantic (X.Y.Z), got {passive.version}"

    @pytest.mark.parametrize("passive_id", list_passives())
    def test_generate_problems_signature(self, passive_id: str):
        """generate_problems must accept count, seed, and level parameters."""
        passive = get_passive(passive_id)
        assert passive is not None

        # Must accept all three parameters without error
        try:
            problems = passive.generate_problems(count=3, seed=42, level=5)
        except TypeError as e:
            pytest.fail(
                f"{passive_id}: generate_problems must accept (count, seed, level). "
                f"Got TypeError: {e}"
            )

        assert isinstance(problems, list), \
            f"{passive_id}: generate_problems must return a list"
        assert len(problems) == 3, \
            f"{passive_id}: generate_problems returned {len(problems)} problems, expected 3"

    @pytest.mark.parametrize("passive_id", list_passives())
    def test_problem_format(self, passive_id: str):
        """Each problem must have prompt, expected, and primitive_id."""
        passive = get_passive(passive_id)
        problems = passive.generate_problems(count=5, seed=42, level=1)

        for i, problem in enumerate(problems):
            assert isinstance(problem, dict), \
                f"{passive_id}[{i}]: problem must be a dict"

            # Required fields
            assert 'prompt' in problem, \
                f"{passive_id}[{i}]: missing 'prompt' field. Keys: {list(problem.keys())}"
            assert 'expected' in problem, \
                f"{passive_id}[{i}]: missing 'expected' field. Keys: {list(problem.keys())}"
            assert 'primitive_id' in problem, \
                f"{passive_id}[{i}]: missing 'primitive_id' field. Keys: {list(problem.keys())}"

            # Type checks
            assert isinstance(problem['prompt'], str), \
                f"{passive_id}[{i}]: 'prompt' must be str"
            assert isinstance(problem['expected'], str), \
                f"{passive_id}[{i}]: 'expected' must be str"
            assert isinstance(problem['primitive_id'], str), \
                f"{passive_id}[{i}]: 'primitive_id' must be str"

    @pytest.mark.parametrize("passive_id", list_passives())
    def test_check_answer_method(self, passive_id: str):
        """check_answer must exist and return bool."""
        passive = get_passive(passive_id)

        # Generate a problem to get expected answer
        problems = passive.generate_problems(count=1, seed=42, level=1)
        expected = problems[0]['expected']

        # Test with correct answer
        result = passive.check_answer(expected, expected)
        assert isinstance(result, bool), \
            f"{passive_id}: check_answer must return bool"
        assert result is True, \
            f"{passive_id}: check_answer should return True for exact match"

        # Test with clearly wrong answer
        result = passive.check_answer(expected, "CLEARLY_WRONG_ANSWER_XYZ123")
        assert isinstance(result, bool), \
            f"{passive_id}: check_answer must return bool"

    @pytest.mark.parametrize("passive_id", list_passives())
    def test_reproducibility_with_seed(self, passive_id: str):
        """Same seed should produce same problems."""
        passive = get_passive(passive_id)

        problems1 = passive.generate_problems(count=5, seed=12345, level=1)
        problems2 = passive.generate_problems(count=5, seed=12345, level=1)

        for i, (p1, p2) in enumerate(zip(problems1, problems2)):
            assert p1['prompt'] == p2['prompt'], \
                f"{passive_id}[{i}]: seed should produce reproducible prompts"
            assert p1['expected'] == p2['expected'], \
                f"{passive_id}[{i}]: seed should produce reproducible expected answers"

    @pytest.mark.parametrize("passive_id", list_passives())
    def test_different_seeds_different_problems(self, passive_id: str):
        """Different seeds should (usually) produce different problems."""
        passive = get_passive(passive_id)

        problems1 = passive.generate_problems(count=5, seed=111, level=1)
        problems2 = passive.generate_problems(count=5, seed=999, level=1)

        # At least some problems should differ
        prompts1 = [p['prompt'] for p in problems1]
        prompts2 = [p['prompt'] for p in problems2]

        # Allow some overlap but not complete identity
        overlap = sum(1 for p in prompts1 if p in prompts2)
        assert overlap < len(prompts1), \
            f"{passive_id}: different seeds produced identical problems"

    def test_get_config_method(self):
        """All passives should have get_config() method."""
        for passive in get_all_passives():
            config = passive.get_config()
            assert config.id == passive.id
            assert config.name == passive.name
            assert config.version == passive.version


class TestPassiveInheritance:
    """Test inheritance from PassiveModule."""

    def test_all_passives_inherit_from_base(self):
        """All passives must inherit from PassiveModule."""
        for passive in get_all_passives():
            assert isinstance(passive, PassiveModule), \
                f"{passive.id} must inherit from PassiveModule"


if __name__ == "__main__":
    # Run quick validation without pytest
    print("Passive Contract Validation")
    print("=" * 60)

    passives = list_passives()
    print(f"Found {len(passives)} passives\n")

    errors = []

    for pid in passives:
        p = get_passive(pid)
        issues = []

        # Check attributes
        for attr in ['id', 'name', 'category', 'description', 'version']:
            if not hasattr(p, attr) or getattr(p, attr) is None:
                issues.append(f"missing {attr}")

        # Check generate_problems signature
        try:
            probs = p.generate_problems(count=2, seed=42, level=5)
            for prob in probs:
                if 'primitive_id' not in prob:
                    issues.append("missing primitive_id in problems")
                    break
        except TypeError as e:
            issues.append(f"bad signature: {e}")

        # Report
        if issues:
            print(f"  {pid}: FAIL - {', '.join(issues)}")
            errors.append(pid)
        else:
            print(f"  {pid}: OK")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} passives have issues")
        exit(1)
    else:
        print(f"PASSED: All {len(passives)} passives comply with contract")
        exit(0)
