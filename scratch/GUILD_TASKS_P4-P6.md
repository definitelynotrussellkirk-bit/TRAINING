# Guild Refactor - Tasks Phase 4-6

**Phases:** Skills Registry, Quest System, Progression Engine
**Prerequisites:** Phases 0-3 complete (guild-p3-complete tag)

---

# Phase 4: Skills Registry

**Goal:** Central skill definitions loaded from YAML configs

---

### P4.1 - Create guild/skills/registry.py

**Description:** Skill registry that loads and manages skill configurations

**File:** `guild/skills/registry.py`

```python
"""Skill registry - central management of skill definitions."""

from typing import Optional
from pathlib import Path

from guild.skills.types import SkillConfig, SkillCategory, MetricDefinition
from guild.config.loader import ConfigLoader, get_config_dir, dict_to_dataclass


class SkillRegistry:
    """
    Central registry for skill definitions.
    Loads skills from configs/skills/*.yaml
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self._skills: dict[str, SkillConfig] = {}
        self._metrics: dict[str, MetricDefinition] = {}
        self._loader = config_loader or ConfigLoader()
        self._register_builtin_metrics()

    def _register_builtin_metrics(self):
        """Register common metrics."""
        builtins = [
            MetricDefinition("accuracy", "Accuracy", "Fraction of correct responses"),
            MetricDefinition("word_accuracy", "Word Accuracy", "Fraction of correct words"),
            MetricDefinition("json_validity", "JSON Validity", "Whether output is valid JSON"),
            MetricDefinition("rouge_l", "ROUGE-L", "Longest common subsequence score"),
            MetricDefinition("loss", "Loss", "Training loss", higher_is_better=False),
            MetricDefinition("perplexity", "Perplexity", "Model perplexity", higher_is_better=False),
            MetricDefinition("format_compliance", "Format Compliance", "Adherence to output format"),
            MetricDefinition("instruction_accuracy", "Instruction Accuracy", "Following instructions"),
        ]
        for metric in builtins:
            self._metrics[metric.id] = metric

    def load_all(self) -> int:
        """
        Load all skill configs from the config directory.
        Returns number of skills loaded.
        """
        skills_dir = get_config_dir() / "skills"
        if not skills_dir.exists():
            return 0

        count = 0
        for path in skills_dir.glob("*.yaml"):
            if path.name.startswith("_"):
                continue
            try:
                self.load_skill(path)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to load skill {path}: {e}")

        return count

    def load_skill(self, path: Path | str) -> SkillConfig:
        """Load a single skill from a YAML file."""
        from guild.config.loader import load_yaml

        data = load_yaml(path)

        # Handle category enum
        if "category" in data and isinstance(data["category"], str):
            data["category"] = SkillCategory(data["category"])

        skill = SkillConfig(**{
            k: v for k, v in data.items()
            if k in SkillConfig.__dataclass_fields__
        })

        self._skills[skill.id] = skill
        return skill

    def register(self, skill: SkillConfig):
        """Register a skill directly (programmatic registration)."""
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Optional[SkillConfig]:
        """Get a skill by ID, returns None if not found."""
        return self._skills.get(skill_id)

    def get_or_raise(self, skill_id: str) -> SkillConfig:
        """Get a skill by ID, raises KeyError if not found."""
        skill = self.get(skill_id)
        if skill is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        return skill

    def list(self, category: Optional[SkillCategory] = None,
             tag: Optional[str] = None) -> list[SkillConfig]:
        """List skills, optionally filtered by category or tag."""
        skills = list(self._skills.values())

        if category:
            skills = [s for s in skills if s.category == category]

        if tag:
            skills = [s for s in skills if tag in s.tags]

        return skills

    def list_ids(self) -> list[str]:
        """List all skill IDs."""
        return list(self._skills.keys())

    def get_metric(self, metric_id: str) -> Optional[MetricDefinition]:
        """Get a metric definition."""
        return self._metrics.get(metric_id)

    def register_metric(self, metric: MetricDefinition):
        """Register a custom metric definition."""
        self._metrics[metric.id] = metric

    def list_metrics(self) -> list[MetricDefinition]:
        """List all registered metrics."""
        return list(self._metrics.values())

    @property
    def count(self) -> int:
        """Number of registered skills."""
        return len(self._skills)

    def to_dict(self) -> dict:
        """Export registry state for debugging/status."""
        return {
            "skills": {sid: {
                "name": s.name,
                "category": s.category.value,
                "tags": s.tags,
                "primary_metric": s.primary_metric,
            } for sid, s in self._skills.items()},
            "metrics": list(self._metrics.keys()),
        }


# Global registry instance
_registry: Optional[SkillRegistry] = None


def get_skill_registry(auto_load: bool = True) -> SkillRegistry:
    """
    Get the global skill registry.

    Args:
        auto_load: If True, automatically load skills from config directory
    """
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        if auto_load:
            _registry.load_all()
    return _registry


def reset_skill_registry():
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None


def get_skill(skill_id: str) -> Optional[SkillConfig]:
    """Get a skill from the global registry."""
    return get_skill_registry().get(skill_id)


def list_skills(category: Optional[SkillCategory] = None,
                tag: Optional[str] = None) -> list[SkillConfig]:
    """List skills from the global registry."""
    return get_skill_registry().list(category, tag)
```

**Dependencies:** P1.2, P2.1

**Acceptance Criteria:**
- [ ] `from guild.skills.registry import get_skill_registry, get_skill` works
- [ ] `registry.load_all()` loads skills from configs/skills/
- [ ] Filtering by category and tag works
- [ ] Built-in metrics are registered

**Effort:** M (30 min)

---

### P4.2 - Update guild/skills/__init__.py

**Description:** Export skill registry functions

**File:** `guild/skills/__init__.py`

```python
"""Skill definitions and registry."""

from guild.skills.types import (
    SkillConfig,
    SkillState,
    SkillCategory,
    MetricDefinition,
)
from guild.skills.registry import (
    SkillRegistry,
    get_skill_registry,
    get_skill,
    list_skills,
    reset_skill_registry,
)

__all__ = [
    "SkillConfig",
    "SkillState",
    "SkillCategory",
    "MetricDefinition",
    "SkillRegistry",
    "get_skill_registry",
    "get_skill",
    "list_skills",
    "reset_skill_registry",
]
```

**Dependencies:** P4.1

**Acceptance Criteria:**
- [ ] `from guild.skills import get_skill, SkillConfig` works

**Effort:** S (5 min)

---

### P4.3 - Create Additional Skill Configs

**Description:** Create more skill configuration files for common skills

**Files to Create:**

`configs/skills/arcane_compression.yaml`:
```yaml
id: arcane_compression
name: Arcane Compression
description: >
  The discipline of distilling long texts to their essential meaning.
  Used for summarization tasks.
category: compression

tags:
  - summarization
  - compression
  - extraction

metrics:
  - rouge_l
  - accuracy
  - format_compliance
primary_metric: rouge_l

accuracy_thresholds:
  1: 0.50
  2: 0.55
  3: 0.60
  4: 0.65
  5: 0.70
  6: 0.75
  7: 0.78
  8: 0.80
  9: 0.82
  10: 0.85

xp_multiplier: 1.0

rpg_name: Arcane Compression
rpg_description: >
  The art of compressing vast knowledge into pure essence.
  Masters can distill a tome into a single potent phrase.
```

`configs/skills/numerical_sorcery.yaml`:
```yaml
id: numerical_sorcery
name: Numerical Sorcery
description: >
  The discipline of mathematical reasoning and calculation.
  Used for arithmetic and quantitative tasks.
category: math

tags:
  - math
  - arithmetic
  - reasoning
  - numbers

metrics:
  - accuracy
  - format_compliance
primary_metric: accuracy

accuracy_thresholds:
  1: 0.50
  2: 0.55
  3: 0.60
  4: 0.65
  5: 0.70
  6: 0.75
  7: 0.80
  8: 0.85
  9: 0.88
  10: 0.90

xp_multiplier: 1.1

rpg_name: Numerical Sorcery
rpg_description: >
  The mystical art of manipulating numbers and quantities.
  Practitioners can calculate trajectories and divine probabilities.
```

`configs/skills/artificer_arts.yaml`:
```yaml
id: artificer_arts
name: Artificer Arts
description: >
  The discipline of understanding and manipulating code.
  Used for code comprehension and generation tasks.
category: code

tags:
  - code
  - programming
  - technical

metrics:
  - accuracy
  - format_compliance
  - json_validity
primary_metric: accuracy

accuracy_thresholds:
  1: 0.45
  2: 0.50
  3: 0.55
  4: 0.60
  5: 0.65
  6: 0.70
  7: 0.75
  8: 0.80
  9: 0.85
  10: 0.88

xp_multiplier: 1.2

rpg_name: Artificer Arts
rpg_description: >
  The craft of reading and weaving the arcane runes of code.
  Artificers can speak directly to constructs and machines.
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] All skill YAML files are valid
- [ ] `get_skill_registry().load_all()` loads all skills
- [ ] `registry.count >= 5` after loading

**Effort:** M (20 min)

---

### P4.4 - Create configs/skills/_schema.yaml

**Description:** Document the skill configuration schema

**File:** `configs/skills/_schema.yaml`

```yaml
# Skill Configuration Schema
# This file documents the expected structure of skill configs.
# Files starting with _ are not loaded as skills.

_schema:
  version: "1.0"
  description: "Schema for skill configuration files"

required_fields:
  - id: "Unique identifier (snake_case)"
  - name: "Display name"
  - description: "What this skill does"
  - category: "One of: reasoning, compression, generation, classification, tool_use, instruction, math, code"

optional_fields:
  - tags: "List of tags for filtering"
  - metrics: "List of metric IDs used to evaluate this skill"
  - primary_metric: "Main metric for progression (default: accuracy)"
  - accuracy_thresholds: "Map of level -> required accuracy for promotion"
  - xp_multiplier: "XP scaling factor (default: 1.0)"
  - rpg_name: "Name in RPG context (default: same as name)"
  - rpg_description: "RPG-flavored description"

example:
  id: example_skill
  name: Example Skill
  description: An example skill configuration
  category: reasoning
  tags:
    - example
    - demo
  metrics:
    - accuracy
  primary_metric: accuracy
  accuracy_thresholds:
    1: 0.60
    2: 0.70
    3: 0.80
  xp_multiplier: 1.0
  rpg_name: Example Skill
  rpg_description: A demonstration of skill configuration.

valid_categories:
  - reasoning
  - compression
  - generation
  - classification
  - tool_use
  - instruction
  - math
  - code

builtin_metrics:
  - accuracy
  - word_accuracy
  - json_validity
  - rouge_l
  - loss
  - perplexity
  - format_compliance
  - instruction_accuracy
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File exists with schema documentation
- [ ] File is NOT loaded as a skill (starts with _)

**Effort:** S (15 min)

---

### P4.5 - Create tests/guild/test_skills.py

**Description:** Tests for skill registry

**File:** `tests/guild/test_skills.py`

```python
"""Tests for skill registry."""

import pytest
import tempfile
from pathlib import Path

from guild.skills.registry import (
    SkillRegistry, get_skill_registry, reset_skill_registry
)
from guild.skills.types import SkillConfig, SkillCategory
from guild.config.loader import set_config_dir, get_config_dir


class TestSkillRegistry:
    @pytest.fixture
    def temp_skills_dir(self):
        """Create temporary config directory with test skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            logic_yaml = """
id: logic_weaving
name: Logic Weaving
description: Deductive reasoning
category: reasoning
tags:
  - reasoning
  - syllo
metrics:
  - accuracy
  - word_accuracy
primary_metric: accuracy
accuracy_thresholds:
  1: 0.6
  2: 0.7
  3: 0.8
"""
            (skills_dir / "logic_weaving.yaml").write_text(logic_yaml)

            oath_yaml = """
id: oath_binding
name: Oath Binding
description: Following instructions
category: instruction
tags:
  - instruction
metrics:
  - accuracy
primary_metric: accuracy
accuracy_thresholds:
  1: 0.55
  2: 0.65
"""
            (skills_dir / "oath_binding.yaml").write_text(oath_yaml)

            # Schema file should be ignored
            schema_yaml = """
_schema:
  version: "1.0"
"""
            (skills_dir / "_schema.yaml").write_text(schema_yaml)

            old_dir = get_config_dir()
            set_config_dir(tmpdir)
            reset_skill_registry()
            yield tmpdir
            set_config_dir(old_dir)
            reset_skill_registry()

    def test_load_all_skills(self, temp_skills_dir):
        registry = SkillRegistry()
        count = registry.load_all()

        assert count == 2
        assert registry.count == 2
        assert "logic_weaving" in registry.list_ids()
        assert "oath_binding" in registry.list_ids()

    def test_get_skill(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        skill = registry.get("logic_weaving")
        assert skill is not None
        assert skill.name == "Logic Weaving"
        assert skill.category == SkillCategory.REASONING

    def test_get_nonexistent_skill(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        assert registry.get("nonexistent") is None

        with pytest.raises(KeyError):
            registry.get_or_raise("nonexistent")

    def test_filter_by_category(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        reasoning = registry.list(category=SkillCategory.REASONING)
        assert len(reasoning) == 1
        assert reasoning[0].id == "logic_weaving"

        instruction = registry.list(category=SkillCategory.INSTRUCTION)
        assert len(instruction) == 1
        assert instruction[0].id == "oath_binding"

    def test_filter_by_tag(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        syllo = registry.list(tag="syllo")
        assert len(syllo) == 1
        assert syllo[0].id == "logic_weaving"

        reasoning = registry.list(tag="reasoning")
        assert len(reasoning) == 1

    def test_threshold_lookup(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        skill = registry.get("logic_weaving")
        assert skill.get_threshold(1) == 0.6
        assert skill.get_threshold(3) == 0.8
        # Beyond defined thresholds
        assert skill.get_threshold(10) == 0.8

    def test_builtin_metrics(self, temp_skills_dir):
        registry = SkillRegistry()

        accuracy = registry.get_metric("accuracy")
        assert accuracy is not None
        assert accuracy.higher_is_better is True

        loss = registry.get_metric("loss")
        assert loss is not None
        assert loss.higher_is_better is False

    def test_register_custom_metric(self, temp_skills_dir):
        from guild.skills.types import MetricDefinition

        registry = SkillRegistry()

        custom = MetricDefinition(
            id="custom_metric",
            name="Custom Metric",
            description="A custom metric",
            higher_is_better=True
        )
        registry.register_metric(custom)

        retrieved = registry.get_metric("custom_metric")
        assert retrieved is not None
        assert retrieved.name == "Custom Metric"

    def test_global_registry(self, temp_skills_dir):
        reset_skill_registry()

        registry1 = get_skill_registry()
        registry2 = get_skill_registry()

        assert registry1 is registry2
        assert registry1.count == 2  # Auto-loaded

    def test_schema_file_ignored(self, temp_skills_dir):
        registry = SkillRegistry()
        count = registry.load_all()

        # Should only load 2 skills, not _schema.yaml
        assert count == 2
        assert "_schema" not in registry.list_ids()

    def test_register_programmatic(self, temp_skills_dir):
        registry = SkillRegistry()

        skill = SkillConfig(
            id="programmatic_skill",
            name="Programmatic Skill",
            description="Added via code",
            category=SkillCategory.GENERATION,
        )
        registry.register(skill)

        assert registry.get("programmatic_skill") is not None
        assert registry.get("programmatic_skill").name == "Programmatic Skill"

    def test_to_dict(self, temp_skills_dir):
        registry = SkillRegistry()
        registry.load_all()

        data = registry.to_dict()
        assert "skills" in data
        assert "metrics" in data
        assert "logic_weaving" in data["skills"]
        assert "accuracy" in data["metrics"]
```

**Dependencies:** P4.1, P4.2

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_skills.py -v` passes all tests

**Effort:** M (30 min)

---

### P4.6 - Commit Phase 4

**Description:** Commit skills registry

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 4 - Skills Registry

- guild/skills/registry.py: SkillRegistry with YAML loading
- configs/skills/: logic_weaving, oath_binding, arcane_compression, numerical_sorcery, artificer_arts
- configs/skills/_schema.yaml: Schema documentation
- tests/guild/test_skills.py: Registry tests

Global registry with auto-loading, filtering by category/tag"
git tag guild-p4-complete
```

**Dependencies:** P4.1-P4.5

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p4-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 5: Quest System

**Goal:** Quest templates, instances, forge, and board

---

### P5.1 - Create guild/quests/registry.py

**Description:** Quest template registry loaded from YAML

**File:** `guild/quests/registry.py`

```python
"""Quest template registry."""

from typing import Optional
from pathlib import Path

from guild.quests.types import QuestTemplate, QuestDifficulty
from guild.config.loader import load_yaml, get_config_dir


class QuestRegistry:
    """
    Registry for quest templates.
    Loads templates from configs/quests/**/*.yaml
    """

    def __init__(self):
        self._templates: dict[str, QuestTemplate] = {}

    def load_all(self) -> int:
        """Load all quest templates from config directory."""
        quests_dir = get_config_dir() / "quests"
        if not quests_dir.exists():
            return 0

        count = 0
        # Load from subdirectories (syllo/, discrimination/, etc.)
        for subdir in quests_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                for path in subdir.glob("*.yaml"):
                    if path.name.startswith("_"):
                        continue
                    try:
                        templates = self.load_templates(path)
                        count += len(templates)
                    except Exception as e:
                        print(f"Warning: Failed to load {path}: {e}")

        # Also load from quests/ directly
        for path in quests_dir.glob("*.yaml"):
            if path.name.startswith("_"):
                continue
            try:
                templates = self.load_templates(path)
                count += len(templates)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        return count

    def load_templates(self, path: Path | str) -> list[QuestTemplate]:
        """
        Load templates from a YAML file.
        File can contain single template or list under 'templates' key.
        """
        data = load_yaml(path)
        templates = []

        # Check if it's a list of templates
        if "templates" in data:
            template_list = data["templates"]
        elif isinstance(data, list):
            template_list = data
        else:
            # Single template
            template_list = [data]

        for tdata in template_list:
            template = self._parse_template(tdata)
            self._templates[template.id] = template
            templates.append(template)

        return templates

    def _parse_template(self, data: dict) -> QuestTemplate:
        """Parse a template from dict data."""
        # Handle difficulty enum
        difficulty = data.get("difficulty", "bronze")
        if isinstance(difficulty, str):
            difficulty = QuestDifficulty[difficulty.upper()]
        elif isinstance(difficulty, int):
            difficulty = QuestDifficulty(difficulty)

        return QuestTemplate(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            skills=data.get("skills", []),
            regions=data.get("regions", []),
            difficulty=difficulty,
            difficulty_level=data.get("difficulty_level", 1),
            generator_id=data.get("generator_id", ""),
            generator_params=data.get("generator_params", {}),
            evaluator_id=data.get("evaluator_id", ""),
            evaluator_params=data.get("evaluator_params", {}),
            base_xp=data.get("base_xp", {}),
            tags=data.get("tags", []),
            enabled=data.get("enabled", True),
        )

    def register(self, template: QuestTemplate):
        """Register a template directly."""
        self._templates[template.id] = template

    def get(self, template_id: str) -> Optional[QuestTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def get_or_raise(self, template_id: str) -> QuestTemplate:
        """Get a template by ID, raise if not found."""
        template = self.get(template_id)
        if template is None:
            raise KeyError(f"Unknown quest template: {template_id}")
        return template

    def list(self,
             skill: Optional[str] = None,
             region: Optional[str] = None,
             difficulty: Optional[QuestDifficulty] = None,
             max_level: Optional[int] = None,
             enabled_only: bool = True) -> list[QuestTemplate]:
        """List templates with optional filters."""
        templates = list(self._templates.values())

        if enabled_only:
            templates = [t for t in templates if t.enabled]

        if skill:
            templates = [t for t in templates if skill in t.skills]

        if region:
            templates = [t for t in templates if region in t.regions]

        if difficulty:
            templates = [t for t in templates if t.difficulty == difficulty]

        if max_level is not None:
            templates = [t for t in templates if t.difficulty_level <= max_level]

        return templates

    def list_ids(self) -> list[str]:
        """List all template IDs."""
        return list(self._templates.keys())

    @property
    def count(self) -> int:
        """Number of registered templates."""
        return len(self._templates)


# Global registry
_registry: Optional[QuestRegistry] = None


def get_quest_registry(auto_load: bool = True) -> QuestRegistry:
    """Get the global quest registry."""
    global _registry
    if _registry is None:
        _registry = QuestRegistry()
        if auto_load:
            _registry.load_all()
    return _registry


def reset_quest_registry():
    """Reset the global registry."""
    global _registry
    _registry = None


def get_quest_template(template_id: str) -> Optional[QuestTemplate]:
    """Get a template from the global registry."""
    return get_quest_registry().get(template_id)
```

**Dependencies:** P1.3, P2.1

**Acceptance Criteria:**
- [ ] `from guild.quests.registry import get_quest_registry` works
- [ ] Templates load from configs/quests/**/*.yaml
- [ ] Filtering works correctly

**Effort:** M (35 min)

---

### P5.2 - Create guild/quests/forge.py

**Description:** QuestForge creates instances from templates using generators

**File:** `guild/quests/forge.py`

```python
"""Quest forge - generates quest instances from templates."""

from typing import Callable, Optional, Any
from datetime import datetime

from guild.quests.types import QuestTemplate, QuestInstance, QuestDifficulty
from guild.quests.registry import get_quest_registry
from guild.types import generate_id


# Generator function signature: (template, **params) -> (prompt, expected, metadata)
GeneratorFn = Callable[..., tuple[str, Optional[dict], dict]]


class QuestForge:
    """
    Generates quest instances from templates.

    Generators are registered functions that create prompt/expected pairs.
    """

    def __init__(self):
        self._generators: dict[str, GeneratorFn] = {}

    def register_generator(self, generator_id: str, fn: GeneratorFn):
        """Register a generator function."""
        self._generators[generator_id] = fn

    def has_generator(self, generator_id: str) -> bool:
        """Check if a generator is registered."""
        return generator_id in self._generators

    def list_generators(self) -> list[str]:
        """List registered generator IDs."""
        return list(self._generators.keys())

    def forge(self, template: QuestTemplate, count: int = 1,
              override_params: Optional[dict] = None) -> list[QuestInstance]:
        """
        Generate quest instances from a template.

        Args:
            template: Quest template to use
            count: Number of instances to generate
            override_params: Override generator params

        Returns:
            List of generated quest instances
        """
        if not template.generator_id:
            raise ValueError(f"Template {template.id} has no generator_id")

        generator = self._generators.get(template.generator_id)
        if generator is None:
            raise ValueError(f"Unknown generator: {template.generator_id}")

        params = {**template.generator_params}
        if override_params:
            params.update(override_params)

        instances = []
        for _ in range(count):
            try:
                prompt, expected, metadata = generator(template, **params)

                instance = QuestInstance(
                    id=generate_id("quest"),
                    template_id=template.id,
                    skills=template.skills.copy(),
                    difficulty=template.difficulty,
                    difficulty_level=template.difficulty_level,
                    prompt=prompt,
                    expected=expected,
                    metadata=metadata,
                    source=f"forge:{template.generator_id}",
                    created_at=datetime.now(),
                )
                instances.append(instance)

            except Exception as e:
                print(f"Warning: Generator {template.generator_id} failed: {e}")

        return instances

    def forge_by_id(self, template_id: str, count: int = 1,
                    override_params: Optional[dict] = None) -> list[QuestInstance]:
        """Generate instances by template ID."""
        template = get_quest_registry().get_or_raise(template_id)
        return self.forge(template, count, override_params)

    def forge_batch(self, templates: list[QuestTemplate],
                    count_per_template: int = 1) -> list[QuestInstance]:
        """Generate instances from multiple templates."""
        instances = []
        for template in templates:
            instances.extend(self.forge(template, count_per_template))
        return instances


# Global forge
_forge: Optional[QuestForge] = None


def get_quest_forge() -> QuestForge:
    """Get the global quest forge."""
    global _forge
    if _forge is None:
        _forge = QuestForge()
    return _forge


def reset_quest_forge():
    """Reset the global forge."""
    global _forge
    _forge = None


def register_generator(generator_id: str, fn: GeneratorFn):
    """Register a generator with the global forge."""
    get_quest_forge().register_generator(generator_id, fn)


# Built-in generators

def _passthrough_generator(template: QuestTemplate,
                           prompt: str = "",
                           expected: Optional[dict] = None,
                           **kwargs) -> tuple[str, Optional[dict], dict]:
    """
    Passthrough generator - uses prompt/expected from params directly.
    Useful for pre-generated quests or manual testing.
    """
    return prompt, expected, {"generator": "passthrough", **kwargs}


def _template_generator(template: QuestTemplate,
                        prompt_template: str = "",
                        variables: Optional[dict] = None,
                        expected: Optional[dict] = None,
                        **kwargs) -> tuple[str, Optional[dict], dict]:
    """
    Template generator - fills in a prompt template with variables.
    """
    variables = variables or {}
    prompt = prompt_template.format(**variables)
    return prompt, expected, {"generator": "template", "variables": variables, **kwargs}


# Register built-in generators
def _register_builtins():
    forge = get_quest_forge()
    forge.register_generator("passthrough", _passthrough_generator)
    forge.register_generator("template", _template_generator)


# Auto-register on import
_register_builtins()
```

**Dependencies:** P1.3, P5.1

**Acceptance Criteria:**
- [ ] `from guild.quests.forge import get_quest_forge, register_generator` works
- [ ] Generators can be registered and invoked
- [ ] `forge.forge(template)` creates valid instances
- [ ] Built-in generators work

**Effort:** M (35 min)

---

### P5.3 - Create guild/quests/board.py

**Description:** Quest board manages pending/active/completed quests

**File:** `guild/quests/board.py`

```python
"""Quest board - manages quest queue and lifecycle."""

from typing import Optional, Iterator
from collections import deque
from datetime import datetime
from pathlib import Path
import json

from guild.quests.types import QuestInstance, QuestResult


class QuestBoard:
    """
    Manages the quest lifecycle: pending -> active -> completed/failed.

    Can optionally persist to disk for crash recovery.
    """

    def __init__(self, persist_dir: Optional[Path | str] = None):
        self._pending: deque[QuestInstance] = deque()
        self._active: dict[str, QuestInstance] = {}  # quest_id -> instance
        self._completed: list[QuestResult] = []
        self._failed: list[tuple[QuestInstance, str]] = []  # (quest, error)

        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def post(self, quest: QuestInstance):
        """Post a new quest to the board."""
        self._pending.append(quest)
        self._persist_pending()

    def post_many(self, quests: list[QuestInstance]):
        """Post multiple quests."""
        self._pending.extend(quests)
        self._persist_pending()

    def draw(self) -> Optional[QuestInstance]:
        """
        Draw the next quest from the pending queue.
        Moves it to active status.
        """
        if not self._pending:
            return None

        quest = self._pending.popleft()
        self._active[quest.id] = quest
        self._persist_pending()
        return quest

    def complete(self, result: QuestResult):
        """Mark a quest as completed with result."""
        if result.quest_id in self._active:
            del self._active[result.quest_id]
        self._completed.append(result)
        self._persist_completed(result)

    def fail(self, quest_id: str, error: str):
        """Mark a quest as failed."""
        if quest_id in self._active:
            quest = self._active.pop(quest_id)
            self._failed.append((quest, error))
            self._persist_failed(quest, error)

    def return_to_board(self, quest_id: str):
        """Return an active quest back to pending (e.g., on timeout)."""
        if quest_id in self._active:
            quest = self._active.pop(quest_id)
            self._pending.appendleft(quest)
            self._persist_pending()

    def get_active(self, quest_id: str) -> Optional[QuestInstance]:
        """Get an active quest by ID."""
        return self._active.get(quest_id)

    def clear_completed(self, keep_last: int = 0):
        """Clear completed quest history, optionally keeping last N."""
        if keep_last > 0:
            self._completed = self._completed[-keep_last:]
        else:
            self._completed = []

    # Properties
    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    @property
    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def pending_quests(self) -> Iterator[QuestInstance]:
        """Iterate over pending quests (read-only)."""
        return iter(self._pending)

    def active_quests(self) -> Iterator[QuestInstance]:
        """Iterate over active quests."""
        return iter(self._active.values())

    def recent_results(self, limit: int = 10) -> list[QuestResult]:
        """Get recent completed results."""
        return self._completed[-limit:] if self._completed else []

    # Persistence
    def _persist_pending(self):
        """Persist pending queue to disk."""
        if not self._persist_dir:
            return
        path = self._persist_dir / "pending.json"
        data = [q.to_dict() for q in self._pending]
        path.write_text(json.dumps(data, indent=2))

    def _persist_completed(self, result: QuestResult):
        """Append completed result to log."""
        if not self._persist_dir:
            return
        path = self._persist_dir / "completed.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def _persist_failed(self, quest: QuestInstance, error: str):
        """Append failed quest to log."""
        if not self._persist_dir:
            return
        path = self._persist_dir / "failed.jsonl"
        data = {"quest": quest.to_dict(), "error": error, "time": datetime.now().isoformat()}
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def load_pending(self):
        """Load pending queue from disk."""
        if not self._persist_dir:
            return
        path = self._persist_dir / "pending.json"
        if path.exists():
            data = json.loads(path.read_text())
            self._pending = deque(QuestInstance.from_dict(d) for d in data)

    # Stats
    def stats(self) -> dict:
        """Get board statistics."""
        return {
            "pending": self.pending_count,
            "active": self.active_count,
            "completed": self.completed_count,
            "failed": self.failed_count,
        }

    def to_dict(self) -> dict:
        """Export board state."""
        return {
            "stats": self.stats(),
            "pending_ids": [q.id for q in self._pending],
            "active_ids": list(self._active.keys()),
            "recent_results": [r.to_dict() for r in self.recent_results(5)],
        }


# Global board
_board: Optional[QuestBoard] = None


def get_quest_board(persist_dir: Optional[Path | str] = None) -> QuestBoard:
    """Get the global quest board."""
    global _board
    if _board is None:
        _board = QuestBoard(persist_dir)
    return _board


def reset_quest_board():
    """Reset the global board."""
    global _board
    _board = None
```

**Dependencies:** P1.3

**Acceptance Criteria:**
- [ ] `from guild.quests.board import get_quest_board, QuestBoard` works
- [ ] Post/draw/complete lifecycle works
- [ ] Persistence to disk works
- [ ] Stats are accurate

**Effort:** M (35 min)

---

### P5.4 - Update guild/quests/__init__.py

**Description:** Export quest system components

**File:** `guild/quests/__init__.py`

```python
"""Quest system - templates, instances, and management."""

from guild.quests.types import (
    QuestTemplate,
    QuestInstance,
    QuestResult,
    QuestDifficulty,
    CombatResult,
)
from guild.quests.registry import (
    QuestRegistry,
    get_quest_registry,
    get_quest_template,
    reset_quest_registry,
)
from guild.quests.forge import (
    QuestForge,
    get_quest_forge,
    register_generator,
    reset_quest_forge,
)
from guild.quests.board import (
    QuestBoard,
    get_quest_board,
    reset_quest_board,
)

__all__ = [
    # Types
    "QuestTemplate",
    "QuestInstance",
    "QuestResult",
    "QuestDifficulty",
    "CombatResult",
    # Registry
    "QuestRegistry",
    "get_quest_registry",
    "get_quest_template",
    "reset_quest_registry",
    # Forge
    "QuestForge",
    "get_quest_forge",
    "register_generator",
    "reset_quest_forge",
    # Board
    "QuestBoard",
    "get_quest_board",
    "reset_quest_board",
]
```

**Dependencies:** P5.1, P5.2, P5.3

**Acceptance Criteria:**
- [ ] `from guild.quests import QuestForge, QuestBoard, get_quest_registry` works

**Effort:** S (5 min)

---

### P5.5 - Create Quest Template Configs

**Description:** Create initial quest template configurations

**Files to Create:**

`configs/quests/syllo/basic.yaml`:
```yaml
# Basic SYLLO quest templates (Levels 1-3)

templates:
  - id: syllo_l1_4word
    name: "SYLLO Level 1 - 4 Words"
    description: "Basic syllable puzzle with 4 words and strong hints"
    skills:
      - logic_weaving
    regions:
      - novice_valley
    difficulty: bronze
    difficulty_level: 1
    generator_id: syllo_generator
    generator_params:
      word_count: 4
      level: 1
      hint_strength: strong
    evaluator_id: syllo_evaluator
    evaluator_params:
      strict_format: false
    base_xp:
      logic_weaving: 10
    tags:
      - syllo
      - beginner

  - id: syllo_l2_4word
    name: "SYLLO Level 2 - 4 Words"
    description: "Basic puzzle with slightly degraded hints"
    skills:
      - logic_weaving
    regions:
      - novice_valley
    difficulty: bronze
    difficulty_level: 2
    generator_id: syllo_generator
    generator_params:
      word_count: 4
      level: 2
      hint_strength: medium
    evaluator_id: syllo_evaluator
    base_xp:
      logic_weaving: 12
    tags:
      - syllo
      - beginner

  - id: syllo_l3_5word
    name: "SYLLO Level 3 - 5 Words"
    description: "Slightly harder with 5 words"
    skills:
      - logic_weaving
    regions:
      - novice_valley
    difficulty: silver
    difficulty_level: 3
    generator_id: syllo_generator
    generator_params:
      word_count: 5
      level: 3
      hint_strength: medium
    evaluator_id: syllo_evaluator
    base_xp:
      logic_weaving: 15
    tags:
      - syllo
```

`configs/quests/syllo/intermediate.yaml`:
```yaml
# Intermediate SYLLO templates (Levels 4-6)

templates:
  - id: syllo_l4_5word
    name: "SYLLO Level 4 - 5 Words"
    description: "Intermediate puzzle with degraded hints"
    skills:
      - logic_weaving
    regions:
      - logic_foothills
    difficulty: silver
    difficulty_level: 4
    generator_id: syllo_generator
    generator_params:
      word_count: 5
      level: 4
    evaluator_id: syllo_evaluator
    base_xp:
      logic_weaving: 18
    tags:
      - syllo
      - intermediate

  - id: syllo_l5_6word
    name: "SYLLO Level 5 - 6 Words"
    description: "Harder puzzle with 6 words"
    skills:
      - logic_weaving
    regions:
      - logic_foothills
    difficulty: gold
    difficulty_level: 5
    generator_id: syllo_generator
    generator_params:
      word_count: 6
      level: 5
    evaluator_id: syllo_evaluator
    base_xp:
      logic_weaving: 22
    tags:
      - syllo
      - intermediate

  - id: syllo_l6_6word
    name: "SYLLO Level 6 - 6 Words"
    description: "Challenging puzzle with weak hints"
    skills:
      - logic_weaving
    regions:
      - logic_foothills
    difficulty: gold
    difficulty_level: 6
    generator_id: syllo_generator
    generator_params:
      word_count: 6
      level: 6
      hint_strength: weak
    evaluator_id: syllo_evaluator
    base_xp:
      logic_weaving: 25
    tags:
      - syllo
      - intermediate
```

`configs/quests/discrimination/templates.yaml`:
```yaml
# Discrimination training templates

templates:
  - id: discrimination_verify
    name: "Answer Verification"
    description: "Verify if a proposed answer is correct"
    skills:
      - logic_weaving
      - oath_binding
    regions:
      - novice_valley
      - logic_foothills
    difficulty: bronze
    difficulty_level: 1
    generator_id: discrimination_generator
    generator_params:
      mode: verification
    evaluator_id: discrimination_evaluator
    base_xp:
      logic_weaving: 8
      oath_binding: 5
    tags:
      - discrimination
      - verification

  - id: discrimination_correct
    name: "Answer Correction"
    description: "Identify error and provide correct answer"
    skills:
      - logic_weaving
      - oath_binding
    regions:
      - novice_valley
      - logic_foothills
    difficulty: silver
    difficulty_level: 3
    generator_id: discrimination_generator
    generator_params:
      mode: correction
    evaluator_id: discrimination_evaluator
    base_xp:
      logic_weaving: 12
      oath_binding: 8
    tags:
      - discrimination
      - correction
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] All YAML files are valid
- [ ] `get_quest_registry().load_all()` loads templates
- [ ] Templates have correct structure

**Effort:** M (25 min)

---

### P5.6 - Create tests/guild/test_quests.py

**Description:** Tests for quest system

**File:** `tests/guild/test_quests.py`

```python
"""Tests for quest system."""

import pytest
import tempfile
from pathlib import Path

from guild.quests.types import (
    QuestTemplate, QuestInstance, QuestResult,
    QuestDifficulty, CombatResult
)
from guild.quests.registry import (
    QuestRegistry, get_quest_registry, reset_quest_registry
)
from guild.quests.forge import (
    QuestForge, get_quest_forge, register_generator, reset_quest_forge
)
from guild.quests.board import QuestBoard, reset_quest_board
from guild.config.loader import set_config_dir, get_config_dir


class TestQuestRegistry:
    @pytest.fixture
    def temp_quests_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            quests_dir = Path(tmpdir) / "quests" / "syllo"
            quests_dir.mkdir(parents=True)

            templates_yaml = """
templates:
  - id: test_quest_1
    name: Test Quest 1
    description: A test quest
    skills:
      - logic_weaving
    regions:
      - novice_valley
    difficulty: bronze
    difficulty_level: 1
    generator_id: test_gen
    evaluator_id: test_eval
    base_xp:
      logic_weaving: 10

  - id: test_quest_2
    name: Test Quest 2
    description: Another test quest
    skills:
      - logic_weaving
    regions:
      - logic_foothills
    difficulty: gold
    difficulty_level: 5
    generator_id: test_gen
    evaluator_id: test_eval
"""
            (quests_dir / "test.yaml").write_text(templates_yaml)

            old_dir = get_config_dir()
            set_config_dir(tmpdir)
            reset_quest_registry()
            yield tmpdir
            set_config_dir(old_dir)
            reset_quest_registry()

    def test_load_templates(self, temp_quests_dir):
        registry = QuestRegistry()
        count = registry.load_all()

        assert count == 2
        assert "test_quest_1" in registry.list_ids()
        assert "test_quest_2" in registry.list_ids()

    def test_get_template(self, temp_quests_dir):
        registry = QuestRegistry()
        registry.load_all()

        template = registry.get("test_quest_1")
        assert template is not None
        assert template.name == "Test Quest 1"
        assert template.difficulty == QuestDifficulty.BRONZE

    def test_filter_by_skill(self, temp_quests_dir):
        registry = QuestRegistry()
        registry.load_all()

        templates = registry.list(skill="logic_weaving")
        assert len(templates) == 2

    def test_filter_by_region(self, temp_quests_dir):
        registry = QuestRegistry()
        registry.load_all()

        templates = registry.list(region="novice_valley")
        assert len(templates) == 1
        assert templates[0].id == "test_quest_1"

    def test_filter_by_max_level(self, temp_quests_dir):
        registry = QuestRegistry()
        registry.load_all()

        templates = registry.list(max_level=3)
        assert len(templates) == 1
        assert templates[0].difficulty_level == 1


class TestQuestForge:
    @pytest.fixture
    def forge_with_generator(self):
        reset_quest_forge()
        forge = get_quest_forge()

        def test_generator(template, **params):
            prompt = f"Test prompt for {template.id}"
            expected = {"answer": "test"}
            metadata = {"generated": True, **params}
            return prompt, expected, metadata

        forge.register_generator("test_gen", test_generator)
        yield forge
        reset_quest_forge()

    def test_register_generator(self, forge_with_generator):
        assert forge_with_generator.has_generator("test_gen")
        assert "test_gen" in forge_with_generator.list_generators()

    def test_forge_instance(self, forge_with_generator):
        template = QuestTemplate(
            id="forge_test",
            name="Forge Test",
            description="Test",
            skills=["logic_weaving"],
            regions=["test"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="test_gen",
            evaluator_id="test_eval",
        )

        instances = forge_with_generator.forge(template, count=3)

        assert len(instances) == 3
        for inst in instances:
            assert inst.template_id == "forge_test"
            assert inst.prompt.startswith("Test prompt")
            assert inst.expected == {"answer": "test"}

    def test_forge_with_override_params(self, forge_with_generator):
        template = QuestTemplate(
            id="param_test",
            name="Param Test",
            description="Test",
            skills=[],
            regions=[],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            generator_id="test_gen",
            generator_params={"default_param": "default"},
            evaluator_id="test_eval",
        )

        instances = forge_with_generator.forge(
            template,
            override_params={"custom_param": "custom"}
        )

        assert instances[0].metadata["custom_param"] == "custom"


class TestQuestBoard:
    @pytest.fixture
    def board(self):
        reset_quest_board()
        return QuestBoard()

    @pytest.fixture
    def sample_quest(self):
        return QuestInstance(
            id="test_quest_123",
            template_id="test_template",
            skills=["logic_weaving"],
            difficulty=QuestDifficulty.BRONZE,
            difficulty_level=1,
            prompt="Test prompt",
        )

    def test_post_and_draw(self, board, sample_quest):
        board.post(sample_quest)
        assert board.pending_count == 1

        drawn = board.draw()
        assert drawn is not None
        assert drawn.id == sample_quest.id
        assert board.pending_count == 0
        assert board.active_count == 1

    def test_complete_quest(self, board, sample_quest):
        board.post(sample_quest)
        board.draw()

        result = QuestResult(
            quest_id=sample_quest.id,
            hero_id="hero_1",
            response="test response",
            combat_result=CombatResult.HIT,
        )
        board.complete(result)

        assert board.active_count == 0
        assert board.completed_count == 1

    def test_fail_quest(self, board, sample_quest):
        board.post(sample_quest)
        board.draw()
        board.fail(sample_quest.id, "Test error")

        assert board.active_count == 0
        assert board.failed_count == 1

    def test_return_to_board(self, board, sample_quest):
        board.post(sample_quest)
        board.draw()
        assert board.active_count == 1

        board.return_to_board(sample_quest.id)
        assert board.active_count == 0
        assert board.pending_count == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create board with persistence
            board1 = QuestBoard(persist_dir=tmpdir)

            quest = QuestInstance(
                id="persist_test",
                template_id="test",
                skills=[],
                difficulty=QuestDifficulty.BRONZE,
                difficulty_level=1,
                prompt="Test",
            )
            board1.post(quest)

            # Create new board, load from disk
            board2 = QuestBoard(persist_dir=tmpdir)
            board2.load_pending()

            assert board2.pending_count == 1
            drawn = board2.draw()
            assert drawn.id == "persist_test"

    def test_stats(self, board, sample_quest):
        board.post(sample_quest)

        stats = board.stats()
        assert stats["pending"] == 1
        assert stats["active"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
```

**Dependencies:** P5.1, P5.2, P5.3

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_quests.py -v` passes all tests

**Effort:** M (35 min)

---

### P5.7 - Commit Phase 5

**Description:** Commit quest system

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 5 - Quest System

- guild/quests/registry.py: QuestRegistry with YAML loading
- guild/quests/forge.py: QuestForge with generator registration
- guild/quests/board.py: QuestBoard with lifecycle management
- configs/quests/syllo/: basic.yaml, intermediate.yaml
- configs/quests/discrimination/: templates.yaml
- tests/guild/test_quests.py: Quest system tests

Complete quest lifecycle: template -> forge -> board -> complete"
git tag guild-p5-complete
```

**Dependencies:** P5.1-P5.6

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p5-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Phase 6: Progression Engine

**Goal:** XP calculation, level-up logic, and status effect tracking

---

### P6.1 - Create guild/progression/engine.py

**Description:** Progression engine handles XP, levels, and trials

**File:** `guild/progression/engine.py`

```python
"""Progression engine - XP, levels, and trial management."""

from typing import Optional
from datetime import datetime

from guild.progression.types import (
    HeroState, HeroIdentity, SkillState, StatusEffect,
    EffectDefinition, EffectRuleConfig, EffectRuleState, EffectType
)
from guild.quests.types import QuestResult, CombatResult
from guild.skills.registry import get_skill_registry
from guild.types import Severity


class ProgressionEngine:
    """
    Manages hero progression: XP awards, level-ups, and trial eligibility.
    """

    def __init__(self, hero_state: HeroState):
        self.hero = hero_state
        self._effect_definitions: dict[str, EffectDefinition] = {}
        self._effect_rules: dict[str, EffectRuleConfig] = {}
        self._rule_states: dict[str, EffectRuleState] = {}

    # XP Management

    def award_xp(self, result: QuestResult) -> dict[str, float]:
        """
        Award XP based on quest result.
        Returns dict of skill_id -> xp_awarded.
        """
        awarded = {}

        for skill_id, base_xp in result.xp_awarded.items():
            skill_state = self.hero.get_skill(skill_id)

            # Apply multipliers
            multiplier = self._get_xp_multiplier(skill_id)
            final_xp = base_xp * multiplier

            skill_state.xp_total += final_xp
            self.hero.total_xp += final_xp
            awarded[skill_id] = final_xp

            # Record result for accuracy tracking
            skill_state.record_result(result.success)

            # Check trial eligibility
            self._check_trial_eligibility(skill_id)

        # Update hero stats
        self.hero.total_quests += 1
        if result.combat_result == CombatResult.CRITICAL_HIT:
            self.hero.total_crits += 1
        elif result.combat_result in [CombatResult.MISS, CombatResult.CRITICAL_MISS]:
            self.hero.total_misses += 1

        self.hero.updated_at = datetime.now()

        return awarded

    def _get_xp_multiplier(self, skill_id: str) -> float:
        """Get total XP multiplier for a skill."""
        multiplier = 1.0

        # Skill-specific multiplier
        skill_config = get_skill_registry().get(skill_id)
        if skill_config:
            multiplier *= skill_config.xp_multiplier

        # Effect-based multipliers
        for effect in self.hero.active_effects:
            if "xp_multiplier" in effect.effects:
                multiplier *= effect.effects["xp_multiplier"]

        return multiplier

    # Level Management

    def check_level_up(self, skill_id: str) -> bool:
        """
        Check if a skill is ready to level up.
        Returns True if eligible for promotion trial.
        """
        skill_state = self.hero.get_skill(skill_id)
        skill_config = get_skill_registry().get(skill_id)

        if not skill_config:
            return False

        # Get threshold for next level
        next_level = skill_state.level + 1
        threshold = skill_config.get_threshold(next_level)

        # Check accuracy meets threshold
        if skill_state.accuracy >= threshold:
            skill_state.eligible_for_trial = True
            return True

        return False

    def _check_trial_eligibility(self, skill_id: str):
        """Update trial eligibility for a skill."""
        skill_state = self.hero.get_skill(skill_id)
        skill_config = get_skill_registry().get(skill_id)

        if not skill_config:
            return

        next_level = skill_state.level + 1
        threshold = skill_config.get_threshold(next_level)

        # Need sufficient accuracy over rolling window
        if len(skill_state.recent_results) >= 20:  # Minimum sample
            if skill_state.accuracy >= threshold:
                skill_state.eligible_for_trial = True
            else:
                skill_state.eligible_for_trial = False

    def execute_promotion(self, skill_id: str, trial_passed: bool) -> bool:
        """
        Execute a promotion trial result.
        Returns True if level-up occurred.
        """
        skill_state = self.hero.get_skill(skill_id)

        if trial_passed:
            skill_state.record_level_up()
            skill_state.consecutive_trial_failures = 0
            return True
        else:
            skill_state.eligible_for_trial = False
            skill_state.consecutive_trial_failures += 1
            skill_state.last_trial_step = self.hero.current_step
            return False

    # Effect Management

    def register_effect(self, definition: EffectDefinition):
        """Register an effect definition."""
        self._effect_definitions[definition.id] = definition

    def register_rule(self, rule: EffectRuleConfig):
        """Register an effect trigger rule."""
        self._effect_rules[rule.id] = rule
        self._rule_states[rule.id] = EffectRuleState(rule_id=rule.id)

    def apply_effect(self, effect_id: str, cause: Optional[dict] = None):
        """Apply an effect to the hero."""
        definition = self._effect_definitions.get(effect_id)
        if not definition:
            raise ValueError(f"Unknown effect: {effect_id}")

        effect = definition.create_instance(
            step=self.hero.current_step,
            cause=cause
        )
        self.hero.add_effect(effect)

    def remove_effect(self, effect_id: str):
        """Remove an effect from the hero."""
        self.hero.remove_effect(effect_id)

    def check_effect_triggers(self, metrics: dict) -> list[str]:
        """
        Check if any effect rules should trigger.
        Returns list of triggered effect IDs.
        """
        triggered = []

        for rule_id, rule in self._effect_rules.items():
            state = self._rule_states[rule_id]

            # Check cooldown
            if (self.hero.current_step - state.last_triggered_step) < rule.cooldown_steps:
                continue

            # Check trigger condition
            if self._check_rule_condition(rule, metrics):
                self.apply_effect(rule.effect_id, cause={"rule": rule_id, "metrics": metrics})
                state.last_triggered_step = self.hero.current_step
                state.trigger_count += 1
                triggered.append(rule.effect_id)

        return triggered

    def _check_rule_condition(self, rule: EffectRuleConfig, metrics: dict) -> bool:
        """Check if a rule's trigger condition is met."""
        trigger_type = rule.trigger_type
        config = rule.trigger_config

        if trigger_type == "metric_threshold":
            metric_name = config.get("metric")
            op = config.get("op", "gt")
            value = config.get("value", 0)

            metric_value = metrics.get(metric_name)
            if metric_value is None:
                return False

            if op == "gt":
                return metric_value > value
            elif op == "lt":
                return metric_value < value
            elif op == "eq":
                return metric_value == value
            elif op == "is_nan":
                import math
                return math.isnan(metric_value) if isinstance(metric_value, float) else False

        elif trigger_type == "consecutive_failures":
            # This would need tracking of consecutive results
            # Simplified: check recent results
            count = config.get("count", 3)
            skill_id = rule.skill_id

            if skill_id:
                skill_state = self.hero.skills.get(skill_id)
                if skill_state:
                    recent = skill_state.recent_results[-count:]
                    if len(recent) >= count and not any(recent):
                        return True

        return False

    def update_step(self, step: int):
        """Update the current step and clear expired effects."""
        self.hero.current_step = step
        self.hero.clear_expired_effects(step)
        self.hero.updated_at = datetime.now()

    # Queries

    def get_eligible_trials(self) -> list[str]:
        """Get list of skill IDs eligible for promotion trial."""
        return [
            skill_id for skill_id, state in self.hero.skills.items()
            if state.eligible_for_trial
        ]

    def get_skill_summary(self, skill_id: str) -> dict:
        """Get summary of a skill's progression state."""
        state = self.hero.get_skill(skill_id)
        config = get_skill_registry().get(skill_id)

        next_threshold = config.get_threshold(state.level + 1) if config else 0.7

        return {
            "skill_id": skill_id,
            "level": state.level,
            "xp_total": state.xp_total,
            "accuracy": state.accuracy,
            "next_threshold": next_threshold,
            "eligible_for_trial": state.eligible_for_trial,
            "progress_to_trial": min(1.0, state.accuracy / next_threshold) if next_threshold > 0 else 0,
        }


# Factory function

def create_progression_engine(hero_id: str, identity: HeroIdentity) -> ProgressionEngine:
    """Create a new progression engine with fresh hero state."""
    state = HeroState(hero_id=hero_id, identity=identity)
    return ProgressionEngine(state)


def load_progression_engine(state_dict: dict) -> ProgressionEngine:
    """Load a progression engine from saved state."""
    state = HeroState.from_dict(state_dict)
    return ProgressionEngine(state)
```

**Dependencies:** P1.2, P1.5, P4.1

**Acceptance Criteria:**
- [ ] `from guild.progression.engine import ProgressionEngine` works
- [ ] XP awards calculate correctly with multipliers
- [ ] Level-up eligibility checks work
- [ ] Effect triggers work

**Effort:** L (45 min)

---

### P6.2 - Create guild/progression/effects.py

**Description:** Effect management utilities

**File:** `guild/progression/effects.py`

```python
"""Effect management utilities."""

from typing import Optional
from pathlib import Path

from guild.progression.types import (
    EffectDefinition, EffectRuleConfig, EffectType
)
from guild.types import Severity
from guild.config.loader import load_yaml, get_config_dir


def load_effects_config(path: Optional[Path] = None) -> tuple[
    dict[str, EffectDefinition],
    list[EffectRuleConfig]
]:
    """
    Load effect definitions and rules from config file.

    Returns:
        Tuple of (effect_definitions, rules)
    """
    if path is None:
        path = get_config_dir() / "progression" / "effects.yaml"

    if not path.exists():
        return {}, []

    data = load_yaml(path)

    # Parse effect definitions
    definitions = {}
    for effect_id, effect_data in data.get("effects", {}).items():
        effect_type = effect_data.get("type", "debuff")
        if isinstance(effect_type, str):
            effect_type = EffectType(effect_type)

        severity = effect_data.get("severity", "medium")
        if isinstance(severity, str):
            severity = Severity(severity)

        definitions[effect_id] = EffectDefinition(
            id=effect_id,
            name=effect_data.get("name", effect_id),
            description=effect_data.get("description", ""),
            type=effect_type,
            severity=severity,
            default_duration_steps=effect_data.get("default_duration_steps"),
            cure_condition=effect_data.get("cure_condition"),
            effects=effect_data.get("effects", {}),
            rpg_name=effect_data.get("rpg_name"),
            rpg_description=effect_data.get("rpg_description"),
        )

    # Parse rules
    rules = []
    for rule_data in data.get("rules", []):
        rules.append(EffectRuleConfig(
            id=rule_data["id"],
            effect_id=rule_data["effect_id"],
            trigger_type=rule_data.get("trigger_type", "metric_threshold"),
            trigger_config=rule_data.get("trigger_config", {}),
            cooldown_steps=rule_data.get("cooldown_steps", 100),
            skill_id=rule_data.get("skill_id"),
        ))

    return definitions, rules


def setup_default_effects(engine: "ProgressionEngine"):
    """Set up default effects on a progression engine."""
    definitions, rules = load_effects_config()

    for definition in definitions.values():
        engine.register_effect(definition)

    for rule in rules:
        engine.register_rule(rule)


# Built-in effect definitions (used if no config file)

BUILTIN_EFFECTS = {
    "tunnel_vision": EffectDefinition(
        id="tunnel_vision",
        name="Tunnel Vision",
        description="Overfitting detected",
        type=EffectType.DEBUFF,
        severity=Severity.MEDIUM,
        effects={"warning": True},
        rpg_name="Tunnel Vision",
    ),
    "confusion": EffectDefinition(
        id="confusion",
        name="Confusion",
        description="Repeated failures",
        type=EffectType.DEBUFF,
        severity=Severity.MEDIUM,
        default_duration_steps=500,
        effects={"xp_multiplier": 0.8},
        rpg_name="Confusion",
    ),
    "exhaustion": EffectDefinition(
        id="exhaustion",
        name="Exhaustion",
        description="Resource exhaustion",
        type=EffectType.DEBUFF,
        severity=Severity.HIGH,
        effects={"training_blocked": True},
        rpg_name="Exhaustion",
    ),
    "reality_tear": EffectDefinition(
        id="reality_tear",
        name="Reality Tear",
        description="NaN loss detected",
        type=EffectType.DEBUFF,
        severity=Severity.CRITICAL,
        effects={"training_blocked": True, "requires_attention": True},
        rpg_name="Reality Tear",
    ),
}
```

**Dependencies:** P1.5, P2.1

**Acceptance Criteria:**
- [ ] `from guild.progression.effects import load_effects_config` works
- [ ] Effects load from YAML config
- [ ] Built-in effects are available

**Effort:** M (25 min)

---

### P6.3 - Update guild/progression/__init__.py

**Description:** Export progression components

**File:** `guild/progression/__init__.py`

```python
"""Progression system - XP, levels, and effects."""

from guild.progression.types import (
    StatusEffect,
    EffectType,
    EffectDefinition,
    EffectRuleConfig,
    EffectRuleState,
    HeroIdentity,
    HeroState,
)
from guild.progression.engine import (
    ProgressionEngine,
    create_progression_engine,
    load_progression_engine,
)
from guild.progression.effects import (
    load_effects_config,
    setup_default_effects,
    BUILTIN_EFFECTS,
)

__all__ = [
    # Types
    "StatusEffect",
    "EffectType",
    "EffectDefinition",
    "EffectRuleConfig",
    "EffectRuleState",
    "HeroIdentity",
    "HeroState",
    # Engine
    "ProgressionEngine",
    "create_progression_engine",
    "load_progression_engine",
    # Effects
    "load_effects_config",
    "setup_default_effects",
    "BUILTIN_EFFECTS",
]
```

**Dependencies:** P6.1, P6.2

**Acceptance Criteria:**
- [ ] `from guild.progression import ProgressionEngine, HeroState` works

**Effort:** S (5 min)

---

### P6.4 - Create tests/guild/test_progression.py

**Description:** Tests for progression engine

**File:** `tests/guild/test_progression.py`

```python
"""Tests for progression engine."""

import pytest
import tempfile
from pathlib import Path

from guild.progression.engine import (
    ProgressionEngine, create_progression_engine
)
from guild.progression.types import (
    HeroState, HeroIdentity, EffectDefinition, EffectType, EffectRuleConfig
)
from guild.progression.effects import BUILTIN_EFFECTS
from guild.quests.types import QuestResult, CombatResult
from guild.skills.types import SkillConfig, SkillCategory
from guild.skills.registry import get_skill_registry, reset_skill_registry
from guild.config.loader import set_config_dir, get_config_dir
from guild.types import Severity


class TestProgressionEngine:
    @pytest.fixture
    def temp_config(self):
        """Set up temp config with test skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            skill_yaml = """
id: test_skill
name: Test Skill
description: For testing
category: reasoning
accuracy_thresholds:
  1: 0.5
  2: 0.6
  3: 0.7
xp_multiplier: 1.5
"""
            (skills_dir / "test_skill.yaml").write_text(skill_yaml)

            old_dir = get_config_dir()
            set_config_dir(tmpdir)
            reset_skill_registry()
            yield tmpdir
            set_config_dir(old_dir)
            reset_skill_registry()

    @pytest.fixture
    def engine(self, temp_config):
        identity = HeroIdentity(
            id="test_hero",
            name="Test Hero",
            architecture="qwen",
            generation="3",
            size="0.6B",
            variant="base",
        )
        return create_progression_engine("test_hero", identity)

    def test_award_xp(self, engine):
        result = QuestResult(
            quest_id="q1",
            hero_id="test_hero",
            response="answer",
            combat_result=CombatResult.HIT,
            xp_awarded={"test_skill": 10},
        )

        awarded = engine.award_xp(result)

        # With 1.5x multiplier
        assert awarded["test_skill"] == 15.0
        assert engine.hero.get_skill("test_skill").xp_total == 15.0
        assert engine.hero.total_quests == 1

    def test_accuracy_tracking(self, engine):
        skill_state = engine.hero.get_skill("test_skill")

        # Record some results
        for _ in range(7):
            skill_state.record_result(True)
        for _ in range(3):
            skill_state.record_result(False)

        assert skill_state.accuracy == pytest.approx(0.7)

    def test_level_up_eligibility(self, engine):
        skill_state = engine.hero.get_skill("test_skill")

        # Need 50% accuracy for level 2
        # Record results to hit threshold
        for _ in range(20):
            skill_state.record_result(True)

        assert engine.check_level_up("test_skill") is True
        assert skill_state.eligible_for_trial is True

    def test_promotion_success(self, engine):
        skill_state = engine.hero.get_skill("test_skill")
        skill_state.eligible_for_trial = True

        result = engine.execute_promotion("test_skill", trial_passed=True)

        assert result is True
        assert skill_state.level == 2
        assert skill_state.eligible_for_trial is False

    def test_promotion_failure(self, engine):
        skill_state = engine.hero.get_skill("test_skill")
        skill_state.eligible_for_trial = True
        engine.hero.current_step = 100

        result = engine.execute_promotion("test_skill", trial_passed=False)

        assert result is False
        assert skill_state.level == 1
        assert skill_state.consecutive_trial_failures == 1
        assert skill_state.last_trial_step == 100

    def test_effect_application(self, engine):
        # Register an effect
        confusion = BUILTIN_EFFECTS["confusion"]
        engine.register_effect(confusion)

        engine.apply_effect("confusion", cause={"reason": "test"})

        assert len(engine.hero.active_effects) == 1
        assert engine.hero.active_effects[0].id == "confusion"
        assert engine.hero.health != "healthy"

    def test_effect_removal(self, engine):
        confusion = BUILTIN_EFFECTS["confusion"]
        engine.register_effect(confusion)
        engine.apply_effect("confusion")

        engine.remove_effect("confusion")

        assert len(engine.hero.active_effects) == 0
        assert engine.hero.health == "healthy"

    def test_xp_multiplier_from_effect(self, engine):
        # Register confusion effect with 0.8x multiplier
        confusion = BUILTIN_EFFECTS["confusion"]
        engine.register_effect(confusion)
        engine.apply_effect("confusion")

        result = QuestResult(
            quest_id="q1",
            hero_id="test_hero",
            response="answer",
            combat_result=CombatResult.HIT,
            xp_awarded={"test_skill": 10},
        )

        awarded = engine.award_xp(result)

        # 10 * 1.5 (skill) * 0.8 (effect) = 12
        assert awarded["test_skill"] == pytest.approx(12.0)

    def test_effect_expiry(self, engine):
        # Create effect with short duration
        short_effect = EffectDefinition(
            id="short",
            name="Short",
            description="Short effect",
            type=EffectType.DEBUFF,
            severity=Severity.LOW,
            default_duration_steps=10,
        )
        engine.register_effect(short_effect)

        engine.hero.current_step = 0
        engine.apply_effect("short")
        assert len(engine.hero.active_effects) == 1

        # Advance past duration
        engine.update_step(15)
        assert len(engine.hero.active_effects) == 0

    def test_skill_summary(self, engine):
        skill_state = engine.hero.get_skill("test_skill")
        skill_state.xp_total = 1000
        for _ in range(10):
            skill_state.record_result(True)
        for _ in range(10):
            skill_state.record_result(False)

        summary = engine.get_skill_summary("test_skill")

        assert summary["skill_id"] == "test_skill"
        assert summary["level"] == 1
        assert summary["xp_total"] == 1000
        assert summary["accuracy"] == pytest.approx(0.5)
        assert summary["next_threshold"] == 0.5  # For level 2

    def test_crit_and_miss_tracking(self, engine):
        crit_result = QuestResult(
            quest_id="q1",
            hero_id="test_hero",
            response="answer",
            combat_result=CombatResult.CRITICAL_HIT,
            xp_awarded={"test_skill": 15},
        )
        engine.award_xp(crit_result)

        miss_result = QuestResult(
            quest_id="q2",
            hero_id="test_hero",
            response="wrong",
            combat_result=CombatResult.MISS,
            xp_awarded={"test_skill": 2},
        )
        engine.award_xp(miss_result)

        assert engine.hero.total_crits == 1
        assert engine.hero.total_misses == 1
        assert engine.hero.total_quests == 2
```

**Dependencies:** P6.1, P6.2, P6.3

**Acceptance Criteria:**
- [ ] `pytest tests/guild/test_progression.py -v` passes all tests

**Effort:** L (40 min)

---

### P6.5 - Create configs/progression/thresholds.yaml

**Description:** Global progression threshold configuration

**File:** `configs/progression/thresholds.yaml`

```yaml
# Global progression thresholds and XP curves

# Default accuracy thresholds by level (used if skill doesn't define own)
default_accuracy_thresholds:
  1: 0.55
  2: 0.60
  3: 0.65
  4: 0.70
  5: 0.73
  6: 0.76
  7: 0.79
  8: 0.82
  9: 0.85
  10: 0.88

# XP requirements (optional - for display purposes)
xp_curves:
  # Level -> total XP needed to reach this level
  1: 0
  2: 5000
  3: 15000
  4: 30000
  5: 50000
  6: 80000
  7: 120000
  8: 180000
  9: 260000
  10: 400000

# Trial configuration
trials:
  # Minimum quests attempted before eligible
  min_quests_for_trial: 20
  # Cooldown after failed trial (steps)
  failed_trial_cooldown: 1000
  # Max consecutive failures before debuff
  max_consecutive_failures: 3

# Accuracy window
accuracy_window_size: 100

# Region unlock requirements
region_unlocks:
  novice_valley:
    required_level: 1
    skills: []
  logic_foothills:
    required_level: 4
    skills:
      - logic_weaving
  reasoning_mountains:
    required_level: 7
    skills:
      - logic_weaving
  summit:
    required_level: 10
    skills:
      - logic_weaving
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File is valid YAML
- [ ] Contains thresholds for levels 1-10

**Effort:** S (15 min)

---

### P6.6 - Create configs/regions/novice_valley.yaml

**Description:** Region configuration for starting area

**File:** `configs/regions/novice_valley.yaml`

```yaml
id: novice_valley
name: Novice Valley
description: >
  The starting region where new heroes begin their journey.
  Quests here are gentle, designed to build confidence and fundamentals.

level_range:
  min: 1
  max: 3

difficulty_range:
  min: bronze
  max: silver

skills_trained:
  - logic_weaving
  - oath_binding

unlock_requirements: null  # Starting region, always available

quest_templates:
  - syllo_l1_4word
  - syllo_l2_4word
  - syllo_l3_5word
  - discrimination_verify

rpg_name: Novice Valley
rpg_description: >
  A peaceful valley where new heroes begin their journey.
  The Guild maintains practice grounds here, with instructors
  ready to guide fledgling adventurers through their first trials.

# Visual/UI hints
theme:
  color: "#4CAF50"  # Green
  icon: "🌳"
  atmosphere: peaceful
```

**Dependencies:** P0.2

**Acceptance Criteria:**
- [ ] File is valid YAML

**Effort:** S (10 min)

---

### P6.7 - Commit Phase 6

**Description:** Commit progression engine

**Commands:**
```bash
git add -A
git commit -m "feat(guild): Phase 6 - Progression Engine

- guild/progression/engine.py: ProgressionEngine with XP, levels, effects
- guild/progression/effects.py: Effect loading and built-ins
- configs/progression/thresholds.yaml: Global progression config
- configs/progression/effects.yaml: Effect definitions (from P2.7)
- configs/regions/novice_valley.yaml: Starting region
- tests/guild/test_progression.py: Progression tests

XP multipliers, level-up eligibility, effect triggers, trial management"
git tag guild-p6-complete
```

**Dependencies:** P6.1-P6.6

**Acceptance Criteria:**
- [ ] Clean git status
- [ ] Tag `guild-p6-complete` exists
- [ ] All tests pass

**Effort:** S (5 min)

---

# Checkpoint: Validate Phase 4-6

```bash
# All guild tests pass
pytest tests/guild/ -v

# Verify skill registry
python -c "
from guild.skills import get_skill_registry
reg = get_skill_registry()
print(f'Skills loaded: {reg.count}')
print(f'Skills: {reg.list_ids()}')
"

# Verify quest system
python -c "
from guild.quests import get_quest_registry, get_quest_forge, QuestBoard
reg = get_quest_registry()
print(f'Quest templates: {reg.count}')
forge = get_quest_forge()
print(f'Generators: {forge.list_generators()}')
"

# Verify progression
python -c "
from guild.progression import create_progression_engine, HeroIdentity
identity = HeroIdentity(
    id='test', name='Test',
    architecture='qwen', generation='3', size='0.6B', variant='base'
)
engine = create_progression_engine('test', identity)
print(f'Hero created: {engine.hero.hero_id}')
print(f'Health: {engine.hero.health}')
"
```

**Decision Point:**
- All tests pass → Continue to Phase 7
- Issues found → Fix before proceeding

---

**Total Tasks in Phases 4-6:** 20 tasks
**Estimated Time:** 1-2 weeks
