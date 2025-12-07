"""Quest forge - generates quest instances from templates."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from guild.quests.types import QuestTemplate, QuestInstance, QuestDifficulty
from guild.quests.registry import get_quest_registry


logger = logging.getLogger(__name__)


class QuestGenerator(ABC):
    """
    Abstract base class for quest generators.

    Generators create quest instances from templates by:
    1. Generating prompts
    2. Computing expected answers
    3. Adding context/metadata

    Subclass this to create custom generators for different quest types.
    """

    generator_id: str = "base"

    @abstractmethod
    def generate(
        self,
        template: QuestTemplate,
        params: Optional[dict] = None,
    ) -> QuestInstance:
        """
        Generate a single quest instance.

        Args:
            template: The quest template
            params: Optional generation parameters

        Returns:
            QuestInstance
        """
        pass

    def generate_batch(
        self,
        template: QuestTemplate,
        count: int,
        params: Optional[dict] = None,
    ) -> list[QuestInstance]:
        """
        Generate multiple quest instances.

        Override for batch-optimized generation.

        Args:
            template: The quest template
            count: Number of instances to generate
            params: Optional generation parameters

        Returns:
            List of QuestInstances
        """
        return [self.generate(template, params) for _ in range(count)]


class StaticGenerator(QuestGenerator):
    """
    Simple generator that uses fixed prompts from template.

    Useful for templates with static content or for testing.
    """

    generator_id: str = "static"

    def generate(
        self,
        template: QuestTemplate,
        params: Optional[dict] = None,
    ) -> QuestInstance:
        """Generate instance with static prompt from params."""
        params = params or {}
        merged = {**template.generator_params, **params}

        prompt = merged.get("prompt", f"Complete this {template.name} quest.")
        expected = merged.get("expected")
        context = merged.get("context", {})

        return QuestInstance.create(
            template=template,
            prompt=prompt,
            expected=expected,
            context=context,
            metadata={"generator": self.generator_id},
        )


class CallbackGenerator(QuestGenerator):
    """
    Generator that uses a callback function.

    Allows registering arbitrary generation functions.
    """

    generator_id: str = "callback"

    def __init__(
        self,
        callback: Callable[[QuestTemplate, dict], tuple[str, Optional[dict], dict]],
        generator_id: str = "callback",
    ):
        """
        Initialize with callback.

        Callback signature: (template, params) -> (prompt, expected, context)
        """
        self.callback = callback
        self.generator_id = generator_id

    def generate(
        self,
        template: QuestTemplate,
        params: Optional[dict] = None,
    ) -> QuestInstance:
        """Generate using callback."""
        params = params or {}
        merged = {**template.generator_params, **params}

        prompt, expected, context = self.callback(template, merged)

        return QuestInstance.create(
            template=template,
            prompt=prompt,
            expected=expected,
            context=context,
            metadata={"generator": self.generator_id},
        )


class QuestForge:
    """
    Factory for creating quest instances.

    The QuestForge:
    1. Manages registered generators
    2. Routes templates to appropriate generators
    3. Creates QuestInstance objects

    Example:
        forge = QuestForge()
        forge.register(MyCustomGenerator())

        template = get_quest("syllo_puzzle")
        instance = forge.create(template)
    """

    def __init__(self):
        self._generators: dict[str, QuestGenerator] = {}
        self._default_generator = StaticGenerator()

        # Register built-in generators
        self.register(self._default_generator)

    def register(self, generator: QuestGenerator):
        """
        Register a generator.

        Args:
            generator: QuestGenerator instance
        """
        self._generators[generator.generator_id] = generator
        logger.debug(f"Registered quest generator: {generator.generator_id}")

    def unregister(self, generator_id: str):
        """Remove a registered generator."""
        self._generators.pop(generator_id, None)

    def get_generator(self, generator_id: str) -> Optional[QuestGenerator]:
        """Get a generator by ID."""
        return self._generators.get(generator_id)

    def list_generators(self) -> list[str]:
        """List registered generator IDs."""
        return list(self._generators.keys())

    def create(
        self,
        template: QuestTemplate,
        params: Optional[dict] = None,
    ) -> QuestInstance:
        """
        Create a quest instance from a template.

        Uses the generator specified in template.generator_id,
        falling back to default if not found.

        Args:
            template: Quest template
            params: Optional override parameters

        Returns:
            QuestInstance
        """
        generator = self._generators.get(template.generator_id)

        if generator is None:
            logger.warning(
                f"Generator '{template.generator_id}' not found, using default"
            )
            generator = self._default_generator

        instance = generator.generate(template, params)

        logger.debug(
            f"Created quest instance: {instance.id} from {template.id} "
            f"using {generator.generator_id}"
        )

        return instance

    def create_batch(
        self,
        template: QuestTemplate,
        count: int,
        params: Optional[dict] = None,
    ) -> list[QuestInstance]:
        """
        Create multiple quest instances from a template.

        Args:
            template: Quest template
            count: Number of instances
            params: Optional override parameters

        Returns:
            List of QuestInstances
        """
        generator = self._generators.get(template.generator_id)

        if generator is None:
            generator = self._default_generator

        return generator.generate_batch(template, count, params)

    def create_from_id(
        self,
        quest_id: str,
        params: Optional[dict] = None,
    ) -> QuestInstance:
        """
        Create a quest instance by template ID.

        Args:
            quest_id: Quest template ID
            params: Optional override parameters

        Returns:
            QuestInstance
        """
        template = get_quest_registry().get(quest_id)
        return self.create(template, params)

    def create_for_skill(
        self,
        skill_id: str,
        difficulty_level: Optional[int] = None,
        params: Optional[dict] = None,
    ) -> Optional[QuestInstance]:
        """
        Create a quest instance for a skill at appropriate difficulty.

        Args:
            skill_id: Skill to train
            difficulty_level: Target difficulty (1-10)
            params: Optional override parameters

        Returns:
            QuestInstance or None if no suitable template found
        """
        registry = get_quest_registry()
        templates = registry.by_skill(skill_id)

        if not templates:
            logger.warning(f"No quest templates found for skill: {skill_id}")
            return None

        # Filter by difficulty if specified
        if difficulty_level is not None:
            level_templates = [
                t for t in templates
                if t.difficulty_level == difficulty_level
            ]
            if level_templates:
                templates = level_templates
            else:
                # Find closest level
                templates.sort(
                    key=lambda t: abs(t.difficulty_level - difficulty_level)
                )

        if not templates:
            return None

        # Use first matching template
        template = templates[0]
        return self.create(template, params)

    def create_for_primitive(
        self,
        primitive: str,
        difficulty_level: Optional[int] = None,
        params: Optional[dict] = None,
    ) -> Optional[QuestInstance]:
        """
        Create a quest instance that trains a specific primitive.

        Args:
            primitive: Primitive ID to train (e.g., "add_single_digit_no_carry")
            difficulty_level: Target difficulty (1-10)
            params: Optional override parameters

        Returns:
            QuestInstance or None if no suitable template found
        """
        registry = get_quest_registry()
        templates = registry.by_primitive(primitive)

        if not templates:
            logger.warning(f"No quest templates found for primitive: {primitive}")
            return None

        # Filter by difficulty if specified
        if difficulty_level is not None:
            level_templates = [
                t for t in templates
                if t.difficulty_level == difficulty_level
            ]
            if level_templates:
                templates = level_templates
            else:
                # Find closest level
                templates.sort(
                    key=lambda t: abs(t.difficulty_level - difficulty_level)
                )

        if not templates:
            return None

        # Use first matching template
        template = templates[0]
        return self.create(template, params)

    def create_for_module(
        self,
        module_id: str,
        difficulty_level: Optional[int] = None,
        params: Optional[dict] = None,
    ) -> Optional[QuestInstance]:
        """
        Create a quest instance from a specific module.

        Args:
            module_id: Module to select from
            difficulty_level: Target difficulty (1-10)
            params: Optional override parameters

        Returns:
            QuestInstance or None if no suitable template found
        """
        registry = get_quest_registry()
        templates = registry.by_module(module_id)

        if not templates:
            logger.warning(f"No quest templates found for module: {module_id}")
            return None

        # Filter by difficulty if specified
        if difficulty_level is not None:
            level_templates = [
                t for t in templates
                if t.difficulty_level == difficulty_level
            ]
            if level_templates:
                templates = level_templates
            else:
                templates.sort(
                    key=lambda t: abs(t.difficulty_level - difficulty_level)
                )

        if not templates:
            return None

        template = templates[0]
        return self.create(template, params)


# Global forge instance
_forge: Optional[QuestForge] = None


def get_forge() -> QuestForge:
    """Get the global quest forge, initializing if needed."""
    global _forge
    if _forge is None:
        _forge = QuestForge()
    return _forge


def reset_forge():
    """Reset the global forge (for testing)."""
    global _forge
    _forge = None


# Convenience functions

def create_quest(
    template: QuestTemplate,
    params: Optional[dict] = None,
) -> QuestInstance:
    """Create a quest instance from a template."""
    return get_forge().create(template, params)


def create_quest_by_id(
    quest_id: str,
    params: Optional[dict] = None,
) -> QuestInstance:
    """Create a quest instance by template ID."""
    return get_forge().create_from_id(quest_id, params)


def create_quest_for_skill(
    skill_id: str,
    difficulty_level: Optional[int] = None,
    params: Optional[dict] = None,
) -> Optional[QuestInstance]:
    """Create a quest instance for a skill."""
    return get_forge().create_for_skill(skill_id, difficulty_level, params)


def create_quest_for_primitive(
    primitive: str,
    difficulty_level: Optional[int] = None,
    params: Optional[dict] = None,
) -> Optional[QuestInstance]:
    """Create a quest instance for a primitive."""
    return get_forge().create_for_primitive(primitive, difficulty_level, params)


def create_quest_for_module(
    module_id: str,
    difficulty_level: Optional[int] = None,
    params: Optional[dict] = None,
) -> Optional[QuestInstance]:
    """Create a quest instance from a module."""
    return get_forge().create_for_module(module_id, difficulty_level, params)


def register_generator(generator: QuestGenerator):
    """Register a generator with the global forge."""
    get_forge().register(generator)
