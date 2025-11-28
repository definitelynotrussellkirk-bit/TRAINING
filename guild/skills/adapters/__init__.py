"""
Skill adapters - connect existing systems to Skill interface.

Adapters wrap existing components (SkillClient, PassiveModule) to provide
the unified Skill interface.
"""

from guild.skills.adapters.generator import GeneratorAdapter
from guild.skills.adapters.passive import PassiveAdapter

__all__ = ["GeneratorAdapter", "PassiveAdapter"]
