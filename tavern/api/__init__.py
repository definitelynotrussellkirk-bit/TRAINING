"""
Tavern API Modules

Extracted from server.py for better organization.
Each module handles a domain of endpoints.
"""

from tavern.api import heroes
from tavern.api import analysis
from tavern.api import skills
from tavern.api import vault
from tavern.api import jobs

__all__ = ["heroes", "analysis", "skills", "vault", "jobs"]
