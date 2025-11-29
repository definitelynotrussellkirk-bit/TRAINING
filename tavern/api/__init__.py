"""
Tavern API Modules

Extracted from server.py for better organization.
Each module handles a domain of endpoints.
"""

from tavern.api import heroes
from tavern.api import analysis

__all__ = ["heroes", "analysis"]
