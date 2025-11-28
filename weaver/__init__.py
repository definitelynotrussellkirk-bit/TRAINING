"""
The Weaver - Daemon orchestrator

Keeps all threads of the tapestry alive:
- Training daemon (the heart)
- Tavern server (the face)
- VaultKeeper (the memory)
- Data flow (the fuel)

Usage:
    from weaver import Weaver

    w = Weaver()
    w.status()     # Check tapestry
    w.mend()       # Fix broken threads
    w.weave()      # Run as daemon
"""

from .weaver import Weaver, Thread

__all__ = ["Weaver", "Thread"]
