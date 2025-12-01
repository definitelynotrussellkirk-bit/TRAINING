"""
Fleet - Distributed node management for the Realm.

The Fleet system manages nodes across the training infrastructure:
- Node Agents run on each machine, reporting health and accepting commands
- Fleet Controller aggregates health data and triggers maintenance

Components:
    - agent.py: Node agent (runs on each node)
    - controller.py: Fleet controller (runs on control plane)
    - types.py: Shared data types

Usage:
    # Start agent on a node
    python3 -m fleet.agent --host-id 3090

    # Query fleet status from controller
    from fleet.controller import get_fleet_status
    status = get_fleet_status()
"""

from fleet.types import NodeHealth, NodeStatus, StorageHealth

__all__ = ["NodeHealth", "NodeStatus", "StorageHealth"]
