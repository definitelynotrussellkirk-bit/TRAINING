"""
Realm State Service - Network-based single source of truth.

This replaces the file-based realm_store.json with an HTTP service backed by SQLite.

Architecture:
    PRODUCERS (POST) -> RealmService (SQLite) -> CONSUMERS (GET)

Producers: Training daemon, eval workers, job system
Consumers: Tavern UI, CLI tools, monitoring, external APIs

Usage:
    # Start service
    python3 -m realm.server --port 8866

    # Producer (from core/realm_store.py client)
    from core.realm_store import update_training
    update_training(status="training", step=100, loss=0.5)

    # Consumer (HTTP)
    curl http://localhost:8866/api/state
"""

from .client import RealmClient

__all__ = ["RealmClient"]
