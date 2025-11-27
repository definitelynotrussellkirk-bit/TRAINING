"""
Sentinels - Guardians of the training realm.

The Sentinels are an order of protectors who watch over the training
systems, detecting threats and responding to emergencies:

    Guardian        - Daemon watchdog (auto-restart)
    HealthInspector - Comprehensive health checks
    Scout           - Stuck training detection
    Healer          - Crash recovery

RPG Mapping:
    daemon_watchdog.py        → Guardian
    comprehensive_health.py   → HealthInspector
    anti_stuck_monitor.py     → Scout
    crash_detector.py         → Healer

Quick Start:
    from sentinels import Guardian, HealthInspector

    # Check system health
    inspector = HealthInspector(base_dir)
    report = inspector.full_patrol()
    if not report.is_all_clear:
        print(f"Threats detected: {len(report.threats_detected)}")

    # Guardian watches daemon
    guardian = Guardian(base_dir)
    if not guardian.is_daemon_alive():
        guardian.revive_daemon()
"""

__version__ = "0.1.0"

# Types
from sentinels.types import (
    SentinelState,
    ThreatLevel,
    HealthDomain,
    HealthReport,
    PatrolReport,
    AlertRecord,
)

# Guardian (daemon watchdog)
from sentinels.guardian import (
    Guardian,
    get_guardian,
)

# Health Inspector
from sentinels.health_inspector import (
    HealthInspector,
    get_health_inspector,
)


__all__ = [
    # Types
    "SentinelState",
    "ThreatLevel",
    "HealthDomain",
    "HealthReport",
    "PatrolReport",
    "AlertRecord",
    # Guardian
    "Guardian",
    "get_guardian",
    # Health Inspector
    "HealthInspector",
    "get_health_inspector",
]


# =============================================================================
# RPG TERMINOLOGY GUIDE
# =============================================================================

"""
SENTINELS GLOSSARY
==================

The Sentinels use RPG terminology for protection and monitoring:

SENTINEL TYPES
--------------
Guardian        = Daemon watchdog (auto-restart on failure)
HealthInspector = Comprehensive health checker
Scout           = Stuck training detector
Healer          = Crash recovery system

STATES
------
Dormant         = Not active
Patrolling      = Actively monitoring
Alerted         = Detected issue
Responding      = Taking action
Resting         = Temporarily paused

THREAT LEVELS
-------------
Clear           = No issues (green)
Minor           = Small concern (yellow)
Moderate        = Needs attention (orange)
Severe          = Urgent action needed (red)
Critical        = System at risk (flashing red)

HEALTH DOMAINS
--------------
Daemon          = Training daemon process
GPU             = GPU resources (VRAM, temp)
Disk            = Disk space
Queue           = Training queue
Model           = Model integrity
Config          = Configuration validity
Network         = Network/API connectivity

ACTIONS
-------
Patrol          = Check all systems
Inspect         = Check specific domain
Revive          = Restart failed daemon
Alert           = Raise notification
Respond         = Take corrective action
"""
