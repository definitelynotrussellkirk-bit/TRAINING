"""
Safety and watchdog modules.

Contains:
- daemon_watchdog.py: Auto-restart daemon
- anti_stuck_monitor.py: Detect hangs
- crash_detector.py: Crash analysis
- comprehensive_health_check.py: System health

NEW: RPG-themed wrappers available in sentinels/:
- Guardian: Daemon watchdog (auto-restart)
- HealthInspector: Comprehensive health checks

Usage:
    # Traditional
    from safety.daemon_watchdog import DaemonWatchdog
    from safety.comprehensive_health_check import run_health_check

    # RPG-themed (new)
    from sentinels import Guardian, HealthInspector
"""

# Note: sentinels/ is independent, no circular import issues
try:
    from sentinels import (
        # Types
        SentinelState,
        ThreatLevel,
        HealthDomain,
        HealthReport,
        PatrolReport,
        AlertRecord,
        # Guardian
        Guardian,
        get_guardian,
        # Health Inspector
        HealthInspector,
        get_health_inspector,
    )
except ImportError:
    # sentinels not yet available
    pass
