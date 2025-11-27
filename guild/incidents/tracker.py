"""Incident tracking and management."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from guild.types import Severity, generate_id
from guild.incidents.types import (
    Incident,
    IncidentCategory,
    IncidentStatus,
    IncidentRule,
)


logger = logging.getLogger(__name__)


class IncidentTracker:
    """
    Tracks and manages incidents throughout training.

    Responsibilities:
    - Create incidents (manually or from rules)
    - Track incident lifecycle (open -> investigating -> resolved/wontfix)
    - Persist incidents to JSON
    - Query incidents by status, category, severity, run
    - Generate statistics and summaries
    """

    def __init__(
        self,
        state_dir: Path,
        state_file: str = "incidents.json",
        history_limit: int = 500,
    ):
        """
        Initialize incident tracker.

        Args:
            state_dir: Directory for state files
            state_file: Name of incidents file
            history_limit: Max resolved incidents to keep
        """
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / state_file
        self.history_limit = history_limit

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._incidents: Dict[str, Incident] = {}
        self._metadata: Dict[str, Any] = {}

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if not self.state_file.exists():
            self._metadata = {
                "created_at": datetime.now().isoformat(),
                "total_created": 0,
                "total_resolved": 0,
            }
            return

        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse incidents file: {e}")
            return

        self._metadata = data.get("metadata", {})

        for inc_id, inc_data in data.get("incidents", {}).items():
            try:
                self._incidents[inc_id] = Incident.from_dict(inc_data)
            except Exception as e:
                logger.warning(f"Failed to load incident '{inc_id}': {e}")

        logger.debug(f"Loaded {len(self._incidents)} incidents")

    def _save_state(self):
        """Save state to disk."""
        data = {
            "metadata": {
                **self._metadata,
                "last_updated": datetime.now().isoformat(),
            },
            "incidents": {
                inc_id: inc.to_dict()
                for inc_id, inc in self._incidents.items()
            },
        }

        # Atomic write
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(self.state_file)

        logger.debug("Incidents saved")

    def _trim_history(self):
        """Trim old resolved incidents if over limit."""
        resolved = [
            inc for inc in self._incidents.values()
            if inc.status in [IncidentStatus.RESOLVED, IncidentStatus.WONTFIX]
        ]

        if len(resolved) <= self.history_limit:
            return

        # Sort by resolution time, oldest first
        resolved.sort(key=lambda i: i.resolved_at or datetime.min)

        # Remove oldest
        to_remove = len(resolved) - self.history_limit
        for inc in resolved[:to_remove]:
            del self._incidents[inc.id]
            logger.debug(f"Trimmed old incident: {inc.id}")

    # --- Incident Creation ---

    def create_incident(
        self,
        category: IncidentCategory,
        severity: Severity,
        title: str,
        description: str,
        detected_at_step: int,
        run_id: Optional[str] = None,
        quest_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        incident_id: Optional[str] = None,
    ) -> Incident:
        """
        Create a new incident.

        Args:
            category: Incident category
            severity: Incident severity
            title: Short title
            description: Detailed description
            detected_at_step: Training step where detected
            run_id: Associated run ID
            quest_id: Associated quest ID
            facility_id: Associated facility ID
            context: Additional context data
            incident_id: Optional explicit ID

        Returns:
            Created Incident
        """
        if incident_id is None:
            incident_id = generate_id("inc")

        if incident_id in self._incidents:
            raise ValueError(f"Incident already exists: {incident_id}")

        incident = Incident(
            id=incident_id,
            category=category,
            severity=severity,
            title=title,
            description=description,
            detected_at_step=detected_at_step,
            detected_at_time=datetime.now(),
            run_id=run_id,
            quest_id=quest_id,
            facility_id=facility_id,
            context=context or {},
            status=IncidentStatus.OPEN,
        )

        self._incidents[incident_id] = incident
        self._metadata["total_created"] = self._metadata.get("total_created", 0) + 1
        self._save_state()

        logger.warning(
            f"[{severity.value.upper()}] Incident created: {incident_id} - {title}"
        )
        return incident

    def create_from_rule(
        self,
        rule: IncidentRule,
        detected_at_step: int,
        template_vars: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        quest_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Incident:
        """
        Create an incident from a rule definition.

        Args:
            rule: The incident rule that triggered
            detected_at_step: Training step where detected
            template_vars: Variables for title/description templates
            run_id: Associated run ID
            quest_id: Associated quest ID
            facility_id: Associated facility ID
            context: Additional context

        Returns:
            Created Incident
        """
        vars_dict = template_vars or {}

        # Format templates
        title = rule.title_template.format(**vars_dict) if rule.title_template else rule.name
        description = rule.description_template.format(**vars_dict) if rule.description_template else ""

        # Add rule info to context
        full_context = {
            "rule_id": rule.id,
            "detector_type": rule.detector_type,
            **(context or {}),
        }

        incident = self.create_incident(
            category=rule.category,
            severity=rule.severity,
            title=title,
            description=description,
            detected_at_step=detected_at_step,
            run_id=run_id,
            quest_id=quest_id,
            facility_id=facility_id,
            context=full_context,
        )

        # Add RPG flavor if defined
        if rule.rpg_name_template:
            incident.rpg_name = rule.rpg_name_template.format(**vars_dict)

        self._save_state()
        return incident

    # --- Incident Lifecycle ---

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        return self._incidents.get(incident_id)

    def get_incident_or_raise(self, incident_id: str) -> Incident:
        """Get an incident by ID, raising if not found."""
        incident = self.get_incident(incident_id)
        if incident is None:
            raise KeyError(f"Unknown incident: {incident_id}")
        return incident

    def start_investigation(self, incident_id: str) -> Incident:
        """
        Mark an incident as under investigation.

        Transitions: OPEN -> INVESTIGATING
        """
        incident = self.get_incident_or_raise(incident_id)

        if incident.status != IncidentStatus.OPEN:
            raise ValueError(
                f"Cannot investigate incident in status {incident.status.value}. "
                f"Must be OPEN."
            )

        incident.status = IncidentStatus.INVESTIGATING
        self._save_state()

        logger.info(f"Investigating incident: {incident_id}")
        return incident

    def resolve(
        self,
        incident_id: str,
        resolution: str,
    ) -> Incident:
        """
        Resolve an incident.

        Transitions: OPEN|INVESTIGATING -> RESOLVED
        """
        incident = self.get_incident_or_raise(incident_id)

        if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.WONTFIX]:
            raise ValueError(
                f"Incident already closed with status {incident.status.value}"
            )

        incident.status = IncidentStatus.RESOLVED
        incident.resolution = resolution
        incident.resolved_at = datetime.now()

        self._metadata["total_resolved"] = self._metadata.get("total_resolved", 0) + 1
        self._save_state()
        self._trim_history()

        logger.info(f"Resolved incident: {incident_id} - {resolution}")
        return incident

    def wontfix(
        self,
        incident_id: str,
        reason: str,
    ) -> Incident:
        """
        Mark an incident as won't fix.

        Transitions: OPEN|INVESTIGATING -> WONTFIX
        """
        incident = self.get_incident_or_raise(incident_id)

        if incident.status in [IncidentStatus.RESOLVED, IncidentStatus.WONTFIX]:
            raise ValueError(
                f"Incident already closed with status {incident.status.value}"
            )

        incident.status = IncidentStatus.WONTFIX
        incident.resolution = reason
        incident.resolved_at = datetime.now()

        self._save_state()
        self._trim_history()

        logger.info(f"Won't fix incident: {incident_id} - {reason}")
        return incident

    def reopen(self, incident_id: str) -> Incident:
        """
        Reopen a closed incident.

        Transitions: RESOLVED|WONTFIX -> OPEN
        """
        incident = self.get_incident_or_raise(incident_id)

        if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.WONTFIX]:
            raise ValueError(
                f"Cannot reopen incident in status {incident.status.value}"
            )

        incident.status = IncidentStatus.OPEN
        incident.resolution = None
        incident.resolved_at = None

        self._save_state()

        logger.info(f"Reopened incident: {incident_id}")
        return incident

    def add_context(
        self,
        incident_id: str,
        key: str,
        value: Any,
    ) -> Incident:
        """Add context data to an incident."""
        incident = self.get_incident_or_raise(incident_id)
        incident.context[key] = value
        self._save_state()
        return incident

    # --- Queries ---

    def list_incidents(self) -> list[str]:
        """List all incident IDs."""
        return list(self._incidents.keys())

    def list_open(self) -> list[Incident]:
        """List all open incidents."""
        return [
            inc for inc in self._incidents.values()
            if inc.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]

    def list_by_status(self, status: IncidentStatus) -> list[Incident]:
        """List incidents by status."""
        return [
            inc for inc in self._incidents.values()
            if inc.status == status
        ]

    def list_by_category(self, category: IncidentCategory) -> list[Incident]:
        """List incidents by category."""
        return [
            inc for inc in self._incidents.values()
            if inc.category == category
        ]

    def list_by_severity(self, severity: Severity) -> list[Incident]:
        """List incidents by severity."""
        return [
            inc for inc in self._incidents.values()
            if inc.severity == severity
        ]

    def list_by_run(self, run_id: str) -> list[Incident]:
        """List incidents associated with a run."""
        return [
            inc for inc in self._incidents.values()
            if inc.run_id == run_id
        ]

    def list_critical(self) -> list[Incident]:
        """List all critical severity incidents that are open."""
        return [
            inc for inc in self._incidents.values()
            if inc.severity == Severity.CRITICAL
            and inc.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]

    def search(
        self,
        status: Optional[IncidentStatus] = None,
        category: Optional[IncidentCategory] = None,
        severity: Optional[Severity] = None,
        run_id: Optional[str] = None,
        since_step: Optional[int] = None,
        title_contains: Optional[str] = None,
    ) -> list[Incident]:
        """
        Search incidents by multiple criteria.

        Args:
            status: Filter by status
            category: Filter by category
            severity: Filter by severity
            run_id: Filter by run
            since_step: Filter by detection step (>= since_step)
            title_contains: Filter by title substring

        Returns:
            List of matching incidents
        """
        results = list(self._incidents.values())

        if status is not None:
            results = [i for i in results if i.status == status]

        if category is not None:
            results = [i for i in results if i.category == category]

        if severity is not None:
            results = [i for i in results if i.severity == severity]

        if run_id is not None:
            results = [i for i in results if i.run_id == run_id]

        if since_step is not None:
            results = [i for i in results if i.detected_at_step >= since_step]

        if title_contains:
            needle = title_contains.lower()
            results = [i for i in results if needle in i.title.lower()]

        return results

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get incident statistics."""
        by_status = {}
        for status in IncidentStatus:
            count = len(self.list_by_status(status))
            if count > 0:
                by_status[status.value] = count

        by_category = {}
        for category in IncidentCategory:
            count = len(self.list_by_category(category))
            if count > 0:
                by_category[category.value] = count

        by_severity = {}
        for severity in Severity:
            count = len(self.list_by_severity(severity))
            if count > 0:
                by_severity[severity.value] = count

        open_incidents = self.list_open()
        critical = self.list_critical()

        return {
            "total": len(self._incidents),
            "total_created": self._metadata.get("total_created", 0),
            "total_resolved": self._metadata.get("total_resolved", 0),
            "open_count": len(open_incidents),
            "critical_count": len(critical),
            "by_status": by_status,
            "by_category": by_category,
            "by_severity": by_severity,
        }

    def get_summary(self) -> str:
        """Get a text summary of current incident status."""
        stats = self.get_stats()
        open_count = stats["open_count"]
        critical_count = stats["critical_count"]

        if critical_count > 0:
            return f"⚠️ {critical_count} CRITICAL, {open_count} open incidents"
        elif open_count > 0:
            return f"📋 {open_count} open incidents"
        else:
            return "✅ No open incidents"

    def delete_incident(self, incident_id: str) -> bool:
        """Delete an incident (only resolved/wontfix)."""
        incident = self.get_incident(incident_id)
        if incident is None:
            return False

        if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.WONTFIX]:
            raise ValueError(
                f"Cannot delete incident in status {incident.status.value}. "
                f"Must be RESOLVED or WONTFIX."
            )

        del self._incidents[incident_id]
        self._save_state()

        logger.info(f"Deleted incident: {incident_id}")
        return True


# Global tracker
_tracker: Optional[IncidentTracker] = None


def init_incident_tracker(
    state_dir: Path,
    state_file: str = "incidents.json",
) -> IncidentTracker:
    """Initialize the global incident tracker."""
    global _tracker
    _tracker = IncidentTracker(state_dir, state_file)
    return _tracker


def get_incident_tracker() -> IncidentTracker:
    """Get the global incident tracker."""
    global _tracker
    if _tracker is None:
        raise RuntimeError(
            "Incident tracker not initialized. "
            "Call init_incident_tracker() first."
        )
    return _tracker


def reset_incident_tracker():
    """Reset the global incident tracker (for testing)."""
    global _tracker
    _tracker = None


# Convenience functions

def create_incident(
    category: IncidentCategory,
    severity: Severity,
    title: str,
    description: str,
    detected_at_step: int,
    **kwargs,
) -> Incident:
    """Create a new incident."""
    return get_incident_tracker().create_incident(
        category=category,
        severity=severity,
        title=title,
        description=description,
        detected_at_step=detected_at_step,
        **kwargs,
    )


def get_incident(incident_id: str) -> Optional[Incident]:
    """Get an incident by ID."""
    return get_incident_tracker().get_incident(incident_id)


def resolve_incident(incident_id: str, resolution: str) -> Incident:
    """Resolve an incident."""
    return get_incident_tracker().resolve(incident_id, resolution)


def list_open_incidents() -> list[Incident]:
    """List all open incidents."""
    return get_incident_tracker().list_open()


def get_incident_stats() -> Dict[str, Any]:
    """Get incident statistics."""
    return get_incident_tracker().get_stats()
