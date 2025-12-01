"""
RealmStore JSON Schema - Single Source of Truth for Data Contracts

This schema defines the structure of realm_store.json and is used for:
1. Producer-side validation (Python - realm_store.py)
2. Consumer-side validation (JavaScript - game.js)
3. Documentation (what fields exist and their types)

CRITICAL: Any changes to this schema must be coordinated across:
- core/realm_store.py (producer)
- tavern/static/js/game.js (consumer)
- Any other consumers of /api/realm-state

Policy: POLICY 1 - Schema Validation at All Boundaries
"""

# JSON Schema v7 for RealmStore
REALM_STORE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RealmStore",
    "description": "Single source of truth for all Realm state",
    "type": "object",
    "required": ["state", "events", "timestamp"],
    "properties": {
        "state": {
            "type": "object",
            "description": "Current state of all Realm components",
            "required": ["training"],
            "properties": {
                "training": {
                    "type": "object",
                    "description": "Current training status",
                    "required": ["status", "step", "updated_at"],
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["idle", "training", "paused", "stopped"],
                            "description": "Current training state"
                        },
                        "step": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Current training step"
                        },
                        "total_steps": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Total steps in current training run"
                        },
                        "loss": {
                            "type": ["number", "null"],
                            "description": "Current training loss"
                        },
                        "learning_rate": {
                            "type": ["number", "null"],
                            "description": "Current learning rate"
                        },
                        "file": {
                            "type": ["string", "null"],
                            "description": "Current training file"
                        },
                        "speed": {
                            "type": ["number", "null"],
                            "description": "Training speed (steps/sec)"
                        },
                        "eta_seconds": {
                            "type": ["integer", "null"],
                            "description": "Estimated time to completion (seconds)"
                        },
                        "strain": {
                            "type": ["number", "null"],
                            "description": "Training strain (loss - floor)"
                        },
                        "started_at": {
                            "type": ["string", "null"],
                            "format": "date-time",
                            "description": "When training started (ISO 8601)"
                        },
                        "updated_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Last update timestamp (ISO 8601)"
                        }
                    }
                },
                "queue": {
                    "type": "object",
                    "description": "Job queue status",
                    "properties": {
                        "depth": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Total jobs in queue"
                        },
                        "high_priority": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "normal_priority": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "low_priority": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "status": {
                            "type": "string",
                            "enum": ["ok", "low", "empty", "stale"],
                            "description": "Queue health status"
                        },
                        "updated_at": {
                            "type": ["string", "null"],
                            "format": "date-time"
                        }
                    }
                },
                "workers": {
                    "type": "object",
                    "description": "Worker heartbeats",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "worker_id": {"type": "string"},
                            "role": {"type": "string"},
                            "status": {"type": "string"},
                            "device": {"type": ["string", "null"]},
                            "current_job": {"type": ["string", "null"]},
                            "last_heartbeat": {"type": "string", "format": "date-time"}
                        }
                    }
                },
                "hero": {
                    "type": "object",
                    "description": "Hero (model) state",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "level": {"type": "integer", "minimum": 0},
                        "xp": {"type": "integer", "minimum": 0},
                        "campaign_id": {"type": ["string", "null"]},
                        "current_skill": {"type": ["string", "null"]},
                        "current_skill_level": {"type": "integer", "minimum": 0},
                        "updated_at": {"type": ["string", "null"], "format": "date-time"}
                    }
                },
                "mode": {
                    "type": "string",
                    "description": "Current realm mode"
                },
                "mode_changed_at": {
                    "type": ["string", "null"],
                    "format": "date-time"
                },
                "mode_reason": {
                    "type": ["string", "null"]
                }
            }
        },
        "events": {
            "type": "array",
            "description": "Recent events (battle log)",
            "items": {
                "type": "object",
                "required": ["id", "timestamp", "kind", "message"],
                "properties": {
                    "id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "kind": {"type": "string"},
                    "channel": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["info", "success", "warning", "error"]
                    },
                    "message": {"type": "string"},
                    "icon": {"type": "string"},
                    "details": {"type": "object"}
                }
            }
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "When this state snapshot was generated"
        },
        "updated_at": {
            "type": "string",
            "format": "date-time",
            "description": "Last update time (alias for timestamp)"
        }
    }
}


# Minimal schema for critical fields only (fast validation)
REALM_STORE_MINIMAL_SCHEMA = {
    "type": "object",
    "required": ["state"],
    "properties": {
        "state": {
            "type": "object",
            "required": ["training"],
            "properties": {
                "training": {
                    "type": "object",
                    "required": ["status", "step", "updated_at"]
                }
            }
        }
    }
}


def validate_realm_store(data: dict, strict: bool = False) -> tuple[bool, list[str]]:
    """
    Validate RealmStore data against schema.

    Args:
        data: Data to validate
        strict: If True, use full schema. If False, use minimal schema.

    Returns:
        (is_valid, errors) where errors is a list of error messages

    Example:
        is_valid, errors = validate_realm_store(data)
        if not is_valid:
            raise ValueError(f"Invalid RealmStore: {errors}")
    """
    try:
        from jsonschema import validate, ValidationError, Draft7Validator
    except ImportError:
        # jsonschema not installed - skip validation
        return True, []

    schema = REALM_STORE_SCHEMA if strict else REALM_STORE_MINIMAL_SCHEMA
    validator = Draft7Validator(schema)

    errors = []
    for error in validator.iter_errors(data):
        # Format error path (e.g., "state.training.step")
        path = ".".join(str(p) for p in error.path) if error.path else "root"
        errors.append(f"{path}: {error.message}")

    is_valid = len(errors) == 0
    return is_valid, errors


def get_schema_version() -> str:
    """Get schema version for compatibility tracking."""
    return "1.0.0"


# JavaScript-compatible schema (for client-side validation)
def get_javascript_validator() -> str:
    """
    Generate JavaScript validation function.

    This can be embedded in game.js to validate data client-side.
    """
    return '''
/**
 * Validate RealmStore data (client-side)
 *
 * POLICY 1: Schema Validation at All Boundaries
 * This catches schema mismatches before they cause UI bugs.
 *
 * @param {object} data - Data from /api/realm-state
 * @returns {{valid: boolean, errors: string[]}}
 */
function validateRealmStore(data) {
    const errors = [];

    // Required: state
    if (!data.state) {
        errors.push("Missing required field: state");
        return {valid: false, errors};
    }

    // Required: state.training
    if (!data.state.training) {
        errors.push("Missing required field: state.training");
        return {valid: false, errors};
    }

    const training = data.state.training;

    // Required training fields
    if (typeof training.status !== 'string') {
        errors.push("state.training.status must be a string");
    }
    if (!['idle', 'training', 'paused', 'stopped'].includes(training.status)) {
        errors.push("state.training.status must be one of: idle, training, paused, stopped");
    }
    if (typeof training.step !== 'number') {
        errors.push("state.training.step must be a number");
    }
    if (!training.updated_at || typeof training.updated_at !== 'string') {
        errors.push("state.training.updated_at must be a string (ISO 8601 timestamp)");
    }

    return {
        valid: errors.length === 0,
        errors: errors
    };
}
'''


if __name__ == "__main__":
    import json

    # Example usage
    example_data = {
        "state": {
            "training": {
                "status": "training",
                "step": 100,
                "total_steps": 1000,
                "loss": 0.5,
                "updated_at": "2025-11-30T12:00:00"
            }
        },
        "events": [],
        "timestamp": "2025-11-30T12:00:00"
    }

    is_valid, errors = validate_realm_store(example_data, strict=False)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    # Test invalid data
    invalid_data = {
        "state": {
            "training": {
                "status": "invalid_status",  # Wrong enum value
                "step": "not_a_number",  # Wrong type
                # Missing updated_at
            }
        }
    }

    is_valid, errors = validate_realm_store(invalid_data, strict=False)
    print(f"\nInvalid data test:")
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
