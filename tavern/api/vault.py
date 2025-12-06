"""
Vault & Ledger API

Extracted from tavern/server.py for better organization.
Handles:
- /api/ledger - Checkpoint history and stats
- /api/ledger/summary - Aggregate stats
- /api/ledger/best - Best checkpoints by metric
- /api/checkpoint/{step} - Checkpoint details
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core import paths

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_ledger_list(handler: "TavernHandler", query: dict):
    """
    GET /api/ledger - List all checkpoints with stats.

    Query params:
    - limit: Max checkpoints to return (default 50)
    - skill: Filter by skill name
    - include_base: Include base model as step 0 (default true)
    """
    import re

    try:
        from core.checkpoint_ledger import get_ledger

        base_dir = paths.get_base_dir()
        ledger = get_ledger()

        limit = int(query.get("limit", [50])[0])
        skill = query.get("skill", [None])[0]
        include_base = query.get("include_base", ["true"])[0].lower() == "true"

        if skill:
            records = ledger.list_by_skill(skill)[:limit]
        else:
            records = ledger.list_all(limit=limit)

        checkpoints = []

        # Get active campaign
        active_hero_id = None
        active_campaign_id = None
        try:
            from guild.campaigns import get_active_campaign
            active = get_active_campaign(base_dir)
            if active:
                active_hero_id = active.hero_id
                active_campaign_id = active.id
        except Exception:
            pass

        # Get eval counts per checkpoint
        eval_counts = {}
        try:
            from core.evaluation_ledger import get_ledger as get_eval_ledger
            eval_ledger = get_eval_ledger()
            for record in eval_ledger.list_all():
                step = record.get("checkpoint_step", 0)
                eval_counts[step] = eval_counts.get(step, 0) + 1
        except Exception:
            pass

        # Add base model as step 0 if requested
        if include_base and not skill:
            base_model_path = base_dir / "models" / "Qwen3-0.6B"
            if base_model_path.exists():
                try:
                    size_bytes = sum(
                        f.stat().st_size for f in base_model_path.rglob("*") if f.is_file()
                    )
                    size_gb = round(size_bytes / (1024**3), 2)

                    config_file = base_model_path / "config.json"
                    model_info = {}
                    if config_file.exists():
                        with open(config_file) as f:
                            model_info = json.load(f)

                    checkpoints.append({
                        "step": 0,
                        "canonical_name": "base-model",
                        "display_name": "Qwen3-0.6B (Base)",
                        "timestamp": datetime.fromtimestamp(base_model_path.stat().st_mtime).isoformat(),
                        "train_loss": None,
                        "val_loss": None,
                        "learning_rate": None,
                        "skill_name": None,
                        "skill_level": None,
                        "size_gb": size_gb,
                        "age_hours": None,
                        "path": str(base_model_path),
                        "is_base": True,
                        "model_type": model_info.get("model_type", "qwen2"),
                        "hidden_size": model_info.get("hidden_size"),
                        "num_layers": model_info.get("num_hidden_layers"),
                        "vocab_size": model_info.get("vocab_size"),
                        "hero_id": None,
                        "campaign_id": None,
                        "is_local": True,
                        "locations": [],
                        "eval_count": 0,
                        "is_active_campaign": False,
                    })
                except Exception as e:
                    logger.warning(f"Failed to get base model info: {e}")

        for r in records:
            # Extract hero/campaign from path
            hero_id = None
            campaign_id = None
            if r.path:
                match = re.search(r'campaigns/([^/]+)/([^/]+)/', r.path)
                if match:
                    hero_id = match.group(1)
                    campaign_id = match.group(2)

            # Check if path exists locally
            is_local = r.path and Path(r.path).exists()

            # Get locations
            locations = r.locations if r.locations else []

            checkpoints.append({
                "step": r.step,
                "canonical_name": r.canonical_name,
                "timestamp": r.timestamp,
                "train_loss": r.train_loss,
                "val_loss": r.val_loss,
                "learning_rate": r.learning_rate,
                "skill_name": r.skill_name,
                "skill_level": r.skill_level,
                "size_gb": r.size_gb,
                "age_hours": round(r.age_hours, 1) if r.age_hours else None,
                "path": r.path,
                "is_base": False,
                "hero_id": hero_id,
                "campaign_id": campaign_id,
                "is_local": is_local,
                "locations": locations,
                "eval_count": eval_counts.get(r.step, 0),
                "is_active_campaign": (hero_id == active_hero_id and campaign_id == active_campaign_id),
            })

        handler._send_json({
            "checkpoints": checkpoints,
            "count": len(checkpoints),
            "active_hero_id": active_hero_id,
            "active_campaign_id": active_campaign_id,
        })

    except ImportError:
        handler._send_json({"checkpoints": [], "count": 0, "error": "Ledger not initialized"})
    except Exception as e:
        logger.error(f"Ledger list error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_ledger_summary(handler: "TavernHandler"):
    """
    GET /api/ledger/summary - Aggregate ledger stats.
    """
    try:
        from core.checkpoint_ledger import get_ledger

        ledger = get_ledger()
        summary = ledger.get_summary()
        handler._send_json(summary)

    except ImportError:
        handler._send_json({"error": "Ledger not initialized"}, 500)
    except Exception as e:
        logger.error(f"Ledger summary error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_ledger_best(handler: "TavernHandler", query: dict):
    """
    GET /api/ledger/best - Best checkpoints by metric.

    Query params:
    - metric: Metric to optimize (default "train_loss")
    - n: Number of results (default 5)
    """
    try:
        from core.checkpoint_ledger import get_ledger

        ledger = get_ledger()
        metric = query.get("metric", ["train_loss"])[0]
        n = int(query.get("n", [5])[0])

        best = ledger.get_best(metric=metric, n=n)

        handler._send_json({
            "metric": metric,
            "checkpoints": [
                {
                    "step": r.step,
                    "canonical_name": r.canonical_name,
                    metric: getattr(r, metric, None),
                    "timestamp": r.timestamp,
                }
                for r in best
            ],
        })

    except ImportError:
        handler._send_json({"error": "Ledger not initialized"}, 500)
    except Exception as e:
        logger.error(f"Ledger best error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_ledger_checkpoint(handler: "TavernHandler", step_str: str):
    """
    GET /api/ledger/checkpoint/{step} - Get specific checkpoint from ledger.
    """
    try:
        from core.checkpoint_ledger import get_ledger

        ledger = get_ledger()
        step = int(step_str)
        record = ledger.get(step)

        if record:
            handler._send_json({
                "step": record.step,
                "canonical_name": record.canonical_name,
                "timestamp": record.timestamp,
                "train_loss": record.train_loss,
                "val_loss": record.val_loss,
                "learning_rate": record.learning_rate,
                "skill_name": record.skill_name,
                "skill_level": record.skill_level,
                "size_gb": record.size_gb,
                "age_hours": round(record.age_hours, 1),
                "path": record.path,
            })
        else:
            handler._send_json({"error": f"Checkpoint {step} not found"}, 404)

    except ImportError:
        handler._send_json({"error": "Ledger not initialized"}, 500)
    except ValueError:
        handler._send_json({"error": f"Invalid step: {step_str}"}, 400)
    except Exception as e:
        logger.error(f"Ledger checkpoint error: {e}")
        handler._send_json({"error": str(e)}, 500)


def serve_checkpoint_data(handler: "TavernHandler", step_str: str):
    """
    GET /api/checkpoint/{step} - Full checkpoint data including training info.

    Combines ledger data with any additional checkpoint metadata.
    """
    try:
        from core.checkpoint_ledger import get_ledger

        base_dir = paths.get_base_dir()
        ledger = get_ledger()
        step = int(step_str)

        # Get ledger record
        record = ledger.get(step)
        if not record:
            handler._send_json({"error": f"Checkpoint {step} not found in ledger"}, 404)
            return

        # Build full response
        data = {
            "step": record.step,
            "canonical_name": record.canonical_name,
            "timestamp": record.timestamp,
            "metrics": {
                "train_loss": record.train_loss,
                "val_loss": record.val_loss,
                "learning_rate": record.learning_rate,
            },
            "skill": {
                "name": record.skill_name,
                "level": record.skill_level,
            },
            "storage": {
                "size_gb": record.size_gb,
                "age_hours": round(record.age_hours, 1),
                "path": record.path,
            },
        }

        # Try to load training_state.json if exists
        checkpoint_path = Path(record.path) if record.path else None
        if checkpoint_path and checkpoint_path.exists():
            state_file = checkpoint_path / "training_state.json"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        data["training_state"] = json.load(f)
                except Exception as e:
                    logger.debug(f"Could not load training_state.json: {e}")

        handler._send_json(data)

    except ImportError:
        handler._send_json({"error": "Ledger not initialized"}, 500)
    except ValueError:
        handler._send_json({"error": f"Invalid step: {step_str}"}, 400)
    except Exception as e:
        logger.error(f"Checkpoint data error: {e}")
        handler._send_json({"error": str(e)}, 500)
