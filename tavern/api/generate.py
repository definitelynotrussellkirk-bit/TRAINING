"""
Data Generation API - Generate training data from the UI.

Handles:
- POST /api/generate - Generate training data and queue it

Uses SkillEngine to generate data and QueueAdapter to submit to queue.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from guild.skills import get_engine
from guild.integration.queue_adapter import get_queue_adapter

if TYPE_CHECKING:
    from tavern.server import TavernHandler

logger = logging.getLogger(__name__)


def serve_generate_post(handler: "TavernHandler"):
    """
    POST /api/generate - Generate training data and add to queue.

    Body:
        {
            "skill": "sy" | "bin" | null,  // Specific skill or null for curriculum
            "count": 1000,                   // Number of examples
            "level": 1                       // Optional level (defaults to current)
        }

    Returns:
        {
            "ok": true,
            "examples_generated": 1000,
            "skill": "sy",
            "level": 5,
            "queue_file": "skill_sy_1000_20251129.jsonl"
        }
    """
    try:
        # Parse request body
        content_length = int(handler.headers.get("Content-Length", 0))
        body = json.loads(handler.rfile.read(content_length).decode()) if content_length else {}

        count = int(body.get("count", 1000))
        skill_id = body.get("skill")
        level = body.get("level")

        # Validate count
        if count < 10:
            handler._send_json({
                "ok": False,
                "error": "Count must be at least 10",
            }, 400)
            return

        if count > 10000:
            handler._send_json({
                "ok": False,
                "error": "Count cannot exceed 10,000 per request",
            }, 400)
            return

        # Get skill engine
        engine = get_engine()
        available_skills = engine.list_skills()

        # If no skill specified, pick based on curriculum or first available
        if not skill_id:
            if available_skills:
                skill_id = available_skills[0]
            else:
                handler._send_json({
                    "ok": False,
                    "error": "No skills available for training",
                }, 400)
                return

        # Validate skill
        if skill_id not in available_skills:
            handler._send_json({
                "ok": False,
                "error": f"Unknown skill: {skill_id}. Available: {available_skills}",
            }, 400)
            return

        # Get skill
        skill = engine.get(skill_id)
        state = engine.get_state(skill_id)

        # Use current level if not specified
        if level is None:
            level = state.level

        # Generate training data
        logger.info(f"[Generate API] Generating {count} examples for {skill_id} L{level}")

        try:
            training_data = skill.generate_training_batch(
                level=level,
                count=count,
            )
        except Exception as e:
            logger.error(f"Failed to generate training data: {e}")
            handler._send_json({
                "ok": False,
                "error": f"Failed to generate data: {e}",
            }, 500)
            return

        if not training_data:
            handler._send_json({
                "ok": False,
                "error": "No training data generated",
            }, 500)
            return

        # Convert to JSONL format
        jsonl_lines = []
        for example in training_data:
            jsonl_lines.append(json.dumps(example))

        content = "\n".join(jsonl_lines)

        # Submit to queue
        queue = get_queue_adapter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"skill_{skill_id}_L{level}_{count}_{timestamp}.jsonl"

        result = queue.submit_content(
            content=content,
            filename=filename,
            priority="normal",
        )

        if not result.success:
            handler._send_json({
                "ok": False,
                "error": f"Failed to queue data: {result.error}",
            }, 500)
            return

        logger.info(f"[Generate API] Generated {len(training_data)} examples → {filename}")

        handler._send_json({
            "ok": True,
            "examples_generated": len(training_data),
            "skill": skill_id,
            "level": level,
            "queue_file": filename,
            "queue_path": str(result.data.queue_path) if result.data else None,
        })

    except json.JSONDecodeError:
        handler._send_json({
            "ok": False,
            "error": "Invalid JSON body",
        }, 400)
    except Exception as e:
        logger.error(f"Generate API error: {e}")
        handler._send_json({
            "ok": False,
            "error": str(e),
        }, 500)


def serve_generate_status(handler: "TavernHandler"):
    """
    GET /api/generate - Get available skills for generation.

    Returns:
        {
            "skills": [
                {"id": "sy", "name": "Syllacrostic", "level": 3, "max_level": 50},
                ...
            ]
        }
    """
    try:
        engine = get_engine()
        skills = []

        for skill_id in engine.list_skills():
            try:
                skill = engine.get(skill_id)
                state = engine.get_state(skill_id)
                skills.append({
                    "id": skill_id,
                    "name": skill.config.name,
                    "level": state.level,
                    "max_level": skill.config.max_level,
                })
            except Exception:
                continue

        handler._send_json({
            "ok": True,
            "skills": skills,
        })

    except Exception as e:
        logger.error(f"Generate status error: {e}")
        handler._send_json({
            "ok": False,
            "error": str(e),
        }, 500)
