"""
Lore Dictionary - Centralized tooltips and UI copy for the Tavern.

This module provides consistent, RPG-flavored explanations for training concepts.
Use these for tooltips, help text, and UI labels throughout the Tavern.

The RPG flavor is functional - it helps visual/story-oriented thinking
map onto technical concepts. Every metaphor should clarify, not obscure.

Usage:
    from tavern.lore import get_lore, get_tooltip, LORE

    # Get full lore entry
    entry = get_lore("training.loss")
    print(entry["label"])    # "Practice Strain"
    print(entry["tooltip"])  # "How hard the hero is working..."

    # Quick tooltip lookup
    tooltip = get_tooltip("validation.loss")
"""

from typing import Any, Dict, Optional

# =============================================================================
# LORE DICTIONARY
# =============================================================================

LORE: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # TRAINING METRICS
    # =========================================================================
    "training.loss": {
        "label": "Strain",
        "short": "Training difficulty",
        "tooltip": (
            "How hard the hero struggles against training challenges. "
            "Lower strain = easier learning. High strain means difficult material "
            "or the hero is confused. Spikes indicate challenging content; "
            "plateaus mean the hero has absorbed what these drills can teach."
        ),
        "technical": "Cross-entropy loss on training batch",
        "icon": "💪",
    },
    "validation.loss": {
        "label": "Trial Strain",
        "short": "Validation difficulty",
        "tooltip": (
            "Strain measured on unseen challenges (not training material). "
            "This shows true capability, not memorization. "
            "If training strain drops but trial strain rises, "
            "the hero is memorizing instead of learning."
        ),
        "technical": "Cross-entropy loss on held-out validation set",
        "icon": "🎯",
    },
    "overfitting": {
        "label": "Over-Drilling",
        "short": "Memorization vs learning",
        "tooltip": (
            "The hero has memorized the training patterns too exactly. "
            "They ace familiar drills but stumble on new challenges. "
            "Solution: new training material, early stopping, or regularization."
        ),
        "technical": "Train loss decreasing while validation loss increases",
        "warning": True,
    },
    "perplexity": {
        "label": "Clarity",
        "short": "Inverse confusion",
        "tooltip": (
            "How clearly the hero understands language. "
            "Higher clarity = less confused, sharper prediction. "
            "Low clarity means the hero is uncertain about what comes next."
        ),
        "technical": "exp(loss) - lower is better, displayed as 1/perplexity",
        "icon": "💎",
    },
    "learning_rate": {
        "label": "Training Intensity",
        "short": "Step size",
        "tooltip": (
            "How aggressively the hero updates their understanding each step. "
            "High intensity = fast but risky learning. "
            "Low intensity = slow but stable refinement."
        ),
        "technical": "Optimizer learning rate",
        "icon": "⚡",
    },
    "gradient_norm": {
        "label": "Momentum",
        "short": "Update magnitude",
        "tooltip": (
            "Size of each learning update. "
            "Extremely high = chaotic learning (may need clipping). "
            "Near zero = the hero has stopped learning from this material."
        ),
        "technical": "L2 norm of gradients before clipping",
    },

    # =========================================================================
    # SKILLS & EVALUATION
    # =========================================================================
    "skill.level": {
        "label": "Skill Level",
        "short": "Mastery tier",
        "tooltip": (
            "Current tier of mastery. Higher levels have harder challenges. "
            "Level up requires passing trials (evaluations) at 80%+ accuracy."
        ),
        "technical": "Curriculum level for this skill",
        "icon": "📊",
    },
    "skill.accuracy": {
        "label": "Accuracy",
        "short": "Success rate",
        "tooltip": (
            "Success rate on recent challenges. "
            "Rolling average over last 100 attempts. "
            "80%+ accuracy qualifies for level-up trials."
        ),
        "technical": "Correct / Total over evaluation window",
        "icon": "🎯",
    },
    "skill.trial": {
        "label": "Trial",
        "short": "Level-up test",
        "tooltip": (
            "A formal evaluation to prove readiness for the next level. "
            "The guild requires proof of mastery before advancement. "
            "Trials use unseen problems to prevent memorization."
        ),
        "technical": "Held-out evaluation set, accuracy threshold required",
    },
    "skill.primitive": {
        "label": "Primitive",
        "short": "Atomic skill",
        "tooltip": (
            "A specific, testable micro-skill within a discipline. "
            "Example: 'binary addition with carry' is a primitive of Binary Alchemy. "
            "Tracking primitives reveals exactly where the hero struggles."
        ),
        "technical": "Tagged problem type for fine-grained accuracy tracking",
    },
    "skill.regression": {
        "label": "Regression",
        "short": "Skill degradation",
        "tooltip": (
            "The hero has gotten worse at something they knew before. "
            "Common when training on new skills causes forgetting. "
            "May need refresher training on affected primitives."
        ),
        "technical": "Accuracy drop >5% from peak on a skill/primitive",
        "warning": True,
        "icon": "📉",
    },

    # =========================================================================
    # VAULT & CHECKPOINTS
    # =========================================================================
    "checkpoint": {
        "label": "Checkpoint",
        "short": "Saved state",
        "tooltip": (
            "A snapshot of the hero at a specific moment in training. "
            "Can be restored if later training goes wrong. "
            "Named by step number and timestamp."
        ),
        "technical": "Model weights + optimizer state + config",
        "icon": "💾",
    },
    "vault.hot": {
        "label": "Hot Vault",
        "short": "Active storage",
        "tooltip": (
            "Fast local storage for recent checkpoints. "
            "Quick to access but limited space. "
            "Old checkpoints move to warm/cold storage."
        ),
        "technical": "NVMe SSD, ~150GB capacity",
    },
    "vault.warm": {
        "label": "Warm Vault",
        "short": "Archive storage",
        "tooltip": (
            "NAS storage for important checkpoints. "
            "Slower access but much more space. "
            "Checkpoints here are verified and indexed."
        ),
        "technical": "Synology NAS, ~500GB allocated",
    },
    "vault.cold": {
        "label": "Deep Vault",
        "short": "Cold archive",
        "tooltip": (
            "Long-term archival storage. "
            "Compressed and rarely accessed. "
            "For historical records and disaster recovery."
        ),
        "technical": "Compressed archives, unlimited retention",
    },
    "checkpoint.promote": {
        "label": "Promote",
        "short": "Mark as important",
        "tooltip": (
            "Flag a checkpoint as significant and worth keeping. "
            "Promoted checkpoints skip automatic cleanup. "
            "Use for milestones, best performers, or before risky experiments."
        ),
        "technical": "Set protected flag in checkpoint metadata",
    },

    # =========================================================================
    # JOBS & INFRASTRUCTURE
    # =========================================================================
    "job.eval": {
        "label": "Evaluation Job",
        "short": "Skill test",
        "tooltip": (
            "Runs the hero through a set of challenges for a specific skill. "
            "Results update skill accuracy and primitive tracking. "
            "Requires inference server access."
        ),
        "technical": "Generate problems, run inference, score results",
    },
    "job.sparring": {
        "label": "Sparring Job",
        "short": "Self-correction training",
        "tooltip": (
            "The hero practices on problems they got wrong. "
            "For each mistake, generates training examples: "
            "identify wrong, correct it, confirm right. "
            "High priority - sparring data is checkpoint-specific."
        ),
        "technical": "Error mining + correction example generation",
    },
    "job.data_gen": {
        "label": "Data Generation",
        "short": "Create training material",
        "tooltip": (
            "Generate new training problems for a skill. "
            "Uses skill-specific generators. "
            "Can target weak primitives for focused improvement."
        ),
        "technical": "Run skill API generator, format as training JSONL",
    },
    "device.trainer": {
        "label": "Training Grounds",
        "short": "Training server",
        "tooltip": (
            "The main training server (RTX 4090). "
            "Where the hero learns and grows. "
            "Runs training daemon, queue manager, and Tavern."
        ),
        "technical": "Primary GPU server for training workloads",
        "icon": "🏋️",
    },
    "device.inference": {
        "label": "Arena",
        "short": "Inference server",
        "tooltip": (
            "The inference server (RTX 3090). "
            "Where the hero is tested. "
            "Runs evals, sparring, and Oracle chat."
        ),
        "technical": "GPU server for inference workloads",
        "icon": "⚔️",
    },

    # =========================================================================
    # HERO & PROGRESSION
    # =========================================================================
    "hero.level": {
        "label": "Total Level",
        "short": "Combined mastery",
        "tooltip": (
            "Sum of all skill levels. "
            "Represents overall training progress. "
            "A level 30 hero with skills at L10+L10+L10 has broad competence."
        ),
        "technical": "sum(skill.level for all skills)",
    },
    "hero.steps": {
        "label": "Training Steps",
        "short": "Experience",
        "tooltip": (
            "Total optimizer steps completed. "
            "Each step processes one batch of training data. "
            "Roughly 1000 steps = 1 level of progression."
        ),
        "technical": "Cumulative training iterations",
        "icon": "⚡",
    },
    "hero.title": {
        "label": "Title",
        "short": "Rank/achievement",
        "tooltip": (
            "Earned designation based on training progress and skill mastery. "
            "Titles reflect capability thresholds, not just time spent. "
            "Some titles can be lost if skills regress."
        ),
        "technical": "Threshold-based label from configs/titles.yaml",
    },
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def get_lore(key: str) -> Optional[Dict[str, Any]]:
    """
    Get a lore entry by key.

    Args:
        key: Dot-notation key like "training.loss" or "skill.accuracy"

    Returns:
        Dict with label, tooltip, technical description, etc.
        None if key not found.
    """
    return LORE.get(key)


def get_tooltip(key: str) -> str:
    """
    Get just the tooltip text for a key.

    Args:
        key: Dot-notation key

    Returns:
        Tooltip text, or empty string if not found.
    """
    entry = LORE.get(key)
    return entry.get("tooltip", "") if entry else ""


def get_label(key: str) -> str:
    """
    Get the display label for a key.

    Args:
        key: Dot-notation key

    Returns:
        Label text, or the key itself if not found.
    """
    entry = LORE.get(key)
    return entry.get("label", key) if entry else key


def get_icon(key: str) -> str:
    """
    Get the icon for a key.

    Args:
        key: Dot-notation key

    Returns:
        Icon emoji, or empty string if not found.
    """
    entry = LORE.get(key)
    return entry.get("icon", "") if entry else ""


def list_keys() -> list[str]:
    """List all available lore keys."""
    return sorted(LORE.keys())


def export_for_js() -> Dict[str, Dict[str, str]]:
    """
    Export lore dictionary in a format suitable for JavaScript.

    Returns a simplified dict with just label, tooltip, and icon.
    Can be JSON-serialized and used in frontend.
    """
    result = {}
    for key, entry in LORE.items():
        result[key] = {
            "label": entry.get("label", key),
            "tooltip": entry.get("tooltip", ""),
            "icon": entry.get("icon", ""),
        }
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key == "--list":
            for k in list_keys():
                entry = LORE[k]
                print(f"{k}: {entry.get('label', k)}")
        elif key == "--json":
            print(json.dumps(export_for_js(), indent=2))
        else:
            entry = get_lore(key)
            if entry:
                print(f"Label: {entry.get('label', key)}")
                print(f"Icon: {entry.get('icon', '-')}")
                print(f"Tooltip: {entry.get('tooltip', '-')}")
                print(f"Technical: {entry.get('technical', '-')}")
            else:
                print(f"Unknown key: {key}")
                print(f"Available: {', '.join(list_keys()[:5])}...")
    else:
        print(f"Lore Dictionary: {len(LORE)} entries")
        print("\nCategories:")
        categories = {}
        for key in LORE:
            cat = key.split(".")[0]
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} entries")
        print("\nUsage: python3 tavern/lore.py <key> | --list | --json")
