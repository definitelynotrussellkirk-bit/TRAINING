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

    # =========================================================================
    # STRAIN & EFFORT (Materials Science Metaphor)
    # =========================================================================
    "strain": {
        "label": "Strain",
        "short": "Instantaneous difficulty",
        "tooltip": (
            "How hard the hero is being pushed right now. "
            "Strain = loss minus floor (the comfortable baseline). "
            "High strain means challenging material; zero strain means coasting."
        ),
        "technical": "loss - floor (baseline loss)",
        "icon": "💪",
    },
    "strain.zone": {
        "label": "Training Zone",
        "short": "Current intensity",
        "tooltip": (
            "Like heart rate zones for exercise. "
            "RECOVERY (too easy), PRODUCTIVE (optimal), STRETCH (challenging), "
            "OVERLOAD (too hard). Stay in Productive/Stretch for best learning."
        ),
        "technical": "StrainZone enum based on strain thresholds",
        "icon": "🎯",
    },
    "effort": {
        "label": "Effort",
        "short": "Cumulative strain",
        "tooltip": (
            "Total work the hero has put in. "
            "Effort is the area under the strain curve over time. "
            "More effort doesn't always mean more learning - quality matters."
        ),
        "technical": "Sum of strain over training steps",
        "icon": "🏋️",
    },
    "experience": {
        "label": "Experience",
        "short": "Validated effort",
        "tooltip": (
            "Effort that actually counted - blessed by the Temple. "
            "Not all training produces learning. Experience = effort × quality_factor. "
            "Cursed training (bad data, broken evals) produces effort but not experience."
        ),
        "technical": "effort × blessing.quality_factor",
        "icon": "✨",
    },
    "blessing": {
        "label": "Blessing",
        "short": "Temple verdict",
        "tooltip": (
            "The Temple's judgment on training quality. "
            "After a Campaign completes, the Cleric runs Rituals and computes a Blessing. "
            "Blessed effort becomes Experience; Cursed effort is wasted. "
            "Quality factor ranges from 0.0 (cursed) to 1.0 (fully blessed)."
        ),
        "technical": "Blessing dataclass with quality_factor, verdict, experience_awarded",
        "icon": "🙏",
    },
    "plastic_gain": {
        "label": "Plastic Gain",
        "short": "Permanent improvement",
        "tooltip": (
            "How much the hero actually improved (loss before - loss after). "
            "Like plastic deformation in materials - permanent change. "
            "High plastic gain with low effort = efficient learning."
        ),
        "technical": "start_loss - end_loss",
        "icon": "📈",
    },
    "efficiency": {
        "label": "Learning Efficiency",
        "short": "Improvement per effort",
        "tooltip": (
            "How efficiently the hero is learning. "
            "Efficiency = plastic_gain / effort. "
            "High efficiency means fast progress; low means spinning wheels."
        ),
        "technical": "plastic_gain / effort_spent",
        "icon": "⚡",
    },

    # =========================================================================
    # SCHOOLS (Job Processing Families)
    # =========================================================================
    "school.inference": {
        "label": "School of Inference",
        "short": "Oracle's Sanctum",
        "tooltip": (
            "Jobs requiring model interaction - evaluation, sparring, queries. "
            "Workers of this school commune with the Oracle to test the Hero. "
            "Needs inference server access."
        ),
        "technical": "JobTypes: EVAL, SPARRING, INFERENCE",
        "icon": "🔮",
    },
    "school.forge": {
        "label": "School of the Forge",
        "short": "Data Forge",
        "tooltip": (
            "Data creation and transformation jobs. "
            "Workers of this school craft the raw materials for training. "
            "CPU-bound, no GPU needed."
        ),
        "technical": "JobTypes: DATA_GEN, DATA_FILTER, DATA_CONVERT",
        "icon": "🔥",
    },
    "school.vault": {
        "label": "School of the Vault",
        "short": "Vault Keepers",
        "tooltip": (
            "Storage and archival operations. "
            "Workers of this school preserve checkpoints across time. "
            "I/O-bound, prefers storage nodes."
        ),
        "technical": "JobTypes: ARCHIVE, RETENTION, SYNC",
        "icon": "🏛️",
    },
    "school.analytics": {
        "label": "School of Analytics",
        "short": "The Scriveners",
        "tooltip": (
            "Reporting and metrics computation. "
            "Workers of this school chronicle progress and system state. "
            "Quick jobs, can run anywhere."
        ),
        "technical": "JobTypes: ANALYTICS, REPORT, HEALTH_CHECK",
        "icon": "📊",
    },
    "school.archaeology": {
        "label": "School of Archaeology",
        "short": "The Seers",
        "tooltip": (
            "Model interpretability and introspection. "
            "Workers of this school peer into the Hero's mind. "
            "May need GPU for activation analysis."
        ),
        "technical": "JobTypes: LAYER_STATS, LAYER_DRIFT",
        "icon": "🔬",
    },

    # =========================================================================
    # DOMAINS (Training Worlds)
    # =========================================================================
    "domain": {
        "label": "Domain",
        "short": "Training world",
        "tooltip": (
            "A world where the Hero trains - defines tasks, data, and evaluation. "
            "Like WoW zones or OASIS planets. "
            "Heroes can specialize in one domain or master many."
        ),
        "technical": "Dataset + tasks + eval sets configuration",
        "icon": "🌍",
    },
    "domain.reasoning": {
        "label": "Domain of Reasoning",
        "short": "The Logic Spire",
        "tooltip": (
            "Structured reasoning and problem-solving tasks. "
            "Includes Syllacrostics and Binary Arithmetic. "
            "Foundation for more complex domains."
        ),
        "technical": "configs/domains/reasoning.yaml",
        "icon": "🧠",
    },

    # =========================================================================
    # PHYSICS & TECHNIQUE (Training Methods)
    # =========================================================================
    "physics": {
        "label": "Physics",
        "short": "Training rules",
        "tooltip": (
            "The laws governing training - optimizer, precision, schedules. "
            "Different physics create different training dynamics. "
            "Same Hero can train under different Physics in different Campaigns."
        ),
        "technical": "Optimizer + precision + gradient handling config",
        "icon": "⚛️",
    },
    "technique": {
        "label": "Technique",
        "short": "Training method",
        "tooltip": (
            "A named training stack - Muon, AdamW, etc. "
            "Techniques wrap Physics configurations into reusable recipes. "
            "Choose based on model size, task type, and stability needs."
        ),
        "technical": "Named physics configuration from configs/physics/",
        "icon": "🔧",
    },
    "technique.muon": {
        "label": "Muon Technique",
        "short": "The Orthogonal Way",
        "tooltip": (
            "Momentum orthogonalized by Newton-Schulz iterations. "
            "Experimental optimizer with different dynamics than AdamW. "
            "May work well for small-medium models."
        ),
        "technical": "trainer/optimizers/muon.py",
        "icon": "⚛️",
    },
    "technique.adamw": {
        "label": "AdamW Technique",
        "short": "The Classical Path",
        "tooltip": (
            "Adam with decoupled weight decay. "
            "The standard choice - well-understood, stable, widely tested. "
            "Safe default for any model size."
        ),
        "technical": "torch.optim.AdamW",
        "icon": "📐",
    },

    # =========================================================================
    # TEMPLE & ORDERS (Diagnostics)
    # =========================================================================
    "temple": {
        "label": "Temple",
        "short": "Diagnostics hub",
        "tooltip": (
            "The place where all system diagnostics happen. "
            "The Cleric runs Rituals here to check system health. "
            "Returns to Temple for judgment after training."
        ),
        "technical": "temple/ module, /api/temple endpoints",
        "icon": "🏛️",
    },
    "temple.cleric": {
        "label": "Cleric",
        "short": "Ritual orchestrator",
        "tooltip": (
            "The service that runs Temple rituals and computes Blessings. "
            "Decides whether training effort becomes experience. "
            "Consults the Nine Orders for judgment."
        ),
        "technical": "temple/cleric.py",
        "icon": "⛪",
    },
    "temple.ritual": {
        "label": "Ritual",
        "short": "Diagnostic check",
        "tooltip": (
            "A single diagnostic check within the Temple. "
            "Rituals are grouped into Orders by domain. "
            "Returns ok/warn/fail status."
        ),
        "technical": "temple/rituals/*.py",
        "icon": "🔮",
    },
    "temple.ceremony": {
        "label": "Ceremony",
        "short": "Multiple rituals",
        "tooltip": (
            "Running multiple rituals together with dependency ordering. "
            "Full ceremony consults all Nine Orders. "
            "Results determine Blessing quality."
        ),
        "technical": "temple.cleric.run_ceremony()",
        "icon": "🎭",
    },
    "temple.blessing": {
        "label": "Blessing",
        "short": "Temple verdict",
        "tooltip": (
            "The Temple's judgment on training validity. "
            "Converts Effort into Experience via quality_factor. "
            "Cursed training (quality=0) produces no experience."
        ),
        "technical": "temple/schemas.py Blessing class",
        "icon": "✨",
    },
    "temple.order.forge": {
        "label": "Order of the Forge",
        "short": "GPU/hardware checks",
        "tooltip": (
            "Checks GPU, CUDA, builds, environment. "
            "Critical order - failure blocks training. "
            "If the Forge fails, nothing can be trained."
        ),
        "technical": "temple/rituals/forge.py",
        "icon": "🔥",
    },
    "temple.order.oracle": {
        "label": "Order of the Oracle",
        "short": "Inference checks",
        "tooltip": (
            "Checks inference server, eval harness, scoring. "
            "Critical order - failure means flying blind on quality. "
            "Oracle sees the future (projected performance)."
        ),
        "technical": "temple/rituals/oracle.py",
        "icon": "🔮",
    },
    "temple.order.champion": {
        "label": "Order of the Champion",
        "short": "Model health checks",
        "tooltip": (
            "Checks model/checkpoint health, regression tests. "
            "Critical order - guards the best checkpoint. "
            "Champions are crowned here."
        ),
        "technical": "temple/rituals/champion.py",
        "icon": "🏆",
    },
    "temple.order.scribe": {
        "label": "Order of the Scribe",
        "short": "Logging/eval checks",
        "tooltip": (
            "Checks logging pipeline, metrics, ledger. "
            "If Scribe fails, history is unreliable. "
            "Chronicles every adventure."
        ),
        "technical": "temple/rituals/scribe.py",
        "icon": "📜",
    },
    "temple.order.weaver": {
        "label": "Order of the Weaver",
        "short": "Daemon/process checks",
        "tooltip": (
            "Checks daemon health, process state. "
            "Weaves the threads of running services. "
            "If Weaver fails, daemons may be unhealthy."
        ),
        "technical": "temple/rituals/weaver.py",
        "icon": "🕸️",
    },
    "temple.order.guild": {
        "label": "Order of the Guild",
        "short": "Skills/curriculum checks",
        "tooltip": (
            "Checks skill servers, curriculum state. "
            "Guild manages skills and progression. "
            "If Guild fails, skill tracking is unreliable."
        ),
        "technical": "temple/rituals/guild.py",
        "icon": "⚔️",
    },

    # =========================================================================
    # CAMPAIGN & PATH
    # =========================================================================
    "campaign": {
        "label": "Campaign",
        "short": "Training playthrough",
        "tooltip": (
            "One attempt to push a Hero to maximum potential. "
            "Like an RPG playthrough - discover the level cap. "
            "Multiple campaigns = multiple attempts with same Hero."
        ),
        "technical": "guild/campaigns/types.py Campaign class",
        "icon": "🗺️",
    },
    "path": {
        "label": "Path",
        "short": "Training recipe",
        "tooltip": (
            "A complete training configuration: Domain + Physics + Technique. "
            "Campaigns follow a Path. "
            "Example: 'Muon Technique in the Domain of Reasoning'"
        ),
        "technical": "Combined domain + physics + technique config",
        "icon": "🛤️",
    },
    "milestone": {
        "label": "Milestone",
        "short": "Notable event",
        "tooltip": (
            "A significant achievement in a Campaign's history. "
            "First level-up, new personal best, skill unlock. "
            "Milestones are recorded in the Ledger."
        ),
        "technical": "guild/campaigns/types.py Milestone class",
        "icon": "🏁",
    },
    "ledger": {
        "label": "Ledger",
        "short": "Historical record",
        "tooltip": (
            "The canonical record of all training history. "
            "Checkpoints, campaigns, rituals, blessings. "
            "The Scribe maintains the Ledger."
        ),
        "technical": "core/checkpoint_ledger.py",
        "icon": "📖",
    },

    # =========================================================================
    # FORWARD PROGRESS (Momentum Engine)
    # =========================================================================
    "forward_momentum": {
        "label": "Forward Momentum",
        "short": "Progress state",
        "tooltip": (
            "Is the system making progress or stuck? "
            "Status: GO (ready to train), BLOCKED (needs action), IDLE (nothing pending). "
            "When blocked, the system suggests how to fix it."
        ),
        "technical": "core/momentum.py MomentumState",
        "icon": "🚀",
    },
    "blocker": {
        "label": "Blocker",
        "short": "Progress obstacle",
        "tooltip": (
            "Something preventing forward progress. "
            "Blockers include: what was attempted, why it failed, how to fix. "
            "The UI surfaces blockers prominently with suggested actions."
        ),
        "technical": "core/momentum.py Blocker dataclass",
        "icon": "🚧",
    },

    # =========================================================================
    # TRAINING SCHOOLS (How Learning Happens)
    # =========================================================================
    "training_school": {
        "label": "Training School",
        "short": "Learning paradigm",
        "tooltip": (
            "A fundamental approach to how the Hero learns. "
            "Different schools use different data formats and objectives. "
            "The Six Schools: Scribe, Mirror, Judge, Champion, Whisper, Oracle."
        ),
        "technical": "guild/training_schools.py TrainingSchool enum",
        "icon": "🎓",
    },
    "school.scribe": {
        "label": "School of the Scribe",
        "short": "SFT - Imitation",
        "tooltip": (
            "\"Copy the master's form until it becomes your own.\" "
            "Learn by directly imitating correct examples. "
            "Foundation of all training - simple, stable, effective."
        ),
        "technical": "Supervised Fine-Tuning (SFT)",
        "icon": "📜",
    },
    "school.mirror": {
        "label": "School of the Mirror",
        "short": "Sparring - Self-correction",
        "tooltip": (
            "\"See your flaws reflected, then correct them.\" "
            "Learn by identifying and correcting your own mistakes. "
            "Teaches judgment: Is this right? How do I fix it?"
        ),
        "technical": "guild/sparring.py - Error mining + correction",
        "icon": "🪞",
    },
    "school.judge": {
        "label": "School of the Judge",
        "short": "DPO - Preferences",
        "tooltip": (
            "\"Between two paths, always choose the better.\" "
            "Learn by comparing options and choosing the superior one. "
            "Develops nuanced quality judgment, not just right/wrong."
        ),
        "technical": "Direct Preference Optimization (DPO)",
        "icon": "⚖️",
    },
    "school.champion": {
        "label": "School of the Champion",
        "short": "RLHF - Rewards",
        "tooltip": (
            "\"Seek the reward, master the arena.\" "
            "Learn by maximizing reward signals from a judge. "
            "Most powerful but complex - for alignment and behavior shaping."
        ),
        "technical": "Reinforcement Learning from Human Feedback",
        "icon": "🏆",
    },
    "school.whisper": {
        "label": "School of the Whisper",
        "short": "Distillation",
        "tooltip": (
            "\"The wisdom of giants flows to those who listen.\" "
            "Learn from a larger, more capable model. "
            "Small hero gains big model capabilities through transfer."
        ),
        "technical": "Knowledge Distillation",
        "icon": "👻",
    },
    "school.oracle": {
        "label": "School of the Oracle",
        "short": "Fortune Teller - Surprise",
        "tooltip": (
            "\"Focus where uncertainty dwells; ignore what is already known.\" "
            "Weight gradients by surprise - focus on what's uncertain. "
            "Enhances other schools with automatic curriculum learning."
        ),
        "technical": "trainer/losses/fortune_teller.py",
        "icon": "🔮",
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
