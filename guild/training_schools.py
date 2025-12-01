"""
Training Schools - Paradigms of How the Hero Learns

Training Schools are distinct methodologies for teaching the Hero. Each School
has its own philosophy, data format, and training objective.

Unlike Job Processing Schools (which define HOW work is dispatched), Training
Schools define HOW LEARNING HAPPENS - the fundamental approach to knowledge
acquisition.

The Five Schools of Training:

    SCRIBE (SFT)     - Learn by copying the master
    MIRROR (Sparring) - Learn by seeing and correcting your mistakes
    JUDGE (DPO)       - Learn by comparing and choosing better
    CHAMPION (RLHF)   - Learn by seeking reward
    WHISPER (Distill) - Learn from the wisdom of elders

Usage:
    from guild.training_schools import (
        TrainingSchool,
        get_school,
        get_data_format,
        list_schools,
    )

    # Get school info
    mirror = get_school(TrainingSchool.MIRROR)
    print(mirror.rpg_name)  # "School of the Mirror"
    print(mirror.philosophy)

    # Get data format for a school
    fmt = get_data_format(TrainingSchool.MIRROR)
    print(fmt.example_types)  # ["identify_incorrect", "correction", "confirm_correct"]

    # Check if sparring data matches Mirror school
    if example["type"].startswith("sparring_"):
        school = TrainingSchool.MIRROR
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TrainingSchool(str, Enum):
    """
    The Six Schools of Training - How the Hero acquires knowledge.

    Each school represents a fundamentally different approach to learning:
    - Different data formats
    - Different loss functions
    - Different training dynamics
    - Different strengths and weaknesses

    The first five are about DATA and FORMAT.
    The sixth (Oracle) is about ATTENTION and FOCUS - it enhances other schools.
    """
    SCRIBE = "scribe"        # SFT - supervised fine-tuning (imitation)
    MIRROR = "mirror"        # Sparring - self-correction learning
    JUDGE = "judge"          # DPO - preference optimization
    CHAMPION = "champion"    # RLHF - reward-guided learning
    WHISPER = "whisper"      # Distillation - knowledge transfer
    ORACLE = "oracle"        # Fortune Teller - surprise-weighted learning

    @property
    def display_name(self) -> str:
        """Human-friendly name."""
        names = {
            TrainingSchool.SCRIBE: "School of the Scribe",
            TrainingSchool.MIRROR: "School of the Mirror",
            TrainingSchool.JUDGE: "School of the Judge",
            TrainingSchool.CHAMPION: "School of the Champion",
            TrainingSchool.WHISPER: "School of the Whisper",
            TrainingSchool.ORACLE: "School of the Oracle",
        }
        return names.get(self, self.value)

    @property
    def rpg_name(self) -> str:
        """Full RPG/lore name."""
        names = {
            TrainingSchool.SCRIBE: "School of the Scribe",
            TrainingSchool.MIRROR: "School of the Mirror",
            TrainingSchool.JUDGE: "School of the Judge",
            TrainingSchool.CHAMPION: "School of the Champion",
            TrainingSchool.WHISPER: "School of the Whisper",
            TrainingSchool.ORACLE: "School of the Oracle",
        }
        return names.get(self, self.value)

    @property
    def motto(self) -> str:
        """The school's motto/mantra."""
        mottos = {
            TrainingSchool.SCRIBE: "Copy the master's form until it becomes your own.",
            TrainingSchool.MIRROR: "See your flaws reflected, then correct them.",
            TrainingSchool.JUDGE: "Between two paths, always choose the better.",
            TrainingSchool.CHAMPION: "Seek the reward, master the arena.",
            TrainingSchool.WHISPER: "The wisdom of giants flows to those who listen.",
            TrainingSchool.ORACLE: "Focus where uncertainty dwells; ignore what is already known.",
        }
        return mottos.get(self, "")

    @property
    def icon(self) -> str:
        """Icon for the school."""
        icons = {
            TrainingSchool.SCRIBE: "📜",
            TrainingSchool.MIRROR: "🪞",
            TrainingSchool.JUDGE: "⚖️",
            TrainingSchool.CHAMPION: "🏆",
            TrainingSchool.WHISPER: "👻",
            TrainingSchool.ORACLE: "🔮",
        }
        return icons.get(self, "📚")

    @property
    def color(self) -> str:
        """Color for UI."""
        colors = {
            TrainingSchool.SCRIBE: "#3B82F6",   # Blue
            TrainingSchool.MIRROR: "#8B5CF6",   # Purple
            TrainingSchool.JUDGE: "#F59E0B",    # Amber
            TrainingSchool.CHAMPION: "#EF4444", # Red
            TrainingSchool.WHISPER: "#6B7280",  # Gray
            TrainingSchool.ORACLE: "#10B981",   # Emerald
        }
        return colors.get(self, "#6B7280")

    @property
    def technical_name(self) -> str:
        """Technical/ML name."""
        names = {
            TrainingSchool.SCRIBE: "Supervised Fine-Tuning (SFT)",
            TrainingSchool.MIRROR: "Self-Correction / Sparring",
            TrainingSchool.JUDGE: "Direct Preference Optimization (DPO)",
            TrainingSchool.CHAMPION: "RLHF / Reward Learning",
            TrainingSchool.WHISPER: "Knowledge Distillation",
            TrainingSchool.ORACLE: "Surprise-Weighted Learning (Fortune Teller)",
        }
        return names.get(self, self.value)

    @property
    def is_implemented(self) -> bool:
        """Whether this school is currently implemented."""
        return self in {
            TrainingSchool.SCRIBE,   # Basic SFT training
            TrainingSchool.MIRROR,   # Sparring system
            TrainingSchool.ORACLE,   # Fortune Teller loss
        }

    @property
    def is_enhancer(self) -> bool:
        """Whether this school enhances other schools (vs standalone)."""
        return self == TrainingSchool.ORACLE


@dataclass
class SchoolPhilosophy:
    """
    The philosophy and characteristics of a Training School.
    """
    school: TrainingSchool
    motto: str
    philosophy: str
    teaches: List[str]          # What capabilities this develops
    strengths: List[str]        # When to use this school
    weaknesses: List[str]       # Limitations
    prerequisites: List[str]    # What should come before
    implementation: str         # Where in codebase


@dataclass
class DataFormatSpec:
    """
    Data format specification for a Training School.
    """
    school: TrainingSchool
    primary_format: str         # "messages", "pairs", "ranked", etc.
    example_types: List[str]    # Types of examples this school produces
    required_fields: List[str]  # Required fields in each example
    optional_fields: List[str]  # Optional metadata fields
    example_template: Dict[str, Any]  # Example structure


# =============================================================================
# SCHOOL PHILOSOPHIES
# =============================================================================

SCHOOL_PHILOSOPHIES: Dict[TrainingSchool, SchoolPhilosophy] = {

    TrainingSchool.SCRIBE: SchoolPhilosophy(
        school=TrainingSchool.SCRIBE,
        motto="Copy the master's form until it becomes your own.",
        philosophy="""
The School of the Scribe is the foundation of all training. Here, the Hero
learns by directly imitating correct examples - seeing the right answer and
learning to reproduce it. This is pure supervised learning: given input X,
produce output Y.

The Scribe's path is straightforward but powerful. Every great warrior begins
by copying the forms of their teachers. The Hero watches, absorbs, and
gradually internalizes the patterns until they flow naturally.

Limitation: The Scribe only learns what they are shown. They cannot judge
quality, identify mistakes, or reason about alternatives - only reproduce.
""",
        teaches=[
            "Basic task completion",
            "Format adherence",
            "Pattern matching",
            "Vocabulary and style",
        ],
        strengths=[
            "Simple and well-understood",
            "Works with any task that has correct answers",
            "Stable training dynamics",
            "Easy to prepare data",
        ],
        weaknesses=[
            "Cannot learn to identify mistakes",
            "No notion of 'better' vs 'worse' - only 'correct'",
            "May memorize rather than generalize",
            "Needs high-quality examples",
        ],
        prerequisites=[],
        implementation="Standard SFT training in trainer/core/engine.py",
    ),

    TrainingSchool.MIRROR: SchoolPhilosophy(
        school=TrainingSchool.MIRROR,
        motto="See your flaws reflected, then correct them.",
        philosophy="""
The School of the Mirror teaches through self-reflection. The Hero attempts
problems, and when they fail, they study their failures. They learn to:
1. Recognize when an answer is wrong
2. Produce the correct answer after seeing the mistake
3. Confirm when an answer is correct

This creates a Hero who can judge their own outputs - a crucial capability
that pure imitation cannot provide. The Mirror teaches discernment.

The three reflections:
- IDENTIFY: "Is this correct?" → "It is incorrect."
- CORRECT: "Fix this mistake." → [correct answer]
- CONFIRM: "Is this correct?" → "It is correct."

The data is checkpoint-specific: mistakes from checkpoint N teach that
specific version of the Hero. This data has short shelf-life.
""",
        teaches=[
            "Error recognition",
            "Self-correction",
            "Answer verification",
            "Judgment and discernment",
        ],
        strengths=[
            "Teaches meta-cognition (knowing when wrong)",
            "Targeted at actual weaknesses",
            "Improves calibration",
            "Creates self-correcting behavior",
        ],
        weaknesses=[
            "Requires inference to generate (expensive)",
            "Data becomes stale as model improves",
            "Needs good answer checking",
            "Only works where correctness is verifiable",
        ],
        prerequisites=["Basic competence from Scribe training"],
        implementation="guild/sparring.py - SparringTrainer",
    ),

    TrainingSchool.JUDGE: SchoolPhilosophy(
        school=TrainingSchool.JUDGE,
        motto="Between two paths, always choose the better.",
        philosophy="""
The School of the Judge teaches comparison and preference. Instead of showing
one correct answer, the Hero sees two options and learns which is better.
This develops nuanced judgment - not just "right vs wrong" but "better vs worse".

DPO (Direct Preference Optimization) is the modern technique: given a prompt
and two responses (chosen/rejected), directly optimize to prefer the chosen.
No reward model needed - preferences are embedded directly in training.

The Judge learns to rank, compare, and evaluate - essential for tasks where
quality exists on a spectrum rather than binary correct/incorrect.
""",
        teaches=[
            "Preference and ranking",
            "Quality gradients",
            "Nuanced judgment",
            "Comparison reasoning",
        ],
        strengths=[
            "Learns quality spectrum, not just binary",
            "Works without explicit reward model (DPO)",
            "Good for subjective tasks (writing, style)",
            "Can learn from human preferences",
        ],
        weaknesses=[
            "Needs paired comparison data",
            "Can be unstable if pairs are noisy",
            "Harder to prepare data than SFT",
            "May learn spurious preferences",
        ],
        prerequisites=["Solid base from Scribe training"],
        implementation="Not yet implemented - reserved for future",
    ),

    TrainingSchool.CHAMPION: SchoolPhilosophy(
        school=TrainingSchool.CHAMPION,
        motto="Seek the reward, master the arena.",
        philosophy="""
The School of the Champion trains through reinforcement - actions that lead
to reward are strengthened, those that don't are weakened. This is RLHF:
Reinforcement Learning from Human Feedback.

The Champion doesn't just copy or compare - they explore, receive feedback,
and adapt. A reward model (trained separately) provides the signal. The Hero
learns to maximize reward while staying close to their base capabilities.

This is the most powerful but most complex school. It can teach objectives
that are hard to specify directly - "be helpful", "be harmless", "be honest".
""",
        teaches=[
            "Reward optimization",
            "Exploration and exploitation",
            "Complex objectives",
            "Behavioral alignment",
        ],
        strengths=[
            "Can optimize complex objectives",
            "Learns from scalar feedback",
            "Good for alignment properties",
            "Can improve beyond training data",
        ],
        weaknesses=[
            "Requires reward model",
            "Complex training setup (PPO, etc.)",
            "Can reward-hack if not careful",
            "Expensive and unstable",
        ],
        prerequisites=["Strong base from Scribe", "Optional: Judge for preference learning"],
        implementation="Not yet implemented - reserved for future",
    ),

    TrainingSchool.WHISPER: SchoolPhilosophy(
        school=TrainingSchool.WHISPER,
        motto="The wisdom of giants flows to those who listen.",
        philosophy="""
The School of the Whisper teaches through knowledge transfer. A larger,
more capable model (the Elder) generates outputs, and the smaller Hero
learns to match them. This is distillation.

The Whisper allows a small Hero to punch above their weight by learning
compressed knowledge from a giant. The Elder's reasoning patterns, style,
and capabilities flow into the smaller vessel.

Variations:
- Response distillation: Match the Elder's outputs
- Logit distillation: Match probability distributions
- Chain-of-thought distillation: Learn the Elder's reasoning
""",
        teaches=[
            "Compressed knowledge from larger models",
            "Reasoning patterns",
            "Style and capabilities beyond training data",
            "Efficient knowledge transfer",
        ],
        strengths=[
            "Small model gets big model capabilities",
            "Can generate unlimited training data",
            "Transfers implicit knowledge",
            "Cost-effective (train small, infer small)",
        ],
        weaknesses=[
            "Requires access to larger model",
            "Student can't exceed teacher",
            "May inherit teacher's biases/errors",
            "Logit distillation needs same tokenizer",
        ],
        prerequisites=["Base training from Scribe"],
        implementation="Not yet implemented - reserved for future",
    ),

    TrainingSchool.ORACLE: SchoolPhilosophy(
        school=TrainingSchool.ORACLE,
        motto="Focus where uncertainty dwells; ignore what is already known.",
        philosophy="""
The School of the Oracle is unique: it does not define WHAT to learn, but
WHERE to focus attention. The Oracle (Fortune Teller) predicts what will
challenge the Hero most, and concentrates gradient updates there.

Standard training treats all tokens equally - a confidently predicted "the"
gets the same gradient as an uncertain "their/there" choice. This is wasteful.
The Oracle sees which predictions are surprising and focuses there.

The key insight: surprise = information. Low-surprise predictions are already
mastered; high-surprise predictions are where learning happens. By weighting
gradients by surprise, training becomes automatically adaptive.

The Oracle enhances other schools:
- Scribe + Oracle = Focus SFT on hard parts
- Mirror + Oracle = Focus self-correction on most uncertain judgments
- Judge + Oracle = Weight preference pairs by difficulty

Surprise metrics:
- Entropy: How uncertain is the distribution?
- Confidence: How low is the max probability?
- Perplexity: How unexpected was the correct token?
- Margin: How close is the runner-up?
""",
        teaches=[
            "Automatic curriculum learning",
            "Efficient gradient allocation",
            "Focus on uncertainty",
            "Reduced forgetting of mastered skills",
        ],
        strengths=[
            "Works with any base school",
            "No extra data needed",
            "Automatic difficulty progression",
            "Reduces wasted compute",
            "Already implemented and tested",
        ],
        weaknesses=[
            "Adds computational overhead (entropy calculation)",
            "Hyperparameter sensitivity (min/max surprise)",
            "May struggle with overconfident wrong predictions",
            "Requires tuning per domain",
        ],
        prerequisites=["Any base school training (typically Scribe)"],
        implementation="trainer/losses/fortune_teller.py - FortuneTellerLoss",
    ),
}


# =============================================================================
# DATA FORMAT SPECIFICATIONS
# =============================================================================

DATA_FORMATS: Dict[TrainingSchool, DataFormatSpec] = {

    TrainingSchool.SCRIBE: DataFormatSpec(
        school=TrainingSchool.SCRIBE,
        primary_format="messages",
        example_types=["sft", "instruction", "completion"],
        required_fields=["messages"],
        optional_fields=["skill", "level", "primitive", "generator_id"],
        example_template={
            "messages": [
                {"role": "user", "content": "<prompt>"},
                {"role": "assistant", "content": "<correct_response>"},
            ],
            "skill": "sy",
            "level": 5,
        },
    ),

    TrainingSchool.MIRROR: DataFormatSpec(
        school=TrainingSchool.MIRROR,
        primary_format="messages",
        example_types=[
            "sparring_identify_incorrect",  # "Is this correct?" → "It is incorrect."
            "sparring_correction",          # "Fix this." → [correct answer]
            "sparring_confirm_correct",     # "Is this correct?" → "It is correct."
        ],
        required_fields=["messages", "type"],
        optional_fields=[
            "skill", "level", "generator_id", "generator_version",
            "session_checkpoint", "session_timestamp"
        ],
        example_template={
            "messages": [
                {
                    "role": "user",
                    "content": "<problem>\n\n---\nProposed answer:\n<model_answer>\n---\n\nIs this answer correct?"
                },
                {
                    "role": "assistant",
                    "content": "It is incorrect."
                }
            ],
            "type": "sparring_identify_incorrect",
            "skill": "binary",
            "level": 3,
            "generator_id": "sparring",
            "generator_version": "1.0.0",
        },
    ),

    TrainingSchool.JUDGE: DataFormatSpec(
        school=TrainingSchool.JUDGE,
        primary_format="preference_pair",
        example_types=["dpo_pair", "preference", "ranking"],
        required_fields=["prompt", "chosen", "rejected"],
        optional_fields=["skill", "level", "margin", "source"],
        example_template={
            "prompt": "<the input prompt>",
            "chosen": "<the preferred response>",
            "rejected": "<the rejected response>",
            "margin": 0.5,  # Optional: how much better is chosen?
        },
    ),

    TrainingSchool.CHAMPION: DataFormatSpec(
        school=TrainingSchool.CHAMPION,
        primary_format="reward_signal",
        example_types=["rlhf_episode", "reward_example"],
        required_fields=["prompt", "response", "reward"],
        optional_fields=["value_estimate", "advantage", "kl_penalty"],
        example_template={
            "prompt": "<input>",
            "response": "<model_output>",
            "reward": 0.85,  # Reward signal (0-1 or unbounded)
        },
    ),

    TrainingSchool.WHISPER: DataFormatSpec(
        school=TrainingSchool.WHISPER,
        primary_format="messages",  # Or logits for logit distillation
        example_types=["distillation", "teacher_response", "cot_distill"],
        required_fields=["messages"],
        optional_fields=["teacher_model", "teacher_logits", "temperature"],
        example_template={
            "messages": [
                {"role": "user", "content": "<prompt>"},
                {"role": "assistant", "content": "<teacher_response>"},
            ],
            "teacher_model": "qwen3-72b",
            "temperature": 0.7,  # Temperature used for teacher generation
        },
    ),

    # Oracle is unique: it's a MODIFIER, not a data format.
    # It enhances other schools by weighting gradients by surprise.
    TrainingSchool.ORACLE: DataFormatSpec(
        school=TrainingSchool.ORACLE,
        primary_format="any",  # Works with any data format!
        example_types=["oracle_enhanced"],  # Marker for oracle-enhanced training
        required_fields=[],  # No special fields required - it's a loss modifier
        optional_fields=[
            "surprise_metric",   # entropy, confidence, perplexity, margin
            "min_surprise",      # Minimum weight (default 0.1)
            "max_surprise",      # Maximum weight (default 10.0)
            "normalize_batch",   # Normalize within batch (default true)
            "temperature",       # Scaling temperature (default 1.0)
        ],
        example_template={
            # Oracle config (in training config, not per-example)
            "_oracle_config": {
                "enabled": True,
                "surprise_metric": "entropy",
                "min_surprise": 0.1,
                "max_surprise": 10.0,
                "normalize_batch": True,
                "temperature": 1.0,
            },
            # Actual data can be any format from other schools
            "messages": [
                {"role": "user", "content": "<any prompt>"},
                {"role": "assistant", "content": "<any response>"},
            ],
        },
    ),
}


# =============================================================================
# ACCESSOR FUNCTIONS
# =============================================================================

def get_school(school: TrainingSchool) -> SchoolPhilosophy:
    """Get philosophy and characteristics of a school."""
    return SCHOOL_PHILOSOPHIES[school]


def get_data_format(school: TrainingSchool) -> DataFormatSpec:
    """Get data format specification for a school."""
    return DATA_FORMATS[school]


def list_schools() -> List[TrainingSchool]:
    """List all training schools."""
    return list(TrainingSchool)


def list_implemented_schools() -> List[TrainingSchool]:
    """List only implemented training schools."""
    return [s for s in TrainingSchool if s.is_implemented]


def school_for_example_type(example_type: str) -> Optional[TrainingSchool]:
    """
    Determine which school an example belongs to based on its type.

    Args:
        example_type: The 'type' field from a training example

    Returns:
        TrainingSchool or None if unknown
    """
    # Check each school's data format
    for school, fmt in DATA_FORMATS.items():
        if example_type in fmt.example_types:
            return school

    # Check prefixes
    if example_type.startswith("sparring_"):
        return TrainingSchool.MIRROR
    if example_type.startswith("dpo_") or example_type.startswith("preference_"):
        return TrainingSchool.JUDGE
    if example_type.startswith("distill") or example_type.startswith("teacher_"):
        return TrainingSchool.WHISPER
    if example_type in ("sft", "instruction", "completion"):
        return TrainingSchool.SCRIBE

    return None


def validate_example(example: Dict[str, Any], school: TrainingSchool) -> List[str]:
    """
    Validate an example against a school's data format.

    Returns:
        List of validation errors (empty if valid)
    """
    fmt = DATA_FORMATS[school]
    errors = []

    for field in fmt.required_fields:
        if field not in example:
            errors.append(f"Missing required field: {field}")

    return errors


def get_school_stats(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count examples by training school.

    Args:
        examples: List of training examples

    Returns:
        Dict mapping school name to count
    """
    stats: Dict[str, int] = {}

    for ex in examples:
        ex_type = ex.get("type", "sft")  # Default to SFT
        school = school_for_example_type(ex_type)

        if school:
            name = school.display_name
            stats[name] = stats.get(name, 0) + 1
        else:
            stats["Unknown"] = stats.get("Unknown", 0) + 1

    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "--list":
            print("Training Schools - Paradigms of Learning")
            print("=" * 50)
            for school in TrainingSchool:
                impl = "✓" if school.is_implemented else "○"
                print(f"{impl} {school.icon} {school.display_name}")
                print(f"     Technical: {school.technical_name}")
                print(f"     Motto: {SCHOOL_PHILOSOPHIES[school].motto}")
                print()

        elif cmd == "--formats":
            print("Data Formats by School")
            print("=" * 50)
            for school, fmt in DATA_FORMATS.items():
                print(f"\n{school.icon} {school.display_name}")
                print(f"   Primary format: {fmt.primary_format}")
                print(f"   Example types: {', '.join(fmt.example_types)}")
                print(f"   Required: {', '.join(fmt.required_fields)}")

        elif cmd in [s.value for s in TrainingSchool]:
            school = TrainingSchool(cmd)
            phil = SCHOOL_PHILOSOPHIES[school]
            fmt = DATA_FORMATS[school]

            print(f"{school.icon} {school.display_name}")
            print("=" * 50)
            print(f"Technical: {school.technical_name}")
            print(f"Implemented: {'Yes' if school.is_implemented else 'Not yet'}")
            print(f"\nMotto: {phil.motto}")
            print(f"\nPhilosophy:\n{phil.philosophy}")
            print(f"\nTeaches: {', '.join(phil.teaches)}")
            print(f"\nStrengths: {', '.join(phil.strengths)}")
            print(f"\nWeaknesses: {', '.join(phil.weaknesses)}")
            print(f"\nImplementation: {phil.implementation}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python3 -m guild.training_schools [--list|--formats|<school_name>]")

    else:
        print("Training Schools - Paradigms of Learning")
        print("=" * 50)
        print(f"Schools defined: {len(TrainingSchool)}")
        print(f"Implemented: {len(list_implemented_schools())}")
        print("\nUsage:")
        print("  python3 -m guild.training_schools --list")
        print("  python3 -m guild.training_schools --formats")
        print("  python3 -m guild.training_schools mirror")
