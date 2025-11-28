#!/usr/bin/env python3
"""
Skill Radar - Generate radar/spider plots of skill performance.

Visualizes multi-dimensional skill scores as radar charts.
Overlay multiple checkpoints to see improvement trajectory.

Usage:
    # Generate radar for current checkpoint
    python3 skill_radar.py --checkpoint latest

    # Compare two checkpoints
    python3 skill_radar.py --compare checkpoint-164000 checkpoint-165000

    # Generate animated evolution GIF
    python3 skill_radar.py --animate --last 10

Output:
    status/visualizations/skill_radar_latest.png
    status/visualizations/skill_radar_animated.gif
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_radar_chart(
    scores: Dict[str, float],
    title: str = "Skill Radar",
    output_path: Optional[Path] = None,
    comparison_scores: Optional[Dict[str, float]] = None,
    comparison_label: str = "Base"
) -> None:
    """
    Create a radar/spider chart of skill scores.

    Args:
        scores: Dict mapping skill names to scores (0.0 to 1.0)
        title: Chart title
        output_path: Where to save PNG (None = display)
        comparison_scores: Optional second set for overlay
        comparison_label: Label for comparison data
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        logger.error("matplotlib required: pip install matplotlib")
        return

    # Prepare data
    categories = list(scores.keys())
    N = len(categories)

    if N < 3:
        logger.warning("Need at least 3 categories for radar chart")
        return

    # Compute angles for each axis
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    # Get values
    values = [scores.get(cat, 0) for cat in categories]
    values += values[:1]  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw the main radar
    ax.plot(angles, values, 'o-', linewidth=2, label='Current', color='#2E86AB')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')

    # Draw comparison if provided
    if comparison_scores:
        comp_values = [comparison_scores.get(cat, 0) for cat in categories]
        comp_values += comp_values[:1]
        ax.plot(angles, comp_values, 'o--', linewidth=2, label=comparison_label, color='#E94F37')
        ax.fill(angles, comp_values, alpha=0.1, color='#E94F37')

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=8)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Title and legend
    ax.set_title(title, size=14, y=1.08)
    if comparison_scores:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved radar chart to {output_path}")
        plt.close()
    else:
        plt.show()


def create_animated_radar(
    history: List[Tuple[str, Dict[str, float]]],
    output_path: Path,
    fps: int = 2
) -> None:
    """
    Create animated GIF showing skill evolution over checkpoints.

    Args:
        history: List of (checkpoint_name, scores_dict)
        output_path: Where to save GIF
        fps: Frames per second
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        logger.error("matplotlib required: pip install matplotlib")
        return

    if not history:
        logger.warning("No history to animate")
        return

    # Get all categories across all checkpoints
    all_categories = set()
    for _, scores in history:
        all_categories.update(scores.keys())
    categories = sorted(all_categories)
    N = len(categories)

    if N < 3:
        logger.warning("Need at least 3 categories for radar chart")
        return

    # Angles
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Initialize empty plot
    line, = ax.plot([], [], 'o-', linewidth=2, color='#2E86AB')
    fill = ax.fill([], [], alpha=0.25, color='#2E86AB')[0]
    title = ax.set_title('', size=14, y=1.08)

    # Set up axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    def init():
        line.set_data([], [])
        fill.set_xy(np.array([[0, 0]]))
        return line, fill

    def animate(frame):
        checkpoint_name, scores = history[frame]
        values = [scores.get(cat, 0) for cat in categories]
        values += values[:1]

        line.set_data(angles, values)

        # Update fill
        verts = list(zip(angles, values))
        fill.set_xy(np.array(verts))

        title.set_text(f'Skill Radar - {checkpoint_name}')
        return line, fill, title

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(history), interval=1000//fps, blit=True
    )

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    logger.info(f"Saved animated radar to {output_path}")
    plt.close()


class SkillRadarGenerator:
    """Generate skill radar visualizations from training data."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.viz_dir = self.base_dir / "status" / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def get_current_skills(self) -> Dict[str, float]:
        """Get current skill scores from curriculum eval."""
        scores = {}

        # Try curriculum eval
        curriculum_eval = self.base_dir / "status" / "curriculum_eval.json"
        if curriculum_eval.exists():
            with open(curriculum_eval) as f:
                data = json.load(f)
                if 'results' in data:
                    for result in data['results']:
                        skill = result.get('skill', 'unknown')
                        level = result.get('level', 0)
                        acc = result.get('accuracy', 0)
                        scores[f"{skill}_L{level}"] = acc

        # Fallback: generate demo data
        if not scores:
            logger.warning("No skill data found, using demo values")
            scores = {
                'SYLLO_L1': 0.85,
                'SYLLO_L2': 0.72,
                'SYLLO_L3': 0.58,
                'SYLLO_L4': 0.45,
                'SYLLO_L5': 0.32,
            }

        return scores

    def get_skill_history(self, n: int = 10) -> List[Tuple[str, Dict[str, float]]]:
        """Get skill scores from learning history."""
        history = []

        history_file = self.base_dir / "status" / "learning_history.jsonl"
        if history_file.exists():
            snapshots = []
            with open(history_file) as f:
                for line in f:
                    if line.strip():
                        snapshots.append(json.loads(line))

            # Get last N
            for snapshot in snapshots[-n:]:
                step = snapshot.get('step', 0)
                skills = snapshot.get('skill_scores', {})
                if skills:
                    history.append((f"Step {step}", skills))

        return history

    def generate_current(self) -> Path:
        """Generate radar for current checkpoint."""
        scores = self.get_current_skills()
        output_path = self.viz_dir / "skill_radar_latest.png"

        create_radar_chart(
            scores=scores,
            title="Current Skill Performance",
            output_path=output_path
        )

        return output_path

    def generate_comparison(
        self,
        checkpoint1: str,
        checkpoint2: str
    ) -> Path:
        """Generate comparison radar between two checkpoints."""
        # TODO: Load historical skill data for specific checkpoints
        # For now, use current vs placeholder
        current = self.get_current_skills()
        base = {k: v * 0.8 for k, v in current.items()}  # Placeholder

        output_path = self.viz_dir / f"skill_radar_comparison.png"

        create_radar_chart(
            scores=current,
            title=f"Skill Comparison",
            output_path=output_path,
            comparison_scores=base,
            comparison_label="Baseline"
        )

        return output_path

    def generate_animation(self, n: int = 10) -> Path:
        """Generate animated radar from history."""
        history = self.get_skill_history(n)

        if not history:
            # Use current as single frame
            current = self.get_current_skills()
            history = [("Current", current)]

        output_path = self.viz_dir / "skill_radar_animated.gif"

        create_animated_radar(
            history=history,
            output_path=output_path
        )

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Skill Radar Generator")
    parser.add_argument('--base-dir', default=None,
                       help='Base directory')
    parser.add_argument('--checkpoint', type=str,
                       help='Generate radar for specific checkpoint')
    parser.add_argument('--compare', nargs=2, metavar=('CKPT1', 'CKPT2'),
                       help='Compare two checkpoints')
    parser.add_argument('--animate', action='store_true',
                       help='Generate animated GIF')
    parser.add_argument('--last', type=int, default=10,
                       help='Number of checkpoints for animation')

    args = parser.parse_args()

    generator = SkillRadarGenerator(args.base_dir)

    if args.compare:
        path = generator.generate_comparison(args.compare[0], args.compare[1])
        print(f"Generated: {path}")

    elif args.animate:
        path = generator.generate_animation(args.last)
        print(f"Generated: {path}")

    else:
        # Default: generate current
        path = generator.generate_current()
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
