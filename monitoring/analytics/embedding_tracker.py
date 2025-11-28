#!/usr/bin/env python3
"""
Embedding Tracker - Track representation geometry over training.

Extracts hidden state embeddings from the model for the probe set,
then visualizes using UMAP to show how the representation space evolves.

Usage:
    # Collect embeddings for current checkpoint (runs on GPU)
    python3 embedding_tracker.py --collect

    # Generate UMAP visualization
    python3 embedding_tracker.py --visualize

    # Generate animated evolution
    python3 embedding_tracker.py --animate

Output:
    status/embeddings/checkpoint_{step}.npz - Raw embeddings
    status/visualizations/umap_latest.png - Current UMAP
    status/visualizations/umap_animated.gif - Evolution animation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingCollector:
    """Collect embeddings from model for probe set."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.embeddings_dir = self.base_dir / "status" / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self) -> bool:
        """Load the model for embedding extraction."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Find model path
            if self.model_path:
                path = Path(self.model_path)
            else:
                # Try to find current checkpoint
                current_model = self.base_dir / "models" / "current_model"
                if current_model.exists():
                    # Find latest checkpoint
                    checkpoints = sorted(current_model.glob("checkpoint-*"))
                    if checkpoints:
                        path = checkpoints[-1]
                    else:
                        path = current_model
                else:
                    # Fall back to base model
                    path = self.base_dir / "models" / "Qwen3-0.6B"

            logger.info(f"Loading model from {path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(path),
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                output_hidden_states=True
            )
            self.model.eval()

            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_embeddings(self, prompts: List[str], layer: int = -1) -> np.ndarray:
        """
        Extract embeddings for a list of prompts.

        Args:
            prompts: List of prompt strings
            layer: Which layer to extract (-1 = last, -2 = second to last, etc.)

        Returns:
            np.ndarray of shape (n_prompts, hidden_dim)
        """
        import torch

        embeddings = []

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.model.device)

                # Forward pass
                outputs = self.model(**inputs)

                # Get hidden states from specified layer
                hidden_states = outputs.hidden_states[layer]

                # Mean pool over sequence length
                # Shape: (1, seq_len, hidden_dim) -> (hidden_dim,)
                embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)

        return np.array(embeddings)

    def collect_for_probe_set(self, checkpoint_name: str = "current") -> Path:
        """Collect embeddings for the entire probe set."""
        # Load probe set
        probe_file = self.base_dir / "config" / "probe_set.json"
        if not probe_file.exists():
            logger.error("Probe set not found. Run probe_set.py --init first.")
            return None

        with open(probe_file) as f:
            data = json.load(f)
            probes = data.get("probes", [])

        prompts = [p["prompt"] for p in probes]
        probe_ids = [p["id"] for p in probes]
        categories = [p["category"] for p in probes]
        difficulties = [p["difficulty"] for p in probes]

        logger.info(f"Collecting embeddings for {len(prompts)} probes...")

        # Load model if needed
        if self.model is None:
            if not self.load_model():
                return None

        # Get current step
        step = 0
        training_status = self.base_dir / "status" / "training_status.json"
        if training_status.exists():
            with open(training_status) as f:
                step = json.load(f).get("current_step", 0)

        # Collect embeddings from last layer and middle layer
        logger.info("Extracting last layer embeddings...")
        last_layer_emb = self.get_embeddings(prompts, layer=-1)

        logger.info("Extracting middle layer embeddings...")
        mid_layer_emb = self.get_embeddings(prompts, layer=len(self.model.model.layers) // 2)

        # Save
        output_path = self.embeddings_dir / f"checkpoint_{step}.npz"
        np.savez(
            output_path,
            last_layer=last_layer_emb,
            mid_layer=mid_layer_emb,
            probe_ids=probe_ids,
            categories=categories,
            difficulties=difficulties,
            step=step,
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"  Shape: {last_layer_emb.shape}")

        return output_path


class EmbeddingVisualizer:
    """Visualize embeddings using UMAP."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            from core.paths import get_base_dir
            base_dir = get_base_dir()
        self.base_dir = Path(base_dir)
        self.embeddings_dir = self.base_dir / "status" / "embeddings"
        self.viz_dir = self.base_dir / "status" / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> List[Tuple[int, Path]]:
        """List available embedding checkpoints."""
        checkpoints = []
        for f in self.embeddings_dir.glob("checkpoint_*.npz"):
            try:
                step = int(f.stem.split("_")[1])
                checkpoints.append((step, f))
            except (ValueError, IndexError):
                continue
        return sorted(checkpoints)

    def load_embeddings(self, path: Path) -> Dict[str, Any]:
        """Load embeddings from file."""
        data = np.load(path, allow_pickle=True)
        return {
            "last_layer": data["last_layer"],
            "mid_layer": data["mid_layer"],
            "probe_ids": data["probe_ids"],
            "categories": data["categories"],
            "difficulties": data["difficulties"],
            "step": int(data["step"]),
        }

    def run_umap(self, embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """Run UMAP dimensionality reduction."""
        try:
            from umap import UMAP
        except ImportError:
            logger.error("UMAP not installed. Install with: pip install umap-learn")
            # Fall back to PCA
            from sklearn.decomposition import PCA
            logger.info("Falling back to PCA")
            pca = PCA(n_components=2)
            return pca.fit_transform(embeddings)

        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        return reducer.fit_transform(embeddings)

    def visualize_checkpoint(
        self,
        checkpoint_path: Path,
        output_path: Optional[Path] = None,
        layer: str = "last_layer"
    ) -> Path:
        """Generate UMAP visualization for a single checkpoint."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib required")
            return None

        # Load embeddings
        data = self.load_embeddings(checkpoint_path)
        embeddings = data[layer]
        categories = list(data["categories"])
        step = data["step"]

        logger.info(f"Running UMAP on {embeddings.shape[0]} embeddings...")

        # Run UMAP
        coords = self.run_umap(embeddings)

        # Create color map for categories
        unique_cats = sorted(set(categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
        cat_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))

        for cat in unique_cats:
            mask = [c == cat for c in categories]
            cat_coords = coords[mask]
            ax.scatter(
                cat_coords[:, 0],
                cat_coords[:, 1],
                c=[cat_to_color[cat]],
                label=cat,
                alpha=0.6,
                s=30
            )

        ax.set_title(f"Representation Space - Step {step}", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is None:
            output_path = self.viz_dir / "umap_latest.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved UMAP visualization to {output_path}")
        return output_path

    def visualize_latest(self) -> Optional[Path]:
        """Visualize the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.error("No embeddings found. Run --collect first.")
            return None

        _, latest_path = checkpoints[-1]
        return self.visualize_checkpoint(latest_path)

    def create_animation(self, output_path: Optional[Path] = None, fps: int = 2) -> Optional[Path]:
        """Create animated UMAP showing evolution over checkpoints."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            logger.error("matplotlib required")
            return None

        checkpoints = self.list_checkpoints()
        if len(checkpoints) < 2:
            logger.warning("Need at least 2 checkpoints for animation")
            return None

        # Load all embeddings
        all_data = []
        for step, path in checkpoints:
            data = self.load_embeddings(path)
            all_data.append(data)

        # Get consistent categories
        categories = list(all_data[0]["categories"])
        unique_cats = sorted(set(categories))

        # Run UMAP on combined embeddings for consistent coordinates
        # (This ensures the same space across all checkpoints)
        combined = np.vstack([d["last_layer"] for d in all_data])
        logger.info(f"Running UMAP on combined embeddings: {combined.shape}")

        combined_coords = self.run_umap(combined)

        # Split back by checkpoint
        n_probes = len(all_data[0]["last_layer"])
        coords_by_checkpoint = []
        for i in range(len(all_data)):
            start = i * n_probes
            end = (i + 1) * n_probes
            coords_by_checkpoint.append(combined_coords[start:end])

        # Create animation
        fig, ax = plt.subplots(figsize=(12, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
        cat_to_color = {cat: colors[i] for i, cat in enumerate(unique_cats)}

        scatters = {}
        for cat in unique_cats:
            scatter = ax.scatter([], [], c=[cat_to_color[cat]], label=cat, alpha=0.6, s=30)
            scatters[cat] = scatter

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Set axis limits based on all coordinates
        all_coords = np.vstack(coords_by_checkpoint)
        margin = 0.1 * (all_coords.max() - all_coords.min())
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)

        title = ax.set_title("", fontsize=14)

        def animate(frame):
            coords = coords_by_checkpoint[frame]
            step = all_data[frame]["step"]

            for cat in unique_cats:
                mask = [c == cat for c in categories]
                cat_coords = coords[mask]
                scatters[cat].set_offsets(cat_coords)

            title.set_text(f"Representation Space - Step {step}")
            return list(scatters.values()) + [title]

        anim = FuncAnimation(
            fig, animate,
            frames=len(coords_by_checkpoint),
            interval=1000 // fps,
            blit=True
        )

        if output_path is None:
            output_path = self.viz_dir / "umap_animated.gif"

        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()

        logger.info(f"Saved animation to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Embedding Tracker")
    parser.add_argument('--base-dir', default=None,
                       help='Base directory')
    parser.add_argument('--collect', action='store_true',
                       help='Collect embeddings for current checkpoint')
    parser.add_argument('--model-path', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate UMAP visualization')
    parser.add_argument('--animate', action='store_true',
                       help='Generate animated UMAP')
    parser.add_argument('--list', action='store_true',
                       help='List available checkpoints')

    args = parser.parse_args()

    if args.collect:
        collector = EmbeddingCollector(args.base_dir, args.model_path)
        path = collector.collect_for_probe_set()
        if path:
            print(f"Collected embeddings: {path}")

    elif args.visualize:
        visualizer = EmbeddingVisualizer(args.base_dir)
        path = visualizer.visualize_latest()
        if path:
            print(f"Generated: {path}")

    elif args.animate:
        visualizer = EmbeddingVisualizer(args.base_dir)
        path = visualizer.create_animation()
        if path:
            print(f"Generated: {path}")

    elif args.list:
        visualizer = EmbeddingVisualizer(args.base_dir)
        checkpoints = visualizer.list_checkpoints()
        print(f"Available checkpoints ({len(checkpoints)}):")
        for step, path in checkpoints:
            print(f"  Step {step}: {path.name}")

    else:
        print("Use --collect, --visualize, --animate, or --list")


if __name__ == "__main__":
    main()
