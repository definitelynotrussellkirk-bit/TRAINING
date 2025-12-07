#!/usr/bin/env python3
"""
Setup Wizard - Interactive first-time setup for Realm of Training.

Consolidates 10+ manual steps into a single guided experience:
1. Check prerequisites (Python, GPU, disk space)
2. Select model based on GPU VRAM
3. Download model from HuggingFace
4. Create first campaign
5. Verify configuration

Usage:
    python3 -m training setup
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

# VRAM requirements by model (in GB)
MODEL_VRAM = {
    "Qwen3-0.6B": {"full": 8, "qlora": 4, "size_gb": 1.2},
    "Qwen3-1.7B": {"full": 12, "qlora": 6, "size_gb": 3.4},
    "Qwen3-4B": {"full": 20, "qlora": 10, "size_gb": 8.0},
    "Qwen3-8B": {"full": 40, "qlora": 18, "size_gb": 16.0},
}

# Pre-configured heroes
HEROES = {
    "dio-qwen3-0.6b": {"model": "Qwen3-0.6B", "name": "DIO", "desc": "The Skeptic - Fast training, good for learning"},
    "gou-qwen3-4b": {"model": "Qwen3-4B", "name": "GOU", "desc": "The Hound - Balanced power and speed"},
    "ojas-qwen3-8b": {"model": "Qwen3-8B", "name": "OJAS", "desc": "The Vital Force - Maximum capability"},
}


def get_base_dir() -> Path:
    """Get base directory."""
    try:
        from core.paths import get_base_dir as _get_base_dir
        return _get_base_dir()
    except ImportError:
        here = Path(__file__).resolve()
        for parent in [here] + list(here.parents):
            if (parent / "CLAUDE.md").exists():
                return parent
        return Path.cwd()


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step: int, total: int, msg: str):
    """Print a step indicator."""
    print(f"\n[{step}/{total}] {msg}")


def check_prerequisites() -> Tuple[bool, List[str]]:
    """Check all prerequisites and return (ok, issues)."""
    issues = []

    # Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required (you have {sys.version_info.major}.{sys.version_info.minor})")

    # Check pip
    try:
        import pip
    except ImportError:
        issues.append("pip not available")

    # Check PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA not available - GPU training won't work")
    except ImportError:
        issues.append("PyTorch not installed - run: pip install torch")

    # Check transformers
    try:
        import transformers
    except ImportError:
        issues.append("transformers not installed - run: pip install -e '.[training]'")

    # Check HuggingFace CLI
    result = subprocess.run(
        ["huggingface-cli", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        issues.append("huggingface-cli not found - run: pip install huggingface_hub")

    return len(issues) == 0, issues


def get_gpu_info() -> Tuple[Optional[str], Optional[int]]:
    """Get GPU name and VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = int(vram_bytes / (1024**3))
            return name, vram_gb
    except Exception:
        pass
    return None, None


def estimate_vram(model_size: str, batch_size: int, max_length: int, qlora: bool) -> float:
    """Estimate VRAM usage in GB."""
    model_info = MODEL_VRAM.get(model_size, {"full": 24, "qlora": 12})
    base = model_info["qlora"] if qlora else model_info["full"]

    # Add per-batch overhead (rough estimate)
    batch_overhead = batch_size * (max_length / 1024) * 0.5

    return base + batch_overhead


def recommend_hero(vram_gb: int) -> str:
    """Recommend a hero based on available VRAM."""
    if vram_gb >= 24:
        return "ojas-qwen3-8b"  # QLoRA 8B fits in 24GB
    elif vram_gb >= 16:
        return "gou-qwen3-4b"
    else:
        return "dio-qwen3-0.6b"


def download_model(model_name: str, models_dir: Path) -> bool:
    """Download a model from HuggingFace."""
    hf_id = f"Qwen/{model_name}"
    local_dir = models_dir / model_name

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  Model already exists at {local_dir}")
        return True

    print(f"  Downloading {hf_id}...")
    print(f"  This may take a few minutes...")

    result = subprocess.run(
        ["huggingface-cli", "download", hf_id, "--local-dir", str(local_dir)],
        capture_output=False
    )

    return result.returncode == 0


def create_campaign(base_dir: Path, hero_id: str) -> Path:
    """Create a new campaign for a hero."""
    campaigns_dir = base_dir / "campaigns" / hero_id

    # Find next campaign number
    existing = list(campaigns_dir.glob("campaign-*")) if campaigns_dir.exists() else []
    next_num = len(existing) + 1
    campaign_id = f"campaign-{next_num:03d}"

    campaign_path = campaigns_dir / campaign_id
    campaign_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (campaign_path / "data").mkdir(exist_ok=True)
    (campaign_path / "data" / "completed").mkdir(exist_ok=True)
    (campaign_path / "checkpoints").mkdir(exist_ok=True)
    (campaign_path / "status").mkdir(exist_ok=True)

    # Create campaign.json
    campaign_config = {
        "hero_id": hero_id,
        "campaign_id": campaign_id,
        "created_at": datetime.now().isoformat(),
        "status": "ready"
    }
    with open(campaign_path / "campaign.json", "w") as f:
        json.dump(campaign_config, f, indent=2)

    # Create status file
    with open(campaign_path / "status" / "campaign_status.json", "w") as f:
        json.dump({"status": "ready", "note": "Drop .jsonl files in data/ and start training"}, f)

    return campaign_path


def set_active_campaign(base_dir: Path, hero_id: str, campaign_path: Path):
    """Set the active campaign."""
    control_dir = base_dir / "control"
    control_dir.mkdir(exist_ok=True)

    active_config = {
        "hero_id": hero_id,
        "campaign_id": campaign_path.name,
        "campaign_path": f"campaigns/{hero_id}/{campaign_path.name}",
        "activated_at": datetime.now().isoformat(),
        "_comment": "Scroll of Destiny - Points to the currently active campaign"
    }

    with open(control_dir / "active_campaign.json", "w") as f:
        json.dump(active_config, f, indent=2)


def update_config_json(base_dir: Path, hero_id: str, model_name: str):
    """Update config.json with model settings."""
    config_path = base_dir / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    config["model_name"] = model_name.lower().replace("-", "_")
    config["model_display_name"] = f"{HEROES[hero_id]['name']} - {model_name}"
    config["model_path"] = f"models/{model_name}"
    config["base_model"] = f"models/{model_name}"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def copy_base_to_current(base_dir: Path, model_name: str):
    """Copy base model to current_model for training."""
    src = base_dir / "models" / model_name
    dst = base_dir / "models" / "current_model"

    if not src.exists():
        return False

    # Clear existing current_model
    if dst.exists():
        shutil.rmtree(dst)

    # Copy model files
    shutil.copytree(src, dst)
    return True


def run_setup_wizard():
    """Run the interactive setup wizard."""
    base_dir = get_base_dir()

    print_header("REALM OF TRAINING - Setup Wizard")
    print()
    print("This wizard will help you set up your training environment.")
    print("It will check prerequisites, download a model, and create your first campaign.")

    # Step 1: Prerequisites
    print_step(1, 5, "Checking prerequisites...")

    ok, issues = check_prerequisites()
    if not ok:
        print("\n  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Please resolve these issues and run setup again.")
        return 1

    print("  All prerequisites satisfied!")

    # Step 2: GPU detection
    print_step(2, 5, "Detecting GPU...")

    gpu_name, vram_gb = get_gpu_info()
    if gpu_name:
        print(f"  Found: {gpu_name}")
        print(f"  VRAM: {vram_gb} GB")
        recommended = recommend_hero(vram_gb)
    else:
        print("  No GPU detected (CPU training will be slow)")
        recommended = "dio-qwen3-0.6b"
        vram_gb = 0

    # Step 3: Hero selection
    print_step(3, 5, "Select your hero...")
    print()
    print("  Available heroes:")
    print()

    hero_list = list(HEROES.keys())
    for i, hero_id in enumerate(hero_list, 1):
        hero = HEROES[hero_id]
        model_info = MODEL_VRAM.get(hero["model"], {})
        qlora_vram = model_info.get("qlora", "?")
        size_gb = model_info.get("size_gb", "?")
        rec = " (recommended)" if hero_id == recommended else ""
        fits = "fits" if vram_gb >= qlora_vram else "needs more VRAM"

        print(f"  {i}. {hero['name']} ({hero_id})")
        print(f"     {hero['desc']}")
        print(f"     Model: {hero['model']} ({size_gb}GB download, needs {qlora_vram}GB VRAM){rec}")
        print()

    while True:
        choice = input(f"  Select hero [1-{len(hero_list)}] (default: {hero_list.index(recommended)+1}): ").strip()
        if not choice:
            selected_hero = recommended
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(hero_list):
                selected_hero = hero_list[idx]
                break
        except ValueError:
            pass
        print("  Invalid choice, try again.")

    hero = HEROES[selected_hero]
    model_name = hero["model"]
    print(f"\n  Selected: {hero['name']} ({selected_hero})")

    # Step 4: Download model
    print_step(4, 5, f"Setting up {model_name}...")

    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    if not download_model(model_name, models_dir):
        print("\n  Failed to download model. Check your internet connection.")
        return 1

    print("  Model ready!")

    # Initialize current_model
    print("  Initializing current_model...")
    if not copy_base_to_current(base_dir, model_name):
        print("  Warning: Could not initialize current_model")

    # Step 5: Create campaign
    print_step(5, 5, "Creating your first campaign...")

    campaign_path = create_campaign(base_dir, selected_hero)
    set_active_campaign(base_dir, selected_hero, campaign_path)
    update_config_json(base_dir, selected_hero, model_name)

    print(f"  Campaign created: {campaign_path.relative_to(base_dir)}")

    # Done!
    print_header("Setup Complete!")
    print()
    print("Your realm is ready. Here's what to do next:")
    print()
    print("  1. Start the services:")
    print("     python3 -m training start-all")
    print()
    print("  2. Open the Tavern UI:")
    print("     http://localhost:8888")
    print()
    print("  3. Drop training data in the inbox:")
    print(f"     cp your_data.jsonl {base_dir}/inbox/")
    print()
    print("  4. Start training:")
    print(f"     python3 -m arena.hero_loop {selected_hero}/campaign-001")
    print()
    print("For more info, see QUICKSTART.md or run: python3 -m training doctor")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(run_setup_wizard())
