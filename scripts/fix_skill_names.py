#!/usr/bin/env python3
"""Fix skill name inconsistencies across all curriculum state files."""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.paths import get_base_dir
    base_dir = get_base_dir()
except ImportError:
    base_dir = Path(__file__).parent.parent

# Files to update
curriculum_files = [
    base_dir / "status/curriculum_state.json",
    base_dir / "data_manager/curriculum_state.json",
    base_dir / "campaigns/dio-qwen3-0.6b/campaign-001/status/curriculum_state.json",
]

def fix_curriculum_file(file_path):
    """Fix skill names in a curriculum state file."""
    if not file_path.exists():
        print(f"⊘ Skipping {file_path} (not found)")
        return

    with open(file_path) as f:
        data = json.load(f)

    # Rename skills if they exist
    if "skills" in data:
        skills = data["skills"]

        # Rename syllo -> sy
        if "syllo" in skills:
            skills["sy"] = skills.pop("syllo")
            print(f"✓ Renamed 'syllo' -> 'sy' in {file_path.name}")

        # Rename binary -> bin
        if "binary" in skills:
            skills["bin"] = skills.pop("binary")
            print(f"✓ Renamed 'binary' -> 'bin' in {file_path.name}")

    # Fix active_skill
    if "active_skill" in data:
        if data["active_skill"] == "syllo":
            data["active_skill"] = "bin"  # Set to bin since sy isn't in SKILL_LEVELS
            print(f"✓ Changed active_skill 'syllo' -> 'bin' in {file_path.name}")
        elif data["active_skill"] == "binary":
            data["active_skill"] = "bin"
            print(f"✓ Changed active_skill 'binary' -> 'bin' in {file_path.name}")
        elif data["active_skill"] == "sy":
            data["active_skill"] = "bin"  # Set to bin since sy isn't in SKILL_LEVELS
            print(f"✓ Changed active_skill 'sy' -> 'bin' in {file_path.name}")

    # Write back
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Updated {file_path}")

print("Fixing skill names across curriculum files...")
print()

for file_path in curriculum_files:
    fix_curriculum_file(file_path)

print()
print("Done!")
