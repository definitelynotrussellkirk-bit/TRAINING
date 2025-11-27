"""
The Tavern - Game UI for Realm of Training

The Tavern is the gathering place where adventurers:
    - View their hero DIO's progress and stats
    - Monitor ongoing battles (training runs)
    - Check quest completion status
    - Review guild skills and progression
    - Access the vault treasures

Start the Tavern:
    python3 tavern/server.py --port 8888

Then visit: http://localhost:8888

Structure:
    tavern/
    ├── server.py           # Tavern server (serves game UI)
    ├── templates/          # HTML templates
    │   └── game.html       # Main game interface
    └── static/             # Static assets
        ├── css/
        │   └── game.css    # Game styling
        └── js/
            └── game.js     # Game logic & animations

RPG Flavor:
    Every realm needs a tavern - a place where heroes gather
    between quests to share tales, count their gold, and plan
    their next adventure. Our Tavern shows the state of the
    entire training realm in one glance.

The Tavern connects to:
    - The Forge (4090) - Training status
    - The Arena (3090) - Evaluation results
    - The Vault - Checkpoint storage
    - The Guild - Skill progression
"""

__version__ = "1.0.0"

from tavern.server import run_tavern, TavernHandler

__all__ = [
    "run_tavern",
    "TavernHandler",
]
