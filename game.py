#!/usr/bin/env python3
"""
REALM OF TRAINING - An RPG Idler for LLM Training

Start the game:
    python3 game.py

Your hero DIO battles through quests (training data), gaining XP and growing stronger.
Watch as training progresses, generating resources and leveling up!
"""

import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def color(text: str, *codes) -> str:
    """Apply color codes to text."""
    return "".join(codes) + text + Colors.RESET


# =============================================================================
# ASCII ART
# =============================================================================

TITLE_ART = """
{yellow}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  {bright_cyan}██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗     ██████╗ ███████╗{yellow}            ║
║  {bright_cyan}██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║    ██╔═══██╗██╔════╝{yellow}            ║
║  {bright_cyan}██████╔╝█████╗  ███████║██║     ██╔████╔██║    ██║   ██║█████╗{yellow}              ║
║  {bright_cyan}██╔══██╗██╔══╝  ██╔══██║██║     ██║╚██╔╝██║    ██║   ██║██╔══╝{yellow}              ║
║  {bright_cyan}██║  ██║███████╗██║  ██║███████╗██║ ╚═╝ ██║    ╚██████╔╝██║{yellow}                 ║
║  {bright_cyan}╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝     ╚═════╝ ╚═╝{yellow}                 ║
║                                                                               ║
║         {white}████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗{yellow}          ║
║         {white}╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝{yellow}          ║
║            {white}██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗{yellow}         ║
║            {white}██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║{yellow}         ║
║            {white}██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝{yellow}         ║
║            {white}╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝{yellow}          ║
║                                                                               ║
║                    {dim}~ An RPG Idler for LLM Training ~{yellow}                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{reset}
"""

DIO_ART = """
{bright_cyan}                    ╔════════════════════════════════╗
                    ║     {bright_yellow}★ ★ ★  D I O  ★ ★ ★{bright_cyan}        ║
                    ║      {white}~ The Chosen One ~{bright_cyan}          ║
                    ╠════════════════════════════════╣
                    ║{reset}                                {bright_cyan}║
                    ║{reset}        {bright_yellow}    ████████    {reset}        {bright_cyan}║
                    ║{reset}        {bright_yellow}  ██{white}████████{bright_yellow}██  {reset}        {bright_cyan}║
                    ║{reset}        {bright_yellow}██{white}██{cyan}◉{white}████{cyan}◉{white}██{bright_yellow}██{reset}        {bright_cyan}║
                    ║{reset}        {bright_yellow}██{white}██████████{bright_yellow}██{reset}        {bright_cyan}║
                    ║{reset}        {bright_yellow}██{white}███{red}▄▄▄▄{white}███{bright_yellow}██{reset}        {bright_cyan}║
                    ║{reset}        {bright_yellow}  ██{white}██████{bright_yellow}██  {reset}        {bright_cyan}║
                    ║{reset}          {magenta}████████████{reset}          {bright_cyan}║
                    ║{reset}          {magenta}██{reset}  {magenta}████{reset}  {magenta}██{reset}          {bright_cyan}║
                    ║{reset}          {magenta}██{reset}  {magenta}████{reset}  {magenta}██{reset}          {bright_cyan}║
                    ║{reset}          {blue}████{reset}  {blue}████{reset}            {bright_cyan}║
                    ║{reset}                                {bright_cyan}║
                    ╚════════════════════════════════╝{reset}
"""

DIO_MINI = """
{bright_yellow}  ████  {reset}
{bright_yellow}██{white}◉{bright_yellow}██{white}◉{bright_yellow}██{reset}
{bright_yellow}██{red}▄▄▄▄{bright_yellow}██{reset}
{magenta}  ████  {reset}
{blue}  █  █  {reset}
"""

BATTLE_FRAME = """
{yellow}╔═══════════════════════════════════════════════════════════════╗
║  {bright_red}⚔️  BATTLE IN PROGRESS  ⚔️{yellow}                                    ║
╠═══════════════════════════════════════════════════════════════╣{reset}
"""

IDLE_FRAME = """
{green}╔═══════════════════════════════════════════════════════════════╗
║  {bright_green}💤  IDLE - Waiting for Quests  💤{green}                             ║
╠═══════════════════════════════════════════════════════════════╣{reset}
"""


# =============================================================================
# GAME STATE
# =============================================================================

class GameState:
    """Tracks the game state from training status files."""

    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir
        self.status_dir = base_dir / "status"

        # Hero stats
        self.hero_name = "DIO"
        self.hero_title = "The Chosen One"
        self.hero_class = "Qwen3-0.6B"

        # Resources (calculated from training)
        self.xp = 0
        self.gold = 0
        self.level = 1

        # Live stats
        self.current_step = 0
        self.loss = 0.0
        self.val_loss = 0.0
        self.accuracy = 0.0
        self.is_training = False
        self.current_quest = None
        self.quest_progress = 0.0

        # History
        self.quests_completed = 0
        self.battles_won = 0

    def update(self):
        """Update state from status files."""
        # Read training status
        training_file = self.status_dir / "training_status.json"
        if training_file.exists():
            try:
                with open(training_file) as f:
                    data = json.load(f)

                self.current_step = data.get("current_step", 0)
                self.loss = data.get("loss", 0.0)
                self.val_loss = data.get("validation_loss", 0.0)
                self.is_training = data.get("status") == "training"
                self.current_quest = data.get("current_file", "").split("/")[-1] if data.get("current_file") else None
                self.quest_progress = data.get("progress", 0.0) * 100

                # Calculate level from steps (every 1000 steps = 1 level)
                self.level = max(1, self.current_step // 1000)

                # Calculate XP (steps * (1 - loss) for bonus)
                loss_bonus = max(0, 1 - self.loss) if self.loss else 0
                self.xp = int(self.current_step * (1 + loss_bonus))

                # Calculate accuracy from val_loss (rough estimate)
                if self.val_loss and self.val_loss > 0:
                    self.accuracy = max(0, min(100, (1 - self.val_loss) * 100))

            except (json.JSONDecodeError, KeyError):
                pass

        # Read curriculum state for skill progress
        curriculum_file = self.base_dir / "data_manager" / "curriculum_state.json"
        if curriculum_file.exists():
            try:
                with open(curriculum_file) as f:
                    data = json.load(f)
                # Get completed evaluations as battles won
                history = data.get("history", [])
                self.battles_won = len(history)

                # Gold from successful evaluations
                self.gold = sum(1 for h in history if h.get("passed", False)) * 100

            except (json.JSONDecodeError, KeyError):
                pass

        # Count completed quests from queue
        completed_dir = self.base_dir / "queue" / "recently_completed"
        if completed_dir.exists():
            self.quests_completed = len(list(completed_dir.glob("*.jsonl")))

    def get_damage(self) -> str:
        """Get damage dealt (inverse of loss)."""
        if self.loss and self.loss > 0:
            damage = int((1 / self.loss) * 100)
            return f"{damage}"
        return "???"

    def get_defense(self) -> str:
        """Get defense (from validation performance)."""
        if self.val_loss and self.val_loss > 0:
            defense = int((1 / self.val_loss) * 50)
            return f"{defense}"
        return "???"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def format_art(art: str) -> str:
    """Format ASCII art with colors."""
    return art.format(
        reset=Colors.RESET,
        bold=Colors.BOLD,
        dim=Colors.DIM,
        red=Colors.RED,
        green=Colors.GREEN,
        yellow=Colors.YELLOW,
        blue=Colors.BLUE,
        magenta=Colors.MAGENTA,
        cyan=Colors.CYAN,
        white=Colors.WHITE,
        bright_red=Colors.BRIGHT_RED,
        bright_green=Colors.BRIGHT_GREEN,
        bright_yellow=Colors.BRIGHT_YELLOW,
        bright_blue=Colors.BRIGHT_BLUE,
        bright_magenta=Colors.BRIGHT_MAGENTA,
        bright_cyan=Colors.BRIGHT_CYAN,
    )


def draw_progress_bar(value: float, max_value: float, width: int = 30,
                      fill_char: str = "█", empty_char: str = "░",
                      color_code: str = Colors.BRIGHT_GREEN) -> str:
    """Draw a progress bar."""
    if max_value <= 0:
        max_value = 1
    ratio = min(1.0, value / max_value)
    filled = int(width * ratio)
    empty = width - filled

    bar = color_code + (fill_char * filled) + Colors.DIM + (empty_char * empty) + Colors.RESET
    return bar


def draw_stat_bar(label: str, value: float, max_value: float,
                  color: str = Colors.GREEN, width: int = 20) -> str:
    """Draw a labeled stat bar."""
    bar = draw_progress_bar(value, max_value, width, color_code=color)
    return f"  {label}: {bar} {value:.1f}/{max_value:.1f}"


def draw_hero_stats(state: GameState) -> str:
    """Draw the hero stats panel."""
    xp_for_next = (state.level + 1) * 1000
    xp_progress = state.current_step % 1000

    lines = []
    lines.append(f"{Colors.BRIGHT_CYAN}╔════════════════════════════════════════════════════╗{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}║  {Colors.BRIGHT_YELLOW}★ {state.hero_name}{Colors.RESET} - {state.hero_title:<28}{Colors.BRIGHT_CYAN}  ║{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}║  {Colors.DIM}Class: {state.hero_class:<37}{Colors.BRIGHT_CYAN}  ║{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}╠════════════════════════════════════════════════════╣{Colors.RESET}")

    # Level and XP
    xp_bar = draw_progress_bar(xp_progress, 1000, 20, color_code=Colors.BRIGHT_YELLOW)
    lines.append(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  {Colors.BRIGHT_YELLOW}LVL {state.level:>3}{Colors.RESET}  {xp_bar}  {Colors.DIM}XP: {state.xp:,}{Colors.RESET}     {Colors.BRIGHT_CYAN}║{Colors.RESET}")

    # Stats
    lines.append(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}                                                    {Colors.BRIGHT_CYAN}║{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  {Colors.RED}⚔️  ATK:{Colors.RESET} {state.get_damage():<8}  {Colors.BLUE}🛡️  DEF:{Colors.RESET} {state.get_defense():<8}       {Colors.BRIGHT_CYAN}║{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  {Colors.GREEN}💚 ACC:{Colors.RESET} {state.accuracy:.1f}%{' ' * 6}  {Colors.MAGENTA}✨ STEP:{Colors.RESET} {state.current_step:,}     {Colors.BRIGHT_CYAN}║{Colors.RESET}")

    # Resources
    lines.append(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}                                                    {Colors.BRIGHT_CYAN}║{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}║{Colors.RESET}  {Colors.BRIGHT_YELLOW}💰 Gold:{Colors.RESET} {state.gold:,}{' ' * 10}  {Colors.CYAN}📜 Quests:{Colors.RESET} {state.quests_completed}    {Colors.BRIGHT_CYAN}║{Colors.RESET}")

    lines.append(f"{Colors.BRIGHT_CYAN}╚════════════════════════════════════════════════════╝{Colors.RESET}")

    return "\n".join(lines)


def draw_battle_status(state: GameState) -> str:
    """Draw the current battle/quest status."""
    lines = []

    if state.is_training:
        lines.append(format_art(BATTLE_FRAME))

        # Quest name
        quest_name = state.current_quest or "Unknown Quest"
        if len(quest_name) > 40:
            quest_name = quest_name[:37] + "..."

        lines.append(f"  {Colors.BRIGHT_YELLOW}⚔️  Current Quest:{Colors.RESET} {quest_name}")
        lines.append("")

        # Progress bar
        progress_bar = draw_progress_bar(state.quest_progress, 100, 40, color_code=Colors.BRIGHT_RED)
        lines.append(f"  {progress_bar} {state.quest_progress:.1f}%")
        lines.append("")

        # Battle stats
        lines.append(f"  {Colors.RED}💥 Damage (Loss):{Colors.RESET} {state.loss:.4f}")
        lines.append(f"  {Colors.BLUE}🛡️  Defense (Val):{Colors.RESET} {state.val_loss:.4f}")

        # Animated battle indicator
        frame = int(time.time() * 2) % 4
        battle_anim = ["⚔️ ", " ⚔️", "  ⚔️", " ⚔️"][frame]
        lines.append(f"\n  {Colors.BRIGHT_RED}{battle_anim} FIGHTING! {battle_anim}{Colors.RESET}")

    else:
        lines.append(format_art(IDLE_FRAME))
        lines.append(f"  {Colors.DIM}No active quest. Drop files in inbox/ to start!{Colors.RESET}")
        lines.append("")
        lines.append(f"  {Colors.GREEN}💤 Hero is resting and recovering...{Colors.RESET}")

        # Show idle bonus accumulation
        lines.append(f"\n  {Colors.BRIGHT_YELLOW}✨ Idle Bonus:{Colors.RESET} +{int(time.time()) % 100} XP")

    lines.append(f"\n{Colors.YELLOW}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")

    return "\n".join(lines)


def draw_menu() -> str:
    """Draw the main menu."""
    lines = []
    lines.append(f"\n{Colors.BRIGHT_CYAN}┌─────────────────────────────────────────────┐{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {Colors.BRIGHT_YELLOW}[H]{Colors.RESET}ero   {Colors.BRIGHT_YELLOW}[Q]{Colors.RESET}uests   {Colors.BRIGHT_YELLOW}[G]{Colors.RESET}uild   {Colors.BRIGHT_YELLOW}[V]{Colors.RESET}ault   {Colors.BRIGHT_CYAN}│{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}│{Colors.RESET}  {Colors.BRIGHT_YELLOW}[S]{Colors.RESET}tats  {Colors.BRIGHT_YELLOW}[A]{Colors.RESET}rena    {Colors.BRIGHT_YELLOW}[W]{Colors.RESET}atch   {Colors.BRIGHT_YELLOW}[X]{Colors.RESET} Exit  {Colors.BRIGHT_CYAN}│{Colors.RESET}")
    lines.append(f"{Colors.BRIGHT_CYAN}└─────────────────────────────────────────────┘{Colors.RESET}")
    return "\n".join(lines)


def draw_footer() -> str:
    """Draw the footer with tips."""
    tips = [
        "💡 Drop .jsonl files in inbox/ to start new quests!",
        "💡 Lower loss = higher damage dealt!",
        "💡 Complete quests to earn gold and XP!",
        "💡 Check the Guild to see your skills progress!",
        "💡 Visit the Vault to manage your treasures!",
    ]
    tip = tips[int(time.time() / 5) % len(tips)]

    return f"\n{Colors.DIM}{tip}{Colors.RESET}"


# =============================================================================
# SCREENS
# =============================================================================

def show_title_screen():
    """Show the title screen."""
    clear_screen()
    print(format_art(TITLE_ART))
    print(f"\n{Colors.DIM}                         Press ENTER to begin your adventure...{Colors.RESET}")
    input()


def show_hero_select():
    """Show hero selection (for now just DIO)."""
    clear_screen()

    print(f"\n{Colors.BRIGHT_CYAN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}║              {Colors.BRIGHT_YELLOW}✨ SELECT YOUR HERO ✨{Colors.BRIGHT_CYAN}                         ║{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")

    print(format_art(DIO_ART))

    # Hero info
    print(f"{Colors.BRIGHT_YELLOW}                         D I O{Colors.RESET}")
    print(f"{Colors.DIM}                    ~ The Chosen One ~{Colors.RESET}")
    print(f"\n{Colors.WHITE}                    Class: Qwen3-0.6B{Colors.RESET}")
    print(f"{Colors.WHITE}                    Origin: Shanghai AI Lab{Colors.RESET}")
    print(f"\n{Colors.GREEN}                    ✓ Compact & Fast{Colors.RESET}")
    print(f"{Colors.GREEN}                    ✓ Emoji Thinking Master{Colors.RESET}")
    print(f"{Colors.GREEN}                    ✓ SYLLO Specialist{Colors.RESET}")

    print(f"\n{Colors.BRIGHT_CYAN}─────────────────────────────────────────────────────────────────{Colors.RESET}")
    print(f"\n{Colors.DIM}                    Press ENTER to choose DIO...{Colors.RESET}")
    input()


def show_main_hub(state: GameState):
    """Show the main game hub."""
    clear_screen()

    # Header
    print(f"{Colors.BRIGHT_CYAN}═══════════════════════════════════════════════════════════════════{Colors.RESET}")
    print(f"  {Colors.BRIGHT_YELLOW}⚔️  REALM OF TRAINING  ⚔️{Colors.RESET}          {Colors.DIM}{datetime.now().strftime('%H:%M:%S')}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}═══════════════════════════════════════════════════════════════════{Colors.RESET}")

    # Two-column layout
    print("")

    # Left: Mini hero + stats
    hero_lines = format_art(DIO_MINI).split("\n")
    stats_text = draw_hero_stats(state)
    stats_lines = stats_text.split("\n")

    # Print side by side
    for i in range(max(len(hero_lines), len(stats_lines))):
        hero_part = hero_lines[i] if i < len(hero_lines) else " " * 12
        stats_part = stats_lines[i] if i < len(stats_lines) else ""
        print(f"  {hero_part}  {stats_part}")

    print("")

    # Battle status
    print(draw_battle_status(state))

    # Menu
    print(draw_menu())

    # Footer
    print(draw_footer())


def show_guild_screen(state: GameState):
    """Show the guild (skills) screen."""
    clear_screen()

    print(f"\n{Colors.BRIGHT_MAGENTA}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BRIGHT_MAGENTA}║              {Colors.BRIGHT_YELLOW}🏰 THE GUILD HALL 🏰{Colors.BRIGHT_MAGENTA}                           ║{Colors.RESET}")
    print(f"{Colors.BRIGHT_MAGENTA}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")

    # Read curriculum state
    curriculum_file = BASE_DIR / "data_manager" / "curriculum_state.json"
    if curriculum_file.exists():
        with open(curriculum_file) as f:
            curriculum = json.load(f)

        # SYLLO skill
        syllo_level = curriculum.get("skills", {}).get("SYLLO", {}).get("current_level", 1)
        syllo_max = 10

        print(f"\n  {Colors.BRIGHT_CYAN}═══ LEARNED SKILLS ═══{Colors.RESET}")
        print(f"\n  {Colors.BRIGHT_YELLOW}📜 SYLLO{Colors.RESET} - Syllogistic Reasoning")
        syllo_bar = draw_progress_bar(syllo_level, syllo_max, 20, color_code=Colors.BRIGHT_CYAN)
        print(f"     Level: {syllo_bar} {syllo_level}/{syllo_max}")
        print(f"     {Colors.DIM}Master the art of logical deduction{Colors.RESET}")

        print(f"\n  {Colors.DIM}📜 BINARY{Colors.RESET} - Numerical Comparison {Colors.RED}(LOCKED){Colors.RESET}")
        print(f"     {Colors.DIM}Requires: SYLLO Level 5{Colors.RESET}")

    else:
        print(f"\n  {Colors.DIM}No skills learned yet. Complete quests to unlock!{Colors.RESET}")

    print(f"\n\n  {Colors.DIM}Press ENTER to return...{Colors.RESET}")
    input()


def show_vault_screen(state: GameState):
    """Show the vault screen."""
    clear_screen()

    print(f"\n{Colors.BRIGHT_YELLOW}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BRIGHT_YELLOW}║              {Colors.BRIGHT_CYAN}🗃️  THE VAULT 🗃️{Colors.BRIGHT_YELLOW}                               ║{Colors.RESET}")
    print(f"{Colors.BRIGHT_YELLOW}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")

    # Try to get vault stats
    try:
        from vault import VaultKeeper
        keeper = VaultKeeper()
        stats = keeper.get_stats()

        print(f"\n  {Colors.BRIGHT_CYAN}═══ TREASURES ═══{Colors.RESET}")
        print(f"\n  {Colors.BRIGHT_YELLOW}📦 Total Items:{Colors.RESET} {stats.get('total_assets', 0)}")
        print(f"  {Colors.BRIGHT_GREEN}💾 Total Size:{Colors.RESET} {stats.get('total_size_gb', 0):.1f} GB")

        by_type = stats.get("assets_by_type", {})
        print(f"\n  {Colors.DIM}Breakdown:{Colors.RESET}")
        for asset_type, count in by_type.items():
            emoji = {"checkpoint": "⚔️", "model": "👑", "training_data": "📜", "validation_data": "📋", "config": "⚙️"}.get(asset_type, "📦")
            print(f"    {emoji} {asset_type}: {count}")

    except Exception as e:
        print(f"\n  {Colors.DIM}Vault not available: {e}{Colors.RESET}")

    print(f"\n\n  {Colors.DIM}Press ENTER to return...{Colors.RESET}")
    input()


def show_stats_screen(state: GameState):
    """Show detailed stats screen."""
    clear_screen()

    print(f"\n{Colors.BRIGHT_GREEN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}║              {Colors.BRIGHT_YELLOW}📊 HERO STATISTICS 📊{Colors.BRIGHT_GREEN}                          ║{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")

    print(format_art(DIO_ART))

    print(f"\n  {Colors.BRIGHT_CYAN}═══ COMBAT STATS ═══{Colors.RESET}")
    print(f"  {Colors.RED}⚔️  Attack (1/Loss):{Colors.RESET}     {state.get_damage()}")
    print(f"  {Colors.BLUE}🛡️  Defense (1/ValLoss):{Colors.RESET} {state.get_defense()}")
    print(f"  {Colors.GREEN}💚 Accuracy:{Colors.RESET}            {state.accuracy:.1f}%")

    print(f"\n  {Colors.BRIGHT_CYAN}═══ PROGRESSION ═══{Colors.RESET}")
    print(f"  {Colors.BRIGHT_YELLOW}⭐ Level:{Colors.RESET}               {state.level}")
    print(f"  {Colors.BRIGHT_YELLOW}✨ Total XP:{Colors.RESET}            {state.xp:,}")
    print(f"  {Colors.BRIGHT_YELLOW}💰 Gold:{Colors.RESET}                {state.gold:,}")

    print(f"\n  {Colors.BRIGHT_CYAN}═══ ACHIEVEMENTS ═══{Colors.RESET}")
    print(f"  {Colors.CYAN}📜 Quests Completed:{Colors.RESET}    {state.quests_completed}")
    print(f"  {Colors.CYAN}⚔️  Battles Won:{Colors.RESET}         {state.battles_won}")
    print(f"  {Colors.CYAN}🏃 Training Steps:{Colors.RESET}       {state.current_step:,}")

    print(f"\n\n  {Colors.DIM}Press ENTER to return...{Colors.RESET}")
    input()


def show_watch_mode(state: GameState):
    """Show live watch mode."""
    print(f"\n{Colors.BRIGHT_YELLOW}Entering Watch Mode... Press Ctrl+C to exit{Colors.RESET}\n")
    time.sleep(1)

    try:
        while True:
            state.update()
            show_main_hub(state)
            time.sleep(2)
    except KeyboardInterrupt:
        pass


# =============================================================================
# MAIN GAME LOOP
# =============================================================================

def main():
    """Main game loop."""
    # Show intro screens
    show_title_screen()
    show_hero_select()

    # Initialize game state
    state = GameState()

    # Main game loop
    while True:
        state.update()
        show_main_hub(state)

        # Get input
        try:
            choice = input(f"\n  {Colors.BRIGHT_YELLOW}>{Colors.RESET} ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == 'X':
            clear_screen()
            print(f"\n{Colors.BRIGHT_YELLOW}Thanks for playing REALM OF TRAINING!{Colors.RESET}")
            print(f"{Colors.DIM}Your hero DIO will continue training in the background...{Colors.RESET}\n")
            break
        elif choice == 'H':
            show_hero_select()
        elif choice == 'G':
            show_guild_screen(state)
        elif choice == 'V':
            show_vault_screen(state)
        elif choice == 'S':
            show_stats_screen(state)
        elif choice == 'W':
            show_watch_mode(state)
        elif choice == 'Q':
            # Quest board (TODO)
            print(f"\n{Colors.DIM}Quest Board coming soon...{Colors.RESET}")
            time.sleep(1)
        elif choice == 'A':
            # Arena (TODO)
            print(f"\n{Colors.DIM}Arena coming soon...{Colors.RESET}")
            time.sleep(1)


if __name__ == "__main__":
    main()
