#!/usr/bin/env python3
"""
Arcana CLI - Run LISP scripts or enter REPL mode.

Usage:
    python -m arcana                    # REPL mode
    python -m arcana script.lisp        # Run a script
    python -m arcana -c "(status)"      # Eval expression
    python -m arcana --world            # Show world state
    python -m arcana --sync             # Sync from filesystem and show
"""

import argparse
import json
import readline  # For REPL history
import sys
from pathlib import Path

from .engine import create_engine, EvalError
from .parser import parse, to_sexpr, ParseError


def run_repl(engine):
    """Run interactive REPL."""
    print("Arcana REPL - The Realm awaits")
    print("Type (help) for commands, Ctrl+D to exit")
    print()

    # Add help verb
    def verb_help(eng):
        return """Available commands:
  (status)           - Show training status
  (world-state)      - Show world as S-expressions
  (list-entities X)  - List entities of type X (hero, campaign, skill, quest)
  (get-hero ID)      - Get hero by ID
  (get-campaign)     - Get active campaign
  (metric :name X)   - Get metric value
  (sync-world)       - Reload from filesystem
  (log)              - Show recent actions
  (queue-status)     - Show training queue
  (train :quest Q)   - Queue training on quest Q
  (pause) (resume)   - Control training
  (quit)             - Exit REPL
"""
    engine.register('help', verb_help)

    def verb_quit(eng):
        print("Farewell, Traveler.")
        sys.exit(0)
    engine.register('quit', verb_quit)
    engine.register('exit', verb_quit)

    while True:
        try:
            line = input("arcana> ")
        except EOFError:
            print("\nFarewell, Traveler.")
            break
        except KeyboardInterrupt:
            print()
            continue

        line = line.strip()
        if not line:
            continue

        # Handle bare commands
        if line in ('status', 'help', 'quit', 'exit'):
            line = f'({line})'

        try:
            results = engine.run(line)
            for result in results:
                if result is not None:
                    _print_result(result)
        except ParseError as e:
            print(f"Parse error: {e}")
        except EvalError as e:
            print(f"Eval error: {e}")
        except Exception as e:
            print(f"Error: {e}")


def _print_result(result):
    """Pretty print a result."""
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))
    elif isinstance(result, list):
        if all(isinstance(x, dict) for x in result):
            print(json.dumps(result, indent=2, default=str))
        else:
            print(to_sexpr(result))
    elif isinstance(result, str) and '\n' in result:
        print(result)
    else:
        print(result)


def main():
    parser = argparse.ArgumentParser(
        description="Arcana - LISP DSL for the Realm of Training"
    )
    parser.add_argument(
        'script', nargs='?',
        help='LISP script to run'
    )
    parser.add_argument(
        '-c', '--command',
        help='Evaluate a single expression'
    )
    parser.add_argument(
        '--repl', action='store_true',
        help='Start REPL (default if no script)'
    )
    parser.add_argument(
        '--world', action='store_true',
        help='Show world state as S-expressions'
    )
    parser.add_argument(
        '--sync', action='store_true',
        help='Sync from filesystem and show counts'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show current training status'
    )
    parser.add_argument(
        '--base-dir', '-d',
        default=None,
        help='Base directory for the Realm (default: auto-detect from core.paths)'
    )
    parser.add_argument(
        '--no-sync', action='store_true',
        help='Skip initial filesystem sync'
    )

    args = parser.parse_args()

    # Create engine
    base_dir = Path(args.base_dir)
    engine = create_engine(base_dir=base_dir, sync=not args.no_sync)

    # Handle different modes
    if args.status:
        result = engine.run('(status)')
        _print_result(result[0])
        return

    if args.sync:
        counts = engine.world.sync_from_filesystem()
        print(f"Synced: {counts}")
        return

    if args.world:
        print(engine.world.to_sexpr())
        return

    if args.command:
        try:
            results = engine.run(args.command)
            for result in results:
                if result is not None:
                    _print_result(result)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Script not found: {script_path}", file=sys.stderr)
            sys.exit(1)

        try:
            results = engine.run_file(str(script_path))
            # Only print non-None results
            for result in results:
                if result is not None:
                    _print_result(result)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Default: REPL mode
    run_repl(engine)


if __name__ == '__main__':
    main()
