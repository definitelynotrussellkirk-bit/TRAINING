#!/usr/bin/env python3
"""CLI interface for adaptive curriculum system."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .controller import DifficultyConfig
from .orchestrator import AdaptiveCurriculumOrchestrator, OrchestratorConfig
from .registry import create_syllo_generator


def cmd_start(args: argparse.Namespace) -> int:
    """Start the adaptive curriculum orchestrator."""
    base_dir = Path(args.base_dir).resolve()

    # Create config
    config = OrchestratorConfig(
        base_dir=base_dir,
        batch_size=args.batch_size,
        queue_threshold=args.queue_threshold,
        eval_interval=args.eval_interval,
        eval_sample_size=args.eval_sample_size,
        target_accuracy=args.target_accuracy,
        accuracy_band=args.accuracy_band,
        window_size=args.window_size,
        min_samples=args.min_samples,
        inference_url=args.inference_url
    )

    # Create orchestrator
    orchestrator = AdaptiveCurriculumOrchestrator(config)

    # Register generators
    if args.generators:
        # Load from config file
        gen_config_file = Path(args.generators)
        if gen_config_file.exists():
            orchestrator.registry._load_from_file(gen_config_file)
        else:
            print(f"Error: Generator config file not found: {gen_config_file}")
            return 1
    else:
        # Default: register SYLLO generator
        print("No generator config provided, using default SYLLO generator")
        syllo_config = create_syllo_generator()
        orchestrator.register_generator(syllo_config)

    # Run loop
    orchestrator.run_loop(check_interval=args.check_interval)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show current status."""
    base_dir = Path(args.base_dir).resolve()

    config = OrchestratorConfig(base_dir=base_dir)
    orchestrator = AdaptiveCurriculumOrchestrator(config)

    # Load any existing generators
    if args.generators:
        gen_config_file = Path(args.generators)
        if gen_config_file.exists():
            orchestrator.registry._load_from_file(gen_config_file)

    if args.json:
        # JSON output
        report = orchestrator.status_report()
        print(json.dumps(report, indent=2))
    else:
        # Human-readable output
        orchestrator.print_status()

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate a single batch."""
    base_dir = Path(args.base_dir).resolve()

    config = OrchestratorConfig(
        base_dir=base_dir,
        batch_size=args.count,
        inference_url=args.inference_url
    )
    orchestrator = AdaptiveCurriculumOrchestrator(config)

    # Register generators
    if args.generators:
        gen_config_file = Path(args.generators)
        if gen_config_file.exists():
            orchestrator.registry._load_from_file(gen_config_file)
    else:
        syllo_config = create_syllo_generator()
        orchestrator.register_generator(syllo_config)

    # Generate batch
    output_path = orchestrator.generate_batch(args.generator)

    if output_path:
        print(f"Successfully generated batch: {output_path}")
        return 0
    else:
        print("Generation failed")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluations."""
    base_dir = Path(args.base_dir).resolve()

    config = OrchestratorConfig(
        base_dir=base_dir,
        inference_url=args.inference_url
    )
    orchestrator = AdaptiveCurriculumOrchestrator(config)

    # Register generators (needed for stats tracking)
    if args.generators:
        gen_config_file = Path(args.generators)
        if gen_config_file.exists():
            orchestrator.registry._load_from_file(gen_config_file)

    # Run evaluations
    results = orchestrator.run_evaluations()

    if args.json:
        print(json.dumps(results, indent=2))

    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    """Initialize generator config file."""
    output_path = Path(args.output)

    # Create example config with SYLLO generator
    syllo_config = create_syllo_generator()

    config_data = {
        "generators": [
            syllo_config.to_dict()
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(config_data, f, indent=2)

    print(f"Created generator config: {output_path}")
    print("\nYou can now edit this file to:")
    print("  - Add more generators")
    print("  - Adjust difficulty levels")
    print("  - Customize toggle mappings")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive curriculum learning system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start orchestrator with default SYLLO generator
  %(prog)s start

  # Start with custom generator config
  %(prog)s start --generators generators.json

  # Generate a single batch
  %(prog)s generate --generator syllo --count 500

  # Run evaluations
  %(prog)s evaluate

  # Check status
  %(prog)s status

  # Create initial generator config
  %(prog)s init-config --output generators.json
        """
    )

    # Global options
    parser.add_argument(
        "--base-dir",
        default="/path/to/training",
        help="Training system base directory"
    )
    parser.add_argument(
        "--generators",
        help="Path to generator config JSON file"
    )
    parser.add_argument(
        "--inference-url",
        default="http://192.168.x.x:8000/generate",
        help="Inference API URL"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # start command
    start_parser = subparsers.add_parser("start", help="Start orchestrator loop")
    start_parser.add_argument("--batch-size", type=int, default=1000, help="Examples per batch")
    start_parser.add_argument("--queue-threshold", type=int, default=2, help="Generate when queue < threshold")
    start_parser.add_argument("--check-interval", type=int, default=300, help="Seconds between checks")
    start_parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")
    start_parser.add_argument("--eval-sample-size", type=int, default=100, help="Examples per eval set")
    start_parser.add_argument("--target-accuracy", type=float, default=0.8, help="Target accuracy (0.0-1.0)")
    start_parser.add_argument("--accuracy-band", type=float, default=0.05, help="Accuracy tolerance band")
    start_parser.add_argument("--window-size", type=int, default=200, help="Rolling window size")
    start_parser.add_argument("--min-samples", type=int, default=20, help="Min samples before adjusting")
    start_parser.set_defaults(func=cmd_start)

    # status command
    status_parser = subparsers.add_parser("status", help="Show status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    status_parser.set_defaults(func=cmd_status)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate single batch")
    gen_parser.add_argument("--generator", required=True, help="Generator ID")
    gen_parser.add_argument("--count", type=int, default=1000, help="Number of examples")
    gen_parser.set_defaults(func=cmd_generate)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluations")
    eval_parser.add_argument("--json", action="store_true", help="Output as JSON")
    eval_parser.set_defaults(func=cmd_evaluate)

    # init-config command
    init_parser = subparsers.add_parser("init-config", help="Create generator config template")
    init_parser.add_argument("--output", default="generators.json", help="Output file path")
    init_parser.set_defaults(func=cmd_init_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
