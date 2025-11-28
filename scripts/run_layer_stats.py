#!/usr/bin/env python3
"""
Run Layer Stats Analysis - Manual trigger for Model Archaeology.

This script can:
1. Submit a layer_stats job to the job queue (remote execution)
2. Run analysis locally (direct execution)

Usage:
    # Submit job to queue (runs on 3090)
    python3 scripts/run_layer_stats.py submit \\
        --checkpoint /path/to/checkpoint \\
        --campaign campaign-001 \\
        --hero dio-qwen3-0.6b

    # Run locally (runs on current machine)
    python3 scripts/run_layer_stats.py local \\
        --checkpoint /path/to/checkpoint \\
        --campaign campaign-001 \\
        --hero dio-qwen3-0.6b

    # Compare two checkpoints
    python3 scripts/run_layer_stats.py submit \\
        --checkpoint /path/to/checkpoint-183000 \\
        --reference /path/to/checkpoint-180000 \\
        --campaign campaign-001

    # List available checkpoints
    python3 scripts/run_layer_stats.py list \\
        --campaign campaign-001 \\
        --hero dio-qwen3-0.6b
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_layer_stats")


def get_base_dir() -> Path:
    """Get training base directory."""
    return Path(os.environ.get("TRAINING_BASE_DIR", "/path/to/training"))


def submit_job(args):
    """Submit a layer_stats job to the job queue."""
    import requests

    payload = {
        "job_type": "layer_stats",
        "payload": {
            "campaign_id": args.campaign,
            "hero_id": args.hero,
            "checkpoint_path": args.checkpoint,
            "model_ref": args.model_ref,
            "compute_activations": not args.no_activations,
        },
        "priority": args.priority,
    }

    if args.reference:
        payload["payload"]["reference_checkpoint_path"] = args.reference

    logger.info(f"Submitting layer_stats job to {args.server}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    if args.reference:
        logger.info(f"  Reference: {args.reference}")

    try:
        response = requests.post(
            f"{args.server}/api/jobs",
            json=payload,
            timeout=10,
        )

        result = response.json()

        if result.get("accepted"):
            print(f"Job submitted: {result['job_id']}")
            print(f"  Queue position: {result.get('queue_position', 'unknown')}")
            if result.get("warnings"):
                print(f"  Warnings: {result['warnings']}")
            return 0
        else:
            print(f"Job rejected: {result.get('message', 'unknown error')}")
            return 1

    except requests.RequestException as e:
        print(f"Failed to submit job: {e}")
        return 1


def run_local(args):
    """Run layer_stats analysis locally."""
    from analysis import run_layer_stats_analysis
    from analysis.probe_datasets import get_default_probes

    logger.info(f"Running layer_stats analysis locally")
    logger.info(f"  Checkpoint: {args.checkpoint}")

    probes = None if args.no_activations else get_default_probes()

    result = run_layer_stats_analysis(
        checkpoint_path=args.checkpoint,
        campaign_id=args.campaign,
        hero_id=args.hero,
        model_ref=args.model_ref,
        reference_checkpoint_path=args.reference,
        probe_sequences=probes,
        compute_activations=not args.no_activations,
        device=args.device,
    )

    # Save result
    if args.output:
        output_path = Path(args.output)
    else:
        base = get_base_dir()
        analysis_dir = base / "campaigns" / args.hero / args.campaign / "analysis" / "layer_stats"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = analysis_dir / f"ckpt-{result.checkpoint_step:06d}.layer_stats.json"

    with open(output_path, "w") as f:
        f.write(result.to_json())

    print(f"Analysis complete!")
    print(f"  Output: {output_path}")
    print(f"  Duration: {result.compute_duration_sec:.1f}s")
    print(f"  Layers: {len(result.weight_stats)}")

    if result.global_drift_stats:
        print(f"  Most changed: {result.global_drift_stats.get('most_changed_layer')}")
        print(f"  Least changed: {result.global_drift_stats.get('least_changed_layer')}")

    return 0


def list_checkpoints(args):
    """List available checkpoints for a campaign."""
    base = get_base_dir()
    models_dir = base / "models" / "current_model"

    print(f"Checkpoints in {models_dir}:")
    print("-" * 60)

    checkpoints = []
    for ckpt in sorted(models_dir.glob("checkpoint-*")):
        if ckpt.is_dir():
            # Try to get step from name
            name = ckpt.name
            try:
                step = int(name.split("-")[1].split("_")[0])
            except:
                step = 0

            # Check size
            size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
            size_gb = size / (1024**3)

            checkpoints.append((step, name, size_gb, str(ckpt)))

    for step, name, size_gb, path in sorted(checkpoints, key=lambda x: x[0]):
        print(f"  {step:>8} | {name:<40} | {size_gb:.2f} GB")

    print(f"\nTotal: {len(checkpoints)} checkpoints")

    # Also check for existing analysis
    analysis_dir = base / "campaigns" / args.hero / args.campaign / "analysis" / "layer_stats"
    if analysis_dir.exists():
        analyzed = list(analysis_dir.glob("*.layer_stats.json"))
        print(f"\nAlready analyzed: {len(analyzed)} checkpoints")
        for f in sorted(analyzed)[-5:]:
            print(f"  {f.name}")

    return 0


def show_results(args):
    """Show analysis results for a checkpoint."""
    base = get_base_dir()
    analysis_dir = base / "campaigns" / args.hero / args.campaign / "analysis" / "layer_stats"

    # Find the file
    if args.checkpoint.isdigit():
        step = int(args.checkpoint)
        filepath = analysis_dir / f"ckpt-{step:06d}.layer_stats.json"
    else:
        filepath = Path(args.checkpoint)

    if not filepath.exists():
        print(f"Not found: {filepath}")
        return 1

    with open(filepath) as f:
        data = json.load(f)

    print(f"Layer Stats: checkpoint-{data.get('checkpoint_step', 'unknown')}")
    print("=" * 60)
    print(f"  Campaign: {data.get('campaign_id')}")
    print(f"  Hero: {data.get('hero_id')}")
    print(f"  Model: {data.get('model_ref')}")
    print(f"  Created: {data.get('created_at')}")
    print(f"  Duration: {data.get('compute_duration_sec', 0):.1f}s")
    print()

    # Weight stats
    global_weight = data.get("global_weight_stats", {})
    print("Weight Statistics:")
    print(f"  Avg norm: {global_weight.get('avg_weight_norm', 0):.4f}")
    print(f"  Max norm: {global_weight.get('max_weight_norm', 0):.4f}")
    print(f"  Min norm: {global_weight.get('min_weight_norm', 0):.4f}")
    print(f"  Total params: {global_weight.get('total_params', 0):,}")
    print()

    # Drift stats
    if data.get("drift_stats"):
        global_drift = data.get("global_drift_stats", {})
        print("Drift Statistics:")
        print(f"  Reference: {data.get('reference_checkpoint_path', 'N/A')}")
        print(f"  Avg drift L2: {global_drift.get('avg_drift_l2', 0):.6f}")
        print(f"  Max drift L2: {global_drift.get('max_drift_l2', 0):.6f}")
        print(f"  Most changed: {global_drift.get('most_changed_layer')}")
        print(f"  Least changed: {global_drift.get('least_changed_layer')}")
        print()

    # Activation stats
    if data.get("activation_stats"):
        probe_info = data.get("probe_info", {})
        print("Activation Statistics:")
        print(f"  Probes: {probe_info.get('num_sequences', 0)} sequences, {probe_info.get('total_tokens', 0)} tokens")

        # Show a few layers
        act_stats = data.get("activation_stats", {})
        print(f"  Layers analyzed: {len(act_stats)}")

        if args.verbose:
            print("\n  Per-layer activation stats:")
            for name, stats in list(act_stats.items())[:5]:
                print(f"    {name}:")
                print(f"      mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                print(f"      range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Model Archaeology - Layer Stats Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit job to queue")
    submit_parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint path")
    submit_parser.add_argument("--reference", "-r", help="Reference checkpoint for drift")
    submit_parser.add_argument("--campaign", default="campaign-001")
    submit_parser.add_argument("--hero", default="dio-qwen3-0.6b")
    submit_parser.add_argument("--model-ref", default="qwen3-0.6b")
    submit_parser.add_argument("--server", default="http://localhost:8767")
    submit_parser.add_argument("--priority", default="normal", choices=["low", "normal", "high"])
    submit_parser.add_argument("--no-activations", action="store_true")

    # Local command
    local_parser = subparsers.add_parser("local", help="Run analysis locally")
    local_parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint path")
    local_parser.add_argument("--reference", "-r", help="Reference checkpoint for drift")
    local_parser.add_argument("--campaign", default="campaign-001")
    local_parser.add_argument("--hero", default="dio-qwen3-0.6b")
    local_parser.add_argument("--model-ref", default="qwen3-0.6b")
    local_parser.add_argument("--device", default="cuda")
    local_parser.add_argument("--output", "-o", help="Output path")
    local_parser.add_argument("--no-activations", action="store_true")

    # List command
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--campaign", default="campaign-001")
    list_parser.add_argument("--hero", default="dio-qwen3-0.6b")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show analysis results")
    show_parser.add_argument("checkpoint", help="Checkpoint step or path")
    show_parser.add_argument("--campaign", default="campaign-001")
    show_parser.add_argument("--hero", default="dio-qwen3-0.6b")
    show_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.command == "submit":
        return submit_job(args)
    elif args.command == "local":
        return run_local(args)
    elif args.command == "list":
        return list_checkpoints(args)
    elif args.command == "show":
        return show_results(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
