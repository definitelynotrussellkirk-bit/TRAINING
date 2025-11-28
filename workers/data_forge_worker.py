"""
Data Forge Worker - Worker for data generation and processing jobs.

Handles:
- DATA_GEN: Generate training data
- DATA_FILTER: Filter/validate data
- DATA_CONVERT: Convert data formats

Usage:
    # Start on a machine (no GPU required)
    python3 -m workers.data_forge_worker --port 8900

    # With specific device ID
    python3 -m workers.data_forge_worker --device macmini_forge_1 --port 8900
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from workers.base_worker import BaseWorker, WorkerConfig

logger = logging.getLogger("data_forge_worker")


class DataForgeWorker(BaseWorker):
    """
    Worker for data generation and processing jobs.

    Does not require GPU - can run on CPU-only machines.
    """

    SUPPORTED_TYPES = ["data_gen", "data_filter", "data_convert"]

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        device_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize data forge worker.

        Args:
            config: Worker configuration
            device_id: Device ID override
            output_dir: Where to save generated data
        """
        # Data forge can run multiple concurrent jobs
        if config is None:
            device_id = device_id or os.environ.get("TRAINING_DEVICE_ID", "data_forge")
            config = WorkerConfig(device_id=device_id, max_concurrent=3)

        super().__init__(config, device_id)

        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(os.environ.get(
                "DATA_FORGE_OUTPUT",
                "/tmp/data_forge"
            ))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_supported_job_types(self) -> List[str]:
        """Get supported job types."""
        return self.SUPPORTED_TYPES

    def handle_job(self, job_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a data generation/processing job.

        Args:
            job_id: Job identifier
            spec: Job specification

        Returns:
            Result dict
        """
        job_type = spec.get("job_type", "unknown")
        payload = spec.get("payload", {})

        logger.info(f"Handling {job_type} job {job_id}")

        if job_type == "data_gen":
            return self._handle_data_gen(job_id, payload)
        elif job_type == "data_filter":
            return self._handle_data_filter(job_id, payload)
        elif job_type == "data_convert":
            return self._handle_data_convert(job_id, payload)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    def _handle_data_gen(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle data generation job.

        Payload:
            generator: Generator name (binary_arithmetic, word_puzzles, etc.)
            count: Number of examples to generate
            level: Optional level/difficulty
            seed: Optional random seed
        """
        generator = payload.get("generator", "binary_arithmetic")
        count = payload.get("count", 1000)
        level = payload.get("level")
        seed = payload.get("seed")

        logger.info(f"Generating {count} examples with {generator}")

        try:
            # Try to use skill engine for generation
            from guild.skills import get_engine

            engine = get_engine()

            # Map generator name to skill
            skill_map = {
                "binary_arithmetic": "bin",
                "binary": "bin",
                "bin": "bin",
                "word_puzzles": "sy",
                "sy": "sy",
                "syllo": "sy",
            }

            skill_id = skill_map.get(generator, generator)

            # Generate training examples
            examples = []
            batch_size = min(count, 100)

            for _ in range(0, count, batch_size):
                try:
                    batch = engine.generate_eval_batch(
                        skill_id,
                        level=level or 1,
                        count=batch_size,
                    )
                    if batch and batch.problems:
                        for problem in batch.problems:
                            examples.append({
                                "prompt": problem.prompt,
                                "response": problem.expected,
                                "metadata": {
                                    "skill": skill_id,
                                    "level": level,
                                    "primitive": problem.primitive_id,
                                },
                            })
                except Exception as e:
                    logger.warning(f"Batch generation failed: {e}")

            # Save to file
            output_file = self.output_dir / f"{job_id}_{generator}.jsonl"
            with open(output_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            return {
                "success": True,
                "generator": generator,
                "count": len(examples),
                "output_file": str(output_file),
            }

        except ImportError:
            # Fallback to simple generation
            logger.warning("Skill engine not available, using fallback")
            return self._fallback_data_gen(job_id, generator, count)

        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _fallback_data_gen(
        self,
        job_id: str,
        generator: str,
        count: int,
    ) -> Dict[str, Any]:
        """Fallback data generation without skill engine."""
        import random

        examples = []

        if generator in ("binary_arithmetic", "binary", "bin"):
            # Simple binary addition
            for _ in range(count):
                a = random.randint(0, 15)
                b = random.randint(0, 15)
                result = a + b
                examples.append({
                    "prompt": f"Compute: {a} + {b} = ?",
                    "response": f"The answer is {result}.",
                })
        else:
            # Generic examples
            for i in range(count):
                examples.append({
                    "prompt": f"Example {i}",
                    "response": f"Response {i}",
                })

        output_file = self.output_dir / f"{job_id}_{generator}.jsonl"
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        return {
            "success": True,
            "generator": generator,
            "count": len(examples),
            "output_file": str(output_file),
            "fallback": True,
        }

    def _handle_data_filter(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle data filtering job.

        Payload:
            input_file: Path to input JSONL file
            filter_type: Type of filter (dedup, quality, length, etc.)
            params: Filter-specific parameters
        """
        input_file = payload.get("input_file")
        filter_type = payload.get("filter_type", "dedup")
        params = payload.get("params", {})

        if not input_file or not Path(input_file).exists():
            return {"success": False, "error": f"Input file not found: {input_file}"}

        logger.info(f"Filtering {input_file} with {filter_type}")

        try:
            # Read input
            examples = []
            with open(input_file) as f:
                for line in f:
                    examples.append(json.loads(line))

            original_count = len(examples)

            # Apply filter
            if filter_type == "dedup":
                # Simple deduplication by prompt
                seen = set()
                filtered = []
                for ex in examples:
                    key = ex.get("prompt", "")
                    if key not in seen:
                        seen.add(key)
                        filtered.append(ex)
                examples = filtered

            elif filter_type == "length":
                # Filter by length
                min_len = params.get("min_length", 10)
                max_len = params.get("max_length", 10000)
                examples = [
                    ex for ex in examples
                    if min_len <= len(ex.get("prompt", "")) + len(ex.get("response", "")) <= max_len
                ]

            elif filter_type == "quality":
                # Placeholder for quality filtering
                # In practice, this would use a quality model
                pass

            # Save filtered output
            output_file = self.output_dir / f"{job_id}_filtered.jsonl"
            with open(output_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            return {
                "success": True,
                "filter_type": filter_type,
                "original_count": original_count,
                "filtered_count": len(examples),
                "removed_count": original_count - len(examples),
                "output_file": str(output_file),
            }

        except Exception as e:
            logger.error(f"Data filtering failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _handle_data_convert(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle data conversion job.

        Payload:
            input_file: Path to input file
            input_format: Source format (jsonl, csv, parquet, etc.)
            output_format: Target format
            template: Optional conversion template
        """
        input_file = payload.get("input_file")
        input_format = payload.get("input_format", "jsonl")
        output_format = payload.get("output_format", "jsonl")
        template = payload.get("template")

        if not input_file or not Path(input_file).exists():
            return {"success": False, "error": f"Input file not found: {input_file}"}

        logger.info(f"Converting {input_file} from {input_format} to {output_format}")

        try:
            # Read input
            examples = []
            input_path = Path(input_file)

            if input_format == "jsonl":
                with open(input_path) as f:
                    for line in f:
                        examples.append(json.loads(line))

            elif input_format == "json":
                with open(input_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        examples = data
                    else:
                        examples = [data]

            elif input_format == "csv":
                import csv
                with open(input_path) as f:
                    reader = csv.DictReader(f)
                    examples = list(reader)

            else:
                return {"success": False, "error": f"Unknown input format: {input_format}"}

            # Apply template if provided
            if template:
                converted = []
                for ex in examples:
                    try:
                        new_ex = {}
                        for key, value_template in template.items():
                            if isinstance(value_template, str) and "{" in value_template:
                                new_ex[key] = value_template.format(**ex)
                            else:
                                new_ex[key] = ex.get(value_template, value_template)
                        converted.append(new_ex)
                    except KeyError as e:
                        logger.warning(f"Template error: {e}")
                examples = converted

            # Write output
            output_file = self.output_dir / f"{job_id}_converted.{output_format}"

            if output_format == "jsonl":
                with open(output_file, "w") as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")

            elif output_format == "json":
                with open(output_file, "w") as f:
                    json.dump(examples, f, indent=2)

            elif output_format == "csv":
                import csv
                if examples:
                    with open(output_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=examples[0].keys())
                        writer.writeheader()
                        writer.writerows(examples)

            return {
                "success": True,
                "input_format": input_format,
                "output_format": output_format,
                "count": len(examples),
                "output_file": str(output_file),
            }

        except Exception as e:
            logger.error(f"Data conversion failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Data Forge Worker - Generate and process data")
    parser.add_argument("--port", type=int, default=8900, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--device", help="Device ID (default from TRAINING_DEVICE_ID)")
    parser.add_argument("--output-dir", help="Output directory for generated data")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent jobs")

    args = parser.parse_args()

    config = WorkerConfig(
        device_id=args.device or os.environ.get("TRAINING_DEVICE_ID", "data_forge"),
        max_concurrent=args.max_concurrent,
    )

    worker = DataForgeWorker(
        config=config,
        output_dir=args.output_dir,
    )

    worker.run(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
