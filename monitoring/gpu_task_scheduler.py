#!/usr/bin/env python3
"""
GPU Task Scheduler - Central coordinator for 3090 GPU workloads

Monitors GPU utilization and dispatches tasks from a priority queue
to ensure efficient use of GPU resources (target: 20-80% utilization).

Runs on the 3090 machine alongside the inference server.

Features:
- Priority queue with task types (CRITICAL > HIGH > NORMAL > LOW > IDLE)
- GPU utilization monitoring with adaptive scheduling
- REST API for task submission from any machine
- Task execution tracking and metrics
- Automatic "fill the gaps" when GPU is idle

API Endpoints:
- POST /api/tasks/submit - Submit a task
- GET /api/tasks/queue - View current queue
- GET /api/tasks/{task_id} - Get task status
- GET /api/tasks/active - Currently running task
- DELETE /api/tasks/{task_id} - Cancel a task
- GET /api/gpu/stats - GPU utilization stats
- GET /api/metrics - Scheduler metrics
- GET /api/health - Health check

Usage:
    # On 3090:
    python3 gpu_task_scheduler.py --port 8766

    # From 4090 or anywhere:
    curl -X POST http://192.168.x.x:8766/api/tasks/submit \
         -H "Content-Type: application/json" \
         -d '{"task_type": "curriculum_eval", "params": {...}}'
"""

import argparse
import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import importlib.util

from flask import Flask, jsonify, request
from flask_cors import CORS

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPUTaskScheduler")


class Priority(IntEnum):
    """Task priority levels (lower = higher priority)"""
    CRITICAL = 0   # Urgent - run immediately if possible
    HIGH = 1       # Important - curriculum evals, regressions
    NORMAL = 2     # Standard - automated testing, baselines
    LOW = 3        # Background - adversarial mining, corrections
    IDLE = 4       # Fill gaps - run when nothing else to do


# Default priorities for task types
TASK_TYPE_PRIORITIES = {
    "critical_eval": Priority.CRITICAL,
    "curriculum_eval": Priority.HIGH,
    "regression_check": Priority.HIGH,
    "baseline_test": Priority.NORMAL,
    "automated_test": Priority.NORMAL,
    "checkpoint_eval": Priority.NORMAL,
    "self_correction": Priority.LOW,
    "adversarial_mine": Priority.LOW,
    "live_prediction": Priority.IDLE,
    "idle_warmup": Priority.IDLE,
    # Analytics tasks
    "collect_embeddings": Priority.LOW,      # GPU intensive, run when idle
    "hard_example_eval": Priority.NORMAL,    # Quick inference tests
    "generate_visualizations": Priority.IDLE, # CPU only, lowest priority
    "update_learning_history": Priority.IDLE, # Quick file append
    # Phase 3: Targeted corrections
    "generate_corrections": Priority.LOW,    # Generate training data for error types
    "error_feedback_loop": Priority.NORMAL,  # Full eval + generate + queue cycle
    # Phase 4: Self-improving pipeline
    "impact_snapshot": Priority.IDLE,        # Take performance snapshot
    "self_improve_cycle": Priority.NORMAL,   # Full self-improving cycle
    # Phase 6: Hard example discovery
    "expand_hard_examples": Priority.LOW,    # Discover new hard examples from failures
    # Dashboard status tasks
    "confidence_calibration": Priority.LOW,   # Calibrate model confidence scores
    "curriculum_optimization": Priority.LOW,  # Optimize curriculum difficulty
    "regression_monitoring": Priority.NORMAL, # Monitor for regressions
    "checkpoint_sync": Priority.LOW,          # Sync checkpoints between machines
    # Storage tasks
    "storage_check": Priority.LOW,            # Check Synology NAS storage status
    "storage_sync": Priority.LOW,             # Sync checkpoints to NAS
}


class TaskStatus:
    """Task status constants"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class Task:
    """A task to be executed on the GPU"""
    priority: int
    task_id: str = field(compare=False)
    task_type: str = field(compare=False)
    params: Dict[str, Any] = field(compare=False, default_factory=dict)
    callback_url: Optional[str] = field(compare=False, default=None)
    submitted_at: str = field(compare=False, default_factory=lambda: datetime.now().isoformat())
    status: str = field(compare=False, default=TaskStatus.QUEUED)
    started_at: Optional[str] = field(compare=False, default=None)
    completed_at: Optional[str] = field(compare=False, default=None)
    result: Optional[Dict] = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)

    def to_dict(self) -> Dict:
        return asdict(self)


class GPUMonitor:
    """Monitors GPU utilization"""

    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self.current_stats = {}
        self.history = []
        self.max_history = 300  # 10 minutes at 2s intervals
        self._running = False
        self._thread = None

    def start(self):
        """Start monitoring thread"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitor started")

    def stop(self):
        """Stop monitoring thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                self.current_stats = self._get_gpu_stats()
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    **self.current_stats
                })
                # Trim history
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")

            time.sleep(self.poll_interval)

    def _get_gpu_stats(self) -> Dict:
        """Get current GPU stats via nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                return {
                    "utilization_pct": float(parts[0]),
                    "memory_used_mb": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "temperature_c": float(parts[3]),
                    "memory_pct": (float(parts[1]) / float(parts[2])) * 100
                }
        except Exception as e:
            logger.error(f"nvidia-smi failed: {e}")

        return {"utilization_pct": 0, "memory_used_mb": 0, "memory_total_mb": 24576, "temperature_c": 0, "memory_pct": 0}

    def get_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        return self.current_stats.get("utilization_pct", 0)

    def get_avg_utilization(self, window_seconds: int = 60) -> float:
        """Get average utilization over a time window"""
        if not self.history:
            return 0

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [h for h in self.history
                  if datetime.fromisoformat(h["timestamp"]) > cutoff]

        if not recent:
            return self.get_utilization()

        return sum(h.get("utilization_pct", 0) for h in recent) / len(recent)


class TaskExecutor:
    """Executes tasks and manages results"""

    def __init__(self, inference_url: str = "http://localhost:8765"):
        self.inference_url = inference_url
        self.task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register built-in task handlers"""
        self.task_handlers = {
            "curriculum_eval": self._handle_curriculum_eval,
            "baseline_test": self._handle_baseline_test,
            "automated_test": self._handle_automated_test,
            "live_prediction": self._handle_live_prediction,
            "self_correction": self._handle_self_correction,
            "adversarial_mine": self._handle_adversarial_mine,
            "checkpoint_eval": self._handle_checkpoint_eval,
            "idle_warmup": self._handle_idle_warmup,
            # Analytics tasks
            "collect_embeddings": self._handle_collect_embeddings,
            "hard_example_eval": self._handle_hard_example_eval,
            "generate_visualizations": self._handle_generate_visualizations,
            "update_learning_history": self._handle_update_learning_history,
            # Phase 3: Targeted corrections
            "generate_corrections": self._handle_generate_corrections,
            "error_feedback_loop": self._handle_error_feedback_loop,
            # Phase 4: Self-improving pipeline
            "impact_snapshot": self._handle_impact_snapshot,
            "self_improve_cycle": self._handle_self_improve_cycle,
            # Phase 6: Hard example discovery
            "expand_hard_examples": self._handle_expand_hard_examples,
            # Dashboard status tasks
            "confidence_calibration": self._handle_confidence_calibration,
            "curriculum_optimization": self._handle_curriculum_optimization,
            "regression_monitoring": self._handle_regression_monitoring,
            "checkpoint_sync": self._handle_checkpoint_sync,
            # Storage tasks
            "storage_check": self._handle_storage_check,
            "storage_sync": self._handle_storage_sync,
        }

    def register_handler(self, task_type: str, handler: Callable):
        """Register a custom task handler"""
        self.task_handlers[task_type] = handler

    def execute(self, task: Task) -> Dict:
        """Execute a task and return results"""
        handler = self.task_handlers.get(task.task_type)

        if not handler:
            raise ValueError(f"Unknown task type: {task.task_type}")

        logger.info(f"Executing task {task.task_id}: {task.task_type}")
        return handler(task.params)

    def _handle_curriculum_eval(self, params: Dict) -> Dict:
        """Run curriculum evaluation"""
        import requests

        skill = params.get("skill", "syllo")
        num_problems = params.get("num_problems", 20)

        # This would call the curriculum eval logic
        # For now, return placeholder
        results = {
            "skill": skill,
            "problems_tested": num_problems,
            "accuracy": 0.0,
            "timestamp": datetime.now().isoformat()
        }

        # Try to call actual eval if available
        try:
            # Import and run actual evaluation
            spec = importlib.util.spec_from_file_location(
                "curriculum_eval",
                "/path/to/training/monitoring/curriculum_eval_loop.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                loop = module.CurriculumEvalLoop(
                    base_dir="/path/to/training",
                    inference_host="localhost",
                    problems_per_eval=num_problems
                )
                results = loop.run_evaluation(skill)
        except Exception as e:
            logger.warning(f"Could not run full curriculum eval: {e}")
            results["error"] = str(e)

        return results

    def _handle_baseline_test(self, params: Dict) -> Dict:
        """Run baseline test"""
        skill = params.get("skill", "primitives")
        max_per_difficulty = params.get("max_per_difficulty", 30)
        model_path = params.get("model_path")
        tag = params.get("tag", "baseline")

        # Run baseline test script
        cmd = [
            "python3", "/path/to/training/tools/analysis/run_baseline_test.py",
            "--tag", tag,
            "--skill", skill,
            "--max-per-difficulty", str(max_per_difficulty),
            "--base-dir", "/path/to/training"
        ]
        if model_path:
            cmd.extend(["--model-path", model_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return {
                "success": result.returncode == 0,
                "output": result.stdout[-2000:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout after 600s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _handle_automated_test(self, params: Dict) -> Dict:
        """Run automated testing - actually runs the test suite"""
        base_dir = params.get("base_dir", "/path/to/training")
        validation_file = params.get("validation_file")

        try:
            # Import and run actual testing daemon
            spec = importlib.util.spec_from_file_location(
                "automated_testing",
                f"{base_dir}/monitoring/automated_testing_daemon.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                daemon = module.AutomatedTestingDaemon(
                    base_dir=base_dir,
                    api_url=self.inference_url,
                    interval=0,  # Not used for single run
                    validation_file=validation_file
                )

                # Get current checkpoint
                checkpoint_name = daemon.get_current_checkpoint_path()
                if checkpoint_name:
                    checkpoint_name = str(checkpoint_name.name)
                else:
                    checkpoint_name = "current"

                # Run test suite
                results = daemon.run_test_suite(checkpoint_name)

                # Save results to status file so dashboard can display them
                daemon.save_results(results)

                return {
                    "task": "automated_testing",
                    "total": results.get("total", 0),
                    "correct": results.get("correct", 0),
                    "accuracy": results.get("accuracy", 0.0),
                    "by_difficulty": dict(results.get("by_difficulty", {})),
                    "checkpoint": checkpoint_name,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Automated testing failed: {e}")
            return {
                "task": "automated_testing",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _handle_live_prediction(self, params: Dict) -> Dict:
        """Make live predictions"""
        import requests

        prompts = params.get("prompts", ["Hello, how are you?"])
        results = []

        for prompt in prompts[:5]:  # Max 5 predictions
            try:
                resp = requests.post(
                    f"{self.inference_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                if resp.ok:
                    data = resp.json()
                    results.append({
                        "prompt": prompt[:100],
                        "response": data["choices"][0]["message"]["content"][:200]
                    })
            except Exception as e:
                results.append({"prompt": prompt[:100], "error": str(e)})

        return {"predictions": results}

    def _handle_self_correction(self, params: Dict) -> Dict:
        """Run self-correction analysis - processes errors and generates corrections"""
        base_dir = params.get("base_dir", "/path/to/training")
        error_threshold = params.get("error_threshold", 50)
        batch_size = params.get("batch_size", 100)

        try:
            # Import and run actual self-correction loop
            spec = importlib.util.spec_from_file_location(
                "self_correction",
                f"{base_dir}/monitoring/self_correction_loop.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                loop = module.SelfCorrectionLoop(
                    api_url=self.inference_url,
                    base_dir=base_dir,
                    batch_size=batch_size,
                    error_threshold=error_threshold
                )

                # Check for unvalidated files and process them
                from pathlib import Path
                unvalidated_dir = Path(base_dir) / "queue" / "unvalidated"
                files_processed = 0
                total_errors = 0
                corrections_generated = 0

                if unvalidated_dir.exists():
                    for jsonl_file in unvalidated_dir.glob("*.jsonl"):
                        try:
                            loop.run_validation_pipeline(jsonl_file)
                            files_processed += 1
                        except Exception as e:
                            logger.warning(f"Failed to process {jsonl_file}: {e}")

                # Process error batch if we have enough
                if len(loop.error_cache) >= error_threshold:
                    loop.process_error_batch()

                # Save status to file so dashboard can display it
                loop.update_status()

                return {
                    "task": "self_correction",
                    "files_processed": files_processed,
                    "tested": loop.stats.get("tested", 0),
                    "errors_captured": loop.stats.get("incorrect", 0),
                    "corrections_generated": loop.stats.get("corrections_generated", 0),
                    "patterns_found": loop.stats.get("patterns_found", 0),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            return {
                "task": "self_correction",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _handle_adversarial_mine(self, params: Dict) -> Dict:
        """Run adversarial mining - finds hard examples from model failures"""
        base_dir = params.get("base_dir", "/path/to/training")
        batch_size = params.get("batch_size", 100)
        categories = params.get("categories", ["negation", "double_negation", "quantifier"])

        try:
            # Import and run actual adversarial miner
            spec = importlib.util.spec_from_file_location(
                "adversarial_miner",
                f"{base_dir}/monitoring/adversarial_miner.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                miner = module.AdversarialMiner(
                    api_url=self.inference_url,
                    base_dir=base_dir
                )

                # Run mining for each category
                results = {
                    "task": "adversarial_mine",
                    "examples_mined": 0,
                    "by_category": {},
                    "timestamp": datetime.now().isoformat()
                }

                for category in categories:
                    try:
                        mined = miner.mine_category(category, max_examples=batch_size)
                        results["examples_mined"] += len(mined)
                        results["by_category"][category] = len(mined)
                    except Exception as e:
                        logger.warning(f"Mining {category} failed: {e}")
                        results["by_category"][category] = 0

                # Save results to status file so dashboard can display them
                miner._save_results()

                return results
        except Exception as e:
            logger.error(f"Adversarial mining failed: {e}")
            return {
                "task": "adversarial_mine",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _handle_checkpoint_eval(self, params: Dict) -> Dict:
        """Evaluate a specific checkpoint against validation data"""
        checkpoint_path = params.get("checkpoint_path")

        # Note: model_comparison_engine was removed - use curriculum eval instead
        return {
            "task": "checkpoint_eval",
            "checkpoint": checkpoint_path,
            "error": "Use curriculum_eval or ledger for checkpoint evaluation",
            "timestamp": datetime.now().isoformat()
        }

    def _handle_idle_warmup(self, params: Dict) -> Dict:
        """Keep GPU warm with light work"""
        import requests

        try:
            resp = requests.post(
                f"{self.inference_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                timeout=30
            )
            return {"warmup": "success", "response_time_ms": resp.elapsed.total_seconds() * 1000}
        except Exception as e:
            return {"warmup": "failed", "error": str(e)}

    def _handle_collect_embeddings(self, params: Dict) -> Dict:
        """Collect embeddings for the probe set (GPU-intensive)"""
        base_dir = params.get("base_dir", "/path/to/training")
        model_path = params.get("model_path")

        try:
            spec = importlib.util.spec_from_file_location(
                "embedding_tracker",
                f"{base_dir}/monitoring/analytics/embedding_tracker.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                collector = module.EmbeddingCollector(base_dir, model_path)
                output_path = collector.collect_for_probe_set()

                if output_path:
                    return {
                        "task": "collect_embeddings",
                        "output": str(output_path),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"task": "collect_embeddings", "error": "Collection failed"}
        except Exception as e:
            logger.error(f"Embedding collection failed: {e}")
            return {"task": "collect_embeddings", "error": str(e)}

    def _handle_hard_example_eval(self, params: Dict) -> Dict:
        """Evaluate model on hard examples"""
        base_dir = params.get("base_dir", "/path/to/training")
        checkpoint = params.get("checkpoint", "current")

        try:
            spec = importlib.util.spec_from_file_location(
                "hard_example_tracker",
                f"{base_dir}/monitoring/analytics/hard_example_tracker.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                tracker = module.HardExampleTracker(base_dir, self.inference_url)
                entry = tracker.evaluate_all(checkpoint)

                return {
                    "task": "hard_example_eval",
                    "correct": entry.total_correct,
                    "total": entry.total,
                    "accuracy": entry.accuracy,
                    "error_types": entry.error_types,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Hard example eval failed: {e}")
            return {"task": "hard_example_eval", "error": str(e)}

    def _handle_generate_visualizations(self, params: Dict) -> Dict:
        """Generate all visualization outputs (CPU-only)"""
        base_dir = params.get("base_dir", "/path/to/training")
        viz_types = params.get("types", ["skill_radar", "hard_example_board"])

        results = {"task": "generate_visualizations", "generated": []}

        try:
            # Skill radar
            if "skill_radar" in viz_types:
                spec = importlib.util.spec_from_file_location(
                    "skill_radar",
                    f"{base_dir}/monitoring/analytics/skill_radar.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    generator = module.SkillRadarGenerator(base_dir)
                    path = generator.generate_current()
                    results["generated"].append(str(path))

            # Hard example board
            if "hard_example_board" in viz_types:
                spec = importlib.util.spec_from_file_location(
                    "hard_example_tracker",
                    f"{base_dir}/monitoring/analytics/hard_example_tracker.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    tracker = module.HardExampleTracker(base_dir)
                    path = tracker.generate_board_image()
                    if path:
                        results["generated"].append(str(path))

            # UMAP visualization
            if "umap" in viz_types:
                spec = importlib.util.spec_from_file_location(
                    "embedding_tracker",
                    f"{base_dir}/monitoring/analytics/embedding_tracker.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    visualizer = module.EmbeddingVisualizer(base_dir)
                    path = visualizer.visualize_latest()
                    if path:
                        results["generated"].append(str(path))

            results["timestamp"] = datetime.now().isoformat()
            return results

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"task": "generate_visualizations", "error": str(e)}

    def _handle_update_learning_history(self, params: Dict) -> Dict:
        """Append current metrics to learning history"""
        base_dir = params.get("base_dir", "/path/to/training")

        try:
            spec = importlib.util.spec_from_file_location(
                "learning_history",
                f"{base_dir}/monitoring/analytics/learning_history.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                history = module.LearningHistory(base_dir)
                snapshot = history.collect_current_metrics()
                history.append(snapshot)

                return {
                    "task": "update_learning_history",
                    "step": snapshot.step,
                    "train_loss": snapshot.train_loss,
                    "val_loss": snapshot.val_loss,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Learning history update failed: {e}")
            return {"task": "update_learning_history", "error": str(e)}

    def _handle_generate_corrections(self, params: Dict) -> Dict:
        """Generate targeted training data for specific error types"""
        base_dir = params.get("base_dir", "/path/to/training")
        error_type = params.get("error_type")  # Optional: specific type
        count = params.get("count", 30)
        queue_data = params.get("queue", True)

        try:
            spec = importlib.util.spec_from_file_location(
                "pattern_generator",
                f"{base_dir}/monitoring/analytics/pattern_generator.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                generator = module.PatternGenerator(base_dir)

                if error_type:
                    # Generate for specific type
                    examples = generator.generate_for_type(error_type, count)
                    if queue_data:
                        filepath = generator.save_to_inbox(examples, error_type)
                        return {
                            "task": "generate_corrections",
                            "error_type": error_type,
                            "examples_generated": len(examples),
                            "queued_file": str(filepath),
                            "timestamp": datetime.now().isoformat()
                        }
                    return {
                        "task": "generate_corrections",
                        "error_type": error_type,
                        "examples_generated": len(examples),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    # Auto-generate based on error distribution
                    if queue_data:
                        files = generator.generate_and_queue(count)
                        return {
                            "task": "generate_corrections",
                            "files_created": len(files),
                            "error_types": list(files.keys()),
                            "total_examples": sum(1 for _ in files.values()),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        all_examples = generator.generate_auto(count)
                        total = sum(len(ex) for ex in all_examples.values())
                        return {
                            "task": "generate_corrections",
                            "examples_generated": total,
                            "by_type": {k: len(v) for k, v in all_examples.items()},
                            "timestamp": datetime.now().isoformat()
                        }

        except Exception as e:
            logger.error(f"Generate corrections failed: {e}")
            return {"task": "generate_corrections", "error": str(e)}

    def _handle_error_feedback_loop(self, params: Dict) -> Dict:
        """
        Full feedback loop: Evaluate -> Analyze errors -> Generate corrections -> Queue

        This is the key Phase 3 integration - runs the complete cycle.
        """
        base_dir = params.get("base_dir", "/path/to/training")
        corrections_per_error = params.get("corrections_per_error", 30)

        results = {
            "task": "error_feedback_loop",
            "steps": [],
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Step 1: Run hard example evaluation
            logger.info("Step 1: Evaluating hard examples...")
            eval_result = self._handle_hard_example_eval({"base_dir": base_dir})
            results["steps"].append({
                "step": "hard_example_eval",
                "accuracy": eval_result.get("accuracy", 0),
                "error_types": eval_result.get("error_types", {})
            })

            # Step 2: Check if we need corrections
            error_types = eval_result.get("error_types", {})
            if not error_types:
                results["message"] = "No errors found - skipping correction generation"
                return results

            # Step 3: Generate corrections for each error type
            logger.info(f"Step 2: Generating corrections for {len(error_types)} error types...")
            gen_result = self._handle_generate_corrections({
                "base_dir": base_dir,
                "count": corrections_per_error,
                "queue": True
            })
            results["steps"].append({
                "step": "generate_corrections",
                "files_created": gen_result.get("files_created", 0),
                "error_types": gen_result.get("error_types", [])
            })

            # Step 4: Summary
            results["summary"] = {
                "accuracy_before": eval_result.get("accuracy", 0),
                "errors_targeted": list(error_types.keys()),
                "corrections_queued": gen_result.get("files_created", 0)
            }

            logger.info(f"Feedback loop complete: {results['summary']}")
            return results

        except Exception as e:
            logger.error(f"Error feedback loop failed: {e}")
            results["error"] = str(e)
            return results

    def _handle_impact_snapshot(self, params: Dict) -> Dict:
        """Take a performance snapshot for impact tracking."""
        base_dir = params.get("base_dir", "/path/to/training")
        notes = params.get("notes")

        try:
            spec = importlib.util.spec_from_file_location(
                "impact_tracker",
                f"{base_dir}/monitoring/analytics/impact_tracker.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                tracker = module.ImpactTracker(base_dir)
                snapshot = tracker.take_snapshot(notes)

                return {
                    "task": "impact_snapshot",
                    "accuracy": snapshot.hard_example_accuracy,
                    "step": snapshot.step,
                    "corrections_pending": snapshot.corrections_pending,
                    "timestamp": snapshot.timestamp
                }

        except Exception as e:
            logger.error(f"Impact snapshot failed: {e}")
            return {"task": "impact_snapshot", "error": str(e)}

    def _handle_self_improve_cycle(self, params: Dict) -> Dict:
        """
        Full self-improving cycle:
        1. Take baseline snapshot
        2. Run hard example evaluation
        3. Generate targeted corrections
        4. Queue corrections for training
        5. (Future: After training, compare to baseline)

        This is designed to run periodically (e.g., every 2-4 hours).
        """
        base_dir = params.get("base_dir", "/path/to/training")
        corrections_per_error = params.get("corrections_per_error", 25)
        min_errors_to_correct = params.get("min_errors", 3)

        results = {
            "task": "self_improve_cycle",
            "phases": [],
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Phase 1: Take baseline snapshot
            logger.info("Phase 1: Taking baseline snapshot...")
            snapshot_result = self._handle_impact_snapshot({
                "base_dir": base_dir,
                "notes": "Pre-cycle baseline"
            })
            results["phases"].append({
                "phase": "baseline_snapshot",
                "accuracy": snapshot_result.get("accuracy", 0),
                "step": snapshot_result.get("step", 0)
            })

            # Phase 2: Evaluate hard examples
            logger.info("Phase 2: Evaluating hard examples...")
            eval_result = self._handle_hard_example_eval({"base_dir": base_dir})
            results["phases"].append({
                "phase": "hard_example_eval",
                "accuracy": eval_result.get("accuracy", 0),
                "error_types": eval_result.get("error_types", {})
            })

            # Phase 3: Check if corrections needed
            error_count = sum(eval_result.get("error_types", {}).values())
            if error_count < min_errors_to_correct:
                results["message"] = f"Only {error_count} errors - below threshold ({min_errors_to_correct})"
                results["action"] = "skipped_corrections"
                return results

            # Phase 4: Generate corrections
            logger.info(f"Phase 3: Generating corrections for {error_count} errors...")
            gen_result = self._handle_generate_corrections({
                "base_dir": base_dir,
                "count": corrections_per_error,
                "queue": True
            })
            results["phases"].append({
                "phase": "generate_corrections",
                "files_created": gen_result.get("files_created", 0),
                "categories": gen_result.get("error_types", [])
            })

            # Summary
            results["summary"] = {
                "baseline_accuracy": snapshot_result.get("accuracy", 0),
                "errors_found": error_count,
                "corrections_queued": gen_result.get("files_created", 0),
                "next_action": "Train corrections, then run another cycle to measure impact"
            }

            logger.info(f"Self-improve cycle complete: {results['summary']}")
            return results

        except Exception as e:
            logger.error(f"Self-improve cycle failed: {e}")
            results["error"] = str(e)
            return results

    def _handle_expand_hard_examples(self, params: Dict) -> Dict:
        """
        Discover and add new hard examples from training failures.
        Grows the test suite organically based on real weaknesses.
        """
        base_dir = params.get("base_dir", "/path/to/training")
        max_new = params.get("max_new", 3)

        try:
            spec = importlib.util.spec_from_file_location(
                "hard_example_expander",
                f"{base_dir}/monitoring/analytics/hard_example_expander.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                expander = module.HardExampleExpander(base_dir, self.inference_url)

                # Generate candidates from failures
                candidates = expander.generate_candidates()
                expander.candidates.extend(candidates)

                # Test and expand
                added = expander.expand_hard_examples(max_new)

                summary = expander.get_summary()

                return {
                    "task": "expand_hard_examples",
                    "candidates_generated": len(candidates),
                    "examples_added": len(added),
                    "new_examples": [ex["id"] for ex in added],
                    "total_hard_examples": summary["current_hard_examples"],
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Hard example expansion failed: {e}")
            return {"task": "expand_hard_examples", "error": str(e)}

    def _handle_confidence_calibration(self, params: Dict) -> Dict:
        """Run confidence calibration - measures model calibration on validation set"""
        base_dir = params.get("base_dir", "/path/to/training")
        num_samples = params.get("num_samples", 100)

        try:
            spec = importlib.util.spec_from_file_location(
                "confidence_calibrator",
                f"{base_dir}/monitoring/confidence_calibrator.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                calibrator = module.ConfidenceCalibrator(
                    base_dir=base_dir,
                    test_samples=num_samples
                )

                # Run calibration (uses inference_url from config or default)
                results = calibrator.run_calibration()

                # Save results to status file
                if hasattr(calibrator, '_save_results'):
                    calibrator._save_results()

                return {
                    "task": "confidence_calibration",
                    "samples_tested": getattr(calibrator, 'test_samples', 0),
                    "results": results if isinstance(results, dict) else {},
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return {"task": "confidence_calibration", "error": str(e)}

    def _handle_curriculum_optimization(self, params: Dict) -> Dict:
        """Run curriculum optimization - adjusts difficulty based on performance"""
        base_dir = params.get("base_dir", "/path/to/training")

        try:
            spec = importlib.util.spec_from_file_location(
                "curriculum_optimizer",
                f"{base_dir}/monitoring/curriculum_optimizer.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                optimizer = module.CurriculumOptimizer(base_dir=base_dir)

                # Run optimization cycle
                results = optimizer.optimize_curriculum()

                # Save results to status file
                if hasattr(optimizer, '_save_results'):
                    optimizer._save_results()

                return {
                    "task": "curriculum_optimization",
                    "results": results if isinstance(results, dict) else {},
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Curriculum optimization failed: {e}")
            return {"task": "curriculum_optimization", "error": str(e)}

    def _handle_regression_monitoring(self, params: Dict) -> Dict:
        """Run regression monitoring - checks for performance degradation"""
        base_dir = params.get("base_dir", "/path/to/training")
        threshold = params.get("threshold", 0.05)  # 5% drop = regression

        try:
            spec = importlib.util.spec_from_file_location(
                "regression_detector",
                f"{base_dir}/monitoring/regression_detector.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                detector = module.RegressionDetector(
                    api_url=self.inference_url,
                    base_dir=base_dir,
                    threshold=threshold
                )

                # Run regression check
                results = detector.check_regression()

                # Save results to status file
                if hasattr(detector, '_save_results'):
                    detector._save_results()

                return {
                    "task": "regression_monitoring",
                    "regression_detected": results.get("regression_detected", False),
                    "current_accuracy": results.get("current_accuracy", 0),
                    "baseline_accuracy": results.get("baseline_accuracy", 0),
                    "accuracy_delta": results.get("accuracy_delta", 0),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Regression monitoring failed: {e}")
            return {"task": "regression_monitoring", "error": str(e)}

    def _handle_checkpoint_sync(self, params: Dict) -> Dict:
        """Run checkpoint sync - syncs checkpoints between machines"""
        base_dir = params.get("base_dir", "/path/to/training")

        try:
            spec = importlib.util.spec_from_file_location(
                "checkpoint_sync",
                f"{base_dir}/monitoring/checkpoint_sync_daemon.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                syncer = module.CheckpointSyncDaemon(base_dir=base_dir)

                # Run sync check
                results = syncer.check_sync_status()

                # Save results to status file
                if hasattr(syncer, '_save_status'):
                    syncer._save_status()

                return {
                    "task": "checkpoint_sync",
                    "in_sync": results.get("in_sync", False),
                    "local_checkpoint": results.get("local_checkpoint", "unknown"),
                    "remote_checkpoint": results.get("remote_checkpoint", "unknown"),
                    "last_sync": results.get("last_sync", "never"),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Checkpoint sync failed: {e}")
            return {"task": "checkpoint_sync", "error": str(e)}

    def _handle_storage_check(self, params: Dict) -> Dict:
        """Check Synology NAS storage status"""
        base_dir = params.get("base_dir", "/path/to/training")

        try:
            spec = importlib.util.spec_from_file_location(
                "storage_manager",
                f"{base_dir}/monitoring/storage_manager.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Load credentials
                creds = module.load_credentials()
                if not creds.get("password"):
                    return {"task": "storage_check", "error": "No credentials found"}

                mgr = module.StorageManager(
                    host=creds.get("host", "192.168.x.x"),
                    username=creds.get("username", "user"),
                    password=creds["password"],
                    port=creds.get("port", 5001),
                    base_dir=base_dir
                )

                status = mgr.get_status()
                mgr._save_status()

                return {
                    "task": "storage_check",
                    "connected": status.get("connected", False),
                    "host": status.get("host"),
                    "health": status.get("health", "unknown"),
                    "total_tb": status.get("total_capacity_tb", 0),
                    "used_tb": status.get("used_capacity_tb", 0),
                    "usage_percent": status.get("usage_percent", 0),
                    "disks": len(status.get("disks", [])),
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return {"task": "storage_check", "error": str(e)}

    def _handle_storage_sync(self, params: Dict) -> Dict:
        """Sync files to Synology NAS using folder types"""
        base_dir = params.get("base_dir", "/path/to/training")
        local_path = params.get("local_path")
        folder_type = params.get("folder_type", "backups")  # checkpoints, models, backups, logs, etc.
        subfolder = params.get("subfolder", "")

        if not local_path:
            return {"task": "storage_sync", "error": "local_path required"}

        try:
            spec = importlib.util.spec_from_file_location(
                "storage_manager",
                f"{base_dir}/monitoring/storage_manager.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Use the new sync_to_nas helper
                result = module.sync_to_nas(
                    local_path=local_path,
                    folder_type=folder_type,
                    subfolder=subfolder,
                    base_dir=base_dir
                )

                return {
                    "task": "storage_sync",
                    "success": result.get("success", False),
                    "local_path": local_path,
                    "folder_type": folder_type,
                    "remote_path": result.get("remote_path"),
                    "error": result.get("error"),
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Storage sync failed: {e}")
            return {"task": "storage_sync", "error": str(e)}


class GPUTaskScheduler:
    """Main scheduler coordinating GPU workloads"""

    def __init__(
        self,
        min_utilization: float = 20.0,
        max_utilization: float = 80.0,
        inference_url: str = "http://localhost:8765",
        base_dir: str = "/path/to/training"
    ):
        self.min_utilization = min_utilization
        self.max_utilization = max_utilization
        self.base_dir = Path(base_dir)

        # Components
        self.gpu_monitor = GPUMonitor()
        self.executor = TaskExecutor(inference_url)

        # Task queue (priority queue)
        self.task_queue: queue.PriorityQueue[Task] = queue.PriorityQueue()
        self.tasks: Dict[str, Task] = {}  # task_id -> Task
        self.active_task: Optional[Task] = None

        # Metrics
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "total_execution_time_sec": 0,
            "started_at": datetime.now().isoformat(),
            "idle_tasks_dispatched": 0,
        }

        # Control
        self._running = False
        self._scheduler_thread = None
        self._lock = threading.Lock()

        # Status file
        self.status_file = self.base_dir / "status" / "gpu_scheduler.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the scheduler"""
        logger.info("="*60)
        logger.info("GPU Task Scheduler Starting")
        logger.info("="*60)
        logger.info(f"Min utilization target: {self.min_utilization}%")
        logger.info(f"Max utilization limit: {self.max_utilization}%")

        self._running = True
        self.gpu_monitor.start()

        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self._running = False
        self.gpu_monitor.stop()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        self._save_status()
        logger.info("Scheduler stopped")

    def submit_task(
        self,
        task_type: str,
        params: Optional[Dict] = None,
        priority: Optional[int] = None,
        callback_url: Optional[str] = None
    ) -> Task:
        """Submit a task to the queue"""
        task_id = f"task_{uuid.uuid4().hex[:12]}"

        # Determine priority
        if priority is None:
            priority = TASK_TYPE_PRIORITIES.get(task_type, Priority.NORMAL)

        task = Task(
            priority=priority,
            task_id=task_id,
            task_type=task_type,
            params=params or {},
            callback_url=callback_url
        )

        with self._lock:
            self.tasks[task_id] = task
            self.task_queue.put(task)
            self.metrics["tasks_submitted"] += 1

        logger.info(f"Task submitted: {task_id} ({task_type}) priority={priority}")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task"""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.QUEUED:
            task.status = TaskStatus.CANCELLED
            self.metrics["tasks_cancelled"] += 1
            logger.info(f"Task cancelled: {task_id}")
            return True
        return False

    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        queued_tasks = [t.to_dict() for t in self.tasks.values()
                       if t.status == TaskStatus.QUEUED]
        queued_tasks.sort(key=lambda x: x["priority"])

        return {
            "queue_length": len(queued_tasks),
            "tasks": queued_tasks[:20],  # First 20
            "active_task": self.active_task.to_dict() if self.active_task else None
        }

    def get_metrics(self) -> Dict:
        """Get scheduler metrics"""
        gpu_stats = self.gpu_monitor.current_stats
        avg_util = self.gpu_monitor.get_avg_utilization(60)

        return {
            **self.metrics,
            "queue_length": self.task_queue.qsize(),
            "active_task": self.active_task.task_type if self.active_task else None,
            "gpu_utilization_pct": gpu_stats.get("utilization_pct", 0),
            "gpu_utilization_avg_1min": avg_util,
            "gpu_memory_pct": gpu_stats.get("memory_pct", 0),
            "uptime_sec": (datetime.now() - datetime.fromisoformat(self.metrics["started_at"])).total_seconds()
        }

    def _scheduler_loop(self):
        """Main scheduling loop"""
        last_idle_check = time.time()
        idle_check_interval = 30  # Check for idle work every 30s

        while self._running:
            try:
                current_util = self.gpu_monitor.get_utilization()

                # Check if we should dispatch a task
                if self.active_task is None:
                    # No task running - check queue
                    if not self.task_queue.empty():
                        self._dispatch_next_task(current_util)
                    elif time.time() - last_idle_check > idle_check_interval:
                        # Check if we should dispatch idle work
                        if current_util < self.min_utilization:
                            self._dispatch_idle_task()
                        last_idle_check = time.time()

                # Save status periodically
                self._save_status()

                time.sleep(1)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
                time.sleep(5)

    def _dispatch_next_task(self, current_util: float):
        """Dispatch the next task from queue"""
        try:
            # Peek at next task
            task = self.task_queue.get_nowait()

            # Skip cancelled tasks
            if task.status == TaskStatus.CANCELLED:
                return

            # Check if we should wait due to high utilization
            if task.priority > Priority.HIGH and current_util > self.max_utilization:
                # Re-queue and wait
                self.task_queue.put(task)
                logger.debug(f"Delaying task {task.task_id} - GPU util {current_util}% > {self.max_utilization}%")
                return

            # Execute task
            self._execute_task(task)

        except queue.Empty:
            pass

    def _execute_task(self, task: Task):
        """Execute a task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        self.active_task = task

        logger.info(f"Executing task {task.task_id}: {task.task_type}")

        try:
            start_time = time.time()
            result = self.executor.execute(task)
            execution_time = time.time() - start_time

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now().isoformat()

            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time_sec"] += execution_time

            logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")

            # Callback if specified
            if task.callback_url:
                self._send_callback(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.task_id} failed: {e}")

        finally:
            self.active_task = None

    def _dispatch_idle_task(self):
        """Dispatch an idle task to keep GPU warm"""
        logger.debug("GPU idle - dispatching warmup task")
        self.submit_task("idle_warmup", priority=Priority.IDLE)
        self.metrics["idle_tasks_dispatched"] += 1

    def _send_callback(self, task: Task):
        """Send callback notification for completed task"""
        try:
            import requests
            requests.post(
                task.callback_url,
                json={
                    "task_id": task.task_id,
                    "status": task.status,
                    "result": task.result,
                    "error": task.error
                },
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Callback failed for {task.task_id}: {e}")

    def _save_status(self):
        """Save scheduler status to file"""
        try:
            status = {
                "scheduler_status": "running" if self._running else "stopped",
                "metrics": self.get_metrics(),
                "queue": self.get_queue_status(),
                "gpu_stats": self.gpu_monitor.current_stats,
                "last_updated": datetime.now().isoformat()
            }

            with open(self.status_file, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")


# Flask API
app = Flask(__name__)
CORS(app)

scheduler: Optional[GPUTaskScheduler] = None


@app.route("/api/health")
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "scheduler": "running" if scheduler and scheduler._running else "stopped",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/tasks/submit", methods=["POST"])
def submit_task():
    """Submit a new task"""
    data = request.get_json() or {}

    task_type = data.get("task_type")
    if not task_type:
        return jsonify({"error": "task_type required"}), 400

    task = scheduler.submit_task(
        task_type=task_type,
        params=data.get("params", {}),
        priority=data.get("priority"),
        callback_url=data.get("callback_url")
    )

    return jsonify({
        "task_id": task.task_id,
        "status": task.status,
        "priority": task.priority,
        "submitted_at": task.submitted_at
    })


@app.route("/api/tasks/queue")
def get_queue():
    """Get queue status"""
    return jsonify(scheduler.get_queue_status())


@app.route("/api/tasks/<task_id>")
def get_task(task_id):
    """Get task status"""
    task = scheduler.get_task(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task.to_dict())


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
def cancel_task(task_id):
    """Cancel a task"""
    if scheduler.cancel_task(task_id):
        return jsonify({"status": "cancelled"})
    return jsonify({"error": "Cannot cancel task"}), 400


@app.route("/api/tasks/active")
def get_active_task():
    """Get currently running task"""
    if scheduler.active_task:
        return jsonify(scheduler.active_task.to_dict())
    return jsonify({"active_task": None})


@app.route("/api/gpu/stats")
def get_gpu_stats():
    """Get GPU statistics"""
    return jsonify({
        "current": scheduler.gpu_monitor.current_stats,
        "avg_1min": scheduler.gpu_monitor.get_avg_utilization(60),
        "avg_5min": scheduler.gpu_monitor.get_avg_utilization(300),
        "history_length": len(scheduler.gpu_monitor.history)
    })


@app.route("/api/metrics")
def get_metrics():
    """Get scheduler metrics"""
    return jsonify(scheduler.get_metrics())


@app.route("/api/task-types")
def get_task_types():
    """List available task types and their priorities"""
    return jsonify({
        "task_types": TASK_TYPE_PRIORITIES,
        "priority_levels": {
            "CRITICAL": 0,
            "HIGH": 1,
            "NORMAL": 2,
            "LOW": 3,
            "IDLE": 4
        }
    })


def main():
    global scheduler

    parser = argparse.ArgumentParser(description="GPU Task Scheduler")
    parser.add_argument("--port", type=int, default=8766, help="API port")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--min-util", type=float, default=20.0,
                        help="Min GPU utilization target (%)")
    parser.add_argument("--max-util", type=float, default=80.0,
                        help="Max GPU utilization limit (%)")
    parser.add_argument("--inference-url", default="http://localhost:8765",
                        help="Inference server URL")
    parser.add_argument("--base-dir", default="/home/user/TRAINING",
                        help="Base directory for status files")

    args = parser.parse_args()

    # Create scheduler
    scheduler = GPUTaskScheduler(
        min_utilization=args.min_util,
        max_utilization=args.max_util,
        inference_url=args.inference_url,
        base_dir=args.base_dir
    )

    # Start scheduler
    scheduler.start()

    # Run Flask app
    try:
        logger.info(f"Starting API server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
