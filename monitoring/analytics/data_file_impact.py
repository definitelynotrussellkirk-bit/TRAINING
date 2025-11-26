#!/usr/bin/env python3
"""
Data File Impact Analyzer - Measure per-file training impact.

For each completed training file, measures:
- Accuracy change on validation set
- Loss change on key skills
- Which skills improved/regressed

This creates a feedback loop: you learn which kinds of training data
are most helpful and can tune your data generation accordingly.

Usage:
    python3 data_file_impact.py --base-dir ~/TRAINING --interval 300

Output:
    status/data_file_impact.jsonl - Append-only log of per-file impacts
    status/data_file_summary.json - Aggregated impact statistics

Integration:
    - Watches queue/recently_completed/ for new files
    - Uses existing validation set for quick impact measurement
    - Correlates with checkpoint steps to attribute changes
"""

import argparse
import json
import logging
import os
import sys
import time
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FileImpact:
    """Impact metrics for a single training file."""
    filename: str
    file_path: str
    step_before: int
    step_after: int
    timestamp: str

    # Per-skill deltas
    skill_impacts: Dict[str, float]  # skill -> accuracy_delta

    # Aggregate metrics
    avg_accuracy_delta: float
    best_skill: Optional[str]
    best_skill_delta: float
    worst_skill: Optional[str]
    worst_skill_delta: float

    # File metadata
    num_examples: Optional[int]
    file_size_kb: float
    detected_type: Optional[str]  # "syllable", "binary", "mixed", etc.

    # Assessment
    overall_impact: str  # "positive", "negative", "neutral", "mixed"


@dataclass
class ImpactSummary:
    """Aggregated impact statistics."""
    timestamp: str
    total_files_analyzed: int
    positive_impact_files: int
    negative_impact_files: int
    neutral_files: int

    # Best/worst files
    top_helpful_files: List[Dict[str, Any]]
    top_harmful_files: List[Dict[str, Any]]

    # By detected type
    impact_by_type: Dict[str, Dict[str, float]]


class DataFileImpactAnalyzer:
    """
    Analyze the impact of individual training files on model performance.

    Workflow:
    1. Watch recently_completed/ for new files
    2. Identify checkpoint before/after that file was trained
    3. Run quick validation on both checkpoints
    4. Compute delta per skill
    5. Log results to status/data_file_impact.jsonl

    This creates empirical evidence for which data types help most.

    Attributes:
        base_dir: Base TRAINING directory
        validation_path: Path to validation dataset
        ssh_host: Remote host for running inference (3090)
    """

    def __init__(
        self,
        base_dir: Path,
        validation_path: Optional[Path] = None,
        ssh_host: str = "192.168.x.x"
    ):
        """
        Initialize the data file impact analyzer.

        Args:
            base_dir: Base TRAINING directory
            validation_path: Path to validation dataset
            ssh_host: Remote host for inference
        """
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(exist_ok=True)

        self.validation_path = validation_path or (
            self.base_dir / "data" / "validation" / "val_easy_200.jsonl"
        )
        self.ssh_host = ssh_host

        self.impact_log_path = self.status_dir / "data_file_impact.jsonl"
        self.summary_path = self.status_dir / "data_file_summary.json"

        self.analyzed_files: set = set()
        self._load_analyzed_files()

    def _load_analyzed_files(self):
        """Load set of already-analyzed files."""
        if self.impact_log_path.exists():
            try:
                with open(self.impact_log_path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.analyzed_files.add(data.get("filename", ""))
            except Exception as e:
                logger.warning(f"Could not load impact log: {e}")

    def _get_recently_completed(self) -> List[Path]:
        """Get list of recently completed training files."""
        completed_dir = self.base_dir / "queue" / "recently_completed"
        if not completed_dir.exists():
            return []

        # Get .jsonl files, sorted by modification time
        files = sorted(
            completed_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return files[:20]  # Last 20

    def _find_checkpoints_for_file(self, file_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find checkpoints before and after a file was trained.

        Uses file modification time and checkpoint timestamps to correlate.

        Args:
            file_path: Path to the training file

        Returns:
            Tuple of (checkpoint_before, checkpoint_after)
        """
        file_mtime = file_path.stat().st_mtime
        checkpoints_dir = self.base_dir / "checkpoints"

        if not checkpoints_dir.exists():
            return None, None

        # Get all checkpoints with their modification times
        checkpoints = []
        for cp in checkpoints_dir.glob("checkpoint-*"):
            try:
                step = int(cp.name.split("-")[1])
                mtime = cp.stat().st_mtime
                checkpoints.append((step, mtime, cp))
            except (ValueError, IndexError):
                continue

        if len(checkpoints) < 2:
            return None, None

        # Sort by step
        checkpoints.sort(key=lambda x: x[0])

        # Find checkpoint just before file was completed
        before = None
        after = None

        for i, (step, mtime, cp) in enumerate(checkpoints):
            if mtime > file_mtime:
                # This checkpoint was created after file completion
                if i > 0:
                    before = checkpoints[i-1][2]
                after = cp
                break

        # If no checkpoint after, use last two
        if after is None and len(checkpoints) >= 2:
            before = checkpoints[-2][2]
            after = checkpoints[-1][2]

        return before, after

    def _extract_step(self, path: Path) -> Optional[int]:
        """Extract step number from checkpoint path."""
        try:
            return int(path.name.split("-")[1])
        except (ValueError, IndexError):
            return None

    def _run_validation(
        self,
        checkpoint_path: Path,
        skills: List[str] = None
    ) -> Dict[str, float]:
        """
        Run validation on a checkpoint and return per-skill accuracy.

        Args:
            checkpoint_path: Path to checkpoint
            skills: List of skills to test (default: syllable, binary)

        Returns:
            Dict mapping skill name to accuracy (0-1)
        """
        if skills is None:
            skills = ["syllable", "binary"]

        results = {}

        for skill in skills:
            try:
                # Run quick validation via SSH
                cmd = f"""ssh {self.ssh_host} "cd ~/TRAINING && python3 tools/analysis/run_baseline_test.py \
                    --tag temp_impact_{checkpoint_path.name} \
                    --model-path {checkpoint_path} \
                    --local \
                    --skill {skill} \
                    --max-per-difficulty 10 \
                    --base-dir ~/TRAINING \
                    --quiet 2>/dev/null" """

                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                # Parse accuracy from output
                # Expected format includes accuracy percentages
                if result.returncode == 0:
                    # Try to extract accuracy from output
                    output = result.stdout
                    # Look for patterns like "accuracy: 0.85" or "85%"
                    import re
                    match = re.search(r'accuracy[:\s]+([0-9.]+)', output, re.IGNORECASE)
                    if match:
                        acc = float(match.group(1))
                        if acc > 1:  # Percentage
                            acc = acc / 100
                        results[skill] = acc
                    else:
                        # Try loading result from status file
                        baseline_file = self.status_dir / f"baselines/baseline_temp_impact_{checkpoint_path.name}.json"
                        if baseline_file.exists():
                            with open(baseline_file) as f:
                                data = json.load(f)
                                skill_data = data.get("skills", {}).get(skill, {})
                                results[skill] = skill_data.get("overall", {}).get("accuracy", 0.0)

            except subprocess.TimeoutExpired:
                logger.warning(f"Validation timeout for {skill} on {checkpoint_path.name}")
            except Exception as e:
                logger.warning(f"Validation failed for {skill}: {e}")

        return results

    def _detect_file_type(self, file_path: Path) -> str:
        """
        Detect the type/domain of a training file.

        Args:
            file_path: Path to training file

        Returns:
            Type string: "syllable", "binary", "mixed", etc.
        """
        try:
            with open(file_path) as f:
                # Read first few lines
                lines = [f.readline() for _ in range(5)]

            content = " ".join(lines).lower()

            if "syllable" in content or "count" in content:
                return "syllable"
            elif "binary" in content or "decimal" in content:
                return "binary"
            elif "math" in content or "calculate" in content:
                return "math"
            elif "logic" in content or "syllogism" in content:
                return "logic"
            else:
                return "mixed"

        except Exception:
            return "unknown"

    def _count_examples(self, file_path: Path) -> int:
        """Count examples in a JSONL file."""
        try:
            with open(file_path) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def analyze_file(self, file_path: Path) -> Optional[FileImpact]:
        """
        Analyze the impact of a single training file.

        Args:
            file_path: Path to the completed training file

        Returns:
            FileImpact with metrics, or None if analysis fails
        """
        filename = file_path.name

        if filename in self.analyzed_files:
            logger.debug(f"Already analyzed: {filename}")
            return None

        logger.info(f"Analyzing impact of: {filename}")

        # Find before/after checkpoints
        before_cp, after_cp = self._find_checkpoints_for_file(file_path)

        if not before_cp or not after_cp:
            logger.warning(f"Could not find checkpoint pair for {filename}")
            return None

        step_before = self._extract_step(before_cp)
        step_after = self._extract_step(after_cp)

        # Run validation on both checkpoints
        logger.info(f"Running validation: {before_cp.name} -> {after_cp.name}")
        before_results = self._run_validation(before_cp)
        after_results = self._run_validation(after_cp)

        if not before_results or not after_results:
            logger.warning("Validation returned no results")
            return None

        # Compute deltas
        skill_impacts = {}
        for skill in set(before_results.keys()) | set(after_results.keys()):
            before_acc = before_results.get(skill, 0)
            after_acc = after_results.get(skill, 0)
            skill_impacts[skill] = after_acc - before_acc

        # Find best/worst skills
        if skill_impacts:
            best_skill = max(skill_impacts, key=skill_impacts.get)
            worst_skill = min(skill_impacts, key=skill_impacts.get)
            avg_delta = sum(skill_impacts.values()) / len(skill_impacts)
        else:
            best_skill = worst_skill = None
            avg_delta = 0.0

        # Determine overall impact
        if avg_delta > 0.05:
            overall = "positive"
        elif avg_delta < -0.05:
            overall = "negative"
        elif any(d > 0.1 for d in skill_impacts.values()) and any(d < -0.1 for d in skill_impacts.values()):
            overall = "mixed"
        else:
            overall = "neutral"

        # File metadata
        file_size = file_path.stat().st_size / 1024
        num_examples = self._count_examples(file_path)
        detected_type = self._detect_file_type(file_path)

        impact = FileImpact(
            filename=filename,
            file_path=str(file_path),
            step_before=step_before or 0,
            step_after=step_after or 0,
            timestamp=datetime.now().isoformat(),
            skill_impacts=skill_impacts,
            avg_accuracy_delta=round(avg_delta, 4),
            best_skill=best_skill,
            best_skill_delta=round(skill_impacts.get(best_skill, 0), 4) if best_skill else 0,
            worst_skill=worst_skill,
            worst_skill_delta=round(skill_impacts.get(worst_skill, 0), 4) if worst_skill else 0,
            num_examples=num_examples,
            file_size_kb=round(file_size, 2),
            detected_type=detected_type,
            overall_impact=overall
        )

        # Log to file
        self._log_impact(impact)
        self.analyzed_files.add(filename)

        logger.info(f"Impact: {overall} (avg delta: {avg_delta:+.2%})")

        return impact

    def _log_impact(self, impact: FileImpact):
        """Append impact to log file."""
        with open(self.impact_log_path, 'a') as f:
            f.write(json.dumps(asdict(impact)) + "\n")

    def generate_summary(self) -> ImpactSummary:
        """
        Generate summary statistics from all analyzed files.

        Returns:
            ImpactSummary with aggregated metrics
        """
        impacts = []

        if self.impact_log_path.exists():
            with open(self.impact_log_path) as f:
                for line in f:
                    if line.strip():
                        impacts.append(json.loads(line))

        # Count by impact type
        positive = [i for i in impacts if i["overall_impact"] == "positive"]
        negative = [i for i in impacts if i["overall_impact"] == "negative"]
        neutral = [i for i in impacts if i["overall_impact"] == "neutral"]

        # Top helpful/harmful files
        sorted_by_impact = sorted(impacts, key=lambda x: x["avg_accuracy_delta"], reverse=True)
        top_helpful = sorted_by_impact[:5]
        top_harmful = sorted_by_impact[-5:][::-1]

        # Aggregate by type
        impact_by_type: Dict[str, Dict[str, float]] = {}
        for impact in impacts:
            dtype = impact.get("detected_type", "unknown")
            if dtype not in impact_by_type:
                impact_by_type[dtype] = {"total": 0, "sum_delta": 0}
            impact_by_type[dtype]["total"] += 1
            impact_by_type[dtype]["sum_delta"] += impact.get("avg_accuracy_delta", 0)

        for dtype in impact_by_type:
            total = impact_by_type[dtype]["total"]
            if total > 0:
                impact_by_type[dtype]["avg_delta"] = round(
                    impact_by_type[dtype]["sum_delta"] / total, 4
                )

        summary = ImpactSummary(
            timestamp=datetime.now().isoformat(),
            total_files_analyzed=len(impacts),
            positive_impact_files=len(positive),
            negative_impact_files=len(negative),
            neutral_files=len(neutral),
            top_helpful_files=top_helpful,
            top_harmful_files=top_harmful,
            impact_by_type=impact_by_type
        )

        # Save summary
        with open(self.summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        return summary

    def run_daemon(self, interval: int = 300):
        """
        Run as daemon, analyzing new files periodically.

        Args:
            interval: Seconds between checks
        """
        logger.info(f"Starting data file impact analyzer (interval={interval}s)")

        while True:
            try:
                # Check for new completed files
                completed = self._get_recently_completed()

                new_files = [f for f in completed if f.name not in self.analyzed_files]

                if new_files:
                    logger.info(f"Found {len(new_files)} new files to analyze")

                    for file_path in new_files:
                        try:
                            self.analyze_file(file_path)
                        except Exception as e:
                            logger.error(f"Failed to analyze {file_path.name}: {e}")

                    # Update summary
                    self.generate_summary()

            except Exception as e:
                logger.error(f"Daemon error: {e}", exc_info=True)

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Data File Impact Analyzer")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--file", type=Path, help="Analyze specific file")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Generate summary")
    parser.add_argument("--ssh-host", default="192.168.x.x")

    args = parser.parse_args()

    analyzer = DataFileImpactAnalyzer(args.base_dir, ssh_host=args.ssh_host)

    if args.summary:
        summary = analyzer.generate_summary()
        print(json.dumps(asdict(summary), indent=2))
    elif args.daemon:
        analyzer.run_daemon(args.interval)
    elif args.file:
        impact = analyzer.analyze_file(args.file)
        if impact:
            print(json.dumps(asdict(impact), indent=2))
    else:
        # Analyze any unanalyzed files
        completed = analyzer._get_recently_completed()
        for f in completed:
            analyzer.analyze_file(f)
        analyzer.generate_summary()


if __name__ == "__main__":
    main()
