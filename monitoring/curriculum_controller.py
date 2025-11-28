#!/usr/bin/env python3
"""
Curriculum Automation Controller

Automatically adjusts training difficulty based on model performance.

Strategy:
- Monitors preview EM (Exact Match) metrics
- Adjusts difficulty distribution based on performance
- Generates new training data with adjusted difficulty
- Provides recommendations for next training batch
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class CurriculumController:
    """
    Adaptive curriculum controller

    Adjusts difficulty based on performance metrics:
    - EM > 80%: Increase hard difficulty (model is learning well)
    - EM 40-80%: Maintain balanced mix (optimal learning zone)
    - EM < 40%: Increase easy difficulty (model struggling)
    """

    def __init__(
        self,
        base_dir: Path = None,
        data_gen_script: str = None,
        output_dir: str = "queue/normal"
    ):
        if base_dir is None:
            from core.paths import require_base_dir
            base_dir = require_base_dir()
        self.base_dir = Path(base_dir)
        if data_gen_script is None:
            from core.paths import get_external_tool_path
            skill_path = get_external_tool_path("singleSKILL")
            data_gen_script = str(skill_path / "skill_syllo_variant" / "scripts" / "export_training_data.py")
        self.data_gen_script = data_gen_script
        self.output_dir = self.base_dir / output_dir

        # Curriculum state file
        self.curriculum_state_file = self.base_dir / "status" / "curriculum_state.json"
        self.curriculum_state_file.parent.mkdir(parents=True, exist_ok=True)

    def get_current_performance(self) -> Optional[Dict[str, float]]:
        """
        Get current model performance from preview history

        Returns dict with EM metrics or None if no data
        """
        preview_history_dir = self.base_dir / "data" / "preview_history"
        if not preview_history_dir.exists():
            return None

        # Get recent previews (last 5)
        preview_files = sorted(preview_history_dir.glob("preview_step_*.json"))[-5:]
        if not preview_files:
            return None

        em_rates = []
        for file in preview_files:
            with open(file, 'r') as f:
                data = json.load(f)
                em_rates.append(data['metrics']['exact_match_rate'])

        return {
            'em_latest': em_rates[-1] if em_rates else 0.0,
            'em_avg_5': sum(em_rates) / len(em_rates) if em_rates else 0.0,
            'em_trend': 'improving' if len(em_rates) >= 2 and em_rates[-1] > em_rates[-2] else
                       'declining' if len(em_rates) >= 2 and em_rates[-1] < em_rates[-2] else
                       'stable',
            'sample_count': len(em_rates)
        }

    def calculate_difficulty_adjustment(
        self,
        performance: Dict[str, float],
        current_distribution: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate recommended difficulty distribution

        Args:
            performance: Current performance metrics
            current_distribution: Current difficulty mix (optional)

        Returns:
            Dict with difficulty distribution (easy, medium, hard)
        """
        em_avg = performance['em_avg_5']

        # Default balanced distribution
        if current_distribution is None:
            current_distribution = {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}

        # Adjustment strategy
        if em_avg >= 0.80:
            # Model doing very well - increase challenge
            return {
                'easy': 0.20,
                'medium': 0.30,
                'hard': 0.50,
                'rationale': f'EM {em_avg:.1%} - increasing hard difficulty'
            }
        elif em_avg >= 0.60:
            # Good performance - moderate challenge
            return {
                'easy': 0.25,
                'medium': 0.35,
                'hard': 0.40,
                'rationale': f'EM {em_avg:.1%} - moderate difficulty increase'
            }
        elif em_avg >= 0.40:
            # Moderate performance - balanced mix
            return {
                'easy': 0.33,
                'medium': 0.34,
                'hard': 0.33,
                'rationale': f'EM {em_avg:.1%} - maintaining balanced mix'
            }
        elif em_avg >= 0.20:
            # Struggling - more easy examples
            return {
                'easy': 0.45,
                'medium': 0.35,
                'hard': 0.20,
                'rationale': f'EM {em_avg:.1%} - increasing easy difficulty'
            }
        else:
            # Very poor performance - focus on fundamentals
            return {
                'easy': 0.60,
                'medium': 0.30,
                'hard': 0.10,
                'rationale': f'EM {em_avg:.1%} - focusing on easy examples'
            }

    def generate_curriculum_batch(
        self,
        difficulty_distribution: Dict[str, float],
        count: int = 100000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate training batch with specified difficulty distribution

        Args:
            difficulty_distribution: Dict with easy/medium/hard ratios
            count: Number of examples to generate
            seed: Random seed for reproducibility

        Returns:
            Dict with generation results
        """
        if seed is None:
            seed = int(datetime.now().timestamp()) % 100000

        # Format difficulty string
        diff_str = f"easy:{difficulty_distribution['easy']:.2f},"
        diff_str += f"medium:{difficulty_distribution['medium']:.2f},"
        diff_str += f"hard:{difficulty_distribution['hard']:.2f}"

        # Generate filename with timestamp and difficulty info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        easy_pct = int(difficulty_distribution['easy'] * 100)
        hard_pct = int(difficulty_distribution['hard'] * 100)
        filename = f"syllo_curriculum_e{easy_pct}h{hard_pct}_{timestamp}_count{count}.jsonl"
        output_path = self.output_dir / filename

        # Build command
        cmd = [
            "python3",
            self.data_gen_script,
            "--count", str(count),
            "--seed", str(seed),
            "--difficulty", diff_str,
            "--output", str(output_path)
        ]

        print(f"\n  🎓 Generating curriculum batch...")
        print(f"     Difficulty: {diff_str}")
        print(f"     Count: {count:,}")
        print(f"     Output: {filename}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print(f"  ✓ Generation complete")
                return {
                    'success': True,
                    'output_file': str(output_path),
                    'filename': filename,
                    'count': count,
                    'difficulty': difficulty_distribution,
                    'seed': seed,
                    'timestamp': timestamp
                }
            else:
                print(f"  ✗ Generation failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'difficulty': difficulty_distribution
                }

        except subprocess.TimeoutExpired:
            print(f"  ✗ Generation timed out")
            return {
                'success': False,
                'error': 'Timeout after 300s',
                'difficulty': difficulty_distribution
            }
        except Exception as e:
            print(f"  ✗ Generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'difficulty': difficulty_distribution
            }

    def update_curriculum_state(
        self,
        performance: Dict[str, float],
        difficulty_distribution: Dict[str, float],
        generation_result: Optional[Dict[str, Any]] = None
    ):
        """Save curriculum state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'performance': performance,
            'difficulty_distribution': difficulty_distribution,
            'generation_result': generation_result
        }

        with open(self.curriculum_state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_curriculum_state(self) -> Optional[Dict[str, Any]]:
        """Load curriculum state from file"""
        if not self.curriculum_state_file.exists():
            return None

        with open(self.curriculum_state_file, 'r') as f:
            return json.load(f)

    def run_curriculum_update(
        self,
        count: int = 100000,
        auto_generate: bool = False
    ) -> Dict[str, Any]:
        """
        Run full curriculum update cycle

        Args:
            count: Number of examples to generate
            auto_generate: If True, automatically generate new batch

        Returns:
            Dict with recommendations and results
        """
        print("\n" + "="*60)
        print("Curriculum Automation - Update Cycle")
        print("="*60)

        # Step 1: Get current performance
        performance = self.get_current_performance()
        if performance is None:
            print("\n  ⚠ No performance data available")
            print("    Run preview inference first to gather metrics")
            return {
                'status': 'no_data',
                'recommendation': 'Run preview inference to gather performance metrics'
            }

        print(f"\n  📊 Current Performance:")
        print(f"     Latest EM: {performance['em_latest']:.1%}")
        print(f"     Avg EM (last 5): {performance['em_avg_5']:.1%}")
        print(f"     Trend: {performance['em_trend']}")

        # Step 2: Calculate difficulty adjustment
        current_state = self.get_curriculum_state()
        current_dist = current_state['difficulty_distribution'] if current_state else None

        new_difficulty = self.calculate_difficulty_adjustment(performance, current_dist)

        print(f"\n  🎯 Recommended Difficulty Distribution:")
        print(f"     Easy:   {new_difficulty['easy']:.0%}")
        print(f"     Medium: {new_difficulty['medium']:.0%}")
        print(f"     Hard:   {new_difficulty['hard']:.0%}")
        print(f"     Rationale: {new_difficulty['rationale']}")

        # Step 3: Generate new batch (if requested)
        generation_result = None
        if auto_generate:
            generation_result = self.generate_curriculum_batch(
                new_difficulty,
                count=count
            )

        # Step 4: Update state
        self.update_curriculum_state(performance, new_difficulty, generation_result)

        print("\n" + "="*60)

        return {
            'status': 'success',
            'performance': performance,
            'difficulty_distribution': new_difficulty,
            'generation_result': generation_result,
            'auto_generated': auto_generate
        }


def main():
    """CLI for curriculum controller"""
    import argparse

    parser = argparse.ArgumentParser(description='Curriculum automation controller')
    parser.add_argument('--base-dir', default=None, help='Base directory (defaults to auto-detected)')
    parser.add_argument('--count', type=int, default=100000, help='Number of examples to generate')
    parser.add_argument('--auto-generate', action='store_true', help='Automatically generate new batch')
    parser.add_argument('--show-state', action='store_true', help='Show current curriculum state')

    args = parser.parse_args()

    controller = CurriculumController(base_dir=Path(args.base_dir) if args.base_dir else None)

    if args.show_state:
        state = controller.get_curriculum_state()
        if state:
            print("\nCurrent Curriculum State:")
            print(json.dumps(state, indent=2))
        else:
            print("\nNo curriculum state found")
        return

    # Run curriculum update
    result = controller.run_curriculum_update(
        count=args.count,
        auto_generate=args.auto_generate
    )

    if result['status'] == 'success':
        print(f"\n✓ Curriculum update complete")
        if result['auto_generated'] and result['generation_result']:
            if result['generation_result']['success']:
                print(f"  Generated: {result['generation_result']['filename']}")
            else:
                print(f"  Generation failed: {result['generation_result']['error']}")


if __name__ == '__main__':
    main()
