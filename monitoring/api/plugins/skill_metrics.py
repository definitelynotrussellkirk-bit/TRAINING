#!/usr/bin/env python3
"""
Skill Metrics Plugin
Fetches per-skill baseline results for trained skills, primitives, and benchmarks.
Compares current trained model performance vs base model.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import subprocess
import os
import re

from .base import BasePlugin

# Skill categories for organization
SKILL_CATEGORIES = {
    'trained': ['syllable', 'binary'],
    'primitives': {
        'counting': ['letter_count', 'vowel_count', 'word_count', 'digit_count', 'char_frequency'],
        'conversion': ['decimal_to_hex', 'decimal_to_octal', 'binary_to_decimal', 'roman_numerals'],
        'string': ['reverse_word', 'reverse_sentence', 'first_n_chars', 'palindrome_check'],
        'arithmetic': ['digit_sum', 'even_odd', 'modulo', 'compare_numbers', 'simple_addition'],
        'logic': ['boolean_and', 'boolean_or', 'boolean_xor'],
        'sequence': ['next_in_sequence', 'alphabetical_order', 'position_in_alphabet'],
        'set': ['membership', 'unique_elements'],
    },
    'benchmarks': {
        'babi': [f'babi_qa{i}' for i in range(1, 21)],
        'bigbench': ['bb_arithmetic', 'bb_elementary_math_qa', 'bb_word_sorting',
                     'bb_word_unscrambling', 'bb_logical_deduction', 'bb_tracking_shuffled_objects',
                     'bb_date_understanding', 'bb_navigate', 'bb_object_counting',
                     'bb_operators', 'bb_list_functions'],
    }
}


class SkillMetricsPlugin(BasePlugin):
    """
    Fetches skill metrics from baseline evaluation results.

    Data sources:
    - Local: /path/to/training/status/baselines/*.json
    - Remote: ssh://192.168.x.x/~/TRAINING/status/baselines/*.json

    Skills tracked:
    - TRAINED: syllable, binary (what we train on)
    - PRIMITIVES: 26 transfer learning tests
    - BENCHMARKS: bAbI (20), BIG-Bench (11+)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ssh_host = self.config.get('ssh_host', '192.168.x.x')
        self.local_baselines_dir = Path(self.config.get(
            'local_baselines_dir',
            '/path/to/training/status/baselines'
        ))
        self.remote_baselines_dir = self.config.get(
            'remote_baselines_dir',
            '/path/to/training/status/baselines'
        )

    def get_name(self) -> str:
        return 'skill_metrics'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': 'Per-skill performance metrics (SYLLABLE, BINARY)',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'skills',
            'machine': 'both',
            'location': 'local+remote',
            'fields': [
                'syllable', 'binary', 'base_model', 'trained_model',
                'by_difficulty', 'improvement'
            ]
        }

    def _read_local_file(self, filepath: Path) -> Optional[Dict]:
        """Read a local JSON file"""
        try:
            if filepath.exists():
                with open(filepath) as f:
                    return json.load(f)
        except Exception as e:
            pass
        return None

    def _read_remote_file(self, remote_path: str) -> Optional[Dict]:
        """Read a remote JSON file via SSH"""
        try:
            cmd = f"ssh {self.ssh_host} 'cat {remote_path}' 2>/dev/null"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except Exception as e:
            pass
        return None

    def _list_remote_files(self, pattern: str) -> list:
        """List remote files matching pattern"""
        try:
            cmd = f"ssh {self.ssh_host} 'ls {pattern}' 2>/dev/null"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        except Exception as e:
            pass
        return []

    def _find_latest_baseline(self, tag_prefix: str = None) -> Optional[Dict]:
        """Find the most recent baseline file for a tag prefix"""
        # Try local first
        if self.local_baselines_dir.exists():
            files = list(self.local_baselines_dir.glob('baseline_*.json'))
            if tag_prefix:
                files = [f for f in files if tag_prefix in f.name]
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                return self._read_local_file(latest)

        # Try remote
        remote_pattern = f"{self.remote_baselines_dir}/baseline_*.json"
        remote_files = self._list_remote_files(remote_pattern)
        if tag_prefix:
            remote_files = [f for f in remote_files if tag_prefix in f]
        if remote_files:
            # Read the last one (most recent by name)
            return self._read_remote_file(sorted(remote_files)[-1])

        return None

    def _list_all_baselines(self) -> list:
        """List all available baseline tags (local + remote)"""
        tags = set()

        # Local baselines
        if self.local_baselines_dir.exists():
            for f in self.local_baselines_dir.glob('baseline_*.json'):
                # Extract tag from filename: baseline_<tag>.json
                tag = f.stem.replace('baseline_', '')
                if tag:
                    tags.add(tag)

        # Remote baselines
        remote_pattern = f"{self.remote_baselines_dir}/baseline_*.json"
        remote_files = self._list_remote_files(remote_pattern)
        for f in remote_files:
            # Extract tag from path
            filename = Path(f).stem
            tag = filename.replace('baseline_', '')
            if tag:
                tags.add(tag)

        return sorted(tags)

    def _get_baseline_by_tag(self, tag: str) -> Optional[Dict]:
        """Load a specific baseline by its tag"""
        # Try local first
        local_path = self.local_baselines_dir / f"baseline_{tag}.json"
        if local_path.exists():
            return self._read_local_file(local_path)

        # Try remote
        remote_path = f"{self.remote_baselines_dir}/baseline_{tag}.json"
        return self._read_remote_file(remote_path)

    def _extract_skill_data(self, baseline_data: Dict, skill: str) -> Dict:
        """Extract data for a specific skill from baseline results"""
        if not baseline_data or 'skills' not in baseline_data:
            return {
                'available': False,
                'overall_accuracy': 0.0,
                'by_difficulty': {},
                'timestamp': None
            }

        skill_data = baseline_data.get('skills', {}).get(skill, {})
        if not skill_data:
            return {
                'available': False,
                'overall_accuracy': 0.0,
                'by_difficulty': {},
                'timestamp': None
            }

        # Extract difficulty breakdown
        by_difficulty = {}
        for diff_name, diff_data in skill_data.get('by_difficulty', {}).items():
            by_difficulty[diff_name] = {
                'accuracy': diff_data.get('accuracy', 0.0) * 100,  # Convert to percent
                'correct': diff_data.get('correct', 0),
                'total': diff_data.get('total', 0)
            }

        # Handle both formats:
        # - Trained skills (syllable, binary): have 'overall.accuracy' (0-1 scale)
        # - Primitives: have direct 'accuracy' field (0-1 scale)
        if 'overall' in skill_data and 'accuracy' in skill_data.get('overall', {}):
            overall_acc = skill_data['overall']['accuracy'] * 100
        elif 'accuracy' in skill_data:
            overall_acc = skill_data['accuracy'] * 100
        else:
            overall_acc = 0.0

        return {
            'available': True,
            'overall_accuracy': overall_acc,
            'by_difficulty': by_difficulty,
            'timestamp': baseline_data.get('timestamp')
        }

    def _get_all_skills_from_baseline(self, baseline_data: Dict) -> List[str]:
        """Extract all skill names from a baseline result"""
        if not baseline_data or 'skills' not in baseline_data:
            return []
        return list(baseline_data.get('skills', {}).keys())

    def fetch(self) -> Dict[str, Any]:
        """Fetch skill metrics from baseline results"""
        from datetime import datetime

        # List all available baseline versions
        available_versions = self._list_all_baselines()

        result = {
            'timestamp': datetime.now().isoformat(),
            'available_versions': available_versions,
            'skill_categories': SKILL_CATEGORIES,
            'base_model': {
                'tag': 'base_qwen3_0.6b',
                'available': False
            },
            'trained_model': {
                'tag': None,
                'available': False
            },
            'versions': {},  # All versions data
            'skills': {},  # Will be populated with all skills found
            'summary': {
                'trained': {'base_avg': 0, 'trained_avg': 0, 'delta': 0},
                'primitives': {'base_avg': 0, 'trained_avg': 0, 'delta': 0},
                'benchmarks': {'base_avg': 0, 'trained_avg': 0, 'delta': 0},
            }
        }

        # Load ALL available baselines and collect all skills
        all_skills = set(['syllable', 'binary'])  # Start with trained skills
        for tag in available_versions:
            baseline_data = self._get_baseline_by_tag(tag)
            if baseline_data:
                skills_in_baseline = self._get_all_skills_from_baseline(baseline_data)
                all_skills.update(skills_in_baseline)

                version_info = {
                    'tag': tag,
                    'timestamp': baseline_data.get('timestamp'),
                    'checkpoint_step': baseline_data.get('checkpoint_step'),
                    'checkpoint_name': baseline_data.get('checkpoint_name'),
                    'model_path': baseline_data.get('model_path'),
                    'skills': {}
                }
                for skill in skills_in_baseline:
                    version_info['skills'][skill] = self._extract_skill_data(baseline_data, skill)
                result['versions'][tag] = version_info

        # Initialize all skills
        for skill in all_skills:
            result['skills'][skill] = {
                'base': {'available': False, 'overall_accuracy': 0},
                'trained': {'available': False, 'overall_accuracy': 0},
                'improvement': 0.0,
                'category': self._categorize_skill(skill)
            }

        # Load base model baseline
        base_data = self._find_latest_baseline('base_')
        if base_data:
            result['base_model']['available'] = True
            result['base_model']['timestamp'] = base_data.get('timestamp')
            result['base_model']['skills_tested'] = self._get_all_skills_from_baseline(base_data)

            # Extract skill data for base model
            for skill in self._get_all_skills_from_baseline(base_data):
                if skill in result['skills']:
                    result['skills'][skill]['base'] = self._extract_skill_data(base_data, skill)

        # Load trained model baseline
        trained_data = self._find_latest_baseline('trained_')
        if not trained_data:
            trained_data = self._find_latest_baseline('checkpoint')

        if trained_data:
            result['trained_model']['available'] = True
            result['trained_model']['tag'] = trained_data.get('tag', 'trained_latest')
            result['trained_model']['timestamp'] = trained_data.get('timestamp')
            result['trained_model']['skills_tested'] = self._get_all_skills_from_baseline(trained_data)

            for skill in self._get_all_skills_from_baseline(trained_data):
                if skill in result['skills']:
                    result['skills'][skill]['trained'] = self._extract_skill_data(trained_data, skill)

        # Calculate improvements and summaries
        trained_deltas, prim_deltas, bench_deltas = [], [], []
        for skill, data in result['skills'].items():
            base_acc = data.get('base', {}).get('overall_accuracy', 0)
            trained_acc = data.get('trained', {}).get('overall_accuracy', 0)
            data['improvement'] = trained_acc - base_acc

            cat = data.get('category', 'unknown')
            if cat == 'trained':
                if base_acc > 0 or trained_acc > 0:
                    trained_deltas.append((base_acc, trained_acc))
            elif cat == 'primitive':
                if base_acc > 0 or trained_acc > 0:
                    prim_deltas.append((base_acc, trained_acc))
            elif cat == 'benchmark':
                if base_acc > 0 or trained_acc > 0:
                    bench_deltas.append((base_acc, trained_acc))

        # Calculate averages
        for name, deltas in [('trained', trained_deltas), ('primitives', prim_deltas), ('benchmarks', bench_deltas)]:
            if deltas:
                base_avg = sum(d[0] for d in deltas) / len(deltas)
                trained_avg = sum(d[1] for d in deltas) / len(deltas)
                result['summary'][name] = {
                    'base_avg': round(base_avg, 1),
                    'trained_avg': round(trained_avg, 1),
                    'delta': round(trained_avg - base_avg, 1),
                    'count': len(deltas)
                }

        return result

    def _categorize_skill(self, skill: str) -> str:
        """Determine category of a skill"""
        if skill in SKILL_CATEGORIES['trained']:
            return 'trained'
        for cat, skills in SKILL_CATEGORIES['primitives'].items():
            if skill in skills:
                return 'primitive'
        for cat, skills in SKILL_CATEGORIES['benchmarks'].items():
            if skill in skills:
                return 'benchmark'
        # Check by prefix
        if skill.startswith('babi_') or skill.startswith('bb_') or skill.startswith('bbh_'):
            return 'benchmark'
        return 'unknown'
