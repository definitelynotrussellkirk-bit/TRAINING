#!/usr/bin/env python3
"""
Curriculum Optimization Plugin
Phase 2, Task 2.2: Curriculum strategy results from 3090 machine
Phase 3: Added local curriculum state for data flow visibility
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from .base import RemoteFilePlugin


class CurriculumPlugin(RemoteFilePlugin):
    """
    Fetches curriculum optimization results from the 3090 intelligence machine
    AND local curriculum state for data flow status.

    Data sources:
    - Remote: ssh://192.168.x.x/~/TRAINING/status/curriculum_optimization.json
    - Local: data_manager/curriculum_state.json

    Refresh: Every 5 minutes
    Critical: No
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ssh_host = (config or {}).get('ssh_host', '192.168.x.x')
        remote_path = (config or {}).get(
            'remote_path',
            '/path/to/training/status/curriculum_optimization.json'
        )

        # Local curriculum state path
        self.local_state_path = Path(
            (config or {}).get('base_dir', '/path/to/training')
        ) / 'data_manager' / 'curriculum_state.json'

        # Cache for 5 minutes (data updates every 5 min)
        config = config or {}
        config['cache_duration'] = config.get('cache_duration', 300)

        super().__init__(ssh_host, remote_path, config)

    def get_name(self) -> str:
        return 'curriculum_optimization'

    def get_metadata(self) -> Dict[str, Any]:
        return {
            'description': '3090 Intelligence Machine - Curriculum strategy optimization + data flow',
            'refresh_interval': 300,
            'critical': False,
            'data_type': 'intelligence',
            'machine': '3090',
            'location': 'hybrid',
            'fields': [
                'evaluations', 'difficulties.easy.accuracy',
                'difficulties.medium.accuracy', 'difficulties.hard.accuracy',
                'data_flow.current_skill', 'data_flow.current_level',
                'data_flow.recent_accuracy', 'data_flow.eval_count'
            ]
        }

    def _load_local_curriculum_state(self) -> Dict[str, Any]:
        """Load local curriculum state file for data flow info"""
        try:
            if self.local_state_path.exists():
                with open(self.local_state_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            return {'error': str(e)}
        return {}

    def _extract_data_flow(self, local_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data flow summary from local curriculum state"""
        data_flow = {
            'available': False,
            'skills': {}
        }

        skills = local_state.get('skills', {})
        if not skills:
            return data_flow

        data_flow['available'] = True

        for skill_name, skill_data in skills.items():
            current_level = skill_data.get('current_level', 1)
            history = skill_data.get('accuracy_history', [])

            # Get recent evaluations (last 5)
            recent = history[-5:] if history else []
            recent_accuracies = [h.get('accuracy', 0.0) for h in recent]
            avg_recent = sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0.0

            # Calculate progress toward level advancement (need 80% avg over 3 evals)
            last_3 = recent_accuracies[-3:] if len(recent_accuracies) >= 3 else recent_accuracies
            avg_last_3 = sum(last_3) / len(last_3) if last_3 else 0.0
            progress_pct = min(100, (avg_last_3 / 80.0) * 100) if avg_last_3 > 0 else 0.0

            # Check if model is progressing
            trend = 'stable'
            if len(recent_accuracies) >= 3:
                early = sum(recent_accuracies[:2]) / 2 if len(recent_accuracies) >= 2 else recent_accuracies[0]
                late = sum(recent_accuracies[-2:]) / 2 if len(recent_accuracies) >= 2 else recent_accuracies[-1]
                if late > early + 5:
                    trend = 'improving'
                elif late < early - 5:
                    trend = 'declining'

            data_flow['skills'][skill_name] = {
                'current_level': current_level,
                'max_level': 10 if skill_name == 'syllo' else 7,
                'eval_count': len(history),
                'recent_accuracy': round(avg_recent * 100, 1),
                'progress_to_next': round(progress_pct, 1),
                'trend': trend,
                'last_eval_step': history[-1].get('step') if history else None,
                'last_eval_time': history[-1].get('timestamp') if history else None
            }

        return data_flow

    def fetch(self) -> Dict[str, Any]:
        """Fetch and extract key curriculum metrics + data flow"""
        # Try to fetch remote data (may fail if 3090 not available)
        try:
            data = super().fetch()
        except Exception:
            data = {'evaluations': [], 'error': 'Remote fetch failed'}

        # Extract latest evaluation summary from remote data
        if 'evaluations' in data and len(data['evaluations']) > 0:
            latest = data['evaluations'][-1]

            summary = {
                'step': latest.get('step'),
                'checkpoint': latest.get('checkpoint'),
                'timestamp': latest.get('timestamp'),
                'accuracies': {}
            }

            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in latest.get('difficulties', {}):
                    diff_data = latest['difficulties'][difficulty]
                    summary['accuracies'][difficulty] = diff_data.get('accuracy', 0.0)

            data['latest_summary'] = summary

        # Load local curriculum state for data flow
        local_state = self._load_local_curriculum_state()
        data['data_flow'] = self._extract_data_flow(local_state)

        return data
