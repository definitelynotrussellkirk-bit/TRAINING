#!/usr/bin/env python3
"""
PRE-TRAINING DATA QUALITY ANALYZER
Assess training data quality WITHOUT training a model.

Calculates multiple quality metrics to predict training effectiveness.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import re


class DataQualityAnalyzer:
    """Analyzes training data quality pre-training."""

    def __init__(self, jsonl_path: str, sample_size: int = 10000):
        self.path = Path(jsonl_path)
        self.sample_size = sample_size
        self.examples = []
        self.metrics = {}

    def load_samples(self):
        """Load examples from JSONL file."""
        print(f"📂 Loading samples from {self.path.name}...")

        with open(self.path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.sample_size:
                    break
                try:
                    self.examples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON at line {i+1}")

        print(f"✓ Loaded {len(self.examples)} examples\n")

    def calculate_diversity_metrics(self):
        """Calculate vocabulary and structural diversity."""
        print("📊 Calculating diversity metrics...")

        all_tokens = []
        all_prompts = []
        all_answers = []

        for ex in self.examples:
            if 'messages' not in ex:
                continue

            for msg in ex['messages']:
                content = msg.get('content', '')
                tokens = content.lower().split()
                all_tokens.extend(tokens)

                if msg['role'] == 'user':
                    all_prompts.append(content)
                elif msg['role'] == 'assistant':
                    all_answers.append(content)

        # Calculate metrics
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        vocab_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0

        # Unique prompts/answers (detect duplicates)
        unique_prompts = len(set(all_prompts))
        unique_answers = len(set(all_answers))

        self.metrics['diversity'] = {
            'unique_vocabulary': unique_tokens,
            'total_tokens': total_tokens,
            'vocabulary_diversity': round(vocab_diversity, 4),
            'unique_prompts': unique_prompts,
            'total_prompts': len(all_prompts),
            'prompt_uniqueness': round(unique_prompts / len(all_prompts), 4) if all_prompts else 0,
            'unique_answers': unique_answers,
            'total_answers': len(all_answers),
            'answer_uniqueness': round(unique_answers / len(all_answers), 4) if all_answers else 0,
        }

        print(f"  ✓ Vocabulary: {unique_tokens:,} unique tokens")
        print(f"  ✓ Vocab diversity: {vocab_diversity:.1%}")
        print(f"  ✓ Prompt uniqueness: {self.metrics['diversity']['prompt_uniqueness']:.1%}\n")

    def calculate_complexity_metrics(self):
        """Calculate compositional complexity."""
        print("🧩 Calculating complexity metrics...")

        prompt_lengths = []
        answer_lengths = []
        compositional_indicators = []

        # Keywords indicating multi-step composition
        composition_keywords = [
            'then', 'after', 'next', 'and then', 'followed by',
            'compare', 'aggregate', 'compute', 'statistics'
        ]

        for ex in self.examples:
            if 'messages' not in ex:
                continue

            for msg in ex['messages']:
                content = msg.get('content', '')
                token_count = len(content.split())

                if msg['role'] == 'user':
                    prompt_lengths.append(token_count)

                    # Count compositional indicators
                    comp_count = sum(1 for kw in composition_keywords if kw in content.lower())
                    compositional_indicators.append(comp_count)

                elif msg['role'] == 'assistant':
                    answer_lengths.append(token_count)

        avg_prompt = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        avg_answer = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        avg_composition = sum(compositional_indicators) / len(compositional_indicators) if compositional_indicators else 0

        self.metrics['complexity'] = {
            'avg_prompt_length': round(avg_prompt, 1),
            'avg_answer_length': round(avg_answer, 1),
            'max_prompt_length': max(prompt_lengths) if prompt_lengths else 0,
            'max_answer_length': max(answer_lengths) if answer_lengths else 0,
            'avg_compositional_depth': round(avg_composition, 2),
            'multi_step_percentage': round(sum(1 for x in compositional_indicators if x > 0) / len(compositional_indicators) * 100, 1) if compositional_indicators else 0,
        }

        print(f"  ✓ Avg prompt length: {avg_prompt:.0f} tokens")
        print(f"  ✓ Avg answer length: {avg_answer:.0f} tokens")
        print(f"  ✓ Multi-step: {self.metrics['complexity']['multi_step_percentage']:.1f}%\n")

    def calculate_quality_metrics(self):
        """Calculate format quality and validity."""
        print("✅ Calculating quality metrics...")

        valid_json_count = 0
        has_messages_count = 0
        has_user_assistant = 0
        json_in_answer = 0

        for ex in self.examples:
            # Check if valid structure
            if 'messages' in ex:
                has_messages_count += 1

                user_found = False
                assistant_found = False

                for msg in ex['messages']:
                    if msg.get('role') == 'user':
                        user_found = True
                    elif msg.get('role') == 'assistant':
                        assistant_found = True

                        # Check if answer contains JSON
                        content = msg.get('content', '')
                        if '[' in content or '{' in content:
                            json_in_answer += 1

                if user_found and assistant_found:
                    has_user_assistant += 1

        total = len(self.examples)

        self.metrics['quality'] = {
            'valid_structure': has_messages_count,
            'valid_structure_pct': round(has_messages_count / total * 100, 1) if total > 0 else 0,
            'complete_conversations': has_user_assistant,
            'complete_conversations_pct': round(has_user_assistant / total * 100, 1) if total > 0 else 0,
            'structured_outputs': json_in_answer,
            'structured_outputs_pct': round(json_in_answer / total * 100, 1) if total > 0 else 0,
        }

        print(f"  ✓ Valid structure: {self.metrics['quality']['valid_structure_pct']:.1f}%")
        print(f"  ✓ Complete conversations: {self.metrics['quality']['complete_conversations_pct']:.1f}%")
        print(f"  ✓ Structured outputs: {self.metrics['quality']['structured_outputs_pct']:.1f}%\n")

    def detect_skills(self):
        """Detect which skills are represented in data."""
        print("🎯 Detecting skill coverage...")

        # Skill indicators from LEO system
        skill_patterns = {
            'filter': r'\b(filter|find|select|where)\b',
            'sort': r'\b(sort|order|arrange)\b',
            'reverse': r'\b(reverse|backward)\b',
            'transform': r'\b(transform|convert|change|capitalize|uppercase|lowercase)\b',
            'aggregate': r'\b(aggregate|count|sum|average|mean|max|min|statistics)\b',
            'compare': r'\b(compare|difference|similar|contrast)\b',
            'group': r'\b(group|categorize|classify)\b',
            'search': r'\b(search|find|locate|lookup)\b',
            'unique': r'\b(unique|distinct|duplicate)\b',
            'conditional': r'\b(if|when|condition|case)\b',
        }

        skill_counts = Counter()

        for ex in self.examples:
            if 'messages' not in ex:
                continue

            for msg in ex['messages']:
                if msg.get('role') != 'user':
                    continue

                content = msg.get('content', '').lower()

                for skill, pattern in skill_patterns.items():
                    if re.search(pattern, content):
                        skill_counts[skill] += 1

        total_examples = len(self.examples)

        self.metrics['skills'] = {
            'detected_skills': len(skill_counts),
            'skill_coverage': {
                skill: {
                    'count': count,
                    'percentage': round(count / total_examples * 100, 1)
                }
                for skill, count in skill_counts.most_common()
            }
        }

        print(f"  ✓ Detected {len(skill_counts)} skill types")
        for skill, data in list(self.metrics['skills']['skill_coverage'].items())[:5]:
            print(f"    • {skill}: {data['percentage']}%")
        print()

    def calculate_overall_score(self):
        """Calculate overall data quality score (0-100)."""
        print("🎯 Calculating overall quality score...")

        scores = {}

        # Diversity score (0-25 points)
        vocab_div = self.metrics['diversity']['vocabulary_diversity']
        prompt_uniq = self.metrics['diversity']['prompt_uniqueness']
        diversity_score = (vocab_div * 12.5) + (prompt_uniq * 12.5)
        scores['diversity'] = round(diversity_score, 1)

        # Complexity score (0-25 points)
        multi_step_pct = self.metrics['complexity']['multi_step_percentage'] / 100
        avg_comp = min(self.metrics['complexity']['avg_compositional_depth'] / 3, 1)  # Cap at 3
        complexity_score = (multi_step_pct * 15) + (avg_comp * 10)
        scores['complexity'] = round(complexity_score, 1)

        # Quality score (0-25 points)
        valid_pct = self.metrics['quality']['valid_structure_pct'] / 100
        complete_pct = self.metrics['quality']['complete_conversations_pct'] / 100
        structured_pct = self.metrics['quality']['structured_outputs_pct'] / 100
        quality_score = (valid_pct * 10) + (complete_pct * 10) + (structured_pct * 5)
        scores['quality'] = round(quality_score, 1)

        # Skill coverage score (0-25 points)
        skill_count = self.metrics['skills']['detected_skills']
        skill_balance = len([s for s in self.metrics['skills']['skill_coverage'].values() if s['percentage'] > 5])
        coverage_score = min(skill_count / 10 * 15, 15) + min(skill_balance / 8 * 10, 10)
        scores['skill_coverage'] = round(coverage_score, 1)

        # Total score
        total_score = sum(scores.values())

        self.metrics['scores'] = {
            'component_scores': scores,
            'total_score': round(total_score, 1),
            'grade': self._get_grade(total_score)
        }

        print(f"\n  📊 COMPONENT SCORES:")
        for component, score in scores.items():
            print(f"    • {component.replace('_', ' ').title()}: {score}/25")
        print(f"\n  🏆 TOTAL SCORE: {total_score:.1f}/100 ({self.metrics['scores']['grade']})\n")

    def _get_grade(self, score):
        """Convert score to letter grade."""
        if score >= 90: return 'A+ (Excellent)'
        elif score >= 85: return 'A (Very Good)'
        elif score >= 80: return 'A- (Good)'
        elif score >= 75: return 'B+ (Above Average)'
        elif score >= 70: return 'B (Average)'
        elif score >= 65: return 'B- (Below Average)'
        elif score >= 60: return 'C (Fair)'
        else: return 'D (Poor)'

    def estimate_value(self):
        """Estimate dataset value based on quality score."""
        print("💰 Estimating dataset value...")

        score = self.metrics['scores']['total_score']
        total_examples = self.metrics['quality']['complete_conversations']

        # Base value per 1k examples based on quality
        if score >= 85:
            value_per_k = 30  # Premium quality
        elif score >= 75:
            value_per_k = 20  # Good quality
        elif score >= 65:
            value_per_k = 10  # Average quality
        else:
            value_per_k = 5   # Below average

        # Calculate for different scales
        estimated_values = {
            '50k_examples': (value_per_k * 50, value_per_k * 50 * 2),
            '100k_examples': (value_per_k * 100, value_per_k * 100 * 2),
            '500k_examples': (value_per_k * 500, value_per_k * 500 * 2),
            '1M_examples': (value_per_k * 1000, value_per_k * 1000 * 2),
        }

        self.metrics['estimated_value'] = {
            'quality_tier': 'Premium' if score >= 85 else 'Good' if score >= 75 else 'Average' if score >= 65 else 'Basic',
            'value_per_1k': value_per_k,
            'pricing': estimated_values
        }

        print(f"\n  Quality Tier: {self.metrics['estimated_value']['quality_tier']}")
        print(f"  Value per 1,000 examples: ${value_per_k}")
        print(f"\n  📈 ESTIMATED PRICING:")
        for scale, (min_val, max_val) in estimated_values.items():
            scale_display = scale.replace('_', ' ').title()
            print(f"    • {scale_display}: ${min_val:,} - ${max_val:,}")
        print()

    def generate_report(self):
        """Generate comprehensive report."""
        print("\n" + "="*60)
        print("📋 DATA QUALITY REPORT")
        print("="*60)
        print()

        # Run all analyses
        self.load_samples()
        self.calculate_diversity_metrics()
        self.calculate_complexity_metrics()
        self.calculate_quality_metrics()
        self.detect_skills()
        self.calculate_overall_score()
        self.estimate_value()

        # Save detailed report
        report_path = self.path.parent / f"{self.path.stem}_quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"💾 Detailed report saved to: {report_path}")
        print()

        return self.metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_training_data.py <path_to_jsonl> [sample_size]")
        print("\nExample:")
        print("  python3 analyze_training_data.py /path/to/training_samples.jsonl")
        print("  python3 analyze_training_data.py /path/to/training_samples.jsonl 5000")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    if not Path(jsonl_path).exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)

    analyzer = DataQualityAnalyzer(jsonl_path, sample_size)
    analyzer.generate_report()

    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
