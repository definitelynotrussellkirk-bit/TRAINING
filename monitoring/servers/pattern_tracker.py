#!/usr/bin/env python3
"""
Pattern Tracker - Coverage & Accuracy by Pattern×Length

Tracks which types of examples are being learned and how well.
Helps identify:
- Gaps in training data (empty cells)
- Patterns that need more examples (low EM)
- Dataset imbalances (one pattern dominates)
"""

from typing import Dict, Callable, Tuple, List
import json


class PatternTracker:
    """Track coverage and exact-match accuracy per pattern×length bucket."""

    def __init__(self, pattern_config: Dict[str, Callable], length_bins: List[int] = None):
        """
        Initialize pattern tracker.

        Args:
            pattern_config: Dict mapping pattern_name -> detector_function
                           detector_function takes (prompt: str) -> bool
            length_bins: List of length boundaries (in tokens)
                        Default: [0, 100, 300, 500, 1000, 2000]

        Example:
            pattern_config = {
                'factual': lambda msg: any(word in msg.lower()
                          for word in ['what is', 'who is', 'when', 'where']),
                'reasoning': lambda msg: any(word in msg.lower()
                            for word in ['explain', 'why', 'how does']),
                'creative': lambda msg: any(word in msg.lower()
                           for word in ['write', 'create', 'compose'])
            }
        """
        self.patterns = pattern_config
        self.length_bins = length_bins or [0, 100, 300, 500, 1000, 2000]

        # Matrix: pattern_id × length_bin → {seen, correct}
        self.matrix = {}
        for pattern in list(self.patterns.keys()) + ['other']:
            self.matrix[pattern] = {}
            for i in range(len(self.length_bins) - 1):
                bin_name = f"{self.length_bins[i]}-{self.length_bins[i+1]}"
                self.matrix[pattern][bin_name] = {'seen': 0, 'correct': 0}

    def classify(self, user_prompt: str, response_tokens: int) -> Tuple[str, str]:
        """
        Classify example into pattern and length bucket.

        Args:
            user_prompt: The user's prompt/question
            response_tokens: Length of the response in tokens

        Returns:
            (pattern_id, bin_name)
        """
        # Determine pattern
        pattern_id = 'other'
        for name, detector in self.patterns.items():
            try:
                if detector(user_prompt):
                    pattern_id = name
                    break
            except Exception as e:
                print(f"Warning: Pattern detector '{name}' failed: {e}")

        # Determine length bucket
        bin_name = None
        for i in range(len(self.length_bins) - 1):
            if self.length_bins[i] <= response_tokens < self.length_bins[i + 1]:
                bin_name = f"{self.length_bins[i]}-{self.length_bins[i+1]}"
                break

        # If longer than last bin
        if bin_name is None:
            bin_name = f"{self.length_bins[-1]}+"
            # Ensure this bin exists
            if bin_name not in self.matrix[pattern_id]:
                self.matrix[pattern_id][bin_name] = {'seen': 0, 'correct': 0}

        return pattern_id, bin_name

    def record(self, pattern_id: str, bin_name: str, correct: bool):
        """
        Record an observation.

        Args:
            pattern_id: Pattern type (e.g., 'factual', 'reasoning')
            bin_name: Length bucket (e.g., '100-300')
            correct: Whether the model got it exactly right
        """
        # Ensure pattern exists
        if pattern_id not in self.matrix:
            self.matrix[pattern_id] = {}

        # Ensure bin exists
        if bin_name not in self.matrix[pattern_id]:
            self.matrix[pattern_id][bin_name] = {'seen': 0, 'correct': 0}

        # Update counts
        self.matrix[pattern_id][bin_name]['seen'] += 1
        if correct:
            self.matrix[pattern_id][bin_name]['correct'] += 1

    def get_matrix(self) -> Dict:
        """
        Export matrix for UI rendering.

        Returns:
            Dict with:
                rows: List of pattern names
                cols: List of length bin names
                data: 2D array of {seen, em}
        """
        # Get all patterns (in order)
        rows = list(self.patterns.keys()) + ['other']

        # Get all length bins (in order)
        cols = []
        for i in range(len(self.length_bins) - 1):
            cols.append(f"{self.length_bins[i]}-{self.length_bins[i+1]}")
        # Add overflow bin if it has data
        overflow_bin = f"{self.length_bins[-1]}+"
        has_overflow = any(overflow_bin in self.matrix.get(p, {}) for p in rows)
        if has_overflow:
            cols.append(overflow_bin)

        # Build data matrix
        data = []
        for pattern in rows:
            row = []
            for bin_name in cols:
                cell = self.matrix.get(pattern, {}).get(bin_name, {'seen': 0, 'correct': 0})
                em = cell['correct'] / cell['seen'] if cell['seen'] > 0 else 0.0
                row.append({
                    'seen': cell['seen'],
                    'em': em
                })
            data.append(row)

        return {
            'rows': rows,
            'cols': cols,
            'data': data
        }

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total_seen = 0
        total_correct = 0
        pattern_stats = {}

        for pattern, bins in self.matrix.items():
            pattern_seen = sum(cell['seen'] for cell in bins.values())
            pattern_correct = sum(cell['correct'] for cell in bins.values())

            total_seen += pattern_seen
            total_correct += pattern_correct

            if pattern_seen > 0:
                pattern_stats[pattern] = {
                    'seen': pattern_seen,
                    'em': pattern_correct / pattern_seen
                }

        return {
            'total_seen': total_seen,
            'total_correct': total_correct,
            'overall_em': total_correct / total_seen if total_seen > 0 else 0.0,
            'by_pattern': pattern_stats
        }

    def save_state(self, filepath: str):
        """Save tracker state to file."""
        state = {
            'patterns': {k: v.__name__ if hasattr(v, '__name__') else str(v)
                        for k, v in self.patterns.items()},
            'length_bins': self.length_bins,
            'matrix': self.matrix
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load tracker state from file (matrix only, not pattern functions)."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.matrix = state['matrix']
        print(f"✅ Loaded pattern tracker state from {filepath}")


# ========== EXAMPLE PATTERN CONFIGURATIONS ==========

def get_default_patterns() -> Dict[str, Callable]:
    """
    Default pattern configuration for general instruction following.

    Returns:
        Dict of pattern_name -> detector_function
    """
    return {
        'factual': lambda msg: any(word in msg.lower() for word in [
            'what is', 'who is', 'when ', 'where ', 'define', 'list'
        ]),
        'reasoning': lambda msg: any(word in msg.lower() for word in [
            'explain', 'why', 'how does', 'analyze', 'compare', 'reason'
        ]),
        'creative': lambda msg: any(word in msg.lower() for word in [
            'write', 'create', 'compose', 'generate', 'imagine', 'design'
        ]),
        'coding': lambda msg: any(word in msg.lower() for word in [
            'code', 'function', 'implement', 'debug', 'script', 'program'
        ]),
        'math': lambda msg: any(word in msg.lower() for word in [
            'calculate', 'solve', 'compute', 'equation', 'proof'
        ])
    }


def get_conversational_patterns() -> Dict[str, Callable]:
    """Pattern config for conversational AI."""
    return {
        'question': lambda msg: msg.strip().endswith('?'),
        'instruction': lambda msg: any(word in msg.lower() for word in [
            'please', 'can you', 'could you', 'would you', 'help me'
        ]),
        'short_query': lambda msg: len(msg.split()) < 10,
        'long_request': lambda msg: len(msg.split()) > 30
    }


if __name__ == '__main__':
    # Example usage
    print("Pattern Tracker - Example Usage")
    print("="*80)

    # Create tracker
    tracker = PatternTracker(get_default_patterns())

    # Simulate some observations
    examples = [
        ("What is machine learning?", 150, True),
        ("Explain how neural networks work", 800, False),
        ("Write a poem about AI", 200, True),
        ("Calculate the derivative of x^2", 100, True),
        ("What is Python?", 120, True),
    ]

    print("\nRecording observations:")
    for prompt, resp_len, correct in examples:
        pattern, bin_name = tracker.classify(prompt, resp_len)
        tracker.record(pattern, bin_name, correct)
        print(f"  {pattern:12s} | {bin_name:10s} | {'✓' if correct else '✗'} | {prompt[:50]}")

    # Get summary
    print("\nSummary:")
    summary = tracker.get_summary()
    print(f"  Total: {summary['total_correct']}/{summary['total_seen']} " +
          f"(EM={summary['overall_em']*100:.1f}%)")
    for pattern, stats in summary['by_pattern'].items():
        print(f"  {pattern:12s}: {stats['em']*100:.1f}% ({stats['seen']} examples)")

    # Get matrix for UI
    print("\nMatrix for UI:")
    matrix = tracker.get_matrix()
    print(f"  Rows: {matrix['rows']}")
    print(f"  Cols: {matrix['cols']}")
    print(f"  Data shape: {len(matrix['data'])}×{len(matrix['data'][0])}")

    print("\n" + "="*80)
