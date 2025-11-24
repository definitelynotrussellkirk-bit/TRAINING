#!/usr/bin/env python3
"""
Compare model performance across multiple checkpoints
"""

import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict
import sys

class CheckpointComparator:
    def __init__(self, api_url: str = "http://192.168.x.x:8765"):
        self.api_url = api_url
        
    def test_checkpoint(
        self, 
        model_id: str, 
        examples: List[Dict],
        max_tokens: int = 500
    ) -> Dict:
        """Test a checkpoint on examples"""
        results = []
        
        print(f"\n🧪 Testing {model_id}...")
        
        for i, ex in enumerate(examples, 1):
            messages = ex['messages'][:-1]  # Exclude assistant response
            expected = ex['messages'][-1]['content']
            
            try:
                resp = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json={
                        'model': model_id,
                        'messages': messages,
                        'max_tokens': max_tokens,
                        'temperature': 0.1
                    },
                    timeout=60
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    actual = data['choices'][0]['message']['content']
                    
                    # Simple EM check
                    match = expected.strip() == actual.strip()
                    
                    results.append({
                        'example': i,
                        'match': match,
                        'expected': expected[:200],
                        'actual': actual[:200],
                        'tokens': data['usage']
                    })
                    
                    print(f"   [{i}/{len(examples)}] {'✅' if match else '❌'}")
                else:
                    print(f"   [{i}/{len(examples)}] ❌ Error: {resp.status_code}")
                    
            except Exception as e:
                print(f"   [{i}/{len(examples)}] ❌ Error: {e}")
                
        em_score = sum(r['match'] for r in results) / len(results) if results else 0
        
        return {
            'model_id': model_id,
            'em_score': em_score,
            'total_examples': len(examples),
            'matches': sum(r['match'] for r in results),
            'results': results
        }
    
    def compare(
        self,
        model_ids: List[str],
        dataset_path: Path,
        sample_size: int = 10
    ):
        """Compare multiple checkpoints"""
        
        # Load examples
        examples = []
        with open(dataset_path) as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                examples.append(json.loads(line))
        
        print(f"\n{'='*80}")
        print(f"📊 CHECKPOINT COMPARISON")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_path.name}")
        print(f"Examples: {len(examples)}")
        print(f"Models: {', '.join(model_ids)}")
        
        # Test each checkpoint
        comparison = []
        for model_id in model_ids:
            result = self.test_checkpoint(model_id, examples)
            comparison.append(result)
        
        # Print comparison table
        print(f"\n{'='*80}")
        print(f"📈 RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<30} {'EM Score':<15} {'Matches'}")
        print("-" * 80)
        
        for c in sorted(comparison, key=lambda x: x['em_score'], reverse=True):
            print(f"{c['model_id']:<30} {c['em_score']:.2%}          {c['matches']}/{c['total_examples']}")
        
        # Show best
        best = max(comparison, key=lambda x: x['em_score'])
        print(f"\n🏆 Best: {best['model_id']} ({best['em_score']:.2%})")
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints")
    parser.add_argument('--models', nargs='+', required=True, help="Model IDs to compare")
    parser.add_argument('--dataset', type=Path, required=True, help="Dataset path")
    parser.add_argument('--samples', type=int, default=10, help="Number of samples")
    parser.add_argument('--api-url', default="http://192.168.x.x:8765")
    
    args = parser.parse_args()
    
    comparator = CheckpointComparator(args.api_url)
    comparator.compare(args.models, args.dataset, args.samples)

if __name__ == '__main__':
    main()
