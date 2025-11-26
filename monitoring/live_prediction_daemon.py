#!/usr/bin/env python3
"""
Live Prediction Daemon - Automatic prediction generation every 5 minutes

Continuously monitors training and generates fresh predictions using
the 3090 inference server.

Features:
- Runs every 5 minutes (configurable)
- Loads validation prompts
- Calls 3090 API /generate endpoint
- Polls for job completion
- Saves to status/latest_predictions.json
- Contract-compliant output format
"""

import json
import time
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib import request, error as urllib_error
from datetime import datetime


class LivePredictionDaemon:
    """
    Daemon that automatically generates predictions every N minutes
    """

    def __init__(
        self,
        base_dir: Path,
        remote_api_url: str = "http://192.168.x.x:8765",
        validation_file: str = "data/validation/syllo_validation_20.jsonl",
        interval: int = 300,  # 5 minutes
        count: int = 5,
        max_tokens: int = 2048,
        temperature: float = 0.1
    ):
        self.base_dir = Path(base_dir)
        self.remote_api_url = remote_api_url
        self.validation_file = self.base_dir / validation_file
        self.interval = interval
        self.count = count
        self.max_tokens = max_tokens
        self.temperature = temperature

        # API authentication and model
        self.api_key = os.environ.get("INFERENCE_API_KEY", "admin123")
        self.model_name = os.environ.get("INFERENCE_MODEL", "checkpoint-175000")

        # Storage
        self.status_dir = self.base_dir / "status"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.status_dir / "latest_predictions.json"

        # Setup logging
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "live_prediction_daemon.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load validation data
        self.validation_data: List[Dict] = []
        self._load_validation_data()

    def _load_validation_data(self):
        """Load validation dataset"""
        if not self.validation_file.exists():
            self.logger.warning(f"Validation file not found: {self.validation_file}")
            return

        with open(self.validation_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.validation_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        self.logger.info(f"Loaded {len(self.validation_data)} validation samples")

    def _get_current_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get current checkpoint info from training status

        Returns:
            dict with path, step, training_step, last_updated, model_name, status
        """
        status_file = self.status_dir / "training_status.json"

        checkpoint_info = {
            "path": str(self.base_dir / "current_model"),
            "step": None,
            "training_step": None,
            "last_updated": datetime.now().isoformat(),
            "model_name": "Qwen3-0.6B",
            "status": "unknown"
        }

        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    current_step = status.get("current_step")
                    checkpoint_info["step"] = current_step
                    checkpoint_info["training_step"] = current_step
                    checkpoint_info["status"] = "active" if status.get("status") == "training" else "idle"
            except Exception as e:
                self.logger.warning(f"Failed to read training status: {e}")

        return checkpoint_info

    def _extract_difficulty(self, metadata: Dict) -> str:
        """Extract difficulty from metadata"""
        return metadata.get("difficulty", "unknown")

    def _call_3090_api(self, prompt: str) -> Optional[Dict]:
        """
        Call 3090 API /v1/chat/completions endpoint (OpenAI-compatible)

        Args:
            prompt: User prompt to generate from

        Returns:
            Result dict with generated_text and inference_time_ms, or None
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.api_key
            }

            req = request.Request(
                f"{self.remote_api_url}/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            start_time = time.time()
            with request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            inference_time = (time.time() - start_time) * 1000

            # Extract text from OpenAI format
            generated_text = result['choices'][0]['message']['content']

            return {
                'generated_text': generated_text,
                'inference_time_ms': inference_time,
                'error': None
            }

        except Exception as e:
            self.logger.error(f"Failed to call API: {e}")
            return {
                'generated_text': '',
                'inference_time_ms': 0.0,
                'error': str(e)
            }

    def _has_thinking_tags(self, text: str) -> bool:
        """Check if output contains thinking tags"""
        import re
        return bool(re.search(r'<think', text, re.IGNORECASE))

    def _check_semantic_match(self, generated: str, expected: str) -> bool:
        """
        Check if generated output has valid JSON structure
        """
        try:
            import re
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if not json_match:
                return False

            generated_json = json.loads(json_match.group(0))

            # Check for required keys
            if 'solutions' not in generated_json:
                return False

            return True

        except Exception:
            return False

    def _calculate_exact_match(self, generated: str, expected: str) -> bool:
        """
        Check if answers are exactly correct
        """
        try:
            import re
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if not json_match:
                return False

            generated_json = json.loads(json_match.group(0))
            expected_json = json.loads(expected)

            gen_solutions = generated_json.get('solutions', [])
            exp_solutions = expected_json.get('solutions', [])

            if len(gen_solutions) != len(exp_solutions):
                return False

            # Check each answer
            for gen_sol, exp_sol in zip(gen_solutions, exp_solutions):
                if gen_sol.get('answer') != exp_sol.get('answer'):
                    return False

            return True

        except Exception:
            return False

    def generate_predictions_once(self) -> Dict[str, Any]:
        """
        Generate predictions once (single cycle)

        Returns:
            Predictions dict matching contract spec
        """
        if not self.validation_data:
            self.logger.error("No validation data loaded")
            return {
                "error": "No validation data",
                "timestamp": datetime.now().isoformat()
            }

        # Get checkpoint info
        checkpoint_info = self._get_current_checkpoint_info()
        self.logger.info(f"Generating predictions for checkpoint step {checkpoint_info['step']}")

        # Sample validation examples
        import random
        samples = random.sample(
            self.validation_data,
            min(self.count, len(self.validation_data))
        )

        predictions = []
        stats = {
            "total": 0,
            "exact_match": 0,
            "semantic_match": 0,
            "has_thinking": 0
        }

        for i, sample in enumerate(samples, 1):
            messages = sample['messages']
            user_content = messages[0]['content']
            expected_output = messages[1]['content']
            metadata = sample.get('metadata', {})

            puzzle_id = metadata.get('puzzle_id', f'puzzle_{i}')
            difficulty = self._extract_difficulty(metadata)

            self.logger.info(f"[{i}/{len(samples)}] Generating {puzzle_id} ({difficulty})")

            # Call API and get result
            result = self._call_3090_api(user_content)
            if not result or result.get('error'):
                self.logger.warning(f"Failed to generate {puzzle_id}: {result.get('error') if result else 'No response'}")
                continue

            generated = result['generated_text']
            inference_time = result['inference_time_ms']

            # Analyze prediction
            exact_match = self._calculate_exact_match(generated, expected_output)
            semantic_match = self._check_semantic_match(generated, expected_output)
            has_thinking = self._has_thinking_tags(generated)

            # Update stats
            stats["total"] += 1
            if exact_match:
                stats["exact_match"] += 1
            if semantic_match:
                stats["semantic_match"] += 1
            if has_thinking:
                stats["has_thinking"] += 1

            # Create prediction entry matching contract spec
            pred_timestamp = datetime.now()
            prediction = {
                "id": f"pred_{pred_timestamp.strftime('%Y%m%d_%H%M%S')}_{i:03d}",
                "difficulty": difficulty,
                "prompt": user_content,
                "expected_answer": expected_output,
                "model_output": generated,
                "extracted_answer": None,  # Could extract JSON structure here
                "metrics": {
                    "exact_match": exact_match,
                    "semantic_match": semantic_match,
                    "has_thinking_tags": has_thinking,
                    "format_valid": semantic_match,  # If semantic match works, format is valid
                    "completion_time_ms": int(inference_time)
                },
                "manual_grade": None,
                "timestamp": pred_timestamp.isoformat()
            }
            predictions.append(prediction)

            # Log status
            em = '✓' if exact_match else '✗'
            sm = '✓' if semantic_match else '✗'
            think = '💭' if has_thinking else '  '
            self.logger.info(f"  {em} EM:{exact_match} SM:{semantic_match} {think} ({inference_time:.0f}ms)")

        # Calculate rates
        accuracy_auto = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0.0
        semantic_rate = stats["semantic_match"] / stats["total"] if stats["total"] > 0 else 0.0
        thinking_rate = stats["has_thinking"] / stats["total"] if stats["total"] > 0 else 0.0

        # Build output matching contract
        output = {
            "checkpoint": checkpoint_info,
            "predictions": predictions,
            "stats": {
                "total": stats["total"],
                "accuracy_auto": accuracy_auto,
                "semantic_match_rate": semantic_rate,
                "thinking_tag_rate": thinking_rate
            },
            "generated_at": datetime.now().isoformat()
        }

        # Save to status file
        with open(self.predictions_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"✓ Generated {stats['total']} predictions")
        self.logger.info(f"  Accuracy: {accuracy_auto:.1%}")
        self.logger.info(f"  Semantic: {semantic_rate:.1%}")
        self.logger.info(f"  Thinking: {thinking_rate:.1%}")

        return output

    def run_daemon(self):
        """
        Run daemon continuously
        """
        self.logger.info("="*60)
        self.logger.info("Live Prediction Daemon Started")
        self.logger.info("="*60)
        self.logger.info(f"Base dir: {self.base_dir}")
        self.logger.info(f"API URL: {self.remote_api_url}")
        self.logger.info(f"Interval: {self.interval}s ({self.interval/60:.1f}min)")
        self.logger.info(f"Predictions per cycle: {self.count}")
        self.logger.info("="*60)

        cycle_num = 0

        while True:
            cycle_num += 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Cycle {cycle_num} starting...")
            self.logger.info(f"{'='*60}")

            try:
                self.generate_predictions_once()
            except Exception as e:
                self.logger.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)

            self.logger.info(f"Cycle {cycle_num} complete. Sleeping {self.interval}s...")
            time.sleep(self.interval)


def main():
    """CLI for live prediction daemon"""
    parser = argparse.ArgumentParser(description='Live prediction generation daemon')
    parser.add_argument('--base-dir', default='/path/to/training', help='Base directory')
    parser.add_argument('--interval', type=int, default=300, help='Generation interval in seconds (default: 300s = 5min)')
    parser.add_argument('--count', type=int, default=5, help='Number of predictions per cycle')
    parser.add_argument('--api-url', default='http://192.168.x.x:8765', help='3090 API URL')
    parser.add_argument('--once', action='store_true', help='Run once and exit (for testing)')

    args = parser.parse_args()

    daemon = LivePredictionDaemon(
        base_dir=args.base_dir,
        remote_api_url=args.api_url,
        interval=args.interval,
        count=args.count
    )

    if args.once:
        # Test mode - run once and exit
        daemon.logger.info("Running once (test mode)")
        result = daemon.generate_predictions_once()
        if 'error' in result:
            daemon.logger.error(f"Error: {result['error']}")
            sys.exit(1)
        daemon.logger.info(f"✓ Test complete. Saved to {daemon.predictions_file}")
        sys.exit(0)
    else:
        # Daemon mode - run continuously
        daemon.run_daemon()


if __name__ == '__main__':
    main()
