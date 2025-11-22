#!/usr/bin/env python3
"""
Detail Collector - Captures detailed training information for monitoring

This module provides a callback that collects:
- Current loss / eval loss
- Complete prompt context (system + user + assistant)
- Golden assistant response (expected output)
- Model prediction
- Token-by-token comparison data

Writes to status/training_detail.json for real-time dashboard viewing
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import TrainerCallback


class DetailCollector(TrainerCallback):
    """
    Callback to collect detailed training information

    Captures sample predictions during evaluation and writes them
    to a JSON file for the monitoring dashboard
    """

    def __init__(self, output_dir, tokenizer, eval_dataset=None, update_frequency=100):
        """
        Initialize detail collector

        Args:
            output_dir: Directory to write detail JSON file
            tokenizer: Tokenizer for decoding predictions
            eval_dataset: Validation dataset to sample from
            update_frequency: Update detail file every N steps
        """
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.update_frequency = update_frequency
        self.detail_file = self.output_dir / "training_detail.json"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detail file
        self._write_detail({
            'status': 'initialized',
            'message': 'Waiting for training to start...'
        })

    def _write_detail(self, data):
        """Write detailed training data to JSON file"""
        try:
            data['timestamp'] = datetime.now().isoformat()
            with open(self.detail_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to write detail file: {e}")

    def _extract_prompt_from_sample(self, sample):
        """
        Extract messages from a training sample

        Args:
            sample: Training sample dict with 'input_ids' or 'messages'

        Returns:
            dict: Prompt information with messages array
        """
        # Try to get original messages if available
        if 'messages' in sample:
            return {'messages': sample['messages']}

        # Otherwise try to reconstruct from input_ids
        if 'input_ids' in sample:
            try:
                text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
                # Parse as messages format if possible
                # For now just return as single text
                return {'text': text}
            except:
                return {'text': '[Could not decode]'}

        return {'text': '[No prompt data]'}

    def _get_model_prediction(self, model, input_ids, max_length=100):
        """
        Generate prediction from model

        Args:
            model: The model to generate from
            input_ids: Input token IDs
            max_length: Maximum tokens to generate

        Returns:
            str: Generated text
        """
        try:
            model.eval()
            with torch.no_grad():
                # Generate continuation
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=min(max_length, 200),
                    do_sample=False,  # Greedy decoding for consistency
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Decode only the new tokens
                generated_ids = outputs[0][len(input_ids[0]):]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                return generated_text
        except Exception as e:
            return f"[Generation error: {e}]"

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        """Called during evaluation - capture detailed sample"""

        # Only update every N evaluations to avoid overhead
        if state.global_step % self.update_frequency != 0:
            return

        try:
            # Get a sample from eval dataset
            if self.eval_dataset is None or len(self.eval_dataset) == 0:
                return

            # Get first sample (or random sample)
            sample_idx = min(0, len(self.eval_dataset) - 1)
            sample = self.eval_dataset[sample_idx]

            # Extract components
            input_ids = sample.get('input_ids', [])
            labels = sample.get('labels', [])

            # Convert to tensor if needed
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids]).to(model.device)
            else:
                input_ids = input_ids.unsqueeze(0).to(model.device)

            # Get prompt and golden response
            prompt_info = self._extract_prompt_from_sample(sample)

            # Decode golden response (labels)
            if len(labels) > 0:
                # Filter out -100 (ignore index)
                golden_ids = [l for l in labels if l != -100]
                golden_text = self.tokenizer.decode(golden_ids, skip_special_tokens=True)
            else:
                golden_text = "[No labels]"

            # Get model prediction
            predicted_text = self._get_model_prediction(model, input_ids)

            # Get current metrics from state
            latest_metrics = state.log_history[-1] if state.log_history else {}

            # Build detail data
            detail_data = {
                'status': 'training',
                'step': state.global_step,
                'epoch': state.epoch,
                'train_loss': latest_metrics.get('loss', None),
                'eval_loss': latest_metrics.get('eval_loss', None),
                'learning_rate': latest_metrics.get('learning_rate', None),
                'sample_idx': sample_idx,
                'prompt': prompt_info,
                'golden': golden_text,
                'predicted': predicted_text,
                'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
            }

            # Write to file
            self._write_detail(detail_data)

        except Exception as e:
            print(f"Warning: Detail collection failed: {e}")
            import traceback
            traceback.print_exc()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging - update basic metrics"""

        # Only update if we have useful logs
        if not logs or state.global_step % self.update_frequency != 0:
            return

        try:
            # Update with latest metrics only (no sample data)
            detail_data = {
                'status': 'training',
                'step': state.global_step,
                'epoch': state.epoch,
                'train_loss': logs.get('loss', None),
                'eval_loss': logs.get('eval_loss', None),
                'learning_rate': logs.get('learning_rate', None),
                'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
            }

            # Only write if we have actual metrics
            if detail_data['train_loss'] is not None or detail_data['eval_loss'] is not None:
                self._write_detail(detail_data)

        except Exception as e:
            print(f"Warning: Metric logging failed: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        self._write_detail({
            'status': 'completed',
            'message': 'Training completed',
            'final_step': state.global_step,
            'final_epoch': state.epoch
        })


def add_detail_collector_to_trainer(trainer, tokenizer, eval_dataset=None, update_frequency=100):
    """
    Add detail collector callback to a Trainer

    Args:
        trainer: HuggingFace Trainer instance
        tokenizer: Tokenizer for decoding
        eval_dataset: Optional validation dataset
        update_frequency: Update every N steps (default: 100)

    Returns:
        DetailCollector: The collector instance
    """
    # Create status directory if needed
    status_dir = Path(trainer.args.output_dir) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)

    # Create collector
    collector = DetailCollector(
        output_dir=status_dir,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        update_frequency=update_frequency
    )

    # Add to trainer callbacks
    trainer.add_callback(collector)

    print(f"✅ Detail collector enabled (updates every {update_frequency} steps)")
    print(f"   Output: {status_dir / 'training_detail.json'}")
    print(f"   Monitor at: http://localhost:8081")

    return collector
