#!/usr/bin/env python3
"""
Ultimate Trainer - Gradio Web UI

A beautiful web interface for the Ultimate Trainer system.
Launch with: python3 web_ui_gradio.py

Features:
- Visual data validation
- Interactive sample browser
- Training configuration wizard
- Live training dashboard
- Training history
"""

import gradio as gr
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

from validator import DatasetValidator, ValidationIssue
from model_db import ModelDatabase
from time_estimator import TimeEstimator


class UltimateTrainerUI:
    """Web UI for Ultimate Trainer."""

    def __init__(self):
        self.model_db = ModelDatabase()
        self.validator = None
        self.current_dataset = None
        self.validation_results = None

    def validate_dataset(self, dataset_file) -> Tuple[str, str, str]:
        """
        Validate a dataset file.

        Returns: (summary, issues, samples_html)
        """
        if dataset_file is None:
            return "⚠️ No file selected", "", ""

        try:
            # Create validator
            dataset_path = Path(dataset_file.name)
            self.validator = DatasetValidator(dataset_path)

            # Load dataset
            if not self.validator.load_dataset():
                issues_text = "\n".join([f"❌ {i.message}" for i in self.validator.issues])
                return "❌ Failed to load dataset", issues_text, ""

            # Run validation
            format_ok = self.validator.validate_format()
            duplicates = self.validator.check_duplicates()
            stats = self.validator.compute_token_stats()
            leakage_samples = self.validator.check_answer_leakage(num_samples=10)

            # Build summary
            num_examples = len(self.validator.examples)
            leakage_count = sum(1 for s in leakage_samples if s['has_leakage'])

            summary = f"""
## 📊 Validation Summary

- **Total Examples:** {num_examples:,}
- **Format:** {'✅ Valid' if format_ok else '❌ Invalid'}
- **Duplicates:** {duplicates:,} ({duplicates/num_examples*100:.1f}%)
- **Leakage Detected:** {leakage_count}/10 samples

### Token Statistics
- **Input:** {stats.min_input}-{stats.max_input} tokens (avg: {stats.avg_input:.0f})
- **Output:** {stats.min_output}-{stats.max_output} tokens (avg: {stats.avg_output:.0f})
- **Total:** {stats.total_tokens:,} tokens
"""

            # Build issues
            errors = [i for i in self.validator.issues if i.severity == 'error']
            warnings = [i for i in self.validator.issues if i.severity == 'warning']

            issues_text = ""
            if errors:
                issues_text += "## ❌ Errors\n"
                for issue in errors:
                    issues_text += f"- {issue.message}\n"

            if warnings:
                issues_text += "\n## ⚠️ Warnings\n"
                for issue in warnings:
                    issues_text += f"- {issue.message}\n"

            if not errors and not warnings:
                issues_text = "## ✅ No Issues Found!"

            # Build samples HTML
            samples_html = self._build_samples_html(leakage_samples)

            self.validation_results = {
                'passed': len(errors) == 0,
                'samples': leakage_samples,
                'stats': stats
            }

            return summary, issues_text, samples_html

        except Exception as e:
            import traceback
            return f"❌ Error: {e}", traceback.format_exc(), ""

    def _build_samples_html(self, samples: List[Dict]) -> str:
        """Build HTML for sample display."""
        html = "<div style='font-family: monospace;'>"

        for sample in samples:
            status = "❌ LEAKAGE" if sample['has_leakage'] else "✅ OK"
            color = "#fee" if sample['has_leakage'] else "#efe"

            html += f"""
<div style='margin: 20px 0; padding: 15px; background: {color}; border-radius: 8px; border-left: 4px solid {"#c00" if sample['has_leakage'] else "#0c0"}'>
    <h3>{status} Sample {sample['index'] + 1}</h3>

    <div style='background: #f8f8f8; padding: 10px; margin: 10px 0; border-radius: 4px;'>
        <strong>INPUT:</strong><br>
        <pre style='white-space: pre-wrap; margin: 5px 0;'>{sample['user'][:500]}</pre>
    </div>

    <div style='background: #f8f8f8; padding: 10px; margin: 10px 0; border-radius: 4px;'>
        <strong>EXPECTED OUTPUT:</strong><br>
        <pre style='white-space: pre-wrap; margin: 5px 0;'>{sample['assistant'][:500]}</pre>
    </div>
"""

            if sample['has_leakage']:
                html += "<div style='background: #fcc; padding: 10px; margin: 10px 0; border-radius: 4px;'>"
                html += "<strong>⚠️ LEAKAGE DETECTED:</strong><br>"
                for leak in sample['leakage']:
                    html += f"<li>{leak}</li>"
                html += "</div>"

            html += "</div>"

        html += "</div>"
        return html

    def list_models(self) -> pd.DataFrame:
        """Get list of available models as DataFrame."""
        models = self.model_db.list_models()

        if not models:
            # Trigger a scan
            search_paths = [
                Path("/media/user/ST/aiPROJECT/models"),
                Path.home() / ".cache" / "huggingface" / "hub",
            ]
            self.model_db.scan_for_models(search_paths)
            models = self.model_db.list_models()

        if not models:
            return pd.DataFrame(columns=["Name", "Size (GB)", "Layers", "Type"])

        data = []
        for model in models:
            data.append({
                "Name": model.name,
                "Size (GB)": model.size_gb,
                "Layers": model.num_layers,
                "Type": model.model_type,
                "Path": model.path
            })

        return pd.DataFrame(data)

    def estimate_training(
        self,
        model_name: str,
        num_examples: int,
        batch_size: int,
        epochs: int
    ) -> str:
        """Estimate training time."""

        if not model_name or num_examples == 0:
            return "⚠️ Please select a model and specify dataset size"

        # Get model size
        model_info = self.model_db.get_model(model_name)
        if model_info:
            model_size_b = (model_info.hidden_size * model_info.num_layers) / 1000
        else:
            model_size_b = 8.0

        # Estimate
        estimate = TimeEstimator.estimate_training(
            num_examples=num_examples,
            batch_size=batch_size,
            num_epochs=epochs,
            model_size_b=model_size_b
        )

        # Format output
        result = f"""
## ⏱️ Training Time Estimate

### Configuration
- **Model:** {model_name} (~{model_size_b:.1f}B parameters)
- **Dataset:** {num_examples:,} examples
- **Batch Size:** {batch_size}
- **Epochs:** {epochs}

### Estimate
- **Total Steps:** {estimate.total_steps:,}
- **Time per Step:** ~{estimate.seconds_per_step} seconds
- **Total Time:** {TimeEstimator.format_duration(estimate.total_seconds)} ({estimate.total_hours:.1f} hours)
- **Estimated Completion:** {estimate.estimated_completion}

### Resources
- **GPU Memory:** ~{estimate.memory_gb:.1f} GB
- **Disk Space (checkpoints):** ~{estimate.checkpoints_gb:.1f} GB

### Timeline
"""

        # Add milestone timeline
        import datetime
        now = datetime.datetime.now()
        milestones = [
            (0.25, "25% Complete"),
            (0.50, "Halfway"),
            (0.75, "75% Complete"),
            (1.00, "DONE")
        ]

        for pct, label in milestones:
            step = int(estimate.total_steps * pct)
            elapsed = step * estimate.seconds_per_step
            eta = now + datetime.timedelta(seconds=elapsed)
            result += f"- **{label}:** {eta.strftime('%I:%M %p')} (step {step:,})\n"

        result += "\n⚠️ *Note: Estimate may vary ±20% based on actual hardware performance*"

        return result

    def scan_models(self) -> Tuple[pd.DataFrame, str]:
        """Scan for models and return updated list."""
        search_paths = [
            Path("/media/user/ST/aiPROJECT/models"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]

        found = self.model_db.scan_for_models(search_paths)

        message = f"✅ Scan complete! Found {len(found)} models."

        return self.list_models(), message


def create_ui():
    """Create the Gradio interface."""

    trainer_ui = UltimateTrainerUI()

    with gr.Blocks(title="Ultimate Trainer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
# 🚀 Ultimate Trainer
## Catch Training Mistakes BEFORE Wasting 9 Hours!

Welcome to the Ultimate Trainer web interface. This tool helps you:
- ✅ Validate training data for common issues
- ✅ Select and configure models
- ✅ Estimate training time and resources
- ✅ Monitor training in real-time

---
""")

        with gr.Tabs():
            # Tab 1: Data Validation
            with gr.Tab("📋 1. Validate Data"):
                gr.Markdown("""
### Step 1: Validate Your Training Data

Upload your training dataset (JSONL format) to check for:
- Format issues
- Answer leakage (the critical check!)
- Duplicate examples
- Token statistics
""")

                with gr.Row():
                    with gr.Column():
                        dataset_file = gr.File(
                            label="Upload Dataset (JSONL)",
                            file_types=[".jsonl", ".json"]
                        )
                        validate_btn = gr.Button("🔍 Validate Dataset", variant="primary", size="lg")

                    with gr.Column():
                        validation_summary = gr.Markdown("Upload a dataset to get started")

                with gr.Row():
                    validation_issues = gr.Markdown("")

                gr.Markdown("### Sample Review")
                samples_display = gr.HTML("")

                validate_btn.click(
                    fn=trainer_ui.validate_dataset,
                    inputs=[dataset_file],
                    outputs=[validation_summary, validation_issues, samples_display]
                )

            # Tab 2: Model Selection
            with gr.Tab("🤖 2. Select Model"):
                gr.Markdown("""
### Step 2: Choose Your Model

Select a model from the database or scan for new models.
""")

                with gr.Row():
                    scan_btn = gr.Button("🔍 Scan for Models", size="sm")
                    scan_status = gr.Markdown("")

                models_table = gr.Dataframe(
                    value=trainer_ui.list_models(),
                    label="Available Models",
                    interactive=False
                )

                scan_btn.click(
                    fn=trainer_ui.scan_models,
                    inputs=[],
                    outputs=[models_table, scan_status]
                )

            # Tab 3: Configuration & Estimation
            with gr.Tab("⚙️ 3. Configure Training"):
                gr.Markdown("""
### Step 3: Configure Training Parameters

Set your training configuration and get a time estimate.
""")

                with gr.Row():
                    with gr.Column():
                        model_name = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., qwen3_8b",
                            value="qwen3_8b"
                        )

                        num_examples = gr.Number(
                            label="Number of Training Examples",
                            value=50000,
                            precision=0
                        )

                        batch_size = gr.Slider(
                            label="Batch Size (per device)",
                            minimum=1,
                            maximum=32,
                            value=4,
                            step=1,
                            info="Larger = faster but more memory"
                        )

                        epochs = gr.Slider(
                            label="Number of Epochs",
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1
                        )

                        estimate_btn = gr.Button("⏱️ Estimate Training Time", variant="primary")

                    with gr.Column():
                        estimate_output = gr.Markdown("Click 'Estimate Training Time' to see estimate")

                estimate_btn.click(
                    fn=trainer_ui.estimate_training,
                    inputs=[model_name, num_examples, batch_size, epochs],
                    outputs=[estimate_output]
                )

            # Tab 4: Training (Placeholder)
            with gr.Tab("🚀 4. Train"):
                gr.Markdown("""
### Step 4: Start Training

**Coming Soon!**

This tab will allow you to:
- Start training with configured settings
- Monitor live training progress
- See real-time accuracy charts
- View live inference examples
- Pause/stop training

For now, use the CLI:
```bash
python3 train.py \\
    --dataset /path/to/data.jsonl \\
    --model qwen3_8b \\
    --output-dir ~/my_adapter \\
    --epochs 2 \\
    --batch-size 4
```
""")

            # Tab 5: History (Placeholder)
            with gr.Tab("📜 5. History"):
                gr.Markdown("""
### Training History

**Coming Soon!**

This tab will show:
- All previous training runs
- Comparison between runs
- Accuracy curves
- Export reports
""")

        gr.Markdown("""
---
### 💡 Quick Tips

1. **Always validate your data first!** The validator catches issues that would waste hours of training.
2. **Check for answer leakage** - this is the most common mistake (happened to us!)
3. **Start with a small test run** (100 steps) to verify everything works
4. **Monitor GPU memory** - if it's maxed out, reduce batch size

### 📚 Documentation

For more information, see:
- `README.md` - Complete guide
- `ULTIMATE_TRAINER_PLAN.md` - System design
- `GUI_IMPROVEMENTS_PLAN.md` - Future enhancements

### 🐛 Found a Bug?

Report issues at: https://github.com/anthropics/claude-code/issues
        """)

    return app


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ULTIMATE TRAINER - WEB UI")
    print("=" * 80)
    print()
    print("Starting Gradio web interface...")
    print("Once started, open your browser to the URL shown below.")
    print()

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to get public URL
        show_error=True
    )
