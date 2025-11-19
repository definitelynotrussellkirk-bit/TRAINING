#!/usr/bin/env python3
"""
Web UI for Auto-Ingest Training System
Monitors and controls the continuous training daemon.
"""

import gradio as gr
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import shutil

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = BASE_DIR / "config.json"
INBOX_DIR = BASE_DIR / "inbox"
MODEL_DIR = BASE_DIR / "model"

# Daemon control (simple file-based signaling)
DAEMON_CONTROL = BASE_DIR / ".daemon_control"
DAEMON_STATUS = BASE_DIR / ".daemon_status"


def load_config():
    """Load current config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save config."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_daemon_status():
    """Check if daemon is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "training_daemon.py"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except:
        return False


def get_training_status():
    """Get current training status."""
    # Check if training is active by looking for python train.py process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py.*--dataset"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return "🟢 Training Active"

        # Check if daemon is running
        if get_daemon_status():
            return "🟡 Daemon Running (Idle)"

        return "🔴 Daemon Stopped"
    except:
        return "❓ Unknown"


def get_inbox_files():
    """Get list of files in inbox."""
    if not INBOX_DIR.exists():
        return []

    files = []
    for f in INBOX_DIR.glob("*.jsonl"):
        size_mb = f.stat().st_size / (1024 * 1024)
        files.append({
            "name": f.name,
            "size": f"{size_mb:.1f} MB",
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        })
    return sorted(files, key=lambda x: x["modified"], reverse=True)


def get_daemon_log():
    """Get recent daemon log entries."""
    logs_dir = BASE_DIR / "logs"
    if not logs_dir.exists():
        return "No logs directory found"

    # Find most recent daemon log
    log_files = sorted(logs_dir.glob("daemon_*.log"))
    if not log_files:
        return "No daemon log files found"

    latest_log = log_files[-1]
    try:
        # Read last 100 lines
        with open(latest_log) as f:
            lines = f.readlines()
            return "".join(lines[-100:])
    except Exception as e:
        return f"Error reading log: {e}"


def get_training_details():
    """Get detailed training status."""
    # Check if training process exists
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train.py.*--dataset"],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            pid = result.stdout.strip().split()[0]

            # Get GPU info
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )

            gpu_info = "GPU info unavailable"
            if gpu_result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = gpu_result.stdout.strip().split(',')
                gpu_info = f"GPU: {gpu_util.strip()}% | VRAM: {mem_used.strip()}/{mem_total.strip()} MB | Temp: {temp.strip()}°C"

            return f"""**Training Active**
PID: {pid}
{gpu_info}

Check daemon log below for details.
"""
        else:
            return "**No active training**\n\nDaemon may be idle or waiting for data."
    except Exception as e:
        return f"Error checking training: {e}"


def get_gpu_status():
    """Get GPU utilization."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(',')
            return f"GPU: {gpu_util.strip()}% | VRAM: {mem_used.strip()}/{mem_total.strip()} MB | Temp: {temp.strip()}°C"
        return "GPU info unavailable"
    except:
        return "nvidia-smi not available"


def switch_profile(profile_name):
    """Switch to different training profile."""
    profiles = {
        "balanced": "config.json",
        "aggressive": "config_aggressive.json",
        "conservative": "config_conservative.json"
    }

    source_file = BASE_DIR / profiles.get(profile_name, "config.json")

    if profile_name == "balanced":
        # Already the default
        config = load_config()
        return f"✓ Using balanced profile\nCurrent settings: LR={config.get('learning_rate', 'N/A')}, LoRA r={config.get('lora_r', 'N/A')}"

    if not source_file.exists():
        return f"❌ Profile file not found: {source_file}"

    try:
        # Backup current config
        backup = BASE_DIR / "config_backup.json"
        shutil.copy(CONFIG_FILE, backup)

        # Copy profile to active config
        shutil.copy(source_file, CONFIG_FILE)

        config = load_config()
        return f"✓ Switched to {profile_name} profile\nLR={config.get('learning_rate', 'N/A')}, LoRA r={config.get('lora_r', 'N/A')}, Grad Accum={config.get('gradient_accumulation', 'N/A')}"
    except Exception as e:
        return f"❌ Failed to switch profile: {e}"


def toggle_daemon(action):
    """Start or stop daemon."""
    if action == "start":
        if get_daemon_status():
            return "⚠️ Daemon already running"

        try:
            # Start daemon in background
            subprocess.Popen(
                ["python3", str(BASE_DIR / "training_daemon.py"), "--base-dir", str(BASE_DIR)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            time.sleep(2)
            if get_daemon_status():
                return "✓ Daemon started successfully"
            return "❌ Failed to start daemon"
        except Exception as e:
            return f"❌ Error starting daemon: {e}"

    elif action == "stop":
        if not get_daemon_status():
            return "⚠️ Daemon not running"

        try:
            subprocess.run(["pkill", "-f", "training_daemon.py"])
            time.sleep(1)
            if not get_daemon_status():
                return "✓ Daemon stopped"
            return "⚠️ Daemon may still be running"
        except Exception as e:
            return f"❌ Error stopping daemon: {e}"

    return "❌ Invalid action"


def get_config_summary():
    """Get summary of current config."""
    config = load_config()

    summary = f"""
**Current Configuration**

**Training:**
- Learning Rate: {config.get('learning_rate', 'N/A')}
- Warmup Steps: {config.get('warmup_steps', 'N/A')}
- Batch Size: {config.get('batch_size', 'N/A')}
- Gradient Accumulation: {config.get('gradient_accumulation', 'N/A')}
- Effective Batch: {config.get('batch_size', 1) * config.get('gradient_accumulation', 1)}

**LoRA:**
- Rank (r): {config.get('lora_r', 'N/A')}
- Alpha: {config.get('lora_alpha', 'N/A')}

**Monitoring:**
- Eval Steps: {config.get('eval_steps', 'N/A')}
- Save Steps: {config.get('save_steps', 'N/A')}
- Max Length: {config.get('max_length', 'N/A')}
"""
    return summary


def create_ui():
    """Create Gradio UI."""

    def update_status():
        return f"""
### System Status
**Daemon:** {get_training_status()}
**GPU:** {get_gpu_status()}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with gr.Blocks(title="Training Control Center", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 Auto-Ingest Training Control Center")
        gr.Markdown("Monitor and control your continuous training system")

        with gr.Row():
            with gr.Column(scale=2):
                status_display = gr.Markdown(value=update_status())

            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")

        # Timer for auto-refresh (every 5 seconds)
        timer = gr.Timer(value=5)

        gr.Markdown("---")

        with gr.Tabs():
            # Tab 1: Daemon Control
            with gr.Tab("⚙️ Daemon Control"):
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Daemon", variant="primary", scale=1)
                    stop_btn = gr.Button("⏹️ Stop Daemon", variant="stop", scale=1)

                daemon_output = gr.Textbox(label="Output", lines=2)

                start_btn.click(
                    fn=lambda: toggle_daemon("start"),
                    outputs=daemon_output
                )
                stop_btn.click(
                    fn=lambda: toggle_daemon("stop"),
                    outputs=daemon_output
                )

            # Tab 2: Training Profile
            with gr.Tab("📊 Training Profile"):
                gr.Markdown("Switch between training configurations")

                profile_radio = gr.Radio(
                    choices=["balanced", "aggressive", "conservative"],
                    value="balanced",
                    label="Select Profile",
                    info="Balanced: Standard | Aggressive: Fast | Conservative: Safe"
                )

                profile_btn = gr.Button("Apply Profile", variant="primary")
                profile_output = gr.Textbox(label="Result", lines=3)

                config_display = gr.Markdown(value=get_config_summary)

                profile_btn.click(
                    fn=switch_profile,
                    inputs=profile_radio,
                    outputs=profile_output
                ).then(
                    fn=get_config_summary,
                    outputs=config_display
                )

            # Tab 3: Inbox Monitor
            with gr.Tab("📥 Inbox Monitor"):
                gr.Markdown("Files waiting to be processed")

                inbox_display = gr.Dataframe(
                    headers=["File Name", "Size", "Last Modified"],
                    value=lambda: [[f["name"], f["size"], f["modified"]] for f in get_inbox_files()],
                    label="Inbox Contents"
                )

                refresh_inbox_btn = gr.Button("🔄 Refresh Inbox")
                refresh_inbox_btn.click(
                    fn=lambda: [[f["name"], f["size"], f["modified"]] for f in get_inbox_files()],
                    outputs=inbox_display
                )

            # Tab 4: Training Logs
            with gr.Tab("📊 Training Logs"):
                gr.Markdown("Real-time training monitor")

                training_details = gr.Markdown(value=get_training_details)

                daemon_log_display = gr.Textbox(
                    label="Daemon Log (Last 100 lines)",
                    value=get_daemon_log,
                    lines=30,
                    max_lines=50,
                    show_label=True,
                    interactive=False
                )

                with gr.Row():
                    refresh_log_btn = gr.Button("🔄 Refresh Logs", variant="secondary")

                refresh_log_btn.click(
                    fn=get_daemon_log,
                    outputs=daemon_log_display
                ).then(
                    fn=get_training_details,
                    outputs=training_details
                )

                # Auto-refresh logs every 5 seconds
                log_timer = gr.Timer(value=5)
                log_timer.tick(
                    fn=get_daemon_log,
                    outputs=daemon_log_display
                ).then(
                    fn=get_training_details,
                    outputs=training_details
                )

            # Tab 5: Quick Actions
            with gr.Tab("⚡ Quick Actions"):
                gr.Markdown("### Common Tasks")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Convert LEO Data**")
                        input_file = gr.Textbox(
                            label="Input File Path",
                            placeholder="/path/to/input.jsonl"
                        )
                        output_name = gr.Textbox(
                            label="Output Name",
                            placeholder="converted_batch.jsonl"
                        )
                        convert_btn = gr.Button("Convert & Copy to Inbox")
                        convert_output = gr.Textbox(label="Result", lines=3)

                        def convert_and_copy(input_path, output_name):
                            try:
                                if not Path(input_path).exists():
                                    return "❌ Input file not found"

                                output_path = INBOX_DIR / output_name
                                result = subprocess.run(
                                    ["python3", str(BASE_DIR / "convert_leo_data.py"), input_path, str(output_path)],
                                    capture_output=True,
                                    text=True
                                )

                                if result.returncode == 0:
                                    return f"✓ Converted and saved to inbox/{output_name}"
                                return f"❌ Conversion failed:\n{result.stderr}"
                            except Exception as e:
                                return f"❌ Error: {e}"

                        convert_btn.click(
                            fn=convert_and_copy,
                            inputs=[input_file, output_name],
                            outputs=convert_output
                        )

        # Manual refresh button
        refresh_btn.click(
            fn=update_status,
            outputs=status_display
        )

        # Auto-refresh with timer
        timer.tick(
            fn=update_status,
            outputs=status_display
        )

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Training Control Center")
    print("=" * 60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Config File: {CONFIG_FILE}")
    print(f"Inbox: {INBOX_DIR}")
    print("=" * 60)

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
