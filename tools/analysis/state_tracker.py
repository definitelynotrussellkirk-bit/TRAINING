#!/usr/bin/env python3
"""
System State Tracker

Generates machine-readable system state for future Claude instances.
Run this to create/update .system_state.json with current system status.

Usage:
    python3 state_tracker.py              # Generate state file
    python3 state_tracker.py --check      # Check state and report
    python3 state_tracker.py --warnings   # Show warnings only
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import hashlib

class StateTracker:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / ".system_state.json"
        
    def get_directory_size(self, path):
        """Get size of directory in GB"""
        if not path.exists():
            return 0.0
        
        try:
            result = subprocess.run(
                ['du', '-sb', str(path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                bytes_size = int(result.stdout.split()[0])
                return round(bytes_size / (1024**3), 2)  # Convert to GB
        except Exception:
            pass
        return 0.0
    
    def check_current_model(self):
        """Check current model status"""
        model_dir = self.base_dir / "current_model"
        
        if not model_dir.exists():
            return {
                "exists": False,
                "path": "current_model/",
                "size_gb": 0,
                "training_steps": 0,
                "last_training": None,
                "warning": "No current model - fresh start"
            }
        
        # Get size
        size_gb = self.get_directory_size(model_dir)
        
        # Check for adapter files
        has_adapter = (model_dir / "adapter_model.safetensors").exists()
        
        # Get training status
        status_file = self.base_dir / "status" / "training_status.json"
        training_steps = 0
        last_training = None
        
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)
                    training_steps = status.get('current_step', 0)
                    # Get last modified time of status file
                    last_training = datetime.fromtimestamp(
                        status_file.stat().st_mtime
                    ).isoformat()
            except Exception:
                pass
        
        return {
            "exists": True,
            "path": "current_model/",
            "size_gb": size_gb,
            "has_adapter": has_adapter,
            "training_steps": training_steps,
            "last_training": last_training,
            "warning": None if has_adapter else "No adapter found - base model only"
        }
    
    def list_versions(self):
        """List all model versions"""
        versions_dir = self.base_dir / "models" / "versions"
        
        if not versions_dir.exists():
            return []
        
        versions = []
        for version_dir in sorted(versions_dir.iterdir()):
            if version_dir.is_dir() and version_dir.name.startswith('v'):
                # Try to read metadata
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            versions.append({
                                "id": version_dir.name.split('_')[0],  # v001, v002, etc
                                "date": metadata.get('timestamp', 'unknown')[:10],
                                "description": metadata.get('description', 'No description'),
                                "training_steps": metadata.get('training_steps', 0),
                                "size_gb": self.get_directory_size(version_dir)
                            })
                    except Exception:
                        pass
                else:
                    # No metadata, just basic info
                    versions.append({
                        "id": version_dir.name.split('_')[0],
                        "date": datetime.fromtimestamp(
                            version_dir.stat().st_mtime
                        ).strftime('%Y-%m-%d'),
                        "description": "Unknown (no metadata)",
                        "training_steps": 0,
                        "size_gb": self.get_directory_size(version_dir)
                    })
        
        return versions
    
    def check_config(self):
        """Check config.json status"""
        config_file = self.base_dir / "config.json"
        
        if not config_file.exists():
            return {
                "exists": False,
                "locked_params": [],
                "warning": "Config file missing!"
            }
        
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            # Critical parameters that should be locked
            critical_params = ['max_length', 'base_model', 'model_name']
            
            return {
                "exists": True,
                "model_name": config.get('model_name', 'unknown'),
                "base_model": config.get('base_model', 'unknown'),
                "max_length": config.get('max_length', 0),
                "locked_params": critical_params,
                "warning": None
            }
        except Exception as e:
            return {
                "exists": True,
                "locked_params": [],
                "warning": f"Error reading config: {e}"
            }
    
    def check_training_status(self):
        """Check current training status"""
        status_file = self.base_dir / "status" / "training_status.json"
        
        if not status_file.exists():
            return {
                "status": "idle",
                "current_step": 0,
                "total_evals": 0,
                "last_update": None
            }
        
        try:
            with open(status_file) as f:
                status = json.load(f)
            
            return {
                "status": status.get('status', 'unknown'),
                "current_step": status.get('current_step', 0),
                "total_evals": status.get('total_evals', 0),
                "last_update": status.get('timestamp', None)
            }
        except Exception:
            return {
                "status": "error",
                "current_step": 0,
                "total_evals": 0,
                "last_update": None
            }
    
    def check_daemon_running(self):
        """Check if training daemon is running"""
        try:
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Look for training_daemon.py process
            for line in result.stdout.split('\n'):
                if 'training_daemon.py' in line and 'grep' not in line:
                    return True
            
            return False
        except Exception:
            return False
    
    def check_disk_space(self):
        """Check available disk space"""
        try:
            result = subprocess.run(
                ['df', '-BG', str(self.base_dir)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse df output
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                fields = lines[1].split()
                if len(fields) >= 4:
                    available_gb = int(fields[3].rstrip('G'))
                    used_percent = int(fields[4].rstrip('%'))
                    
                    warning = None
                    if available_gb < 10:
                        warning = f"LOW DISK SPACE: Only {available_gb}GB free!"
                    elif available_gb < 50:
                        warning = f"Disk space getting low: {available_gb}GB free"
                    
                    return {
                        "available_gb": available_gb,
                        "used_percent": used_percent,
                        "warning": warning
                    }
        except Exception:
            pass
        
        return {
            "available_gb": -1,
            "used_percent": -1,
            "warning": "Could not check disk space"
        }
    
    def generate_warnings(self, state):
        """Generate list of warnings from state"""
        warnings = []
        
        # Check current model
        if state['current_model'].get('warning'):
            warnings.append(('model', state['current_model']['warning']))
        
        # Check config
        if state['config'].get('warning'):
            warnings.append(('config', state['config']['warning']))
        
        # Check if daemon should be running but isn't
        if state['current_model']['exists'] and state['current_model']['training_steps'] > 0:
            if not state['daemon_running']:
                warnings.append(('daemon', 'Training daemon not running (model exists with training progress)'))
        
        # Check disk space
        if state['disk_space'].get('warning'):
            warnings.append(('disk', state['disk_space']['warning']))
        
        # Check if current_model exists but no versions
        if state['current_model']['exists'] and len(state['versions']) == 0:
            warnings.append(('backup', 'Current model exists but no versions saved - create backup!'))
        
        return warnings
    
    def generate_state(self):
        """Generate complete system state"""
        state = {
            "last_updated": datetime.now().isoformat(),
            "current_model": self.check_current_model(),
            "versions": self.list_versions(),
            "config": self.check_config(),
            "training_status": self.check_training_status(),
            "daemon_running": self.check_daemon_running(),
            "disk_space": self.check_disk_space(),
        }
        
        # Add warnings
        state['warnings'] = self.generate_warnings(state)
        
        return state
    
    def save_state(self):
        """Generate and save state file"""
        state = self.generate_state()
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        return state
    
    def print_state(self, state):
        """Print human-readable state"""
        print("\n" + "="*80)
        print("SYSTEM STATE REPORT")
        print("="*80)
        print(f"\nLast Updated: {state['last_updated']}")
        
        # Current Model
        print("\n📦 CURRENT MODEL:")
        model = state['current_model']
        if model['exists']:
            print(f"   ✓ Exists: {model['path']}")
            print(f"   Size: {model['size_gb']} GB")
            print(f"   Training Steps: {model['training_steps']}")
            if model['last_training']:
                print(f"   Last Training: {model['last_training']}")
            if model.get('has_adapter'):
                print(f"   Has Adapter: Yes")
        else:
            print(f"   ✗ No current model")
        
        # Versions
        print(f"\n📚 VERSIONS: {len(state['versions'])} saved")
        for v in state['versions'][:5]:  # Show first 5
            print(f"   • {v['id']}: {v['description'][:50]} ({v['date']}, {v['size_gb']}GB)")
        if len(state['versions']) > 5:
            print(f"   ... and {len(state['versions']) - 5} more")
        
        # Config
        print("\n⚙️  CONFIG:")
        config = state['config']
        if config['exists']:
            print(f"   Model: {config['model_name']}")
            print(f"   Base: {config['base_model']}")
            print(f"   Max Length: {config['max_length']}")
            print(f"   Locked Params: {', '.join(config['locked_params'])}")
        else:
            print(f"   ✗ Config missing!")
        
        # Training Status
        print("\n🎯 TRAINING STATUS:")
        status = state['training_status']
        print(f"   Status: {status['status']}")
        print(f"   Steps: {status['current_step']}")
        print(f"   Evaluations: {status['total_evals']}")
        
        # Daemon
        print(f"\n🔄 DAEMON: {'Running ✓' if state['daemon_running'] else 'Not Running ✗'}")
        
        # Disk Space
        print("\n💾 DISK SPACE:")
        disk = state['disk_space']
        if disk['available_gb'] > 0:
            print(f"   Available: {disk['available_gb']} GB")
            print(f"   Used: {disk['used_percent']}%")
        
        # Warnings
        if state['warnings']:
            print("\n⚠️  WARNINGS:")
            for category, warning in state['warnings']:
                print(f"   • [{category}] {warning}")
        else:
            print("\n✅ NO WARNINGS - System healthy")
        
        print("\n" + "="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="System State Tracker")
    parser.add_argument('--check', action='store_true', help='Check and display state')
    parser.add_argument('--warnings', action='store_true', help='Show warnings only')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    
    args = parser.parse_args()
    
    tracker = StateTracker()
    
    # Generate state
    state = tracker.save_state()
    
    if args.json:
        # Just output JSON
        print(json.dumps(state, indent=2))
    elif args.warnings:
        # Show warnings only
        if state['warnings']:
            print("\n⚠️  SYSTEM WARNINGS:")
            for category, warning in state['warnings']:
                print(f"   • [{category}] {warning}")
        else:
            print("\n✅ NO WARNINGS - System healthy")
    elif args.check:
        # Show full report
        tracker.print_state(state)
    else:
        # Default: save and show summary
        print(f"✅ System state saved to: {tracker.state_file}")
        if state['warnings']:
            print(f"\n⚠️  {len(state['warnings'])} warnings found:")
            for category, warning in state['warnings']:
                print(f"   • [{category}] {warning}")
        else:
            print("✅ No warnings - system healthy")


if __name__ == '__main__':
    main()
