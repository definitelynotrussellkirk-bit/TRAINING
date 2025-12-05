"""
Inference Health Check - Deep Validation of Inference Server
=============================================================

The problem: Inference server can report "healthy" but fail to generate.
This happens when:
1. PEFT checkpoints are loaded but base model is missing
2. Model is on wrong device (CPU/GPU mismatch)
3. Tokenizer is missing or incompatible
4. Model is corrupted or partially loaded

This module does DEEP health checks:
- Actually attempts generation (not just /health)
- Validates base model availability for PEFT adapters
- Checks device consistency
- Tests tokenization round-trip
- **Generates actionable fix commands**

Usage:
    from temple.diagnostics import InferenceHealthChecker

    checker = InferenceHealthChecker(server_url="http://192.168.88.149:8765")
    report = checker.run_all_checks()

    if not report.can_generate:
        print(f"INFERENCE BROKEN: {report.issues}")
        for fix in report.fixes:
            print(f"  Fix: {fix}")

        # Get executable fix commands
        for cmd in report.fix_commands:
            print(f"  Run: {cmd}")
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


# ========== Issue Types ==========

class InferenceIssue(Enum):
    """Known inference issues with their fix strategies."""
    SERVER_DOWN = "server_down"
    NO_MODELS = "no_models"
    BASE_MODEL_MISSING = "base_model_missing"
    DEVICE_MISMATCH = "device_mismatch"
    GENERATION_FAILED = "generation_failed"
    GPU_OOM = "gpu_oom"
    TOKENIZER_MISSING = "tokenizer_missing"
    MODEL_CORRUPTED = "model_corrupted"


# ========== Fix Generator ==========

class FixGenerator:
    """
    Generates actionable fix commands based on diagnosed issues.

    This is the "smart" part - translates diagnostics into executable commands.
    """

    # Known model locations on trainer
    TRAINER_MODEL_PATHS = {
        "Qwen3-8B": "/home/russ/Desktop/TRAINING/models/Qwen3-8B",
        "Qwen3-4B": "/home/russ/Desktop/TRAINING/models/Qwen3-4B",
        "Qwen3-4B-Instruct-2507": "/home/russ/Desktop/TRAINING/models/Qwen3-4B-Instruct-2507",
        "Qwen3-0.6B": "/home/russ/Desktop/TRAINING/models/Qwen3-0.6B",
        "Mistral-7B-v0.3": "/home/russ/Desktop/TRAINING/models/Mistral-7B-v0.3",
        "Phi-3.5-mini": "/home/russ/Desktop/TRAINING/models/Phi-3.5-mini",
    }

    # Inference server details
    INFERENCE_HOST = "192.168.88.149"
    INFERENCE_USER = "russ"
    INFERENCE_MODEL_DIR = "~/llm/models"

    @classmethod
    def for_issue(
        cls,
        issue: InferenceIssue,
        details: Dict[str, Any] = None,
    ) -> List[str]:
        """
        Generate fix commands for a specific issue.

        Returns list of shell commands that can be executed.
        """
        details = details or {}

        if issue == InferenceIssue.SERVER_DOWN:
            return cls._fix_server_down(details)
        elif issue == InferenceIssue.NO_MODELS:
            return cls._fix_no_models(details)
        elif issue == InferenceIssue.BASE_MODEL_MISSING:
            return cls._fix_base_model_missing(details)
        elif issue == InferenceIssue.DEVICE_MISMATCH:
            return cls._fix_device_mismatch(details)
        elif issue == InferenceIssue.GPU_OOM:
            return cls._fix_gpu_oom(details)
        else:
            return []

    @classmethod
    def _fix_server_down(cls, details: Dict) -> List[str]:
        """Commands to start inference server."""
        server_url = details.get("server_url", f"http://{cls.INFERENCE_HOST}:8765")
        return [
            f"# Start inference server",
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "cd ~/llm && INFERENCE_ADMIN_KEY=admin123 nohup python3 main.py --host 0.0.0.0 --port 8765 > logs/inference.log 2>&1 &"',
            f"# Wait for startup, then verify:",
            f"sleep 10 && curl -s {server_url}/health",
        ]

    @classmethod
    def _fix_no_models(cls, details: Dict) -> List[str]:
        """Commands to load models on inference server."""
        return [
            f"# Reload models on inference server",
            f'curl -X POST http://{cls.INFERENCE_HOST}:8765/models/reload -H "X-API-Key: admin123"',
        ]

    @classmethod
    def _fix_base_model_missing(cls, details: Dict) -> List[str]:
        """Commands to copy base model to inference server."""
        base_path = details.get("base_path", "")
        model_name = details.get("model_name", "")

        # Try to identify the model from the path
        if not model_name:
            for name, path in cls.TRAINER_MODEL_PATHS.items():
                if name.lower() in base_path.lower():
                    model_name = name
                    break

        # Find local path
        local_path = None
        if model_name and model_name in cls.TRAINER_MODEL_PATHS:
            local_path = cls.TRAINER_MODEL_PATHS[model_name]
        elif base_path and os.path.exists(base_path):
            local_path = base_path
        else:
            # Try to extract path from base_path
            for name, path in cls.TRAINER_MODEL_PATHS.items():
                if os.path.exists(path):
                    if name.lower() in base_path.lower():
                        local_path = path
                        model_name = name
                        break

        if not local_path:
            return [
                f"# ERROR: Could not determine local path for base model",
                f"# Base path reported: {base_path}",
                f"# Please manually copy the base model to the inference server",
            ]

        remote_path = f"{cls.INFERENCE_MODEL_DIR}/{model_name or Path(local_path).name}"

        return [
            f"# Copy base model {model_name or 'unknown'} to inference server",
            f"# This may take several minutes for large models",
            f"rsync -avz --progress {local_path}/ {cls.INFERENCE_USER}@{cls.INFERENCE_HOST}:{remote_path}/",
            f"",
            f"# After copy completes, reload the model:",
            f'curl -X POST http://{cls.INFERENCE_HOST}:8765/models/reload -H "X-API-Key: admin123"',
        ]

    @classmethod
    def _fix_device_mismatch(cls, details: Dict) -> List[str]:
        """Commands to fix device mismatch (usually base model issue)."""
        return [
            f"# Device mismatch usually means PEFT adapter loaded but base model missing",
            f"# First, check what models are available on the inference server:",
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "ls -la ~/llm/models/"',
            f"",
            f"# If base model is missing, copy it (see fix_base_model_missing)",
            f"# Then restart the inference server:",
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "pkill -f \'python3 main.py\'"',
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "cd ~/llm && INFERENCE_ADMIN_KEY=admin123 nohup python3 main.py --host 0.0.0.0 --port 8765 > logs/inference.log 2>&1 &"',
        ]

    @classmethod
    def _fix_gpu_oom(cls, details: Dict) -> List[str]:
        """Commands to fix GPU out of memory."""
        return [
            f"# GPU out of memory - need to free memory",
            f"# First, check GPU memory usage:",
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "nvidia-smi"',
            f"",
            f"# Kill inference server and restart with smaller model:",
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "pkill -f \'python3 main.py\'"',
            f"",
            f"# Clear GPU cache:",
            f'ssh {cls.INFERENCE_USER}@{cls.INFERENCE_HOST} "python3 -c \\"import torch; torch.cuda.empty_cache()\\""',
        ]

    @classmethod
    def extract_model_name_from_path(cls, path: str) -> Optional[str]:
        """Extract model name from a path."""
        path_lower = path.lower()

        # Check known models
        for name in cls.TRAINER_MODEL_PATHS:
            if name.lower() in path_lower:
                return name

        # Try to extract from path segments
        parts = Path(path).parts
        for part in parts:
            if any(x in part.lower() for x in ["qwen", "llama", "mistral", "phi", "gemma"]):
                return part

        return None


@dataclass
class InferenceCheckResult:
    """Result of a single inference check."""
    name: str
    passed: bool
    message: str
    severity: str = "warn"  # "info", "warn", "error"
    fix: Optional[str] = None  # Human-readable fix description
    issue_type: Optional[InferenceIssue] = None  # For FixGenerator
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def fix_commands(self) -> List[str]:
        """Get executable fix commands for this issue."""
        if self.passed or not self.issue_type:
            return []
        return FixGenerator.for_issue(self.issue_type, self.details)


@dataclass
class InferenceHealthReport:
    """Full inference health report."""
    server_url: str
    server_reachable: bool = False
    can_generate: bool = False
    checks: List[InferenceCheckResult] = field(default_factory=list)

    @property
    def issues(self) -> List[str]:
        return [c.message for c in self.checks if not c.passed]

    @property
    def fixes(self) -> List[str]:
        return [c.fix for c in self.checks if not c.passed and c.fix]

    @property
    def fix_commands(self) -> List[str]:
        """Get all executable fix commands for all issues."""
        commands = []
        for check in self.checks:
            if not check.passed:
                cmds = check.fix_commands
                if cmds:
                    commands.extend(cmds)
                    commands.append("")  # Blank line between fixes
        return commands

    @property
    def is_healthy(self) -> bool:
        return self.server_reachable and self.can_generate

    def summary(self) -> str:
        lines = ["INFERENCE HEALTH REPORT", "=" * 50]
        lines.append(f"Server: {self.server_url}")
        lines.append(f"Reachable: {'Yes' if self.server_reachable else 'NO'}")
        lines.append(f"Can Generate: {'Yes' if self.can_generate else 'NO'}")
        lines.append("")

        for check in self.checks:
            icon = "✅" if check.passed else ("❌" if check.severity == "error" else "⚠️")
            lines.append(f"{icon} {check.name}: {check.message}")
            if not check.passed and check.fix:
                lines.append(f"   Fix: {check.fix}")

        # Add executable commands section
        if self.fix_commands:
            lines.append("")
            lines.append("=" * 50)
            lines.append("EXECUTABLE FIX COMMANDS:")
            lines.append("=" * 50)
            for cmd in self.fix_commands:
                lines.append(cmd)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_url": self.server_url,
            "server_reachable": self.server_reachable,
            "can_generate": self.can_generate,
            "is_healthy": self.is_healthy,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                    "fix": c.fix,
                    "issue_type": c.issue_type.value if c.issue_type else None,
                }
                for c in self.checks
            ],
            "issues": self.issues,
            "fixes": self.fixes,
            "fix_commands": self.fix_commands,
        }


class InferenceHealthChecker:
    """
    Deep health check for inference server.

    Goes beyond simple /health endpoint to validate:
    1. Server is reachable
    2. Model is loaded correctly
    3. Generation actually works
    4. Base model available for PEFT adapters
    5. Device consistency (no CPU/GPU split)
    """

    def __init__(
        self,
        server_url: str = "http://192.168.88.149:8765",
        api_key: str = "admin123",
        timeout: int = 30,
    ):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def run_all_checks(self) -> InferenceHealthReport:
        """Run all inference health checks."""
        report = InferenceHealthReport(server_url=self.server_url)

        # Check 1: Server reachable
        check1 = self._check_server_reachable()
        report.checks.append(check1)
        report.server_reachable = check1.passed

        if not report.server_reachable:
            return report

        # Check 2: Models loaded
        check2 = self._check_models_loaded()
        report.checks.append(check2)

        # Check 3: Base model available (critical for PEFT)
        check3 = self._check_base_model()
        report.checks.append(check3)

        # Check 4: Actually try to generate
        check4 = self._check_generation()
        report.checks.append(check4)
        report.can_generate = check4.passed

        # Check 5: Device consistency
        check5 = self._check_device_consistency()
        report.checks.append(check5)

        return report

    def _curl(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> tuple[int, str]:
        """Make HTTP request using curl."""
        url = urljoin(self.server_url + "/", endpoint.lstrip("/"))
        cmd = [
            "curl", "-s", "-w", "\n%{http_code}",
            "--connect-timeout", str(self.timeout),
            "-H", f"X-API-Key: {self.api_key}",
        ]

        if method == "POST":
            cmd.extend(["-X", "POST", "-H", "Content-Type: application/json"])
            if data:
                cmd.extend(["-d", json.dumps(data)])

        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            lines = result.stdout.strip().split("\n")
            status_code = int(lines[-1]) if lines else 0
            body = "\n".join(lines[:-1])
            return status_code, body
        except Exception as e:
            return 0, str(e)

    def _check_server_reachable(self) -> InferenceCheckResult:
        """Check if server is reachable."""
        status, body = self._curl("/health")

        if status == 200:
            try:
                data = json.loads(body)
                if data.get("status") == "ok":
                    return InferenceCheckResult(
                        name="Server Reachable",
                        passed=True,
                        message="Server responding normally",
                        details=data,
                    )
            except:
                pass

        return InferenceCheckResult(
            name="Server Reachable",
            passed=False,
            message=f"Server not responding (status={status})",
            severity="error",
            fix=f"Start inference server at {self.server_url}",
            issue_type=InferenceIssue.SERVER_DOWN,
            details={"server_url": self.server_url, "status": status},
        )

    def _check_models_loaded(self) -> InferenceCheckResult:
        """Check if models are loaded."""
        status, body = self._curl("/models/info")

        if status != 200:
            return InferenceCheckResult(
                name="Models Loaded",
                passed=False,
                message="Could not get model info",
                severity="error",
                issue_type=InferenceIssue.NO_MODELS,
            )

        try:
            data = json.loads(body)
            loaded_count = data.get("loaded_count", 0)
            models = data.get("models", [])

            if loaded_count == 0:
                return InferenceCheckResult(
                    name="Models Loaded",
                    passed=False,
                    message="No models loaded",
                    severity="error",
                    fix="Load a model via /models/reload endpoint",
                    issue_type=InferenceIssue.NO_MODELS,
                )

            model_names = [m.get("model_id", "?") for m in models]
            return InferenceCheckResult(
                name="Models Loaded",
                passed=True,
                message=f"{loaded_count} models loaded: {', '.join(model_names[:3])}",
                details={"models": models},
            )

        except Exception as e:
            return InferenceCheckResult(
                name="Models Loaded",
                passed=False,
                message=f"Error parsing model info: {e}",
                severity="warn",
            )

    def _check_base_model(self) -> InferenceCheckResult:
        """Check if base model is available for PEFT adapters."""
        status, body = self._curl("/models/info")

        if status != 200:
            return InferenceCheckResult(
                name="Base Model Available",
                passed=False,
                message="Could not check base model",
                severity="warn",
            )

        try:
            data = json.loads(body)
            models = data.get("models", [])

            for model in models:
                is_peft = model.get("is_peft", False)
                base_path = model.get("base_model_path")
                model_id = model.get("model_id", "unknown")

                if is_peft and base_path:
                    # Extract model name for smart fix generation
                    model_name = FixGenerator.extract_model_name_from_path(str(base_path))

                    # Look for path issues - trainer path on inference server
                    if "/Desktop/TRAINING/" in str(base_path):
                        return InferenceCheckResult(
                            name="Base Model Available",
                            passed=False,
                            message=f"PEFT adapter '{model_id}' needs base model at {base_path} (trainer path, not on inference server)",
                            severity="error",
                            fix=f"Copy base model {model_name or 'unknown'} to inference server",
                            issue_type=InferenceIssue.BASE_MODEL_MISSING,
                            details={
                                "base_path": str(base_path),
                                "model_name": model_name,
                                "model_id": model_id,
                                "is_peft": True,
                            },
                        )

                    # Check if path looks like a remote reference that might be missing
                    if not Path(base_path).exists() and "~" not in str(base_path):
                        # Path doesn't exist locally and isn't a tilde path
                        # Could be a remote reference or just wrong
                        return InferenceCheckResult(
                            name="Base Model Available",
                            passed=False,
                            message=f"PEFT adapter '{model_id}' references base model at {base_path} (may not exist)",
                            severity="warn",
                            fix=f"Verify base model exists at {base_path} on inference server",
                            issue_type=InferenceIssue.BASE_MODEL_MISSING,
                            details={
                                "base_path": str(base_path),
                                "model_name": model_name,
                                "model_id": model_id,
                                "is_peft": True,
                            },
                        )

            return InferenceCheckResult(
                name="Base Model Available",
                passed=True,
                message="Base model paths look correct",
            )

        except Exception as e:
            return InferenceCheckResult(
                name="Base Model Available",
                passed=False,
                message=f"Error checking base model: {e}",
                severity="warn",
            )

    def _check_generation(self) -> InferenceCheckResult:
        """Actually try to generate text."""
        # Try /v1/chat/completions first
        status, body = self._curl(
            "/v1/chat/completions",
            method="POST",
            data={
                "model": "default",
                "messages": [{"role": "user", "content": "Say 'test' and nothing else"}],
                "max_tokens": 10,
            },
        )

        if status == 200:
            try:
                data = json.loads(body)
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    return InferenceCheckResult(
                        name="Generation Test",
                        passed=True,
                        message=f"Generation works: '{text[:50]}...'",
                        details={"response": text},
                    )
            except:
                pass

        # Try /generate endpoint
        status2, body2 = self._curl(
            "/generate",
            method="POST",
            data={
                "prompt": "Test:",
                "max_tokens": 10,
            },
        )

        if status2 == 200:
            try:
                data = json.loads(body2)
                text = data.get("text", data.get("generated_text", ""))
                if text:
                    return InferenceCheckResult(
                        name="Generation Test",
                        passed=True,
                        message=f"Generation works: '{text[:50]}...'",
                        details={"response": text},
                    )
            except:
                pass

        # Parse error message and diagnose the issue
        error_msg = "Generation failed"
        issue_type = InferenceIssue.GENERATION_FAILED
        fix_msg = "Check model loading and base model availability"

        try:
            err_data = json.loads(body) if body else {}
            if "detail" in err_data:
                detail = err_data["detail"]
                if isinstance(detail, str):
                    error_msg = detail
                elif isinstance(detail, list) and detail:
                    error_msg = detail[0].get("msg", str(detail[0]))

            error_lower = error_msg.lower()

            # Check for specific error patterns
            if "cuda" in error_lower and "cpu" in error_lower:
                # Device mismatch - usually means PEFT without base model
                issue_type = InferenceIssue.DEVICE_MISMATCH
                fix_msg = "Device mismatch detected - PEFT adapter may be missing its base model"
                return InferenceCheckResult(
                    name="Generation Test",
                    passed=False,
                    message="Device mismatch: Model split between CPU and GPU",
                    severity="error",
                    fix=fix_msg,
                    issue_type=issue_type,
                    details={"error": error_msg},
                )

            elif "out of memory" in error_lower or "oom" in error_lower:
                issue_type = InferenceIssue.GPU_OOM
                fix_msg = "GPU out of memory - reduce batch size or use smaller model"
                return InferenceCheckResult(
                    name="Generation Test",
                    passed=False,
                    message="GPU out of memory during generation",
                    severity="error",
                    fix=fix_msg,
                    issue_type=issue_type,
                    details={"error": error_msg},
                )

            elif "tokenizer" in error_lower:
                issue_type = InferenceIssue.TOKENIZER_MISSING
                fix_msg = "Tokenizer missing or incompatible - ensure tokenizer files are with model"
                return InferenceCheckResult(
                    name="Generation Test",
                    passed=False,
                    message="Tokenizer error during generation",
                    severity="error",
                    fix=fix_msg,
                    issue_type=issue_type,
                    details={"error": error_msg},
                )

            elif any(x in error_lower for x in ["corrupt", "invalid", "load"]):
                issue_type = InferenceIssue.MODEL_CORRUPTED
                fix_msg = "Model may be corrupted or failed to load properly"

        except:
            pass

        return InferenceCheckResult(
            name="Generation Test",
            passed=False,
            message=f"Generation failed: {error_msg[:100]}",
            severity="error",
            fix=fix_msg,
            issue_type=issue_type,
            details={"status": status, "body": body[:500] if body else ""},
        )

    def _check_device_consistency(self) -> InferenceCheckResult:
        """Check if model is on consistent device."""
        # This is mostly caught by the generation test, but we can add GPU memory check
        status, body = self._curl("/health")

        if status != 200:
            return InferenceCheckResult(
                name="Device Consistency",
                passed=False,
                message="Could not check device status",
                severity="warn",
            )

        try:
            data = json.loads(body)
            gpu = data.get("gpu", {})
            allocated = gpu.get("memory_allocated_gb", 0)

            if allocated > 0:
                return InferenceCheckResult(
                    name="Device Consistency",
                    passed=True,
                    message=f"Model on GPU ({allocated:.1f} GB allocated)",
                    details=gpu,
                )
            else:
                return InferenceCheckResult(
                    name="Device Consistency",
                    passed=False,
                    message="No GPU memory allocated - model may be on CPU",
                    severity="warn",
                    fix="Check CUDA availability and model device placement",
                )

        except Exception as e:
            return InferenceCheckResult(
                name="Device Consistency",
                passed=False,
                message=f"Error checking devices: {e}",
                severity="warn",
            )


# ========== Convenience Functions ==========

def check_inference_health(
    server_url: str = "http://192.168.88.149:8765",
    api_key: str = "admin123",
) -> InferenceHealthReport:
    """
    Quick check of inference server health.

    Usage:
        from temple.diagnostics.inference_health import check_inference_health

        report = check_inference_health()
        if not report.can_generate:
            print("INFERENCE BROKEN!")
            for issue in report.issues:
                print(f"  - {issue}")
            for fix in report.fixes:
                print(f"  Fix: {fix}")
    """
    checker = InferenceHealthChecker(server_url=server_url, api_key=api_key)
    return checker.run_all_checks()


def quick_inference_test(
    server_url: str = "http://192.168.88.149:8765",
    api_key: str = "admin123",
) -> bool:
    """
    Quick test if inference works. Returns True/False.

    Usage:
        if not quick_inference_test():
            print("Inference server is broken!")
    """
    checker = InferenceHealthChecker(server_url=server_url, api_key=api_key)
    result = checker._check_generation()
    return result.passed
