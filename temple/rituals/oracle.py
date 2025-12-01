"""
Ritual of the Oracle - Inference server diagnostics.

This ritual checks the health of inference capabilities:
- Remote inference server connectivity
- Inference API health
- Test inference with simple prompt
"""

import json
import os
from datetime import datetime
from typing import List
import urllib.request
import urllib.error

from temple.schemas import RitualCheckResult
from temple.cleric import register_ritual


@register_ritual("oracle", "Ritual of the Oracle", "Inference server diagnostics")
def run() -> List[RitualCheckResult]:
    """Execute all oracle ritual checks."""
    results = []
    results.append(_check_inference_connectivity())
    results.append(_check_inference_health())
    results.append(_check_inference_test())
    return results


def _get_inference_url() -> str:
    """Get inference server URL from config."""
    try:
        from core.paths import get_base_dir
        hosts_file = get_base_dir() / "config" / "hosts.json"
        if hosts_file.exists():
            with open(hosts_file) as f:
                hosts_config = json.load(f)
            # Get default inference host ID
            default_inference = hosts_config.get("default_inference", "3090")
            # Look up that host's inference service
            host_entry = hosts_config.get("hosts", {}).get(default_inference, {})
            host = host_entry.get("host", "localhost")
            inference_service = host_entry.get("services", {}).get("inference", {})
            port = inference_service.get("port", 8765)
            return f"http://{host}:{port}"
    except:
        pass
    # Fallback - use localhost or INFERENCE_HOST env var
    host = os.environ.get("INFERENCE_HOST", "localhost")
    port = os.environ.get("INFERENCE_PORT", "8765")
    return f"http://{host}:{port}"


def _check_inference_connectivity() -> RitualCheckResult:
    """Check basic connectivity to inference server."""
    start = datetime.utcnow()
    base_url = _get_inference_url()
    url = f"{base_url}/health"

    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            status_code = resp.status
            body = resp.read().decode('utf-8')

        return RitualCheckResult(
            id="inference_connectivity",
            name="Inference Connectivity",
            description="Check basic connectivity to inference server",
            status="ok" if status_code == 200 else "warn",
            category="inference",
            details={
                "url": url,
                "status_code": status_code,
                "response": body[:200] if body else None,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="inference_connectivity",
            name="Inference Connectivity",
            description="Check basic connectivity to inference server",
            status="fail",
            category="inference",
            details={
                "url": url,
                "error": str(e.reason),
            },
            remediation=f"Start inference server on remote machine or check network. URL: {base_url}",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="inference_connectivity",
            name="Inference Connectivity",
            description="Check basic connectivity to inference server",
            status="fail",
            category="inference",
            details={
                "url": url,
                "error": str(e),
            },
            remediation="Check inference server configuration in config/hosts.json",
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_inference_health() -> RitualCheckResult:
    """Check inference server reports healthy."""
    start = datetime.utcnow()
    base_url = _get_inference_url()
    url = f"{base_url}/v1/models"

    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = resp.read().decode('utf-8')
            data = json.loads(body) if body else {}

        models = data.get("data", [])
        model_ids = [m.get("id") for m in models]

        return RitualCheckResult(
            id="inference_health",
            name="Inference Health",
            description="Check inference server reports models available",
            status="ok" if models else "warn",
            category="inference",
            details={
                "url": url,
                "model_count": len(models),
                "models": model_ids,
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="inference_health",
            name="Inference Health",
            description="Check inference server reports models available",
            status="fail",
            category="inference",
            details={
                "url": url,
                "error": str(e.reason),
            },
            remediation="Inference server may not be running or model not loaded",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="inference_health",
            name="Inference Health",
            description="Check inference server reports models available",
            status="fail",
            category="inference",
            details={
                "url": url,
                "error": str(e),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )


def _check_inference_test() -> RitualCheckResult:
    """Test actual inference with a simple prompt."""
    start = datetime.utcnow()
    base_url = _get_inference_url()
    url = f"{base_url}/v1/completions"

    try:
        payload = json.dumps({
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.1,
        }).encode('utf-8')

        req = urllib.request.Request(
            url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = resp.read().decode('utf-8')
            data = json.loads(body) if body else {}

        choices = data.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        usage = data.get("usage", {})

        # Check if we got a reasonable response
        has_output = len(text.strip()) > 0

        return RitualCheckResult(
            id="inference_test",
            name="Inference Test",
            description="Test actual inference with a simple prompt",
            status="ok" if has_output else "warn",
            category="inference",
            details={
                "url": url,
                "prompt": "The capital of France is",
                "response": text[:100],
                "tokens_generated": usage.get("completion_tokens", 0),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except urllib.error.URLError as e:
        return RitualCheckResult(
            id="inference_test",
            name="Inference Test",
            description="Test actual inference with a simple prompt",
            status="fail",
            category="inference",
            details={
                "url": url,
                "error": str(e.reason),
            },
            remediation="Inference server not responding. Check if model is loaded.",
            started_at=start,
            finished_at=datetime.utcnow(),
        )
    except Exception as e:
        return RitualCheckResult(
            id="inference_test",
            name="Inference Test",
            description="Test actual inference with a simple prompt",
            status="fail",
            category="inference",
            details={
                "url": url,
                "error": str(e),
            },
            started_at=start,
            finished_at=datetime.utcnow(),
        )
