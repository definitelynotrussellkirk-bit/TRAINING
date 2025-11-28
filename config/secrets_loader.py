"""Load secrets from config/secrets.json."""
import json
import os

_secrets = None

def get_secrets() -> dict:
    """Load secrets from config/secrets.json."""
    global _secrets
    if _secrets is None:
        secrets_path = os.path.join(os.path.dirname(__file__), "secrets.json")
        if os.path.exists(secrets_path):
            with open(secrets_path) as f:
                _secrets = json.load(f)
        else:
            _secrets = {}
    return _secrets

def get_inference_api_key() -> str:
    """Get the inference API key."""
    # Environment variable takes priority
    env_key = os.environ.get("INFERENCE_API_KEY")
    if env_key:
        return env_key
    # Fall back to secrets file
    secrets = get_secrets()
    return secrets.get("inference_api_key", secrets.get("admin_key", "admin123"))
