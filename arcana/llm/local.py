"""
Local inference server interface for Arcana planner.

Connects to a local vLLM, llama.cpp, or similar OpenAI-compatible server.
"""

import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from .base import BaseLLM


class LocalLLM(BaseLLM):
    """
    Local inference server interface (OpenAI-compatible API).

    Usage:
        llm = LocalLLM()  # Defaults to localhost:8765
        llm = LocalLLM(base_url="http://192.168.1.100:8000")

        response = llm.complete(system_prompt, user_prompt)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call local inference server and return response."""
        # Build request
        endpoint = f"{self.base_url}/v1/chat/completions"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.model:
            payload["model"] = self.model

        # Make request
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
        }

        req = Request(endpoint, data=data, headers=headers, method='POST')

        try:
            with urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
        except URLError as e:
            raise ConnectionError(f"Failed to connect to {endpoint}: {e}")

        # Extract response text
        try:
            return result['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format: {result}")

    def health_check(self) -> bool:
        """Check if the server is reachable."""
        try:
            req = Request(f"{self.base_url}/health", method='GET')
            with urlopen(req, timeout=5) as response:
                return response.status == 200
        except:
            return False

    def __repr__(self):
        return f"LocalLLM(base_url={self.base_url!r})"
