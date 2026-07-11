"""Pluggable LLM backends.

One abstraction, two wire protocols:
- ``AnthropicBackend`` speaks the Anthropic Messages API natively.
- ``OpenAICompatBackend`` speaks the OpenAI chat-completions protocol, which also covers
  Groq, OpenRouter, and a local Ollama server — so the $0 no-key path and the paid paths
  share one code path.

Raw ``httpx`` is used instead of provider SDKs to keep the dependency surface small and
the retry/timeout behaviour identical across providers.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out
        self.calls += 1


class BackendError(RuntimeError):
    pass


@dataclass
class LLMBackend(ABC):
    model: str
    timeout: float = 120.0
    max_retries: int = 5
    usage: Usage = field(default_factory=Usage)

    @abstractmethod
    def _request(self, system: str, user: str, temperature: float, max_tokens: int) -> str: ...

    def complete(
        self, system: str, user: str, *, temperature: float = 0.7, max_tokens: int = 512
    ) -> str:
        delay = 2.0
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                return self._request(system, user, temperature, max_tokens)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in RETRYABLE_STATUS:
                    raise BackendError(
                        f"{self.model}: HTTP {e.response.status_code}: {e.response.text[:300]}"
                    ) from e
                last_err = e
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise BackendError(f"{self.model}: retries exhausted: {last_err}")


@dataclass
class AnthropicBackend(LLMBackend):
    api_key_env: str = "ANTHROPIC_API_KEY"

    def _request(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise BackendError(f"{self.api_key_env} is not set")
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        u = data.get("usage", {})
        self.usage.add(u.get("input_tokens", 0), u.get("output_tokens", 0))
        return "".join(b.get("text", "") for b in data["content"])


@dataclass
class OpenAICompatBackend(LLMBackend):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    require_key: bool = True

    def _request(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key and self.require_key:
            raise BackendError(f"{self.api_key_env} is not set")
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key or 'local'}",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        u = data.get("usage") or {}
        self.usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return data["choices"][0]["message"]["content"] or ""


# provider name -> (factory kwargs, default model)
PROVIDERS: dict[str, dict] = {
    "anthropic": {"default_model": "claude-sonnet-5"},
    "openai": {
        "default_model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "default_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "default_model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "require_key": False,
    },
}


def make_backend(
    provider: str, model: str | None = None, base_url: str | None = None
) -> LLMBackend:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    spec = dict(PROVIDERS[provider])
    resolved_model = model or spec.pop("default_model")
    spec.pop("default_model", None)
    if base_url:
        spec["base_url"] = base_url
    if provider == "anthropic":
        return AnthropicBackend(model=resolved_model)
    return OpenAICompatBackend(model=resolved_model, **spec)
