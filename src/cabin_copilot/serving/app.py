"""FastAPI service around the fine-tuned model.

The model itself runs in any OpenAI-compatible server (locally: Ollama with the
exported GGUF). Configuration via env:

- CABIN_PROVIDER (default "ollama"), CABIN_MODEL (default "cabin-copilot"),
  CABIN_BASE_URL (optional override, e.g. host.docker.internal from a container)

`create_app(backend=...)` allows dependency injection for tests.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from cabin_copilot import __version__
from cabin_copilot.schemas import CoachResponse, Signals
from cabin_copilot.teacher.base import BackendError, LLMBackend, make_backend
from cabin_copilot.teacher.generate import parse_response
from cabin_copilot.teacher.prompts import STUDENT_SYSTEM_PROMPT


def create_app(backend: LLMBackend | None = None) -> FastAPI:
    if backend is None:
        backend = make_backend(
            os.environ.get("CABIN_PROVIDER", "ollama"),
            os.environ.get("CABIN_MODEL", "cabin-copilot"),
            os.environ.get("CABIN_BASE_URL") or None,
        )

    app = FastAPI(title="CabinCopilot", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model": backend.model, "version": __version__}

    @app.post("/coach", response_model=CoachResponse)
    def coach(signals: Signals) -> CoachResponse:
        try:
            raw = backend.complete(
                STUDENT_SYSTEM_PROMPT, signals.signals_json(), temperature=0.0
            )
            response = parse_response(raw)
        except BackendError as e:
            raise HTTPException(status_code=502, detail=f"model backend error: {e}") from e
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            raise HTTPException(
                status_code=502, detail=f"model returned invalid contract: {e}"
            ) from e
        return response

    return app


app = create_app()
