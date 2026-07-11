FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

# The model runs in an external OpenAI-compatible server (e.g. Ollama on the host);
# point the service at it, e.g. CABIN_BASE_URL=http://host.docker.internal:11434/v1
ENV CABIN_PROVIDER=ollama CABIN_MODEL=cabin-copilot

EXPOSE 8080
CMD ["uvicorn", "cabin_copilot.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
