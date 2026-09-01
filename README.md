# model-agent-frontend

A Streamlit-based web UI for [model-agent](https://github.com/model-verse/model-agent), a local runtime provisioner. Provides a visual interface for managing model artifacts, syncing from MinIO to backend storage (ComfyUI or vLLM), and adopting local files.

## Features

- **Managed Repos** — List repositories managed by model-agent with metrics (total size, artifact count, pinned status). Supports removing repos with cascade option.

- **Sync** — Core artifact synchronization flow:
  - Select backend: **ComfyUI** or **vLLM**
  - Enter HuggingFace `Repo ID` (e.g., `Comfy-Org/MiniMax-H3`)
  - Click **"Discover from MinIO"** to fetch available artifacts
  - Intelligent folder suggestions based on artifact names
  - Per-artifact destination folder selection
  - Progress tracking with status indicators
  - Sync all or sync selected artifacts
  - 10-minute default timeout for large model downloads

- **Adopt** — Fold existing local files into the managed store by specifying a local path and repo ID.

## Setup

### Prerequisites

- Docker and Docker Compose (recommended)
- Or Python 3.12+

### Using Docker Compose (recommended)

```bash
docker compose up --build
```

- Builds the image from the local `Dockerfile`
- Runs `streamlit run app.py` on port 8501
- Exposes port mapping: `8501:8501`
- Sets `MODEL_AGENT_API_URL` env var to `http://model-agent-api:8500`

### Running directly

```bash
python3 -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

- App available at `http://localhost:8501`
- Default API URL: `http://model-agent-api:8500` (overridable via `MODEL_AGENT_API_URL` env var)

### Configuration

- Modify `.env` to change `MODEL_AGENT_API_URL`
- No authentication — intended for internal LAN use only

## Architecture

Three-tab interface:

1. **📦 Managed Repos** — Lists all repositories managed by model-agent with metrics and cascade removal support.

2. **⬇️ Sync** — Core artifact synchronization flow with backend selection, HuggingFace repo discovery, intelligent folder suggestions, and progress tracking.

3. **📥 Adopt** — Fold existing local files into the managed store.

## Technical Details

- **Framework**: Streamlit
- **HTTP Client**: httpx (async)
- **Dependencies**: See `requirements.txt`
- **Docker**: `python:3.12-slim` base image