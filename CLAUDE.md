# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA AI Blueprint: LLM Router — an intelligent routing proxy that classifies user prompts and routes them to the most appropriate LLM based on task type or complexity. It exposes an OpenAI API-compatible endpoint (`/v1/chat/completions`) so it can serve as a drop-in replacement.

## Architecture

Three main components connected via Docker network:

1. **Router Controller** (`src/router-controller/`) — Rust HTTP proxy (port 8084) built with hyper/tokio. Receives OpenAI-compatible requests, calls the Router Server for classification, then proxies to the selected downstream LLM. Supports streaming.
2. **Router Server** (`src/router-server/`) — NVIDIA Triton Inference Server (port 8000) hosting trained classification models. Returns a one-hot vector indicating which LLM category the prompt belongs to.
3. **Downstream LLMs** — Any OpenAI-compatible endpoint (NVIDIA API Catalog, local NIMs, etc.) configured in `src/router-controller/config.yaml`.

**Request flow:** Client → Router Controller → Router Server (classify) → Router Controller → selected LLM → response proxied back with routing metadata in headers.

**Routing policies** are defined in `src/router-controller/config.yaml`. Each policy maps to a Triton model and lists LLMs in order matching the one-hot vector indices. Two pre-trained policies ship with the project:
- `task_router` — 12 task categories (Brainstorming, Chatbot, Classification, Code Generation, etc.)
- `complexity_router` — 7 complexity categories (Creativity, Reasoning, Contextual-Knowledge, etc.)

API keys in config.yaml use `${ENV_VAR}` substitution syntax.

## Key Source Files (Router Controller)

The Rust codebase is a Cargo workspace at `src/router-controller/`:
- `crates/llm-router-gateway-api/src/proxy.rs` — Core routing logic
- `crates/llm-router-gateway-api/src/triton.rs` — Triton inference client
- `crates/llm-router-gateway-api/src/config.rs` — YAML config loading with env var substitution
- `crates/llm-router-gateway-api/src/stream.rs` — SSE streaming support
- `crates/llm-router-gateway-api/src/metrics.rs` — Prometheus metrics

## Build & Run Commands

All services run via Docker Compose. Requires Docker Compose 2.29.1+, NVIDIA Container Toolkit, and a GPU (V100+ with 4GB for pre-trained models).

```bash
# Initial setup (installs uv, creates venv, installs Python deps)
bash setup.sh

# Download pre-trained router models from NGC (requires NGC_CLI_API_KEY)
make download

# Start core services — router-server + router-controller (requires NVIDIA_API_KEY)
make up

# Stop all services
make down

# Start demo web app (Gradio on localhost:8008)
make app

# Start Prometheus (9090) + Grafana (3000, admin/secret)
make metrics

# Start Locust load testing (localhost:8089)
make loadtest

# Start Jupyter Lab for custom model training (localhost:9999)
make build-router

# Start core services + demo app together
make all
```

## Testing

```bash
# Test router with Python client (streaming + non-streaming)
python src/test_router.py

# Test with curl
bash src/test_router.sh

# Load testing via Locust (start with `make loadtest`, then open localhost:8089)
```

CI runs on push to main: executes the Jupyter notebook via papermill, then runs pytest through a custom NVIDIA test Docker image.

## Rust Development

```bash
# Build the router controller
cd src/router-controller && cargo build

# Run clippy (workspace enforces warnings=deny and clippy restriction=deny)
cd src/router-controller && cargo clippy

# Run tests (uses wiremock for HTTP mocking)
cd src/router-controller && cargo test
```

The workspace Cargo.toml sets `warnings = "deny"` and `clippy::restriction = "deny"` — all warnings and clippy restriction lints are errors.

## Kubernetes Deployment

Helm charts are in `deploy/helm/llm-router/`. See `deploy/helm/llm-router/values.yaml` for all configurable options. Example configs in `deploy/helm/llm-router/examples/`.

## Custom Router Training

Jupyter notebooks in `customize/router-builder/` for training custom classification models. Start the environment with `make build-router`, then access JupyterLab at localhost:9999.

## Environment Variables

- `NVIDIA_API_KEY` — Required for downstream LLM calls via NVIDIA API Catalog
- `NGC_CLI_API_KEY` — Required for downloading pre-trained models from NGC
