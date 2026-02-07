# Ollama + Blackwell Validation Testing

End-to-end integration tests that exercise the full LLM Router pipeline using Ollama as the downstream LLM backend and Triton running on a Blackwell GPU.

**Pipeline:** Client → Router Controller (port 8084) → Router Server / Triton (classify prompt) → Router Controller → Ollama model → response

## Prerequisites

- Docker Compose 2.29.1+
- NVIDIA Container Toolkit
- NVIDIA Blackwell GPU (RTX PRO 6000, etc.) with driver 560+
- Ollama running on the host with these models pulled:
  - `gemma3:1b`
  - `qwen3:0.6b`
  - `qwen3-vl:30b-a3b-instruct-q4_K_M`
- Pre-trained router models downloaded:
  ```bash
  export NGC_CLI_API_KEY=<your-key>
  make download
  ```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.ollama-test.yaml` | Compose override — swaps in Blackwell Triton image, mounts Ollama router config, adds `host.docker.internal` networking |
| `src/ollama_router_config.yaml` | Router controller config mapping 12 task categories → 3 Ollama models |
| `src/router-server/router-server-blackwell.dockerfile` | Triton container built from `tritonserver:25.01-py3` (first version with Blackwell / sm_120 support) |
| `src/router-server/requirements-ollama-test.txt` | Python deps for 25.01 container (relaxed pins — original `numpy==1.25.1` and `torch==2.4.0` are incompatible with 25.01's Python 3.12 / CUDA 12.8) |
| `src/test_router_integration.py` | pytest integration tests (12 tests) |
| `src/conftest.py` | Adds `--router-url` pytest option |

## Blackwell GPU Support

The upstream Triton container (`tritonserver:24.10-py3`) does not support Blackwell GPUs. The `router-server-blackwell.dockerfile` uses `25.01-py3` which is the minimum version with Blackwell support:

| Container | CUDA | Blackwell |
|-----------|------|-----------|
| `tritonserver:24.10-py3` (upstream) | 12.6 | No |
| `tritonserver:25.01-py3` (this override) | 12.8 | Yes |

The original `requirements.txt` pins `torch==2.4.0` and `numpy==1.25.1`, which are incompatible with the 25.01 container (Python 3.12, CUDA 12.8). The override requirements file (`requirements-ollama-test.txt`) uses relaxed pins so pip resolves compatible versions.

## Route Mapping

The `ollama_router_config.yaml` maps the task_router's 12 classification categories to 3 Ollama models:

| Categories | Ollama Model | Rationale |
|------------|-------------|-----------|
| Code Generation | `gemma3:1b` | Coding tasks |
| Brainstorming, Closed QA, Open QA, Summarization | `qwen3-vl:30b-a3b-instruct-q4_K_M` | Complex reasoning / analysis |
| Chatbot, Classification, Extraction, Rewrite, Text Generation, Other, Unknown | `qwen3:0.6b` | Simple / general tasks |

## Running

```bash
# 1. Start the test stack (builds Blackwell Triton image on first run):
docker compose -f docker-compose.yaml -f docker-compose.ollama-test.yaml \
  up router-server router-controller --build -d

# 2. Wait for Triton to load all models (typically ~10s):
until curl -s -o /dev/null -w "%{http_code}" http://0.0.0.0:8000/v2/health/ready | grep -q 200; do
  sleep 2
done

# 3. Run tests:
.venv/bin/pytest src/test_router_integration.py -v

# 4. Tear down:
docker compose -f docker-compose.yaml -f docker-compose.ollama-test.yaml down
```

## Test Suite

12 tests across 4 categories:

**Triton classification (non-streaming)**
- Coding prompt → classified as "Code Generation", routed to `gemma3:1b`
- General prompt → classified to a valid category, response returned
- Complex prompt → classified and routed, response returned
- Token usage included in response

**Triton classification (streaming)**
- Coding prompt classified correctly via streaming
- Stream produces a `finish_reason` chunk
- Stream includes usage chunk when `include_usage` is set

**Manual routing**
- Explicitly select "Code Generation" category → routes to `gemma3:1b`
- Explicitly select "Chatbot" category → routes to `qwen3:0.6b`

**Error handling**
- Invalid policy name → 404
- Invalid manual model category → 404
- Missing `nim-llm-router` params → 400
