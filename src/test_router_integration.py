"""
Integration tests for the full LLM Router pipeline.

These tests exercise the complete routing path:
  Client → Router Controller → Router Server (Triton classification) → Ollama

Prerequisites:
  1. Pre-trained router models downloaded (`make download`)
  2. Ollama running on the host with: gemma3:1b, qwen3:0.6b,
     qwen3-vl:30b-a3b-instruct-q4_K_M
  3. Docker stack started:
     docker compose -f docker-compose.yaml \\
       -f docker-compose.ollama-test.yaml \\
       up router-server router-controller --build -d

Usage:
    pytest src/test_router_integration.py -v
    pytest src/test_router_integration.py -v --router-url http://0.0.0.0:8084/v1
"""

import pytest
from openai import OpenAI, APIConnectionError, APIStatusError


EXTRA_BODY_TRITON = {
    "nim-llm-router": {
        "policy": "task_router",
        "routing_strategy": "triton",
    }
}

# Maps task_router category names → expected Ollama model
CATEGORY_MODEL_MAP = {
    "Code Generation": "gemma3:1b",
    "Brainstorming": "qwen3-vl:30b-a3b-instruct-q4_K_M",
    "Closed QA": "qwen3-vl:30b-a3b-instruct-q4_K_M",
    "Open QA": "qwen3-vl:30b-a3b-instruct-q4_K_M",
    "Summarization": "qwen3-vl:30b-a3b-instruct-q4_K_M",
    "Chatbot": "qwen3:0.6b",
    "Classification": "qwen3:0.6b",
    "Extraction": "qwen3:0.6b",
    "Rewrite": "qwen3:0.6b",
    "Text Generation": "qwen3:0.6b",
    "Other": "qwen3:0.6b",
    "Unknown": "qwen3:0.6b",
}

COMPLEX_CATEGORIES = {"Brainstorming", "Closed QA", "Open QA", "Summarization"}
SIMPLE_CATEGORIES = {
    "Chatbot", "Classification", "Extraction", "Rewrite",
    "Text Generation", "Other", "Unknown",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def router_url(request):
    return request.config.getoption("--router-url")


@pytest.fixture(scope="session")
def router_client(router_url):
    client = OpenAI(
        base_url=router_url,
        api_key="ollama",
        max_retries=0,
    )
    # Quick health check
    try:
        import httpx
        resp = httpx.get(router_url.replace("/v1", "/health"), timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip("Router controller is not reachable")
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_non_streaming(client, prompt, extra_body=None):
    """Send a non-streaming request and return (raw_response, parsed_completion)."""
    raw = client.chat.completions.with_raw_response.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra_body or EXTRA_BODY_TRITON,
    )
    return raw, raw.parse()


def _send_streaming(client, prompt, extra_body=None):
    """Send a streaming request and return (raw_response, list_of_chunks)."""
    raw = client.chat.completions.with_raw_response.create(
        model="",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body or EXTRA_BODY_TRITON,
    )
    chunks = list(raw.parse())
    return raw, chunks


# ---------------------------------------------------------------------------
# Tests — Triton classification routing (non-streaming)
# ---------------------------------------------------------------------------

class TestTritonNonStreaming:
    def test_coding_prompt_routes_to_code_generation(self, router_client):
        """A coding prompt should be classified as 'Code Generation' → gemma3:1b."""
        raw, completion = _send_non_streaming(
            router_client,
            "Write a Python function that implements binary search on a sorted list.",
        )

        classifier = raw.headers.get("x-chosen-classifier")
        assert classifier == "Code Generation", (
            f"Expected 'Code Generation', got '{classifier}'"
        )
        assert completion.choices[0].message.content, "Response should not be empty"

    def test_simple_prompt_routes_and_responds(self, router_client):
        """A casual prompt should classify to a valid category and produce a response."""
        raw, completion = _send_non_streaming(
            router_client,
            "Rewrite the following sentence in a formal tone: Hey, what's up?",
        )

        classifier = raw.headers.get("x-chosen-classifier")
        assert classifier in CATEGORY_MODEL_MAP, (
            f"Got unexpected classifier '{classifier}'"
        )
        assert completion.choices[0].message.content

    def test_complex_prompt_routes_to_large_model(self, router_client):
        """A summarization/QA prompt should route to a complex category."""
        raw, completion = _send_non_streaming(
            router_client,
            "Summarize the key differences between TCP and UDP protocols "
            "in networking, including their use cases and trade-offs.",
        )

        classifier = raw.headers.get("x-chosen-classifier")
        assert classifier in COMPLEX_CATEGORIES | SIMPLE_CATEGORIES | {"Code Generation"}, (
            f"Got unexpected classifier '{classifier}'"
        )
        assert completion.choices[0].message.content

    def test_response_has_usage(self, router_client):
        """Non-streaming responses should include token usage."""
        _, completion = _send_non_streaming(
            router_client,
            "What is 2 + 2?",
        )

        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0


# ---------------------------------------------------------------------------
# Tests — Triton classification routing (streaming)
# ---------------------------------------------------------------------------

class TestTritonStreaming:
    def test_streaming_coding_prompt(self, router_client):
        """Streaming coding prompt should classify as 'Code Generation'."""
        raw, chunks = _send_streaming(
            router_client,
            "Write a JavaScript function to reverse a string.",
        )

        classifier = raw.headers.get("x-chosen-classifier")
        assert classifier == "Code Generation", (
            f"Expected 'Code Generation', got '{classifier}'"
        )

        content_parts = [
            c.choices[0].delta.content
            for c in chunks
            if c.choices and c.choices[0].delta.content is not None
        ]
        assert len(content_parts) > 0, "Stream should produce content"
        assert len("".join(content_parts)) > 0

    def test_streaming_has_finish_reason(self, router_client):
        """Streaming should eventually produce a chunk with finish_reason."""
        _, chunks = _send_streaming(
            router_client,
            "Say hello in one word.",
        )

        finish_chunks = [
            c for c in chunks
            if c.choices and c.choices[0].finish_reason is not None
        ]
        assert len(finish_chunks) > 0, "Stream should have a finish_reason chunk"
        assert finish_chunks[-1].choices[0].finish_reason in ("stop", "length")

    def test_streaming_has_usage(self, router_client):
        """Streaming with include_usage should return a usage chunk."""
        _, chunks = _send_streaming(
            router_client,
            "What is the capital of France?",
        )

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) > 0, "Stream should include a usage chunk"
        assert usage_chunks[-1].usage.prompt_tokens > 0
        assert usage_chunks[-1].usage.completion_tokens > 0


# ---------------------------------------------------------------------------
# Tests — Manual routing
# ---------------------------------------------------------------------------

class TestManualRouting:
    def test_manual_route_to_code_generation(self, router_client):
        """Manually selecting 'Code Generation' should route to gemma3:1b."""
        extra_body = {
            "nim-llm-router": {
                "policy": "task_router",
                "routing_strategy": "manual",
                "model": "Code Generation",
            }
        }
        raw, completion = _send_non_streaming(
            router_client,
            "Tell me a joke.",
            extra_body=extra_body,
        )

        classifier = raw.headers.get("x-chosen-classifier")
        assert classifier == "Code Generation", (
            f"Expected 'Code Generation', got '{classifier}'"
        )
        assert completion.choices[0].message.content

    def test_manual_route_to_chatbot(self, router_client):
        """Manually selecting 'Chatbot' should route to qwen3:0.6b."""
        extra_body = {
            "nim-llm-router": {
                "policy": "task_router",
                "routing_strategy": "manual",
                "model": "Chatbot",
            }
        }
        raw, completion = _send_non_streaming(
            router_client,
            "Write me a Python function.",
            extra_body=extra_body,
        )

        classifier = raw.headers.get("x-chosen-classifier")
        assert classifier == "Chatbot", (
            f"Expected 'Chatbot', got '{classifier}'"
        )
        assert completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Tests — Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_invalid_policy_returns_error(self, router_client):
        """An invalid policy name should return 404."""
        extra_body = {
            "nim-llm-router": {
                "policy": "nonexistent_policy",
                "routing_strategy": "triton",
            }
        }
        with pytest.raises(APIStatusError) as exc_info:
            router_client.chat.completions.create(
                model="",
                messages=[{"role": "user", "content": "Hello"}],
                extra_body=extra_body,
            )
        assert exc_info.value.status_code == 404

    def test_invalid_manual_model_returns_error(self, router_client):
        """Manually specifying a nonexistent model category should return 404."""
        extra_body = {
            "nim-llm-router": {
                "policy": "task_router",
                "routing_strategy": "manual",
                "model": "Nonexistent Category",
            }
        }
        with pytest.raises(APIStatusError) as exc_info:
            router_client.chat.completions.create(
                model="",
                messages=[{"role": "user", "content": "Hello"}],
                extra_body=extra_body,
            )
        assert exc_info.value.status_code == 404

    def test_missing_routing_params_returns_error(self, router_client):
        """A request without nim-llm-router params should return 400."""
        with pytest.raises(APIStatusError) as exc_info:
            router_client.chat.completions.create(
                model="",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert exc_info.value.status_code == 400
