# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for LLM Router targeting a local Ollama instance.

These tests validate OpenAI-compatible chat completions in both streaming
and non-streaming modes against multiple Ollama models. They can run
directly against Ollama (bypassing the router) or through the router
controller when it is available.

Usage:
    # Run all tests against Ollama (default):
    pytest src/test_router_ollama.py -v

    # Run against the router controller instead:
    pytest src/test_router_ollama.py -v --base-url http://0.0.0.0:8084/v1

    # Run only a specific model:
    pytest src/test_router_ollama.py -v -k "qwen3_0_6b"

    # Run only streaming tests:
    pytest src/test_router_ollama.py -v -k "streaming"
"""

import pytest
from openai import OpenAI, APIConnectionError, APIStatusError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/v1"

MODELS = [
    "qwen3:0.6b",
    "gemma3:1b",
    "qwen3-vl:30b-a3b-instruct-q4_K_M",
]

PROMPTS = [
    {"role": "user", "content": "What is 2 + 2? Answer with just the number."},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url") or OLLAMA_BASE_URL


@pytest.fixture(scope="session")
def client(base_url):
    return OpenAI(
        base_url=base_url,
        api_key="ollama",  # Ollama ignores the key but the client requires one
    )


@pytest.fixture(scope="session")
def available_models(client):
    """Fetch the model list once and return the set of IDs."""
    try:
        return {m.id for m in client.models.list()}
    except APIConnectionError:
        pytest.skip("LLM endpoint is not reachable")


# ---------------------------------------------------------------------------
# Tests — non-streaming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS, ids=[m.replace(":", "_") for m in MODELS])
def test_non_streaming_completion(client, available_models, model):
    if model not in available_models:
        pytest.skip(f"Model {model!r} not available on this endpoint")

    response = client.chat.completions.create(
        model=model,
        messages=PROMPTS,
    )

    # Structure checks
    assert response.id, "Response should have an id"
    assert len(response.choices) > 0, "Response should contain at least one choice"

    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content, "Assistant message should not be empty"
    assert choice.finish_reason in ("stop", "length")

    # Usage
    assert response.usage is not None, "Non-streaming response should include usage"
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


@pytest.mark.parametrize("model", MODELS, ids=[m.replace(":", "_") for m in MODELS])
def test_non_streaming_with_system_message(client, available_models, model):
    if model not in available_models:
        pytest.skip(f"Model {model!r} not available on this endpoint")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be brief."},
        {"role": "user", "content": "Say hello."},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    assert len(response.choices) > 0
    assert response.choices[0].message.content


# ---------------------------------------------------------------------------
# Tests — streaming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS, ids=[m.replace(":", "_") for m in MODELS])
def test_streaming_completion(client, available_models, model):
    if model not in available_models:
        pytest.skip(f"Model {model!r} not available on this endpoint")

    stream = client.chat.completions.create(
        model=model,
        messages=PROMPTS,
        stream=True,
    )

    chunks = list(stream)

    assert len(chunks) > 0, "Stream should produce at least one chunk"

    # First chunk should have a role
    first_delta = chunks[0].choices[0].delta
    assert first_delta.role == "assistant" or first_delta.content is not None

    # Collect all content pieces
    content_parts = [
        c.choices[0].delta.content
        for c in chunks
        if c.choices and c.choices[0].delta.content is not None
    ]
    full_content = "".join(content_parts)
    assert len(full_content) > 0, "Streamed content should not be empty"

    # Last chunk with choices should signal stop
    chunks_with_choices = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert len(chunks_with_choices) > 0, "Stream should contain a chunk with finish_reason"
    assert chunks_with_choices[-1].choices[0].finish_reason in ("stop", "length")


@pytest.mark.parametrize("model", MODELS, ids=[m.replace(":", "_") for m in MODELS])
def test_streaming_with_usage(client, available_models, model):
    """Verify stream_options=include_usage returns a final usage chunk."""
    if model not in available_models:
        pytest.skip(f"Model {model!r} not available on this endpoint")

    stream = client.chat.completions.create(
        model=model,
        messages=PROMPTS,
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = list(stream)
    usage_chunks = [c for c in chunks if c.usage is not None]

    assert len(usage_chunks) > 0, "Stream with include_usage should return a usage chunk"
    usage = usage_chunks[-1].usage
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------

def test_invalid_model(client):
    with pytest.raises(APIStatusError) as exc_info:
        client.chat.completions.create(
            model="nonexistent-model-xyz",
            messages=PROMPTS,
        )
    assert exc_info.value.status_code in (400, 404)


def test_empty_messages(client, available_models):
    model = next((m for m in MODELS if m in available_models), None)
    if model is None:
        pytest.skip("No test models available")

    with pytest.raises(APIStatusError) as exc_info:
        client.chat.completions.create(
            model=model,
            messages=[],
        )
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Tests — parameter variations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS, ids=[m.replace(":", "_") for m in MODELS])
def test_max_tokens_respected(client, available_models, model):
    if model not in available_models:
        pytest.skip(f"Model {model!r} not available on this endpoint")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Write a long essay about the ocean."}],
        max_tokens=5,
    )

    assert response.usage.completion_tokens <= 10  # small tolerance
    assert response.choices[0].finish_reason == "length"


@pytest.mark.parametrize("model", MODELS, ids=[m.replace(":", "_") for m in MODELS])
def test_temperature_zero(client, available_models, model):
    """Temperature 0 should produce deterministic output."""
    if model not in available_models:
        pytest.skip(f"Model {model!r} not available on this endpoint")

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": "What is 1+1? Reply with just the number."}],
        temperature=0,
        max_tokens=3,
    )

    r1 = client.chat.completions.create(**kwargs)
    r2 = client.chat.completions.create(**kwargs)
    assert r1.choices[0].message.content.strip() == r2.choices[0].message.content.strip()
