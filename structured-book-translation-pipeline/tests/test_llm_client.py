"""P03 llm_client extraction regression + complete_json + resolve_model.

The three auth/header tests keep their original assertions from
``tests/test_pipeline.py``; only the monkeypatch/import target moved to
``book_pipeline.llm_client`` (plan §5.2).  No live network is ever touched:
``urlopen`` is monkeypatched or env vars are unset.
"""
import json
import socket

import pytest
from pydantic import BaseModel, Field

import book_pipeline.llm_client as llm_client
import book_pipeline.translate as translate
from book_pipeline.llm_client import (
    OpenAICompatibleClient,
    _ExactHeaderName,
    resolve_model,
)


class _Model(BaseModel):
    kind: str
    level: int = Field(ge=1, le=6)


class _FakeHTTPResponse:
    def __init__(self, content):
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": self._content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9},
        }).encode("utf-8")


def _client(monkeypatch, **provider):
    monkeypatch.setenv("TEST_LLM_KEY", "dummy-not-a-secret")
    base = {
        "api_key_env": "TEST_LLM_KEY",
        "api_url": "https://example.invalid/v1/chat/completions",
        "model": "fixture-model",
        "retries": 1,
    }
    base.update(provider)
    return OpenAICompatibleClient(base)


# ------------------------------------------------------------------ extraction
def test_configured_auth_header_preserves_exact_case(monkeypatch):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "dummy-not-a-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = request.header_items()
        return _FakeHTTPResponse("[]")

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient({
        "api_key_env": "TEST_DEEPSEEK_KEY",
        "api_url": "https://example.invalid/v1/chat/completions",
        "model": "fixture-model",
        "auth_header": "x-custom-auth",
        "auth_scheme": "",
    })
    client.complete("system", "user")

    header_names = [name for name, _ in captured["headers"]]
    header_values = dict(captured["headers"])
    assert "x-custom-auth" in header_names
    assert "X-custom-auth" not in header_names
    assert "Authorization" not in header_names
    assert header_values["x-custom-auth"] == "dummy-not-a-secret"


def test_default_authorization_scheme_is_bearer(monkeypatch):
    monkeypatch.setenv("TEST_DEFAULT_KEY", "dummy-not-a-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        return _FakeHTTPResponse("[]")

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient({
        "api_key_env": "TEST_DEFAULT_KEY",
        "api_url": "https://example.invalid/v1/chat/completions",
        "model": "fixture-model",
    })
    client.complete("system", "user")

    assert captured["headers"]["Authorization"] == "Bearer dummy-not-a-secret"


def test_deepseek_official_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-not-a-secret")
    client = OpenAICompatibleClient({})
    assert client.api_url == "https://api.deepseek.com/chat/completions"
    assert client.model == "deepseek-v4-flash"
    assert client.auth_header == "Authorization"
    assert client.auth_scheme == "Bearer"


@pytest.mark.parametrize("bad_key", [" key", "key\n", "'key'", '"key"'])
def test_provider_rejects_malformed_environment_key(monkeypatch, bad_key):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", bad_key)
    with pytest.raises(RuntimeError, match="API key environment variable"):
        OpenAICompatibleClient({
            "api_key_env": "TEST_DEEPSEEK_KEY",
            "api_url": "https://example.invalid/v1/chat/completions",
            "model": "fixture-model",
            "auth_header": "x-custom-auth",
            "auth_scheme": "",
        })


def test_exact_header_name_travels_with_llm_client():
    # A custom lowercase gateway field must survive urllib's .title() rewrite.
    assert _ExactHeaderName("x-custom-auth").title() == "x-custom-auth"
    assert isinstance(llm_client._ExactHeaderName("x"), _ExactHeaderName)


def test_translate_reexports_client_names():
    assert translate.ChatClient is llm_client.ChatClient
    assert translate.OpenAICompatibleClient is llm_client.OpenAICompatibleClient
    assert translate._ExactHeaderName is llm_client._ExactHeaderName
    # Import direction: translate imports FROM llm_client, never the reverse.
    assert not hasattr(llm_client, "translate_book")


# --------------------------------------------------------------- complete_json
def test_complete_json_valid_response_returns_model_instance(monkeypatch):
    client = _client(monkeypatch)

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(json.dumps({"kind": "chapter", "level": 2}))

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    model, usage = client.complete_json("system", "user", _Model)
    assert isinstance(model, _Model)
    assert model.kind == "chapter" and model.level == 2
    assert usage["prompt_tokens"] == 7


def test_complete_json_accepts_typing_generic_list(monkeypatch):
    """Pass B calls complete_json(..., List[StructureDecision]); typing generics
    have no .model_validate — TypeAdapter must be used (live-run regression)."""
    from typing import List

    client = _client(monkeypatch)
    payload = json.dumps([{"kind": "chapter", "level": 2},
                          {"kind": "section", "level": 3}])

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(payload)

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    models, usage = client.complete_json("system", "user", List[_Model])
    assert isinstance(models, list) and len(models) == 2
    assert all(isinstance(m, _Model) for m in models)
    assert models[0].level == 2 and models[1].kind == "section"


def test_complete_retries_bare_socket_timeout_then_runtime_error(monkeypatch):
    """urlopen can raise bare socket.timeout (OSError subclass) unwrapped;
    it must retry with backoff and fail closed after retries (live-run
    regression: previously escaped the (URLError, TimeoutError) handler)."""
    import time

    client = _client(monkeypatch, retries=2)
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        raise socket.timeout("The read operation timed out")

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="provider unavailable after retries"):
        client.complete("system", "user")
    assert len(calls) == 2  # retries=2 attempts


def test_complete_retries_transient_404_then_raises(monkeypatch):
    """A 404 indicates an invalid route or model and should fail fast."""
    import time
    from urllib.error import HTTPError
    from io import BytesIO

    client = _client(monkeypatch, retries=3)
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        raise HTTPError("https://example.invalid/v1/chat/completions", 404,
                        "Not Found", {}, BytesIO(b""))

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Translation HTTP error 404"):
        client.complete("system", "user")
    assert len(calls) == 1


def test_complete_non_retryable_status_fails_fast(monkeypatch):
    """A hard client error (e.g. 400) must not consume the bounded retry budget."""
    from urllib.error import HTTPError
    from io import BytesIO

    client = _client(monkeypatch, retries=4)
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(timeout)
        raise HTTPError("https://example.invalid/v1/chat/completions", 400,
                        "Bad Request", {}, BytesIO(b""))

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Translation HTTP error 400"):
        client.complete("system", "user")
    assert len(calls) == 1


def test_complete_json_malformed_json_retries_then_runtime_error(monkeypatch):
    client = _client(monkeypatch)
    responses = [_FakeHTTPResponse("{not json"), _FakeHTTPResponse("{still bad")]

    def fake_urlopen(request, timeout):
        return responses.pop(0)

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Structured response failed validation after retries"):
        client.complete_json("system", "user", _Model, decode_retries=2)


def test_complete_json_validation_error_recovers_on_second_attempt(monkeypatch):
    client = _client(monkeypatch)
    responses = [
        _FakeHTTPResponse(json.dumps({"kind": "chapter", "level": 0})),  # level out of range
        _FakeHTTPResponse(json.dumps({"kind": "chapter", "level": 3})),
    ]

    def fake_urlopen(request, timeout):
        return responses.pop(0)

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    model, _ = client.complete_json("system", "user", _Model, decode_retries=2)
    assert model.level == 3


def test_complete_json_strips_code_fences(monkeypatch):
    client = _client(monkeypatch)
    fenced = "```json\n" + json.dumps({"kind": "body", "level": 1}) + "\n```"

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse(fenced)

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    model, _ = client.complete_json("system", "user", _Model)
    assert model.kind == "body"


# ---------------------------------------------------------------- resolve_model
def test_resolve_model_role_present_wins():
    config = {
        "provider": {"model": "deepseek-v4-flash"},
        "llm_roles": {
            "structure_global": "deepseek-v4-flash",
            "structure_local": "deepseek-v4-flash",
        },
    }
    assert resolve_model(config, "structure_global") == "deepseek-v4-flash"
    assert resolve_model(config, "structure_local") == "deepseek-v4-flash"


def test_resolve_model_falls_back_to_provider_model():
    config = {"provider": {"model": "deepseek-v4-flash"}, "llm_roles": {}}
    assert resolve_model(config, "structure_review") == "deepseek-v4-flash"
    assert resolve_model({"provider": {"model": "m"}}, "structure_local") == "m"
    assert resolve_model({}, "structure_local") == ""


def test_complete_wire_attempts_count_retries(monkeypatch):
    """Every actual HTTP POST increments client.wire_attempts (reviewer M1:
    budget must meter wire requests, not logical steps)."""
    import time

    client = _client(monkeypatch, retries=3)
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise socket.timeout("stall")
        return _FakeHTTPResponse("ok")

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    content, _ = client.complete("system", "user")
    assert content == "ok"
    assert client.wire_attempts == 3
