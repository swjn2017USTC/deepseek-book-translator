from __future__ import annotations

"""DeepSeek chat client used by translation and structure resolution.

Extracted verbatim-behavior from ``translate.py`` (P03 §5.1/§5.2).  This module
is the single home of wire/auth/retry semantics; ``translate.py`` imports FROM
here and never the reverse.  It depends only on the standard library so the
import DAG stays acyclic.
"""

import json
import os
import re
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Type, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

try:  # pydantic is an optional runtime dependency for structured completion.
    from pydantic import BaseModel, TypeAdapter, ValidationError
except ImportError:  # pragma: no cover - exercised only in minimal installs
    BaseModel = None  # type: ignore[assignment,misc]
    ValidationError = None  # type: ignore[assignment,misc]


ModelT = TypeVar("ModelT")


def aggregate_usage(usages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum billable usage across every successful provider completion.

    Parse/validation retries are separate billable completions.  Returning only
    the final attempt materially under-reports cost, so callers use this helper
    whenever one logical operation can make more than one completion request.
    Missing cache-detail fields are treated conservatively as cache misses.
    """
    rows = [dict(usage or {}) for usage in usages]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_tokens = sum(
        int(row.get("total_tokens") or (int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)))
        for row in rows
    )
    cache_hit = sum(int(row.get("prompt_cache_hit_tokens") or 0) for row in rows)
    cache_miss = 0
    reasoning_tokens = 0
    for row in rows:
        if "prompt_cache_miss_tokens" in row:
            cache_miss += int(row.get("prompt_cache_miss_tokens") or 0)
        else:
            cache_miss += max(0, int(row.get("prompt_tokens") or 0) - int(row.get("prompt_cache_hit_tokens") or 0))
        details = row.get("completion_tokens_details") or {}
        reasoning_tokens += int(details.get("reasoning_tokens") or 0)
    prompt_accounted = cache_hit + cache_miss
    cache_hit_ratio = (cache_hit / prompt_accounted) if prompt_accounted else None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
        "prompt_tokens_details": {"cached_tokens": cache_hit},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "visible_completion_tokens": max(0, completion_tokens - reasoning_tokens),
        "request_count": len(rows),
        "cache_hit_ratio": round(cache_hit_ratio, 6) if cache_hit_ratio is not None else None,
    }


class _ExactHeaderName(str):
    """Keep a configured header spelling through urllib's .title()."""

    def title(self) -> str:
        return str(self)


class ChatClient:
    """Structural protocol: a chat model with a text completion entry point."""

    model: str

    def complete(self, system: str, user: str) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class OpenAICompatibleClient:
    def __init__(self, provider: Dict[str, Any]):
        key_env = str(provider.get("api_key_env") or "DEEPSEEK_API_KEY")
        url_env = str(provider.get("api_url_env") or "DEEPSEEK_API_URL")
        model_env = str(provider.get("model_env") or "DEEPSEEK_MODEL")
        thinking_env = str(provider.get("thinking_env") or "DEEPSEEK_THINKING_MODE")
        self.api_key = os.environ.get(key_env, "")
        self.api_url = os.environ.get(url_env) or str(provider.get("api_url") or "https://api.deepseek.com/chat/completions")
        self.model = os.environ.get(model_env) or str(provider.get("model") or "deepseek-flash")
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {key_env}")
        if self.api_key != self.api_key.strip():
            raise RuntimeError(f"API key environment variable contains surrounding whitespace: {key_env}")
        if self.api_key[:1] in {"'", '"'} or self.api_key[-1:] in {"'", '"'}:
            raise RuntimeError(f"API key environment variable must not include quote characters: {key_env}")
        if any(ord(character) < 32 or ord(character) == 127 for character in self.api_key):
            raise RuntimeError(f"API key environment variable contains a control character: {key_env}")
        if not self.api_url:
            raise RuntimeError(f"Missing API URL: set {url_env} or provider.api_url")
        if not self.model:
            raise RuntimeError(f"Missing model: set {model_env} or provider.model")
        self.auth_header = str(provider.get("auth_header") or "Authorization")
        # An explicitly empty scheme remains supported for custom test clients.
        auth_scheme = provider.get("auth_scheme")
        self.auth_scheme = "Bearer" if auth_scheme is None else str(auth_scheme)
        self.timeout = int(provider.get("timeout_seconds", 600))
        self.max_tokens = int(provider.get("max_tokens", 8192))
        self.temperature = float(provider.get("temperature", 0.2))
        self.retries = int(provider.get("retries", 5))
        hostname = (urlparse(self.api_url).hostname or "").lower()
        self.is_deepseek = hostname == "api.deepseek.com" or hostname.endswith(".deepseek.com")
        configured_thinking = os.environ.get(thinking_env)
        if configured_thinking is None:
            configured_thinking = provider.get("thinking_mode")
        if configured_thinking is None and self.is_deepseek:
            configured_thinking = "disabled"
        self.thinking_mode = (
            str(configured_thinking).strip().lower()
            if configured_thinking is not None
            else None
        )
        if self.thinking_mode is not None and self.thinking_mode not in {"disabled", "low", "high", "max"}:
            raise ValueError("provider.thinking_mode must be one of: disabled, low, high, max")
        native_json_setting = provider.get("native_json_mode")
        self.native_json_mode = self.is_deepseek if native_json_setting is None else bool(native_json_setting)
        # Monotonic wire-request counter: every actual HTTP POST attempt this
        # client instance makes (including complete_json re-requests). Lets the
        # resolver meter the true request budget (reviewer MAJOR finding: step
        # counting alone undercounts client-internal retries).
        self.wire_attempts = 0

    def complete(self, system: str, user: str) -> Tuple[str, Dict[str, Any]]:
        return self._complete(system, user, json_mode=False)

    def complete_json_text(self, system: str, user: str) -> Tuple[str, Dict[str, Any]]:
        """Return raw JSON text while enabling the provider's native JSON mode."""
        return self._complete(system, user, json_mode=True)

    def complete_multimodal_json_text(
        self,
        system: str,
        content: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return JSON text for an OpenAI-compatible multimodal content list.

        DeepSeek Flash accepts standard text and image_url blocks. Reuse the
        same auth, retry, usage, thinking and native-JSON transport as text.
        """
        if not isinstance(content, list) or not content:
            raise ValueError("Multimodal content must be a non-empty block list")
        return self._complete(system, content, json_mode=True)

    def _complete(self, system: str, user: Any, *, json_mode: bool) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking_mode is not None:
            payload["thinking"] = {"type": "disabled" if self.thinking_mode == "disabled" else "enabled"}
            if self.thinking_mode != "disabled":
                payload["reasoning_effort"] = self.thinking_mode
        if json_mode and self.native_json_mode:
            payload["response_format"] = {"type": "json_object"}
        auth = f"{self.auth_scheme} {self.api_key}".strip() if self.auth_scheme else self.api_key
        headers = {"Content-Type": "application/json"}
        for attempt in range(1, self.retries + 1):
            self.wire_attempts += 1
            request = Request(self.api_url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
            # urllib's HTTP handler applies .title() before writing headers.
            request.headers[_ExactHeaderName(self.auth_header)] = auth
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = data.get("choices", [{}])[0]
                content = str(choice.get("message", {}).get("content") or "")
                usage = dict(data.get("usage") or {})
                # DeepSeek JSON mode can occasionally return empty content; a
                # length-truncated JSON response is also unusable. Return an
                # empty payload WITH usage so the structured caller can retry
                # while still accounting for the billable completion.
                if json_mode and (not content or choice.get("finish_reason") == "length"):
                    return "", usage
                if not content:
                    raise RuntimeError("Provider returned empty content")
                if choice.get("finish_reason") == "length":
                    raise RuntimeError("Provider truncated the batch")
                return content, usage
            except HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == self.retries:
                    detail = (
                        "; authentication rejected: verify the key is set in this process "
                        "and the configured auth_header/auth_scheme match the provider"
                        if exc.code in {401, 403}
                        else ""
                    )
                    raise RuntimeError(f"Translation HTTP error {exc.code}{detail}") from exc
                retry_after = float(exc.headers.get("Retry-After") or 0)
                time.sleep(max(retry_after, min(60.0, 1.5 * (2 ** (attempt - 1)))))
            except (URLError, TimeoutError, OSError) as exc:
                # Network-level failures. URLError is raised by urlopen for
                # DNS/conn errors; some paths surface bare socket.timeout /
                # ConnectionResetError (OSError subclasses) unwrapped — cover
                # them all with the same bounded retry/backoff.
                if attempt == self.retries:
                    raise RuntimeError("Translation provider unavailable after retries") from exc
                time.sleep(min(60.0, 1.5 * (2 ** (attempt - 1))))
        raise RuntimeError("Unreachable retry state")

    def complete_json(
        self,
        system: str,
        user: str,
        model_cls: Type[ModelT],
        *,
        decode_retries: int = 3,
    ) -> Tuple[ModelT, Dict[str, Any]]:
        """Request a structured response validated into ``model_cls``.

        The provider is asked for a single JSON object (no code fences).  Fence
        stripping is applied defensively, then ``model_cls.model_validate``.
        ``json.JSONDecodeError``/``pydantic.ValidationError`` trigger a bounded
        re-request (``decode_retries`` attempts total); persistent failures
        raise ``RuntimeError``.
        """
        if BaseModel is None or ValidationError is None:  # pragma: no cover
            raise RuntimeError("pydantic is required for structured completion")
        content, usage = self.complete_json_text(system, user)
        usages = [dict(usage)]
        last_error: Optional[BaseException] = None
        for attempt in range(1, decode_retries + 1):
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned, flags=re.I)
                cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.I)
            try:
                data = json.loads(cleaned)
                return TypeAdapter(model_cls).validate_python(data), aggregate_usage(usages)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt == decode_retries:
                    break
                try:
                    content, usage = self.complete_json_text(system, user)
                    usages.append(dict(usage))
                except RuntimeError as provider_exc:
                    last_error = provider_exc
                    break
        raise RuntimeError("Structured response failed validation after retries") from last_error


def resolve_model(config: Dict[str, Any], role: str) -> str:
    """Model for ``role`` from ``config["llm_roles"]``, falling back to the
    provider default model (plan §5.1)."""
    provider = config.get("provider") or {}
    return (config.get("llm_roles") or {}).get(role) or str(provider.get("model") or "")
