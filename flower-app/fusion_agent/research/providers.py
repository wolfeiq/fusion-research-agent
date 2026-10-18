"""Server-side Nebius inference and Tavily retrieval; no training dependencies."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ProviderError(RuntimeError):
    """A safe error message, never a provider body or credential."""

    def __init__(
        self, message, *, finish_reason=None, usage=None,
        error_type=None, http_status=None,
    ):
        super().__init__(message)
        self.error_type = (
            error_type if isinstance(error_type, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", error_type)
            else None
        )
        self.http_status = (
            http_status if type(http_status) is int and 100 <= http_status <= 599
            else None
        )
        self.finish_reason = (
            finish_reason if finish_reason in {
                "stop", "length", "tool_calls", "content_filter", "function_call",
            } else None
        )
        self.usage = {
            key: value for key, value in (usage or {}).items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and type(value) is int and value >= 0
        }


def _output_token_limit(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,5}", value.strip()):
        value = int(value)
    if type(value) is not int or not 256 <= value <= 16000:
        raise ValueError("NEBIUS_MAX_OUTPUT_TOKENS must be an integer from 256 to 16000")
    return value


def _thinking_control(value):
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError("NEBIUS_ENABLE_THINKING must be true, false or unset")


def _token_usage(response):
    usage = getattr(response, "usage", None)
    return {
        key: value
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if type(value := getattr(usage, key, None)) is int and value >= 0
    }


def public_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 3000:
        raise ValueError("Expected a public web URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Expected an HTTP(S) URL without credentials")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local URLs cannot be retrieved")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or re.fullmatch(r"[0-9.]+", host):
            raise ValueError("Expected a public hostname") from None
    else:
        if not address.is_global:
            raise ValueError("Private addresses cannot be retrieved")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Only standard web ports are accepted")
    return value


def _endpoint(value: str) -> str:
    public_url(value)
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.query or parsed.fragment:
        raise ValueError("Provider endpoint must use HTTPS without query or fragment")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    nebius_key: str = field(default="", repr=False)
    tavily_key: str = field(default="", repr=False)
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    tavily_url: str = "https://api.tavily.com"
    model: str = "MiniMaxAI/MiniMax-M3"
    max_output_tokens: int = 5000
    enable_thinking: bool | None = None

    @classmethod
    def from_env(cls):
        return cls(
            nebius_key=os.environ.get("NEBIUS_API_KEY", "").strip(),
            tavily_key=os.environ.get("TAVILY_API_KEY", "").strip(),
            base_url=os.environ.get("NEBIUS_BASE_URL", cls.base_url).strip(),
            tavily_url=os.environ.get("TAVILY_BASE_URL", cls.tavily_url).strip(),
            model=os.environ.get("NEBIUS_MODEL", cls.model).strip(),
            max_output_tokens=_output_token_limit(
                os.environ.get("NEBIUS_MAX_OUTPUT_TOKENS", cls.max_output_tokens)
            ),
            enable_thinking=_thinking_control(os.environ.get("NEBIUS_ENABLE_THINKING")),
        )

    def validate(self):
        _endpoint(self.base_url)
        _endpoint(self.tavily_url)
        _output_token_limit(self.max_output_tokens)
        _thinking_control(self.enable_thinking)
        if not self.nebius_key or not self.tavily_key:
            raise ValueError(
                "Set NEBIUS_API_KEY and TAVILY_API_KEY on the server first"
            )
        if not self.model or len(self.model) > 200:
            raise ValueError("Configure a tool-capable NEBIUS_MODEL")

    def public(self):
        # Never return query parameters, credentials, or unvalidated endpoint text.
        try:
            base_url = _endpoint(self.base_url)
        except ValueError:
            base_url = "invalid endpoint"
        try:
            max_output_tokens = _output_token_limit(self.max_output_tokens)
        except ValueError:
            max_output_tokens = "invalid token limit"
        try:
            enable_thinking = _thinking_control(self.enable_thinking)
        except ValueError:
            enable_thinking = "invalid thinking setting"
        return {
            "nebius": bool(self.nebius_key),
            "tavily": bool(self.tavily_key),
            "model": self.model[:200],
            "base_url": base_url,
            "max_output_tokens": max_output_tokens,
            "enable_thinking": enable_thinking,
        }


class NebiusModel:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.model = settings.model
        self.max_output_tokens = _output_token_limit(settings.max_output_tokens)
        self.enable_thinking = _thinking_control(settings.enable_thinking)
        self.client = OpenAI(
            api_key=settings.nebius_key,
            base_url=_endpoint(settings.base_url),
            timeout=60.0,
            max_retries=0,
        )

    def complete(self, messages, tools, tool_choice="auto"):
        # This is a model-specific extension, enabled only by explicit config.
        # Unset leaves every provider/model's own default behavior untouched.
        extra = {}
        enable_thinking = getattr(self, "enable_thinking", None)
        if enable_thinking is not None:
            extra["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=getattr(self, "max_output_tokens", 5000),
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary must not leak keys
            from openai import APIStatusError

            status = getattr(exc, "status_code", None)
            valid_status = type(status) is int and 100 <= status <= 599
            suffix = f" (HTTP {status})" if valid_status else ""
            raise ProviderError(
                "Nebius request failed"
                + suffix
                + "; check endpoint, model, key and quota",
                error_type=type(exc).__name__,
                http_status=status if isinstance(exc, APIStatusError) else None,
            ) from None

        usage = _token_usage(response)
        choices = getattr(response, "choices", None)
        if not choices:
            raise ProviderError("Nebius returned no completion choices", usage=usage)
        choice = choices[0]
        # A truncated tool call must never reach JSON parsing or tool execution.
        # Missing finish_reason remains compatible with provider test doubles.
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise ProviderError(
                "Nebius response was truncated at the output token limit; "
                "no partial content or tool calls were accepted",
                finish_reason=finish_reason, usage=usage,
            )
        if finish_reason == "content_filter":
            raise ProviderError(
                "Nebius completion was blocked by content filtering",
                finish_reason=finish_reason, usage=usage,
            )
        msg = getattr(choice, "message", None)
        if msg is None:
            raise ProviderError(
                "Nebius returned no completion message",
                finish_reason=finish_reason, usage=usage,
            )
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls and not (isinstance(content, str) and content.strip()):
            raise ProviderError(
                "Nebius returned an empty completion",
                finish_reason=finish_reason, usage=usage,
            )

        try:
            # Do not retain private reasoning fields from provider responses.
            message = {"role": "assistant", "content": content or ""}
            if tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            return message, usage
        except Exception:  # noqa: BLE001 - never expose malformed provider data
            raise ProviderError(
                "Nebius returned an invalid completion",
                finish_reason=finish_reason, usage=usage,
            ) from None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class Tavily:
    def __init__(self, settings: Settings):
        self.base_url = _endpoint(settings.tavily_url)
        self.key = settings.tavily_key

    def _post(self, path, payload):
        req = Request(
            self.base_url + path,
            method="POST",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
        )
        try:
            with build_opener(_NoRedirect()).open(req, timeout=45) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderError("Tavily response exceeds the retrieval limit")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Invalid payload")  # noqa: TRY004
            return result
        except HTTPError as exc:
            raise ProviderError(
                f"Tavily request failed (HTTP {exc.code}); "
                "check key, endpoint and quota"
            ) from None
        except (URLError, TimeoutError, ValueError, OSError):
            raise ProviderError(
                "Tavily request failed or returned invalid JSON"
            ) from None

    def search(self, query):
        return self._post(
            "/search",
            {
                "query": query,
                "search_depth": "advanced",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
                "include_usage": True,
            },
        )

    def extract(self, url):
        return self._post(
            "/extract",
            {
                "urls": [public_url(url)],
                "extract_depth": "advanced",
                "format": "text",
                "include_usage": True,
            },
        )
