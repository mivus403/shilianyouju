from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


class ModelAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelResponse:
    text: str
    elapsed_seconds: float
    usage: dict[str, Any] | None
    request_id: str | None
    response_model: str | None


def _safe_error(text: str, api_key: str) -> str:
    cleaned = text.replace(api_key, "[密钥已隐藏]") if api_key else text
    cleaned = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[密钥已隐藏]", cleaned)
    return cleaned[-1200:]


def chat_completion_stream(
    *,
    api_key: str,
    base_url: str,
    model: str,
    content: list[dict[str, Any]],
    timeout_seconds: float,
    reasoning_effort: str = "none",
) -> ModelResponse:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "modalities": ["text"],
        "reasoning_effort": reasoning_effort,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started = time.perf_counter()
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    request_id: str | None = None
    response_model: str | None = None
    timeout = httpx.Timeout(timeout_seconds, connect=min(20.0, timeout_seconds))
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise ModelAPIError(f"接口返回 HTTP {response.status_code}：{_safe_error(body, api_key)}")
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ModelAPIError("接口流中包含无法解析的 JSON 事件") from exc
                    request_id = event.get("id") or request_id
                    response_model = event.get("model") or response_model
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        text_parts.append(piece)
                    elif isinstance(piece, list):
                        for item in piece:
                            if isinstance(item, dict) and isinstance(item.get("text"), str):
                                text_parts.append(item["text"])
    except ModelAPIError:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise ModelAPIError(f"接口请求失败：{_safe_error(str(exc), api_key)}") from exc
    text = "".join(text_parts).strip()
    if not text:
        raise ModelAPIError("接口完成但没有返回文本内容")
    return ModelResponse(
        text=text,
        elapsed_seconds=time.perf_counter() - started,
        usage=usage,
        request_id=request_id,
        response_model=response_model,
    )
