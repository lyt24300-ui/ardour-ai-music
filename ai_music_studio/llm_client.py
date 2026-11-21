"""Universal OpenAI-compatible async LLM client with up to 10 reconnects + ReAct fallback."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Awaitable, Callable, Optional

import httpx
from json_repair import repair_json

from .models import ProviderConfig

MAX_RECONNECTS = 10
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.NetworkError,
    OSError,
)


class _RetryableStreamError(RuntimeError):
    """Raised when an HTTP status or stream interruption should trigger a reconnect."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _compute_backoff(retry_num: int, retry_after: Optional[float] = None, cap: float = 12.0) -> float:
    if retry_after is not None and retry_after > 0:
        return min(max(retry_after, 0.5), 30.0)
    return min(cap, 0.8 * (1.45 ** max(0, retry_num - 1)))


async def _sleep_with_cancel(delay: float, cancel_event: Optional[asyncio.Event]) -> bool:
    """Sleep up to delay seconds; return False if cancelled early."""
    if cancel_event is None:
        await asyncio.sleep(delay)
        return True
    try:
        await asyncio.wait_for(cancel_event.wait(), timeout=delay)
        return False
    except asyncio.TimeoutError:
        return True


def _build_headers(provider: ProviderConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    for k, v in provider.custom_headers.items():
        if k and v:
            headers[k] = v
    return headers


def _endpoint(provider: ProviderConfig, suffix: str) -> str:
    base = provider.base_url.rstrip("/")
    return f"{base}/{suffix.lstrip('/')}"


async def list_provider_models(provider: ProviderConfig, max_reconnects: int = MAX_RECONNECTS) -> dict[str, Any]:
    """Fetch available model IDs from any OpenAI-compatible /models endpoint with reconnects."""
    url = _endpoint(provider, "models")
    retries = max(0, int(max_reconnects))
    last_err = ""
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=_build_headers(provider))
                if resp.status_code in _RETRYABLE_STATUS and attempt < retries:
                    raise _RetryableStreamError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
            raw_models = data.get("data", data.get("models", []))
            ids: list[str] = []
            for item in raw_models:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
                elif isinstance(item, str):
                    ids.append(item)
            ids.sort()
            return {"ok": True, "models": ids, "reconnects_used": attempt}
        except _RETRYABLE_EXC + (_RetryableStreamError,) as exc:
            last_err = str(exc)
            if attempt < retries:
                await asyncio.sleep(_compute_backoff(attempt + 1, cap=4.0))
                continue
            break
        except Exception as exc:
            last_err = str(exc)
            break
    return {"ok": False, "error": last_err, "models": []}


async def test_provider_connection(provider: ProviderConfig) -> dict[str, Any]:
    """Verify endpoint auth and reachability via /models or minimal completion."""
    models_res = await list_provider_models(provider, max_reconnects=3)
    if models_res["ok"]:
        return {
            "ok": True,
            "message": f"Подключено! Доступно моделей: {len(models_res['models'])}",
            "models_count": len(models_res["models"]),
            "models": models_res["models"][:80],
        }
    # Fallback minimal chat check for providers that hide /models
    url = _endpoint(provider, "chat/completions")
    payload = {
        "model": provider.model,
        "messages": [{"role": "user", "content": "Reply with OK"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=_build_headers(provider), json=payload)
            resp.raise_for_status()
            return {"ok": True, "message": f"Подключено к модели {provider.model}", "models": [provider.model]}
    except Exception as exc:
        return {"ok": False, "message": f"Ошибка подключения: {exc}", "models": []}


def parse_react_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract ```tool_call {...} ``` blocks for models without native function calling."""
    pattern = re.compile(r"```(?:tool_call|json)\s*(\{.*?\})\s*```", re.DOTALL)
    calls: list[dict[str, Any]] = []
    for match in pattern.finditer(text or ""):
        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = json.loads(repair_json(raw))
            except Exception:
                continue
        if isinstance(parsed, dict) and parsed.get("name"):
            calls.append({
                "id": f"react_{uuid.uuid4().hex[:10]}",
                "type": "function",
                "function": {
                    "name": str(parsed["name"]),
                    "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False),
                },
            })
    return calls


async def _single_stream_attempt(
    provider: ProviderConfig,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    use_native_tools: bool,
    tools: Optional[list[dict[str, Any]]],
    messages: list[dict[str, Any]],
    on_token: Optional[Callable[[str], Awaitable[None]]],
    cancel_event: Optional[asyncio.Event],
) -> dict[str, Any]:
    content_chunks: list[str] = []
    tool_calls_map: dict[int, dict[str, Any]] = {}
    fallback_idx = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=180.0, write=30.0, pool=10.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code == 400 and use_native_tools:
                err_body = (await resp.aread()).decode("utf-8", errors="ignore")
                low = err_body.lower()
                if any(k in low for k in ("tool", "function", "schema", "additionalproperties")):
                    return {"_fallback_non_tool": True}
                raise RuntimeError(f"LLM HTTP 400: {err_body[:600]}")

            if resp.status_code in _RETRYABLE_STATUS:
                err_body = (await resp.aread()).decode("utf-8", errors="ignore")
                ra_hdr = resp.headers.get("retry-after")
                ra_val = float(ra_hdr) if ra_hdr and ra_hdr.replace(".", "", 1).isdigit() else None
                raise _RetryableStreamError(f"LLM HTTP {resp.status_code}: {err_body[:300]}", retry_after=ra_val)

            if resp.status_code >= 400:
                err_body = (await resp.aread()).decode("utf-8", errors="ignore")
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {err_body[:600]}")

            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                raw_body = (await resp.aread()).decode("utf-8", errors="ignore")
                data = json.loads(raw_body)
                if isinstance(data, dict) and data.get("error"):
                    err_obj = data["error"]
                    msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                    raise _RetryableStreamError(f"LLM error payload: {msg[:300]}")
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message", {})
                text = msg.get("content") or ""
                if text and on_token:
                    await on_token(text)
                tcalls = msg.get("tool_calls") or []
                used_react = False
                if not tcalls and tools and text:
                    tcalls = parse_react_tool_calls(text)
                    used_react = bool(tcalls)
                return {
                    "content": text,
                    "tool_calls": tcalls,
                    "used_react_fallback": used_react,
                }

            saw_done = False
            async for line in resp.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err_obj = chunk["error"]
                    msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                    raise _RetryableStreamError(f"LLM stream error: {msg[:300]}")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or choices[0].get("message") or {}

                piece = delta.get("content")
                if piece:
                    content_chunks.append(piece)
                    if on_token:
                        await on_token(piece)

                for tc in delta.get("tool_calls") or []:
                    tc_id = tc.get("id")
                    if "index" in tc and tc["index"] is not None:
                        idx = int(tc["index"])
                        # Guard against proxies reusing index=0 for separate calls with distinct IDs
                        if idx in tool_calls_map and tc_id and tool_calls_map[idx]["id"] != tc_id and tool_calls_map[idx]["function"]["name"]:
                            fallback_idx = max(max(tool_calls_map.keys()) + 1, fallback_idx + 1)
                            idx = fallback_idx
                    else:
                        if tc_id:
                            existing = [k for k, v in tool_calls_map.items() if v["id"] == tc_id]
                            if existing:
                                idx = existing[0]
                            else:
                                fallback_idx = (max(tool_calls_map.keys()) + 1) if tool_calls_map else 0
                                idx = fallback_idx
                        else:
                            idx = fallback_idx

                    entry = tool_calls_map.setdefault(
                        idx,
                        {
                            "id": tc_id or f"call_{uuid.uuid4().hex[:10]}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc_id:
                        entry["id"] = tc_id
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]

            # Treat completely empty SSE stream (without DONE and without content/tools) as interrupted
            if not saw_done and not content_chunks and not tool_calls_map and not (cancel_event and cancel_event.is_set()):
                raise _RetryableStreamError("Пустой поток ответа от LLM (обрыв соединения)")

    full_text = "".join(content_chunks)
    tool_calls = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())]

    used_react = False
    if not tool_calls and tools:
        react_calls = parse_react_tool_calls(full_text)
        if react_calls:
            tool_calls = react_calls
            used_react = True

    return {
        "content": full_text,
        "tool_calls": tool_calls,
        "used_react_fallback": used_react,
    }


async def stream_chat_completion(
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    on_reset: Optional[Callable[[], Awaitable[None]]] = None,
    max_reconnects: int = MAX_RECONNECTS,
    cancel_event: Optional[asyncio.Event] = None,
) -> dict[str, Any]:
    """Stream OpenAI-compatible /chat/completions with up to 10 automatic reconnects."""
    url = _endpoint(provider, "chat/completions")
    headers = _build_headers(provider)

    use_native_tools = bool(tools and provider.supports_tools)
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
        "stream": True,
    }
    if use_native_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    retries = max(0, int(max_reconnects))
    last_error = ""

    for attempt in range(retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            return {"content": "", "tool_calls": [], "used_react_fallback": False}

        emitted_any = False

        async def tracked_on_token(tok: str) -> None:
            nonlocal emitted_any
            emitted_any = True
            if on_token:
                await on_token(tok)

        try:
            res = await _single_stream_attempt(
                provider=provider,
                url=url,
                headers=headers,
                payload=payload,
                use_native_tools=use_native_tools,
                tools=tools,
                messages=messages,
                on_token=tracked_on_token,
                cancel_event=cancel_event,
            )
            if res.get("_fallback_non_tool"):
                return await _non_tool_fallback(
                    provider, messages, tools, on_token, on_status, on_reset, max_reconnects, cancel_event
                )
            res["reconnects_used"] = attempt
            return res
        except _RETRYABLE_EXC + (_RetryableStreamError,) as exc:
            last_error = str(exc)
            if attempt >= retries or (cancel_event is not None and cancel_event.is_set()):
                break
            ra = exc.retry_after if isinstance(exc, _RetryableStreamError) else None
            delay = _compute_backoff(attempt + 1, retry_after=ra)
            if emitted_any and on_reset:
                await on_reset()
            if on_status:
                await on_status(
                    f"LLM API: реконнект {attempt + 1}/{retries} через {delay:.1f}с ({last_error[:90]})"
                )
            ok_sleep = await _sleep_with_cancel(delay, cancel_event)
            if not ok_sleep:
                return {"content": "", "tool_calls": [], "used_react_fallback": False}

    raise RuntimeError(f"LLM недоступен после {retries} попыток переподключения: {last_error}")


async def _non_tool_fallback(
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    on_token: Optional[Callable[[str], Awaitable[None]]],
    on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    on_reset: Optional[Callable[[], Awaitable[None]]] = None,
    max_reconnects: int = MAX_RECONNECTS,
    cancel_event: Optional[asyncio.Event] = None,
) -> dict[str, Any]:
    """Re-run request using textual ReAct ```tool_call``` blocks."""
    tool_specs = []
    for t in tools or []:
        fn = t.get("function", {})
        tool_specs.append(f"- `{fn.get('name')}`: {fn.get('description', '')} | params: {json.dumps(fn.get('parameters', {}), ensure_ascii=False)}")
    instruction = (
        "\n\n[TOOL CALL PROTOCOL]\n"
        "You can execute tools by outputting one or more fenced blocks:\n"
        "```tool_call\n{\"name\": \"tool_name\", \"arguments\": {...}}\n```\n"
        "Available tools:\n" + "\n".join(tool_specs)
    )
    patched = [dict(m) for m in messages]
    if patched and patched[0]["role"] == "system":
        patched[0]["content"] = str(patched[0]["content"]) + instruction
    else:
        patched.insert(0, {"role": "system", "content": instruction})

    fake_prov = provider.model_copy(update={"supports_tools": False})
    return await stream_chat_completion(
        fake_prov,
        patched,
        tools=tools,
        on_token=on_token,
        on_status=on_status,
        on_reset=on_reset,
        max_reconnects=max_reconnects,
        cancel_event=cancel_event,
    )
