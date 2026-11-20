"""Universal OpenAI-compatible async LLM client with native tools + ReAct fallback."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Awaitable, Callable, Optional

import httpx
from json_repair import repair_json

from .models import ProviderConfig


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


async def list_provider_models(provider: ProviderConfig) -> dict[str, Any]:
    """Fetch available model IDs from any OpenAI-compatible /models endpoint."""
    url = _endpoint(provider, "models")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_build_headers(provider))
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
        return {"ok": True, "models": ids}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "models": []}


async def test_provider_connection(provider: ProviderConfig) -> dict[str, Any]:
    """Verify endpoint auth and reachability via /models or minimal completion."""
    models_res = await list_provider_models(provider)
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
    pattern = re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL)
    calls: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
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


async def stream_chat_completion(
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    on_token: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict[str, Any]:
    """Stream OpenAI-compatible /chat/completions and return content + tool_calls."""
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

    content_chunks: list[str] = []
    tool_calls_map: dict[int, dict[str, Any]] = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=180.0, write=30.0, pool=10.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code == 400 and use_native_tools:
                # Some endpoints reject 'tools' or 'stream' with tools — retry without native tools
                err_body = (await resp.aread()).decode("utf-8", errors="ignore")
                if "tool" in err_body.lower() or "function" in err_body.lower():
                    return await _non_tool_fallback(provider, messages, tools, on_token)
                raise RuntimeError(f"LLM HTTP 400: {err_body[:600]}")

            if resp.status_code >= 400:
                err_body = (await resp.aread()).decode("utf-8", errors="ignore")
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {err_body[:600]}")

            ctype = resp.headers.get("content-type", "")
            if "application/json" in ctype:
                # Provider ignored stream=True and returned a plain JSON body
                data = json.loads((await resp.aread()).decode("utf-8"))
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message", {})
                text = msg.get("content") or ""
                if text and on_token:
                    await on_token(text)
                return {
                    "content": text,
                    "tool_calls": msg.get("tool_calls") or [],
                    "used_react_fallback": False,
                }

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                piece = delta.get("content")
                if piece:
                    content_chunks.append(piece)
                    if on_token:
                        await on_token(piece)

                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    entry = tool_calls_map.setdefault(
                        idx,
                        {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:10]}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]

    full_text = "".join(content_chunks)
    tool_calls = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())]

    used_react = False
    if not tool_calls and not use_native_tools and tools:
        react_calls = parse_react_tool_calls(full_text)
        if react_calls:
            tool_calls = react_calls
            used_react = True

    return {
        "content": full_text,
        "tool_calls": tool_calls,
        "used_react_fallback": used_react,
    }


async def _non_tool_fallback(
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    on_token: Optional[Callable[[str], Awaitable[None]]],
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
    return await stream_chat_completion(fake_prov, patched, tools=tools, on_token=on_token)
