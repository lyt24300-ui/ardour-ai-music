"""Async Streamable HTTP JSON-RPC 2.0 client for Ardour MCP HTTP surface."""

from __future__ import annotations

import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import WORKSPACE_ROOT


def _discover_soundfonts() -> list[str]:
    """Locate installed .sf2/.sf3 SoundFont banks on this machine."""
    search_dirs: list[Path] = [
        WORKSPACE_ROOT / "share" / "ardour9" / "soundfonts",
        Path.home() / "Documents" / "SoundFonts",
        Path.home() / ".lv2",
        WORKSPACE_ROOT / "lib" / "ardour9" / "LV2",
    ]
    data_env = os.environ.get("ARDOUR_DATA_PATH", "")
    for part in data_env.split(os.pathsep):
        if part.strip():
            search_dirs.append(Path(part.strip()) / "soundfonts")
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        search_dirs.append(Path(appdata) / "LV2")

    found: list[str] = []
    seen: set[str] = set()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for ext in ("*.sf2", "*.sf3", "*.SF2", "*.SF3"):
            for f in d.rglob(ext):
                key = str(f.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    found.append(str(f))
    return found


def _discover_fluidsynth_presets() -> list[str]:
    """Locate LV2 preset labels registered for urn:ardour:a-fluidsynth."""
    lv2_dirs: list[Path] = [
        WORKSPACE_ROOT / "lib" / "ardour9" / "LV2",
        Path.home() / ".lv2",
    ]
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        lv2_dirs.append(Path(appdata) / "LV2")

    labels: list[str] = []
    seen: set[str] = set()
    label_re = re.compile(r'rdfs:label\s+"([^"]+)"')
    for base in lv2_dirs:
        if not base.is_dir():
            continue
        for ttl in base.glob("*.lv2/manifest.ttl"):
            try:
                txt = ttl.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "urn:ardour:a-fluidsynth" in txt and "pset:Preset" in txt:
                for m in label_re.finditer(txt):
                    lbl = m.group(1).strip()
                    if lbl and lbl not in seen:
                        seen.add(lbl)
                        labels.append(lbl)
    return labels


class ArdourMCPClient:
    """Talks directly to Ardour's built-in libs/surfaces/mcp_http server."""

    def __init__(self, endpoint_url: str = "http://127.0.0.1:4820/mcp") -> None:
        self.endpoint_url = endpoint_url
        self._id_counter = itertools.count(1)
        self._initialized = False
        self._tools_cache: list[dict[str, Any]] = []
        self._last_error: str = ""
        self._server_info: dict[str, Any] = {}

    def set_url(self, url: str) -> None:
        if url != self.endpoint_url:
            self.endpoint_url = url
            self._initialized = False
            self._tools_cache = []

    async def _rpc(self, method: str, params: Optional[dict[str, Any]] = None, timeout: float = 20.0) -> dict[str, Any]:
        req_id = next(self._id_counter)
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=1.2, read=timeout, write=timeout, pool=2.0)) as client:
            resp = await client.post(self.endpoint_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data and data["error"] is not None:
            err = data["error"]
            raise RuntimeError(f"Ardour MCP error {err.get('code')}: {err.get('message')}")
        return data.get("result", {})

    async def check_connection(self) -> dict[str, Any]:
        """Ping Ardour MCP server via initialize + session/get_info."""
        t0 = time.perf_counter()
        try:
            init_res = await self._rpc(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "ardour-ai-music-studio", "version": "1.0.0"},
                },
                timeout=4.0,
            )
            self._initialized = True
            self._server_info = init_res.get("serverInfo", {})
            self._last_error = ""

            tools = await self.list_tools(force=True)
            session_info = {}
            try:
                s_res = await self.call_tool("session_get_info", {})
                session_info = s_res.get("structured", s_res)
            except Exception:
                pass

            return {
                "connected": True,
                "url": self.endpoint_url,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "server_info": self._server_info,
                "tool_count": len(tools),
                "session": session_info,
            }
        except Exception as exc:
            self._initialized = False
            self._last_error = str(exc)
            return {
                "connected": False,
                "url": self.endpoint_url,
                "error": str(exc),
                "tool_count": 0,
                "session": {},
            }

    async def list_tools(self, force: bool = False) -> list[dict[str, Any]]:
        if self._tools_cache and not force:
            return self._tools_cache
        res = await self._rpc("tools/list", {}, timeout=10.0)
        tools = res.get("tools", [])
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: Optional[dict[str, Any]] = None, timeout: float = 60.0) -> dict[str, Any]:
        """Call a single Ardour MCP tool and normalize structured/text output."""
        if not self._initialized:
            try:
                await self._rpc(
                    "initialize",
                    {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ardour-ai-music-studio", "version": "1.0.0"},
                    },
                    timeout=5.0,
                )
                self._initialized = True
            except Exception as exc:
                raise RuntimeError(
                    f"Ardour MCP server unavailable at {self.endpoint_url}. "
                    f"Make sure Ardour is open and MCPHttp surface is active. ({exc})"
                ) from exc

        res = await self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )
        is_error = bool(res.get("isError", False))
        structured = res.get("structuredContent")
        text_parts = [
            c.get("text", "")
            for c in res.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        text = "\n".join(text_parts).strip()
        if structured is None and text:
            try:
                structured = json.loads(text)
            except Exception:
                structured = {"text": text}

        if is_error:
            raise RuntimeError(text or json.dumps(structured, ensure_ascii=False))

        return {
            "ok": True,
            "tool": name,
            "structured": structured if structured is not None else {},
            "text": text,
        }

    async def get_session_snapshot_context(self) -> dict[str, Any]:
        """Gather compact live session state (info, tracks, markers, available plugins)."""
        try:
            from .sound_library import inspect_sound_library
            lib = inspect_sound_library(category="all", limit=15)
            library_summary: dict[str, Any] = {
                "soundfonts_count": lib.get("soundfonts_count", 0),
                "nam_models_count": lib.get("nam_models_count", 0),
                "total_audio_samples": lib.get("total_audio_samples", 0),
                "top_sample_categories": dict(list(lib.get("category_counts", {}).items())[:15]),
                "nam_models": [m.get("name") for m in lib.get("nam_models", [])],
            }
        except Exception:
            library_summary = {}

        ctx: dict[str, Any] = {
            "connected": False,
            "soundfonts": _discover_soundfonts(),
            "fluid_synth_presets": _discover_fluidsynth_presets(),
            "sound_library_summary": library_summary,
        }
        try:
            info = await self.call_tool("session_get_info", {})
            tracks = await self.call_tool("tracks_list", {})
            markers = await self.call_tool("markers_list", {})
            ctx.update({
                "connected": True,
                "session": info.get("structured", {}),
                "tracks": tracks.get("structured", {}),
                "markers": markers.get("structured", {}),
            })
            try:
                master = await self.call_tool("master_get_info", {})
                ctx["master"] = master.get("structured", {})
            except Exception:
                pass
        except Exception as exc:
            ctx["error"] = str(exc)
        return ctx
