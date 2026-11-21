"""FastAPI backend for AI Music Studio."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import AIProducerAgent
from .audio_analyzer import align_vocal_with_lyrics, analyze_audio_file
from .config import DATA_DIR, STATIC_DIR, ensure_dirs
from .llm_client import list_provider_models, test_provider_connection
from .mcp_client import ArdourMCPClient
from .models import (
    ChatRequest,
    ProviderConfig,
    SkillItem,
    StudioSettings,
)
from .storage import StudioStorage
from .theme_loader import list_ardour_themes, load_ardour_theme
from .web_search import web_search

ensure_dirs()

_studio_logger = logging.getLogger("ai_music_studio")
if not _studio_logger.handlers:
    _studio_logger.setLevel(logging.INFO)
    _fh = logging.FileHandler(DATA_DIR / "studio.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _studio_logger.addHandler(_fh)

storage = StudioStorage()
settings_init = storage.get_settings()
mcp_client = ArdourMCPClient(settings_init.ardour_mcp_url)
agent = AIProducerAgent(storage, mcp_client)

app = FastAPI(title="Ardour AI Music Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _focus_studio_window() -> bool:
    """Bring the standalone AI Music Studio window to foreground on Windows."""
    if sys.platform != "win32":
        return False
    try:
        import win32con
        import win32gui

        found_hwnds: list[int] = []

        def enum_cb(hwnd: int, _ctx: Any) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd) or ""
            if "AI Music Studio" in title:
                found_hwnds.append(hwnd)

        win32gui.EnumWindows(enum_cb, None)
        if found_hwnds:
            hwnd = found_hwnds[0]
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
            return True
    except Exception:
        pass
    return False


# ---------- Root & Health ----------
@app.get("/", response_class=HTMLResponse)
async def index_page() -> HTMLResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>AI Music Studio UI missing</h1>", status_code=500)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "ardour-ai-music-studio", "agent_running": agent.is_running}


@app.post("/api/focus")
async def focus_window() -> dict[str, Any]:
    focused = _focus_studio_window()
    return {"ok": True, "focused": focused}


# ---------- Themes (Exact Ardour XML .colors Sync) ----------
@app.get("/api/themes")
async def api_list_themes() -> dict[str, Any]:
    cur = storage.get_settings().theme_name
    return {
        "current": cur,
        "themes": list_ardour_themes(),
        "active_theme": load_ardour_theme(cur),
    }


@app.get("/api/themes/{theme_id}")
async def api_get_theme(theme_id: str) -> dict[str, Any]:
    return load_ardour_theme(theme_id)


# ---------- Settings ----------
@app.get("/api/settings")
async def api_get_settings() -> StudioSettings:
    return storage.get_settings()


@app.put("/api/settings")
async def api_save_settings(payload: StudioSettings) -> StudioSettings:
    saved = storage.save_settings(payload)
    mcp_client.set_url(saved.ardour_mcp_url)
    return saved


# ---------- Ardour MCP Engine ----------
@app.get("/api/ardour/status")
async def api_ardour_status() -> dict[str, Any]:
    mcp_client.set_url(storage.get_settings().ardour_mcp_url)
    return await mcp_client.check_connection(max_reconnects=1)


@app.post("/api/ardour/reconnect")
async def api_ardour_reconnect(max_reconnects: int = 10) -> dict[str, Any]:
    mcp_client.set_url(storage.get_settings().ardour_mcp_url)
    return await mcp_client.check_connection(max_reconnects=max(1, min(max_reconnects, 10)))


@app.get("/api/ardour/tools")
async def api_ardour_tools() -> dict[str, Any]:
    mcp_client.set_url(storage.get_settings().ardour_mcp_url)
    try:
        tools = await mcp_client.list_tools(force=True)
        return {"ok": True, "count": len(tools), "tools": tools}
    except Exception as exc:
        return {"ok": False, "count": 0, "tools": [], "error": str(exc)}


@app.post("/api/ardour/undo")
async def api_ardour_undo(count: int = 1) -> dict[str, Any]:
    mcp_client.set_url(storage.get_settings().ardour_mcp_url)
    try:
        res = await mcp_client.call_tool("session_undo", {"count": max(1, count)})
        return {"ok": True, "result": res}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/ardour/rescan-plugins")
async def api_ardour_rescan_plugins() -> dict[str, Any]:
    mcp_client.set_url(storage.get_settings().ardour_mcp_url)
    try:
        try:
            res = await mcp_client.call_tool("plugin_rescan", {})
        except Exception:
            res = await mcp_client.call_tool("plugin_list_available", {})
        return {"ok": True, "result": res}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------- Providers (Any OpenAI-Compatible API) ----------
@app.get("/api/providers")
async def api_list_providers() -> list[ProviderConfig]:
    return storage.list_providers()


@app.post("/api/providers")
async def api_upsert_provider(provider: ProviderConfig) -> ProviderConfig:
    if not provider.id:
        provider.id = f"prov_{uuid.uuid4().hex[:8]}"
    return storage.upsert_provider(provider)


@app.delete("/api/providers/{provider_id}")
async def api_delete_provider(provider_id: str) -> dict[str, Any]:
    ok = storage.delete_provider(provider_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"ok": True}


@app.post("/api/providers/test")
async def api_test_provider(provider: ProviderConfig) -> dict[str, Any]:
    return await test_provider_connection(provider)


@app.post("/api/providers/models")
async def api_list_provider_models(provider: ProviderConfig) -> dict[str, Any]:
    return await list_provider_models(provider)


# ---------- Skills ----------
@app.get("/api/skills")
async def api_list_skills() -> list[SkillItem]:
    return storage.list_skills()


@app.post("/api/skills")
async def api_upsert_skill(skill: SkillItem) -> SkillItem:
    return storage.upsert_skill(skill)


class SkillToggleReq(BaseModel):
    enabled: bool


@app.patch("/api/skills/{skill_id}")
async def api_toggle_skill(skill_id: str, req: SkillToggleReq) -> SkillItem:
    updated = storage.toggle_skill(skill_id, req.enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="Skill not found")
    return updated


@app.delete("/api/skills/{skill_id}")
async def api_delete_skill(skill_id: str) -> dict[str, Any]:
    if not storage.delete_skill(skill_id):
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"ok": True}


# ---------- Audio & File Assets (References / Vocals / Samples / Attachments) ----------
@app.get("/api/assets")
async def api_list_assets(category: Optional[str] = None) -> list[dict[str, Any]]:
    return [a.model_dump() for a in storage.list_assets(category)]


@app.post("/api/assets/upload")
async def api_upload_asset(
    file: UploadFile = File(...),
    category: str = Form("references"),
    role: str = Form(""),
    notes: str = Form(""),
    auto_analyze: bool = Form(True),
) -> dict[str, Any]:
    if category not in ("references", "vocals", "samples", "attachments"):
        raise HTTPException(status_code=400, detail="Invalid asset category")
    data = await file.read()
    asset = storage.save_uploaded_asset(
        category=category,
        filename=file.filename or "audio.wav",
        data=data,
        role=role,
        notes=notes,
    )

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff", ".opus", ".aac", ".wma"}
    ext = Path(asset.filename).suffix.lower()
    if auto_analyze and (ext in audio_exts or category in ("references", "vocals", "samples")):
        try:
            analysis = await asyncio.to_thread(analyze_audio_file, asset.stored_path, category)
            asset.analysis = analysis
            storage.update_asset(asset)
        except Exception as exc:
            asset.analysis = {"error": str(exc)}
            storage.update_asset(asset)

    return asset.model_dump()


class AssetMetaPatch(BaseModel):
    role: Optional[str] = None
    notes: Optional[str] = None


@app.patch("/api/assets/{asset_id}")
async def api_patch_asset(asset_id: str, patch: AssetMetaPatch) -> dict[str, Any]:
    asset = storage.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    if patch.role is not None:
        asset.role = patch.role
    if patch.notes is not None:
        asset.notes = patch.notes
    storage.update_asset(asset)
    return asset.model_dump()


@app.post("/api/assets/{asset_id}/analyze")
async def api_reanalyze_asset(asset_id: str) -> dict[str, Any]:
    asset = storage.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    analysis = await asyncio.to_thread(analyze_audio_file, asset.stored_path, asset.category)
    asset.analysis = analysis
    storage.update_asset(asset)
    return asset.model_dump()


@app.post("/api/assets/{asset_id}/align")
async def api_align_vocal(asset_id: str, language: str = "") -> dict[str, Any]:
    asset = storage.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Vocal asset not found")
    settings = storage.get_settings()
    lyrics = storage.get_lyrics()
    vad_phrases = asset.analysis.get("vocal", {}).get("phrases") if asset.analysis else None
    alignment = await asyncio.to_thread(
        align_vocal_with_lyrics,
        asset.stored_path,
        lyrics.sections,
        vad_phrases,
        settings.whisper_enabled,
        settings.whisper_model,
        language or settings.whisper_language,
    )
    asset.alignment = alignment
    storage.update_asset(asset)
    storage.save_lyrics(
        lyrics.title,
        lyrics.raw_text,
        aligned_vocal_id=asset.id,
        aligned_lines=alignment.get("aligned_lines", []),
    )
    return {"asset": asset.model_dump(), "alignment": alignment}


@app.get("/api/assets/{asset_id}/audio")
async def api_stream_audio(asset_id: str) -> FileResponse:
    asset = storage.get_asset(asset_id)
    if not asset or not Path(asset.stored_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(asset.stored_path, filename=asset.filename)


@app.delete("/api/assets/{asset_id}")
async def api_delete_asset(asset_id: str) -> dict[str, Any]:
    if not storage.delete_asset(asset_id):
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"ok": True}


# ---------- Lyrics ----------
@app.get("/api/lyrics")
async def api_get_lyrics() -> dict[str, Any]:
    return storage.get_lyrics().model_dump()


class LyricsSaveReq(BaseModel):
    title: str = ""
    raw_text: str = ""


@app.put("/api/lyrics")
async def api_save_lyrics(req: LyricsSaveReq) -> dict[str, Any]:
    saved = storage.save_lyrics(req.title, req.raw_text)
    return saved.model_dump()


# ---------- Web Search Test ----------
class SearchTestReq(BaseModel):
    query: str


@app.post("/api/search/test")
async def api_test_search(req: SearchTestReq) -> dict[str, Any]:
    return await web_search(req.query, storage.get_settings().web_search, max_results=3)


# ---------- Chat & Agent Stream ----------
@app.get("/api/chat/state")
async def api_chat_state() -> dict[str, Any]:
    return {
        "messages": [m.model_dump() for m in storage.get_chat_history()],
        "plan": [s.model_dump() for s in storage.get_plan()],
        "running": agent.is_running,
    }


@app.post("/api/chat/clear")
async def api_chat_clear() -> dict[str, Any]:
    agent.stop()
    storage.clear_chat()
    return {"ok": True}


@app.post("/api/chat/stop")
async def api_chat_stop() -> dict[str, Any]:
    agent.stop()
    return {"ok": True}


class PluginCardActionReq(BaseModel):
    status: str  # installed | skipped


@app.post("/api/plugin-cards/{card_id}/status")
async def api_plugin_card_status(card_id: str, req: PluginCardActionReq) -> dict[str, Any]:
    if req.status not in ("installed", "skipped", "pending"):
        raise HTTPException(status_code=400, detail="Invalid status")
    ok = storage.update_plugin_card_status(card_id, req.status)
    return {"ok": ok}


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest) -> StreamingResponse:
    async def event_gen():
        async for ev in agent.run_chat_stream(
            user_message=req.message,
            provider_id=req.provider_id,
            attachment_ids=req.attachment_ids,
            mode_override=req.mode,
        ):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
