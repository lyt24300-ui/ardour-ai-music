"""Pydantic domain models for AI Music Studio."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderConfig(BaseModel):
    id: str
    name: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=128000)
    supports_tools: bool = True
    supports_audio: bool = False
    custom_headers: dict[str, str] = Field(default_factory=dict)
    is_default: bool = False


class WebSearchConfig(BaseModel):
    enabled: bool = True
    provider: Literal["duckduckgo", "tavily", "brave", "serper", "searxng"] = "duckduckgo"
    api_key: str = ""
    base_url: str = ""
    max_results: int = Field(default=6, ge=1, le=15)


class StudioSettings(BaseModel):
    ardour_mcp_url: str = "http://127.0.0.1:4820/mcp"
    theme_name: str = "dark"
    autonomy_mode: Literal["autonomous", "step_by_step"] = "autonomous"
    auto_snapshot: bool = True
    max_agent_steps: int = Field(default=60, ge=3, le=250)
    max_reconnects: int = Field(default=10, ge=1, le=20)
    whisper_model: str = "base"
    whisper_enabled: bool = True
    whisper_language: str = ""  # empty = auto-detect
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class SkillItem(BaseModel):
    id: str
    name: str
    description: str = ""
    category: str = "general"
    enabled: bool = True
    builtin: bool = False
    content: str = ""
    updated_at: str = Field(default_factory=utc_now_iso)


class AudioAsset(BaseModel):
    id: str
    category: Literal["references", "vocals", "samples", "attachments"]
    filename: str
    stored_path: str
    size_bytes: int = 0
    role: str = ""
    notes: str = ""
    uploaded_at: str = Field(default_factory=utc_now_iso)
    analysis: dict[str, Any] = Field(default_factory=dict)
    alignment: dict[str, Any] = Field(default_factory=dict)


class LyricsData(BaseModel):
    title: str = ""
    raw_text: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    aligned_vocal_id: Optional[str] = None
    aligned_lines: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)


class PlanStep(BaseModel):
    id: str
    title: str
    status: Literal["pending", "in_progress", "done", "skipped", "error"] = "pending"
    detail: str = ""


class PluginInstallCard(BaseModel):
    id: str
    plugin_name: str
    format: str = "VST3"
    price_type: str = "Free"
    purpose: str = ""
    download_url: str = ""
    install_path_windows: str = r"C:\Program Files\Common Files\VST3"
    steps: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    status: Literal["pending", "installed", "skipped"] = "pending"


class ToolCallTrace(BaseModel):
    id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    ok: bool = True
    duration_ms: int = 0


class ChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str = Field(default_factory=utc_now_iso)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    plugin_cards: list[PluginInstallCard] = Field(default_factory=list)
    plan_snapshot: list[PlanStep] = Field(default_factory=list)
    snapshot_name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    provider_id: Optional[str] = None
    attachment_ids: list[str] = Field(default_factory=list)
    mode: Optional[Literal["autonomous", "step_by_step"]] = None
