"""Persistent JSON & Markdown storage for AI Music Studio."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import (
    DATA_DIR,
    DEFAULT_SKILLS_DIR,
    SKILLS_DIR,
    UPLOADS_DIR,
    ensure_dirs,
)
from .models import (
    AudioAsset,
    ChatMessage,
    LyricsData,
    PlanStep,
    ProviderConfig,
    SkillItem,
    StudioSettings,
    utc_now_iso,
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return slug or f"skill_{uuid.uuid4().hex[:8]}"


def _parse_lyrics_sections(raw_text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    cur_name = "Main"
    cur_lines: list[str] = []
    header_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")

    for raw_line in raw_text.splitlines():
        m = header_re.match(raw_line)
        if m:
            if cur_lines:
                sections.append({"section": cur_name, "lines": cur_lines})
                cur_lines = []
            cur_name = m.group(1).strip()
        else:
            stripped = raw_line.strip()
            if stripped:
                cur_lines.append(stripped)

    if cur_lines:
        sections.append({"section": cur_name, "lines": cur_lines})
    return sections


class StudioStorage:
    """Thread-safe local JSON + Markdown file persistence."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        ensure_dirs()
        self.data_dir = data_dir or DATA_DIR
        self.uploads_dir = self.data_dir / "uploads"
        self.skills_dir = self.data_dir / "skills"
        for sub in ("references", "vocals", "samples", "attachments"):
            (self.uploads_dir / sub).mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.settings_file = self.data_dir / "settings.json"
        self.providers_file = self.data_dir / "providers.json"
        self.assets_file = self.data_dir / "assets.json"
        self.lyrics_file = self.data_dir / "lyrics.json"
        self.chat_file = self.data_dir / "chat_history.json"
        self.plan_file = self.data_dir / "plan.json"

        self._seed_default_skills()
        self._seed_default_providers()

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ---------- Settings ----------
    def get_settings(self) -> StudioSettings:
        data = self._read_json(self.settings_file, {})
        return StudioSettings(**data)

    def save_settings(self, settings: StudioSettings) -> StudioSettings:
        self._write_json(self.settings_file, settings.model_dump())
        return settings

    # ---------- Providers ----------
    def _seed_default_providers(self) -> None:
        if self.providers_file.exists():
            return
        defaults = [
            ProviderConfig(
                id="openai-default",
                name="OpenAI",
                base_url="https://api.openai.com/v1",
                api_key="",
                model="gpt-4o",
                is_default=True,
            ),
            ProviderConfig(
                id="openrouter-default",
                name="OpenRouter (Любые модели)",
                base_url="https://openrouter.ai/api/v1",
                api_key="",
                model="anthropic/claude-sonnet-4",
                custom_headers={"X-Title": "Ardour AI Music Studio"},
                is_default=False,
            ),
            ProviderConfig(
                id="ollama-local",
                name="Ollama (Локально)",
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama",
                model="qwen2.5:14b",
                is_default=False,
            ),
        ]
        self._write_json(self.providers_file, [p.model_dump() for p in defaults])

    def list_providers(self) -> list[ProviderConfig]:
        items = self._read_json(self.providers_file, [])
        return [ProviderConfig(**i) for i in items]

    def get_provider(self, provider_id: Optional[str] = None) -> Optional[ProviderConfig]:
        providers = self.list_providers()
        if not providers:
            return None
        if provider_id:
            for p in providers:
                if p.id == provider_id:
                    return p
        for p in providers:
            if p.is_default:
                return p
        return providers[0]

    def upsert_provider(self, provider: ProviderConfig) -> ProviderConfig:
        providers = self.list_providers()
        if provider.is_default:
            for p in providers:
                p.is_default = False
        replaced = False
        for idx, p in enumerate(providers):
            if p.id == provider.id:
                providers[idx] = provider
                replaced = True
                break
        if not replaced:
            if not providers:
                provider.is_default = True
            providers.append(provider)
        self._write_json(self.providers_file, [p.model_dump() for p in providers])
        return provider

    def delete_provider(self, provider_id: str) -> bool:
        providers = self.list_providers()
        new_list = [p for p in providers if p.id != provider_id]
        if len(new_list) == len(providers):
            return False
        if new_list and not any(p.is_default for p in new_list):
            new_list[0].is_default = True
        self._write_json(self.providers_file, [p.model_dump() for p in new_list])
        return True

    # ---------- Skills (.md with YAML frontmatter) ----------
    def _seed_default_skills(self) -> None:
        if not DEFAULT_SKILLS_DIR.exists():
            return
        for src in DEFAULT_SKILLS_DIR.glob("*.md"):
            dst = self.skills_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                continue
            try:
                cur = self._parse_skill_md(dst)
                nxt = self._parse_skill_md(src)
                if cur.builtin and cur.content.strip() != nxt.content.strip():
                    nxt.enabled = cur.enabled
                    self._write_skill_md(nxt)
            except Exception:
                pass

    def _parse_skill_md(self, path: Path) -> SkillItem:
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, Any] = {}
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) == 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    meta = {}
                body = parts[2].lstrip("\r\n")
        skill_id = path.stem
        return SkillItem(
            id=skill_id,
            name=str(meta.get("name", skill_id.replace("_", " ").title())),
            description=str(meta.get("description", "")),
            category=str(meta.get("category", "general")),
            enabled=bool(meta.get("enabled", True)),
            builtin=bool(meta.get("builtin", False)),
            content=body,
            updated_at=str(meta.get("updated_at", utc_now_iso())),
        )

    def _write_skill_md(self, skill: SkillItem) -> Path:
        meta = {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "enabled": skill.enabled,
            "builtin": skill.builtin,
            "updated_at": utc_now_iso(),
        }
        fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
        text = f"---\n{fm}\n---\n\n{skill.content.strip()}\n"
        path = self.skills_dir / f"{skill.id}.md"
        path.write_text(text, encoding="utf-8")
        return path

    def list_skills(self) -> list[SkillItem]:
        skills: list[SkillItem] = []
        for p in sorted(self.skills_dir.glob("*.md")):
            try:
                skills.append(self._parse_skill_md(p))
            except Exception:
                continue
        return skills

    def upsert_skill(self, skill: SkillItem) -> SkillItem:
        if not skill.id:
            skill.id = _slugify(skill.name)
        else:
            skill.id = _slugify(skill.id)
        skill.updated_at = utc_now_iso()
        self._write_skill_md(skill)
        return skill

    def toggle_skill(self, skill_id: str, enabled: bool) -> Optional[SkillItem]:
        path = self.skills_dir / f"{skill_id}.md"
        if not path.exists():
            return None
        sk = self._parse_skill_md(path)
        sk.enabled = enabled
        sk.updated_at = utc_now_iso()
        self._write_skill_md(sk)
        return sk

    def delete_skill(self, skill_id: str) -> bool:
        path = self.skills_dir / f"{skill_id}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    # ---------- Audio & File Assets ----------
    def list_assets(self, category: Optional[str] = None) -> list[AudioAsset]:
        raw = self._read_json(self.assets_file, [])
        assets = [AudioAsset(**a) for a in raw]
        if category:
            assets = [a for a in assets if a.category == category]
        return assets

    def get_asset(self, asset_id: str) -> Optional[AudioAsset]:
        for a in self.list_assets():
            if a.id == asset_id:
                return a
        return None

    def save_uploaded_asset(
        self,
        category: str,
        filename: str,
        data: bytes,
        role: str = "",
        notes: str = "",
    ) -> AudioAsset:
        asset_id = uuid.uuid4().hex[:12]
        safe_name = re.sub(r"[^\w.\- ()А-Яа-яЁё]+", "_", filename).strip() or f"file_{asset_id}.bin"
        dest = self.uploads_dir / category / f"{asset_id}_{safe_name}"
        dest.write_bytes(data)

        asset = AudioAsset(
            id=asset_id,
            category=category,  # type: ignore[arg-type]
            filename=filename,
            stored_path=str(dest),
            size_bytes=len(data),
            role=role,
            notes=notes,
        )
        all_assets = self.list_assets()
        all_assets.append(asset)
        self._write_json(self.assets_file, [a.model_dump() for a in all_assets])
        return asset

    def update_asset(self, asset: AudioAsset) -> AudioAsset:
        all_assets = self.list_assets()
        for idx, a in enumerate(all_assets):
            if a.id == asset.id:
                all_assets[idx] = asset
                break
        self._write_json(self.assets_file, [a.model_dump() for a in all_assets])
        return asset

    def delete_asset(self, asset_id: str) -> bool:
        all_assets = self.list_assets()
        target = next((a for a in all_assets if a.id == asset_id), None)
        if not target:
            return False
        try:
            Path(target.stored_path).unlink(missing_ok=True)
        except Exception:
            pass
        new_list = [a for a in all_assets if a.id != asset_id]
        self._write_json(self.assets_file, [a.model_dump() for a in new_list])
        return True

    # ---------- Lyrics ----------
    def get_lyrics(self) -> LyricsData:
        raw = self._read_json(self.lyrics_file, {})
        return LyricsData(**raw)

    def save_lyrics(
        self,
        title: str,
        raw_text: str,
        aligned_vocal_id: Optional[str] = None,
        aligned_lines: Optional[list[dict[str, Any]]] = None,
    ) -> LyricsData:
        existing = self.get_lyrics()
        sections = _parse_lyrics_sections(raw_text)
        lyrics = LyricsData(
            title=title,
            raw_text=raw_text,
            sections=sections,
            aligned_vocal_id=aligned_vocal_id if aligned_vocal_id is not None else existing.aligned_vocal_id,
            aligned_lines=aligned_lines if aligned_lines is not None else existing.aligned_lines,
            updated_at=utc_now_iso(),
        )
        self._write_json(self.lyrics_file, lyrics.model_dump())
        return lyrics

    # ---------- Chat & Plan ----------
    def get_chat_history(self) -> list[ChatMessage]:
        raw = self._read_json(self.chat_file, [])
        return [ChatMessage(**m) for m in raw]

    def append_message(self, msg: ChatMessage) -> ChatMessage:
        history = self.get_chat_history()
        history.append(msg)
        self._write_json(self.chat_file, [m.model_dump() for m in history])
        return msg

    def update_plugin_card_status(self, card_id: str, status: str) -> bool:
        history = self.get_chat_history()
        changed = False
        for msg in history:
            for card in msg.plugin_cards:
                if card.id == card_id:
                    card.status = status  # type: ignore[assignment]
                    changed = True
        if changed:
            self._write_json(self.chat_file, [m.model_dump() for m in history])
        return changed

    def clear_chat(self) -> None:
        self._write_json(self.chat_file, [])
        self._write_json(self.plan_file, [])

    def get_plan(self) -> list[PlanStep]:
        raw = self._read_json(self.plan_file, [])
        return [PlanStep(**s) for s in raw]

    def save_plan(self, steps: list[PlanStep]) -> list[PlanStep]:
        self._write_json(self.plan_file, [s.model_dump() for s in steps])
        return steps
