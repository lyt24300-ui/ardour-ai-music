"""Configuration paths and defaults for AI Music Studio."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent
ARDOUR_ROOT = WORKSPACE_ROOT / "ardour"


def _resolve_themes_dir() -> Path:
    candidates: list[Path] = []
    data_env = os.environ.get("ARDOUR_DATA_PATH", "")
    for part in data_env.split(os.pathsep):
        if part.strip():
            candidates.append(Path(part.strip()) / "themes")
    candidates.extend([
        WORKSPACE_ROOT / "share" / "ardour9" / "themes",
        WORKSPACE_ROOT / "gtk2_ardour" / "themes",
        ARDOUR_ROOT / "gtk2_ardour" / "themes",
    ])
    for cand in candidates:
        if cand.is_dir():
            return cand
    return WORKSPACE_ROOT / "share" / "ardour9" / "themes"


ARDOUR_THEMES_DIR = _resolve_themes_dir()

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SKILLS_DIR = DATA_DIR / "skills"
EXPORTS_DIR = DATA_DIR / "exports"
DEFAULT_SKILLS_DIR = BASE_DIR / "default_skills"
STATIC_DIR = BASE_DIR / "static"

STUDIO_HOST = os.environ.get("AI_MUSIC_HOST", "127.0.0.1")
STUDIO_PORT = int(os.environ.get("AI_MUSIC_PORT", "4821"))
ARDOUR_MCP_URL = os.environ.get("ARDOUR_MCP_URL", "http://127.0.0.1:4820/mcp")

DEFAULT_THEME = os.environ.get("AI_MUSIC_THEME", "dark")
WHISPER_MODEL_SIZE = os.environ.get("AI_MUSIC_WHISPER_MODEL", "base")


def find_ffmpeg() -> str:
    """Locate ffmpeg executable on Windows or PATH."""
    env_ff = os.environ.get("FFMPEG_PATH")
    if env_ff and Path(env_ff).exists():
        return env_ff
    for candidate in (
        Path(r"C:\ffmpeg\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    which = shutil.which("ffmpeg")
    return which or "ffmpeg"


FFMPEG_BIN = find_ffmpeg()


def ensure_dirs() -> None:
    """Create persistent storage directories if missing."""
    for d in (
        DATA_DIR,
        UPLOADS_DIR / "references",
        UPLOADS_DIR / "vocals",
        UPLOADS_DIR / "samples",
        UPLOADS_DIR / "attachments",
        SKILLS_DIR,
        EXPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
