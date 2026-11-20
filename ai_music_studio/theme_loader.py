"""Parse Ardour's native XML .colors themes into exact CSS variables.

Reads directly from ardour/gtk2_ardour/themes/*.colors so AI Music Studio
matches the exact palette, contrast, LED accents, and dark/light tone of
Ardour's interface.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .config import ARDOUR_THEMES_DIR, DEFAULT_THEME


# Fallback palette copied verbatim from ardour/gtk2_ardour/themes/dark-ardour.colors
DEFAULT_DARK_COLORS: dict[str, str] = {
    "theme:bg": "#3d3d3d",
    "theme:bg1": "#373737",
    "theme:bg2": "#101010",
    "neutral:background": "#202020",
    "neutral:background2": "#555555",
    "neutral:backgroundest": "#000000",
    "neutral:foreground": "#e5e5e5",
    "neutral:foreground2": "#cccccc",
    "neutral:foregroundest": "#ffffff",
    "neutral:midground": "#7f7f7f",
    "widget:bg": "#4f4d4d",
    "widget:blue": "#526d94",
    "widget:blue darker": "#20242a",
    "widget:blue lighter": "#628db4",
    "widget:gray": "#595755",
    "widget:green": "#617552",
    "widget:green darker": "#3c403a",
    "widget:orange": "#904010",
    "widget:ruddy": "#7c3a3a",
    "widget:yellow": "#eddc3d",
    "alert:blue": "#2e41e6",
    "alert:cyan": "#20b2b2",
    "alert:green": "#3dcc4c",
    "alert:greenish": "#459952",
    "alert:orange": "#e58b05",
    "alert:red": "#f10000",
    "alert:ruddy": "#bb2828",
    "alert:yellow": "#e0e000",
    "theme:contrasting": "#f10000",
    "theme:contrasting alt": "#cca336",
    "theme:contrasting clock": "#78cd2f",
    "theme:contrasting less": "#46b357",
    "theme:contrasting selection": "#5da3c1",
    "meter color0": "#008800",
    "meter color2": "#00ff00",
    "meter color4": "#fff000",
    "meter color6": "#ff8800",
    "meter color8": "#ff0000",
}


def _parse_ardour_rgba(raw: str) -> str:
    """Convert Ardour's RRGGBBAA or 0xRRGGBBAA string to CSS hex or rgba()."""
    val = raw.strip().lower()
    if val.startswith("0x"):
        val = val[2:]
    if val.startswith("#"):
        val = val[1:]
    if len(val) == 6:
        return f"#{val}"
    if len(val) == 8:
        r = int(val[0:2], 16)
        g = int(val[2:4], 16)
        b = int(val[4:6], 16)
        a = int(val[6:8], 16)
        if a >= 254:
            return f"#{val[0:6]}"
        return f"rgba({r}, {g}, {b}, {a / 255.0:.3f})"
    return "#3d3d3d"


def _hex_luma(hex_color: str) -> float:
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return 0.2
    r = int(c[0:2], 16) / 255.0
    g = int(c[2:4], 16) / 255.0
    b = int(c[4:6], 16) / 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _adjust_hex(hex_color: str, delta: int) -> str:
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return hex_color
    r = max(0, min(255, int(c[0:2], 16) + delta))
    g = max(0, min(255, int(c[2:4], 16) + delta))
    b = max(0, min(255, int(c[4:6], 16) + delta))
    return f"#{r:02x}{g:02x}{b:02x}"


def list_ardour_themes() -> list[dict[str, str]]:
    """Return available Ardour themes discovered in gtk2_ardour/themes."""
    themes: list[dict[str, str]] = []
    if ARDOUR_THEMES_DIR.exists():
        for path in sorted(ARDOUR_THEMES_DIR.glob("*-ardour.colors")):
            slug = path.name.replace("-ardour.colors", "")
            label = slug.replace("_", " ").title()
            try:
                root = ET.parse(path).getroot()
                tname = root.attrib.get("theme-name")
                if tname:
                    label = tname.replace("_", " ").title()
            except Exception:
                pass
            themes.append({"id": slug, "name": label, "file": str(path)})
    if not themes:
        themes.append({"id": "dark", "name": "Dark", "file": ""})
    return themes


def load_ardour_theme(theme_name: str = DEFAULT_THEME) -> dict[str, Any]:
    """Load an Ardour .colors file and map it to semantic CSS variables."""
    colors = dict(DEFAULT_DARK_COLORS)
    aliases: dict[str, str] = {}
    source_file = ""

    candidate = ARDOUR_THEMES_DIR / f"{theme_name}-ardour.colors"
    if not candidate.exists():
        candidate = ARDOUR_THEMES_DIR / "dark-ardour.colors"

    if candidate.exists():
        source_file = str(candidate)
        try:
            tree = ET.parse(candidate)
            root = tree.getroot()
            for el in root.findall("./Colors/Color"):
                name = el.attrib.get("name", "")
                val = el.attrib.get("value", "")
                if name and val:
                    colors[name] = _parse_ardour_rgba(val)
            for el in root.findall("./ColorAliases/ColorAlias"):
                name = el.attrib.get("name", "")
                alias = el.attrib.get("alias", "")
                if name and alias:
                    aliases[name] = alias
        except Exception:
            pass

    def resolve(key: str, fallback: str) -> str:
        seen = set()
        cur = key
        while cur in aliases and cur not in seen:
            seen.add(cur)
            cur = aliases[cur]
        return colors.get(cur, colors.get(key, fallback))

    bg = resolve("gtk_background", resolve("theme:bg", "#3d3d3d"))
    bg1 = resolve("gtk_audio_track", resolve("theme:bg1", "#373737"))
    bg2 = resolve("gtk_bases", resolve("theme:bg2", "#101010"))
    track_bg = resolve("audio track base", resolve("neutral:background", "#202020"))
    bg_elevated = resolve("neutral:background2", "#555555")
    border = resolve("border color", resolve("neutral:backgroundest", "#000000"))
    fg = resolve("gtk_foreground", resolve("neutral:foreground", "#e5e5e5"))
    fg2 = resolve("gtk_light_text_on_dark", resolve("neutral:foreground2", "#cccccc"))
    fg_muted = resolve("neutral:midground", "#7f7f7f")
    fg_bright = resolve("neutral:foregroundest", "#ffffff")
    btn_bg = resolve("generic button: fill", resolve("widget:bg", "#4f4d4d"))
    btn_top = _adjust_hex(btn_bg, 14)
    btn_bot = _adjust_hex(btn_bg, -12)
    btn_hover = _adjust_hex(btn_bg, 18)

    page_active = resolve("page switch button: fill active", resolve("alert:green", "#3dcc4c"))
    selection = resolve("gtk_bg_selected", resolve("theme:contrasting selection", "#5da3c1"))
    clock_fg = resolve("clock: text", resolve("theme:contrasting clock", "#78cd2f"))
    clock_bg = resolve("clock: background", bg2)

    is_light = _hex_luma(bg) > 0.52
    active_text = "#101010" if _hex_luma(page_active) > 0.45 else "#ffffff"

    css_vars = {
        "--ardour-bg": bg,
        "--ardour-bg-panel": bg1,
        "--ardour-bg-dark": bg2,
        "--ardour-bg-track": track_bg,
        "--ardour-bg-elevated": bg_elevated,
        "--ardour-border": border,
        "--ardour-border-soft": _adjust_hex(bg, -24) if not is_light else _adjust_hex(bg, -40),
        "--ardour-fg": fg,
        "--ardour-fg-secondary": fg2,
        "--ardour-fg-muted": fg_muted,
        "--ardour-fg-bright": fg_bright,
        "--ardour-btn-bg": btn_bg,
        "--ardour-btn-top": btn_top,
        "--ardour-btn-bot": btn_bot,
        "--ardour-btn-hover": btn_hover,
        "--ardour-page-active": page_active,
        "--ardour-page-active-fg": active_text,
        "--ardour-selection": selection,
        "--ardour-clock-bg": clock_bg,
        "--ardour-clock-fg": clock_fg,
        "--ardour-widget-blue": resolve("widget:blue", "#526d94"),
        "--ardour-widget-blue-dark": resolve("widget:blue darker", "#20242a"),
        "--ardour-widget-blue-light": resolve("widget:blue lighter", "#628db4"),
        "--ardour-widget-green": resolve("widget:green", "#617552"),
        "--ardour-widget-green-dark": resolve("widget:green darker", "#3c403a"),
        "--ardour-widget-orange": resolve("widget:orange", "#904010"),
        "--ardour-widget-ruddy": resolve("widget:ruddy", "#7c3a3a"),
        "--ardour-alert-green": resolve("alert:green", "#3dcc4c"),
        "--ardour-alert-orange": resolve("alert:orange", "#e58b05"),
        "--ardour-alert-red": resolve("alert:red", "#f10000"),
        "--ardour-alert-yellow": resolve("alert:yellow", "#e0e000"),
        "--ardour-alert-cyan": resolve("alert:cyan", "#20b2b2"),
        "--ardour-contrasting-alt": resolve("theme:contrasting alt", "#cca336"),
        "--ardour-meter-0": resolve("meter color0", "#008800"),
        "--ardour-meter-2": resolve("meter color2", "#00ff00"),
        "--ardour-meter-4": resolve("meter color4", "#fff000"),
        "--ardour-meter-6": resolve("meter color6", "#ff8800"),
        "--ardour-meter-8": resolve("meter color8", "#ff0000"),
    }

    return {
        "theme_id": theme_name,
        "source_file": source_file,
        "is_light": is_light,
        "css_vars": css_vars,
    }
