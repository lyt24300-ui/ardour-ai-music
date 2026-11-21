"""Discovery and inspection helpers for local SoundFonts, WAV samples, NAM models, and presets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _candidate_share_roots() -> list[Path]:
    roots: list[Path] = []
    env_data = os.environ.get("ARDOUR_DATA_PATH", "").strip()
    if env_data:
        for part in env_data.split(os.pathsep):
            p = Path(part.strip())
            if p.exists():
                roots.append(p)
    pkg_share = Path(__file__).resolve().parent.parent / "share" / "ardour9"
    if pkg_share.exists():
        roots.append(pkg_share)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def inspect_sound_library(
    category: str = "all",
    query: str = "",
    limit: int = 40,
) -> dict[str, Any]:
    """Browse installed SoundFonts (.sf2), WAV samples, and NAM amp models (.nam)."""
    cat_norm = (category or "all").strip().lower()
    q_norm = (query or "").strip().lower()
    max_items = max(5, min(int(limit or 40), 120))

    roots = _candidate_share_roots()
    soundfonts: list[dict[str, Any]] = []
    nam_models: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    sample_categories: dict[str, int] = {}

    user_docs = Path.home() / "Documents"
    sf2_dirs = [r / "soundfonts" for r in roots] + [user_docs / "SoundFonts"]
    sample_dirs = [r / "samples" for r in roots] + [user_docs / "Samples"]

    seen_files: set[str] = set()

    for sdir in sf2_dirs:
        if not sdir.exists():
            continue
        for p in sorted(sdir.glob("*.sf2")):
            rp = str(p.resolve())
            if rp.lower() in seen_files:
                continue
            seen_files.add(rp.lower())
            if q_norm and q_norm not in p.name.lower():
                continue
            soundfonts.append({
                "name": p.stem,
                "file": p.name,
                "path": rp,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            })

    for bdir in sample_dirs:
        if not bdir.exists():
            continue
        for p in sorted(bdir.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in {".wav", ".flac", ".aif", ".aiff", ".nam"}:
                continue
            rp = str(p.resolve())
            if rp.lower() in seen_files:
                continue
            seen_files.add(rp.lower())
            rel = p.relative_to(bdir)
            rel_str = rel.as_posix()
            top_cat = rel.parts[0] if len(rel.parts) > 1 else "root"
            sub_cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "root"

            if ext == ".nam":
                if not q_norm or q_norm in rel_str.lower():
                    nam_models.append({
                        "name": p.stem,
                        "path": rp,
                        "relative": rel_str,
                        "size_kb": round(p.stat().st_size / 1024, 1),
                    })
                continue

            sample_categories[sub_cat] = sample_categories.get(sub_cat, 0) + 1
            if q_norm and q_norm not in rel_str.lower():
                continue
            if cat_norm not in ("all", "") and cat_norm not in sub_cat.lower() and cat_norm not in rel_str.lower():
                continue
            if len(samples) < max_items:
                samples.append({
                    "name": p.stem,
                    "category": sub_cat,
                    "top_pack": top_cat,
                    "path": rp,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                })

    result: dict[str, Any] = {
        "soundfonts_count": len(soundfonts),
        "nam_models_count": len(nam_models),
        "sample_categories_total": len(sample_categories),
        "total_audio_samples": sum(sample_categories.values()),
        "category_counts": dict(sorted(sample_categories.items(), key=lambda x: (-x[1], x[0]))[:40]),
    }
    if cat_norm in ("all", "sf2", "soundfont", "soundfonts"):
        result["soundfonts"] = soundfonts[:max_items]
    if cat_norm in ("all", "nam", "amp", "guitar"):
        result["nam_models"] = nam_models[:max_items]
    if cat_norm not in ("sf2", "soundfont", "soundfonts", "nam"):
        result["samples"] = samples
    return result
