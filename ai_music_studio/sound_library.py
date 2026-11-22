"""Discovery and inspection helpers for local SoundFonts, SFZ instruments, Surge XT patches, WAV samples, and NAM models."""

from __future__ import annotations

import os
import struct
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
    build_share = Path(__file__).resolve().parent.parent / "BUILD" / "share" / "ardour9"
    if build_share.exists():
        roots.append(build_share)
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _list_sf2_presets(sf2_path: Path, query: str = "") -> list[dict[str, Any]]:
    """Parse SoundFont RIFF pdta/phdr chunk in pure Python to expose real bank/program presets."""
    try:
        data = sf2_path.read_bytes()
    except OSError:
        return []
    pos = 12
    q_low = query.strip().lower()
    presets: list[dict[str, Any]] = []
    while pos + 8 < len(data):
        cid = data[pos : pos + 4]
        sz = struct.unpack_from("<I", data, pos + 4)[0]
        if cid == b"LIST" and data[pos + 8 : pos + 12] == b"pdta":
            p2 = pos + 12
            end = min(len(data), pos + 8 + sz)
            while p2 + 8 < end:
                subid = data[p2 : p2 + 4]
                subsz = struct.unpack_from("<I", data, p2 + 4)[0]
                if subid == b"phdr":
                    for off in range(p2 + 8, min(end, p2 + 8 + subsz - 37), 38):
                        name = data[off : off + 20].split(b"\x00", 1)[0].decode("latin1", "ignore").strip()
                        prog, bank = struct.unpack_from("<HH", data, off + 20)
                        if not name or name == "EOP":
                            continue
                        if q_low and q_low not in name.lower():
                            continue
                        presets.append({"bank": int(bank), "program": int(prog), "name": name})
                    presets.sort(key=lambda x: (x["bank"], x["program"], x["name"]))
                    return presets
                p2 += 8 + subsz + (subsz & 1)
        pos += 8 + sz + (sz & 1)
    return presets


def _discover_surge_xt_dirs() -> list[Path]:
    dirs: list[Path] = []
    base = Path(__file__).resolve().parent.parent
    roots = [
        base / "share" / "surge-xt",
        base / "BUILD" / "share" / "surge-xt",
        Path.home() / "AppData" / "Local" / "Surge XT",
    ]
    for r in roots:
        for base_data in (r, r / "Surge Synth Team" / "SurgeXTData"):
            for sub in ("patches_factory", "patches_3rdparty"):
                cand = base_data / sub
                if cand.is_dir():
                    dirs.append(cand)
    seen: set[str] = set()
    unique: list[Path] = []
    for d in dirs:
        k = str(d.resolve()).lower()
        if k not in seen:
            seen.add(k)
            unique.append(d)
    # Deduplicate SurgeXTData copies between BUILD/share and AppData while keeping both factory + 3rdparty from first root
    by_rel_root: list[Path] = []
    seen_roots: set[str] = set()
    for d in unique:
        parent_key = str(d.parent.resolve()).lower()
        if not seen_roots or parent_key in seen_roots:
            seen_roots.add(parent_key)
            by_rel_root.append(d)
    return by_rel_root


def inspect_sound_library(
    category: str = "all",
    query: str = "",
    limit: int = 40,
) -> dict[str, Any]:
    """Browse installed SoundFonts (.sf2 + presets), SFZ kits, Surge XT patches, WAV samples, and NAM models."""
    cat_norm = (category or "all").strip().lower()
    q_norm = (query or "").strip().lower()
    max_items = max(5, min(int(limit or 40), 120))

    roots = _candidate_share_roots()
    soundfonts: list[dict[str, Any]] = []
    sfz_instruments: list[dict[str, Any]] = []
    nam_models: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    sample_categories: dict[str, int] = {}
    surge_patches: list[dict[str, Any]] = []
    surge_categories: dict[str, int] = {}

    user_docs = Path.home() / "Documents"
    sf2_dirs = [r / "soundfonts" for r in roots] + [user_docs / "SoundFonts"]
    sample_dirs = [r / "samples" for r in roots] + [user_docs / "Samples"]

    seen_files: set[str] = set()

    # 1. SoundFonts + internal presets
    for sdir in sf2_dirs:
        if not sdir.exists():
            continue
        for p in sorted(sdir.glob("*.sf2")):
            rp = str(p.resolve())
            if rp.lower() in seen_files:
                continue
            seen_files.add(rp.lower())
            all_presets = _list_sf2_presets(p)
            matched_presets = [
                pr for pr in all_presets
                if not q_norm or q_norm in p.name.lower() or q_norm in pr["name"].lower()
            ]
            if q_norm and q_norm not in p.name.lower() and not matched_presets:
                continue
            soundfonts.append({
                "name": p.stem,
                "file": p.name,
                "path": rp,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                "preset_count": len(all_presets),
                "presets_sample": (matched_presets or all_presets)[:24],
            })

    # 2. WAV/NAM samples + SFZ rock/bass instruments
    _sfz_preset_map = {
        "black_pearl_5pc.sfz": "AVL Black Pearl 5pc",
        "black_pearl_4pc.sfz": "AVL Black Pearl 4pc",
        "red_zeppelin_5pc.sfz": "AVL Red Zeppelin 5pc",
        "red_zeppelin_4pc.sfz": "AVL Red Zeppelin 4pc",
        "pick_bass.sfz": "Pick Bass",
        "mute_bass.sfz": "Mute Pick Bass",
        "slap_bass.sfz": "Slap Bass",
    }

    for bdir in sample_dirs:
        if not bdir.exists():
            continue
        for p in sorted(bdir.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            rp = str(p.resolve())
            if rp.lower() in seen_files:
                continue

            if ext == ".sfz":
                seen_files.add(rp.lower())
                rel_str = p.relative_to(bdir).as_posix()
                if q_norm and q_norm not in rel_str.lower() and q_norm not in p.stem.lower():
                    continue
                sfz_instruments.append({
                    "name": p.stem,
                    "sfizz_preset_label": _sfz_preset_map.get(p.name.lower(), p.stem),
                    "path": rp,
                    "relative": rel_str,
                })
                continue

            if ext not in {".wav", ".flac", ".aif", ".aiff", ".nam"}:
                continue
            seen_files.add(rp.lower())
            rel = p.relative_to(bdir)
            rel_str = rel.as_posix()
            # Exclude internal sfz wav pools from cluttering standalone one-shot list unless queried specifically
            if "sfz_instruments" in rel.parts and ext == ".wav" and cat_norm != "sfz" and not q_norm:
                continue

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
            if cat_norm not in ("all", "", "presets", "surge") and cat_norm not in sub_cat.lower() and cat_norm not in rel_str.lower():
                continue
            if len(samples) < max_items:
                samples.append({
                    "name": p.stem,
                    "category": sub_cat,
                    "top_pack": top_cat,
                    "path": rp,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                })

    # 3. Surge XT factory & 3rd-party .fxp patches
    for sroot in _discover_surge_xt_dirs():
        for fx in sorted(sroot.rglob("*.fxp")):
            rel = fx.relative_to(sroot)
            cat_name = rel.parts[0] if len(rel.parts) > 1 else "Other"
            surge_categories[cat_name] = surge_categories.get(cat_name, 0) + 1
            rel_str = f"{sroot.name}/{rel.as_posix()}"
            if q_norm and q_norm not in rel_str.lower():
                continue
            if cat_norm not in ("all", "", "surge", "presets") and cat_norm not in cat_name.lower():
                continue
            if len(surge_patches) < max_items:
                surge_patches.append({
                    "name": fx.stem,
                    "category": cat_name,
                    "relative": rel_str,
                })

    result: dict[str, Any] = {
        "soundfonts_count": len(soundfonts),
        "sfz_instruments_count": len(sfz_instruments),
        "surge_patches_count": sum(surge_categories.values()),
        "nam_models_count": len(nam_models),
        "sample_categories_total": len(sample_categories),
        "total_audio_samples": sum(sample_categories.values()),
        "category_counts": dict(sorted(sample_categories.items(), key=lambda x: (-x[1], x[0]))[:40]),
        "surge_category_counts": dict(sorted(surge_categories.items())),
    }
    if cat_norm in ("all", "sf2", "soundfont", "soundfonts", "presets", "guitar", "bass", "drums"):
        result["soundfonts"] = soundfonts[:max_items]
    if cat_norm in ("all", "sfz", "drums", "bass", "presets"):
        result["sfz_instruments"] = sfz_instruments[:max_items]
    if cat_norm in ("all", "surge", "synth", "keys", "bass", "lead", "pad", "presets") or q_norm:
        result["surge_patches"] = surge_patches[:max_items]
    if cat_norm in ("all", "nam", "amp", "guitar"):
        result["nam_models"] = nam_models[:max_items]
    if cat_norm not in ("sf2", "soundfont", "soundfonts", "nam", "sfz", "surge", "presets"):
        result["samples"] = samples
    return result
