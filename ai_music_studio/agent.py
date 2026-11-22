"""Autonomous AI Music Producer Agent orchestrating Ardour MCP + DSP + Web Search."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from json_repair import repair_json

from .audio_analyzer import (
    align_vocal_with_lyrics,
    analyze_audio_file,
    compare_mix_to_reference,
    compute_project_vocal_harmony,
)
from .llm_client import stream_chat_completion
from .mcp_client import ArdourMCPClient
from .models import (
    ChatMessage,
    PlanStep,
    PluginInstallCard,
    ProviderConfig,
    ToolCallTrace,
)
from .storage import StudioStorage
from .web_search import fetch_webpage_text, web_search


STUDIO_NATIVE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "studio_update_plan",
            "description": "Update the visible step-by-step production plan checklist in the AI Music Studio UI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done", "skipped", "error"],
                                },
                                "detail": {"type": "string"},
                            },
                            "required": ["id", "title", "status"],
                        },
                    }
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_analyze_audio",
            "description": "Run deep DSP audio analysis (BPM, Key, LUFS, True Peak, 7-band EQ, stereo, vocal VAD/phrases/breaths/pitch) on an uploaded asset ID or absolute file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Asset ID from References/Vocals/Samples/Attachments."},
                    "file_path": {"type": "string", "description": "Absolute path to an audio file (if not using asset_id)."},
                    "category": {"type": "string", "enum": ["references", "vocals", "samples"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_align_vocal_lyrics",
            "description": "Align a vocal recording (asset_id) to the lines in the Lyrics tab using local faster-whisper word timestamps + VAD, returning exact [start_sec, end_sec] per lyric line for cutting/timing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vocal_asset_id": {"type": "string", "description": "Asset ID of the vocal audio file."},
                    "language": {"type": "string", "description": "Optional ISO language code (e.g. 'ru', 'en'), empty for auto."},
                },
                "required": ["vocal_asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_compare_mix_to_reference",
            "description": "Compare a bounced mix WAV file against a reference track (by reference asset_id). Set deep=true to also run HTDemucs + CLAP stem & vocal-FX comparison (vocal level vs backing, vocal reverb tail, tone match, drums/bass balance).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mix_wav_path": {"type": "string", "description": "Absolute path to exported mix preview WAV."},
                    "reference_asset_id": {"type": "string", "description": "Asset ID of the reference track."},
                    "deep": {"type": "boolean", "description": "Run stem-level + vocal FX neural comparison (default true)."},
                },
                "required": ["mix_wav_path", "reference_asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_deep_analyze_reference",
            "description": "Run local neural audio intelligence (HTDemucs 4-stem split + AST instrument recognition + CLAP timbre/FX tags + Basic Pitch chord/bass/melody MIDI transcription + 16-step drum rhythm extraction + Reference Vocal FX Detective) on a reference asset or audio file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Asset ID from References/Samples/Vocals."},
                    "file_path": {"type": "string", "description": "Absolute path to audio file if not using asset_id."},
                    "offset_sec": {"type": "number", "description": "Optional start second for a custom window (e.g. 45.0)."},
                    "duration_sec": {"type": "number", "description": "Optional duration for custom window (default 7.8s)."},
                    "save_stems": {"type": "boolean", "description": "Save separated stem WAVs to exports/stems."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_analyze_vocal_fx",
            "description": "Analyze uploaded vocal tracks (unified musical key lock across all takes, F0 pitch range, raw compression/EQ/room tail) and compare directly with the reference vocal stem's FX recipe (reverb tail ms, delay lag ms, bandpass/saturation dirt, stereo doubling, SnapTune/EQ/Comp targets).",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_reference_comparison": {"type": "boolean", "description": "Include target Vocal FX Blueprint extracted from uploaded references (default true)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_web_search",
            "description": "Search the web for VST3/LV2/CLAP audio plugins, synths, drum machines, vocal chains, download links, and Windows installation instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_fetch_webpage",
            "description": "Fetch and read clean text from a webpage URL (e.g., official VST3 plugin page or installation guide).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_request_plugin_install",
            "description": "Present an interactive Plugin/Addon Installation Card to the user in chat explaining which VST3/CLAP/LV2 plugin to install, why it's needed, official download URL, Windows install folder, and step-by-step instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_name": {"type": "string"},
                    "format": {"type": "string", "description": "VST3, CLAP, LV2, or Addon"},
                    "price_type": {"type": "string", "description": "Free, Freemium, or Paid"},
                    "purpose": {"type": "string", "description": "Why this plugin helps achieve the user's target sound."},
                    "download_url": {"type": "string"},
                    "install_path_windows": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "alternatives": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["plugin_name", "purpose", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_inspect_sound_library",
            "description": "Inspect installed SoundFonts (.sf2 + internal presets), live SFZ rock/bass kits, Surge XT factory patches (3500+), WAV audio samples/loops, and Neural Amp Modeler (.nam) amp captures. Filter by category (all, sf2, sfz, surge, presets, nam, drums, bass, guitar, keys, synth, pad) or search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter category: all, sf2, sfz, surge, presets, nam, drums, 808, 909, bass, guitar, piano, synth, pad, etc."},
                    "query": {"type": "string", "description": "Optional filename/preset/patch substring filter."},
                    "limit": {"type": "integer", "minimum": 5, "maximum": 120},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_setup_track_instrument",
            "description": "Configure a MIDI instrument track in 1 atomic call: optionally creates a MIDI track, inserts sfizz / ACE Fluid Synth / Surge XT / TAL / Dexed, loads an exact LV2/synth preset by label (e.g. 'AVL Black Pearl 5pc', 'Pick Bass', 'FS Fender Telecaster DI', 'FluidR3 GM', 'GeneralUser GS'), sends MIDI channel + Bank/Program Change, and optionally adds a guitar/bass amp plugin (e.g. NeuralAmpModeler or BOD) right after it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string", "description": "Existing MIDI track ID. Leave empty to create a new MIDI track."},
                    "track_name": {"type": "string", "description": "Name when creating a new MIDI track (e.g. 'Rock Drums', 'Telecaster Rhythm', 'Pick Bass')."},
                    "plugin_type": {"type": "string", "enum": ["sfizz", "fluidsynth", "vst3", "lv2"], "description": "Instrument engine to insert."},
                    "plugin_search": {"type": "string", "description": "Plugin search string if plugin_type is vst3/lv2 (e.g. 'Surge XT', 'TAL-NoiseMaker', 'Dexed')."},
                    "preset_label": {"type": "string", "description": "Exact preset label to load (e.g. 'AVL Black Pearl 5pc', 'AVL Red Zeppelin 5pc', 'Pick Bass', 'Mute Pick Bass', 'FS Fender Telecaster DI', 'FluidR3 GM', 'GeneralUser GS')."},
                    "midi_channel": {"type": "integer", "description": "0-based MIDI channel (0..15; use 9 for Channel 10 GM drums). Default 0."},
                    "bank_msb": {"type": "integer", "description": "MIDI Bank MSB (0..128; for GM drums use midi_channel=9 and bank_msb=0 or 128). Default 0."},
                    "program": {"type": "integer", "description": "MIDI Program Change (0..127), e.g. 0..3 for Telecaster DI presets, 16..18 for FluidR3 Power Kits on ch 9. Default 0."},
                    "amp_plugin_search": {"type": "string", "description": "Optional follow-up amp/FX plugin to insert after instrument (e.g. 'NeuralAmpModeler', 'BOD')."},
                },
                "required": ["plugin_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_compose_humanized_midi",
            "description": "Write organic, non-robotic MIDI music into an Ardour MIDI track with micro-timing humanization, dynamic velocity groove, and optional guitar chord strumming stagger (so chords & rhythms never sound stiff/cliche).",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string", "description": "Target Ardour MIDI track ID."},
                    "name": {"type": "string", "description": "Created MIDI region name (e.g. 'Chorus Guitars', 'Verse Drums')."},
                    "role": {"type": "string", "enum": ["drums", "bass", "guitar_strum", "guitar_arp", "lead", "keys"], "description": "Musical role controlling humanization & strum curve."},
                    "start_bar": {"type": "integer", "description": "1-based start bar for the region (default 1)."},
                    "length_bars": {"type": "integer", "description": "Total length in bars (default 8)."},
                    "bpm": {"type": "number", "description": "Tempo BPM for converting ms humanization into beat fractions (fetched from session if omitted)."},
                    "midi_channel": {"type": "integer", "description": "1-based MIDI channel (10 for drums, 1 for melodic instruments)."},
                    "humanize_timing_ms": {"type": "number", "description": "Random human micro-timing jitter in milliseconds (e.g. 12). Set 0 for strict quantization."},
                    "humanize_velocity": {"type": "integer", "description": "Random human velocity variation +/- (e.g. 10)."},
                    "strum_ms": {"type": "number", "description": "Stagger delay in ms between notes occurring on the exact same beat in chords (ideal 14..24ms for guitar_strum)."},
                    "notes": {
                        "type": "array",
                        "description": "List of MIDI note events.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "bar": {"type": "integer", "description": "Absolute 1-based bar >= 1 (e.g. 1, 2, 9...)."},
                                "b": {"type": "number", "description": "Beat inside bar >= 1.0 (e.g. 1.0, 1.5, 2.25, 3.5)."},
                                "n": {"type": "integer", "description": "MIDI pitch 0..127 (e.g. Kick=36, Snare=38, ClosedHat=42, Guitar E2=40..84)."},
                                "l": {"type": "number", "description": "Note duration in beats (e.g. 0.5, 1.0, 2.0)."},
                                "v": {"type": "integer", "description": "Velocity 1..127 (default 92)."},
                            },
                            "required": ["bar", "b", "n", "l"],
                        },
                    },
                },
                "required": ["track_id", "role", "length_bars", "notes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "studio_set_plugin_parameters",
            "description": "Batch-set multiple parameters on an Ardour plugin in a single step (saves dozens of tool calls). Call ardour__plugin_get_description first to confirm exact parameterIndex and min..max range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "route_id": {"type": "string", "description": "Track or bus ID (e.g. '302' or '82' for Master)."},
                    "plugin_index": {"type": "integer", "description": "0-based plugin slot index on the track/bus."},
                    "parameters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "parameter_index": {"type": "integer"},
                                "value": {"type": "number"},
                            },
                            "required": ["parameter_index", "value"],
                        },
                    },
                },
                "required": ["route_id", "plugin_index", "parameters"],
            },
        },
    },
]


def _sanitize_schema_for_llm(node: Any) -> Any:
    """Remove strict JSON-Schema fields (like additionalProperties) rejected by some LLM gateways."""
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for k, v in node.items():
            if k == "additionalProperties":
                continue
            cleaned[k] = _sanitize_schema_for_llm(v)
        return cleaned
    if isinstance(node, list):
        return [_sanitize_schema_for_llm(x) for x in node]
    return node


def _coerce_value_type(val: Any, target_type: str) -> Any:
    if val is None or not target_type:
        return val
    if target_type == "string":
        if isinstance(val, (int, float, bool)):
            return str(val).lower() if isinstance(val, bool) else str(val)
        return str(val)
    if target_type == "integer":
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(round(val))
        if isinstance(val, str):
            try:
                return int(float(val.strip()))
            except Exception:
                return val
    if target_type == "number":
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
        if isinstance(val, str):
            try:
                return float(val.strip())
            except Exception:
                return val
    if target_type == "boolean":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        if isinstance(val, (int, float)):
            return bool(val)
    return val


def _coerce_args_to_schema(args: dict[str, Any], schema: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Coerce LLM-provided tool arguments to match the MCP tool's JSON schema cleanly."""
    if not isinstance(args, dict):
        return {}
    if not isinstance(schema, dict):
        return args
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else None
    strict = schema.get("additionalProperties") is False
    out: dict[str, Any] = {}
    for k, v in args.items():
        if props is not None and strict and k not in props:
            # Drop hallucinated metadata keys (e.g. 'reason') when MCP forbids additionalProperties
            continue
        if props and k in props and isinstance(props[k], dict):
            t = str(props[k].get("type", ""))
            v = _coerce_value_type(v, t)
        out[k] = v
    return out


_LOGGER = logging.getLogger("ai_music_studio")


def _sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _safe_parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        try:
            parsed = json.loads(repair_json(str(raw)))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


class AIProducerAgent:
    """Orchestrates LLM reasoning, DSP tools, Web Search, and Ardour MCP calls."""

    def __init__(self, storage: StudioStorage, mcp: ArdourMCPClient) -> None:
        self.storage = storage
        self.mcp = mcp
        self._cancel_event = asyncio.Event()
        self._running = False
        self._region_id_alias: dict[str, str] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._cancel_event.set()

    def _build_system_prompt(self, mode: str, session_ctx: dict[str, Any]) -> str:
        skills = [s for s in self.storage.list_skills() if s.enabled]
        refs = self.storage.list_assets("references")
        vocals = self.storage.list_assets("vocals")
        samples = self.storage.list_assets("samples")
        lyrics = self.storage.get_lyrics()
        vocal_harmony = compute_project_vocal_harmony(vocals) if vocals else {}

        def compact_analysis(a: dict[str, Any]) -> dict[str, Any]:
            # Strip dense waveform_peaks array so we don't waste LLM tokens
            out: dict[str, Any] = {}
            for k, v in a.items():
                if k == "waveform_peaks":
                    continue
                if k == "neural_blueprint" and isinstance(v, dict):
                    nb = dict(v)
                    wins = []
                    for w in nb.get("windows", [])[:3]:
                        wc = dict(w)
                        for mk in ("bass_midi", "chords_and_melody_midi"):
                            if isinstance(wc.get(mk), dict):
                                mcopy = dict(wc[mk])
                                mcopy["notes_sample"] = mcopy.get("notes_sample", [])[:12]
                                wc[mk] = mcopy
                        wins.append(wc)
                    nb["windows"] = wins
                    out[k] = nb
                else:
                    out[k] = v
            return out

        refs_block = [
            {
                "id": r.id,
                "filename": r.filename,
                "path": r.stored_path,
                "role": r.role or "general_reference",
                "notes": r.notes,
                "dsp_analysis": compact_analysis(r.analysis),
            }
            for r in refs
        ]
        vocals_block = [
            {
                "id": v.id,
                "filename": v.filename,
                "path": v.stored_path,
                "role": v.role or "lead_vocal",
                "notes": v.notes,
                "dsp_analysis": compact_analysis(v.analysis),
                "lyrics_alignment": {
                    "method": v.alignment.get("method"),
                    "aligned_lines": v.alignment.get("aligned_lines", []),
                }
                if v.alignment
                else None,
            }
            for v in vocals
        ]
        samples_block = [
            {
                "id": s.id,
                "filename": s.filename,
                "path": s.stored_path,
                "role": s.role or "extra_sound",
                "notes": s.notes,
                "dsp_analysis": compact_analysis(s.analysis),
            }
            for s in samples
        ]

        skills_text = "\n\n".join(
            f"### Skill: {s.name} ({s.category})\n{s.content}" for s in skills
        ) or "(Нет активных скиллов)"

        mode_instruction = (
            "РЕЖИМ: ПОЛНЫЙ АВТОМАТ. Самостоятельно планируй и выполняй весь цикл производства песни "
            "(разбор нейронного паспорта референсов и вокальных FX -> единая тональность/темп/маркеры -> "
            "подбор живых инструментов sfizz/DI+NAM/Surge -> гуманизированный MIDI без клише -> импорт/нарезка/тайминг вокала -> "
            "профессиональная цепь обработки голоса под стиль референса -> сведение и нейросверка экспорта со стемами референса). "
            "Останавливайся только если нужно попросить пользователя установить внешний плагин."
            if mode == "autonomous"
            else "РЕЖИМ: ПОШАГОВЫЙ КОНТРОЛЬ. Сначала обнови план через `studio_update_plan`, выполни анализ и "
            "согласуй крупный этап с пользователем перед массовыми изменениями в сессии."
        )

        return f"""Ты — профессиональный AI-музыкальный продюсер, композитор-аранжировщик и звукорежиссёр, встроенный напрямую в DAW **Ardour** (AI Music Studio).
Отвечай пользователю на языке его запроса (по умолчанию — на русском), чётко, по делу и дружелюбно.

{mode_instruction}

## Главные принципы работы (НИКАКИХ КЛИШЕ!)
1. **Сначала «услышь» референс и вокал нейросетями:**
   - У тебя есть локальные нейро-модели (`HTDemucs`, `AST`, `CLAP`, `Basic Pitch`). Проверь блок `neural_blueprint` в референсах и `unified_vocal_harmony`, или вызови **`studio_deep_analyze_reference`** и **`studio_analyze_vocal_fx`**.
   - **Пиши музыку ПОХОЖУЮ на референсы, а не шаблоны:** бери реальные инструменты (`clap_timbre`, `ast_instruments`), реальную сетку ударных (`drum_pattern.kick_steps_16 / snare_steps_16 / hat_steps_16`), аккордовое движение (`detected_chord_flow` / `chords_and_melody_midi`) и согласуй их с **единой тональностью и диапазоном нот вокала пользователя (`Согласованная тональность вокала`)**.
   - **Жёсткий запрет на дежурные заглушки:** НИКОГДА не ставь «грустное пианино (Lonely Piano)», `alien-monolith-pad` или случайные `808`, если их нет в тембре референса!
2. **Выбор живых инструментов и быстрая настройка:**
   - Изучай библиотеку через **`studio_inspect_sound_library`** (включая `sfz`, `sf2`, `surge`, `nam`, `drums`, `guitar`, `bass`) и ставь инструменты в 1 вызов через **`studio_setup_track_instrument`**:
     - Рок/панк/инди барабаны -> `plugin_type="sfizz"`, `preset_label="AVL Black Pearl 5pc"` (или `"AVL Red Zeppelin 5pc"`), либо `FluidR3 GM` `midi_channel=9` `program=16..18`.
     - Живой бас -> `plugin_type="sfizz"`, `preset_label="Pick Bass"` / `"Mute Pick Bass"` (+ при нужде `amp_plugin_search="BOD"`).
     - Живые гитары -> `plugin_type="fluidsynth"`, `preset_label="FS Fender Telecaster DI"` (`program=0` Clean, `1` Muted, `2` Muted 5ths & Solo), **и сразу следом `amp_plugin_search="NeuralAmpModeler"`**!
   - Для создания MIDI-партий используй **`studio_compose_humanized_midi`** (с `strum_ms=14..24` для гитарных аккордов, микротаймингом `humanize_timing_ms=10..18`, живым разбросом velocity и брейками на смене секций).
3. **Понимание и обработка вокала (как в референсе + чисто по паузам):**
   - Используй рецепт из **`studio_analyze_vocal_fx`** (`reverb_type`, `reverb_tail_ms`, `delay_lag_ms`, `eq_character`, `compression_style`, `stereo_mode`, `tuning_style`, `vocal_to_instrumental_db`).
   - Вырезай паузы через `ardour__region_split` + `ardour__region_move*` + `ardour__region_set_fades` (`audio_import_file` автоматически возвращает живой `regionId` плейлиста!).
   - На голос ставь актуальную цепь: **`SnapTune.vst3`** (питч/автотюн в тональность трека) -> **`TDR Nova.vst3`** (HPF, уборка мутности 200-400 Гц, де-эссер или lo-fi bandpass) -> **`TDR Molotok.vst3`** / `Kotelnikov` -> **`Airwindows Consolidated.vst3`** (грязь/лента/перегруз по стилю) -> посылы 100% Wet на **`Dragonfly*Reverb`** / `TAL-Reverb-4`, **`a-delay`**, и **`TAL-Chorus-LX`** при широких даблах.
4. **Не крути параметры вслепую и проверяй микс со стемами:**
   - Для любого нового плагина вызови `ardour__plugin_get_description` и выставляй все параметры за 1 шаг через **`studio_set_plugin_parameters`**.
   - **Обязательная финальная проверка:** после `ardour__session_export_preview` всегда прогоняй экспорт через **`studio_compare_mix_to_reference`** (`deep=true`). Она сравнит **отдельно твой вокал с вокалом референса и твои инструменты с инструментами референса**. Если вокал тихий, слишком сухой или микс молчит — поправь фейдеры/цепи и переэкспортируй!
5. **Безопасность:** если Ardour MCP не подключён (`connected: false`), честно сообщи об этом и при этом подготовь полный нейронный анализ референсов, вокальных FX и аранжировки.

## Критические правила Ardour Lua и пресетов
- Инструмент `studio_setup_track_instrument` сам грузит пресеты `sfizz` / `ACE Fluid Synth` и отправляет Bank MSB/LSB + Program Change. Если пишешь Lua вручную: никогда не гадай несуществующие методы (`Session:region_by_id`, `to_midi_track`, `Evoral.PatchChange`, `plug:get_presets()`, `ins:load_preset()` — правильно `plug:get_info():get_presets(false)` и `plug:load_preset(ps)`).

## Активные Скиллы пользователя
{skills_text}

## Контекст проекта (из вкладок AI Music Studio)
### Согласованная тональность и вокальная гармония проекта
```json
{json.dumps(vocal_harmony, ensure_ascii=False, indent=2)}
```

### Вкладка «Референсы (Примеры звучания + Нейронный паспорт)»
```json
{json.dumps(refs_block, ensure_ascii=False, indent=2)}
```

### Вкладка «Вокал (Голос пользователя)»
```json
{json.dumps(vocals_block, ensure_ascii=False, indent=2)}
```

### Вкладка «Дополнительные звуки / Сэмплы»
```json
{json.dumps(samples_block, ensure_ascii=False, indent=2)}
```

### Вкладка «Текст песни (Lyrics)»
Название: {lyrics.title or "(без названия)"}
Секции и текст:
```
{lyrics.raw_text or "(текст не добавлен)"}
```

## Живое состояние Ardour DAW
```json
{json.dumps(session_ctx, ensure_ascii=False, indent=2)}
```
"""

    async def _collect_tools(self) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
        """Combine studio tools + sanitized Ardour MCP tool schemas."""
        tools: list[dict[str, Any]] = list(STUDIO_NATIVE_TOOLS)
        ardour_map: dict[str, str] = {}
        raw_schemas: dict[str, dict[str, Any]] = {}

        try:
            mcp_tools = await self.mcp.list_tools()
            for t in mcp_tools:
                orig_name = str(t.get("name", ""))
                if not orig_name or orig_name == "hello_world":
                    continue
                fn_name = f"ardour__{_sanitize_tool_name(orig_name)}"
                ardour_map[fn_name] = orig_name
                raw_schema = copy.deepcopy(t.get("inputSchema") or {"type": "object", "properties": {}})
                raw_schemas[fn_name] = raw_schema
                tools.append({
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "description": f"[Ardour DAW] {t.get('description') or t.get('title') or orig_name}",
                        "parameters": _sanitize_schema_for_llm(raw_schema),
                    },
                })
        except Exception as exc:
            _LOGGER.warning("MCP list_tools failed during _collect_tools: %s", exc)

        return tools, ardour_map, raw_schemas

    def _mcp_tool_name(self, ardour_map: dict[str, str], fn_key: str, default: str) -> str:
        return ardour_map.get(fn_key) or default

    async def _resolve_playlist_region_id(
        self,
        ardour_map: dict[str, str],
        orig_region_id: str,
        track_id_hint: str = "",
        region_name_hint: str = "",
        pos_hint: Optional[int] = None,
    ) -> Optional[str]:
        """Resolve an orphaned whole-file regionId to its live cloned playlist regionId."""
        if not orig_region_id:
            return None
        if orig_region_id in self._region_id_alias:
            return self._region_id_alias[orig_region_id]

        # Inspect orphaned region if hints are missing
        if not region_name_hint or pos_hint is None:
            try:
                g_tool = self._mcp_tool_name(ardour_map, "ardour__region_get_info", "region/get_info")
                info_res = await self.mcp.call_tool(g_tool, {"regionId": orig_region_id})
                st = info_res.get("structured", {})
                if st.get("playlistId"):
                    self._region_id_alias[orig_region_id] = orig_region_id
                    return orig_region_id
                region_name_hint = region_name_hint or str(st.get("name", ""))
                if pos_hint is None and st.get("startSample") is not None:
                    pos_hint = int(st["startSample"])
                track_id_hint = track_id_hint or str(st.get("trackId", ""))
            except Exception:
                pass

        t_get = self._mcp_tool_name(ardour_map, "ardour__track_get_regions", "track/get_regions")
        track_ids: list[str] = [track_id_hint] if track_id_hint else []
        if not track_ids:
            try:
                tl_tool = self._mcp_tool_name(ardour_map, "ardour__tracks_list", "tracks/list")
                tl_res = await self.mcp.call_tool(tl_tool, {})
                routes = tl_res.get("structured", {}).get("routes") or tl_res.get("structured", {}).get("tracks") or []
                track_ids = [str(rt.get("id", "")) for rt in routes if rt.get("id")]
            except Exception:
                pass

        best_id: Optional[str] = None
        best_score = -10**12
        for tid in track_ids:
            try:
                r_res = await self.mcp.call_tool(t_get, {"id": tid})
                regs = r_res.get("structured", {}).get("regions") or []
            except Exception:
                continue
            for reg in regs:
                if not reg.get("playlistId"):
                    continue
                rid = str(reg.get("regionId", ""))
                if not rid:
                    continue
                rname = str(reg.get("name", ""))
                rpos = int(reg.get("startSample", 0) or 0)
                score = 0
                if region_name_hint and rname == region_name_hint:
                    score += 1000
                elif region_name_hint and (rname.startswith(region_name_hint) or region_name_hint.startswith(rname)):
                    score += 500
                if pos_hint is not None:
                    score -= abs(rpos - pos_hint) // 100
                if score > best_score:
                    best_score = score
                    best_id = rid
        if best_id and (region_name_hint or pos_hint is not None):
            self._region_id_alias[orig_region_id] = best_id
            return best_id
        return None

    async def _execute_tool(
        self,
        fn_name: str,
        args: dict[str, Any],
        ardour_map: dict[str, str],
        raw_schemas: dict[str, dict[str, Any]],
        plugin_cards_out: list[PluginInstallCard],
        plan_out: list[PlanStep],
        emit: Any,
    ) -> tuple[bool, Any]:
        settings = self.storage.get_settings()

        if fn_name == "studio_update_plan":
            raw_steps = args.get("steps") or []
            steps = [PlanStep(**s) for s in raw_steps if isinstance(s, dict) and s.get("title")]
            self.storage.save_plan(steps)
            plan_out[:] = steps
            await emit({"type": "plan", "steps": [s.model_dump() for s in steps]})
            return True, {"updated": len(steps)}

        if fn_name == "studio_analyze_audio":
            asset_id = args.get("asset_id")
            file_path = args.get("file_path")
            category = args.get("category", "references")
            asset = self.storage.get_asset(asset_id) if asset_id else None
            if asset:
                file_path = asset.stored_path
                category = asset.category
            if not file_path or not Path(file_path).exists():
                return False, {"error": f"Audio file not found: {file_path}"}
            analysis = await asyncio.to_thread(analyze_audio_file, file_path, category)
            if asset:
                existing_nb = (asset.analysis or {}).get("neural_blueprint")
                if existing_nb:
                    analysis["neural_blueprint"] = existing_nb
                asset.analysis = analysis
                self.storage.update_asset(asset)
                await emit({"type": "assets_updated"})
            compact = {k: v for k, v in analysis.items() if k != "waveform_peaks"}
            return True, compact

        if fn_name == "studio_deep_analyze_reference":
            from .ai_audio_models import deep_analyze_reference

            asset_id = args.get("asset_id")
            file_path = args.get("file_path")
            asset = self.storage.get_asset(asset_id) if asset_id else None
            if asset:
                file_path = asset.stored_path
            if not file_path or not Path(file_path).exists():
                return False, {"error": f"Audio file not found: {file_path}"}
            offset_sec = args.get("offset_sec")
            duration_sec = float(args.get("duration_sec", 7.8) or 7.8)
            save_stems_dir = None
            if args.get("save_stems"):
                save_stems_dir = self.storage.data_dir / "exports" / "stems" / Path(file_path).stem
            bpm_hint = (asset.analysis or {}).get("tempo", {}).get("bpm") if asset else None
            sections = (asset.analysis or {}).get("sections") if asset else None

            await emit({"type": "status", "text": f"Запускаю нейроразбор {Path(file_path).name} (HTDemucs + AST + CLAP + Basic Pitch)..."})
            bp = await asyncio.to_thread(
                deep_analyze_reference,
                file_path,
                bpm_hint,
                sections,
                save_stems_dir,
                float(offset_sec) if offset_sec is not None else None,
                duration_sec,
            )
            if asset and offset_sec is None and not bp.get("error"):
                an = dict(asset.analysis or {})
                an["neural_blueprint"] = bp
                asset.analysis = an
                self.storage.update_asset(asset)
                await emit({"type": "assets_updated"})
            return "error" not in bp, bp

        if fn_name == "studio_analyze_vocal_fx":
            from .ai_audio_models import analyze_project_vocals_and_harmony

            vocals = self.storage.list_assets("vocals")
            refs = self.storage.list_assets("references") if args.get("include_reference_comparison", True) else []
            if not vocals and not refs:
                return False, {"error": "No vocals or references uploaded yet."}
            await emit({"type": "status", "text": "Анализирую вокальные эффекты, питч-контур и гармонию проекта..."})
            res = await asyncio.to_thread(analyze_project_vocals_and_harmony, vocals, refs)
            return True, res

        if fn_name == "studio_align_vocal_lyrics":
            vocal_id = args.get("vocal_asset_id", "")
            asset = self.storage.get_asset(vocal_id)
            if not asset:
                return False, {"error": f"Vocal asset '{vocal_id}' not found."}
            lyrics = self.storage.get_lyrics()
            vad_phrases = asset.analysis.get("vocal", {}).get("phrases") if asset.analysis else None
            lang = args.get("language") or settings.whisper_language
            alignment = await asyncio.to_thread(
                align_vocal_with_lyrics,
                asset.stored_path,
                lyrics.sections,
                vad_phrases,
                settings.whisper_enabled,
                settings.whisper_model,
                lang,
            )
            asset.alignment = alignment
            self.storage.update_asset(asset)
            self.storage.save_lyrics(
                lyrics.title,
                lyrics.raw_text,
                aligned_vocal_id=asset.id,
                aligned_lines=alignment.get("aligned_lines", []),
            )
            await emit({"type": "lyrics_updated"})
            return True, {
                "method": alignment.get("method"),
                "detected_language": alignment.get("detected_language"),
                "word_count": alignment.get("word_count"),
                "aligned_lines": alignment.get("aligned_lines", []),
                "full_transcript": alignment.get("full_transcript", "")[:1200],
            }

        if fn_name == "studio_compare_mix_to_reference":
            mix_path = args.get("mix_wav_path", "")
            ref_id = args.get("reference_asset_id", "")
            ref = self.storage.get_asset(ref_id)
            if not ref:
                return False, {"error": f"Reference asset '{ref_id}' not found."}
            if not mix_path or not Path(mix_path).exists():
                return False, {"error": f"Mix WAV not found: {mix_path}"}
            use_deep = bool(args.get("deep", True))
            if use_deep:
                from .ai_audio_models import compare_mix_neural
                await emit({"type": "status", "text": "Разделяю экспорт на стемы и сверяю вокал/инструменты с референсом..."})
                report = await asyncio.to_thread(compare_mix_neural, mix_path, ref.stored_path)
            else:
                report = await asyncio.to_thread(compare_mix_to_reference, mix_path, ref.stored_path)
            return True, report

        if fn_name == "studio_web_search":
            res = await web_search(
                str(args.get("query", "")),
                settings.web_search,
                max_results=args.get("max_results"),
            )
            return bool(res.get("ok")), res

        if fn_name == "studio_fetch_webpage":
            res = await fetch_webpage_text(str(args.get("url", "")))
            return bool(res.get("ok")), res

        if fn_name == "studio_inspect_sound_library":
            from .sound_library import inspect_sound_library
            res = inspect_sound_library(
                category=str(args.get("category", "all")),
                query=str(args.get("query", "")),
                limit=int(args.get("limit", 40) or 40),
            )
            return True, res

        if fn_name == "studio_setup_track_instrument":
            try:
                ptype = str(args.get("plugin_type", "sfizz")).lower().strip()
                track_id = str(args.get("track_id", "")).strip()
                track_name = str(args.get("track_name") or args.get("preset_label") or "Instrument").strip()
                preset_label = str(args.get("preset_label", "")).strip()
                plugin_search = str(args.get("plugin_search", "")).strip()
                midi_ch = max(0, min(15, int(args.get("midi_channel", 0) or 0)))
                bank_msb = int(args.get("bank_msb", 0) or 0)
                program = max(0, min(127, int(args.get("program", 0) or 0)))
                amp_search = str(args.get("amp_plugin_search", "")).strip()

                t_add_tool = self._mcp_tool_name(ardour_map, "ardour__tracks_add", "tracks/add")
                p_list_tool = self._mcp_tool_name(ardour_map, "ardour__plugin_list_available", "plugin/list_available")
                p_add_tool = self._mcp_tool_name(ardour_map, "ardour__plugin_add_by_plugin_id", self._mcp_tool_name(ardour_map, "ardour__plugin_add", "plugin/add"))
                lua_tool = self._mcp_tool_name(ardour_map, "ardour__lua_run_script", "lua/run_script")

                # 1. Create MIDI track if track_id not specified
                if not track_id:
                    tr_res = await self.mcp.call_tool(
                        t_add_tool,
                        {"name": track_name, "type": "midi", "count": 1, "inputChannels": 0, "outputChannels": 2, "insert": "end"},
                    )
                    st = tr_res.get("structured", {})
                    track_id = str(
                        st.get("id")
                        or (st.get("tracks") or [{}])[0].get("id")
                        or (st.get("routes") or [{}])[0].get("id")
                        or st.get("track", {}).get("id")
                        or ""
                    )
                    if not track_id:
                        tl_res = await self.mcp.call_tool(self._mcp_tool_name(ardour_map, "ardour__tracks_list", "tracks/list"), {})
                        for rt in reversed(tl_res.get("structured", {}).get("routes") or []):
                            if str(rt.get("name", "")).startswith(track_name):
                                track_id = str(rt.get("id", ""))
                                break
                    if not track_id:
                        return False, {"error": f"Failed to resolve created MIDI track ID from response: {st}"}

                # 2. Lookup exact pluginId
                search_term = plugin_search
                if ptype == "sfizz":
                    search_term = search_term or "sfizz"
                elif ptype == "fluidsynth":
                    search_term = search_term or "a-fluidsynth"
                pl_res = await self.mcp.call_tool(p_list_tool, {"search": search_term, "includeHidden": False, "includeInternal": False})
                candidates = pl_res.get("structured", {}).get("plugins") or []
                chosen = None
                for c in candidates:
                    pid = str(c.get("pluginId", ""))
                    if ptype == "sfizz" and "sfizz" in pid.lower() and "multi" not in pid.lower():
                        chosen = pid
                        break
                    if ptype == "fluidsynth" and "a-fluidsynth" in pid.lower():
                        chosen = pid
                        break
                if not chosen and candidates:
                    chosen = str(candidates[0].get("pluginId", ""))
                if not chosen:
                    return False, {"error": f"Plugin not found for type={ptype}, search={search_term}"}

                # 3. Insert instrument plugin
                add_res = await self.mcp.call_tool(p_add_tool, {"id": track_id, "pluginId": chosen, "enabled": True})
                add_st = add_res.get("structured", {})
                p_idx = int(add_st.get("pluginIndex", 0))

                # 4. Load LV2/plugin preset + send Bank/Program Change via verified Lua
                cc0 = 0 if (midi_ch == 9 and bank_msb == 128) else max(0, min(127, bank_msb))
                status_cc = 0xB0 + midi_ch
                status_pc = 0xC0 + midi_ch
                preset_esc = preset_label.replace("\\", "\\\\").replace('"', '\\"')
                tid_esc = track_id.replace('"', '\\"')
                lua_lines = [
                    f'local r = Session:route_by_id(PBD.ID("{tid_esc}"))',
                    f'local ins = r:nth_plugin({p_idx}):to_plugininsert()',
                    'local plug = ins:plugin(0)',
                ]
                if preset_label:
                    lua_lines.extend([
                        f'local ps = plug:preset_by_label("{preset_esc}")',
                        'if ps and ps.valid then plug:load_preset(ps) end',
                        'ARDOUR.LuaAPI.usleep(600000)',
                    ])
                if ptype == "fluidsynth" or program > 0 or bank_msb > 0 or midi_ch > 0:
                    lua_lines.extend([
                        'local parser = ARDOUR.RawMidiParser()',
                        f'for _, b in ipairs({{{status_cc}, 0x00, {cc0}, {status_cc}, 0x20, 0x00, {status_pc}, {program}}}) do',
                        '  if parser:process_byte(b) then',
                        '    ins:write_immediate_event(Evoral.EventType.MIDI_EVENT, parser:buffer_size(), parser:midi_buffer())',
                        '  end',
                        'end',
                    ])
                await self.mcp.call_tool(lua_tool, {"script": "\n".join(lua_lines)})

                # 5. Optional amp/FX follow-up plugin (e.g. NeuralAmpModeler after Fender Telecaster DI)
                amp_info = None
                if amp_search:
                    ap_res = await self.mcp.call_tool(p_list_tool, {"search": amp_search, "includeHidden": False, "includeInternal": False})
                    acands = ap_res.get("structured", {}).get("plugins") or []
                    if acands:
                        amp_pid = str(acands[0].get("pluginId", ""))
                        a_add = await self.mcp.call_tool(p_add_tool, {"id": track_id, "pluginId": amp_pid, "enabled": True})
                        amp_info = {"pluginId": amp_pid, "pluginIndex": a_add.get("structured", {}).get("pluginIndex")}

                return True, {
                    "track_id": track_id,
                    "track_name": track_name,
                    "plugin_id": chosen,
                    "plugin_index": p_idx,
                    "preset_label": preset_label,
                    "midi_channel_0_based": midi_ch,
                    "bank_msb": cc0,
                    "program": program,
                    "amp_plugin": amp_info,
                }
            except Exception as exc:
                return False, {"error": str(exc)}

        if fn_name == "studio_compose_humanized_midi":
            try:
                track_id = str(args.get("track_id", "")).strip()
                if not track_id:
                    return False, {"error": "Provide track_id."}
                role = str(args.get("role", "guitar_strum")).strip().lower()
                region_name = str(args.get("name") or f"Part_{role}").strip()
                start_bar = max(1, int(args.get("start_bar", 1) or 1))
                length_bars = max(1, int(args.get("length_bars", 8) or 8))
                raw_notes = args.get("notes") or []
                if not raw_notes:
                    return False, {"error": "notes array is empty."}

                bpm = float(args.get("bpm") or 0.0)
                if bpm <= 10:
                    try:
                        s_info = await self.mcp.call_tool(self._mcp_tool_name(ardour_map, "ardour__session_get_info", "session/get_info"), {})
                        bpm = float(s_info.get("structured", {}).get("tempoBpm") or 120.0)
                    except Exception:
                        bpm = 120.0

                is_drum = role == "drums"
                midi_ch = int(args.get("midi_channel") or (10 if is_drum else 1))
                timing_ms = float(args.get("humanize_timing_ms", 12.0 if not is_drum else 9.0) or 0.0)
                vel_jit = int(args.get("humanize_velocity", 10) or 0)
                strum_ms = float(args.get("strum_ms", 16.0 if role == "guitar_strum" else 0.0) or 0.0)

                beats_per_bar = 4.0
                ms_to_beats = bpm / 60000.0
                rng = random.Random(42)

                # Group notes happening on same (bar, rounded beat) for chord strumming stagger
                normalized_input: list[dict[str, Any]] = []
                for idx, ev in enumerate(raw_notes):
                    if not isinstance(ev, dict) or "n" not in ev:
                        continue
                    bar = max(1, int(ev.get("bar", start_bar)))
                    beat = float(ev.get("b", 1.0))
                    pitch = max(0, min(127, int(ev.get("n"))))
                    dur = max(0.05 if not is_drum else 0.25, float(ev.get("l", 0.5)))
                    vel = max(1, min(127, int(ev.get("v", 92))))
                    normalized_input.append({"bar": bar, "b": beat, "n": pitch, "l": dur, "v": vel, "idx": idx})

                strum_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
                for ev in normalized_input:
                    key = (ev["bar"], int(round(ev["b"] * 100)))
                    strum_groups.setdefault(key, []).append(ev)

                events_out: list[dict[str, Any]] = []
                for key, grp in strum_groups.items():
                    grp.sort(key=lambda x: x["n"])
                    for order_i, ev in enumerate(grp):
                        delta_ms = 0.0
                        if strum_ms > 0 and len(grp) > 1 and not is_drum:
                            delta_ms += order_i * strum_ms
                        if timing_ms > 0:
                            delta_ms += rng.uniform(-timing_ms, timing_ms)

                        # Keep notes anchored strictly at or after bar 1, beat 1.0
                        abs_beat = (ev["bar"] - 1) * beats_per_bar + (ev["b"] - 1.0) + (delta_ms * ms_to_beats)
                        abs_beat = max((start_bar - 1) * beats_per_bar, abs_beat)
                        new_bar = int(math.floor(abs_beat / beats_per_bar)) + 1
                        new_b = round((abs_beat - (new_bar - 1) * beats_per_bar) + 1.0, 4)

                        # Beat groove accent: slight boost on beats 1 & 3, subtle dip on syncopations
                        groove_boost = 4 if abs((new_b - 1.0) - round(new_b - 1.0)) < 0.08 and int(round(new_b)) in (1, 3) else -3
                        new_vel = ev["v"] + groove_boost + (rng.randint(-vel_jit, vel_jit) if vel_jit > 0 else 0)
                        new_vel = max(20, min(127, new_vel))

                        events_out.append({
                            "bar": new_bar,
                            "b": max(1.0, new_b),
                            "n": ev["n"],
                            "l": round(ev["l"], 4),
                            "v": new_vel,
                        })

                events_out.sort(key=lambda x: (x["bar"], x["b"], x["n"]))
                max_ev_bar = max((e["bar"] for e in events_out), default=start_bar)
                end_bar = max(start_bar + length_bars, max_ev_bar + 1)

                m_tool = self._mcp_tool_name(
                    ardour_map,
                    "ardour__midi_note_import_json_to_new_region_bbt",
                    self._mcp_tool_name(ardour_map, "ardour__midi_note_import_json", "midi_note/import_json"),
                )
                payload = {
                    "trackId": track_id,
                    "name": region_name,
                    "startBar": start_bar,
                    "startBeat": 1.0,
                    "endBar": end_bar,
                    "endBeat": 1.0,
                    "midi": {
                        "is_drum_mode": is_drum,
                        "channel": midi_ch,
                        "ticks_per_quarter": 480,
                        "time_signature": "4/4",
                        "midi_events": events_out,
                    },
                }
                res = await self.mcp.call_tool(m_tool, payload)
                st = res.get("structured", res)
                return True, {"imported_notes": len(events_out), "bpm_used": bpm, "role": role, "ardour_result": st}
            except Exception as exc:
                return False, {"error": str(exc)}

        if fn_name == "studio_set_plugin_parameters":
            route_id = str(args.get("route_id", "")).strip()
            try:
                plugin_idx = int(args.get("plugin_index", 0))
            except Exception:
                return False, {"error": f"Invalid plugin_index: {args.get('plugin_index')}"}
            params_list = args.get("parameters") or []
            if not route_id or not isinstance(params_list, list) or not params_list:
                return False, {"error": "Provide route_id, plugin_index, and non-empty parameters array."}
            results: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for p_item in params_list:
                if not isinstance(p_item, dict):
                    continue
                try:
                    p_idx = int(p_item.get("parameter_index"))
                    val = float(p_item.get("value"))
                    r = await self.mcp.call_tool(
                        "plugin_set_parameter_by_index_value",
                        {"id": route_id, "pluginIndex": plugin_idx, "parameterIndex": p_idx, "value": val},
                    )
                    st = r.get("structured", {})
                    results.append({"parameterIndex": p_idx, "value": val, "label": st.get("label", ""), "ok": True})
                except Exception as exc:
                    errors.append({"parameterIndex": p_item.get("parameter_index"), "error": str(exc)})
            return len(errors) == 0, {
                "route_id": route_id,
                "plugin_index": plugin_idx,
                "applied": len(results),
                "failed": len(errors),
                "results": results[:30],
                "errors": errors[:10],
            }

        if fn_name == "studio_request_plugin_install":
            card = PluginInstallCard(
                id=f"plug_{uuid.uuid4().hex[:8]}",
                plugin_name=str(args.get("plugin_name", "Unknown Plugin")),
                format=str(args.get("format", "VST3")),
                price_type=str(args.get("price_type", "Free")),
                purpose=str(args.get("purpose", "")),
                download_url=str(args.get("download_url", "")),
                install_path_windows=str(args.get("install_path_windows", r"C:\Users\Admin\Desktop\FL Radio\BUILD\lib\ardour9\VST3")),
                steps=[str(x) for x in (args.get("steps") or [])],
                alternatives=[str(x) for x in (args.get("alternatives") or [])],
                status="pending",
            )
            plugin_cards_out.append(card)
            await emit({"type": "plugin_card", "card": card.model_dump()})
            return True, {
                "card_displayed": True,
                "card_id": card.id,
                "note": "Plugin install card shown to user. Pause here and let the user install & click Rescan, or ask if they prefer built-in plugins.",
            }

        if fn_name in ardour_map:
            mcp_name = ardour_map[fn_name]
            coerced = _coerce_args_to_schema(args, raw_schemas.get(fn_name))
            # Remap known orphaned whole_file regionIds before dispatching region/* calls
            if "regionId" in coerced and str(coerced["regionId"]) in self._region_id_alias:
                coerced["regionId"] = self._region_id_alias[str(coerced["regionId"])]
            try:
                res = await self.mcp.call_tool(mcp_name, coerced)
                st = res.get("structured", res)
                # Fix whole_file clone mismatch immediately after audio/import_file
                if (mcp_name == "audio/import_file" or "audio_import" in fn_name) and isinstance(st, dict) and st.get("regionId"):
                    orig_rid = str(st["regionId"])
                    real_rid = await self._resolve_playlist_region_id(
                        ardour_map,
                        orig_rid,
                        track_id_hint=str(st.get("trackId", "")),
                        region_name_hint=str(st.get("regionName", "")),
                        pos_hint=int(st["positionSample"]) if st.get("positionSample") is not None else None,
                    )
                    if real_rid and real_rid != orig_rid:
                        st = dict(st)
                        st["regionId"] = real_rid
                        st["resolvedFromWholeFileId"] = orig_rid
                return True, st
            except Exception as exc:
                err_txt = str(exc)
                # Auto-recover older unmapped regionId throwing "Region has no playlist"
                if "Region has no playlist" in err_txt and coerced.get("regionId"):
                    orig_rid = str(coerced["regionId"])
                    real_rid = await self._resolve_playlist_region_id(ardour_map, orig_rid)
                    if real_rid and real_rid != orig_rid:
                        coerced_retry = dict(coerced)
                        coerced_retry["regionId"] = real_rid
                        try:
                            res2 = await self.mcp.call_tool(mcp_name, coerced_retry)
                            return True, res2.get("structured", res2)
                        except Exception as exc2:
                            return False, {"error": str(exc2)}
                return False, {"error": err_txt}

        return False, {"error": f"Unknown tool: {fn_name}"}

    async def run_chat_stream(
        self,
        user_message: str,
        provider_id: Optional[str] = None,
        attachment_ids: Optional[list[str]] = None,
        mode_override: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute a full agent turn, yielding real-time SSE event dicts."""
        self._cancel_event.clear()
        self._running = True
        queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

        async def emit(ev: dict[str, Any]) -> None:
            await queue.put(ev)

        async def worker() -> None:
            try:
                settings = self.storage.get_settings()
                self.mcp.set_url(settings.ardour_mcp_url)
                mode = mode_override or settings.autonomy_mode

                provider: Optional[ProviderConfig] = self.storage.get_provider(provider_id)
                if not provider:
                    await emit({"type": "error", "message": "Не настроен ни один LLM-провайдер. Откройте вкладку «Провайдеры (API)»."})
                    return
                if not provider.api_key and "127.0.0.1" not in provider.base_url and "localhost" not in provider.base_url:
                    await emit({"type": "error", "message": f"У провайдера «{provider.name}» не указан API Key. Задайте ключ во вкладке «Провайдеры (API)»."})
                    return

                # Gather attached files for this specific message
                attached_meta: list[dict[str, Any]] = []
                for aid in attachment_ids or []:
                    asset = self.storage.get_asset(aid)
                    if asset:
                        attached_meta.append({
                            "id": asset.id,
                            "filename": asset.filename,
                            "path": asset.stored_path,
                            "category": asset.category,
                        })

                user_msg_obj = ChatMessage(
                    id=f"msg_{uuid.uuid4().hex[:10]}",
                    role="user",
                    content=user_message,
                    attachments=attached_meta,
                )
                self.storage.append_message(user_msg_obj)
                await emit({"type": "user_saved", "message": user_msg_obj.model_dump()})

                async def on_status_cb(status_text: str) -> None:
                    await emit({"type": "status", "text": status_text})

                self.mcp.set_runtime_hooks(on_status=on_status_cb, cancel_event=self._cancel_event)

                # Ensure real uploaded references (>=7s) have neural blueprint + vocal FX profile cached
                try:
                    from .ai_audio_models import deep_analyze_reference, models_available

                    if models_available().get("htdemucs"):
                        refs_now = self.storage.list_assets("references")
                        bp_cache_by_sig: dict[tuple[str, int], dict[str, Any]] = {}
                        for r in refs_now:
                            sig_k = (r.filename, int(r.size_bytes or 0))
                            if isinstance(r.analysis, dict) and r.analysis.get("neural_blueprint"):
                                bp_cache_by_sig[sig_k] = r.analysis["neural_blueprint"]
                        for r in refs_now:
                            if self._cancel_event.is_set():
                                break
                            an = dict(r.analysis or {})
                            dur_s = float(an.get("duration_sec", 0.0) or 0.0)
                            if an.get("neural_blueprint") or dur_s < 7.0:
                                continue
                            if not r.stored_path or not Path(r.stored_path).exists():
                                continue
                            sig_k = (r.filename, int(r.size_bytes or 0))
                            if sig_k in bp_cache_by_sig:
                                an["neural_blueprint"] = bp_cache_by_sig[sig_k]
                                r.analysis = an
                                self.storage.update_asset(r)
                                continue
                            await emit({
                                "type": "status",
                                "text": f"Слушаю референс нейросетями (HTDemucs + AST + CLAP + Basic Pitch): {r.filename}...",
                            })
                            bp = await asyncio.to_thread(
                                deep_analyze_reference,
                                r.stored_path,
                                (an.get("tempo") or {}).get("bpm"),
                                an.get("sections"),
                            )
                            if bp and not bp.get("error"):
                                bp_cache_by_sig[sig_k] = bp
                                an["neural_blueprint"] = bp
                                r.analysis = an
                                self.storage.update_asset(r)
                                await emit({"type": "assets_updated"})
                except Exception as neural_exc:
                    _LOGGER.warning("Auto neural reference enrichment skipped: %s", neural_exc)

                await emit({"type": "status", "text": "Подключаюсь к движку Ardour и собираю контекст..."})
                session_ctx = await self.mcp.get_session_snapshot_context()
                tools, ardour_map, raw_schemas = await self._collect_tools()
                system_prompt = self._build_system_prompt(mode, session_ctx)

                # Build conversation history (last 18 messages for bounded context)
                history = self.storage.get_chat_history()[-18:]
                llm_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
                for m in history:
                    text = m.content
                    if m.attachments:
                        att_str = ", ".join(f"{a['filename']} (id={a['id']}, path={a['path']})" for a in m.attachments)
                        text += f"\n[Прикреплённые файлы: {att_str}]"
                    llm_messages.append({"role": m.role, "content": text})

                tool_traces: list[ToolCallTrace] = []
                plugin_cards: list[PluginInstallCard] = []
                current_plan: list[PlanStep] = self.storage.get_plan()
                assistant_full_text_parts: list[str] = []
                snapshot_taken: Optional[str] = None
                mutating_done = False
                consecutive_errors: dict[str, int] = {}
                max_reconnects = max(1, min(int(getattr(settings, "max_reconnects", 10) or 10), 20))
                hit_step_limit = False

                for step_idx in range(settings.max_agent_steps):
                    if self._cancel_event.is_set():
                        await emit({"type": "status", "text": "Остановлено пользователем."})
                        break

                    await emit({"type": "status", "text": f"Думаю ({provider.model}) — шаг {step_idx + 1}/{settings.max_agent_steps}..."})

                    step_tokens: list[str] = []

                    async def on_tok(tok: str) -> None:
                        step_tokens.append(tok)
                        await emit({"type": "token", "text": tok})

                    async def on_reset() -> None:
                        step_tokens.clear()
                        await emit({"type": "retry_reset"})

                    resp = await stream_chat_completion(
                        provider=provider,
                        messages=llm_messages,
                        tools=tools,
                        on_token=on_tok,
                        on_status=on_status_cb,
                        on_reset=on_reset,
                        max_reconnects=max_reconnects,
                        cancel_event=self._cancel_event,
                    )

                    text_piece = resp.get("content", "")
                    tool_calls = resp.get("tool_calls") or []
                    if text_piece:
                        assistant_full_text_parts.append(text_piece)

                    if not tool_calls:
                        break

                    # Record assistant message with tool calls (avoid content: null for proxy compat)
                    if resp.get("used_react_fallback"):
                        llm_messages.append({"role": "assistant", "content": text_piece or ""})
                    else:
                        llm_messages.append({
                            "role": "assistant",
                            "content": text_piece or "",
                            "tool_calls": tool_calls,
                        })

                    paused_for_plugin = False

                    for tc in tool_calls:
                        if self._cancel_event.is_set():
                            break
                        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                        fn = tc.get("function") or {}
                        fn_name = str(fn.get("name", ""))
                        fn_args = _safe_parse_args(fn.get("arguments"))

                        # Auto-snapshot before first mutating Ardour call
                        if (
                            not mutating_done
                            and settings.auto_snapshot
                            and session_ctx.get("connected")
                            and fn_name in ardour_map
                            and not any(x in fn_name for x in ("get_", "_list", "list_"))
                        ):
                            mutating_done = True
                            try:
                                snap_name = f"AI_Before_{int(time.time())}"
                                await self.mcp.call_tool("session_save", {"snapshotName": snap_name, "switchToSnapshot": False})
                                snapshot_taken = snap_name
                                await emit({"type": "snapshot", "name": snap_name})
                            except Exception:
                                pass

                        await emit({
                            "type": "tool_start",
                            "id": tc_id,
                            "tool_name": fn_name,
                            "arguments": fn_args,
                        })
                        t0 = time.perf_counter()
                        ok, result_obj = await self._execute_tool(
                            fn_name, fn_args, ardour_map, raw_schemas, plugin_cards, current_plan, emit
                        )
                        dur_ms = int((time.perf_counter() - t0) * 1000)

                        if ok and not (isinstance(result_obj, dict) and result_obj.get("error")):
                            consecutive_errors[fn_name] = 0
                        else:
                            consecutive_errors[fn_name] = consecutive_errors.get(fn_name, 0) + 1
                            if isinstance(result_obj, dict) and consecutive_errors[fn_name] >= 2:
                                result_obj = dict(result_obj)
                                result_obj["_guard_warning"] = (
                                    f"Tool '{fn_name}' failed {consecutive_errors[fn_name]} times in a row. "
                                    "STOP guessing API methods! Use the verified Lua preset/MIDI snippets from your prompt, "
                                    "standard MCP tools, studio_request_plugin_install, or report cleanly."
                                )

                        result_json = json.dumps(result_obj, ensure_ascii=False)
                        summary = result_json[:280] + ("…" if len(result_json) > 280 else "")
                        trace = ToolCallTrace(
                            id=tc_id,
                            tool_name=fn_name,
                            arguments=fn_args,
                            result_summary=summary,
                            ok=ok,
                            duration_ms=dur_ms,
                        )
                        tool_traces.append(trace)
                        await emit({"type": "tool_end", "trace": trace.model_dump()})

                        # Feed tool result back to LLM (truncated safely at 14k chars)
                        tool_payload = result_json[:14000]
                        if resp.get("used_react_fallback"):
                            llm_messages.append({
                                "role": "user",
                                "content": f"[TOOL RESULT: {fn_name}]\n```json\n{tool_payload}\n```",
                            })
                        else:
                            llm_messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": tool_payload,
                            })

                        if fn_name == "studio_request_plugin_install":
                            paused_for_plugin = True

                    if paused_for_plugin:
                        continue

                    if step_idx == settings.max_agent_steps - 1 and not self._cancel_event.is_set():
                        hit_step_limit = True

                # If step limit reached right after executing tools, ask LLM for a clean status summary
                if hit_step_limit and not self._cancel_event.is_set():
                    try:
                        await emit({"type": "status", "text": "Достигнут лимит шагов — формирую отчёт..."})
                        llm_messages.append({
                            "role": "user",
                            "content": (
                                f"[SYSTEM]: Достигнут лимит шагов агента ({settings.max_agent_steps}). "
                                "НЕ вызывай инструменты. Честно и кратко перечисли пользователю, что уже сделано в сессии, "
                                "что ещё не успели сделать (например, экспорт WAV / MIDI), и предложи написать «продолжай» для завершения."
                            ),
                        })

                        async def on_wrap_tok(tok: str) -> None:
                            await emit({"type": "token", "text": tok})

                        async def on_wrap_reset() -> None:
                            await emit({"type": "retry_reset"})

                        wrap = await stream_chat_completion(
                            provider=provider,
                            messages=llm_messages,
                            tools=None,
                            on_token=on_wrap_tok,
                            on_status=on_status_cb,
                            on_reset=on_wrap_reset,
                            max_reconnects=max_reconnects,
                            cancel_event=self._cancel_event,
                        )
                        if wrap.get("content"):
                            assistant_full_text_parts.append(wrap["content"])
                    except Exception as wrap_exc:
                        _LOGGER.warning("Wrap-up summary call failed: %s", wrap_exc)

                final_text = "\n\n".join(p.strip() for p in assistant_full_text_parts if p.strip())
                if not final_text and tool_traces:
                    if hit_step_limit:
                        final_text = (
                            f"Достигнут лимит шагов ({settings.max_agent_steps}): выполнено {len(tool_traces)} действий, "
                            "но задача ещё не завершена до конца. Напишите «продолжай» (или увеличьте «Макс. шагов агента» во вкладке Провайдеры)."
                        )
                    else:
                        final_text = f"Готово! Выполнено действий с движком: {len(tool_traces)}."
                elif not final_text and not self._cancel_event.is_set():
                    final_text = (
                        "⚠️ Модель вернула пустой ответ без текста и вызова инструментов. "
                        "Проверьте выбранного провайдера/модель и баланс/регион во вкладке «Провайдеры»."
                    )

                assistant_msg = ChatMessage(
                    id=f"msg_{uuid.uuid4().hex[:10]}",
                    role="assistant",
                    content=final_text,
                    tool_calls=tool_traces,
                    plugin_cards=plugin_cards,
                    plan_snapshot=current_plan,
                    snapshot_name=snapshot_taken,
                )
                self.storage.append_message(assistant_msg)
                await emit({"type": "done", "message": assistant_msg.model_dump()})
            except Exception as exc:
                _LOGGER.exception("Agent worker failed: %s", exc)
                err_msg = ChatMessage(
                    id=f"msg_{uuid.uuid4().hex[:10]}",
                    role="assistant",
                    content=f"**Ошибка выполнения:** {exc}",
                    tool_calls=locals().get("tool_traces", []),
                    plugin_cards=locals().get("plugin_cards", []),
                    plan_snapshot=locals().get("current_plan", []),
                )
                self.storage.append_message(err_msg)
                await emit({"type": "error", "message": str(exc)})
            finally:
                self.mcp.set_runtime_hooks(None, None)
                self._running = False
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                self._cancel_event.set()
                task.cancel()
