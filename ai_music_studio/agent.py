"""Autonomous AI Music Producer Agent orchestrating Ardour MCP + DSP + Web Search."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
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
            "description": "Compare a bounced mix WAV file against a reference track (by reference asset_id) and get LUFS, spectral band deltas, stereo width comparison, and actionable mixing recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mix_wav_path": {"type": "string", "description": "Absolute path to exported mix preview WAV."},
                    "reference_asset_id": {"type": "string", "description": "Asset ID of the reference track."},
                },
                "required": ["mix_wav_path", "reference_asset_id"],
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
            "description": "Inspect installed SoundFonts (.sf2), WAV audio samples/loops, and Neural Amp Modeler (.nam) amp captures in the project sound library. Filter by category (e.g. drums, 808, 909, bass, piano, guitar, synth, nam, sf2, all) or search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter category: all, sf2, nam, drums, 808, 909, bass, guitar, piano, synth, pad, etc."},
                    "query": {"type": "string", "description": "Optional filename/folder substring filter."},
                    "limit": {"type": "integer", "minimum": 5, "maximum": 120},
                },
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

        def compact_analysis(a: dict[str, Any]) -> dict[str, Any]:
            # Strip dense waveform_peaks array so we don't waste LLM tokens
            return {k: v for k, v in a.items() if k != "waveform_peaks"}

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
            "(анализ -> настройка сессии/темпа/маркеров -> аранжировка/MIDI/сэмплы -> импорт/нарезка/тайминг вокала -> "
            "сведение/эффекты на треках и мастере -> контрольный экспорт и сравнение с референсом). "
            "Останавливайся только если нужно попросить пользователя установить внешний плагин."
            if mode == "autonomous"
            else "РЕЖИМ: ПОШАГОВЫЙ КОНТРОЛЬ. Сначала обнови план через `studio_update_plan`, выполни анализ и "
            "согласуй крупный этап с пользователем перед массовыми изменениями в сессии."
        )

        return f"""Ты — профессиональный AI-музыкальный продюсер, аранжировщик и звукорежиссёр, встроенный напрямую в DAW **Ardour** (AI Music Studio).
Отвечай пользователю на языке его запроса (по умолчанию — на русском), чётко, по делу и дружелюбно.

{mode_instruction}

## Главные принципы работы
1. **Полный контроль над движком Ardour:** все инструменты с префиксом `ardour__` напрямую управляют открытой сессией Ardour (треки, шины, мастер, импорт аудио, разрезка/перемещение/фейды регионов, MIDI JSON, автоматизация, плагины и их параметры, темп, маркеры, рендер, Lua-скрипты).
2. **Не работай вслепую (Исследование перед применением):**
   - **Изучай плагины перед настройкой:** добавив или выбрав плагин (`ardour__plugin_list_available`), ВСЕГДА вызывай `ardour__plugin_get_description` (изучи имена параметров, их реальные диапазоны min..max и единицы измерения) и проверяй пресеты/доки через Lua (`plug:get_info():get_presets(false)`, `plug:get_docs()`). Для незнакомых или продвинутых сторонних плагинов (Airwindows Consolidated, Surge XT / Effects, Dexed, Neural Amp Modeler, TDR Nova / Kotelnikov / SlickEQ / Molotok, Dragonfly Reverb, TAL-Reverb/Chorus/NoiseMaker, TSE BOD, LoudMax) используй `studio_web_search` / `studio_fetch_webpage`, чтобы узнать назначение ручек и оптимальные настройки («sweet spots»).
   - **Экономь шаги — настройку параметров делай пакетом:** после `ardour__plugin_get_description` используй **`studio_set_plugin_parameters`**, чтобы выставить все нужные ручки плагина за **один вызов** вместо десятков одиночных вызовов! Это позволяет успеть полностью написать MIDI/аудио, свести и экспортировать WAV.
   - **Изучай звуки и инструменты:** просматривай библиотеку через `studio_inspect_sound_library` (SoundFonts `.sf2`, WAV-сэмплы, `.nam` модели усилителей), проверяй референсы и сэмплы через `studio_analyze_audio` (BPM, тональность, LUFS, 7-полосный спектр, транзиенты) и таймкоды вокала (`studio_align_vocal_lyrics`).
3. **План работы:** в начале сложной задачи всегда вызывай `studio_update_plan`, чтобы пользователь видел живой прогресс.
4. **Плагины и Web Search:** сначала проверяй реально установленные плагины через `ardour__plugin_list_available`. Если не хватает важного инструмента/эффекта (например, редкого синтезатора или специфической обработки), найди лучший актуальный вариант через `studio_web_search` и отправь карточку через `studio_request_plugin_install` с понятной инструкцией для Windows (папка `BUILD\\lib\\ardour9\\VST3` или `%LOCALAPPDATA%\\Programs\\Common\\VST3`).
5. **Безопасность:** если Ardour MCP не подключён (`connected: false`), честно сообщи об этом и предложи, что нужно сделать (открыть Ardour / включить MCPHttp), при этом всё равно можешь анализировать аудио, изучать локальную библиотеку звуков и подготовить детальный план продакшна.

## Критические правила по инструментам, SoundFont (.sf2) и Ardour Lua
- **`ACE Fluid Synth` (`lv2:urn:ardour:a-fluidsynth`) НЕ имеет встроенных звуков!** Пока в него не загружен `.sf2` банк (SoundFont), он выдаёт полную тишину.
- Проверяй `soundfonts` и `fluid_synth_presets` в разделе `Живое состояние Ardour DAW`:
  - Если в `fluid_synth_presets` есть пресет (например `"GeneralUser GS"`), то **сразу после добавления `ACE Fluid Synth`** на MIDI-трек выполни **ровно один** вызов `ardour__lua_run_script` по этому проверенному шаблону (загружает `.sf2` и при необходимости выбирает GM-инструмент `0..127` на канале 1; для барабанов на 10-м канале Program Change не нужен):
    ```lua
    local r = Session:route_by_id(PBD.ID("<TRACK_ID>"))
    local ins = r:nth_plugin(0):to_plugininsert()
    local plug = ins:plugin(0)
    local ps = plug:preset_by_label("GeneralUser GS")
    if ps and ps.valid then plug:load_preset(ps) end
    ARDOUR.LuaAPI.usleep(800000)
    local parser = ARDOUR.RawMidiParser()
    for _, b in ipairs({{0xB0, 0x00, 0x00, 0xB0, 0x20, 0x00, 0xC0, <PGM_0_127>}}) do
      if parser:process_byte(b) then
        ins:write_immediate_event(Evoral.EventType.MIDI_EVENT, parser:buffer_size(), parser:midi_buffer())
      end
    end
    ```
  - Частые GM-программы (`<PGM_0_127>`): `0`=Acoustic Grand Piano, `25`=Acoustic Steel Guitar, `27`=Clean Electric Guitar, `29`=Overdriven Guitar, `30`=Distortion Guitar, `33`=Electric Bass Finger, `38`=Synth Bass 1, `48`=Strings, `81`=Saw Lead.
  - Если `soundfonts` и `fluid_synth_presets` **пусты** и сторонних VSTi нет — **НЕ вешай пустой `ACE Fluid Synth`!** Вызови `studio_request_plugin_install` (предложи `.sf2` или бесплатный VST3), либо по согласованию используй `ACE Reasonable Synth` (`lv2:https://community.ardour.org/node/7596`).
- **Анти-зацикливание по `ardour__lua_run_script`:** НИКОГДА не гадай методы Lua API вслепую! В Ardour Lua **НЕТ** `Session:region_by_id`, `to_midi_track`, `Evoral.PatchChange`, `PatchChangePtr`, `MidiModel:new_patch_change_command`, `plug:get_presets()` (`get_presets` только у `plug:get_info():get_presets(false)`), `ins:load_preset()` (`load_preset` только у `plug = ins:plugin(0)`). Если инструмент ошибся 2 раза — СТОП, не подбирай API дальше.
- **Обязательная проверка экспорта:** после `ardour__session_export_preview` всегда прогоняй итоговый WAV через `studio_analyze_audio` (`file_path=...`). Если `integrated_lufs < -60` или файл без сигнала — НЕ сдавай тишину, сперва почини синт/пресет/маршрутизацию и переэкспортируй!

## Активные Скиллы пользователя
{skills_text}

## Контекст проекта (из вкладок AI Music Studio)
### Вкладка «Референсы (Примеры звучания)»
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
                asset.analysis = analysis
                self.storage.update_asset(asset)
                await emit({"type": "assets_updated"})
            compact = {k: v for k, v in analysis.items() if k != "waveform_peaks"}
            return True, compact

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
            try:
                res = await self.mcp.call_tool(mcp_name, coerced)
                return True, res.get("structured", res)
            except Exception as exc:
                return False, {"error": str(exc)}

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
