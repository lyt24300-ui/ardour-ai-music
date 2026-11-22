"""End-to-end automated verification suite for AI Music Studio."""

from __future__ import annotations

import json
import math
import socket
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ai_music_studio.audio_analyzer import (
    align_vocal_with_lyrics,
    analyze_audio_file,
    compare_mix_to_reference,
)
from ai_music_studio.llm_client import parse_react_tool_calls
from ai_music_studio.models import ProviderConfig, SkillItem
from ai_music_studio.storage import StudioStorage
from ai_music_studio.theme_loader import list_ardour_themes, load_ardour_theme


def _write_wav(path: Path, stereo: np.ndarray, sr: int = 44100) -> None:
    clipped = np.clip(stereo, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _synth_reference_wav(path: Path, bpm: float = 128.0, dur: float = 6.0, sr: int = 44100, gain: float = 0.45) -> None:
    t = np.arange(int(dur * sr), dtype=np.float64) / sr
    # A minor triad: A3 (220), C4 (261.63), E4 (329.63) + A2 bass (110)
    chord = (
        0.35 * np.sin(2 * math.pi * 110.0 * t)
        + 0.25 * np.sin(2 * math.pi * 220.0 * t)
        + 0.20 * np.sin(2 * math.pi * 261.63 * t)
        + 0.20 * np.sin(2 * math.pi * 329.63 * t)
    )
    # 128 BPM kick envelope
    beat_sec = 60.0 / bpm
    beat_phase = np.mod(t, beat_sec)
    kick_env = np.exp(-beat_phase * 28.0)
    kick = 0.75 * kick_env * np.sin(2 * math.pi * (55.0 + 90.0 * np.exp(-beat_phase * 40.0)) * beat_phase)
    left = gain * (chord + kick)
    right = gain * (0.9 * chord + 0.1 * np.sin(2 * math.pi * 329.63 * t + 0.7) + kick)
    _write_wav(path, np.stack([left, right], axis=1), sr)


def _synth_vocal_wav(path: Path, sr: int = 44100) -> None:
    dur = 5.0
    n = int(dur * sr)
    sig = np.random.normal(0, 0.0005, n)  # quiet noise floor
    t = np.arange(n, dtype=np.float64) / sr
    # Phrase 1: 0.6s - 1.9s at 220 Hz (A3)
    m1 = (t >= 0.6) & (t <= 1.9)
    sig[m1] += 0.35 * np.sin(2 * math.pi * 220.0 * t[m1]) + 0.12 * np.sin(2 * math.pi * 440.0 * t[m1])
    # Phrase 2: 2.8s - 4.2s at 246.94 Hz (B3)
    m2 = (t >= 2.8) & (t <= 4.2)
    sig[m2] += 0.32 * np.sin(2 * math.pi * 246.94 * t[m2]) + 0.10 * np.sin(2 * math.pi * 493.88 * t[m2])
    _write_wav(path, np.stack([sig, sig], axis=1), sr)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_themes() -> None:
    themes = list_ardour_themes()
    assert len(themes) >= 10, f"Expected Ardour themes from gtk2_ardour/themes, got {len(themes)}"
    dark = load_ardour_theme("dark")
    assert dark["css_vars"]["--ardour-bg"] == "#3d3d3d"
    assert dark["css_vars"]["--ardour-page-active"] == "#3dcc4c"
    assert dark["css_vars"]["--ardour-clock-fg"] == "#78cd2f"
    assert dark["is_light"] is False
    light = load_ardour_theme("captain_light")
    assert light["is_light"] is True
    print(f"[PASS] Ardour themes loaded ({len(themes)} themes synced)")


def test_audio_dsp_and_alignment() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        ref_wav = tdp / "ref_128_am.wav"
        mix_wav = tdp / "mix_hot.wav"
        voc_wav = tdp / "vocal.wav"

        _synth_reference_wav(ref_wav, bpm=128.0, gain=0.4)
        _synth_reference_wav(mix_wav, bpm=128.0, gain=0.99)
        _synth_vocal_wav(voc_wav)

        ref_an = analyze_audio_file(ref_wav, category="references")
        assert 122.0 <= ref_an["tempo"]["bpm"] <= 134.0, f"BPM off: {ref_an['tempo']}"
        assert "key" in ref_an["key"] and ref_an["key"]["key"] != "Unknown"
        assert len(ref_an["waveform_peaks"]) == 220
        assert len(ref_an["spectral"]["bands_relative_db"]) == 7
        assert ref_an["loudness"]["integrated_lufs"] < 0.0

        voc_an = analyze_audio_file(voc_wav, category="vocals")
        assert voc_an["vocal"]["phrase_count"] >= 2, f"Expected >=2 phrases: {voc_an['vocal']}"
        assert voc_an["vocal"]["pitch"]["median_hz"] > 180.0

        al = align_vocal_with_lyrics(
            voc_wav,
            [{"section": "Verse 1", "lines": ["First vocal line", "Second vocal line"]}],
            existing_vad_phrases=voc_an["vocal"]["phrases"],
            whisper_enabled=False,
        )
        assert len(al["aligned_lines"]) == 2
        assert al["aligned_lines"][0]["start_sec"] is not None
        assert al["aligned_lines"][1]["start_sec"] > al["aligned_lines"][0]["start_sec"]

        cmp = compare_mix_to_reference(mix_wav, ref_wav)
        assert cmp["lufs_delta_db"] > 2.0
        assert len(cmp["recommendations"]) >= 1
        print(
            f"[PASS] Audio DSP: BPM={ref_an['tempo']['bpm']}, Key={ref_an['key']['key']}, "
            f"LUFS={ref_an['loudness']['integrated_lufs']}, Vocal phrases={voc_an['vocal']['phrase_count']}"
        )


def test_react_parser() -> None:
    sample = 'Thinking...\n```tool_call\n{"name": "ardour__tracks_add", "arguments": {"name": "Lead Vocal",}}\n```'
    calls = parse_react_tool_calls(sample)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ardour__tracks_add"
    print("[PASS] ReAct JSON fallback tool parser")


def test_full_agent_and_api_end_to_end() -> None:
    mcp_port = _free_port()
    llm_port = _free_port()
    mcp_calls_seen: list[str] = []
    llm_turn = {"n": 0}

    class MockMCPHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            method = req.get("method")
            rid = req.get("id", 1)

            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "serverInfo": {"name": "ArdourMock", "version": "9.10"}}
            elif method == "tools/list":
                result = {
                    "tools": [
                        {"name": "session_get_info", "description": "Get info", "inputSchema": {"type": "object", "properties": {}}},
                        {"name": "session_save", "description": "Save", "inputSchema": {"type": "object", "properties": {"snapshotName": {"type": "string"}}}},
                        {"name": "tracks_list", "description": "List tracks", "inputSchema": {"type": "object", "properties": {}}},
                        {"name": "markers_list", "description": "List markers", "inputSchema": {"type": "object", "properties": {}}},
                        {"name": "tracks_add", "description": "Add track", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}},
                    ]
                }
            elif method == "tools/call":
                tname = req["params"]["name"]
                mcp_calls_seen.append(tname)
                if tname == "session_get_info":
                    sc = {"name": "FLRadioDemo", "tempoBpm": 128.0, "sampleRate": 48000}
                elif tname == "tracks_add":
                    sc = {"id": "101", "name": req["params"]["arguments"].get("name", "Track")}
                else:
                    sc = {"ok": True}
                result = {"structuredContent": sc, "content": [{"type": "text", "text": json.dumps(sc)}]}
            else:
                result = {}

            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class MockLLMHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "mock-producer-llm"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            _ = json.loads(self.rfile.read(length).decode("utf-8"))
            llm_turn["n"] += 1
            if llm_turn["n"] == 1:
                payload = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tc_plan",
                                    "type": "function",
                                    "function": {
                                        "name": "studio_update_plan",
                                        "arguments": json.dumps({"steps": [{"id": "1", "title": "Создать вокальную дорожку", "status": "in_progress"}]}),
                                    },
                                },
                                {
                                    "id": "tc_track",
                                    "type": "function",
                                    "function": {
                                        "name": "ardour__tracks_add",
                                        "arguments": json.dumps({"name": "Lead Vocal"}),
                                    },
                                },
                                {
                                    "id": "tc_plug",
                                    "type": "function",
                                    "function": {
                                        "name": "studio_request_plugin_install",
                                        "arguments": json.dumps({
                                            "plugin_name": "Vital",
                                            "format": "VST3",
                                            "price_type": "Free",
                                            "purpose": "Современный синтезатор для лида",
                                            "download_url": "https://vital.audio",
                                            "steps": ["Скачайте установщик Vital", "Установите VST3"],
                                        }),
                                    },
                                },
                            ],
                        }
                    }]
                }
            else:
                payload = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Дорожка Lead Vocal создана в Ardour! Рекомендую установить Vital (см. карточку выше).",
                        }
                    }]
                }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    mcp_srv = HTTPServer(("127.0.0.1", mcp_port), MockMCPHandler)
    llm_srv = HTTPServer(("127.0.0.1", llm_port), MockLLMHandler)
    threading.Thread(target=mcp_srv.serve_forever, daemon=True).start()
    threading.Thread(target=llm_srv.serve_forever, daemon=True).start()

    try:
        import ai_music_studio.server as srv

        with tempfile.TemporaryDirectory() as td:
            test_storage = StudioStorage(Path(td))
            srv.storage = test_storage
            srv.mcp_client.set_url(f"http://127.0.0.1:{mcp_port}/mcp")
            srv.agent.storage = test_storage
            srv.agent.mcp = srv.mcp_client

            client = TestClient(srv.app)

            assert client.get("/").status_code == 200
            assert client.get("/api/health").json()["ok"] is True

            # Settings pointing to mock MCP
            st_data = client.get("/api/settings").json()
            st_data["ardour_mcp_url"] = f"http://127.0.0.1:{mcp_port}/mcp"
            assert client.put("/api/settings", json=st_data).status_code == 200

            # Ardour MCP status check
            ard_st = client.get("/api/ardour/status").json()
            assert ard_st["connected"] is True
            assert ard_st["session"]["name"] == "FLRadioDemo"

            # Add custom OpenAI-compatible provider
            prov = ProviderConfig(
                id="mock-prov",
                name="Mock OpenAI",
                base_url=f"http://127.0.0.1:{llm_port}/v1",
                api_key="test-key",
                model="mock-producer-llm",
                is_default=True,
            )
            assert client.post("/api/providers", json=prov.model_dump()).status_code == 200
            models_resp = client.post("/api/providers/models", json=prov.model_dump()).json()
            assert models_resp["ok"] and "mock-producer-llm" in models_resp["models"]

            # Skills CRUD
            sk = SkillItem(id="custom_trap", name="Trap Mix", category="mixing", content="Heavy 808 + crisp vocal")
            assert client.post("/api/skills", json=sk.model_dump()).status_code == 200

            # Lyrics save
            assert client.put("/api/lyrics", json={"title": "Test Song", "raw_text": "[Verse 1]\nLine one"}).status_code == 200

            # Upload synthetic reference audio
            ref_wav = Path(td) / "ref.wav"
            _synth_reference_wav(ref_wav)
            with ref_wav.open("rb") as f:
                up = client.post(
                    "/api/assets/upload",
                    data={"category": "references", "role": "general_style", "auto_analyze": "true"},
                    files={"file": ("ref.wav", f, "audio/wav")},
                )
            assert up.status_code == 200
            assert up.json()["analysis"]["tempo"]["bpm"] > 100

            # Run full SSE chat turn triggering Ardour MCP tool + auto-snapshot + plan + plugin card
            events = []
            with client.stream("POST", "/api/chat/stream", json={"message": "Собери вокал и посоветуй синт", "provider_id": "mock-prov"}) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:].strip()))

            ev_types = [e["type"] for e in events]
            assert "snapshot" in ev_types, f"Expected auto-snapshot in {ev_types}"
            assert "plan" in ev_types
            assert "plugin_card" in ev_types
            assert "done" in ev_types
            assert "tracks_add" in mcp_calls_seen
            assert "session_save" in mcp_calls_seen

            print(f"[PASS] Full E2E FastAPI + Agent + Ardour MCP + OpenAI-compat stream ({len(events)} SSE events)")
    finally:
        mcp_srv.shutdown()
        llm_srv.shutdown()


def test_sound_library_and_mastery_skill() -> None:
    from ai_music_studio.agent import (
        STUDIO_NATIVE_TOOLS,
        _coerce_args_to_schema,
        _sanitize_schema_for_llm,
    )
    from ai_music_studio.sound_library import inspect_sound_library

    tool_names = {t["function"]["name"] for t in STUDIO_NATIVE_TOOLS}
    assert "studio_inspect_sound_library" in tool_names
    assert "studio_set_plugin_parameters" in tool_names

    raw_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "pluginIndex": {"type": "integer"},
            "value": {"type": "number"},
            "enabled": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    sanitized = _sanitize_schema_for_llm(raw_schema)
    assert "additionalProperties" not in sanitized
    coerced = _coerce_args_to_schema(
        {"id": 302, "pluginIndex": "1", "value": "-3.5", "enabled": "true", "reason": "hallucinated"},
        raw_schema,
    )
    assert coerced == {"id": "302", "pluginIndex": 1, "value": -3.5, "enabled": True}

    lib = inspect_sound_library(category="all", limit=10)
    assert "soundfonts_count" in lib and "total_audio_samples" in lib

    with tempfile.TemporaryDirectory() as td:
        st = StudioStorage(Path(td))
        skills = {s.id: s for s in st.list_skills()}
        assert "plugin_sound_mastery" in skills
        assert skills["plugin_sound_mastery"].enabled is True
    print(f"[PASS] Sound library ({lib['soundfonts_count']} sf2, {lib['nam_models_count']} nam, {lib['total_audio_samples']} samples), mastery skill & schema coercion")


def test_ten_reconnects_llm_and_mcp() -> None:
    import asyncio
    from ai_music_studio import llm_client, mcp_client

    llm_attempts = {"n": 0}
    mcp_call_attempts = {"n": 0}
    status_msgs: list[str] = []

    class FlakyHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            n_read = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n_read).decode("utf-8") if n_read else "{}"
            body = json.loads(raw)

            if self.path.endswith("/chat/completions"):
                llm_attempts["n"] += 1
                if llm_attempts["n"] <= 10:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Temporary overload")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "Reconnected OK"}}]}).encode("utf-8"))
                return

            if self.path == "/mcp":
                method = body.get("method")
                if method == "initialize":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"serverInfo": {"name": "flaky"}}}).encode("utf-8"))
                    return
                mcp_call_attempts["n"] += 1
                if mcp_call_attempts["n"] <= 10:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": json.dumps({"ok": True})}], "isError": False},
                }).encode("utf-8"))

    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), FlakyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    orig_sleep = asyncio.sleep

    async def fast_sleep(_delay: float) -> None:
        await orig_sleep(0.001)

    asyncio.sleep = fast_sleep
    try:
        async def run_checks() -> None:
            prov = ProviderConfig(id="p", name="Flaky", base_url=f"http://127.0.0.1:{port}/v1", model="m")

            async def on_st(s: str) -> None:
                status_msgs.append(s)

            res = await llm_client.stream_chat_completion(
                prov,
                [{"role": "user", "content": "hi"}],
                on_status=on_st,
                max_reconnects=10,
            )
            assert res["content"] == "Reconnected OK"
            assert res["reconnects_used"] == 10

            mc = mcp_client.ArdourMCPClient(f"http://127.0.0.1:{port}/mcp")
            mc.set_runtime_hooks(on_status=on_st)
            mres = await mc.call_tool("session_get_info", {}, max_reconnects=10)
            assert mres["ok"] is True

        asyncio.run(run_checks())
        assert llm_attempts["n"] == 11
        assert mcp_call_attempts["n"] == 11
        assert any("реконнект 10/10" in s for s in status_msgs)
        print("[PASS] 10x auto-reconnect verified for both LLM API and Ardour MCP")
    finally:
        asyncio.sleep = orig_sleep
        srv.shutdown()


def test_neural_audio_and_region_fix() -> None:
    import asyncio
    from ai_music_studio import ai_audio_models
    from ai_music_studio.agent import AIProducerAgent, STUDIO_NATIVE_TOOLS
    from ai_music_studio.mcp_client import ArdourMCPClient
    from ai_music_studio.sound_library import inspect_sound_library

    tool_names = {t["function"]["name"] for t in STUDIO_NATIVE_TOOLS}
    for req_tool in (
        "studio_deep_analyze_reference",
        "studio_analyze_vocal_fx",
        "studio_setup_track_instrument",
        "studio_compose_humanized_midi",
    ):
        assert req_tool in tool_names, f"Missing tool: {req_tool}"

    avail = ai_audio_models.models_available()
    assert isinstance(avail, dict) and set(avail.keys()) == {"htdemucs", "ast", "clap", "basic_pitch"}

    lib = inspect_sound_library(category="all", limit=20)
    assert lib["soundfonts_count"] >= 0 and "sfz_instruments_count" in lib and "surge_patches_count" in lib

    # Test Vocal FX Detective, Drum Grid, and Basic Pitch ONNX on synthetic audio
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        voc = tdp / "voc.wav"
        ref = tdp / "ref.wav"
        _synth_vocal_wav(voc)
        _synth_reference_wav(ref, bpm=120.0, dur=8.0)

        from ai_music_studio.audio_analyzer import load_audio_pcm
        v_st, v_sr = load_audio_pcm(voc, target_sr=44100)
        vfx = ai_audio_models.analyze_vocal_fx_profile(v_st.T, sr=v_sr, inst_rms_db=-18.0, use_clap=avail["clap"])
        assert vfx["has_vocal"] is True
        assert "reverb_type" in vfx and "recommended_ardour_chain" in vfx

        r_st, r_sr = load_audio_pcm(ref, target_sr=44100)
        dgrid = ai_audio_models.extract_drum_pattern_grid(r_st.T, bpm=120.0, sr=r_sr)
        assert dgrid["has_drums"] is True
        assert 0 in dgrid["kick_steps_16"] or len(dgrid["kick_steps_16"]) >= 1

        if avail["basic_pitch"]:
            bp = ai_audio_models.transcribe_midi_basic_pitch(r_st.mean(axis=1), sr=r_sr)
            assert bp["note_count"] >= 1
            assert len(bp["active_pitch_classes"]) >= 1

        if avail["ast"]:
            ast_res = ai_audio_models.classify_stem_ast(r_st.mean(axis=1), sr=r_sr)
            assert len(ast_res) >= 1

    # Mock MCP test for whole_file regionId remapping + studio_setup_track_instrument + studio_compose_humanized_midi
    port = _free_port()
    seen_calls: list[tuple[str, dict]] = []

    class MockRegionMCP(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
            method, rid = req.get("method"), req.get("id", 1)
            if method == "initialize":
                res = {"serverInfo": {"name": "MockArdour", "version": "9.10"}}
            elif method == "tools/list":
                res = {
                    "tools": [
                        {"name": "tracks/add", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}}},
                        {"name": "tracks/list", "inputSchema": {"type": "object", "properties": {}}},
                        {"name": "track/get_regions", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}},
                        {"name": "audio/import_file", "inputSchema": {"type": "object", "properties": {"trackId": {"type": "string"}, "filePath": {"type": "string"}}}},
                        {"name": "region/resize_samples", "inputSchema": {"type": "object", "properties": {"regionId": {"type": "string"}, "startSample": {"type": "integer"}, "endSample": {"type": "integer"}}}},
                        {"name": "plugin/list_available", "inputSchema": {"type": "object", "properties": {"search": {"type": "string"}}}},
                        {"name": "plugin/add", "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "pluginId": {"type": "string"}}}},
                        {"name": "lua/run_script", "inputSchema": {"type": "object", "properties": {"script": {"type": "string"}}}},
                        {"name": "session/get_info", "inputSchema": {"type": "object", "properties": {}}},
                        {"name": "midi_note/import_json", "inputSchema": {"type": "object", "properties": {"trackId": {"type": "string"}, "midi": {"type": "object"}}}},
                    ]
                }
            elif method == "tools/call":
                tn = req["params"]["name"]
                ar = req["params"].get("arguments") or {}
                seen_calls.append((tn, ar))
                if tn == "audio/import_file":
                    sc = {"trackId": "311", "regionId": "463", "regionName": "Vocal_1", "positionSample": 1000}
                elif tn == "track/get_regions":
                    sc = {"id": "311", "regions": [{"regionId": "999", "name": "Vocal_1", "playlistId": "pl_1", "startSample": 1000}]}
                elif tn == "region/resize_samples":
                    assert ar["regionId"] == "999", f"Expected remapped playlist region 999, got {ar['regionId']}"
                    sc = {"regionId": "999", "resized": True}
                elif tn == "tracks/add":
                    sc = {"id": "501", "name": ar.get("name", "MIDI")}
                elif tn == "plugin/list_available":
                    sc = {"plugins": [{"pluginId": "lv2:http://sfztools.github.io/sfizz", "name": "sfizz"}]}
                elif tn == "plugin/add":
                    sc = {"id": ar["id"], "pluginIndex": 0}
                elif tn == "session/get_info":
                    sc = {"tempoBpm": 100.0}
                else:
                    sc = {"ok": True}
                res = {"structuredContent": sc, "content": [{"type": "text", "text": json.dumps(sc)}]}
            else:
                res = {}
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", port), MockRegionMCP)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as td:
            ag = AIProducerAgent(StudioStorage(Path(td)), ArdourMCPClient(f"http://127.0.0.1:{port}/mcp"))

            async def dummy_emit(_ev):
                return None

            async def run_mock():
                tools, amap, rsch = await ag._collect_tools()
                ok1, st1 = await ag._execute_tool("ardour__audio_import_file", {"trackId": "311", "filePath": "x.ogg"}, amap, rsch, [], [], dummy_emit)
                assert ok1 and st1["regionId"] == "999"
                ok2, st2 = await ag._execute_tool("ardour__region_resize_samples", {"regionId": "463", "startSample": 1100, "endSample": 5000}, amap, rsch, [], [], dummy_emit)
                assert ok2 and st2["resized"] is True
                ok3, st3 = await ag._execute_tool("studio_setup_track_instrument", {"track_name": "Drums", "plugin_type": "sfizz", "preset_label": "AVL Black Pearl 5pc", "midi_channel": 9}, amap, rsch, [], [], dummy_emit)
                assert ok3 and st3["track_id"] == "501"
                ok4, st4 = await ag._execute_tool("studio_compose_humanized_midi", {"track_id": "501", "role": "guitar_strum", "length_bars": 2, "notes": [{"bar": 1, "b": 1.0, "n": 40, "l": 1.0}, {"bar": 1, "b": 1.0, "n": 47, "l": 1.0}]}, amap, rsch, [], [], dummy_emit)
                assert ok4 and st4["imported_notes"] == 2

            asyncio.run(run_mock())
            midi_calls = [c for c in seen_calls if c[0] == "midi_note/import_json"]
            assert len(midi_calls) == 1
            evs = midi_calls[0][1]["midi"]["midi_events"]
            assert evs[0]["b"] != evs[1]["b"], "Guitar strum stagger should separate identical chord beats"
    finally:
        srv.shutdown()

    print(f"[PASS] Neural ONNX ({avail}), Vocal FX Detective, SFZ/SF2/Surge ({lib.get('surge_patches_count', 0)} patches), MIDI Humanize & Region Fix")


if __name__ == "__main__":
    test_themes()
    test_audio_dsp_and_alignment()
    test_react_parser()
    test_sound_library_and_mastery_skill()
    test_ten_reconnects_llm_and_mcp()
    test_neural_audio_and_region_fix()
    test_full_agent_and_api_end_to_end()
    print("\nALL STAGE 1-7 VERIFICATION TESTS PASSED!")
