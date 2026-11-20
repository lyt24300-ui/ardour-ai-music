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


if __name__ == "__main__":
    test_themes()
    test_audio_dsp_and_alignment()
    test_react_parser()
    test_full_agent_and_api_end_to_end()
    print("\nALL STAGE 1-4 VERIFICATION TESTS PASSED!")
