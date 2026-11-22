"""Neural Audio Intelligence Layer — pure ONNX Runtime + NumPy + FFmpeg.

Gives text-only LLM producers real 'ears' for references, vocal FX, rhythms,
instruments, chords, and mix-vs-reference evaluation without PyTorch bloat:
- HTDemucs (FP16 weights ONNX): 4-stem separation (drums, bass, other, vocals)
- AST AudioSet (INT8 ONNX): 527-class instrument & acoustic tag recognition
- LAION CLAP (FP16 Audio + INT8 Text ONNX): zero-shot stem & vocal FX profiling
- Spotify Basic Pitch (nmp.onnx): polyphonic audio-to-MIDI & chord extraction
- Drum 16-step pattern & groove/swing extractor
- Vocal FX Detective: reverb decay/type, slap/tempo echo, bandpass/lo-fi EQ,
  saturation/dirt, stereo doubling vs reverb width, compression & pitch stability
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import BASE_DIR, FFMPEG_BIN, WORKSPACE_ROOT

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_STEM_NAMES = ("drums", "bass", "other", "vocals")
_HTDEMUCS_SR = 44100
_HTDEMUCS_SAMPLES = 343980  # 7.8s @ 44.1 kHz
_BP_SR = 22050
_BP_SAMPLES = 43844
_BP_FRAMES = 172
_BP_OVERLAP = 7680
_MIDI_OFFSET = 21

_LOCK = threading.RLock()
_SESSIONS: dict[str, Any] = {}
_CLAP_TEXT_CACHE: dict[str, np.ndarray] = {}
_AST_LABELS: Optional[dict[str, str]] = None
_CLAP_TOKENIZER: Any = None


def resolve_ai_models_dir() -> Optional[Path]:
    """Locate the local ONNX ai_models directory across BUILD/root/ARDOUR_DATA_PATH."""
    candidates: list[Path] = []
    env_dir = os.environ.get("AI_MUSIC_MODELS_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    data_env = os.environ.get("ARDOUR_DATA_PATH", "")
    for part in data_env.split(os.pathsep):
        if part.strip():
            candidates.append(Path(part.strip()) / "ai_models")
    candidates.extend([
        WORKSPACE_ROOT / "share" / "ardour9" / "ai_models",
        WORKSPACE_ROOT / "BUILD" / "share" / "ardour9" / "ai_models",
        BASE_DIR.parent / "BUILD" / "share" / "ardour9" / "ai_models",
        BASE_DIR / "data" / "ai_models",
    ])
    for cand in candidates:
        if cand.is_dir() and (cand / "htdemucs").is_dir():
            return cand
    return None


def models_available() -> dict[str, bool]:
    """Report which local ONNX models are present on disk."""
    mdir = resolve_ai_models_dir()
    if not mdir:
        return {"htdemucs": False, "ast": False, "clap": False, "basic_pitch": False}
    return {
        "htdemucs": (mdir / "htdemucs" / "htdemucs_fp16weights.onnx").is_file()
        or (mdir / "htdemucs" / "htdemucs.onnx").is_file(),
        "ast": (mdir / "ast" / "onnx" / "model_quantized.onnx").is_file()
        and (mdir / "ast" / "config.json").is_file(),
        "clap": (mdir / "clap" / "onnx" / "audio_model_fp16.onnx").is_file()
        and (mdir / "clap" / "onnx" / "text_model_quantized.onnx").is_file()
        and (mdir / "clap" / "tokenizer.json").is_file(),
        "basic_pitch": (mdir / "basic_pitch" / "nmp.onnx").is_file(),
    }


def _get_ort_session(key: str, model_path: Path) -> Any:
    import onnxruntime as ort

    with _LOCK:
        if key not in _SESSIONS:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Keep threads reasonable on 12-core laptops to avoid system hitching
            opts.intra_op_num_threads = min(6, max(2, (os.cpu_count() or 4) // 2))
            _SESSIONS[key] = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return _SESSIONS[key]


def _decode_window_stereo(
    path: str | Path,
    offset_sec: float = 0.0,
    duration_sec: float = 15.6,
    target_sr: int = 44100,
) -> tuple[np.ndarray, int]:
    """Decode an audio window via ffmpeg into (2, N) float32 array."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-ss", f"{max(0.0, offset_sec):.3f}",
        "-t", f"{max(0.5, duration_sec):.3f}",
        "-i", str(p),
        "-ac", "2",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90)
    raw = proc.stdout
    if len(raw) < 8:
        return np.zeros((2, 0), dtype=np.float32), target_sr
    samples = np.frombuffer(raw, dtype=np.float32)
    n_frames = len(samples) // 2
    stereo = samples[: n_frames * 2].reshape(n_frames, 2).T.copy()
    return stereo, target_sr


def _write_stereo_wav(path: Path, stereo_2xn: np.ndarray, sr: int = 44100) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(stereo_2xn.T, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _resample_linear(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or len(mono) < 2:
        return mono.astype(np.float32, copy=False)
    dst_len = max(1, int(round(len(mono) * dst_sr / float(src_sr))))
    idx = np.linspace(0.0, len(mono) - 1, dst_len)
    return np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)


def _db_rms(arr: np.ndarray) -> float:
    if arr.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    if rms <= 1e-12:
        return -120.0
    return round(max(-120.0, 20.0 * math.log10(rms)), 2)


# ---------------------------------------------------------------------------
# 1. HTDemucs 4-Stem Separation (ONNX)
# ---------------------------------------------------------------------------
def separate_stems_array(mix_2xn: np.ndarray, sr: int = 44100) -> dict[str, np.ndarray]:
    """Run HTDemucs ONNX on (2, N) audio @ 44.1 kHz -> {drums, bass, other, vocals}."""
    mdir = resolve_ai_models_dir()
    if not mdir:
        raise RuntimeError("Local ai_models directory not found.")
    mpath = mdir / "htdemucs" / "htdemucs_fp16weights.onnx"
    if not mpath.is_file():
        mpath = mdir / "htdemucs" / "htdemucs.onnx"
    if not mpath.is_file():
        raise FileNotFoundError("HTDemucs ONNX model not installed.")

    if mix_2xn.ndim != 2 or mix_2xn.shape[0] not in (1, 2):
        raise ValueError(f"Expected (2, N) array, got {mix_2xn.shape}")
    if mix_2xn.shape[0] == 1:
        mix_2xn = np.repeat(mix_2xn, 2, axis=0)
    if sr != _HTDEMUCS_SR:
        left = _resample_linear(mix_2xn[0], sr, _HTDEMUCS_SR)
        right = _resample_linear(mix_2xn[1], sr, _HTDEMUCS_SR)
        mix_2xn = np.stack([left, right], axis=0)

    sess = _get_ort_session("htdemucs", mpath)
    total = mix_2xn.shape[1]
    if total == 0:
        empty = np.zeros((2, 0), dtype=np.float32)
        return {name: empty for name in _STEM_NAMES}

    overlap = _HTDEMUCS_SAMPLES // 4
    stride = _HTDEMUCS_SAMPLES - overlap
    win = np.ones(_HTDEMUCS_SAMPLES, dtype=np.float32)
    fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    win[:overlap] = fade
    win[-overlap:] = fade[::-1]

    n_chunks = max(1, (total + stride - 1) // stride)
    out = np.zeros((len(_STEM_NAMES), 2, total), dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)

    for i in range(n_chunks):
        start = i * stride
        end = min(start + _HTDEMUCS_SAMPLES, total)
        chunk = mix_2xn[:, start:end]
        if chunk.shape[1] < _HTDEMUCS_SAMPLES:
            chunk = np.pad(chunk, ((0, 0), (0, _HTDEMUCS_SAMPLES - chunk.shape[1])))
        x = chunk[np.newaxis, ...].astype(np.float32, copy=False)
        stems = sess.run(["stems"], {"mix": x})[0][0]
        clen = end - start
        w = win[:clen]
        out[:, :, start:end] += stems[:, :, :clen] * w
        weight[start:end] += w

    out /= np.maximum(weight, 1e-8)
    return {name: out[i] for i, name in enumerate(_STEM_NAMES)}


def separate_stems_onnx(
    audio_path: str | Path,
    offset_sec: float = 0.0,
    duration_sec: float = 7.8,
    stems_out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Separate an audio segment via HTDemucs and optionally export stem WAV files."""
    mix, sr = _decode_window_stereo(audio_path, offset_sec=offset_sec, duration_sec=duration_sec, target_sr=_HTDEMUCS_SR)
    stems = separate_stems_array(mix, sr=sr)
    saved: dict[str, str] = {}
    if stems_out_dir is not None:
        stems_out_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in stems.items():
            out_p = stems_out_dir / f"{name}.wav"
            _write_stereo_wav(out_p, arr, sr=_HTDEMUCS_SR)
            saved[name] = str(out_p)
    return {"stems": stems, "sr": _HTDEMUCS_SR, "saved_paths": saved, "mix": mix}


# ---------------------------------------------------------------------------
# 2. AST AudioSet 527-Class Instrument & Sound Recognition (INT8 ONNX)
# ---------------------------------------------------------------------------
_GENERIC_AUDIOSET_TAGS = {
    "Music", "Musical instrument", "Silence", "Sound", "Noise", "Inside, small room",
    "Song", "Background music", "Theme music", "Soundtrack music", "Recording",
}


def _ast_log_mel(mono_16k: np.ndarray, sr: int = 16000) -> np.ndarray:
    if len(mono_16k) < 400:
        mono_16k = np.pad(mono_16k, (0, 400 - len(mono_16k)))
    preemph = np.append(mono_16k[0], mono_16k[1:] - 0.97 * mono_16k[:-1])
    win_len, hop, n_fft, n_mels, max_len = 400, 160, 512, 128, 1024
    need = (max_len - 1) * hop + win_len
    if len(preemph) < need:
        preemph = np.pad(preemph, (0, need - len(preemph)))
    else:
        preemph = preemph[:need]
    frames = np.lib.stride_tricks.sliding_window_view(preemph, win_len)[::hop][:max_len]
    frames = frames - frames.mean(axis=1, keepdims=True)
    han = np.hanning(win_len).astype(np.float32)
    spec = np.fft.rfft(frames * han, n=n_fft)
    pow_spec = spec.real**2 + spec.imag**2

    def hz_to_mel(f: Any) -> Any:
        return 1127.0 * np.log(1.0 + f / 700.0)

    def mel_to_hz(m: Any) -> Any:
        return 700.0 * (np.exp(m / 1127.0) - 1.0)

    m_pts = np.linspace(hz_to_mel(20.0), hz_to_mel(sr / 2.0), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        fb[:, m] = np.maximum(
            0.0,
            np.minimum((fft_freqs - lo) / max(ctr - lo, 1e-6), (hi - fft_freqs) / max(hi - ctr, 1e-6)),
        )
    mel_e = np.maximum(pow_spec @ fb, 1.1920929e-7)
    log_mel = np.log(mel_e).astype(np.float32)
    return ((log_mel + 4.2677393) / (4.5689974 * 2.0)).astype(np.float32)


def classify_stem_ast(mono_44k: np.ndarray, sr: int = 44100, top_k: int = 6) -> list[dict[str, Any]]:
    """Run AST AudioSet INT8 recognizer on mono audio."""
    global _AST_LABELS
    if _db_rms(mono_44k) < -55.0:
        return []
    mdir = resolve_ai_models_dir()
    if not mdir:
        return []
    mpath = mdir / "ast" / "onnx" / "model_quantized.onnx"
    cfg_path = mdir / "ast" / "config.json"
    if not mpath.is_file() or not cfg_path.is_file():
        return []

    if _AST_LABELS is None:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        _AST_LABELS = cfg.get("id2label", {})

    mono16 = _resample_linear(mono_44k, sr, 16000)
    feat = _ast_log_mel(mono16)[np.newaxis, ...]
    sess = _get_ort_session("ast", mpath)
    logits = sess.run(None, {"input_values": feat})[0][0]
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    order = np.argsort(probs)[::-1]

    specific: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for idx in order[:40]:
        lbl = str(_AST_LABELS.get(str(int(idx)), f"class_{idx}"))
        score = round(float(probs[idx]), 3)
        if score < 0.03:
            break
        entry = {"label": lbl, "score": score}
        if lbl in _GENERIC_AUDIOSET_TAGS:
            fallback.append(entry)
        else:
            specific.append(entry)
        if len(specific) >= top_k:
            break
    return specific if specific else fallback[:top_k]


# ---------------------------------------------------------------------------
# 3. CLAP Zero-Shot Audio-Text Stem & Vocal FX Recognizer (ONNX)
# ---------------------------------------------------------------------------
CLAP_ARRANGEMENT_PROMPTS = (
    "acoustic strumming guitar",
    "clean electric guitar arpeggio with chorus",
    "distorted electric guitar riff or punk power chords",
    "bright shoegaze atmospheric electric guitar",
    "acoustic piano chords ballad",
    "vintage analog synthesizer or lo-fi organ keys",
    "electronic synthesizer pad or synthwave lead",
    "orchestral strings ensemble",
)

CLAP_DRUM_PROMPTS = (
    "live acoustic rock or punk drum kit with real snare and cymbals",
    "dry lo-fi indie drum kit",
    "punchy electronic 808 or 909 trap drum machine",
    "sparse minimal click percussion without full drums",
)

CLAP_BASS_PROMPTS = (
    "picked or fingered electric bass guitar",
    "gritty overdriven fuzz rock bass guitar",
    "deep synthetic 808 sub bass",
    "electronic analog synth bass",
)

CLAP_VOCAL_FX_PROMPTS = (
    "dry close intimate vocals without heavy reverb",
    "vocals with heavy wet hall or plate reverb and spatial tail",
    "vocals with rhythmic delay echo tail",
    "lo-fi distorted telephone or megaphone bandpass vocals",
    "heavily auto-tuned pitch-corrected vocals",
    "double-tracked wide chorus vocals",
    "raw emotional punk indie rock male or female vocals",
)


def _clap_mel_features(mono: np.ndarray, sr: int = 44100) -> np.ndarray:
    sig = _resample_linear(mono, sr, 48000)
    target_n = 480000
    if len(sig) == 0:
        sig = np.zeros(target_n, dtype=np.float32)
    elif len(sig) < target_n:
        reps = int(math.ceil(target_n / len(sig)))
        sig = np.tile(sig, reps)[:target_n]
    else:
        sig = sig[:target_n]

    sig_p = np.pad(sig, (512, 512), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(sig_p, 1024)[::480][:1001]
    han = np.hanning(1024).astype(np.float32)
    spec = np.fft.rfft(frames * han, n=1024)
    pow_spec = spec.real**2 + spec.imag**2

    def hz_to_mel_slaney(f: Any) -> Any:
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        f = np.asarray(f, dtype=np.float64)
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep, f / f_sp)

    def mel_to_hz_slaney(m: Any) -> Any:
        f_sp = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    m_pts = np.linspace(hz_to_mel_slaney(50.0), hz_to_mel_slaney(14000.0), 66)
    f_pts = mel_to_hz_slaney(m_pts)
    fft_freqs = np.fft.rfftfreq(1024, d=1.0 / 48000.0)
    fb = np.zeros((513, 64), dtype=np.float32)
    for m in range(64):
        lo, ctr, hi = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        enorm = 2.0 / max(hi - lo, 1e-6)
        fb[:, m] = enorm * np.maximum(
            0.0,
            np.minimum((fft_freqs - lo) / max(ctr - lo, 1e-6), (hi - fft_freqs) / max(hi - ctr, 1e-6)),
        )
    mel = pow_spec @ fb
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, float(db.max()) - 80.0)
    return db[np.newaxis, np.newaxis, :, :].astype(np.float32)


def _get_clap_text_embeds(prompts: tuple[str, ...] | list[str]) -> Optional[np.ndarray]:
    global _CLAP_TOKENIZER
    mdir = resolve_ai_models_dir()
    if not mdir:
        return None
    t_path = mdir / "clap" / "onnx" / "text_model_quantized.onnx"
    tok_path = mdir / "clap" / "tokenizer.json"
    if not t_path.is_file() or not tok_path.is_file():
        return None

    from tokenizers import Tokenizer

    if _CLAP_TOKENIZER is None:
        _CLAP_TOKENIZER = Tokenizer.from_file(str(tok_path))
    sess = _get_ort_session("clap_text", t_path)
    out: list[np.ndarray] = []
    with _LOCK:
        for p in prompts:
            if p not in _CLAP_TEXT_CACHE:
                ids = np.array([_CLAP_TOKENIZER.encode(p).ids], dtype=np.int64)
                feed: dict[str, Any] = {"input_ids": ids}
                for inp in sess.get_inputs():
                    if inp.name == "attention_mask":
                        feed["attention_mask"] = np.ones_like(ids, dtype=np.int64)
                emb = sess.run(None, feed)[0][0]
                emb = emb / max(float(np.linalg.norm(emb)), 1e-9)
                _CLAP_TEXT_CACHE[p] = emb.astype(np.float32)
            out.append(_CLAP_TEXT_CACHE[p])
    return np.stack(out)


def score_clap_prompts(
    mono_44k: np.ndarray,
    prompts: tuple[str, ...] | list[str],
    sr: int = 44100,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Zero-shot CLAP classification over a list of candidate descriptions."""
    if _db_rms(mono_44k) < -55.0:
        return []
    mdir = resolve_ai_models_dir()
    if not mdir:
        return []
    a_path = mdir / "clap" / "onnx" / "audio_model_fp16.onnx"
    if not a_path.is_file():
        return []
    te = _get_clap_text_embeds(prompts)
    if te is None:
        return []

    feat = _clap_mel_features(mono_44k, sr=sr)
    sess = _get_ort_session("clap_audio", a_path)
    ae = sess.run(None, {"input_features": feat})[0][0]
    ae = ae / max(float(np.linalg.norm(ae)), 1e-9)

    sims = (te @ ae) * 14.285714
    probs = np.exp(sims - np.max(sims))
    probs /= np.sum(probs)
    order = np.argsort(probs)[::-1]
    return [
        {"description": prompts[int(j)], "probability": round(float(probs[j]), 3)}
        for j in order[:top_k]
    ]


# ---------------------------------------------------------------------------
# 4. Spotify Basic Pitch (nmp.onnx) Polyphonic MIDI & Chord Extraction
# ---------------------------------------------------------------------------
_CHORD_TEMPLATES: list[tuple[str, tuple[int, ...]]] = [
    ("maj", (0, 4, 7)),
    ("min", (0, 3, 7)),
    ("5", (0, 7)),
    ("maj7", (0, 4, 7, 11)),
    ("min7", (0, 3, 7, 10)),
    ("7", (0, 4, 7, 10)),
    ("sus2", (0, 2, 7)),
    ("sus4", (0, 5, 7)),
]


def _midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def _infer_chord_label(pitch_weights: np.ndarray) -> str:
    if np.sum(pitch_weights) <= 1e-6:
        return "N.C."
    pw = pitch_weights / np.sum(pitch_weights)
    best_name = NOTE_NAMES[int(np.argmax(pw))]
    best_score = -1.0
    for root in range(12):
        for suffix, intervals in _CHORD_TEMPLATES:
            tones = {(root + iv) % 12 for iv in intervals}
            in_sum = sum(float(pw[t]) for t in tones)
            # Small bonus for root dominance and slight penalty for cluttered non-chord tones
            score = in_sum + 0.18 * float(pw[root]) - 0.04 * len(intervals)
            if score > best_score:
                best_score = score
                best_name = f"{NOTE_NAMES[root]}{'' if suffix == 'maj' else ('m' if suffix == 'min' else suffix)}"
    return best_name


def transcribe_midi_basic_pitch(
    mono_44k: np.ndarray,
    sr: int = 44100,
    min_midi: int = 28,
    max_midi: int = 88,
    onset_thresh: float = 0.42,
    frame_thresh: float = 0.34,
    min_note_len_frames: int = 9,
    offset_base_sec: float = 0.0,
) -> dict[str, Any]:
    """Polyphonic audio-to-MIDI transcription using Spotify Basic Pitch ONNX + pure NumPy."""
    if _db_rms(mono_44k) < -52.0:
        return {"note_count": 0, "notes": [], "active_pitch_classes": [], "estimated_chords": []}
    mdir = resolve_ai_models_dir()
    if not mdir:
        return {"note_count": 0, "notes": [], "active_pitch_classes": [], "estimated_chords": []}
    mpath = mdir / "basic_pitch" / "nmp.onnx"
    if not mpath.is_file():
        return {"note_count": 0, "notes": [], "active_pitch_classes": [], "estimated_chords": []}

    sig = _resample_linear(mono_44k, sr, _BP_SR)
    # Normalize softly without blowing up quiet noise
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    if peak > 1e-4:
        sig = sig / max(peak, 0.25)

    stride = _BP_SAMPLES - _BP_OVERLAP
    half_ov = _BP_OVERLAP // 2
    n_ov_frames = max(0, int(round((half_ov / _BP_SAMPLES) * _BP_FRAMES)))

    sess = _get_ort_session("basic_pitch", mpath)
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    # Verified: :1 is note (frame sustain), :2 is onset (attack peak), :0 is contour
    note_out_idx = out_names.index("StatefulPartitionedCall:1") if "StatefulPartitionedCall:1" in out_names else 1
    onset_out_idx = out_names.index("StatefulPartitionedCall:2") if "StatefulPartitionedCall:2" in out_names else 2

    pad_front = np.zeros(half_ov, dtype=np.float32)
    padded = np.concatenate([pad_front, sig])
    note_chunks: list[np.ndarray] = []
    onset_chunks: list[np.ndarray] = []

    pos = 0
    while pos < len(padded):
        seg = padded[pos : pos + _BP_SAMPLES]
        if len(seg) < _BP_SAMPLES:
            seg = np.pad(seg, (0, _BP_SAMPLES - len(seg)))
        outs = sess.run(None, {in_name: seg.reshape(1, _BP_SAMPLES, 1)})
        n_mat = outs[note_out_idx][0]
        o_mat = outs[onset_out_idx][0]
        if n_ov_frames > 0 and n_mat.shape[0] > 2 * n_ov_frames:
            n_mat = n_mat[n_ov_frames:-n_ov_frames]
            o_mat = o_mat[n_ov_frames:-n_ov_frames]
        note_chunks.append(n_mat)
        onset_chunks.append(o_mat)
        pos += stride
        if len(padded) <= _BP_SAMPLES:
            break

    frames = np.concatenate(note_chunks, axis=0)
    onsets = np.concatenate(onset_chunks, axis=0)
    fps = _BP_FRAMES / (_BP_SAMPLES / _BP_SR)  # ~86.13 Hz
    dur_audio = len(sig) / float(_BP_SR)
    max_frames = min(frames.shape[0], int(round(dur_audio * fps)))
    frames = frames[:max_frames].copy()
    onsets = onsets[:max_frames].copy()
    if frames.shape[0] < 4:
        return {"note_count": 0, "notes": [], "active_pitch_classes": [], "estimated_chords": []}

    min_idx = max(1, int(min_midi - _MIDI_OFFSET))  # avoid edge bin 0 noise
    max_idx = min(87, int(max_midi - _MIDI_OFFSET) + 1)
    onsets[:, :min_idx] = 0.0
    frames[:, :min_idx] = 0.0
    onsets[:, max_idx:] = 0.0
    frames[:, max_idx:] = 0.0

    # Infer additional onsets from sharp frame jumps (Basic Pitch get_infered_onsets)
    diffs = []
    for n in (1, 2):
        app = np.concatenate([np.zeros((n, 88), dtype=np.float32), frames], axis=0)
        diffs.append(app[n:, :] - app[:-n, :])
    fdiff = np.min(diffs, axis=0)
    fdiff[fdiff < 0] = 0.0
    fdiff[:2, :] = 0.0
    mx_diff = float(np.max(fdiff))
    if mx_diff > 1e-6:
        fdiff = float(np.max(onsets)) * fdiff / mx_diff
    onsets = np.maximum(onsets, fdiff)

    # Local time maxima along axis 0 (replaces scipy.signal.argrelmax)
    peaks_mask = np.zeros_like(onsets, dtype=bool)
    peaks_mask[1:-1, :] = (onsets[1:-1, :] > onsets[:-2, :]) & (onsets[1:-1, :] > onsets[2:, :])
    peak_mat = np.where(peaks_mask, onsets, 0.0)
    t_idxs, f_idxs = np.where(peak_mat >= onset_thresh)

    rem = frames.copy()
    n_t = frames.shape[0]
    energy_tol = 11
    raw_events: list[tuple[int, int, int, float]] = []

    for start_i, freq_i in zip(t_idxs[::-1], f_idxs[::-1]):
        if start_i >= n_t - 1:
            continue
        i = int(start_i) + 1
        k = 0
        while i < n_t - 1 and k < energy_tol:
            if rem[i, freq_i] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        end_i = i - k
        if end_i - start_i < min_note_len_frames:
            continue
        amp = float(np.mean(frames[start_i:end_i, freq_i]))
        rem[start_i:end_i, freq_i] = 0.0
        if freq_i < 87:
            rem[start_i:end_i, freq_i + 1] = 0.0
        if freq_i > 0:
            rem[start_i:end_i, freq_i - 1] = 0.0
        raw_events.append((int(start_i), int(end_i), int(freq_i + _MIDI_OFFSET), amp))

    # Melodia trick for un-peaked sustained tones
    guard = 0
    while float(np.max(rem)) > max(frame_thresh, 0.42) and guard < 120:
        guard += 1
        i_mid, freq_i = np.unravel_index(int(np.argmax(rem)), rem.shape)
        rem[i_mid, freq_i] = 0.0
        i = int(i_mid) + 1
        k = 0
        while i < n_t - 1 and k < energy_tol:
            k = k + 1 if rem[i, freq_i] < frame_thresh else 0
            rem[i, freq_i] = 0.0
            i += 1
        i_end = i - 1 - k
        i = int(i_mid) - 1
        k = 0
        while i > 0 and k < energy_tol:
            k = k + 1 if rem[i, freq_i] < frame_thresh else 0
            rem[i, freq_i] = 0.0
            i -= 1
        i_start = max(0, i + 1 + k)
        if i_end - i_start >= min_note_len_frames:
            amp = float(np.mean(frames[i_start:i_end, freq_i]))
            raw_events.append((i_start, i_end, int(freq_i + _MIDI_OFFSET), amp))

    raw_events.sort(key=lambda x: (x[0], x[2]))
    notes_out: list[dict[str, Any]] = []
    pc_hist = np.zeros(12, dtype=np.float64)
    for s_f, e_f, midi_p, amp in raw_events:
        s_t = round(offset_base_sec + s_f / fps, 3)
        e_t = round(offset_base_sec + e_f / fps, 3)
        dur = round(max(0.05, e_t - s_t), 3)
        pc_hist[midi_p % 12] += dur * max(amp, 0.2)
        notes_out.append({
            "midi": midi_p,
            "note": _midi_to_name(midi_p),
            "start_sec": s_t,
            "dur_sec": dur,
            "vel": int(max(30, min(127, round(amp * 135)))),
        })

    # Estimate harmonic chords in ~2-second slices across the window
    chords_seq: list[dict[str, Any]] = []
    slice_sec = 2.0
    n_slices = max(1, int(math.ceil(dur_audio / slice_sec)))
    for sl in range(n_slices):
        t0 = offset_base_sec + sl * slice_sec
        t1 = t0 + slice_sec
        pw = np.zeros(12, dtype=np.float64)
        for n in notes_out:
            ns, ne = n["start_sec"], n["start_sec"] + n["dur_sec"]
            ov = min(ne, t1) - max(ns, t0)
            if ov > 0:
                pw[n["midi"] % 12] += ov * (n["vel"] / 127.0)
        if np.sum(pw) > 0.05:
            ch = _infer_chord_label(pw)
            chords_seq.append({"time_sec": round(t0, 2), "chord": ch})

    pc_order = np.argsort(pc_hist)[::-1]
    active_pcs = [NOTE_NAMES[int(pc)] for pc in pc_order[:7] if pc_hist[pc] > 0.05]

    midis = [n["midi"] for n in notes_out]
    return {
        "note_count": len(notes_out),
        "pitch_range": f"{_midi_to_name(min(midis))}..{_midi_to_name(max(midis))}" if midis else "",
        "active_pitch_classes": active_pcs,
        "estimated_chords": chords_seq,
        "notes_sample": notes_out[:28],
    }


# ---------------------------------------------------------------------------
# 5. Drum Pattern 16-Step Grid & Groove Extractor
# ---------------------------------------------------------------------------
def extract_drum_pattern_grid(drums_2xn: np.ndarray, bpm: float, sr: int = 44100) -> dict[str, Any]:
    """Extract kick/snare/hat 16-step pattern and swing from separated drum stem."""
    if drums_2xn.size == 0 or _db_rms(drums_2xn) < -52.0 or bpm <= 20:
        return {"has_drums": False}

    mono = drums_2xn.mean(axis=0).astype(np.float32)
    hop = 256
    n_fft = 1024
    if len(mono) < n_fft * 4:
        return {"has_drums": False}

    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop]
    win = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * win, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fps = sr / float(hop)

    def band_onsets(f_lo: float, f_hi: float, min_gap_sec: float = 0.08, pct: float = 86.0, k_mad: float = 4.0) -> np.ndarray:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        e = np.sum(mag[:, mask], axis=1)
        flux = np.maximum(0.0, np.diff(e, prepend=e[0]))
        med = float(np.median(flux))
        mad = float(np.median(np.abs(flux - med))) + 1e-6
        thresh = max(float(np.percentile(flux, pct)), med + k_mad * mad, float(np.max(flux)) * 0.18)
        min_gap = max(2, int(min_gap_sec * fps))
        peaks: list[int] = []
        last_i = -min_gap
        for i in range(1, len(flux) - 1):
            if flux[i] >= thresh and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
                if i - last_i >= min_gap:
                    peaks.append(i)
                    last_i = i
        return np.array(peaks, dtype=np.float64) / fps

    kick_t = band_onsets(25.0, 120.0, min_gap_sec=0.125, pct=88.0, k_mad=4.5)
    snare_t = band_onsets(190.0, 2100.0, min_gap_sec=0.145, pct=89.0, k_mad=4.8)
    hat_t = band_onsets(5200.0, 16000.0, min_gap_sec=0.075, pct=84.0, k_mad=3.8)

    # Align phase to kick+snare anchor so bar step 0 lands on downbeat
    sixteenth_sec = 60.0 / (bpm * 4.0)
    bar_sec = 16.0 * sixteenth_sec
    anchor_t = np.concatenate([kick_t, snare_t])
    best_phase = 0.0
    best_score = -1.0
    if len(anchor_t) > 0:
        for ph in np.linspace(0.0, sixteenth_sec, 24, endpoint=False):
            # Score distance to 8th notes weighted slightly higher on beats 1 and 3 for kicks
            q = np.round((anchor_t - ph) / sixteenth_sec).astype(int) % 16
            score = sum(2 if s in (0, 4, 8, 12) else (1 if s % 2 == 0 else 0) for s in q)
            if score > best_score:
                best_score = score
                best_phase = float(ph)

    def quantize_steps(times: np.ndarray, min_ratio: float = 0.48) -> tuple[list[int], float]:
        if len(times) == 0:
            return [], 0.0
        n_bars = max(1.0, (len(mono) / sr) / bar_sec)
        counts = np.zeros(16, dtype=int)
        offsets_ms: list[float] = []
        for t_ev in times:
            pos_16 = (t_ev - best_phase) / sixteenth_sec
            idx = int(round(pos_16)) % 16
            counts[idx] += 1
            offsets_ms.append((pos_16 - round(pos_16)) * sixteenth_sec * 1000.0)
        min_hits = max(1, int(math.ceil(n_bars * min_ratio)))
        steps = [s for s in range(16) if counts[s] >= min_hits]
        swing_ms = round(float(np.std(offsets_ms)) if offsets_ms else 0.0, 1)
        return steps, swing_ms

    kick_steps, k_swing = quantize_steps(kick_t, min_ratio=0.45)
    snare_steps, s_swing = quantize_steps(snare_t, min_ratio=0.45)
    hat_steps, h_swing = quantize_steps(hat_t, min_ratio=0.40)

    style = "standard_rock_pop"
    if 8 in snare_steps and 4 not in snare_steps and 12 not in snare_steps:
        style = "half_time_backbeat"
    elif len(hat_steps) >= 10:
        style = "driving_16th_or_fast_punk"
    elif len(hat_steps) >= 6 and len(kick_steps) >= 3:
        style = "driving_8th_rock_indie"
    elif len(kick_steps) + len(snare_steps) <= 2:
        style = "sparse_minimal"

    return {
        "has_drums": True,
        "bpm_used": round(bpm, 1),
        "style": style,
        "kick_steps_16": kick_steps,
        "snare_steps_16": snare_steps,
        "hat_steps_16": hat_steps,
        "microtiming_jitter_ms": round(float(np.mean([k_swing, s_swing, h_swing])), 1),
        "kick_hits_per_sec": round(len(kick_t) / max(1.0, len(mono) / sr), 2),
        "snare_hits_per_sec": round(len(snare_t) / max(1.0, len(mono) / sr), 2),
    }


# ---------------------------------------------------------------------------
# 6. Vocal FX Detective (Reverb, Delay, Bandpass/Distortion, Comp, Doubling, Pitch)
# ---------------------------------------------------------------------------
def analyze_vocal_fx_profile(
    vocal_2xn: np.ndarray,
    sr: int = 44100,
    inst_rms_db: Optional[float] = None,
    use_clap: bool = True,
) -> dict[str, Any]:
    """Inspect vocal audio/stem and reconstruct the exact vocal FX & mix recipe."""
    if vocal_2xn.ndim == 1:
        vocal_2xn = np.stack([vocal_2xn, vocal_2xn], axis=0)
    elif vocal_2xn.shape[0] != 2 and vocal_2xn.shape[1] == 2:
        vocal_2xn = vocal_2xn.T

    if vocal_2xn.shape[1] < sr // 2 or _db_rms(vocal_2xn) < -55.0:
        return {"has_vocal": False}

    left = vocal_2xn[0].astype(np.float64)
    right = vocal_2xn[1].astype(np.float64)
    mono = 0.5 * (left + right)
    side = 0.5 * (left - right)

    frame_ms = 20
    frame_n = max(128, int(sr * frame_ms / 1000.0))
    n_frames = len(mono) // frame_n
    if n_frames < 10:
        return {"has_vocal": False}

    m_mat = mono[: n_frames * frame_n].reshape(n_frames, frame_n)
    s_mat = side[: n_frames * frame_n].reshape(n_frames, frame_n)
    m_rms = np.sqrt(np.mean(m_mat**2, axis=1))
    s_rms = np.sqrt(np.mean(s_mat**2, axis=1))
    frame_db = np.array([max(-110.0, 20.0 * math.log10(max(float(r), 1e-12))) for r in m_rms])

    noise_floor = float(np.percentile(frame_db, 8))
    peak98 = float(np.percentile(frame_db, 98))
    # Cap threshold below peak98 - 8 dB so continuous chorus singing stays marked active
    active_thresh = max(min(noise_floor + 10.0, peak98 - 8.0), peak98 - 22.0, -45.0)
    active_mask = frame_db >= active_thresh
    if int(np.sum(active_mask)) < 4:
        return {"has_vocal": False}

    # Measure active vocal dynamics, EQ bands, and stereo doubling
    act_samples = m_mat[active_mask].ravel()
    act_peak_db = 20.0 * math.log10(max(float(np.max(np.abs(act_samples))), 1e-9))
    act_rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(act_samples**2))), 1e-9))
    crest_db = round(act_peak_db - act_rms_db, 2)

    if crest_db < 7.5:
        comp_style = "heavy_compressed_brickwall"
    elif crest_db < 11.5:
        comp_style = "tight_studio_compressed"
    elif crest_db < 15.0:
        comp_style = "moderate_natural_compression"
    else:
        comp_style = "raw_uncompressed_dynamic"

    act_side_ratio = float(np.mean(s_rms[active_mask]) / (np.mean(m_rms[active_mask]) + 1e-9))
    # Post-phrase pause frames: 40ms..1200ms immediately after an active frame drops
    tail_frames: list[int] = []
    reverb_decays_ms: list[float] = []
    echo_lags_ms: list[float] = []

    i = 0
    while i < n_frames:
        if active_mask[i]:
            while i < n_frames and active_mask[i]:
                i += 1
            # i is first inactive frame right after vocal phrase
            if i < n_frames:
                start_db = float(frame_db[max(0, i - 1)])
                decay_frames = 0
                seg_env = []
                max_look = min(n_frames, i + 65)  # up to 1.3s
                for j in range(i, max_look):
                    if active_mask[j]:
                        break
                    tail_frames.append(j)
                    seg_env.append(float(m_rms[j]))
                    if frame_db[j] > start_db - 20.0 and frame_db[j] > noise_floor + 4.0:
                        decay_frames += 1
                if 2 <= decay_frames <= 65:
                    reverb_decays_ms.append(decay_frames * frame_ms)
                # Detect delay echo spike in 80..650ms window after vocal drop
                if len(seg_env) >= 15:
                    env = np.array(seg_env)
                    for k in range(4, min(33, len(env) - 1)):
                        if env[k] > env[k - 1] * 1.28 and env[k] > env[k + 1] * 1.1 and env[k] > np.mean(env[:3]) * 0.25:
                            echo_lags_ms.append(k * frame_ms)
                            break
        else:
            i += 1

    tail_ms = round(float(np.median(reverb_decays_ms)) if reverb_decays_ms else 120.0, 1)
    if tail_ms < 180:
        reverb_type = "dry_intimate"
    elif tail_ms < 420:
        reverb_type = "room_short"
    elif tail_ms < 900:
        reverb_type = "plate_or_medium_space"
    else:
        reverb_type = "long_hall_wet"

    tail_side_ratio = (
        float(np.mean(s_rms[tail_frames]) / (np.mean(m_rms[tail_frames]) + 1e-9))
        if tail_frames
        else act_side_ratio
    )
    if act_side_ratio > 0.28:
        stereo_mode = "wide_doubled_or_chorus_vocal"
    elif tail_side_ratio > act_side_ratio * 1.5 and tail_side_ratio > 0.16:
        stereo_mode = "centered_vocal_into_stereo_reverb_delay"
    else:
        stereo_mode = "focused_mono_center"

    # Spectral tilt & telephone/lo-fi bandpass analysis on active vocal chunks
    n_fft = 2048
    act_mono_chunks = m_mat[active_mask]
    win = np.hanning(min(frame_n * 2, n_fft)).astype(np.float64)
    # Use 2048 chunks built from active samples
    if len(act_samples) < n_fft:
        act_samples = np.pad(act_samples, (0, n_fft - len(act_samples)))
    step = n_fft // 2
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    cnt = 0
    w_fft = np.hanning(n_fft)
    for off in range(0, len(act_samples) - n_fft + 1, step):
        sp = np.fft.rfft(act_samples[off : off + n_fft] * w_fft)
        acc += sp.real**2 + sp.imag**2
        cnt += 1
        if cnt >= 120:
            break
    avg_p = acc / max(1, cnt)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    tot = float(np.sum(avg_p)) + 1e-12

    def band_rel(f0: float, f1: float) -> float:
        m = (freqs >= f0) & (freqs < f1)
        return round(10.0 * math.log10(max(float(np.sum(avg_p[m])) / tot, 1e-9)), 2)

    low_cut_band = band_rel(30.0, 140.0)
    mud_band = band_rel(200.0, 420.0)
    mid_core = band_rel(500.0, 2200.0)
    presence_band = band_rel(2200.0, 5000.0)
    sib_band = band_rel(5000.0, 9000.0)
    air_band = band_rel(9000.0, 18000.0)
    centroid = float(np.sum(freqs * np.sqrt(avg_p)) / (np.sum(np.sqrt(avg_p)) + 1e-12))

    # Bandpass / telephone / lo-fi grit detection
    is_bandpass_lofi = (air_band < mid_core - 18.0) and (low_cut_band < mid_core - 16.0)
    eq_character = "telephone_or_lofi_bandpass" if is_bandpass_lofi else (
        "bright_airy_modern" if air_band > mid_core - 12.0 else (
            "warm_dark_vintage" if presence_band < mid_core - 6.0 else "balanced_present"
        )
    )

    # Pitch stability (autocorrelation F0 tracking)
    f0s: list[float] = []
    min_lag = max(1, int(sr / 620.0))
    max_lag = max(min_lag + 4, int(sr / 85.0))
    win_f0 = max(max_lag + 64, int(sr * 0.04))
    act_indices = np.where(active_mask)[0]
    if len(act_indices) > 90:
        act_indices = act_indices[:: max(1, len(act_indices) // 90)]
    for fi in act_indices:
        s0 = int(fi * frame_n)
        if s0 + win_f0 > len(mono):
            continue
        seg = mono[s0 : s0 + win_f0]
        seg = seg - np.mean(seg)
        en = float(np.dot(seg, seg))
        if en < 1e-6:
            continue
        ac = np.correlate(seg, seg, mode="full")[win_f0 - 1 :]
        sub = ac[min_lag : min(max_lag, len(ac))]
        if len(sub) == 0:
            continue
        pk = int(np.argmax(sub))
        if sub[pk] / en > 0.48:
            f0s.append(sr / float(min_lag + pk))

    pitch_cents_err: list[float] = []
    midi_vals: list[float] = []
    for hz in f0s:
        m = 69.0 + 12.0 * math.log2(max(hz, 40.0) / 440.0)
        midi_vals.append(m)
        pitch_cents_err.append(abs(m - round(m)) * 100.0)

    med_cent_err = round(float(np.median(pitch_cents_err)) if pitch_cents_err else 28.0, 1)
    if med_cent_err < 12.0 and len(f0s) >= 8:
        tuning_style = "tight_pitch_tuned_or_autotune"
    elif med_cent_err < 21.0 and len(f0s) >= 8:
        tuning_style = "mild_natural_pitch_control"
    else:
        tuning_style = "raw_live_uncorrected"

    clap_fx: list[dict[str, Any]] = []
    if use_clap:
        clap_fx = score_clap_prompts(mono.astype(np.float32), CLAP_VOCAL_FX_PROMPTS, sr=sr, top_k=4)

    vox_vs_inst = round(act_rms_db - float(inst_rms_db), 2) if inst_rms_db is not None else None

    # Build concrete Ardour FX chain recommendations for the LLM
    chain: list[str] = [
        f"1. EQ (TDR Nova / a-eq): HPF {'160-260 Hz' if is_bandpass_lofi else '80-100 Hz'}; "
        + ("cut mud -3dB @ 280Hz; " if mud_band > mid_core - 2.0 else "")
        + ("de-ess -3..5dB @ 6-7.5kHz; " if sib_band > -13.0 else "")
        + ("LPF 4.5-6.5kHz + +3dB @ 1.5-2.5kHz for lo-fi/radio tone." if is_bandpass_lofi else "presence +2dB @ 3-4kHz."),
        f"2. Pitch (SnapTune.vst3): "
        + ("enable key-locked hard/medium retune (Retune Speed ~15-30ms) matching song key." if "tight" in tuning_style else "gentle transparent pitch assist (Retune ~50-80ms) or bypass for raw grit."),
        f"3. Compression (TDR Molotok / Kotelnikov / a-comp): ratio "
        + ("5:1..8:1, fast attack 5-10ms, heavy squeeze (target 7-10dB GR)" if "heavy" in comp_style or "tight" in comp_style else "3:1, 15ms attack, 4-6dB GR"),
        "4. Saturation / Grit (Airwindows Consolidated ToTape/PurestDrive/Mackity or NeuralAmpModeler subtle): "
        + ("add warm/dirty overdrive harmonics to match lo-fi/punk texture." if (is_bandpass_lofi or any("distort" in c["description"] for c in clap_fx[:2])) else "light analog tape warmth only."),
        f"5. Reverb Send ({reverb_type}, measured tail ~{tail_ms} ms): "
        + ("keep dry (-26..-30dB send to DragonflyRoom)" if reverb_type == "dry_intimate" else f"DragonflyPlateReverb / Hall send (-16..-22dB, decay ~{round(tail_ms/1000.0, 2)}s)"),
    ]
    if echo_lags_ms:
        med_lag = round(float(np.median(echo_lags_ms)), 0)
        chain.append(f"6. Delay Send (a-delay): detected pause echo ~{med_lag} ms -> set slapback/tempo delay send (-18..-24 dB).")
    if stereo_mode == "wide_doubled_or_chorus_vocal":
        chain.append("7. Doubling / Width (TAL-Chorus-LX or L/R pitch-detuned doubles): light stereo chorus/width on vocal bus.")

    return {
        "has_vocal": True,
        "vocal_active_rms_db": round(act_rms_db, 2),
        "vocal_to_instrumental_db": vox_vs_inst,
        "active_crest_factor_db": crest_db,
        "compression_style": comp_style,
        "eq_character": eq_character,
        "spectral_bands_db": {
            "low_30_140": low_cut_band,
            "mud_200_420": mud_band,
            "mid_500_2200": mid_core,
            "presence_2200_5000": presence_band,
            "sibilance_5000_9000": sib_band,
            "air_9000_18000": air_band,
            "centroid_hz": round(centroid, 1),
        },
        "reverb_tail_ms": tail_ms,
        "reverb_type": reverb_type,
        "delay_detected": bool(echo_lags_ms),
        "delay_lag_ms": round(float(np.median(echo_lags_ms)), 0) if echo_lags_ms else None,
        "stereo_mode": stereo_mode,
        "active_side_to_mid": round(act_side_ratio, 3),
        "tail_side_to_mid": round(tail_side_ratio, 3),
        "tuning_style": tuning_style,
        "pitch_deviation_cents": med_cent_err,
        "pitch_range_midi": [int(round(np.percentile(midi_vals, 10))), int(round(np.percentile(midi_vals, 90)))] if len(midi_vals) >= 5 else [],
        "clap_fx_tags": clap_fx,
        "recommended_ardour_chain": chain,
    }


# ---------------------------------------------------------------------------
# 7. Complete Deep Reference Analyzer + Unified Vocal Harmony + Mix Compare
# ---------------------------------------------------------------------------
def _pick_representative_windows(audio_path: Path, total_dur: float, sections: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Select quiet active verse window & loudest main chorus window (skipping pure intro/outro tails)."""
    if total_dur <= 11.0:
        return [("main", 0.0)]
    if not sections:
        mid = max(0.0, total_dur * 0.45 - 3.9)
        return [("verse_or_quiet", max(4.0, total_dur * 0.2)), ("chorus_or_peak", min(mid, max(0.0, total_dur - 7.8)))]

    # Prefer non-extreme-edge sections when picking chorus peak (avoid t=0 intro & last 12% outro)
    core_cands = [
        s for s in sections
        if float(s.get("rms_db", -90.0)) > -42.0
        and float(s.get("start_sec", 0.0)) >= min(4.0, total_dur * 0.08)
        and float(s.get("start_sec", 0.0)) <= total_dur * 0.88
    ] or [s for s in sections if float(s.get("rms_db", -90.0)) > -42.0] or sections

    loudest = max(core_cands, key=lambda s: float(s.get("rms_db", -90.0)))
    chorus_off = max(0.0, min(float(loudest.get("start_sec", total_dur * 0.5)), total_dur - 7.8))

    # Active verse/quieter window distinct from chorus peak (favor earlier verse)
    active_cands = [
        s for s in core_cands
        if abs(float(s.get("start_sec", 0.0)) - chorus_off) >= 7.5
    ]
    quiet_cand = sorted(active_cands, key=lambda x: (float(x.get("rms_db", 0.0)), float(x.get("start_sec", 0.0))))[0] if active_cands else None

    windows: list[tuple[str, float]] = []
    if quiet_cand is not None:
        q_off = max(0.0, min(float(quiet_cand.get("start_sec", 0.0)), total_dur - 7.8))
        windows.append(("verse_or_quiet", q_off))
    windows.append(("chorus_or_peak", chorus_off))
    return windows


def deep_analyze_reference(
    audio_path: str | Path,
    bpm_hint: Optional[float] = None,
    existing_sections: Optional[list[dict[str, Any]]] = None,
    save_stems_dir: Optional[Path] = None,
    custom_offset_sec: Optional[float] = None,
    custom_duration_sec: float = 7.8,
) -> dict[str, Any]:
    """Full neural blueprint of a reference track using HTDemucs + AST + CLAP + Basic Pitch."""
    from .audio_analyzer import compute_energy_sections, estimate_bpm, load_audio_pcm

    p = Path(audio_path)
    avail = models_available()
    if not avail["htdemucs"]:
        return {"error": "HTDemucs ONNX model is not available locally.", "models_available": avail}

    # Fast header/BPM/sections scan when needed
    if bpm_hint is None or existing_sections is None:
        stereo_full, sr_full = load_audio_pcm(p, target_sr=44100, max_seconds=240.0)
        mono_full = stereo_full.mean(axis=1).astype(np.float32)
        total_dur = len(mono_full) / float(sr_full)
        if bpm_hint is None:
            bpm_hint = float(estimate_bpm(mono_full, sr_full).get("bpm", 120.0))
        if existing_sections is None:
            existing_sections = compute_energy_sections(mono_full, sr_full)
    else:
        total_dur = float(existing_sections[-1]["end_sec"]) if existing_sections else 60.0

    if custom_offset_sec is not None:
        windows = [("custom_window", float(custom_offset_sec))]
    else:
        windows = _pick_representative_windows(p, total_dur, existing_sections or [])

    window_reports: list[dict[str, Any]] = []
    chorus_stems_bundle: Optional[dict[str, Any]] = None

    for label, off in windows:
        dur_req = float(custom_duration_sec) if custom_offset_sec is not None else 7.8
        sep_dir = (save_stems_dir / label) if save_stems_dir else None
        sep = separate_stems_onnx(p, offset_sec=off, duration_sec=dur_req, stems_out_dir=sep_dir)
        stems = sep["stems"]
        mix = sep["mix"]
        if label in ("chorus_or_peak", "main", "custom_window"):
            chorus_stems_bundle = sep

        drums_mono = stems["drums"].mean(axis=0)
        bass_mono = stems["bass"].mean(axis=0)
        other_mono = stems["other"].mean(axis=0)
        vocals_mono = stems["vocals"].mean(axis=0)
        inst_only = stems["drums"] + stems["bass"] + stems["other"]
        inst_rms = _db_rms(inst_only)

        report: dict[str, Any] = {
            "window": label,
            "offset_sec": round(off, 2),
            "stem_levels_rms_db": {k: _db_rms(v) for k, v in stems.items()},
            "ast_instruments": {
                "drums": classify_stem_ast(drums_mono, top_k=5),
                "bass": classify_stem_ast(bass_mono, top_k=5),
                "other": classify_stem_ast(other_mono, top_k=6),
                "vocals": classify_stem_ast(vocals_mono, top_k=5),
            },
            "clap_timbre": {
                "arrangement": score_clap_prompts(other_mono, CLAP_ARRANGEMENT_PROMPTS, top_k=4),
                "drums": score_clap_prompts(drums_mono, CLAP_DRUM_PROMPTS, top_k=3),
                "bass": score_clap_prompts(bass_mono, CLAP_BASS_PROMPTS, top_k=3),
            },
            "drum_pattern": extract_drum_pattern_grid(stems["drums"], bpm=float(bpm_hint)),
            "bass_midi": transcribe_midi_basic_pitch(
                bass_mono, min_midi=28, max_midi=64, onset_thresh=0.40, frame_thresh=0.32, min_note_len_frames=8, offset_base_sec=off
            ),
            "chords_and_melody_midi": transcribe_midi_basic_pitch(
                other_mono, min_midi=40, max_midi=92, onset_thresh=0.42, frame_thresh=0.34, min_note_len_frames=9, offset_base_sec=off
            ),
            "vocal_fx_profile": analyze_vocal_fx_profile(stems["vocals"], sr=44100, inst_rms_db=inst_rms, use_clap=True),
        }
        if sep["saved_paths"]:
            report["saved_stem_wavs"] = sep["saved_stem_wavs"] if "saved_stem_wavs" in sep else sep["saved_paths"]
        window_reports.append(report)

    # Synthesize concise overall summary picking active full-band & active vocal windows
    vocal_wins = [w for w in window_reports if (w.get("vocal_fx_profile") or {}).get("has_vocal")]
    best_vocal_win = (
        max(vocal_wins, key=lambda w: float(w["vocal_fx_profile"].get("vocal_active_rms_db", -90.0)))
        if vocal_wins
        else window_reports[-1]
    )
    arr_win = max(
        window_reports,
        key=lambda w: float(w["stem_levels_rms_db"].get("other", -90.0)) + 0.5 * float(w["stem_levels_rms_db"].get("drums", -90.0)),
    )
    drums_win = max(window_reports, key=lambda w: float(w["stem_levels_rms_db"].get("drums", -90.0)))
    bass_win = max(window_reports, key=lambda w: float(w["stem_levels_rms_db"].get("bass", -90.0)))

    inst_summary: list[str] = []
    if arr_win["clap_timbre"].get("arrangement"):
        inst_summary.append("Main harmony/guitar: " + arr_win["clap_timbre"]["arrangement"][0]["description"])
    if drums_win["clap_timbre"].get("drums"):
        inst_summary.append("Drums: " + drums_win["clap_timbre"]["drums"][0]["description"])
    if bass_win["clap_timbre"].get("bass"):
        inst_summary.append("Bass: " + bass_win["clap_timbre"]["bass"][0]["description"])

    chords_collected = []
    for wr in window_reports:
        for c in wr["chords_and_melody_midi"].get("estimated_chords", []):
            if not chords_collected or chords_collected[-1] != c["chord"]:
                chords_collected.append(c["chord"])

    return {
        "models_used": [k for k, v in avail.items() if v],
        "bpm_used": round(float(bpm_hint), 1),
        "instrumentation_summary": inst_summary,
        "detected_chord_flow": chords_collected[:16],
        "vocal_fx_summary": best_vocal_win.get("vocal_fx_profile", {}),
        "best_drum_pattern": drums_win.get("drum_pattern", {}),
        "windows": window_reports,
    }


def analyze_project_vocals_and_harmony(vocal_assets: list[Any], ref_assets: Optional[list[Any]] = None) -> dict[str, Any]:
    """Compute unified musical key across all uploaded vocals + Vocal FX Blueprint against reference."""
    from .audio_analyzer import load_audio_pcm

    all_key_votes: dict[str, float] = {}
    vocal_reports: list[dict[str, Any]] = []
    midi_lows: list[int] = []
    midi_highs: list[int] = []

    for v in vocal_assets:
        an = getattr(v, "analysis", None) or {}
        key_info = an.get("key", {})
        for cand in key_info.get("top_candidates", []):
            kname = cand.get("key")
            sc = float(cand.get("score", 0.0))
            if kname and sc > 0:
                all_key_votes[kname] = all_key_votes.get(kname, 0.0) + sc

        v_path = getattr(v, "stored_path", "")
        fx = {}
        if v_path and Path(v_path).exists():
            try:
                stereo, sr = load_audio_pcm(v_path, target_sr=44100, max_seconds=60.0)
                fx = analyze_vocal_fx_profile(stereo.T, sr=sr, inst_rms_db=None, use_clap=False)
                pr = fx.get("pitch_range_midi", [])
                if len(pr) == 2:
                    midi_lows.append(int(pr[0]))
                    midi_highs.append(int(pr[1]))
            except Exception as exc:
                fx = {"error": str(exc)}

        vocal_reports.append({
            "id": getattr(v, "id", ""),
            "filename": getattr(v, "filename", ""),
            "role": getattr(v, "role", "lead_vocal"),
            "detected_solo_key": key_info.get("key", ""),
            "vocal_fx_raw": fx,
        })

    ranked_keys = sorted(all_key_votes.items(), key=lambda x: x[1], reverse=True)
    unified_key = ranked_keys[0][0] if ranked_keys else ""

    # Compare against reference vocal stem FX if available
    ref_vocal_target = None
    if ref_assets:
        for r in ref_assets:
            ran = getattr(r, "analysis", None) or {}
            nb = ran.get("neural_blueprint", {})
            vf = nb.get("vocal_fx_summary")
            if vf and vf.get("has_vocal"):
                ref_vocal_target = {
                    "reference_filename": getattr(r, "filename", ""),
                    **vf,
                }
                break

    return {
        "unified_recommended_key": unified_key,
        "key_candidates_aggregated": [{"key": k, "combined_score": round(s, 3)} for k, s in ranked_keys[:4]],
        "combined_vocal_midi_range": (
            [min(midi_lows), max(midi_highs), f"{_midi_to_name(min(midi_lows))}..{_midi_to_name(max(midi_highs))}"]
            if midi_lows and midi_highs
            else []
        ),
        "per_vocal_analysis": vocal_reports,
        "reference_vocal_fx_target": ref_vocal_target,
    }


def compare_mix_neural(mix_wav_path: str | Path, ref_path: str | Path) -> dict[str, Any]:
    """Compare a bounced mix against a reference at BOTH full-mix and neural stem/vocal-FX level."""
    from .audio_analyzer import compare_mix_to_reference

    base = compare_mix_to_reference(mix_wav_path, ref_path)
    avail = models_available()
    if not avail["htdemucs"]:
        base["neural_comparison"] = {"available": False, "note": "HTDemucs ONNX unavailable"}
        return base

    # Compare climax/loudest active segment of mix & reference
    mix_bp = deep_analyze_reference(mix_wav_path)
    ref_bp = deep_analyze_reference(ref_path)
    m_win = mix_bp["windows"][-1]
    r_win = ref_bp["windows"][-1]

    m_lvl = m_win["stem_levels_rms_db"]
    r_lvl = r_win["stem_levels_rms_db"]
    stem_deltas = {k: round(m_lvl.get(k, -90.0) - r_lvl.get(k, -90.0), 2) for k in _STEM_NAMES}

    m_vox = m_win.get("vocal_fx_profile", {})
    r_vox = r_win.get("vocal_fx_profile", {})
    neural_recs = list(base.get("recommendations", []))

    if m_vox.get("has_vocal") and r_vox.get("has_vocal"):
        mv_rel = m_vox.get("vocal_to_instrumental_db")
        rv_rel = r_vox.get("vocal_to_instrumental_db")
        if mv_rel is not None and rv_rel is not None:
            vdelta = round(mv_rel - rv_rel, 2)
            if vdelta < -2.0:
                neural_recs.append(f"Vocal sits {abs(vdelta)} dB quieter vs instrumental than reference ({mv_rel} vs {rv_rel} dB): raise Lead Vocal fader or duck mid-range on instruments.")
            elif vdelta > 2.5:
                neural_recs.append(f"Vocal sits {vdelta} dB hotter vs instrumental than reference ({mv_rel} vs {rv_rel} dB): lower Lead Vocal slightly or raise backing.")
        m_tail = float(m_vox.get("reverb_tail_ms", 0))
        r_tail = float(r_vox.get("reverb_tail_ms", 0))
        if abs(m_tail - r_tail) > 280:
            direction = "longer/wetter" if r_tail > m_tail else "shorter/drier"
            neural_recs.append(f"Vocal reverb tail mismatch ({m_tail}ms vs reference {r_tail}ms): adjust Vocal Reverb send to be {direction}.")
        if m_vox.get("eq_character") != r_vox.get("eq_character"):
            neural_recs.append(f"Vocal tone differs ('{m_vox.get('eq_character')}' vs reference '{r_vox.get('eq_character')}'): tweak TDR Nova / Airwindows saturation.")
    elif not m_vox.get("has_vocal") and r_vox.get("has_vocal"):
        neural_recs.append("CRITICAL: Almost no vocal detected in mix export! Check Lead Vocal mute/routing/region placement.")

    if stem_deltas.get("drums", 0) < -4.0:
        neural_recs.append(f"Drums stem is {abs(stem_deltas['drums'])} dB weaker than reference: raise Drums track/bus or transient punch.")
    if stem_deltas.get("bass", 0) < -4.5:
        neural_recs.append(f"Bass stem is {abs(stem_deltas['bass'])} dB weaker than reference: raise Bass track or add harmonics/TSE BOD.")

    base["recommendations"] = neural_recs
    base["neural_comparison"] = {
        "available": True,
        "stem_rms_deltas_db": stem_deltas,
        "mix_vocal_fx": m_vox,
        "reference_vocal_fx": r_vox,
        "mix_instrumentation": mix_bp.get("instrumentation_summary", []),
        "reference_instrumentation": ref_bp.get("instrumentation_summary", []),
    }
    return base
