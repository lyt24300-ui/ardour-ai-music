"""Audio Intelligence Layer — local DSP analysis & Whisper lyrics alignment.

Gives ANY text-only LLM precise 'ears' for references, vocals, samples,
and bounced mix previews:
- ffmpeg decoding for WAV/MP3/FLAC/OGG/M4A/AIFF
- ITU-R BS.1770 K-weighted Integrated LUFS, Short-Term Max, LRA, True Peak
- 7-band spectral balance & centroid
- Stereo correlation & Mid/Side width
- BPM / Tempo estimation (spectral flux autocorrelation)
- Musical Key & Scale detection (Krumhansl-Schmuckler chromagram)
- Vocal VAD (phrases, silences, breaths), noise floor, sibilance, muddiness, F0 pitch
- Local faster-whisper word-level forced alignment to Lyrics tab
- Mix-vs-Reference comparator for iterative autonomous mixing/mastering
"""

from __future__ import annotations

import difflib
import math
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import FFMPEG_BIN

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_CAMELOT_MAJOR = {
    "C": "8B", "C#": "3B", "D": "10B", "D#": "5B", "E": "12B", "F": "7B",
    "F#": "2B", "G": "9B", "G#": "4B", "A": "11B", "A#": "6B", "B": "1B",
}
_CAMELOT_MINOR = {
    "A": "8A", "A#": "3A", "B": "10A", "C": "5A", "C#": "12A", "D": "7A",
    "D#": "2A", "E": "9A", "F": "4A", "F#": "11A", "G": "6A", "G#": "1A",
}

SPECTRAL_BANDS = [
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("upper_mid", 2000.0, 4000.0),
    ("presence", 4000.0, 8000.0),
    ("air", 8000.0, 20000.0),
]

# Cached faster-whisper model instance by size
_WHISPER_CACHE: dict[str, Any] = {}


def _db(val: float, floor: float = -120.0) -> float:
    if val <= 1e-12:
        return floor
    return float(max(floor, 20.0 * math.log10(val)))


def _hz_to_note(hz: float) -> str:
    if hz <= 0:
        return "N/A"
    midi = int(round(69 + 12 * math.log2(hz / 440.0)))
    name = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return f"{name}{octave}"


def load_audio_pcm(path: str | Path, target_sr: int = 44100, max_seconds: float = 480.0) -> tuple[np.ndarray, int]:
    """Decode any audio file into shape (N, 2) float32 array in [-1, 1]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    cmd = [
        FFMPEG_BIN,
        "-v", "error",
        "-i", str(p),
        "-t", str(max_seconds),
        "-ac", "2",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=90,
        )
        raw = proc.stdout
        if len(raw) >= 8:
            samples = np.frombuffer(raw, dtype=np.float32)
            n_frames = len(samples) // 2
            stereo = samples[: n_frames * 2].reshape(n_frames, 2).copy()
            return stereo, target_sr
    except Exception:
        pass

    # Fallback: built-in wave module for uncompressed PCM WAV
    if p.suffix.lower() == ".wav":
        with wave.open(str(p), "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = min(wf.getnframes(), int(sr * max_seconds))
            raw = wf.readframes(n_frames)
            if sw == 2:
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 4:
                arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sw == 1:
                arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise RuntimeError(f"Unsupported WAV sample width: {sw}")
            arr = arr.reshape(-1, n_ch)
            if n_ch == 1:
                stereo = np.repeat(arr, 2, axis=1)
            else:
                stereo = arr[:, :2]
            return stereo, sr

    raise RuntimeError(f"Failed to decode audio file '{p.name}' via ffmpeg ({FFMPEG_BIN}).")


def _k_weight_gain_curve(freqs: np.ndarray) -> np.ndarray:
    """ITU-R BS.1770-4 K-weighting magnitude curve (high-shelf + RLB HPF)."""
    f = np.maximum(freqs, 1.0)
    # Stage 1: +4 dB high shelf around 1681.97 Hz
    f_shelf = 1681.974450955533
    ratio = f / f_shelf
    shelf_mag = np.sqrt((1.0 + 2.508 * ratio**2 + ratio**4) / (1.0 + 1.0 * ratio**2 + ratio**4))
    # Stage 2: 2nd-order RLB high-pass around 38.135 Hz
    f_hp = 38.13547087602444
    hp_ratio = f_hp / f
    hp_mag = 1.0 / np.sqrt(1.0 + hp_ratio**4)
    return shelf_mag * hp_mag


def compute_loudness_and_dynamics(stereo: np.ndarray, sr: int) -> dict[str, Any]:
    """Compute Integrated LUFS, Short-Term Max LUFS, LRA, True Peak, RMS, Crest."""
    if len(stereo) == 0:
        return {}

    mono = 0.5 * (stereo[:, 0] + stereo[:, 1])
    peak = float(np.max(np.abs(stereo)))
    rms = float(np.sqrt(np.mean(stereo**2)))

    # 4x oversampled True Peak estimate via FFT zero-padding on highest-peak window
    tp = peak
    peak_idx = int(np.argmax(np.max(np.abs(stereo), axis=1)))
    win = 512
    s0 = max(0, peak_idx - win // 2)
    s1 = min(len(stereo), s0 + win)
    if s1 - s0 >= 32:
        for ch in (0, 1):
            seg = stereo[s0:s1, ch]
            spec = np.fft.rfft(seg)
            padded = np.zeros(len(spec) * 4 - 3, dtype=np.complex64)
            padded[: len(spec)] = spec
            up = np.fft.irfft(padded) * 4.0
            tp = max(tp, float(np.max(np.abs(up))))

    # K-weighted loudness via windowed FFT blocks (400 ms, 100 ms hop)
    block_n = max(1024, int(0.4 * sr))
    hop_n = max(256, int(0.1 * sr))
    fft_n = 1 << int(math.ceil(math.log2(block_n)))
    freqs = np.fft.rfftfreq(fft_n, d=1.0 / sr)
    k_curve = _k_weight_gain_curve(freqs) ** 2

    block_powers: list[float] = []
    if len(stereo) < block_n:
        pad = np.zeros((block_n, 2), dtype=np.float32)
        pad[: len(stereo)] = stereo
        stereo_eval = pad
    else:
        stereo_eval = stereo

    # Limit evaluation blocks for very long files while staying representative
    starts = list(range(0, len(stereo_eval) - block_n + 1, hop_n))
    if len(starts) > 1500:
        step = max(1, len(starts) // 1500)
        starts = starts[::step]

    for start in starts:
        sl = stereo_eval[start : start + block_n]
        pwr = 0.0
        for ch in (0, 1):
            sp = np.fft.rfft(sl[:, ch], n=fft_n)
            # Parseval normalized mean square energy
            mag2 = (sp.real**2 + sp.imag**2) * k_curve
            # rfft double-count interior bins
            ch_e = (mag2[0] + mag2[-1] + 2.0 * np.sum(mag2[1:-1])) / (fft_n * block_n)
            pwr += float(ch_e)
        block_powers.append(pwr)

    arr_p = np.array(block_powers, dtype=np.float64)
    block_lufs = np.where(arr_p > 1e-12, -0.691 + 10.0 * np.log10(np.maximum(arr_p, 1e-12)), -120.0)

    # BS.1770 gating: absolute -70 LUFS, then relative -10 dB below ungated mean
    above_abs = arr_p[block_lufs > -70.0]
    if len(above_abs) > 0:
        abs_mean = float(np.mean(above_abs))
        rel_thresh = -0.691 + 10.0 * math.log10(max(abs_mean, 1e-12)) - 10.0
        above_rel = arr_p[block_lufs > rel_thresh]
        gated_mean = float(np.mean(above_rel)) if len(above_rel) > 0 else abs_mean
        integrated_lufs = round(-0.691 + 10.0 * math.log10(max(gated_mean, 1e-12)), 2)
    else:
        integrated_lufs = -70.0

    # Short-term (~3s moving average of 100ms hops -> 27 overlapping blocks)
    if len(arr_p) >= 27:
        kernel = np.ones(27) / 27.0
        st = np.convolve(arr_p, kernel, mode="valid")
        st_lufs = -0.691 + 10.0 * np.log10(np.maximum(st, 1e-12))
        st_max = round(float(np.max(st_lufs)), 2)
        valid_st = st_lufs[st_lufs > -70.0]
        lra = round(float(np.percentile(valid_st, 95) - np.percentile(valid_st, 10)), 2) if len(valid_st) > 5 else 0.0
    else:
        st_max = integrated_lufs
        lra = 0.0

    peak_db = round(_db(peak), 2)
    true_peak_db = round(_db(tp), 2)
    rms_db = round(_db(rms), 2)
    crest_db = round(peak_db - rms_db, 2)

    return {
        "integrated_lufs": integrated_lufs,
        "short_term_max_lufs": st_max,
        "loudness_range_lu": lra,
        "peak_db": peak_db,
        "true_peak_db": true_peak_db,
        "rms_db": rms_db,
        "crest_factor_db": crest_db,
        "clipping_risk": true_peak_db > -0.3,
    }


def compute_spectral_profile(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Compute 7-band spectral balance in dB and spectral centroid."""
    n_fft = 4096
    hop = 2048
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))

    starts = list(range(0, len(mono) - n_fft + 1, hop))
    if len(starts) > 400:
        step = max(1, len(starts) // 400)
        starts = starts[::step]

    win = np.hanning(n_fft).astype(np.float32)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for s in starts:
        sp = np.fft.rfft(mono[s : s + n_fft] * win)
        acc += sp.real**2 + sp.imag**2
    avg_pow = acc / max(1, len(starts))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    total_e = float(np.sum(avg_pow)) + 1e-12
    bands_db: dict[str, float] = {}
    for name, f_lo, f_hi in SPECTRAL_BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_e = float(np.sum(avg_pow[mask]))
        # Relative dB vs total spectrum energy (normalized reference)
        bands_db[name] = round(10.0 * math.log10(max(band_e / total_e, 1e-9)), 2)

    mag = np.sqrt(avg_pow)
    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))

    # Specific vocal health metrics
    sib_mask = (freqs >= 5000.0) & (freqs <= 9000.0)
    mud_mask = (freqs >= 200.0) & (freqs <= 400.0)
    sib_db = round(10.0 * math.log10(max(float(np.sum(avg_pow[sib_mask])) / total_e, 1e-9)), 2)
    mud_db = round(10.0 * math.log10(max(float(np.sum(avg_pow[mud_mask])) / total_e, 1e-9)), 2)

    brightness = "dark"
    if centroid > 2800:
        brightness = "bright"
    elif centroid > 1600:
        brightness = "balanced"

    return {
        "bands_relative_db": bands_db,
        "spectral_centroid_hz": round(centroid, 1),
        "tonal_character": brightness,
        "sibilance_5k_9k_db": sib_db,
        "muddiness_200_400_db": mud_db,
    }


def compute_stereo_metrics(stereo: np.ndarray) -> dict[str, Any]:
    """Compute stereo correlation and Mid/Side balance."""
    left = stereo[:, 0].astype(np.float64)
    right = stereo[:, 1].astype(np.float64)
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    mid_rms = float(np.sqrt(np.mean(mid**2))) + 1e-12
    side_rms = float(np.sqrt(np.mean(side**2)))
    side_ratio = side_rms / mid_rms

    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2))) + 1e-12
    corr = float(np.sum(left * right) / denom)
    corr = max(-1.0, min(1.0, corr))

    if side_ratio < 0.05:
        width_label = "mono"
    elif side_ratio < 0.25:
        width_label = "narrow"
    elif side_ratio < 0.55:
        width_label = "balanced_wide"
    else:
        width_label = "very_wide"

    return {
        "correlation": round(corr, 3),
        "side_to_mid_ratio": round(side_ratio, 3),
        "width_label": width_label,
        "mono_compatible": corr >= 0.2,
    }


def estimate_bpm(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Estimate BPM via spectral-flux onset envelope + autocorrelation."""
    # Downsample to ~11025 Hz for fast onset detection
    factor = max(1, sr // 11025)
    sig = mono[::factor]
    ds_sr = sr / factor
    if len(sig) < 4096:
        return {"bpm": 120.0, "confidence": 0.0}

    n_fft = 1024
    hop = 256
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = (len(sig) - n_fft) // hop
    if n_frames < 32:
        return {"bpm": 120.0, "confidence": 0.0}

    # Cap to first 90s for speed
    max_frames = int(90.0 * ds_sr / hop)
    n_frames = min(n_frames, max_frames)

    flux = np.zeros(n_frames, dtype=np.float32)
    prev_mag = np.zeros(n_fft // 2 + 1, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        mag = np.abs(np.fft.rfft(sig[s : s + n_fft] * win))
        diff = mag - prev_mag
        flux[i] = float(np.sum(np.maximum(diff, 0.0)))
        prev_mag = mag

    flux -= np.mean(flux)
    if np.std(flux) < 1e-8:
        return {"bpm": 120.0, "confidence": 0.0}

    # Autocorrelation via FFT
    ac_n = 1 << int(math.ceil(math.log2(len(flux) * 2)))
    f_sp = np.fft.rfft(flux, n=ac_n)
    ac = np.fft.irfft(f_sp * np.conj(f_sp))[: len(flux)]
    ac /= max(float(ac[0]), 1e-9)

    fps = ds_sr / hop
    min_lag = max(1, int(round(fps * 60.0 / 185.0)))
    max_lag = min(len(ac) - 1, int(round(fps * 60.0 / 65.0)))
    if max_lag <= min_lag:
        return {"bpm": 120.0, "confidence": 0.0}

    # Slight perceptual prior around 90–145 BPM
    lags = np.arange(min_lag, max_lag + 1)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * ((np.log2(bpms / 120.0)) / 0.65) ** 2)
    scores = ac[lags] * prior

    best_idx = int(np.argmax(scores))
    best_lag = int(lags[best_idx])
    bpm = 60.0 * fps / best_lag
    conf = float(max(0.0, min(1.0, ac[best_lag])))

    # Normalize into musical range 70..175
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 175.0:
        bpm /= 2.0

    return {
        "bpm": round(bpm, 1),
        "confidence": round(conf, 3),
        "half_time_bpm": round(bpm / 2.0, 1),
        "double_time_bpm": round(bpm * 2.0, 1),
    }


def estimate_key(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Estimate musical key & scale using chromagram + Krumhansl-Schmuckler."""
    n_fft = 8192
    hop = 4096
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))

    starts = list(range(0, len(mono) - n_fft + 1, hop))
    if len(starts) > 250:
        step = max(1, len(starts) // 250)
        starts = starts[::step]

    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    valid = (freqs >= 65.0) & (freqs <= 2093.0)
    v_freqs = freqs[valid]
    midi_bins = np.round(69.0 + 12.0 * np.log2(v_freqs / 440.0)).astype(int) % 12

    chroma = np.zeros(12, dtype=np.float64)
    for s in starts:
        mag = np.abs(np.fft.rfft(mono[s : s + n_fft] * win))[valid]
        for pc in range(12):
            chroma[pc] += float(np.sum(mag[midi_bins == pc] ** 2))

    if np.sum(chroma) <= 1e-12:
        return {"key": "Unknown", "camelot": "", "confidence": 0.0}

    chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-12)
    maj_norm = (_KS_MAJOR - np.mean(_KS_MAJOR)) / np.std(_KS_MAJOR)
    min_norm = (_KS_MINOR - np.mean(_KS_MINOR)) / np.std(_KS_MINOR)

    candidates: list[tuple[float, str, str]] = []
    for root in range(12):
        rot = np.roll(chroma, -root)
        maj_c = float(np.dot(rot, maj_norm) / 12.0)
        min_c = float(np.dot(rot, min_norm) / 12.0)
        note = NOTE_NAMES[root]
        candidates.append((maj_c, f"{note} major", _CAMELOT_MAJOR[note]))
        candidates.append((min_c, f"{note} minor", _CAMELOT_MINOR[note]))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_key, best_camelot = candidates[0]
    second_score = candidates[1][0]
    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))

    return {
        "key": best_key,
        "camelot": best_camelot,
        "confidence": round(confidence, 3),
        "margin": round(best_score - second_score, 3),
        "top_candidates": [
            {"key": k, "camelot": cam, "score": round(sc, 3)}
            for sc, k, cam in candidates[:3]
        ],
    }


def compute_waveform_peaks(mono: np.ndarray, num_bars: int = 220) -> list[float]:
    """Compute compact peak envelope [0..1] for Ardour-style canvas waveform."""
    if len(mono) == 0:
        return [0.0] * num_bars
    chunk = max(1, len(mono) // num_bars)
    peaks: list[float] = []
    for i in range(num_bars):
        s = i * chunk
        e = min(len(mono), s + chunk)
        if s >= len(mono):
            peaks.append(0.0)
        else:
            peaks.append(round(float(np.max(np.abs(mono[s:e]))), 4))
    max_p = max(peaks) if peaks else 1.0
    if max_p > 1e-6:
        peaks = [round(p / max_p, 4) for p in peaks]
    return peaks


def compute_energy_sections(mono: np.ndarray, sr: int, num_segments: int = 12) -> list[dict[str, Any]]:
    """Map song energy across equal timeline sections for structural understanding."""
    dur = len(mono) / float(sr)
    if dur <= 0.1:
        return []
    n_seg = min(num_segments, max(1, int(dur // 4)))
    seg_len = len(mono) // n_seg
    overall_rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    out: list[dict[str, Any]] = []
    for i in range(n_seg):
        s = i * seg_len
        e = len(mono) if i == n_seg - 1 else (i + 1) * seg_len
        seg = mono[s:e]
        r = float(np.sqrt(np.mean(seg**2)))
        rel = r / overall_rms
        level = "high" if rel > 1.15 else ("low" if rel < 0.65 else "medium")
        out.append({
            "segment": i + 1,
            "start_sec": round(s / sr, 2),
            "end_sec": round(e / sr, 2),
            "rms_db": round(_db(r), 1),
            "energy": level,
        })
    return out


def analyze_vocal_vad_and_pitch(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Detect vocal phrases, silences, breaths, noise floor, and F0 pitch range."""
    frame_ms = 20
    frame_n = max(64, int(sr * frame_ms / 1000.0))
    n_frames = len(mono) // frame_n
    if n_frames == 0:
        return {"phrases": [], "silences": [], "breaths": [], "noise_floor_db": -90.0}

    trimmed = mono[: n_frames * frame_n].reshape(n_frames, frame_n)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    frame_db = np.array([_db(float(r)) for r in rms])
    zcr = np.mean(np.abs(np.diff(np.signbit(trimmed), axis=1)), axis=1)

    noise_floor = float(np.percentile(frame_db, 10))
    peak_frame_db = float(np.percentile(frame_db, 98))
    # Adaptive speech/singing threshold
    active_thresh = max(noise_floor + 10.0, peak_frame_db - 26.0, -45.0)
    breath_low = max(noise_floor + 3.0, active_thresh - 12.0)

    is_active = frame_db >= active_thresh
    # Morphological hangover: bridge gaps <= 220 ms (11 frames) inside a phrase
    bridge_frames = 11
    smoothed = is_active.copy()
    i = 0
    while i < n_frames:
        if smoothed[i]:
            j = i
            while j < n_frames and smoothed[j]:
                j += 1
            # Check silence gap length
            k = j
            while k < n_frames and not smoothed[k]:
                k += 1
            if k < n_frames and (k - j) <= bridge_frames:
                smoothed[j:k] = True
                i = k
                continue
            i = j
        else:
            i += 1

    phrases: list[dict[str, Any]] = []
    silences: list[dict[str, Any]] = []
    breaths: list[dict[str, Any]] = []

    idx = 0
    while idx < n_frames:
        state = bool(smoothed[idx])
        end = idx
        while end < n_frames and bool(smoothed[end]) == state:
            end += 1
        start_sec = round(idx * frame_ms / 1000.0, 3)
        end_sec = round(end * frame_ms / 1000.0, 3)
        dur = round(end_sec - start_sec, 3)

        if state and dur >= 0.14:
            seg_audio = mono[idx * frame_n : end * frame_n]
            phrases.append({
                "index": len(phrases) + 1,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": dur,
                "peak_db": round(_db(float(np.max(np.abs(seg_audio)))), 1),
                "rms_db": round(_db(float(np.sqrt(np.mean(seg_audio**2)))), 1),
            })
        elif not state and dur >= 0.18:
            silences.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": dur,
            })
            # Check for breath noise in silence (high ZCR + moderate energy just above noise floor)
            seg_db = frame_db[idx:end]
            seg_zcr = zcr[idx:end]
            breath_mask = (seg_db >= breath_low) & (seg_db < active_thresh) & (seg_zcr > 0.12)
            if int(np.sum(breath_mask)) >= 4:
                b_indices = np.where(breath_mask)[0]
                b_start = round((idx + int(b_indices[0])) * frame_ms / 1000.0, 3)
                b_end = round((idx + int(b_indices[-1]) + 1) * frame_ms / 1000.0, 3)
                breaths.append({
                    "start_sec": b_start,
                    "end_sec": b_end,
                    "peak_db": round(float(np.max(seg_db[b_indices])), 1),
                })
        idx = end

    # Estimate vocal fundamental F0 via autocorrelation on active frames
    f0_values: list[float] = []
    min_lag = max(1, int(sr / 650.0))
    max_lag = min(frame_n * 2 - 1, int(sr / 80.0))
    win_f0 = max(max_lag + 64, int(sr * 0.04))
    active_idxs = np.where(is_active)[0]
    if len(active_idxs) > 120:
        active_idxs = active_idxs[:: max(1, len(active_idxs) // 120)]

    for fi in active_idxs:
        s = int(fi * frame_n)
        if s + win_f0 > len(mono):
            continue
        seg = mono[s : s + win_f0]
        seg = seg - np.mean(seg)
        energy = float(np.dot(seg, seg))
        if energy < 1e-6:
            continue
        ac = np.correlate(seg, seg, mode="full")[win_f0 - 1 :]
        if max_lag >= len(ac):
            continue
        sub = ac[min_lag:max_lag]
        if len(sub) == 0:
            continue
        pk = int(np.argmax(sub))
        if sub[pk] / energy > 0.45:
            f0 = sr / float(min_lag + pk)
            f0_values.append(f0)

    pitch_info: dict[str, Any] = {}
    if len(f0_values) >= 5:
        arr_f0 = np.array(f0_values)
        p10, p50, p90 = np.percentile(arr_f0, [10, 50, 90])
        pitch_info = {
            "median_hz": round(float(p50), 1),
            "median_note": _hz_to_note(float(p50)),
            "range_low_note": _hz_to_note(float(p10)),
            "range_high_note": _hz_to_note(float(p90)),
        }

    return {
        "noise_floor_db": round(noise_floor, 1),
        "phrase_count": len(phrases),
        "phrases": phrases[:80],
        "silences": silences[:80],
        "breaths": breaths[:40],
        "pitch": pitch_info,
    }


def analyze_audio_file(path: str | Path, category: str = "references") -> dict[str, Any]:
    """Run full category-appropriate DSP analysis on any audio file."""
    stereo, sr = load_audio_pcm(path)
    duration = round(len(stereo) / float(sr), 3)
    mono = 0.5 * (stereo[:, 0] + stereo[:, 1])

    loudness = compute_loudness_and_dynamics(stereo, sr)
    spectral = compute_spectral_profile(mono, sr)
    stereo_info = compute_stereo_metrics(stereo)
    peaks = compute_waveform_peaks(mono, num_bars=220)

    result: dict[str, Any] = {
        "duration_sec": duration,
        "sample_rate": sr,
        "channels": 2,
        "loudness": loudness,
        "spectral": spectral,
        "stereo": stereo_info,
        "waveform_peaks": peaks,
    }

    if category == "references":
        result["tempo"] = estimate_bpm(mono, sr)
        result["key"] = estimate_key(mono, sr)
        result["sections"] = compute_energy_sections(mono, sr)
    elif category == "vocals":
        result["vocal"] = analyze_vocal_vad_and_pitch(mono, sr)
        result["key"] = estimate_key(mono, sr)
    elif category == "samples":
        sample_type = "one_shot" if duration < 2.2 else ("loop" if duration <= 16.0 else "stem_or_fx")
        result["sample_type"] = sample_type
        result["key"] = estimate_key(mono, sr)
        if duration >= 2.0:
            result["tempo"] = estimate_bpm(mono, sr)
    else:
        result["tempo"] = estimate_bpm(mono, sr)
        result["key"] = estimate_key(mono, sr)

    return result


def _normalize_word(w: str) -> str:
    return re.sub(r"[^\wа-яА-ЯёЁ]+", "", w.lower())


def align_vocal_with_lyrics(
    vocal_path: str | Path,
    lyrics_sections: list[dict[str, Any]],
    existing_vad_phrases: Optional[list[dict[str, Any]]] = None,
    whisper_enabled: bool = True,
    whisper_model_size: str = "base",
    language: str = "",
) -> dict[str, Any]:
    """Align lyrics lines to exact timestamps in a vocal recording.

    Uses local faster-whisper with word_timestamps=True when available,
    falling back gracefully to VAD phrase-to-line mapping if offline.
    """
    flat_lines: list[dict[str, Any]] = []
    for sec in lyrics_sections:
        sec_name = sec.get("section", "Main")
        for line in sec.get("lines", []):
            if line.strip():
                flat_lines.append({
                    "section": sec_name,
                    "line_index": len(flat_lines) + 1,
                    "text": line.strip(),
                })

    transcript_words: list[dict[str, Any]] = []
    full_transcript = ""
    detected_language = language or "auto"
    method = "vad_fallback"
    whisper_error = ""

    if whisper_enabled:
        try:
            from faster_whisper import WhisperModel

            model = _WHISPER_CACHE.get(whisper_model_size)
            if model is None:
                model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
                _WHISPER_CACHE[whisper_model_size] = model

            kwargs: dict[str, Any] = {
                "word_timestamps": True,
                "vad_filter": True,
                "beam_size": 3,
            }
            if language:
                kwargs["language"] = language

            segments, info = model.transcribe(str(vocal_path), **kwargs)
            detected_language = getattr(info, "language", detected_language) or "auto"
            seg_texts: list[str] = []
            for seg in segments:
                seg_texts.append(seg.text.strip())
                if seg.words:
                    for w in seg.words:
                        transcript_words.append({
                            "word": w.word.strip(),
                            "norm": _normalize_word(w.word),
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                            "prob": round(float(getattr(w, "probability", 0.8)), 3),
                        })
            full_transcript = " ".join(seg_texts).strip()
            if transcript_words:
                method = "faster_whisper"
        except Exception as exc:
            whisper_error = str(exc)

    aligned_lines: list[dict[str, Any]] = []

    if method == "faster_whisper" and flat_lines and transcript_words:
        # Sequential fuzzy alignment of each lyrics line to transcript words
        cursor = 0
        n_words = len(transcript_words)
        for line_obj in flat_lines:
            line_tokens = [_normalize_word(t) for t in line_obj["text"].split() if _normalize_word(t)]
            if not line_tokens or cursor >= n_words:
                aligned_lines.append({
                    **line_obj,
                    "start_sec": None,
                    "end_sec": None,
                    "confidence": 0.0,
                    "matched_text": "",
                })
                continue

            win_len = max(1, len(line_tokens))
            best_score = -1.0
            best_span = (cursor, min(n_words, cursor + win_len))
            search_limit = min(n_words, cursor + max(20, win_len * 4))

            for start_i in range(cursor, search_limit):
                for span_len in range(max(1, win_len - 2), min(win_len + 4, n_words - start_i + 1)):
                    cand = " ".join(w["norm"] for w in transcript_words[start_i : start_i + span_len])
                    target = " ".join(line_tokens)
                    ratio = difflib.SequenceMatcher(None, target, cand).ratio()
                    # Mild penalty for jumping far ahead
                    penalty = 0.004 * (start_i - cursor)
                    score = ratio - penalty
                    if score > best_score:
                        best_score = score
                        best_span = (start_i, start_i + span_len)

            s_i, e_i = best_span
            matched = transcript_words[s_i:e_i]
            if matched and best_score >= 0.25:
                aligned_lines.append({
                    **line_obj,
                    "start_sec": matched[0]["start"],
                    "end_sec": matched[-1]["end"],
                    "confidence": round(max(0.0, min(1.0, best_score)), 3),
                    "matched_text": " ".join(w["word"] for w in matched),
                })
                cursor = e_i
            else:
                aligned_lines.append({
                    **line_obj,
                    "start_sec": None,
                    "end_sec": None,
                    "confidence": round(max(0.0, best_score), 3),
                    "matched_text": "",
                })
    else:
        # VAD phrase-based mapping fallback
        phrases = existing_vad_phrases
        if phrases is None:
            stereo, sr = load_audio_pcm(vocal_path)
            mono = 0.5 * (stereo[:, 0] + stereo[:, 1])
            phrases = analyze_vocal_vad_and_pitch(mono, sr).get("phrases", [])

        if flat_lines and phrases:
            for i, line_obj in enumerate(flat_lines):
                p_idx = min(len(phrases) - 1, int(i * len(phrases) / max(1, len(flat_lines))))
                ph = phrases[p_idx]
                aligned_lines.append({
                    **line_obj,
                    "start_sec": ph["start_sec"],
                    "end_sec": ph["end_sec"],
                    "confidence": 0.45,
                    "matched_text": "(VAD phrase estimate)",
                })

    return {
        "method": method,
        "detected_language": detected_language,
        "full_transcript": full_transcript,
        "word_count": len(transcript_words),
        "words": transcript_words[:300],
        "aligned_lines": aligned_lines,
        "whisper_note": whisper_error,
    }


def compare_mix_to_reference(mix_path: str | Path, ref_path: str | Path) -> dict[str, Any]:
    """Compare a bounced mix WAV against a reference track and emit mixing advice."""
    mix = analyze_audio_file(mix_path, category="references")
    ref = analyze_audio_file(ref_path, category="references")

    mix_l = mix["loudness"]
    ref_l = ref["loudness"]
    lufs_delta = round(mix_l["integrated_lufs"] - ref_l["integrated_lufs"], 2)

    band_deltas: dict[str, float] = {}
    recommendations: list[str] = []

    for band, _, _ in SPECTRAL_BANDS:
        m_val = mix["spectral"]["bands_relative_db"].get(band, 0.0)
        r_val = ref["spectral"]["bands_relative_db"].get(band, 0.0)
        delta = round(m_val - r_val, 2)
        band_deltas[band] = delta
        if delta > 2.5:
            recommendations.append(f"Band '{band}' is +{delta} dB hotter than reference: consider attenuating EQ or lowering tracks in this range.")
        elif delta < -2.5:
            recommendations.append(f"Band '{band}' is {delta} dB quieter than reference: consider subtle EQ boost or bringing up instruments in this range.")

    if mix_l.get("clipping_risk"):
        recommendations.append(
            f"True Peak is {mix_l['true_peak_db']} dBTP (clipping risk): add/tighten Master True Peak Limiter ceiling to -1.0 dBTP."
        )
    if lufs_delta < -2.0:
        recommendations.append(
            f"Mix is {abs(lufs_delta)} LU quieter than reference ({mix_l['integrated_lufs']} vs {ref_l['integrated_lufs']} LUFS): raise master limiter input gain carefully."
        )
    elif lufs_delta > 2.0:
        recommendations.append(
            f"Mix is {lufs_delta} LU louder/more crushed than reference: ease back master limiter drive for better dynamics."
        )

    stereo_delta = round(mix["stereo"]["side_to_mid_ratio"] - ref["stereo"]["side_to_mid_ratio"], 3)
    if not mix["stereo"]["mono_compatible"]:
        recommendations.append("Low stereo correlation (<0.2): check phase issues on stereo wideners or layered bass.")
    elif stereo_delta < -0.15:
        recommendations.append("Mix is noticeably narrower than reference: pan guitars/synths/hats wider or widen reverb returns.")

    return {
        "mix_summary": {
            "lufs": mix_l["integrated_lufs"],
            "true_peak_db": mix_l["true_peak_db"],
            "crest_db": mix_l["crest_factor_db"],
            "stereo": mix["stereo"],
        },
        "reference_summary": {
            "lufs": ref_l["integrated_lufs"],
            "true_peak_db": ref_l["true_peak_db"],
            "crest_db": ref_l["crest_factor_db"],
            "stereo": ref["stereo"],
        },
        "lufs_delta_db": lufs_delta,
        "band_deltas_db": band_deltas,
        "stereo_side_ratio_delta": stereo_delta,
        "recommendations": recommendations,
    }
