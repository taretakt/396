"""Core feature extraction — the standard card.

Everything in :mod:`m396` runs on librosa 0.11 (2026-10 era API drift handled:
``beat_track`` returns a tuple, ``beat_times`` was removed — use ``units='time'``).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import librosa

warnings.filterwarnings("ignore")

SR = 22050
HOP = 512


def load(path: str) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


def features(path: str) -> dict[str, Any]:
    """The static card: tempo (arithmetic cross-check), timbre, rhythm, dynamics, key."""
    y, sr = load(path)
    dur = len(y) / sr

    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    _, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo_cc = len(beat_times) / dur * 60.0  # beats counted, authoritative

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_frac = float(S[freqs <= 128].sum() / S.sum())

    H, P = librosa.decompose.hpss(S)
    perc = float((P**2).sum() / ((H**2).sum() + 1e-8))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    if len(beat_times) > 2:
        beat_idx = np.clip(librosa.util.match_events(times, beat_times), 0, len(onset_env) - 1)
        on = np.zeros_like(onset_env, dtype=bool)
        on[beat_idx] = True
        sync = float(onset_env[~on].mean() / (onset_env[on].mean() + 1e-9))
    else:
        sync = float("nan")

    rms = librosa.feature.rms(y=y)[0]
    db = np.array([20 * np.log10(s.mean() + 1e-8) for s in np.array_split(rms, 8)])
    db = db - db.max()

    return {
        "duration_s": round(dur, 2),
        "tempo": round(tempo_cc, 1),          # beat-counted — the honest number
        "tempo_beat_track_bpm": round(tempo, 1),  # tracker read, for comparison only
        "spectral_centroid_hz": round(centroid, 1),
        "bass_fraction": round(bass_frac, 3),
        "percussive_ratio": round(perc, 3),
        "syncopation": round(sync, 3),
        "dyn_mean_db": round(float(db.mean()), 1),
        "dyn_std_db": round(float(db.std()), 1),
        "fade_delta_db": round(float(db[-1] - db[0]), 1),
        "key": _key(y, sr),
    }


def _key(y: np.ndarray, sr: int) -> str:
    """Krumhansl-Schmuckler key guess. Tiebreaker only — never a filter."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    ks = np.array([6.35, 2.23, 3.48, 2.33, 3.38, 2.33, 3.38, 2.23, 3.48, 2.33, 3.38, 2.23])
    sm = np.array([6.33, 2.68, 3.52, 2.68, 3.52, 2.23, 3.48, 2.33, 3.48, 2.33, 3.48, 2.33])
    ch = chroma.mean(axis=1)
    ck = np.correlate(ch, ks, "full")[11:]
    cs = np.correlate(ch, sm, "full")[11:]
    names = "C C# D D# E F F# G G# A A# B".split()
    best, key = 0.0, "?"
    for i, (a, b) in enumerate(zip(ck, cs)):
        if max(a, b) > best:
            best, key = max(a, b), f"{names[i]}{'m' if b > a else ''}"
    return key