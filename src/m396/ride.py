"""Gait analysis: does the cadence ride the piano, the beat, or thread between?

Faithful port of the validated standalone (2026-10-03, four real tracks):
  - piano-hit novelty on the HPSS HARMONIC magnitude (80-1400 Hz)
  - vocal-band novelty on the full mix (900-3800 Hz)
  - envelopes normalized before peak-picking (wait=, per the reference)
  - lock window +-48 ms; shuffle null (200 draws, same count); z-score
  - beat-grid lock as the control

Reference values reproduced by design: Tonedeff Porcelain lock=0.094 z=5.11,
Immortal Technique lock=0.046 z=-4.71, JMT lock=0.039 z=-0.83.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import librosa

SR = 22050
HOP = 512
N_FFT = 2048
LOCK = 0.048


def ride(y: np.ndarray, sr: int = SR) -> dict[str, Any]:
    dur = len(y) / sr

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    H, _ = librosa.decompose.hpss(S)
    times = librosa.times_like(S, sr=sr, hop_length=HOP)

    piano_ons = librosa.onset.onset_strength(S=H, sr=sr, hop_length=HOP, fmin=80.0, fmax=1400.0)
    piano_ons = librosa.util.normalize(piano_ons)
    piano_idx = librosa.onset.onset_detect(
        onset_envelope=piano_ons, sr=sr, hop_length=HOP,
        pre_max=int(0.04 * SR / HOP), post_max=int(0.05 * SR / HOP),
        pre_avg=int(0.18 * SR / HOP), post_avg=int(0.35 * SR / HOP),
        delta=0.30, wait=int(0.09 * SR / HOP),
    )
    piano_hits = times[piano_idx]

    voc_ons = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, fmin=900.0, fmax=3800.0)
    voc_ons = librosa.util.normalize(voc_ons)
    voc_idx = librosa.onset.onset_detect(
        onset_envelope=voc_ons, sr=sr, hop_length=HOP,
        pre_max=max(1, int(0.05 * SR / HOP)), post_max=max(1, int(0.02 * SR / HOP)),
        pre_avg=int(0.12 * SR / HOP), post_avg=int(0.20 * SR / HOP),
        delta=0.22, wait=int(0.03 * SR / HOP),
    )
    voc_times = times[voc_idx]

    # safety margins
    voc_times = voc_times[(voc_times > 0.2) & (voc_times < dur - 0.2)]
    piano_hits = piano_hits[(piano_hits > 0.2) & (piano_hits < dur - 0.2)]

    # alignment + shuffle null
    lock_obs = _lock_frac(voc_times, piano_hits)
    rng = np.random.default_rng(7)
    nulls = [_lock_frac(rng.uniform(0.2, dur - 0.2, size=len(voc_times)), piano_hits)
             for _ in range(200)]
    nulls = np.array(nulls)
    null_mean, null_std = float(nulls.mean()), float(nulls.std())
    z = (lock_obs - null_mean) / (null_std + 1e-12)

    _, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beat_lock = _lock_frac(voc_times, beat_times)

    return {
        "piano_hits": int(len(piano_hits)),
        "vocal_onsets": int(len(voc_times)),
        "syllables_per_s": round(len(voc_times) / dur, 2),
        "vocal_lock_vs_piano": round(lock_obs, 3),
        "null_lock_expected": round(null_mean, 3),
        "null_lock_std": round(null_std, 3),
        "lock_z": round(float(z), 2),
        "vocal_lock_vs_beat": round(beat_lock, 3),
    }


def _lock_frac(onsets: np.ndarray, refs: np.ndarray) -> float:
    if len(onsets) == 0 or len(refs) == 0:
        return 0.0
    nearest = np.abs(onsets[:, None] - refs[None, :]).min(axis=1)
    return float((nearest <= LOCK).mean())