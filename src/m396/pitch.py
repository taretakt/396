"""Layer probes — faithful port of the validated standalone pitch_knob.py.

How far the lead line wanders (trace), plus the halo tells: stereo side/mid
width and bright-band ratio over voice-active frames. Single-f0 tracking on
dense mixes is honest-but-limited: the gate is energy-based, so frames can mix
voice and keys at pitch. The metrics that survive are exactly these.

Reference values reproduced by design (2026-10-03): Tonedeff Porcelain
trace_p5_p95=14.53, side_mid=0.17, high_low=0.281; Slomo trace=12.05.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import librosa
from scipy import signal

SR = 22050
HOP = 1024
N_FFT = 2048
CAP = 270.0


def pitch(path: str) -> dict[str, Any]:
    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y[: int(CAP * sr)]

    sos = signal.butter(4, [80.0, 3000.0], btype="bandpass", fs=sr, output="sos")
    yb = signal.sosfilt(sos, y)
    yb = yb / (np.abs(yb).max() + 1e-9)

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    f0 = librosa.yin(yb, fmin=75.0, fmax=900.0, sr=sr, hop_length=HOP)
    env = np.abs(librosa.stft(yb, n_fft=N_FFT, hop_length=HOP)).mean(axis=0)
    gate = env > np.percentile(env, 30)
    f = f0[gate]
    f = f[np.isfinite(f)]
    if len(f) < 400:
        return {"median_f0_hz": None, "raw_pitch_std_st": None, "raw_p5_p95_st": None,
                "trace_pitch_std_st": None, "trace_p5_p95_st": None,
                "vocal_band_centroid_hz": None, "high_low_vocal_ratio": None,
                "side_mid_vocal_ratio": None}

    # raw spread = lead voice + layer crossings (the audible knob)
    sem = 12.0 * np.log2(f / np.nanmedian(f))
    # traced spread = median-filtered contour (the lead line)
    fmed = signal.medfilt(f0, kernel_size=31)
    fm = fmed[gate]
    fm = fm[np.isfinite(fm)]
    semm = 12.0 * np.log2(fm / np.nanmedian(fm))
    p5m, p95m = np.percentile(semm, 5), np.percentile(semm, 95)

    vf = np.flatnonzero(gate)
    lo = (freqs >= 300) & (freqs < 1200)
    hi = (freqs >= 1200) & (freqs < 4000)
    high_low = float(S[hi][:, vf].mean() / (S[lo][:, vf].mean() + 1e-9))
    vb = (freqs >= 300) & (freqs < 4000)
    cents = float((S[vb][:, vf] * freqs[vb][:, None]).sum() / (S[vb][:, vf].sum() + 1e-9))

    side_ratio = None
    stereo = librosa.load(path, sr=sr, mono=False)[0][:, : int(CAP * sr)]
    if stereo.ndim == 2 and stereo.shape[0] == 2:
        Ls = np.abs(librosa.stft(stereo[0], n_fft=N_FFT, hop_length=HOP))
        Rs = np.abs(librosa.stft(stereo[1], n_fft=N_FFT, hop_length=HOP))
        mid = (Ls + Rs) / 2.0
        side = np.abs(Ls - Rs) / 2.0
        side_ratio = float(side[vb][:, vf].mean() / (mid[vb][:, vf].mean() + 1e-9))

    return {
        "median_f0_hz": round(float(np.nanmedian(f)), 1),
        "raw_pitch_std_st": round(float(np.nanstd(sem)), 2),
        "raw_p5_p95_st": round(float(np.nanpercentile(sem, 95) - np.nanpercentile(sem, 5)), 2),
        "trace_pitch_std_st": round(float(np.nanstd(semm)), 2),
        "trace_p5_p95_st": round(float(p95m - p5m), 2),
        "vocal_band_centroid_hz": round(cents, 0),
        "high_low_vocal_ratio": round(high_low, 3),
        "side_mid_vocal_ratio": round(side_ratio, 3) if side_ratio is not None else None,
    }