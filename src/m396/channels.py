"""Mood channels: named rooms anchored by measured tracks.

A channel is a codeword (named like a scent, a bar, a feeling), a plain-language
story, and one or more anchor fingerprints. Candidate tracks score against the
channel centroid on FIXED ABSOLUTE SCALES — never z-scored against the pool —
so a score means the same thing no matter what else is in the folder.

Scales (fixed, musically meaningful):
  tempo 10 BPM | centroid 500 Hz | bass 0.20 | perc 0.20 | sync 0.15 |
  dyn 3 dB | beat_lock 0.15 | piano_lock_z 2.0 | syll_s 0.5 |
  trace_p5_p95 5 st | vocal_centroid 600 Hz | high_low 0.30 | side_mid 0.10

Claim discipline: 396 scores STRUCTURE. It cannot taste. "Rides the piano",
"bulldozes the beat", "threads between the hits" are measured gaits; whether
that moves you is your weather.
"""
from __future__ import annotations

import json
from typing import Any

FEATURES = [
    "tempo", "spectral_centroid_hz", "bass_fraction", "percussive_ratio",
    "syncopation", "dyn_mean_db", "dyn_std_db", "fade_delta_db",
    "beat_lock", "piano_lock_z", "syllables_per_s",
    "trace_p5_p95_st", "vocal_band_centroid_hz",
    "high_low_vocal_ratio", "side_mid_vocal_ratio",
]

SCALES: dict[str, float] = {
    "tempo": 10.0, "spectral_centroid_hz": 500.0, "bass_fraction": 0.20,
    "percussive_ratio": 0.20, "syncopation": 0.15,
    "dyn_mean_db": 3.0, "dyn_std_db": 3.0, "fade_delta_db": 3.0,
    "beat_lock": 0.15, "piano_lock_z": 2.0, "syllables_per_s": 0.5,
    "trace_p5_p95_st": 5.0, "vocal_band_centroid_hz": 600.0,
    "high_low_vocal_ratio": 0.30, "side_mid_vocal_ratio": 0.10,
}

WEIGHTS: dict[str, float] = {
    "tempo": 0.14, "spectral_centroid_hz": 0.10, "bass_fraction": 0.05,
    "percussive_ratio": 0.05, "syncopation": 0.10,
    "dyn_mean_db": 0.02, "dyn_std_db": 0.01, "fade_delta_db": 0.04,
    "beat_lock": 0.14, "piano_lock_z": 0.14, "syllables_per_s": 0.05,
    "trace_p5_p95_st": 0.10, "vocal_band_centroid_hz": 0.03,
    "high_low_vocal_ratio": 0.05, "side_mid_vocal_ratio": 0.07,
}


def _dyn_composite(fp: dict[str, Any]) -> float:
    return (fp.get("dyn_mean_db", 0.0) ** 2 + fp.get("dyn_std_db", 0.0) ** 2
            + fp.get("fade_delta_db", 0.0) ** 2) ** 0.5


def _vec(fp: dict[str, Any]) -> dict[str, float]:
    v = {k: fp.get(k, 0.0) for k in FEATURES}
    v["dyn_composite"] = _dyn_composite(fp)
    return v


def channel_distance(cand: dict[str, Any], anchors: list[dict[str, Any]]) -> float:
    """Weighted Euclidean distance from the candidate to the anchor centroid."""
    c = _vec(cand)
    centroid = {k: sum(_vec(a)[k] for a in anchors) / len(anchors) for k in c}
    total = sum(WEIGHTS.values())
    d2 = 0.0
    for k, w in WEIGHTS.items():
        if k in ("dyn_mean_db", "dyn_std_db", "fade_delta_db"):
            continue
        d2 += (w / total) * ((c[k] - centroid[k]) / SCALES[k]) ** 2
    # the three dynamics legs folded into one scaled composite
    d2 += (WEIGHTS["dyn_mean_db"] + WEIGHTS["dyn_std_db"] + WEIGHTS["fade_delta_db"]) / total \
        * ((c["dyn_composite"] - centroid["dyn_composite"]) / 3.0) ** 2
    return d2 ** 0.5


def score_channels(cand: dict[str, Any], channels: dict[str, dict[str, Any]],
                   fingerprints: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cid, ch in channels.items():
        anchors = [fingerprints[a] for a in ch["anchors"] if a in fingerprints]
        if not anchors:
            continue
        d = channel_distance(cand, anchors)
        rows.append({
            "channel": cid,
            "name": ch.get("name", cid),
            "distance": round(d, 3),
            "score": round(1.0 / (1.0 + d), 3),
            "anchors": ch["anchors"],
        })
    return sorted(rows, key=lambda r: r["distance"])