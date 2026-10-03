import json
import os
import wave

import numpy as np
import pytest

from m396 import fingerprint as FP
from m396 import channels as CH

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

REQUIRED_KEYS = [
    "tempo", "spectral_centroid_hz", "bass_fraction", "percussive_ratio", "syncopation",
    "dyn_mean_db", "dyn_std_db", "fade_delta_db", "key",
    "beat_lock", "piano_lock_z", "syllables_per_s", "trace_p5_p95_st",
    "vocal_band_centroid_hz", "high_low_vocal_ratio", "side_mid_vocal_ratio",
]


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory):
    """30 s of 440 Hz tone + noise bursts — a synthetic, deterministic track.
    30 s clears the 400-voiced-frame floor in the pitch probe (~21.5 fps @ hop 1024)."""
    sr = 22050
    t = np.arange(sr * 30) / sr
    tone = 0.4 * np.sin(2 * np.pi * 440 * t)
    bursts = (t % 0.5 < 0.08).astype(float) * 0.3 * np.random.default_rng(0).normal(0, 1, len(t))
    sig = (tone + bursts).astype(np.float32)
    # interleave stereo so mono downmix keeps alignment
    stereo = np.stack([sig, sig], axis=1)
    path = tmp_path_factory.mktemp("wav") / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
    return str(path)


def test_measure_runs_and_schema(tone_wav):
    fp = FP.measure(tone_wav, name="tone")
    for k in REQUIRED_KEYS:
        assert k in fp, f"missing {k}"
    assert fp["schema"] == "m396-fingerprint-v0"
    assert 40 < fp["tempo"] < 220
    assert fp["duration_s"] == pytest.approx(30.0, abs=0.3)


def test_fingerprint_roundtrip(tone_wav, tmp_path):
    fp = FP.measure(tone_wav, name="tone")
    dest = FP.save(fp, str(tmp_path))
    loaded = FP.load_fingerprints(str(tmp_path))
    assert loaded["tone"] == fp


def test_packaged_seed_channels_and_anchors_consistent():
    channels = json.load(open(os.path.join(DATA, "channels.json")))
    fps = FP.load_fingerprints(os.path.join(DATA, "fingerprints"))
    assert set(channels) == {"white-vanilla-orchids", "porcelain", "razorblade", "dread-drift"}
    for cid, ch in channels.items():
        for a in ch["anchors"]:
            assert a in fps, f"anchor {a} missing from packaged fingerprints"


def test_anchor_routes_to_own_channel():
    channels = json.load(open(os.path.join(DATA, "channels.json")))
    fps = FP.load_fingerprints(os.path.join(DATA, "fingerprints"))
    for cid, ch in channels.items():
        fp = fps[ch["anchors"][0]]
        rows = CH.score_channels(fp, channels, fps)
        assert rows[0]["channel"] == cid, f"{ch['anchors'][0]} should route to {cid}"


def test_anchor_separation():
    """The four anchors must be distinct animals — cross-distance stays substantial."""
    channels = json.load(open(os.path.join(DATA, "channels.json")))
    fps = FP.load_fingerprints(os.path.join(DATA, "fingerprints"))
    f = {cid: fps[ch["anchors"][0]] for cid, ch in channels.items()}
    pairs = [("porcelain", "razorblade"), ("porcelain", "dread-drift"),
             ("porcelain", "white-vanilla-orchids"), ("razorblade", "dread-drift"),
             ("razorblade", "white-vanilla-orchids"), ("dread-drift", "white-vanilla-orchids")]
    for a, b in pairs:
        d = CH.channel_distance(f[a], [f[b]])
        assert d > 0.30, f"{a} vs {b} distance {d:.3f} — fingerprints collapsed"