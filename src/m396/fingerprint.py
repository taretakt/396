"""The fingerprint: one pass over the audio, one measured card.

Fingerprint = features card + ride card + layer card. Stable JSON schema so
fingerprints are diffable, committed data — the repo ships fingerprints, never
audio. Nobody needs the song file to score a track already measured.
"""
from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from . import features as F
from . import ride as R
from . import pitch as P

SCHEMA = "m396-fingerprint-v0"


def measure(path: str, name: str | None = None, source: str | None = None) -> dict[str, Any]:
    y, sr = F.load(path)
    card = F.features(path)
    ride = R.ride(y, sr)
    pitch = P.pitch(path)
    fp = {
        "schema": SCHEMA,
        "track": name or os.path.splitext(os.path.basename(path))[0],
        "source": source or os.path.basename(path),
        **card,
        "beat_lock": ride.pop("vocal_lock_vs_beat"),
        "piano_lock_z": ride.pop("lock_z"),
        **ride,
        **pitch,
        "measured_at": date.today().isoformat(),
    }
    return {k: v for k, v in fp.items() if v is not None}


def save(fp: dict[str, Any], directory: str, key: str | None = None) -> str:
    os.makedirs(directory, exist_ok=True)
    ident = (key or fp["track"]).lower().replace(" ", "-").replace("/", "-")
    path = os.path.join(directory, f"{ident}.json")
    with open(path, "w") as fh:
        json.dump(fp, fh, indent=2)
    return path


def load_fingerprints(directory: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(directory, fn)) as fh:
            fp = json.load(fh)
        out[fp.get("track", fn[:-5])] = fp
    return out