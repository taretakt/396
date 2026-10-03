"""396 CLI — the mood-channel machine.

Subcommands:
  analyze <audio-or-url>   measure one track into a fingerprint
  index <dir>              measure every audio file in a directory
  route <audio-or-track>   score a track against every channel
  channels                 list the mood channels
  seed                     (re)write the seed channels.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

from . import channels as CH
from . import fingerprint as FP

PKG_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_FPRINTS = os.path.join(PKG_DATA, "fingerprints")
DEFAULT_CHANNELS = os.path.join(PKG_DATA, "channels.json")
SUPPORTED = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".webm")


def _resolve_audio(arg: str) -> tuple[str, str | None]:
    """Local file, or a URL/name that yt-dlp can fetch (user's own responsibility)."""
    if os.path.exists(arg):
        return arg, None
    name = os.path.splitext(os.path.basename(arg))[0]
    out = tempfile.mktemp(suffix=".%(ext)s", prefix="m396_")
    found = None
    for cand in ("yt-dlp",):
        r = subprocess.run([cand, "-f", "bestaudio", "--extract-audio", "--audio-format", "mp3",
                            "--audio-quality", "0", "-o", out, f"ytsearch1:{arg}"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(out.replace(".%(ext)s", ".mp3")):
            found = out.replace(".%(ext)s", ".mp3")
            break
    if not found:
        raise SystemExit(f"could not fetch audio for: {arg}")
    return found, name


def cmd_analyze(args) -> None:
    for path, name in [_resolve_audio(args.audio)]:
        fp = FP.measure(path, name=args.name or name, source=args.audio)
        dest = FP.save(fp, args.out)
        print(json.dumps(fp, indent=2))
        print(f"saved: {dest}")


def cmd_index(args) -> None:
    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(SUPPORTED))
    if not files:
        raise SystemExit(f"no audio files in {args.dir}")
    for fn in files:
        path = os.path.join(args.dir, fn)
        name = os.path.splitext(fn)[0]
        dest = os.path.join(args.out, f"{name}.json")
        if os.path.exists(dest) and not getattr(args, "force", False):
            print(f"{name:<32} exists, skipping (--force to re-measure)")
            continue
        fp = FP.measure(path, name=name)
        dest = FP.save(fp, args.out, key=name)
        print(f"{name:<32} {fp['tempo']:>6} bpm  {fp['spectral_centroid_hz']:>6} Hz  "
              f"z={fp['piano_lock_z']:>6}  syll/s={fp['syllables_per_s']:>4}  saved")


def cmd_route(args) -> None:
    fingerprints = FP.load_fingerprints(args.fingerprints)
    fp = fingerprints.get(args.track)
    if fp is None and os.path.exists(args.track):
        fp = FP.measure(args.track, name=os.path.splitext(os.path.basename(args.track))[0])
    if fp is None:
        raise SystemExit(f"no fingerprint for {args.track!r} and not a local file — run `m396 analyze`")
    channels = json.load(open(args.channels))
    rows = CH.score_channels(fp, channels, fingerprints)
    print(f"routing {fp['track']}:")
    for r in rows:
        print(f"  {r['score']:>6}  {r['name']:<26} (d={r['distance']:.3f})")


def cmd_channels(args) -> None:
    channels = json.load(open(args.channels))
    for cid, ch in channels.items():
        print(f"{cid}:")
        print(f"  {ch.get('name')} — {ch.get('story')}")
        print(f"  anchors: {', '.join(ch['anchors'])}")
        print()


def cmd_seed(args) -> None:
    src = os.path.join(os.path.dirname(__file__), "..", "data", "channels.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} exists (use --force to overwrite)")
    else:
        data = json.load(open(src))
        json.dump(data, open(args.out, "w"), indent=2)
        print(f"seeded {len(data)} channels -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="m396", description="the mood-channel machine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="measure one track")
    a.add_argument("audio"); a.add_argument("--name"); a.add_argument("--out", default=DEFAULT_FPRINTS)
    a.set_defaults(fn=cmd_analyze)

    a = sub.add_parser("index", help="measure every audio file in a directory")
    a.add_argument("dir"); a.add_argument("--out", default=DEFAULT_FPRINTS); a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_index)

    a = sub.add_parser("route", help="score a track against the channels")
    a.add_argument("track")
    a.add_argument("--fingerprints", default=DEFAULT_FPRINTS)
    a.add_argument("--channels", default=DEFAULT_CHANNELS)
    a.set_defaults(fn=cmd_route)

    a = sub.add_parser("channels", help="list mood channels")
    a.add_argument("--channels", default=DEFAULT_CHANNELS)
    a.set_defaults(fn=cmd_channels)

    a = sub.add_parser("seed", help="write the seed channels.json")
    a.add_argument("--out", default=DEFAULT_CHANNELS); a.add_argument("--force", action="store_true")
    a.set_defaults(fn=cmd_seed)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()