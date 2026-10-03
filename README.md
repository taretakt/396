# 396 — the mood-channel machine

[![CI](https://github.com/taretakt/396/actions/workflows/ci.yml/badge.svg)](https://github.com/taretakt/396/actions/workflows/ci.yml)

A track is a **fingerprint**: gait (does the voice ride the beat, the piano, or
thread between the hits?), texture (warm or midnight), layer (wide or mono),
pace. 396 measures the fingerprint from the audio and **routes it to mood
channels** — rooms named the way you'd name a scent. Point it at any track and
it tells you which rooms it belongs in, and why.

Built and verified on four real tracks (2026-10):

| anchor | gait | the measured tell |
|---|---|---|
| Porcelain (Tonedeff) | dance | cadence locks to piano hits at **z=+5.1** (2.5× above chance) |
| Razorblade Salvation (Jedi Mind Tricks) | bulldoze | 67.5% beat lock, piano-lock at chance |
| Dance with the Devil (Immortal Technique) | drift | piano-lock **below** chance (z=−4.7) — threads between the hits |
| Slomo (Oliver Francis) | slow drift | slowest lane, brightest, only builder (+5.4 dB) |

## What it measures

| card | feature | meaning |
|---|---|---|
| features | tempo (beat-counted), spectral centroid, bass fraction, percussive ratio, syncopation, dynamics arc, key | the static card |
| ride | piano hits, vocal onsets, **lock vs piano (z-scored against a shuffle null)**, beat lock | gait |
| layer | lead-line pitch wander (p5–p95 semitones), vocal-band centroid, high/low ratio, side/mid width | the halo tells |

Everything runs on fixed absolute scales — never z-scored against the pool — so
a score means the same thing no matter what else is in the folder.

## Quickstart

```bash
uv sync --extra dev
uv run m396 analyze path/to/track.mp3        # one fingerprint
uv run m396 index ~/music                     # a whole folder
uv run m396 route path/to/track.mp3           # which rooms does it live in
uv run m396 channels                          # the mood channels
```

`analyze` also accepts a track name (fetches audio via yt-dlp — your machine,
your call, supply files you're allowed to have).

## How routing works

A channel is a codeword + story + anchor fingerprints. Candidates score against
the channel centroid (weighted Euclidean distance on the fixed scales,
`score = 1/(1+distance)`). More anchors = sharper rooms. Seed channels:

- **white-vanilla-orchids** — the sweet floral room, slow and bright
- **porcelain** — piano-ride soul food, unhurried
- **razorblade** — the dark passenger, keys under a steamroller
- **dread-drift** — the creeper, flat and steely

## Honest limits

- 396 scores **structure**, not taste. "Rides the piano" is a measured gait;
  whether it moves you is your weather.
- Single-pitch tracking on dense mixes mixes voice and keys at pitch — the
  layer card ships the metrics that survive that fuzz (lead-line wander,
  stereo width, bright-band ratio).
- One-anchor channels self-match by construction; they sharpen as anchors join.
- The repo ships fingerprints and channel data — **never audio**.

## Roadmap

- Multi-anchor channels (the rooms get sharper)
- Public web face: name a mood, get a room, get tracks
- The bench: pre-register a read, get scored (it's how 396 was born)

MIT — (c) 2026 Ross MacDonald