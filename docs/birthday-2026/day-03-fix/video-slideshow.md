# Day 3 — Demo script & rebuild

## Theme

Day 3: Fix a broken project with Kiro Autopilot (red → green).

## Overview

Silent slideshow (~81s): intro (old GenAI course code) → before composite → short prompt →
Kiro conversation → after composite → summary → lesson / guard rails → Built with Kiro.

- Local MP4: [`captures/day-03-fix-demo.mp4`](captures/day-03-fix-demo.mp4)
- Pages: https://jajera.github.io/geonet-quake-classifier/
- Published YouTube: https://youtu.be/3BJNqb9sP6w

```bash
cd docs/birthday-2026/day-03-fix/captures && python3 build-demo.py
```

Requires Pillow + ffmpeg (and `libx264` or `libopenh264`).

---

## Beat table

| # | Beat | Source |
| - | ---- | ------ |
| 1 | Intro — old GenAI course code | generated + `ref-genai-aws-course.png` |
| 2 | Before card | generated |
| 3 | Before composite (dashboard + 404s) | before-01..03 |
| 4 | Short Kiro prompt (“only knew a little”) | generated |
| 5–8 | Kiro Autopilot conversation | kiro-01..04 |
| 9 | After card | generated |
| 10 | After composite (OK badges + working thumbs) | after-01 + thumbs |
| 11 | Summary | generated |
| 12 | Lesson — YMMV / right question / context | generated |
| 13 | Guard rails — verify + scrutinize | generated |
| 14 | Built with Kiro | generated |

`before-04-investigate-mismatch` is intentionally **not** in the demo.

---

## Smoke (after fix)

1. Open https://jajera.github.io/geonet-quake-classifier/
2. Earthquake Maps — all badges OK with timestamps
3. Open Decision Tree Model + Transformer Visualization — files load
