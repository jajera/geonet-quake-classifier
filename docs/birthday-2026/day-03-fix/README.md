# Day 3 — Fix a Broken Project

## Theme

Day 3: Use Kiro Autopilot to diagnose and fix a broken project (red → green).

Challenge: [Kiro Birthday 2026](https://kiro.dev/birthday/)

## What Was Broken

The **Earthquake Maps** section on `index.html` linked to generated artifacts that did not
exist under the expected filenames. Dashboard, Python `savefig` paths, and CI artifact uploads
disagreed — and `transformer_model.py` never wrote a PNG at all.

## What Was Fixed

- Standardized `{model}.png` / `{model}.html` across scripts, dashboard, and CI
- Added transformer visualization + matplotlib requirement
- Status badges (OK / Missing) + last-generated timestamps on the dashboard
- CI regenerated the missing PNGs after merge ([PR #21](https://github.com/jajera/geonet-quake-classifier/pull/21))
- Root `README.md` postmortem (~230 words)

## Deliverables

| File | Purpose |
| ---- | ------- |
| [form-submission.md](form-submission.md) | Paste-ready challenge form fields |
| [builder-center-post.md](builder-center-post.md) | Builder Center article draft |
| [social-post.md](social-post.md) | Social post draft |
| [video-slideshow.md](video-slideshow.md) | Demo beat table + rebuild notes |
| [before-state.md](before-state.md) | Pre-fix audit |
| [captures/](captures/) | Before/after/Kiro stills + demo MP4 |
| [OG image](captures/og-geonet-quake-classifier.png) | 1200×630 social / Open Graph |
| [Demo video](https://youtu.be/3BJNqb9sP6w) | YouTube (~81s silent) |
| [Demo MP4](captures/day-03-fix-demo.mp4) | Local rebuild source |
| Live Pages | https://jajera.github.io/geonet-quake-classifier/ |

## Rebuild demo

```bash
cd docs/birthday-2026/day-03-fix/captures
# needs Pillow + ffmpeg (e.g. /tmp/pillow-venv/bin/python)
python3 build-demo.py
```

## Submission Checklist

- [x] Public repository: https://github.com/jajera/geonet-quake-classifier
- [x] Root README postmortem (150–300 words)
- [x] Demo MP4 in `captures/`
- [x] Demo video: https://youtu.be/3BJNqb9sP6w
- [x] Social post: https://www.linkedin.com/posts/john-ajera_kiro-birthday-week-day-3-fix-a-broken-project-share-7483291944148914178-VYEM/
- [ ] Builder Center article URL (optional)
- [ ] `.kiro/` folder present (if required by the form)
