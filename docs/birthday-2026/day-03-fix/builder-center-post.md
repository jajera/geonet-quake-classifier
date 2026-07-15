# Day 3: Fix a functional problem in a broken project

Paste-ready fields for AWS Builder Center article submission (optional).

---

## Builder Center form fields

### Title

```text
Day 3: Fixing broken artifact links with Kiro Autopilot
```

### Description

```text
GeoNet Quake Classifier had silent 404s on the Earthquake Maps dashboard — UI, Python
generators, and CI expected different filenames. Kiro Autopilot diagnosed the drift,
fixed the contract, and added status badges. Short prompts + verify + human scrutiny.
```

### Tags

- `kiro` / `BuildWithKiro`
- `developer-tools`
- `generative-ai`
- `ci-cd` / `github-actions`
- `debugging`

### Body

---

## Article body

Kiro Birthday Week Day 3 asks for a **meaningful fix** of a broken project — investigate,
repair, show before/after.

### This was old code

GeoNet Quake Classifier started as a learning project while taking *Introducing Generative
AI with AWS*. The Earthquake Maps section listed links that looked complete; several model
PNG URLs returned 404.

### How Kiro helped

A short Autopilot prompt (symptom + badges, not a pre-written root cause) led Kiro to
compare `index.html`, `savefig` paths, and CI artifact globs. It aligned names, added a
transformer visualization, and put OK/Missing badges on the dashboard. A follow-up verify
pass and README postmortem closed the loop. CI then regenerated the missing files.

### Lesson

Vibe-coding mileage varies. Ask the right question, give useful context (faster and
cheaper — less thrash), ask the agent to verify as a guard rail — and still scrutinize
the result yourself.

### Links

- Repo: https://github.com/jajera/geonet-quake-classifier
- Pages: https://jajera.github.io/geonet-quake-classifier/
- Demo: https://youtu.be/3BJNqb9sP6w
- Demo MP4: [`captures/day-03-fix-demo.mp4`](captures/day-03-fix-demo.mp4)
