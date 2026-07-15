# Day 3 — Challenge Form Submission

Paste-ready fields for the Kiro Birthday 2026 challenge form.

---

## Challenge Day

Day 3

## Project Name

GeoNet Quake Classifier

## Repository URL

https://github.com/jajera/geonet-quake-classifier

## Short Description

For Day 3 I fixed a broken GeoNet earthquake classifier dashboard: Earthquake Maps links
404'd because `index.html`, Python output paths, and GitHub Actions used different filenames.
Kiro Autopilot traced the mismatch, aligned the artifact contract, added status badges and
timestamps, and a CI run regenerated the missing files.

## How Kiro Was Used

I pointed Kiro Autopilot at the repo with a short symptom-only prompt (broken Earthquake Maps
links + status badges). Autopilot compared dashboard hrefs, `savefig` paths, and CI uploads,
fixed naming drift and the missing transformer PNG, then verified end-to-end and wrote a
README postmortem. I asked it to verify as a guard rail and still checked the live Pages site
myself.

## Demo Video URL

https://youtu.be/3BJNqb9sP6w

## Social Post URL

https://www.linkedin.com/posts/john-ajera_kiro-birthday-week-day-3-fix-a-broken-project-share-7483291944148914178-VYEM/

## Builder Center Article URL

`TBD` (optional)
