# Day 3 — Before State (captured 2026-07-16)

Pre-fix audit. Used for the demo “before” beats.

## Symptom

Opening `index.html` → **Earthquake Maps**: HTML maps work; several model PNG links 404.

## Link audit (before CI regeneration)

| Link | Status |
| ---- | ------ |
| `decision_tree.html` | OK |
| `decision_tree.png` | **BROKEN** (script wrote `decision_tree_model.png`) |
| `neural_network.html` | OK |
| `neural_network.png` | **BROKEN** (script wrote `neural_network_model.png`) |
| `statistical_model.*` | OK |
| `ml_model.*` | OK |
| `neural_model.*` | OK |
| `transformer_model.html` | OK |
| `transformer_model.png` | **BROKEN** (no PNG generated) |

## Root cause

Artifact contract drift across `index.html`, Python `savefig` paths, and CI
`${{ matrix.model }}.png` uploads. Transformer never saved a visualization.

## After (post PR #21 + CI)

All 12 Earthquake Maps links resolve; dashboard badges show **OK** + timestamps on
https://jajera.github.io/geonet-quake-classifier/
