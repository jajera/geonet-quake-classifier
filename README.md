# GeoNet Quake Classifier

A lightweight, real-time earthquake classifier and static map using live data from GeoNet New Zealand.

## Features

- Fetches live earthquake data from GeoNet's public API
- Classifies earthquakes as "High" or "Low" intensity using a Decision Tree
- Generates a self-contained static HTML map with color-coded markers
- Includes popup information for each earthquake event

## Setup

1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the classifier:

    ```bash
    python decision_tree.py
    ```

This will:

- Fetch the latest earthquake data from GeoNet
- Train a decision tree classifier
- Generate an interactive map (`decision_tree.html`)

## Map Features

- Red markers: High intensity earthquakes (MMI >= 5)
- Green markers: Low intensity earthquakes (MMI < 5)
- Marker size corresponds to earthquake magnitude
- Click markers to view detailed information
- Includes a legend and OpenStreetMap base layer

## Requirements

To run this project, you need:

- Python 3.7+ (or later)
- The core dependencies listed in `requirements.txt`:
  - pandas
  - scikit-learn
  - requests

For development and linting, you also need the development dependencies listed in `requirements-dev.txt`:

- flake8
- pylint

## License

This code is available under the [MIT license](http://opensource.org/licenses/MIT).

## Postmortem: Broken Earthquake Maps Links (2026-07-16)

**What was broken:** Several links in the "Earthquake Maps" section of `index.html` returned 404s.
The dashboard and CI both expected files named `<model>.png` (e.g. `decision_tree.png`), but two
scripts used a different convention — `decision_tree.py` wrote `decision_tree_model.png` and
`neural_network.py` wrote `neural_network_model.png`. Additionally, `transformer_model.py` never
generated a PNG at all, so `transformer_model.png` was always missing.

**How it was diagnosed:** Compared the `href` values in `index.html` and the CI artifact upload
glob (`${{ matrix.model }}.png`) against the actual `plt.savefig(...)` calls in each Python script.
The mismatch was immediately visible: two scripts appended `_model` to the filename, one script
skipped PNG generation entirely.

**What changed:**

- `decision_tree.py` / `neural_network.py`: renamed `savefig` output to `decision_tree.png` /
  `neural_network.png`.
- `transformer_model.py`: added a `generate_visualization()` function producing
  `transformer_model.png`, plus added `matplotlib` to its requirements file.
- Deleted the orphaned `*_model.png` files from the repo.
- Enhanced the Earthquake Maps section in `index.html` with status badges (OK / Missing) and
  last-generated timestamps via lightweight `HEAD` requests.

**Still incomplete:** The three corrected PNG files (`decision_tree.png`, `neural_network.png`,
`transformer_model.png`) won't physically exist until the next CI run regenerates them. The
dashboard will show them as "Missing" until then.

**Lesson learned:** When a CI pipeline builds filenames from a matrix variable, every generator
must follow that exact naming convention. A single source of truth for output filenames — ideally
in `config.py` — would have prevented this drift entirely.
