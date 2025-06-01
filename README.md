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
    python quake_classifier.py
    ```

This will:

- Fetch the latest earthquake data from GeoNet
- Train a decision tree classifier
- Generate an interactive map (`quake_map.html`)

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
