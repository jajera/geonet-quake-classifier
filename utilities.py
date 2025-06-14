"""
Utilities module for GeoNet Earthquake Classifier

This module contains common functions used across
all earthquake classifier models
to reduce code duplication and maintain consistency.
"""

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pandas as pd

from config import (
    API_BASE_URL,
    DEFAULT_FILTERS,
    INTENSITY_THRESHOLD,
    DAYS_FILTER,
    MIN_MAGNITUDE,
    GEONET_EARTHQUAKE_URL_BASE,
)


def build_api_url(filters=None):
    """
    Build the GeoNet API URL with configured filters.

    Args:
        filters (dict, optional): Custom filters for the GeoNet API query.
            Defaults to DEFAULT_FILTERS from config.

    Returns:
        str: The complete API URL with query parameters.
    """
    filters = filters or DEFAULT_FILTERS
    params = {k: str(v) for k, v in filters.items() if v is not None}
    return f"{API_BASE_URL}?{urlencode(params)}"


def fetch_data(api_url):
    """
    Fetch earthquake data from the GeoNet API.

    Args:
        api_url (str): The API URL to fetch data from.

    Returns:
        list: List of earthquake features from the API response.
    """
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.load(response)
            return data.get("features", [])
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"Error fetching data: {e}")
        return []


def process_data(features):
    """
    Process raw earthquake features into a structured DataFrame.

    Args:
        features (list): List of earthquake features from the API.

    Returns:
        pd.DataFrame: Processed earthquake data.
    """
    records = []
    print(f"\nProcessing {len(features)} earthquakes")

    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        ts = props.get("origintime", "")
        if not ts:
            continue

        try:
            quake_time = datetime.strptime(
                ts.split(".")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        magnitude = props.get("magnitude", 0)
        depth = props.get("depth", 0)
        publicid = props.get("publicid", "")

        record = {
            "publicid": publicid,
            "magnitude": magnitude,
            "depth": depth,
            "latitude": coords[1],
            "longitude": coords[0],
            "timestamp": ts,
            "quake_time": quake_time,
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Created DataFrame with {len(df)} records")
    return df.dropna()


def filter_data(df):
    """
    Filter earthquake data based on time and magnitude criteria.

    Args:
        df (pd.DataFrame): Input earthquake data.

    Returns:
        pd.DataFrame: Filtered earthquake data.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=DAYS_FILTER)
    # fmt: off
    filtered = df[
        (df["quake_time"] >= cutoff) & (df["magnitude"] >= MIN_MAGNITUDE)
    ]  # noqa: E501
    # fmt: on
    print(f"Filtered to {len(filtered)} records")
    return filtered


def generate_map_data(df):
    """
    Generate data for map visualization.

    Args:
        df (pd.DataFrame): Processed earthquake data.

    Returns:
        list: List of dictionaries containing map marker data.
    """
    map_data = [
        {
            "lat": row["latitude"],
            "lon": row["longitude"],
            "magnitude": row["magnitude"],
            "depth": row["depth"],
            "intensity": row["intensity"],
            "timestamp": row["quake_time"].strftime(
                "%a, %b %d %Y, %I:%M:%S %p"
            ),  # noqa: E501
            "publicid": row["publicid"],
        }
        for _, row in df.iterrows()
    ]
    print(f"Generated map data for {len(map_data)} earthquakes")
    return map_data


def generate_map_html(
    data,
    min_magnitude,
    model_config,
    intensity_type="predicted",
):  # noqa: E501
    """
    Generate an interactive HTML map with earthquake data.

    Args:
        data (list): List of earthquake data for map markers.
        min_magnitude (float): Minimum magnitude threshold
          for intensity classification.
        model_config (dict): Dictionary containing model display settings:
            - name (str): Name of the model (e.g., "Decision Tree Model")
            - emoji (str): Emoji to display for the model (e.g., "🌳")
            - header_color (str): CSS color for the header background
        intensity_type (str): Type of intensity classification
          ("predicted" or "actual").
    """
    model_name = model_config.get("name", "Unknown Model")
    model_emoji = model_config.get("emoji", "🌍")
    header_color = model_config.get("header_color", "#2c3e50")
    leaflet_css = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    leaflet_js = "https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoNet Earthquake Classifier - {model_name}</title>
    <link rel="stylesheet" href="{leaflet_css}" />
    <link rel="icon" type="image/png" href="favicon.png" />
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        .header {{
            background-color: {header_color};
            color: white;
            padding: 1rem;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.5rem;
        }}
        .header p {{
            margin: 0.5rem 0 0;
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        #map {{
            height: calc(100vh - 200px);
            width: 100%;
            margin: 0;
        }}
        .footer {{
            background-color: #34495e;
            color: white;
            text-align: center;
            padding: 1rem;
            font-size: 0.8rem;
        }}
        .footer a {{
            color: #3498db;
            text-decoration: none;
        }}
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{model_emoji} GeoNet Earthquake Classifier - {model_name}</h1>
        <p>
            Real-time earthquake intensity predictions using {model_name}
        </p>
    </div>
    <div id="map"></div>
    <div class="footer">
        <p>Data source:
            <a href="https://www.geonet.org.nz/" target="_blank">
                GeoNet New Zealand</a> |
                Earthquakes from the last {DAYS_FILTER}
                days with magnitude ≥ {min_magnitude}
        </p>
        <p>
            High intensity: Magnitude ≥ {INTENSITY_THRESHOLD} |
                Low intensity: Magnitude < {INTENSITY_THRESHOLD}</p>
    </div>

    <script src="{leaflet_js}"></script>
    <script>
        // Initialize map centered on New Zealand
        var map = L.map('map').setView([-41.2865, 174.7762], 6);

        // Add OpenStreetMap tiles
        L.tileLayer(
        'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Earthquake data
        var earthquakes = {json.dumps(data)};

        // Add markers for each earthquake
        earthquakes.forEach(function(quake) {{
            var color = quake.intensity === 'High' ? 'red' : 'green';
            var size = Math.max(5, quake.magnitude * 2);

            var marker = L.circleMarker([quake.lat, quake.lon], {{
                radius: size,
                fillColor: color,
                color: color,
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.6
            }}).addTo(map);

            // Create popup with earthquake details
            var popupContent = `
                <div style="font-family: Arial, sans-serif;">
                    <h3 style="margin: 0 0 10px 0; color: {header_color};">
                        Magnitude ${{quake.magnitude}}
                    </h3>
                    <p style="margin: 5px 0;"><strong>Intensity:</strong>
                        <span style="color: ${{color}}; font-weight: bold;">
                            ${{quake.intensity}}
                        </span>
                    </p>
                    <p style="margin: 5px 0;">
                        <strong>Depth:</strong> ${{quake.depth}} km
                    </p>
                    <p style="margin: 5px 0;">
                        <strong>Time:</strong> ${{quake.timestamp}}
                    </p>
                    <p style="margin: 5px 0;"><strong>Location:</strong>
                        ${{quake.lat.toFixed(4)}}, ${{quake.lon.toFixed(4)}}
                    </p>
                    <p style="margin: 10px 0 0 0;">
            <a href="${{"{GEONET_EARTHQUAKE_URL_BASE}"}}${{quake.publicid}}"
                           target="_blank"
                           style="color: #3498db; text-decoration: none;">
                            View on GeoNet →
                        </a>
                    </p>
                </div>
            `;

            marker.bindPopup(popupContent);
        }});

        // Add legend
        var legend = L.control({{position: 'bottomright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.style.backgroundColor = 'white';
            div.style.padding = '10px';
            div.style.border = '2px solid #ccc';
            div.style.borderRadius = '5px';
            div.innerHTML = `
                <h4 style="margin: 0 0 10px 0;">
                    Intensity ({intensity_type})
                </h4>
                <div style="margin: 5px 0;">
                    <span
                        style="
                            display: inline-block;
                            width: 12px;
                            height: 12px;
                            background-color: red;
                            border-radius: 50%;
                            margin-right: 8px;"
                    ></span>
                    High (Mag >= {INTENSITY_THRESHOLD})
                </div>
                <div style="margin: 5px 0;">
                    <span style="
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        background-color: green;
                        border-radius: 50%;
                        margin-right: 8px;"
                    ></span>
                    Low (Mag < {INTENSITY_THRESHOLD})
                </div>
            `;
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
"""
    return html


def calculate_intensity_class(magnitude):
    """
    Calculate intensity class based on magnitude threshold.

    Args:
        magnitude (float): Earthquake magnitude.

    Returns:
        int: 1 for High intensity, 0 for Low intensity.
    """
    return 1 if magnitude >= INTENSITY_THRESHOLD else 0


def apply_intensity_labels(df):
    """
    Apply intensity labels to DataFrame based on intensity_class.

    Args:
        df (pd.DataFrame): DataFrame with intensity_class column.

    Returns:
        pd.DataFrame: DataFrame with added intensity column.
    """
    df["intensity"] = df["intensity_class"].apply(
        lambda x: "High" if x else "Low"
    )  # noqa: E501
    return df


def print_dataset_summary(df):
    """
    Print a summary of the earthquake dataset.

    Args:
        df (pd.DataFrame): DataFrame with intensity_class column.
    """
    high_count = df["intensity_class"].sum()
    low_count = len(df) - high_count
    print("\nDataset Summary:")
    print(f"Total earthquakes: {len(df)}")
    print(f"High intensity: {high_count}")
    print(f"Low intensity: {low_count}")


def save_html_file(html_content, filename):
    """
    Save HTML content to a file.

    Args:
        html_content (str): The HTML content to save.
        filename (str): The filename to save to.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated map: {filename}")
