"""
GeoNet Quake Classifier - Neural Network

This module provides functionality to fetch, process,
and classify earthquake data from GeoNet New Zealand's API
using a neural network model. It includes features for data
processing, model training, and generating an interactive
map visualization.
"""

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from config import (
    API_BASE_URL,
    DEFAULT_FILTERS,
    INTENSITY_THRESHOLD,
    DAYS_FILTER,
    NEURAL_NETWORK_MAP_INTENSITY_TYPE,
    MIN_MAGNITUDE,
    GEONET_EARTHQUAKE_URL_BASE,
    NEURAL_NETWORK_CONFIG,
)


class QuakeClassifierNeuralNetwork:
    """
    A class to handle earthquake data processing and classification
    using a neural network model.

    This class provides methods to fetch earthquake data from GeoNet,
    process and filter the data, train a neural network classifier,
    and generate map visualization data.
    """

    def __init__(self, filters=None):
        """
        Initialize the QuakeClassifierNeuralNetwork with optional filters.

        Args:
            filters (dict, optional): Custom filters for the GeoNet API query.
                Defaults to DEFAULT_FILTERS from config.
        """
        self.filters = filters or DEFAULT_FILTERS
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.api_url = self.build_api_url()

    def _build_model(self):
        """
        Build and compile the neural network model.

        Returns:
            tf.keras.Model: Compiled neural network model.
        """
        model = Sequential()

        # Input layer with batch normalization
        model.add(
            Dense(
                NEURAL_NETWORK_CONFIG["hidden_layers"][0],
                activation=NEURAL_NETWORK_CONFIG["activation"],
                input_shape=(2,),  # magnitude and depth
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Hidden layers with batch normalization and dropout
        # fmt: off
        for units in NEURAL_NETWORK_CONFIG["hidden_layers"][1:]:
            model.add(Dense(
                units,
                activation=NEURAL_NETWORK_CONFIG["activation"])
            )
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(
            1,
            activation=NEURAL_NETWORK_CONFIG["output_activation"])
        )

        # Compile model with a lower learning rate
        model.compile(
            optimizer=Adam(
                learning_rate=NEURAL_NETWORK_CONFIG["learning_rate"]
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        # fmt: on

        return model

    def build_api_url(self):
        """
        Build the GeoNet API URL with configured filters.

        Returns:
            str: The complete API URL with query parameters.
        """
        params = {k: str(v) for k, v in self.filters.items() if v is not None}
        return f"{API_BASE_URL}?{urlencode(params)}"

    def fetch_data(self):
        """
        Fetch earthquake data from the GeoNet API.

        Returns:
            list: List of earthquake features from the API response.
        """
        try:
            with urllib.request.urlopen(self.api_url) as response:
                data = json.load(response)
                return data.get("features", [])
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            return []

    def process_data(self, features):
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

    def filter_data(self, df):
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
            (df["quake_time"] >= cutoff)
            & (df["magnitude"] >= MIN_MAGNITUDE)
        ]
        # fmt: on
        print(f"Filtered to {len(filtered)} records")
        return filtered

    def predict_intensity(self, magnitude, depth):
        """
        Predict earthquake intensity using the trained model.

        Args:
            magnitude (float): Earthquake magnitude.
            depth (float): Earthquake depth in kilometers.

        Returns:
            dict: Predicted intensity class and label.
        """
        # Scale the input features
        scaled_input = self.scaler.transform([[magnitude, depth]])
        pred = self.model.predict(scaled_input, verbose=0)[0][0]
        intensity_class = 1 if pred >= 0.5 else 0
        return {
            "intensity_class": intensity_class,
            "intensity": "High" if intensity_class else "Low",
        }

    # fmt: off
    def train_model(
            self,
            features_train,
            labels_train,
            features_test,
            labels_test
    ):
        """
        Train the neural network model and generate visualization.

        Args:
            features_train (pd.DataFrame): Training features.
            labels_train (pd.Series): Training labels.
            features_test (pd.DataFrame): Test features.
            labels_test (pd.Series): Test labels.

        Returns:
            float: Model accuracy score.
        """
        # Scale the features
        features_train_scaled = self.scaler.fit_transform(features_train)
        features_test_scaled = self.scaler.transform(features_test)

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
        ]

        # Train the model
        history = self.model.fit(
            features_train_scaled,
            labels_train,
            epochs=NEURAL_NETWORK_CONFIG["epochs"],
            batch_size=NEURAL_NETWORK_CONFIG["batch_size"],
            validation_split=NEURAL_NETWORK_CONFIG["validation_split"],
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate accuracy
        _, accuracy = self.model.evaluate(
            features_test_scaled,
            labels_test,
            verbose=0
        )

        # Generate predictions for classification report
        predictions = (
            self.model.predict(features_test_scaled) >= 0.5
        ).astype(int)
        print("\nClassification Report:")
        print(classification_report(labels_test, predictions))

        # Plot training history
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        plt.tight_layout()
        plt.savefig("neural_network_model.png")
        plt.close()

        return accuracy

    def generate_map_data(self, df):
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
                ),
                "publicid": row["publicid"],
            }
            for _, row in df.iterrows()
        ]
        print(f"Generated map data for {len(map_data)} earthquakes")
        return map_data
    # fmt: on


def generate_map_html(data, min_magnitude):
    """
    Generate an interactive HTML map with earthquake data.

    Args:
        data (list): List of earthquake data for map markers.
        min_magnitude (float): Minimum magnitude threshold
          for intensity classification.
    """
    leaflet_css = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    leaflet_js = "https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Earthquake Map - Neural Network</title>
        <link rel="icon" type="image/png" href="favicon.png">
        <link rel="stylesheet" href="{leaflet_css}" />
        <script src="{leaflet_js}"></script>
        <style>
            #map {{ height: 100vh; }}
            .legend {{
                padding: 6px 8px;
                background: white;
                background: rgba(255,255,255,0.8);
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
            }}
            .legend i {{
                width: 18px;
                height: 18px;
                float: left;
                margin-right: 8px;
                opacity: 0.7;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var GEONET_EARTHQUAKE_URL_BASE = "{GEONET_EARTHQUAKE_URL_BASE}";
            var map = L.map('map').setView([-40.9, 174.9], 6);
            L.tileLayer(
                'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
                {{
                    'attribution': '© OpenStreetMap contributors'
                }}
            ).addTo(map);

            var data = {json.dumps(data)};
            console.log("Data loaded:", data.length, "earthquakes");

            data.forEach(function(q) {{
                var color = q.intensity === 'High' ? 'red' : 'green';
                var marker = L.circleMarker([q.lat, q.lon], {{
                    radius: q.magnitude * 2,
                    fillColor: color,
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }});

                marker.bindPopup(
                    '<b>Public ID:</b> <a href="' +
                    GEONET_EARTHQUAKE_URL_BASE + q.publicid +
                    '" target="_blank">' + q.publicid +
                    '</a><br>' +
                    '<b>Mag:</b> ' + q.magnitude.toFixed(1) + '<br>' +
                    '<b>Depth:</b> ' + q.depth.toFixed(1) + ' km<br>' +
                    '<b>Intensity:</b> ' + q.intensity + '<br>' +
                    '<b>Time:</b> ' + q.timestamp
                );
                marker.addTo(map);
            }});

            // Add legend
            var legend = L.control({{position: 'bottomright'}});
            legend.onAdd = function(map) {{
                var div = L.DomUtil.create('div', 'legend');
                div.innerHTML = '<h4>Intensity (' +
                    '{NEURAL_NETWORK_MAP_INTENSITY_TYPE})</h4>' +
                    '<i style="background: red"></i>' +
                    ' High (Mag >= {min_magnitude})<br>' +
                    '<i style="background: green"></i>' +
                    ' Low (Mag < {min_magnitude})';
                return div;
            }};
            legend.addTo(map);
        </script>
    </body>
    </html>
    """
    with open("neural_network.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Generated map: neural_network.html")


def main():
    """
    Main function to run the earthquake classification and mapping process.
    """
    qc = QuakeClassifierNeuralNetwork()
    features = qc.fetch_data()
    if not features:
        print("No features fetched.")
        return

    df = qc.process_data(features)
    if df.empty:
        print("No data after processing.")
        return

    df = qc.filter_data(df)
    if df.empty:
        print("No data after filtering.")
        return

    # Always calculate intensity based on magnitude
    df["intensity"] = df["magnitude"].apply(
        lambda x: "High" if x >= INTENSITY_THRESHOLD else "Low"
    )

    # Train the model if needed,
    # but don't use its predictions for visualization
    # fmt: off
    if NEURAL_NETWORK_MAP_INTENSITY_TYPE == "predicted":
        # Prepare data for training
        features = df[["magnitude", "depth"]]
        labels = df["magnitude"].apply(
            lambda x: 1 if x >= INTENSITY_THRESHOLD else 0
        )

        # Ensure there's enough data to split and train
        if len(df) >= 2:
            # Split data for training and testing
            (
                features_train,
                features_test,
                labels_train,
                labels_test,
            ) = train_test_split(
                features,
                labels,
                test_size=0.2,
                random_state=42,
            )

            # Train the model and get accuracy
            acc = qc.train_model(
                features_train, labels_train, features_test, labels_test
            )
            print(f"Model accuracy: {acc:.2f}")

    map_data = qc.generate_map_data(df)
    if not map_data:
        print("No map data generated.")
        return
    # fmt: on

    generate_map_html(map_data, INTENSITY_THRESHOLD)


if __name__ == "__main__":
    main()
