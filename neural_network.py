"""
GeoNet Quake Classifier - Neural Network

This module provides functionality to fetch, process,
and classify earthquake data from GeoNet New Zealand's API
using a neural network model. It uses the utilities module
to reduce code duplication.
"""

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from config import (
    NEURAL_NETWORK_MAP_INTENSITY_TYPE,
    MIN_MAGNITUDE,
    NEURAL_NETWORK_CONFIG,
)

import utilities


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
        self.filters = filters
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.api_url = utilities.build_api_url(filters)

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
        for units in NEURAL_NETWORK_CONFIG["hidden_layers"][1:]:
            model.add(
                Dense(units, activation=NEURAL_NETWORK_CONFIG["activation"])
            )  # noqa: E501
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

        # Output layer
        model.add(
            Dense(1, activation=NEURAL_NETWORK_CONFIG["output_activation"])
        )  # noqa: E501

        # Compile model with a lower learning rate
        model.compile(
            optimizer=Adam(
                learning_rate=NEURAL_NETWORK_CONFIG["learning_rate"]
            ),  # noqa: E501
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

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

    def train_model(
        self,
        features_train,
        labels_train,
        features_test,
        labels_test,
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
                monitor="val_loss", patience=10, restore_best_weights=True
            ),  # noqa: E501
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),  # noqa: E501
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
            features_test_scaled, labels_test, verbose=0
        )  # noqa: E501

        # Generate predictions for classification report
        predictions = (self.model.predict(features_test_scaled) >= 0.5).astype(
            int
        )  # noqa: E501
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


def main():
    """
    Main function to run the earthquake classification and mapping process.
    """
    print("🧠 GeoNet Earthquake Neural Network Classifier")
    print("=" * 48)
    print("Fetching earthquake data from GeoNet...")

    qc = QuakeClassifierNeuralNetwork()
    features = utilities.fetch_data(qc.api_url)
    if not features:
        print("No features fetched.")
        return

    df = utilities.process_data(features)
    if df.empty:
        print("No data after processing.")
        return

    df = utilities.filter_data(df)
    if df.empty:
        print("No data after filtering.")
        return

    # Calculate intensity_class for training
    df["intensity_class"] = df["magnitude"].apply(
        utilities.calculate_intensity_class
    )  # noqa: E501

    # Print dataset summary
    utilities.print_dataset_summary(df)

    # Always calculate intensity based on magnitude
    df = utilities.apply_intensity_labels(df)

    # Train the model if needed,
    # but don't use its predictions for visualization
    if NEURAL_NETWORK_MAP_INTENSITY_TYPE == "predicted":
        # Prepare data for training
        features = df[["magnitude", "depth"]]
        labels = df["intensity_class"]

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

    map_data = utilities.generate_map_data(df)
    if not map_data:
        print("No map data generated.")
        return

    # Generate HTML using utilities
    model_config = {
        "name": "Neural Network Model",
        "emoji": "🧠",
        "header_color": "#2d3748",
    }
    html_content = utilities.generate_map_html(
        map_data,
        MIN_MAGNITUDE,
        model_config,
        NEURAL_NETWORK_MAP_INTENSITY_TYPE,
    )
    utilities.save_html_file(html_content, "neural_network.html")

    print("\n✅ Classification complete!")
    print(f"Processed {len(df)} earthquakes")
    print("🧠 View results: neural_network.html")


if __name__ == "__main__":
    main()
